import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import numpy as np
import scipy.io as sio
import scipy.signal as ss
import os
import os.path as osp

from src.data.dataset_3d import *
from src.data.sampler import UDAImbalancedDatasetSampler
from src.build_model import loss_function, build_distance, \
    build_training_records, build_validation_records, build_acnn_models
from src.centers_acnn import *
from src.utils import *
from src.config import get_cfg_defaults
from src.regularizers import *

import argparse
import pprint


parser = argparse.ArgumentParser()
parser.add_argument('--config', default=None, type=str,
                    help='The directory of config .yaml file')
parser.add_argument('--check_epoch', default=-1, type=int,
                    help='The checkpoint ID for recovering training procedure')
parser.add_argument('--use_ema', dest='use_ema',
                    action='store_true')
args = parser.parse_args()

ROOT_PATH = '/home/workspace/mingchen/ECG_UDA/data_index'


def get_optimizer(optimizer, params, lr, weight_decay):

    if optimizer == 'Adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'Adagrad':
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'SGD':
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        print('No available optimizer')
        raise ValueError


def get_cb_weights(source, beta_cb):
    categories = {'N': 0, 'V': 0, 'S': 0, 'F': 0}
    dataset_path = osp.join(osp.join(ROOT_PATH, source), 'entire')
    for cate in categories.keys():
        files = os.listdir(osp.join(dataset_path, cate))
        categories[cate] = len(files)
    weights = {0: (1 - beta_cb) / (1 - beta_cb ** categories['N']),
               1: (1 - beta_cb) / (1 - beta_cb ** categories['V']),
               2: (1 - beta_cb) / (1 - beta_cb ** categories['S']),
               3: (1 - beta_cb) / (1 - beta_cb ** categories['F'])}

    return weights


def update_thrs(thrs, running_epoch, epochs):

    interval = 10

    if running_epoch % interval == 0:
        for l in range(4):
            if l == 0:
                thrs[l] = thrs[l] + 0.001
            if l == 1:
                thrs[l] = thrs[l] + 0.001
            if l == 2:
                thrs[l] = thrs[l] + 0.001
            if l == 3:
                thrs[l] = thrs[l] + 0.001
    return thrs


def train(args):

    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg_dir = args.config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_dir)
    cfg.freeze()

    '''Setting the random seed used in the experiment'''

    if cfg.SETTING.SEED != -1:
        torch.manual_seed(cfg.SETTING.SEED)
        torch.cuda.manual_seed(cfg.SETTING.SEED)

    source = cfg.SETTING.TRAIN_DATASET
    target = cfg.SETTING.TEST_DATASET

    batch_size = cfg.TRAIN.BATCH_SIZE
    pre_train_epochs = cfg.TRAIN.PRE_TRAIN_EPOCHS
    epochs = cfg.TRAIN.EPOCHS
    lr = cfg.TRAIN.LR
    decay_rate = cfg.TRAIN.DECAY_RATE
    decay_step = cfg.TRAIN.DECAY_STEP
    flag_intra = cfg.SETTING.INTRA_LOSS
    flag_inter = cfg.SETTING.INTER_LOSS
    flag_norm = cfg.SETTING.NORM_ALIGN
    optimizer_ = cfg.SETTING.OPTIMIZER

    w_l2 = cfg.PARAMETERS.W_L2
    w_cls = cfg.PARAMETERS.W_CLS
    w_norm = cfg.PARAMETERS.W_NORM
    w_cs = cfg.PARAMETERS.BETA1
    w_ct = cfg.PARAMETERS.BETA2
    w_cst = cfg.PARAMETERS.BETA
    w_mmd = cfg.PARAMETERS.BETA_MMD
    w_inter = cfg.PARAMETERS.BETA_INTER
    w_intra = cfg.PARAMETERS.BETA_INTRA
    thr_m = cfg.PARAMETERS.THR_M
    thrs_ = cfg.PARAMETERS.THRS
    entropy_w = 0.001

    emsemble_num = cfg.PARAMETERS.EMSEMBLE_NUM
    emsemble_step = cfg.PARAMETERS.EMSEMBLE_STEP

    lr_c = cfg.PARAMETERS.LR_C
    lr_cs = cfg.PARAMETERS.LR_C_S
    lr_ct = cfg.PARAMETERS.LR_C_T

    thrs = {}
    for l in range(len(thrs_)):
        thrs[l] = thrs_[l]

    exp_id = os.path.basename(cfg_dir).split('.')[0]

    save_path = os.path.join(cfg.SYSTEM.SAVE_PATH, exp_id)
    if not osp.exists(save_path):
        os.makedirs(save_path)

    check_epoch = args.check_epoch
    check_point_dir = osp.join(save_path, '{}.pkl'.format(check_epoch))
    flag_loading = True if osp.exists(check_point_dir) else False

    data_dict_source = load_dataset_to_memory(source)
    data_dict_target = load_dataset_to_memory(target) if (source != target) else data_dict_source

    transform = augmentation_transform_with_rr if cfg.SETTING.AUGMENTATION else None

    source_records = build_training_records(source)
    target_records = build_validation_records(target)

    dataset = UDA_DATASET(source, target,
                          data_dict_source, data_dict_target,
                          source_records, target_records,
                          cfg.SETTING.UDA_NUM, load_beat_with_rr,
                          transform=transform, beat_num=cfg.SETTING.BEAT_NUM,
                          fixed_len=cfg.SETTING.FIXED_LEN, lead=cfg.SETTING.LEAD)

    dset_val = MULTI_ECG_EVAL_DATASET(target,
                                      load_beat_with_rr,
                                      data_dict_target,
                                      test_records=target_records,
                                      beat_num=cfg.SETTING.BEAT_NUM,
                                      fixed_len=cfg.SETTING.FIXED_LEN,
                                      lead=cfg.SETTING.LEAD,
                                      unlabel_num=0)
    dloader_val = DataLoader(dset_val, batch_size=batch_size,
                             num_workers=cfg.SYSTEM.NUM_WORKERS)

    if cfg.TRAIN.IMBALANCE_SAMPLE:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                num_workers=cfg.SYSTEM.NUM_WORKERS,
                                sampler=UDAImbalancedDatasetSampler(dataset))
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                num_workers=cfg.SYSTEM.NUM_WORKERS,
                                shuffle=True)

    iter_num = int(len(dataset) / batch_size)

    net = build_acnn_models(cfg.SETTING.NETWORK, cfg.SETTING.ASPP_BN,
                            cfg.SETTING.ASPP_ACT, cfg.SETTING.LEAD,
                            cfg.PARAMETERS.P, cfg.SETTING.DILATIONS,
                            act_func=cfg.SETTING.ACT, f_act_func=cfg.SETTING.F_ACT,
                            apply_residual=cfg.SETTING.RESIDUAL,
                            bank_num=cfg.SETTING.BANK_NUM)
    # Initialization of the model
    net.apply(init_weights)

    teacher_net = build_acnn_models(cfg.SETTING.NETWORK, cfg.SETTING.ASPP_BN,
                                    cfg.SETTING.ASPP_ACT, cfg.SETTING.LEAD,
                                    cfg.PARAMETERS.P, cfg.SETTING.DILATIONS,
                                    act_func=cfg.SETTING.ACT, f_act_func=cfg.SETTING.F_ACT,
                                    apply_residual=cfg.SETTING.RESIDUAL,
                                    bank_num=cfg.SETTING.BANK_NUM)

    print("The network {} has {} parameters in total".format(cfg.SETTING.NETWORK,
                                                             sum(x.numel() for x in net.parameters())))

    if flag_loading:
        net.load_state_dict(torch.load(check_point_dir)['model_state_dict'])
        print("The saved model is loaded.")
    net = net.cuda()

    criterion_cls_4 = loss_function(cfg.SETTING.LOSS, dataset=source, num_ew=cfg.PARAMETERS.N, T=cfg.PARAMETERS.T)
    criterion_dist = build_distance(cfg.SETTING.DISTANCE)

    optimizer_pre = get_optimizer(optimizer_, net.parameters(), lr, w_l2)
    scheduler_pre = optim.lr_scheduler.StepLR(optimizer_pre,
                                              step_size=decay_step,
                                              gamma=decay_rate)

    optimizer_main = get_optimizer(optimizer_, net.parameters(), lr * 0.1, w_l2)
    scheduler_main = optim.lr_scheduler.StepLR(optimizer_main,
                                               step_size=decay_step * 10,
                                               gamma=decay_rate)
    evaluator = Eval(num_class=4)
    '''Initial and register the EMA'''
    ema = EMA(model=net, decay=0.99)
    ema.register()

    if check_epoch < pre_train_epochs - 1:
        print("Starting STAGE I: pre-training the model using source data")

        best_f1_s = 0.0

        for epoch in range(max(0, check_epoch), pre_train_epochs):
            for idb, data_batch in enumerate(dataloader):
                net.train()

                s_batch, sl_batch, t_batch, tl_batch = data_batch
                s_batch = s_batch.cuda()
                sl_batch = sl_batch.cuda()
                t_batch = t_batch.cuda()
                tl_batch = tl_batch.cuda()

                _, preds = net(s_batch)
                loss = criterion_cls_4(preds, sl_batch)

                # Add an entropy regularizer
                # p_softmax = nn.Softmax(dim=1)(preds)
                # loss -= get_entropy_loss(p_softmax, entropy_w)

                optimizer_pre.zero_grad()
                loss.backward()
                optimizer_pre.step()
                scheduler_pre.step()
                if args.use_ema:
                    ema.update()
                    ema.apply_shadow()

                running_lr = optimizer_pre.state_dict()['param_groups'][0]['lr']

                print("[{}, {}] cls loss: {:.4f}, lr: {:.4f}".format(
                        epoch, idb, loss, running_lr
                    ), end='\r')
                if idb == iter_num - 1:
                    torch.save({"model_state_dict": net.state_dict()},
                               osp.join(save_path, '{}.pkl'.format(epoch)))

            if epoch % 10 == 9:
                net.eval()
                preds_entire = []
                labels_entire = []

                with torch.no_grad():
                    for idb, data_batch in enumerate(dloader_val):
                        s_batch, l_batch = data_batch

                        s_batch = s_batch.cuda()
                        l_batch = l_batch.numpy()

                        _, logits = net(s_batch)

                        preds_softmax = F.log_softmax(logits, dim=1).exp()
                        preds_softmax_np = preds_softmax.detach().cpu().numpy()
                        preds = np.argmax(preds_softmax_np, axis=1)

                        preds_entire.append(preds)
                        labels_entire.append(l_batch)

                        torch.cuda.empty_cache()

                preds_entire = np.concatenate(preds_entire, axis=0)
                labels_entire = np.concatenate(labels_entire, axis=0)

                Pp, Se = evaluator._sklean_metrics(y_pred=preds_entire,
                                                   y_label=labels_entire)
                results = evaluator._metrics(predictions=preds_entire, labels=labels_entire)
                # con_matrix = evaluator._confusion_matrix(y_pred=preds_entire,
                #                                          y_label=labels_entire)

                f1_scores = evaluator._f1_score(y_pred=preds_entire, y_true=labels_entire)

                print('The overall accuracy is: {}'.format(results['Acc']))
                print("The confusion matrix is: ")
                print('Pp: ')
                pprint.pprint(Pp)
                print('Se: ')
                pprint.pprint(Se)
                print('The F1 score is: {}'.format(f1_scores))

                if f1_scores[2] >= best_f1_s:
                    best_f1_s = f1_scores[2]
                    torch.save({"model_state_dict": net.state_dict()},
                               osp.join(save_path, 'best_model.pkl'))

                torch.cuda.empty_cache()

    print('Start obtaining centers of each cluster and distribution')

    best_model_dir = osp.join(save_path, 'best_model.pkl')
    net.load_state_dict(torch.load(best_model_dir)['model_state_dict'])

    centers_source_dir = osp.join(save_path, "centers_source.mat")
    centers_target_dir = osp.join(save_path, "centers_target.mat")
    flag_centers = osp.exists(centers_source_dir) and osp.exists(centers_target_dir)

    center_source_dir = osp.join(save_path, "center_s.mat")
    center_target_dir = osp.join(save_path, "center_t.mat")
    flag_center = osp.exists(center_source_dir) and osp.exists(center_target_dir)

    if flag_centers:
        centers_s_ = sio.loadmat(centers_source_dir)
        centers_t_ = sio.loadmat(centers_target_dir)
        centers_s = {}
        centers_t = {}
        for l in range(4):
            if 'c{}'.format(l) in centers_s_.keys():
                centers_s[l] = torch.from_numpy(centers_s_['c{}'.format(l)].squeeze()).cuda()
            if 'c{}'.format(l) in centers_t_.keys():
                centers_t[l] = torch.from_numpy(centers_t_['c{}'.format(l)].squeeze()).cuda()
    else:
        net.eval()
        centers_s, counter_s = init_source_centers(net, source, source_records, data_dict_source,
                                                   batch_size=batch_size, num_workers=cfg.SYSTEM.NUM_WORKERS,
                                                   beat_num=cfg.SETTING.BEAT_NUM, fixed_len=cfg.SETTING.FIXED_LEN,
                                                   lead=cfg.SETTING.LEAD)
        centers_t, counter_t = init_target_centers(net, target, target_records, data_dict_target,
                                                   batch_size=batch_size, num_workers=cfg.SYSTEM.NUM_WORKERS,
                                                   beat_num=cfg.SETTING.BEAT_NUM, fixed_len=cfg.SETTING.FIXED_LEN,
                                                   lead=cfg.SETTING.LEAD, thrs=thrs)
        centers_s_np = {}
        centers_t_np = {}
        for l in range(4):
            if l in centers_s.keys():
                centers_s_np['c{}'.format(l)] = centers_s[l].detach().cpu().numpy()
            if l in centers_t.keys():
                centers_t_np['c{}'.format(l)] = centers_t[l].detach().cpu().numpy()
        sio.savemat(centers_source_dir, centers_s_np)
        sio.savemat(centers_target_dir, centers_t_np)

    if flag_center:
        center_s = torch.from_numpy(sio.loadmat(center_source_dir)['c'].squeeze()).cuda()
        center_t = torch.from_numpy(sio.loadmat(center_target_dir)['c'].squeeze()).cuda()
    else:

        center_s = init_entire_center(net, source, source_records, data_dict_source,
                                      batch_size=batch_size, num_workers=cfg.SYSTEM.NUM_WORKERS,
                                      beat_num=cfg.SETTING.BEAT_NUM, fixed_len=cfg.SETTING.FIXED_LEN,
                                      lead=cfg.SETTING.LEAD)
        center_t = init_entire_center(net, target, target_records, data_dict_target,
                                      batch_size=batch_size, num_workers=cfg.SYSTEM.NUM_WORKERS,
                                      beat_num=cfg.SETTING.BEAT_NUM, fixed_len=cfg.SETTING.FIXED_LEN,
                                      lead=cfg.SETTING.LEAD)
        sio.savemat(center_source_dir, {'c': center_s.detach().cpu().numpy()})
        sio.savemat(center_target_dir, {'c': center_t.detach().cpu().numpy()})

    print("Starting STAGE III: adaptation process")

    low_bound = max(2 * pre_train_epochs, check_epoch) if cfg.SETTING.RE_TRAIN else max(pre_train_epochs, check_epoch)
    high_bound = 2 * pre_train_epochs + epochs if cfg.SETTING.RE_TRAIN else pre_train_epochs + epochs
    for epoch in range(low_bound, high_bound):
        best_f1_s = 0.0
        dataset.shuffle_target()

        # load_dir = osp.join(save_path, 'best_model.pkl')
        # loaded_models = torch.load(load_dir)['model_state_dict']
        loaded_models = []
        for idx in range(emsemble_num):
            load_dir = osp.join(save_path, '{}.pkl'.format(epoch - idx * emsemble_step - 1))
            loaded_models.append(torch.load(load_dir)['model_state_dict'])

        for idb, data_batch in enumerate(dataloader):
            net.train()

            s_batch, sl_batch, t_batch, tl_batch = data_batch
            s_batch = s_batch.cuda()
            sl_batch = sl_batch.cuda()
            t_batch = t_batch.cuda()
            tl_batch = tl_batch.cuda()

            feat_s, preds_s = net(s_batch)
            feat_t, preds_t = net(t_batch)

            loss_cls = criterion_cls_4(preds_s, sl_batch)
            loss = loss_cls * w_cls

            # Add an entropy regularizer
            # p_softmax = nn.Softmax(dim=1)(preds_s)
            # loss -= get_entropy_loss(p_softmax, entropy_w)

            delta_s = center_s - torch.mean(feat_s, dim=0)
            delta_t = center_t - torch.mean(feat_t, dim=0)

            center_s = center_s - lr_c * delta_s
            center_t = center_t - lr_c * delta_t

            loss_mmd = criterion_dist(center_s, center_t)
            loss += loss_mmd * w_mmd

            loss_intra = 0
            loss_inter = 0
            loss_ct = 0
            loss_cs = 0
            loss_cst = 0

            if flag_norm:
                if cfg.SETTING.ALIGN_SET == 'soft':
                    loss += get_L2norm_loss_self_driven(feat_s, w_norm)
                    loss += get_L2norm_loss_self_driven(feat_t, w_norm)
                else:
                    loss += get_L2norm_loss_self_driven_hard(feat_s, cfg.PARAMETERS.RADIUS, w_norm)
                    loss += get_L2norm_loss_self_driven_hard(feat_t, cfg.PARAMETERS.RADIUS, w_norm)

            '''Obtaining the pesudo labels of target samples'''
            pseudo_label_nums = {0: 0, 1: 0, 2: 0, 3: 0}
            pseudo_labels, legal_indices = obtain_pseudo_labels(teacher_net, loaded_models, t_batch, thrs)
            # pseudo_labels: (NUM, ); legal_indices: (NUM, ),the indices of legal pseudo labels;

            tmp_centers_t = {}
            tmp_feats_t = {}
            # if len(pesudo_labels):
            if pseudo_labels.size(0) > 0:
                # feat_t_pesudo = torch.index_select(feat_t, dim=0, index=torch.LongTensor(legal_indices).cuda())
                feat_t_pseudo = torch.index_select(feat_t, dim=0, index=legal_indices)

                for l in range(4):
                    # _index = np.argwhere(pseudo_labels == l)
                    _index = torch.nonzero(pseudo_labels == l).squeeze(dim=1)
                    if _index.size(0) > 0:
                        pseudo_label_nums[l] = _index.size(0)
                        # _index = np.squeeze(_index, axis=1)
                        # _feat_t = torch.index_select(feat_t_pesudo, dim=0, index=torch.LongTensor(_index).cuda())
                        _feat_t = torch.index_select(feat_t_pseudo, dim=0, index=_index)
                        tmp_feats_t[l] = _feat_t
                        bs_ = _feat_t.size(0)

                        local_centers_tl = torch.mean(_feat_t, dim=0)
                        tmp_centers_t[l] = local_centers_tl

                        if l in centers_t.keys():
                            delta_ct = centers_t[l] - local_centers_tl
                            centers_t[l] = centers_t[l] - lr_ct * delta_ct
                            loss_ct_l = criterion_dist(local_centers_tl, centers_t[l])
                            loss_ct += loss_ct_l
                        else:
                            centers_t[l] = local_centers_tl

                        if flag_intra:
                            m_feat_t = centers_t[l].repeat((bs_, 1))
                            loss_intra_l = criterion_dist(_feat_t, m_feat_t, dim=1)
                            loss_intra += loss_intra_l

            if cfg.SETTING.CLoss:
                loss += loss_ct * w_ct

            # sl_batch_np = sl_batch.detach().cpu().numpy()
            sl_batch_ = sl_batch.detach()
            true_label_nums = {0: 0, 1: 0, 2: 0, 3: 0}

            tmp_centers_s = {}
            tmp_feats_s = {}

            for l in range(4):
                # _index = np.argwhere(sl_batch_np == l)
                _index = torch.nonzero(sl_batch_ == l).squeeze(dim=1)
                if _index.size(0) > 0:
                    true_label_nums[l] = _index.size(0)
                    # _feat_s = torch.index_select(feat_s, dim=0, index=torch.LongTensor(_index).cuda())
                    _feat_s = torch.index_select(feat_s, dim=0, index=_index)
                    tmp_feats_s[l] = _feat_s
                    bs_ = _feat_s.size(0)

                    local_centers_sl = torch.mean(_feat_s, dim=0)
                    tmp_centers_s[l] = local_centers_sl
                    delta_cs = centers_s[l] - local_centers_sl
                    centers_s[l] = centers_s[l] - lr_cs * delta_cs

                    loss_cs_l = criterion_dist(local_centers_sl, centers_s[l])
                    loss_cs += loss_cs_l

                    if flag_intra:
                        m_feat_s = centers_s[l].repeat((bs_, 1))
                        loss_intra_l = criterion_dist(_feat_s, m_feat_s, dim=1)
                        loss_intra += loss_intra_l

            if cfg.SETTING.CLoss:
                loss += loss_cs * w_cs

            for l in centers_t.keys():
                loss_cst_l = criterion_dist(centers_s[l], centers_t[l])
                loss_cst += loss_cst_l

            if cfg.SETTING.CLoss:
                loss += loss_cst * w_cst

            for i in range(4 - 1):
                for j in range(i + 1, 4):
                    loss_inter_ij_s = torch.max(thr_m - criterion_dist(centers_s[i], centers_s[j]),
                                                torch.FloatTensor([0]).cuda()).squeeze()
                    loss_inter_ij_t = torch.max(thr_m - criterion_dist(centers_t[i], centers_t[j]),
                                                torch.FloatTensor([0]).cuda()).squeeze()

                    '''Add items between two domains'''
                    loss_inter_ij_st = torch.max(thr_m - criterion_dist(centers_s[i], centers_t[j]),
                                                 torch.FloatTensor([0]).cuda()).squeeze()
                    loss_inter_ij_ts = torch.max(thr_m - criterion_dist(centers_t[i], centers_s[j]),
                                                 torch.FloatTensor([0]).cuda()).squeeze()

                    loss_inter_ij = (loss_inter_ij_s + loss_inter_ij_t + loss_inter_ij_st + loss_inter_ij_ts) / 4
                    # loss_inter_ij = (loss_inter_ij_s + loss_inter_ij_t)
                    # loss_inter_ij = loss_inter_ij_s

                    loss_inter += loss_inter_ij

            if flag_inter:
                loss += loss_inter * w_inter
            if flag_intra:
                loss += loss_intra * w_intra

            loss_coral = 0
            if cfg.SETTING.CORAL:
                for l in tmp_feats_s.keys():
                    if l in tmp_feats_t.keys():
                        loss_coral += coral(tmp_feats_s[l], tmp_feats_t[l])
                loss += loss_coral

            optimizer_main.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_main.step()
            scheduler_main.step()
            if args.use_ema:
                ema.update()
                ema.apply_shadow()

            running_lr = optimizer_main.state_dict()['param_groups'][0]['lr']
            torch.cuda.empty_cache()
            for l in centers_s.keys():
                centers_s[l] = centers_s[l].detach()
            for l in centers_t.keys():
                centers_t[l] = centers_t[l].detach()
            center_s = center_s.detach()
            center_t = center_t.detach()

            if idb == iter_num - 1:
                print("[{}, {}] cls loss: {:.4f}, cs loss: {:.4f}, "
                      "ct loss: {:.4f}, cst loss: {:.4f}, mmd loss: {:.4f}, "
                      "inter loss: {:.4f}, intra loss: {:.4f}, "
                      "CORAL: {:.4f}, "
                      "lr: {:.5f}".format(epoch, idb, loss_cls, loss_cs, loss_ct, loss_cst, loss_mmd,
                                          loss_inter, loss_intra, loss_coral, running_lr))

                print("The number of pesudo labels and true labels:")
                pprint.pprint(pseudo_label_nums)
                pprint.pprint(true_label_nums)

                torch.save({'model_state_dict': net.state_dict()},
                           osp.join(save_path, '{}.pkl'.format(epoch)))

        if epoch % 10 == 9:
            net.eval()

            preds_entire = []
            labels_entire = []

            with torch.no_grad():
                for idb, data_batch in enumerate(dloader_val):
                    s_batch, l_batch = data_batch

                    s_batch = s_batch.cuda()
                    l_batch = l_batch.numpy()

                    _, logits = net(s_batch)

                    preds_softmax = F.log_softmax(logits, dim=1).exp()
                    preds_softmax_np = preds_softmax.detach().cpu().numpy()
                    preds = np.argmax(preds_softmax_np, axis=1)

                    preds_entire.append(preds)
                    labels_entire.append(l_batch)

                    torch.cuda.empty_cache()

            preds_entire = np.concatenate(preds_entire, axis=0)
            labels_entire = np.concatenate(labels_entire, axis=0)

            Pp, Se = evaluator._sklean_metrics(y_pred=preds_entire,
                                               y_label=labels_entire)
            results = evaluator._metrics(predictions=preds_entire, labels=labels_entire)
            f1_scores = evaluator._f1_score(y_pred=preds_entire, y_true=labels_entire)
            con_matrix = evaluator._confusion_matrix(preds_entire, labels_entire)

            print('The overall accuracy is: {}'.format(results['Acc']))
            print("The confusion matrix is: ")
            print(con_matrix)
            # print('The sklearn metrics are: ')
            print('Pp: ')
            pprint.pprint(Pp)
            print('Se: ')
            pprint.pprint(Se)
            print('The F1 score is: {}'.format(f1_scores))

            if f1_scores[2] >= best_f1_s:
                best_f1_s = f1_scores[2]
                torch.save({"model_state_dict": net.state_dict()},
                           osp.join(save_path, 'best_model.pkl'))

        # Updating thresholds for pesudo labels
        if cfg.SETTING.INCRE_THRS:
            if cfg.SETTING.RE_TRAIN:
                epoch_ = epoch - 2 * pre_train_epochs
            else:
                epoch_ = epoch - pre_train_epochs
            thrs = update_thrs(thrs, epoch_, epochs)


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)

