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

from src.data.multibeat_dataset import *
from src.data.sampler import UDAImbalancedDatasetSampler
from src.build_model import loss_function, build_distance, \
    build_training_records, build_validation_records, build_cnn_models
from src.centers_multibeat import *
from src.utils import *
from src.config import get_cfg_defaults

import argparse
import pprint


parser = argparse.ArgumentParser()
parser.add_argument('--config', default=None, type=str,
                    help='The directory of config .yaml file')
parser.add_argument('--check_epoch', default=-1, type=int,
                    help='The checkpoint ID for recovering training procedure')
args = parser.parse_args()

ROOT_PATH = '/home/workspace/mingchen/ECG_UDA/data_index'


def get_optimizer(optimizer, params, lr, weight_decay):

    if optimizer == 'Adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
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
    weights = {0: (1 - beta_cb) / (1 - beta_cb ** categories['N']), 1: (1 - beta_cb) / (1 - beta_cb ** categories['V']),
               2: (1 - beta_cb) / (1 - beta_cb ** categories['S']), 3: (1 - beta_cb) / (1 - beta_cb ** categories['F'])}

    return weights


def update_thrs(thrs, running_epoch, epochs):

    for l in range(4):
        if l == 0:
            thrs[l] = thrs[l] + 0.05 * (running_epoch / epochs)
        if l == 1 or l == 2:
            thrs[l] = thrs[l] + 0.1 * (running_epoch / epochs)
    return thrs


def obtain_pesudo_labels(net, loaded_models, t_batch, tr_batch, thrs, use_thr=True):

    predicts = []
    confidences = []
    for idx in range(len(loaded_models)):
        net.load_state_dict(loaded_models[idx])
        net.cuda()
        net.eval()

        _, _, _, preds = net(t_batch, tr_batch)
        preds = F.log_softmax(preds, dim=1).exp()
        preds = preds.detach().cpu().numpy()
        confidences.append(np.expand_dims(preds, axis=2))

        preds = np.expand_dims(np.argmax(preds, axis=1), axis=1)
        predicts.append(preds)

        net.cpu()

    predicts = np.concatenate(predicts, axis=1)
    confidences = np.concatenate(confidences, axis=2)
    confidences = np.max(np.mean(confidences, axis=2), axis=1)
    pesudo_labels = []

    for i in range(predicts.shape[0]):
        predict_line = predicts[i]
        pesudo_labels.append(np.argmax(np.bincount(predict_line)))

    pesudo_labels = np.array(pesudo_labels)

    if use_thr:
        _legal_indices = []
        for l in range(4):
            indices_l = np.argwhere(pesudo_labels == l)
            if len(indices_l) > 0:
                indices_l = np.squeeze(indices_l, axis=1)
                confidences_l = confidences[indices_l]
                legal_indices_l = np.argwhere(confidences_l >= thrs[l])
                if len(legal_indices_l) > 0:
                    legal_indices_l = np.squeeze(legal_indices_l, axis=1)
                    _legal_indices.append(indices_l[legal_indices_l])
        _legal_indices = np.concatenate(_legal_indices)
        pesudo_labels = pesudo_labels[_legal_indices]
    else:
        _legal_indices = np.arange(len(pesudo_labels))

    return pesudo_labels, _legal_indices


def get_L2norm_loss_self_driven(x, weight_L2norm):

    radius = x.norm(p=2, dim=1).detach()
    assert radius.requires_grad == False
    radius = radius + 0.3
    l = ((x.norm(p=2, dim=1) - radius) ** 2).mean()
    return weight_L2norm * l


def get_L2norm_loss_self_driven_hard(x, radius, weight_L2norm):
    l = (x.norm(p=2, dim=1).mean() - radius) ** 2
    return weight_L2norm * l


def get_entropy_loss(p_softmax, weight_entropy):
    mask = p_softmax.ge(0.000001)
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return weight_entropy * (entropy / float(p_softmax.size(0)))


def train(args):

    cudnn.benchmark = True

    cfg_dir = args.config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_dir)
    cfg.freeze()
    print(cfg)

    source = cfg.SETTING.TRAIN_DATASET
    target = cfg.SETTING.TEST_DATASET

    batch_size = cfg.TRAIN.BATCH_SIZE
    pre_train_epochs = cfg.TRAIN.PRE_TRAIN_EPOCHS
    epochs = cfg.TRAIN.EPOCHS
    lr = cfg.TRAIN.LR
    decay_rate = cfg.TRAIN.DECAY_RATE
    decay_step = cfg.TRAIN.DECAY_STEP
    flag_c = cfg.SETTING.CENTER
    flag_intra = cfg.SETTING.INTRA_LOSS
    flag_inter = cfg.SETTING.INTER_LOSS
    flag_norm = cfg.SETTING.NORM_ALIGN
    optimizer_ = cfg.SETTING.OPTIMIZER

    w_l2 = cfg.PARAMETERS.W_L2
    w_cls = cfg.PARAMETERS.W_CLS
    w_norm = cfg.PARAMETERS.W_NORM
    w_c = cfg.PARAMETERS.BETA_C
    w_cs = cfg.PARAMETERS.BETA1
    w_ct = cfg.PARAMETERS.BETA2
    w_cst = cfg.PARAMETERS.BETA
    w_bin = cfg.PARAMETERS.W_BIN
    w_mmd = cfg.PARAMETERS.BETA_MMD
    w_inter = cfg.PARAMETERS.BETA_INTER
    w_intra = cfg.PARAMETERS.BETA_INTRA
    thr_m = cfg.PARAMETERS.THR_M
    thrs_ = cfg.PARAMETERS.THRS

    emsemble_num = cfg.PARAMETERS.EMSEMBLE_NUM
    emsemble_step = cfg.PARAMETERS.EMSEMBLE_STEP

    lr_c = cfg.PARAMETERS.LR_C
    lr_cs = cfg.PARAMETERS.LR_C_S
    lr_ct = cfg.PARAMETERS.LR_C_T
    beta_cb = cfg.PARAMETERS.BETA_CB

    weights = get_cb_weights(source, beta_cb)

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
                          cfg.SETTING.UDA_NUM,
                          load_beat_with_rr,
                          transform=transform,
                          beat_num=cfg.SETTING.BEAT_NUM,
                          fixed_len=cfg.SETTING.FIXED_LEN,
                          use_dbscan=cfg.SETTING.USE_DBSCAN)

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=cfg.SYSTEM.NUM_WORKERS,
                            sampler=UDAImbalancedDatasetSampler(dataset))

    iter_num = int(len(dataset) / batch_size)

    net = build_cnn_models(cfg.SETTING.NETWORK,
                           fixed_len=cfg.SETTING.FIXED_LEN,
                           p=cfg.PARAMETERS.P)
    # Initialization of the model
    net.apply(init_weights)

    teacher_net = build_cnn_models(cfg.SETTING.NETWORK,
                                   fixed_len=cfg.SETTING.FIXED_LEN,
                                   p=cfg.PARAMETERS.P)

    print("The network {} has {} parameters in total".format(cfg.SETTING.NETWORK,
                                                             sum(x.numel() for x in net.parameters())))

    if flag_loading:
        net.load_state_dict(torch.load(check_point_dir)['model_state_dict'])
    net = net.cuda()

    criterion_cls_4 = loss_function(cfg.SETTING.LOSS, dataset=source, num_ew=cfg.PARAMETERS.N)
    criterion_cls_2 = loss_function('BinCBLoss', dataset=source)
    criterion_dist = build_distance(cfg.SETTING.DISTANCE)

    optimizer_pre = get_optimizer(optimizer_, net.parameters(), lr, w_l2)
    scheduler_pre = optim.lr_scheduler.StepLR(optimizer_pre,
                                              step_size=decay_step,
                                              gamma=decay_rate)

    optimizer_re = get_optimizer(optimizer_, net.parameters(), lr * 0.1, w_l2)
    scheduler_re = optim.lr_scheduler.StepLR(optimizer_re,
                                             step_size=decay_step * 10,
                                             gamma=decay_rate)

    optimizer_main = get_optimizer(optimizer_, net.parameters(), lr * 0.1, w_l2)
    scheduler_main = optim.lr_scheduler.StepLR(optimizer_main,
                                               step_size=decay_step * 10,
                                               gamma=decay_rate)
    evaluator = Eval(num_class=4)

    if check_epoch <= pre_train_epochs - 1:
        print("Starting STAGE I: pre-training the model using source data")

        for epoch in range(max(0, check_epoch), pre_train_epochs):
            for idb, data_batch in enumerate(dataloader):
                net.train()

                s_batch, sl_batch, sr_batch, sb_batch, \
                t_batch, tl_batch, tr_batch, tb_batch = data_batch

                s_batch = s_batch.unsqueeze(dim=1)
                t_batch = t_batch.unsqueeze(dim=1)
                s_batch = s_batch.cuda()
                sl_batch = sl_batch.cuda()
                t_batch = t_batch.cuda()
                tl_batch = tl_batch.cuda()
                sr_batch = sr_batch.cuda()
                sb_batch = sb_batch.cuda()
                tr_batch = tr_batch.cuda()
                tb_batch = tb_batch.cuda()

                _, pef, _, preds = net(s_batch, sr_batch)

                cls_loss = criterion_cls_4(preds, sl_batch)
                bin_loss = criterion_cls_2(pef, sb_batch)
                loss = cls_loss * w_cls + bin_loss * w_bin

                optimizer_pre.zero_grad()
                loss.backward()
                optimizer_pre.step()
                scheduler_pre.step()

                running_lr = optimizer_pre.state_dict()['param_groups'][0]['lr']

                if idb % 10 == 9:
                    print("[{}, {}] cls loss: {:.4f}, lr: {:.4f}".format(
                        epoch, idb, cls_loss, running_lr
                    ))

                    torch.save({"model_state_dict": net.state_dict()},
                               osp.join(save_path, '{}.pkl'.format(epoch)))

                if idb == iter_num - 1:
                    net.eval()
                    _, _, _, preds = net(t_batch, tr_batch)
                    preds_softmax = F.log_softmax(preds, dim=1).exp()
                    preds_softmax_np = preds_softmax.detach().cpu().numpy()
                    preds_ = np.argmax(preds_softmax_np, axis=1)

                    loss_eval = criterion_cls_4(preds, tl_batch)
                    print("The loss on target mini-batch is {:.4f}".format(loss_eval))
                    results = evaluator._metrics(predictions=preds_,
                                                 labels=tl_batch.detach().cpu().numpy())
                    pprint.pprint(results)

                torch.cuda.empty_cache()

    net.eval()
    centers_s, _ = init_source_centers(net, source, source_records, data_dict_source,
                                       batch_size=batch_size, num_workers=cfg.SYSTEM.NUM_WORKERS,
                                       beat_num=cfg.SETTING.BEAT_NUM, fixed_len=cfg.SETTING.FIXED_LEN)

    if cfg.SETTING.RE_TRAIN and (check_epoch <= pre_train_epochs * 2 - 1):
        print("Starting STAGE II: re-training the model using source data and extra constraints")

        for epoch in range(max(pre_train_epochs, check_epoch), 2 * pre_train_epochs):
            for idb, data_batch in enumerate(dataloader):
                net.train()

                s_batch, sl_batch, sr_batch, sb_batch,\
                t_batch, tl_batch, tr_batch, tb_batch = data_batch

                s_batch = s_batch.unsqueeze(dim=1)
                t_batch = t_batch.unsqueeze(dim=1)
                s_batch = s_batch.cuda()
                sl_batch = sl_batch.cuda()
                t_batch = t_batch.cuda()
                tl_batch = tl_batch.cuda()
                sr_batch = sr_batch.cuda()
                sb_batch = sb_batch.cuda()
                tr_batch = tr_batch.cuda()
                tb_batch = tb_batch.cuda()

                _, pef_s, feat_s, preds = net(s_batch, sr_batch)

                loss_cls = criterion_cls_4(preds, sl_batch)
                loss_bin = criterion_cls_2(pef_s, sb_batch)
                loss = loss_cls * w_cls + loss_bin * w_bin

                loss_cs = 0
                loss_intra = 0

                sl_batch_np = sl_batch.detach().cpu().numpy()

                for l in range(4):
                    _index = np.argwhere(sl_batch_np == l)
                    if len(_index):
                        _index = np.squeeze(_index, axis=1)
                        _feat_s = torch.index_select(feat_s, dim=0, index=torch.LongTensor(_index).cuda())
                        bs_ = _feat_s.size()[0]
                        m_feat_s = torch.mean(_feat_s, dim=0)

                        delta_cs_l = centers_s[l] - m_feat_s
                        centers_s[l] = centers_s[l] - lr_cs * delta_cs_l

                        loss_cs_l = criterion_dist(m_feat_s, centers_s[l])
                        loss_cs += loss_cs_l

                        if flag_intra:
                            cl_feat_s = centers_s[l].repeat((bs_, 1))
                            loss_intra_l = criterion_dist(_feat_s, cl_feat_s, dim=1) / bs_
                            loss_intra += loss_intra_l

                loss_intra = loss_intra / 4
                loss += loss_cs * w_cs

                loss_inter = 0
                for i in range(4 - 1):
                    for j in range(i + 1, 4):
                        loss_inter_ij = torch.max(thr_m - criterion_dist(centers_s[i], centers_s[j]),
                                                  torch.FloatTensor([0]).cuda()).squeeze()
                        loss_inter += loss_inter_ij
                loss_inter = loss_inter / 6

                if flag_inter:
                    loss += loss_inter * w_inter
                if flag_intra:
                    loss += loss_intra * w_intra

                optimizer_re.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_re.step()
                scheduler_re.step()

                running_lr = optimizer_pre.state_dict()['param_groups'][0]['lr']

                if idb % 10 == 9:
                    print("[{}, {}] cls loss: {:.4f}, cs loss: {:.4f}, intra loss: {:.4f}, "
                          "inter_loss: {:.4f}, lr: {:.4f}".format(epoch, idb, loss_cls, loss_cs,
                                                                  loss_intra, loss_inter, running_lr))
                    torch.save({'model_state_dict': net.state_dict()},
                               osp.join(save_path, '{}.pkl'.format(epoch)))

                if idb == iter_num - 1:
                    net.eval()
                    _, _, _, preds = net(t_batch, tr_batch)
                    preds_softmax = F.log_softmax(preds, dim=1).exp()
                    preds_softmax_np = preds_softmax.detach().cpu().numpy()
                    preds_ = np.argmax(preds_softmax_np, axis=1)

                    loss_eval = criterion_cls_4(preds, tl_batch)
                    print("The loss on target mini-batch is: {:.4f}".format(loss_eval))
                    results = evaluator._metrics(predictions=preds_,
                                                 labels=tl_batch.detach().cpu().numpy())
                    pprint.pprint(results)

                for l in range(4):
                    centers_s[l] = centers_s[l].detach()

                torch.cuda.empty_cache()

    print('Start obtaining centers of each cluster and distribution')

    centers_source_dir = osp.join(save_path, "centers_source.mat")
    centers_target_dir = osp.join(save_path, "centers_target.mat")
    centers_dir = osp.join(save_path, "centers.mat")
    flag_centers = osp.exists(centers_source_dir) and osp.exists(centers_target_dir) and osp.exists(centers_dir)

    center_source_dir = osp.join(save_path, "center_s.mat")
    center_target_dir = osp.join(save_path, "center_t.mat")
    flag_center = osp.exists(center_source_dir) and osp.exists(center_target_dir)

    if flag_centers:
        centers_s = sio.loadmat(centers_source_dir)
        centers_t = sio.loadmat(centers_target_dir)
        centers = sio.loadmat(centers_dir)
        for l in range(4):
            centers_s[l] = torch.from_numpy(centers_s['c{}'.format(l)].squeeze()).cuda()
            centers_t[l] = torch.from_numpy(centers_t['c{}'.format(l)].squeeze()).cuda()
            centers[l] = torch.from_numpy(centers['c{}'.format(l)].squeeze().astype(np.float32)).cuda()

    else:
        net.eval()
        centers_s, counter_s = init_source_centers(net, source, source_records, data_dict_source,
                                                   batch_size=batch_size, num_workers=cfg.SYSTEM.NUM_WORKERS,
                                                   beat_num=cfg.SETTING.BEAT_NUM, fixed_len=cfg.SETTING.FIXED_LEN)
        centers_t, counter_t = init_target_centers(net, target, target_records, data_dict_target,
                                                   batch_size=batch_size, num_workers=cfg.SYSTEM.NUM_WORKERS,
                                                   beat_num=cfg.SETTING.BEAT_NUM, fixed_len=cfg.SETTING.FIXED_LEN,
                                                   thrs=thrs)
        centers_s_np = {}
        centers_t_np = {}
        centers_np = {}
        centers = {}
        for l in range(4):
            centers_s_np['c{}'.format(l)] = centers_s[l].detach().cpu().numpy()
            centers_t_np['c{}'.format(l)] = centers_t[l].detach().cpu().numpy()
            centers_np['c{}'.format(l)] = ((centers_s_np['c{}'.format(l)] * counter_s[l] +
                                            centers_t_np['c{}'.format(l)] * counter_t[l])
                                           / (counter_s[l] + counter_t[l])).astype(np.float32)
            centers[l] = torch.from_numpy(centers_np['c{}'.format(l)]).cuda()
        sio.savemat(centers_source_dir, centers_s_np)
        sio.savemat(centers_target_dir, centers_t_np)
        sio.savemat(centers_dir, centers_np)

    if flag_center:
        center_s = torch.from_numpy(sio.loadmat(center_source_dir)['c'].squeeze()).cuda()
        center_t = torch.from_numpy(sio.loadmat(center_target_dir)['c'].squeeze()).cuda()
    else:

        center_s = init_entire_center(net, source, source_records, data_dict_source,
                                      batch_size=batch_size, num_workers=cfg.SYSTEM.NUM_WORKERS,
                                      beat_num=cfg.SETTING.BEAT_NUM, fixed_len=cfg.SETTING.FIXED_LEN)
        center_t = init_entire_center(net, target, target_records, data_dict_target,
                                      batch_size=batch_size, num_workers=cfg.SYSTEM.NUM_WORKERS,
                                      beat_num=cfg.SETTING.BEAT_NUM, fixed_len=cfg.SETTING.FIXED_LEN)

        sio.savemat(center_source_dir, {'c': center_s.detach().cpu().numpy()})
        sio.savemat(center_target_dir, {'c': center_t.detach().cpu().numpy()})

    print("Starting STAGE III: adaptation process")

    low_bound = max(2 * pre_train_epochs, check_epoch) if cfg.SETTING.RE_TRAIN else max(pre_train_epochs, check_epoch)
    high_bound = 2 * pre_train_epochs + epochs if cfg.SETTING.RE_TRAIN else pre_train_epochs + epochs
    for epoch in range(low_bound, high_bound):
        dataset.shuffle_target()

        loaded_models = []
        for idx in range(emsemble_num):
            load_dir = osp.join(save_path, '{}.pkl'.format(epoch - idx * emsemble_step - 1))
            loaded_models.append(torch.load(load_dir)['model_state_dict'])

        for idb, data_batch in enumerate(dataloader):
            net.train()

            s_batch, sl_batch, sr_batch, sb_batch,\
            t_batch, tl_batch, tr_batch, tb_batch = data_batch

            s_batch = s_batch.unsqueeze(dim=1)
            t_batch = t_batch.unsqueeze(dim=1)
            s_batch = s_batch.cuda()
            sl_batch = sl_batch.cuda()
            t_batch = t_batch.cuda()
            tl_batch = tl_batch.cuda()
            sr_batch = sr_batch.cuda()
            sb_batch = sb_batch.cuda()
            tr_batch = tr_batch.cuda()
            tb_batch = tb_batch.cuda()

            _, pef_s, feat_s, preds_s = net(s_batch, sr_batch)
            _, _, feat_t, preds_t = net(t_batch, tr_batch)

            loss_cls = criterion_cls_4(preds_s, sl_batch)
            loss_bin = criterion_cls_2(pef_s, sb_batch)
            loss = loss_cls * w_cls + loss_bin * w_bin

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

            pesudo_label_nums = {0: 0, 1: 0, 2: 0, 3: 0}
            pesudo_labels, legal_indices = obtain_pesudo_labels(teacher_net, loaded_models, t_batch, tr_batch, thrs)

            tmp_centers_t = {}
            if len(pesudo_labels):
                feat_t_pesudo = torch.index_select(feat_t, dim=0, index=torch.LongTensor(legal_indices).cuda())

                for l in range(4):
                    _index = np.argwhere(pesudo_labels == l)
                    if len(_index):
                        pesudo_label_nums[l] = len(_index)
                        _index = np.squeeze(_index, axis=1)
                        _feat_t = torch.index_select(feat_t_pesudo, dim=0, index=torch.LongTensor(_index).cuda())
                        bs_ = _feat_t.size()[0]

                        local_centers_tl = torch.mean(_feat_t, dim=0)
                        tmp_centers_t[l] = local_centers_tl
                        delta_ct = centers_t[l] - local_centers_tl
                        centers_t[l] = centers_t[l] - lr_ct * delta_ct

                        loss_ct_l = criterion_dist(local_centers_tl, centers_t[l])
                        loss_ct += loss_ct_l

                        if flag_intra:
                            m_feat_t = centers_t[l].repeat((bs_, 1))
                            loss_intra_l = criterion_dist(_feat_t, m_feat_t, dim=1)
                            loss_intra += loss_intra_l

            loss += loss_ct * w_ct

            sl_batch_np = sl_batch.detach().cpu().numpy()
            true_label_nums = {0: 0, 1: 0, 2: 0, 3: 0}

            tmp_centers_s = {}
            for l in range(4):
                _index = np.argwhere(sl_batch_np == l)
                if len(_index):
                    true_label_nums[l] = len(_index)
                    _index = np.squeeze(_index, axis=1)
                    _feat_s = torch.index_select(feat_s, dim=0, index=torch.LongTensor(_index).cuda())
                    bs_ = _feat_s.size()[0]
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

            loss += loss_cs * w_cs

            for l in range(4):
                loss_cst_l = criterion_dist(centers_s[l], centers_t[l])
                loss_cst += loss_cst_l

            loss += loss_cst * w_cst

            for i in range(4 - 1):
                for j in range(i + 1, 4):
                    loss_inter_ij_s = torch.max(thr_m - criterion_dist(centers_s[i], centers_s[j]),
                                                torch.FloatTensor([0]).cuda()).squeeze()
                    loss_inter_ij_t = torch.max(thr_m - criterion_dist(centers_t[i], centers_t[j]),
                                                torch.FloatTensor([0]).cuda()).squeeze()
                    loss_inter_ij = (loss_inter_ij_s + loss_inter_ij_t)
                    loss_inter += loss_inter_ij

            if flag_inter:
                loss += loss_inter * w_inter
            if flag_intra:
                loss += loss_intra * w_intra

            loss_c = 0

            for l in range(4):
                if (l in tmp_centers_s.keys()) and (l in tmp_centers_t.keys()):
                    tmp_centers_sl = tmp_centers_s[l]
                    tmp_centers_tl = tmp_centers_t[l]
                    m_centers_stl = (pesudo_label_nums[l] * tmp_centers_tl + true_label_nums[l] * tmp_centers_sl) \
                                    / (pesudo_label_nums[l] + true_label_nums[l])
                    delta_l = centers[l] - m_centers_stl
                    centers[l] = centers[l] - lr_c * delta_l

                    loss_cl = criterion_dist(m_centers_stl, centers[l])
                    loss_c += loss_cl

            if flag_c:
                loss += loss_c * w_c

            optimizer_main.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_main.step()
            scheduler_main.step()

            running_lr = optimizer_main.state_dict()['param_groups'][0]['lr']
            torch.cuda.empty_cache()
            for l in range(4):
                centers[l] = centers[l].detach()
                centers_s[l] = centers_s[l].detach()
                centers_t[l] = centers_t[l].detach()
            center_s = center_s.detach()
            center_t = center_t.detach()

            if idb % 10 == 9:
                print("[{}, {}] cls loss: {:.4f}, cs loss: {:.4f}, "
                      "ct loss: {:.4f}, cst loss: {:.4f}, mmd loss: {:.4f}, "
                      "inter loss: {:.4f}, intra loss: {:.4f}, c loss: {:.4f}, "
                      "lr: {:.5f}".format(epoch, idb, loss_cls, loss_cs, loss_ct, loss_cst, loss_mmd,
                                          loss_inter, loss_intra, loss_c, running_lr))

                print("The number of pesudo labels and true labels:")
                pprint.pprint(pesudo_label_nums)
                pprint.pprint(true_label_nums)

                torch.save({'model_state_dict': net.state_dict()},
                           osp.join(save_path, '{}.pkl'.format(epoch)))

            if idb == iter_num - 1:
                net.eval()
                _, _, _, preds = net(t_batch, tr_batch)
                preds_softmax = F.log_softmax(preds, dim=1).exp()
                preds_softmax_np = preds_softmax.detach().cpu().numpy()
                preds_ = np.argmax(preds_softmax_np, axis=1)
                loss_eval = criterion_cls_4(preds, tl_batch)
                print('The loss on target mini-batch is: {:.4f}'.format(loss_eval))
                results = evaluator._metrics(predictions=preds_,
                                             labels=tl_batch.detach().cpu().numpy())
                pprint.pprint(results)

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

