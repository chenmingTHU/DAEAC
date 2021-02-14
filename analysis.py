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

import argparse
import pprint

import matplotlib.pyplot as plt
plt.switch_backend('Agg')


def eval_epochs():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str,
                        help='The directory of config .yaml file')
    # parser.add_argument('--check_epoch', default=-1, type=int,
    #                     help='The checkpoint ID for recovering training procedure')
    parser.add_argument('--target', dest='target',
                        default=None, type=str,
                        choices=['mitdb', 'fmitdb', 'svdb', 'incartdb'],
                        help='One can choose another dataset for validation '
                             'beyond the training setting up')
    args = parser.parse_args()

    cfg_dir = args.config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_dir)
    cfg.freeze()

    target = cfg.SETTING.TEST_DATASET
    if args.target:
        target = args.target

    batch_size = cfg.TRAIN.BATCH_SIZE

    exp_id = os.path.basename(cfg_dir).split('.')[0]
    save_path = os.path.join(cfg.SYSTEM.SAVE_PATH, exp_id)

    img_path = os.path.join('./figures', exp_id)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    res_path = os.path.join('./results', exp_id)
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    evaluator = Eval(num_class=4)
    data_dict_target = load_dataset_to_memory(target)
    target_records = build_validation_records(target)

    net = build_acnn_models(cfg.SETTING.NETWORK,
                            aspp_bn=cfg.SETTING.ASPP_BN,
                            aspp_act=cfg.SETTING.ASPP_ACT,
                            lead=cfg.SETTING.LEAD,
                            p=cfg.PARAMETERS.P,
                            dilations=cfg.SETTING.DILATIONS,
                            act_func=cfg.SETTING.ACT,
                            f_act_func=cfg.SETTING.F_ACT,
                            apply_residual=cfg.SETTING.RESIDUAL)

    for check_epoch in range(0, 109, 1):

        net = net.cpu()
        check_point_dir = osp.join(save_path, '{}.pkl'.format(check_epoch))
        net.load_state_dict(torch.load(check_point_dir)['model_state_dict'])
        net = net.cuda()
        net.eval()

        dataset = MULTI_ECG_EVAL_DATASET(target,
                                         load_beat_with_rr,
                                         data_dict_target,
                                         test_records=target_records,
                                         beat_num=cfg.SETTING.BEAT_NUM,
                                         fixed_len=cfg.SETTING.FIXED_LEN,
                                         lead=cfg.SETTING.LEAD, unlabel_num=cfg.SETTING.UDA_NUM)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=cfg.SYSTEM.NUM_WORKERS)
        preds_entire = []
        labels_entire = []
        probs_entire = []

        with torch.no_grad():
            for idb, data_batch in enumerate(dataloader):

                s_batch, l_batch = data_batch

                s_batch = s_batch.cuda()
                l_batch = l_batch.numpy()

                _, logits = net(s_batch)

                preds_softmax = F.log_softmax(logits, dim=1).exp()
                preds_softmax_np = preds_softmax.detach().cpu().numpy()
                preds = np.argmax(preds_softmax_np, axis=1)

                preds_entire.append(preds)
                labels_entire.append(l_batch)
                probs_entire.append(preds_softmax_np)

                torch.cuda.empty_cache()

        preds_entire = np.concatenate(preds_entire, axis=0)
        labels_entire = np.concatenate(labels_entire, axis=0)
        probs_entire = np.concatenate(probs_entire, axis=0)

        Pp, Se = evaluator._sklean_metrics(y_pred=preds_entire,
                                           y_label=labels_entire)
        con_matrix = evaluator._confusion_matrix(y_pred=preds_entire,
                                                 y_label=labels_entire)
        F1_scores = evaluator._f1_score(y_pred=preds_entire, y_true=labels_entire)

        print("The confusion matrix is: ")
        print(con_matrix)
        print('The sklearn metrics are: ')
        print('Pp: ')
        pprint.pprint(Pp)
        print('Se: ')
        pprint.pprint(Se)
        print('The F1 score is: {}'.format(F1_scores))

        se = evaluator._get_recall(y_pred=preds_entire, y_label=labels_entire)
        pp = evaluator._get_precisions(y_pred=preds_entire, y_label=labels_entire)

        np.savez(os.path.join(res_path, "checkepoch_{}.npz".format(check_epoch)), Se=se, Pp=pp,
                 F1=F1_scores)


if __name__ == '__main__':
    eval_epochs()

