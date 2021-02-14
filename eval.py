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


parser = argparse.ArgumentParser()
parser.add_argument('--config', default=None, type=str,
                    help='The directory of config .yaml file')
parser.add_argument('--check_epoch', default=-1, type=int,
                    help='The checkpoint ID for recovering training procedure')
parser.add_argument('--target', dest='target',
                    default=None, type=str,
                    choices=['mitdb', 'fmitdb', 'svdb', 'incartdb', 'ltdb'],
                    help='One can choose another dataset for validation '
                         'beyond the training setting up')
args = parser.parse_args()

ROOT_PATH = '/home/workspace/mingchen/ECG_UDA/data_index'


def eval(args):

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

    # img_path = os.path.join('./figures', exp_id)
    # if not os.path.exists(img_path):
    #     os.makedirs(img_path)

    check_epoch = args.check_epoch
    check_point_dir = osp.join(save_path, '{}.pkl'.format(check_epoch))

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
                            apply_residual=cfg.SETTING.RESIDUAL,
                            bank_num=cfg.SETTING.BANK_NUM)
    net.load_state_dict(torch.load(check_point_dir)['model_state_dict'])
    net = net.cuda()
    net.eval()

    evaluator = Eval(num_class=4)

    print("The network {} has {} "
          "parameters in total".format(cfg.SETTING.NETWORK,
                                       sum(x.numel() for x in net.parameters())))

    dataset = MULTI_ECG_EVAL_DATASET(target,
                                     load_beat_with_rr,
                                     data_dict_target,
                                     test_records=target_records,
                                     beat_num=cfg.SETTING.BEAT_NUM,
                                     fixed_len=cfg.SETTING.FIXED_LEN,
                                     lead=cfg.SETTING.LEAD, unlabel_num=cfg.SETTING.UDA_NUM)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=cfg.SYSTEM.NUM_WORKERS)

    print("The size of the validation dataset is {}".format(len(dataset)))

    preds_entire = []
    labels_entire = []
    probs_entire = []
    samples = []

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
            samples.append(s_batch.detach().cpu().numpy())

            torch.cuda.empty_cache()

    preds_entire = np.concatenate(preds_entire, axis=0)
    labels_entire = np.concatenate(labels_entire, axis=0)
    probs_entire = np.concatenate(probs_entire, axis=0)
    samples = np.concatenate(samples, axis=0)

    # '''Visualize incorrect samples'''
    # indices_preds_2 = np.argwhere(preds_entire == 2).squeeze(axis=1)
    # labels_ = labels_entire[indices_preds_2]
    # indices_labels_0 = np.argwhere(labels_ == 0).squeeze(axis=1)
    # samples_2_to_0 = samples[indices_preds_2[indices_labels_0]]
    #
    # for k in range(samples_2_to_0.shape[0]):
    #     plt.figure(figsize=(20.5, 15.5))
    #     plt.plot(samples_2_to_0[k, 0, 0])
    #     plt.savefig(os.path.join(img_path, '2_to_0_{}.png'.format(k)), bbox_inches='tight')
    #     plt.close()

    Pp, Se = evaluator._sklean_metrics(y_pred=preds_entire,
                                       y_label=labels_entire)
    results = evaluator._metrics(predictions=preds_entire, labels=labels_entire)
    con_matrix = evaluator._confusion_matrix(y_pred=preds_entire,
                                             y_label=labels_entire)

    print('The overall accuracy is: {}'.format(results['Acc']))
    print("The confusion matrix is: ")
    print(con_matrix)
    print('The sklearn metrics are: ')
    print('Pp: ')
    pprint.pprint(Pp)
    print('Se: ')
    pprint.pprint(Se)
    print('The F1 score is: {}'.format(evaluator._f1_score(y_pred=preds_entire, y_true=labels_entire)))

    # for l in range(4):
    #     indices_l = np.argwhere(labels_entire == l).squeeze(axis=1)
    #     probs_l = probs_entire[indices_l]
    #
    #     plt.figure(figsize=(30, 20))
    #     plt.subplot(411)
    #     _, _, _ = plt.hist(probs_l[:, 0], bins=300)
    #     plt.subplot(412)
    #     _, _, _ = plt.hist(probs_l[:, 1], bins=300)
    #     plt.subplot(413)
    #     _, _, _ = plt.hist(probs_l[:, 2], bins=300)
    #     plt.subplot(414)
    #     _, _, _ = plt.hist(probs_l[:, 3], bins=300)
    #     plt.xticks(np.arange(0, 0.05, 1))
    #     plt.savefig(osp.join(img_path, '{}_hist_{}.png'.format(check_epoch, l)), bbox_inches='tight')
    #     plt.close()


if __name__ == '__main__':
    args = parser.parse_args()
    eval(args)

