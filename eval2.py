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
from src.utils import *
from src.config import get_cfg_defaults

import argparse
import pprint


parser = argparse.ArgumentParser()
parser.add_argument('--config', default=None, type=str,
                    help='The directory of config .yaml file')
parser.add_argument('--check_epoch', default=-1, type=int,
                    help='The checkpoint ID for recovering training procedure')
parser.add_argument('--target', dest='target',
                    default=None, type=str,
                    choices=['mitdb', 'fmitdb', 'svdb', 'incartdb'],
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

    check_epoch = args.check_epoch
    check_point_dir = osp.join(save_path, '{}.pkl'.format(check_epoch))

    data_dict_target = load_dataset_to_memory(target)
    target_records = build_validation_records(target)

    net = build_cnn_models(cfg.SETTING.NETWORK,
                           fixed_len=cfg.SETTING.FIXED_LEN,
                           p=cfg.PARAMETERS.P)
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
                                     fixed_len=cfg.SETTING.FIXED_LEN)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=cfg.SYSTEM.NUM_WORKERS)

    print("The size of the validation dataset is {}".format(len(dataset)))

    preds_entire = []
    labels_entire = []

    with torch.no_grad():
        for idb, data_batch in enumerate(dataloader):

            s_batch, l_batch, r_batch, b_batch = data_batch

            s_batch = s_batch.unsqueeze(dim=1)
            s_batch = s_batch.cuda()
            r_batch = r_batch.cuda()
            l_batch = l_batch.numpy()

            _, _, _, logits = net(s_batch, r_batch)

            preds_softmax = F.log_softmax(logits, dim=1).exp()
            preds_softmax_np = preds_softmax.detach().cpu().numpy()
            preds = np.argmax(preds_softmax_np, axis=1)

            preds_entire.append(preds)
            labels_entire.append(l_batch)

            torch.cuda.empty_cache()

    preds_entire = np.concatenate(preds_entire, axis=0)
    labels_entire = np.concatenate(labels_entire, axis=0)

    results = evaluator._metrics(predictions=preds_entire,
                                 labels=labels_entire)

    Pp, Se = evaluator._sklean_metrics(y_pred=preds_entire,
                                       y_label=labels_entire)

    con_matrix = evaluator._confusion_matrix(y_pred=preds_entire,
                                             y_label=labels_entire)

    pprint.pprint(results)

    print("The confusion matrix is: ")
    print(con_matrix)
    print('The sklearn metrics are: ')
    print('Pp: ')
    pprint.pprint(Pp)
    print('Se: ')
    pprint.pprint(Se)
    print('The F1 score is: {}'.format(evaluator._f1_score(y_pred=preds_entire, y_true=labels_entire)))


if __name__ == '__main__':
    args = parser.parse_args()
    eval(args)

