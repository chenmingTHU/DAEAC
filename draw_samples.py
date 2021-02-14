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

    result_files = os.listdir(res_path)
    result_files = [filename for filename in result_files if filename.split('.')[1] == 'npz']

    F1 = []
    SE = []
    PP = []
    checkepochs = []

    for filename in result_files:
        checkepoch = int(filename.split('.')[0].split('_')[1])

        if checkepoch >= 0:
            checkepochs.append(checkepoch)
            data = np.load(os.path.join(res_path, filename))
            se = data['Se']
            pp = data['Pp']
            f1 = data['F1']

            SE.append(se)
            PP.append(pp)
            F1.append(f1)

    F1 = np.stack(F1, axis=1)
    SE = np.stack(SE, axis=1)
    PP = np.stack(PP, axis=1)
    checkepochs = np.sort(np.array(checkepochs))

    plt.figure(figsize=(20, 15))
    plt.plot(checkepochs, F1[0], color='red', linestyle='-', label='N')
    plt.plot(checkepochs, F1[1], color='blue', linestyle='-', label='V')
    plt.plot(checkepochs, F1[2], color='green', linestyle='-', label='S')
    plt.plot(checkepochs, F1[3], color='orange', linestyle='-', label='F')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(res_path, 'F1.png'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(20, 15))
    plt.plot(checkepochs, SE[0], color='red', linestyle='-', label='N')
    plt.plot(checkepochs, SE[1], color='blue', linestyle='-', label='V')
    plt.plot(checkepochs, SE[2], color='green', linestyle='-', label='S')
    plt.plot(checkepochs, SE[3], color='orange', linestyle='-', label='F')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(res_path, 'sen.png'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(20, 15))
    plt.plot(checkepochs, PP[0], color='red', linestyle='-', label='N')
    plt.plot(checkepochs, PP[1], color='blue', linestyle='-', label='V')
    plt.plot(checkepochs, PP[2], color='green', linestyle='-', label='S')
    plt.plot(checkepochs, PP[3], color='orange', linestyle='-', label='F')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(res_path, 'pre.png'), bbox_inches='tight')
    plt.close()


def draw_samples():
    data_dict_target = load_dataset_to_memory('mitdb')

    dataset = MULTI_ECG_EVAL_DATASET('mitdb',
                                     load_beat_with_rr,
                                     data_dict_target,
                                     test_records=DS1,
                                     beat_num=0,
                                     fixed_len=200,
                                     lead=1, unlabel_num=300)
    dataloader = DataLoader(dataset, batch_size=1,
                            num_workers=1)

    for idx, data_batch in enumerate(dataloader):

        s_batch, l_batch = data_batch
        s_batch = s_batch.numpy().squeeze()

        plt.figure(idx, figsize=(20.5, 12.5))
        plt.plot(s_batch[0])
        plt.savefig('./figures/samples/len1_{}.png'.format(idx), bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    draw_samples()

