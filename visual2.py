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
from src.data.sampler import UDAImbalancedDatasetSampler, ImbalancedDatasetSampler
from src.build_model import loss_function, build_distance, \
    build_training_records, build_validation_records, build_cnn_models
from src.utils import *
from src.config import get_cfg_defaults

from sklearn import manifold

import argparse
import pprint


parser = argparse.ArgumentParser()
parser.add_argument('--config', default=None, type=str,
                    help='The directory of config .yaml file')
parser.add_argument('--check_epoch', default=0, type=int,
                    help='The checkpoint ID for recovering training procedure')
parser.add_argument('--target', dest='target',
                    default=None, type=str,
                    choices=['mitdb', 'fmitdb', 'svdb', 'incartdb'],
                    help='One can choose another dataset for validation '
                         'beyond the training setting up')

parser.add_argument('--epochs', default=30, type=int,
                    help='The number of mini-batch to visualize')
parser.add_argument('--component', default='entire', type=str,
                    choices=['entire', 'wave', 'pef'])

# parser.add_argument('--class', dest='cls',
#                     action='store_true',
#                     help='One can visualize the feature maps in class-specific view')
# parser.add_argument('--source_only', dest='so',
#                     action='store_true',
#                     help='One can only visualize the feature maps from source domain(training dataset)')
# parser.add_argument('--target_only', dest='to',
#                     action='store_true',
#                     help='One can only visualize the feature maps from target domain(validation dataset)')


args = parser.parse_args()

ROOT_PATH = '/home/workspace/mingchen/ECG_UDA/data_index'


def vis(args):
    colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'black'}
    metrics = {'Cosine': 'cosine',
               'L2': 'euclidean'}

    cfg_dir = args.config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_dir)
    cfg.freeze()
    print(cfg)

    source = cfg.SETTING.TRAIN_DATASET
    target = cfg.SETTING.TEST_DATASET
    if args.target:
        target = args.target

    batch_size = cfg.TRAIN.BATCH_SIZE
    distance = cfg.SETTING.DISTANCE

    exp_id = os.path.basename(cfg_dir).split('.')[0]
    save_path = os.path.join(cfg.SYSTEM.SAVE_PATH, exp_id)

    check_epoch = args.check_epoch
    check_point_dir = osp.join(save_path, '{}.pkl'.format(check_epoch))

    data_dict_source = load_dataset_to_memory(source)
    data_dict_target = load_dataset_to_memory(target)
    source_records = build_training_records(source)
    target_records = build_validation_records(target)

    net = build_cnn_models(cfg.SETTING.NETWORK, cfg.SETTING.FIXED_LEN)
    net.load_state_dict(torch.load(check_point_dir)['model_state_dict'])
    net.cuda()
    net.eval()

    dataset_source = MULTI_ECG_EVAL_DATASET(source, load_beat_with_rr,
                                            data_dict_source, test_records=source_records,
                                            beat_num=cfg.SETTING.BEAT_NUM,
                                            fixed_len=cfg.SETTING.FIXED_LEN)
    dataloader_source = DataLoader(dataset_source, batch_size=batch_size,
                                   num_workers=cfg.SYSTEM.NUM_WORKERS,
                                   sampler=ImbalancedDatasetSampler(dataset_source))

    dataset_target = MULTI_ECG_EVAL_DATASET(target, load_beat_with_rr,
                                            data_dict_target, test_records=target_records,
                                            beat_num=cfg.SETTING.BEAT_NUM,
                                            fixed_len=cfg.SETTING.FIXED_LEN)
    dataloader_target = DataLoader(dataset_target, batch_size=batch_size,
                                   num_workers=cfg.SYSTEM.NUM_WORKERS,
                                   sampler=ImbalancedDatasetSampler(dataset_target))

    features_source = []
    features_target = []
    labels_source = []
    labels_target = []

    source_logits = []
    target_logits = []

    raw_signals_source = []
    raw_signals_target = []

    with torch.no_grad():

        tsne = manifold.TSNE(n_components=2, metric=metrics[distance], perplexity=30,
                             early_exaggeration=4.0, learning_rate=500.0,
                             n_iter=2000, init='pca',
                             random_state=2389)

        for idb, data_batch in enumerate(dataloader_source):
            s_batch, l_batch, sr_batch, _ = data_batch
            s_batch_cpu = s_batch.detach().numpy()
            s_batch = s_batch.unsqueeze(dim=1)
            s_batch = s_batch.cuda()
            sr_batch = sr_batch.cuda()

            _, _, features_s, logits_s = net(s_batch, sr_batch)

            feat_s_cpu = features_s.detach().cpu().numpy()
            logits_s_cpu = logits_s.detach().cpu().numpy()

            source_logits.append(logits_s_cpu)
            features_source.append(feat_s_cpu)
            raw_signals_source.append(s_batch_cpu)
            labels_source.append(l_batch)

            if idb == args.epochs - 1:
                break

        for idb, data_batch in enumerate(dataloader_target):
            s_batch, l_batch, tr_batch, _ = data_batch
            s_batch_cpu = s_batch.detach().numpy()
            s_batch = s_batch.unsqueeze(dim=1)
            s_batch = s_batch.cuda()
            tr_batch = tr_batch.cuda()

            _, _, features_t, logits_t = net(s_batch, tr_batch)

            feat_t_cpu = features_t.detach().cpu().numpy()
            logits_t_cpu = logits_t.detach().cpu().numpy()

            target_logits.append(logits_t_cpu)
            features_target.append(feat_t_cpu)
            raw_signals_target.append(s_batch_cpu)
            labels_target.append(l_batch)

            if idb == args.epochs - 1:
                break

        labels_source = np.concatenate(labels_source, axis=0)
        labels_target = np.concatenate(labels_target, axis=0)

        labels = np.concatenate([labels_source, labels_target], axis=0)

        count_source = {'N': 0, 'V': 0, 'S': 0, 'F': 0}
        count_target = {'N': 0, 'V': 0, 'S': 0, 'F': 0}
        keys = ['N', 'V', 'S', 'F']

        num_source = len(labels_source)
        num_target = len(labels_target)

        for i in range(num_source):
            count_source[keys[labels_source[i]]] += 1
        for j in range(num_target):
            count_target[keys[labels_target[j]]] += 1

        for k in keys:
            print('The number of {} in source: {}; in target: {}'.format(k, count_source[k], count_target[k]))

        features_source = np.concatenate(features_source, axis=0)
        features_target = np.concatenate(features_target, axis=0)

        features = np.concatenate([features_source, features_target], axis=0)

        if args.component == 'entire':
            features = features
        elif args.component == 'wave':
            features = features[:, 0: 512]
        else:
            features = features[:, 512:]

        feat_tsne = tsne.fit_transform(features)

        x_min, x_max = feat_tsne.min(0), feat_tsne.max(0)
        feat_norm = (feat_tsne - x_min) / (x_max - x_min)

        '''The class-specific view'''
        plt.figure(figsize=(20, 20))
        for i in range(feat_norm.shape[0]):
            if i < num_source:
                plt.scatter(feat_norm[i, 0], feat_norm[i, 1],
                            marker='.', color=colors[labels[i]])
            else:
                plt.scatter(feat_norm[i, 0], feat_norm[i, 1],
                            marker='x', color=colors[labels[i]])
        plt.xticks([])
        plt.yticks([])
        img_save_path = 'figures/tsne_{}_{}_{}_cls.png'.format(exp_id, args.check_epoch, args.component)
        plt.savefig(img_save_path, bbox_inches='tight')
        plt.close()

        '''The domain-specific view'''
        # plt.figure(figsize=(20, 20))
        # for i in range(feat_norm.shape[0]):
        #     if i < num_source:
        #         plt.scatter(feat_norm[i, 0], feat_norm[i, 1],
        #                     marker='.', color='red')
        #     else:
        #         plt.scatter(feat_norm[i, 0], feat_norm[i, 1],
        #                     marker='x', color='blue')
        # plt.xticks([])
        # plt.yticks([])
        # img_save_path = 'figures/tsne_{}_{}_{}_dom.png'.format(exp_id, args.check_epoch, args.component)
        # plt.savefig(img_save_path, bbox_inches='tight')
        # plt.close()


if __name__ == '__main__':
    args = parser.parse_args()
    vis(args)

