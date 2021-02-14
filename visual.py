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
from src.data.sampler import UDAImbalancedDatasetSampler, ImbalancedDatasetSampler
from src.build_model import loss_function, build_distance, \
    build_training_records, build_validation_records, build_acnn_models
from src.centers_acnn import *
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
                    choices=['mitdb', 'fmitdb', 'svdb', 'incartdb', 'ltdb'],
                    help='One can choose another dataset for validation '
                         'beyond the training setting up')
parser.add_argument('--distance', default=None, type=str,
                    choices=['Cosine', 'L2'])
parser.add_argument('--epochs', default=30, type=int,
                    help='The number of mini-batch to visualize')

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
    colors_s = {0: 'red', 1: 'blue', 2: 'green', 3: 'black'}
    colors_t = {0: 'lightcoral', 1: 'lightskyblue', 2: 'lightgreen', 3: 'gray'}
    categories = {0: 'N', 1: 'V', 2: 'S', 3: 'F'}
    metrics = {'Cosine': 'cosine',
               'L2': 'euclidean'}

    cfg_dir = args.config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_dir)
    cfg.freeze()

    source = cfg.SETTING.TRAIN_DATASET
    target = cfg.SETTING.TEST_DATASET
    if args.target:
        target = args.target

    batch_size = cfg.TRAIN.BATCH_SIZE
    distance = args.distance if args.distance else cfg.SETTING.DISTANCE

    exp_id = os.path.basename(cfg_dir).split('.')[0]
    save_path = os.path.join(cfg.SYSTEM.SAVE_PATH, exp_id)

    img_path = os.path.join('./figures', exp_id)
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    check_epoch = args.check_epoch
    check_point_dir = osp.join(save_path, '{}.pkl'.format(check_epoch))

    data_dict_source = load_dataset_to_memory(source)
    data_dict_target = load_dataset_to_memory(target)
    source_records = build_training_records(source)
    target_records = build_validation_records(target)

    net = build_acnn_models(cfg.SETTING.NETWORK, cfg.SETTING.ASPP_BN,
                            cfg.SETTING.ASPP_ACT, cfg.SETTING.LEAD,
                            cfg.PARAMETERS.P, cfg.SETTING.DILATIONS,
                            act_func=cfg.SETTING.ACT, f_act_func=cfg.SETTING.F_ACT,
                            apply_residual=cfg.SETTING.RESIDUAL,
                            bank_num=cfg.SETTING.BANK_NUM)
    net.load_state_dict(torch.load(check_point_dir)['model_state_dict'])
    net.cuda()
    net.eval()

    dataset_source = MULTI_ECG_EVAL_DATASET(source, load_beat_with_rr,
                                            data_dict_source, test_records=source_records,
                                            beat_num=cfg.SETTING.BEAT_NUM,
                                            fixed_len=cfg.SETTING.FIXED_LEN,
                                            lead=cfg.SETTING.LEAD)
    dataloader_source = DataLoader(dataset_source, batch_size=batch_size,
                                   num_workers=cfg.SYSTEM.NUM_WORKERS,
                                   sampler=ImbalancedDatasetSampler(dataset_source)
                                   )

    dataset_target = MULTI_ECG_EVAL_DATASET(target, load_beat_with_rr,
                                            data_dict_target, test_records=target_records,
                                            beat_num=cfg.SETTING.BEAT_NUM,
                                            fixed_len=cfg.SETTING.FIXED_LEN,
                                            lead=cfg.SETTING.LEAD)
    dataloader_target = DataLoader(dataset_target, batch_size=batch_size,
                                   num_workers=cfg.SYSTEM.NUM_WORKERS,
                                   sampler=ImbalancedDatasetSampler(dataset_target)
                                   )

    features_source = []
    features_target = []
    labels_source = []
    labels_target = []

    source_logits = []
    target_logits = []
    source_probs = []
    target_probs = []

    raw_signals_source = []
    raw_signals_target = []

    with torch.no_grad():

        tsne = manifold.TSNE(n_components=2, metric=metrics[distance], perplexity=30,
                             early_exaggeration=4.0, learning_rate=500.0,
                             n_iter=2000, init='pca',
                             random_state=2389)

        for idb, data_batch in enumerate(dataloader_source):
            s_batch, l_batch = data_batch
            s_batch_cpu = s_batch.detach().numpy()
            s_batch = s_batch.cuda()

            features_s, logits_s = net(s_batch)

            # feats = net.get_feature_maps(s_batch)
            # feats = feats.detach().cpu().numpy()
            # plt.figure(figsize=(12.5, 10))
            # plt.plot(feats[0])
            # plt.savefig(osp.join(img_path, '{}.png'.format(idb)), bbox_inches='tight')
            # plt.close()

            feat_s_cpu = features_s.detach().cpu().numpy()
            logits_s_cpu = logits_s.detach().cpu().numpy()
            probs_s = F.log_softmax(logits_s, dim=1).exp().detach().cpu().numpy()

            source_logits.append(logits_s_cpu)
            source_probs.append(probs_s)
            features_source.append(feat_s_cpu)
            raw_signals_source.append(s_batch_cpu)
            labels_source.append(l_batch)

            if idb == args.epochs - 1:
                break

        for idb, data_batch in enumerate(dataloader_target):
            s_batch, l_batch = data_batch
            s_batch_cpu = s_batch.detach().numpy()
            s_batch = s_batch.cuda()

            features_t, logits_t = net(s_batch)

            feat_t_cpu = features_t.detach().cpu().numpy()
            logits_t_cpu = logits_t.detach().cpu().numpy()
            probs_t = F.log_softmax(logits_t, dim=1).exp().detach().cpu().numpy()

            target_logits.append(logits_t_cpu)
            target_probs.append(probs_t)
            features_target.append(feat_t_cpu)
            raw_signals_target.append(s_batch_cpu)
            labels_target.append(l_batch)

            if idb == args.epochs - 1:
                break

        labels_source = np.concatenate(labels_source, axis=0)
        labels_target = np.concatenate(labels_target, axis=0)

        # target_probs = np.concatenate(target_probs, axis=0)
        # preds_t = np.argmax(target_probs, axis=1)
        # probs_t = np.max(target_probs, axis=1)
        #
        # for l in range(4):
        #     indices_tl = np.argwhere(preds_t == l)
        #     if len(indices_tl) > 0:
        #         indices_tl = indices_tl.squeeze(axis=1)
        #         probs_tl = probs_t[indices_tl]
        #         gt_tl = labels_target[indices_tl]
        #         indices_l = np.where(gt_tl == l, 1, 0)
        #
        #         plt.figure(figsize=(20, 15))
        #         n, bins, patches = plt.hist(probs_tl, bins=300)
        #         plt.savefig(osp.join(img_path, 'cls_{}.png'.format(l)), bbox_inches='tight')
        #         plt.close()
        #
        #         corr_indices_l = np.argwhere(indices_l == 1)
        #         incorr_indices_l = np.argwhere(indices_l == 0)
        #
        #         if len(corr_indices_l):
        #             plt.figure(figsize=(20, 15))
        #             corr_indices_l = corr_indices_l.squeeze(axis=1)
        #             corr_probs_tl = probs_tl[corr_indices_l]
        #             _, _, _ = plt.hist(corr_probs_tl, bins=300)
        #             plt.savefig(osp.join(img_path, 'corr_cls{}.png'.format(l)), bbox_inches='tight')
        #             plt.close()
        #         if len(incorr_indices_l):
        #             plt.figure(figsize=(20, 15))
        #             incorr_indices_l = incorr_indices_l.squeeze(axis=1)
        #             incorr_probs_tl = probs_tl[incorr_indices_l]
        #             _, _, _ = plt.hist(incorr_probs_tl, bins=300, color='red')
        #             plt.savefig(osp.join(img_path, 'incorr_cls{}.png'.format(l)), bbox_inches='tight')
        #             plt.close()

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
        feat_tsne = tsne.fit_transform(features)

        x_min, x_max = feat_tsne.min(0), feat_tsne.max(0)
        feat_norm = (feat_tsne - x_min) / (x_max - x_min)

        feat_norm_s = feat_norm[0: num_source]
        feat_norm_t = feat_norm[num_source: num_target + num_source]

        features_s_dict = {}
        feat_norm_s_dict = {}
        features_t_dict = {}
        feat_norm_t_dict = {}
        for l in range(4):
            l_indices = np.argwhere(labels_source == l).squeeze(axis=1)
            features_s_dict[l] = features_source[l_indices]
            feat_norm_s_dict[l] = feat_norm_s[l_indices]

            l_indices_t = np.argwhere(labels_target == l).squeeze(axis=1)

            features_t_dict[l] = features_target[l_indices_t]
            feat_norm_t_dict[l] = feat_norm_t[l_indices_t]

        '''The feature visualization'''
        # plt.figure(figsize=(30, 15))
        # for i in range(features_source.shape[0]):
        #     if labels_source[i] == 0:
        #         plt.subplot(411)
        #         plt.plot(features_source[i], color=colors[labels_source[i]])
        #     elif labels_source[i] == 1:
        #         plt.subplot(412)
        #         plt.plot(features_source[i], color=colors[labels_source[i]])
        #     elif labels_source[i] == 2:
        #         plt.subplot(413)
        #         plt.plot(features_source[i], color=colors[labels_source[i]])
        #     else:
        #         plt.subplot(414)
        #         plt.plot(features_source[i], color=colors[labels_source[i]])
        # img_save_path = osp.join(img_path, 'feat_s_{}_{}.png'.format(exp_id, args.check_epoch))
        # plt.savefig(img_save_path, bbox_inches='tight')
        # plt.close()
        #
        # plt.figure(figsize=(30, 15))
        # for i in range(features_target.shape[0]):
        #     if labels_target[i] == 0:
        #         plt.subplot(411)
        #         plt.plot(features_target[i], color=colors[labels_target[i]])
        #     elif labels_target[i] == 1:
        #         plt.subplot(412)
        #         plt.plot(features_target[i], color=colors[labels_target[i]])
        #     elif labels_target[i] == 2:
        #         plt.subplot(413)
        #         plt.plot(features_target[i], color=colors[labels_target[i]])
        #     else:
        #         plt.subplot(414)
        #         plt.plot(features_target[i], color=colors[labels_target[i]])
        # img_save_path = osp.join(img_path, 'feat_t_{}_{}.png'.format(exp_id, args.check_epoch))
        # plt.savefig(img_save_path, bbox_inches='tight')
        # plt.close()

        '''The class-specific view'''

        if 'mitdb' in target:
            plt.figure(figsize=(20, 20))
            for l in range(4):
                plt.scatter(feat_norm_s_dict[l][:, 0], feat_norm_s_dict[l][:, 1],
                            marker='o', color=colors_s[l], label='source {}'.format(categories[l]))
                plt.scatter(feat_norm_t_dict[l][:, 0], feat_norm_t_dict[l][:, 1],
                            marker='X', color=colors_t[l], label='target {}'.format(categories[l]))
            plt.xticks([])
            plt.yticks([])
            plt.legend(loc='upper right', fontsize=30)
            img_save_path = osp.join(img_path, 'tsne_{}_{}_cls.png'.format(exp_id, args.check_epoch))
            plt.savefig(img_save_path, bbox_inches='tight')
            plt.close()

        else:
            plt.figure(figsize=(20, 20))
            for l in range(3):
                plt.scatter(feat_norm_s_dict[l][:, 0], feat_norm_s_dict[l][:, 1],
                            marker='o', color=colors_s[l], label='source {}'.format(categories[l]))
                plt.scatter(feat_norm_t_dict[l][:, 0], feat_norm_t_dict[l][:, 1],
                            marker='X', color=colors_t[l], label='target {}'.format(categories[l]))
            plt.xticks([])
            plt.yticks([])
            plt.legend(loc='upper right', fontsize=30)
            img_save_path = osp.join(img_path, 'tsne_{}_{}_cls.png'.format(exp_id, args.check_epoch))
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
        # img_save_path = 'figures/tsne_{}_{}_dom.png'.format(exp_id, args.check_epoch)
        # plt.savefig(img_save_path, bbox_inches='tight')
        # plt.close()


if __name__ == '__main__':
    args = parser.parse_args()
    vis(args)

