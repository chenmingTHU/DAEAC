import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset_3d import *


# def init_centers(net, source, target, source_records, target_records,
#                  data_dict_source, data_dict_target, batch_size, num_workers, thrs):
#
#     dataset_source = MULTI_ECG_EVAL_DATASET(source, load_beat_with_rr,
#                                             data_dict_source, test_records=source_records)
#     dataloader_source = DataLoader(dataset_source, batch_size=batch_size, num_workers=num_workers)
#
#     features = {0: 0, 1: 0, 2: 0, 3: 0}
#     counters = {0: 0, 1: 0, 2: 0, 3: 0}
#
#     for idb, data_batch in enumerate(dataloader_source):
#         s_batch, sl_batch = data_batch
#         s_batch = s_batch.cuda()
#         sl_batch = sl_batch.numpy()
#
#         feat_s, _ = net(s_batch)
#
#         for l in range(4):
#             _index = np.argwhere(sl_batch == l)
#             if _index:
#                 counters[l] += len(_index)
#                 _index = np.squeeze(_index, axis=1)
#                 _feat = torch.index_select(feat_s, dim=0, index=torch.LongTensor(_index).cuda()).detach().cpu()
#                 _feat_sum = torch.sum(_feat, dim=0)
#                 features[l] += _feat_sum
#         torch.cuda.empty_cache()
#
#     dataset_target = MULTI_ECG_EVAL_DATASET(target, load_beat_with_rr,
#                                             data_dict_target, test_records=target_records)
#     dataloader_target = DataLoader(dataset_target, batch_size=batch_size, num_workers=num_workers)
#
#     for idb, data_batch in enumerate(dataloader_target):
#
#         t_batch, _ = data_batch
#         t_batch = t_batch.cuda()
#
#         feat_t, logits = net(t_batch)
#
#         probs = F.log_softmax(logits, dim=1).exp().detach().cpu().numpy()
#         max_probs = np.max(probs, axis=1)
#
#         indices = []
#         for l in range(4):
#             max_indices_l = np.argwhere(np.argmax(probs, axis=1) == l)
#             if max_indices_l:
#                 max_indices_l = np.squeeze(max_indices_l, axis=1)
#                 max_probs_l = max_probs[max_indices_l]
#                 legal_indices_l = np.argwhere(max_probs_l >= thrs[l])
#                 if legal_indices_l:
#                     legal_indices_l = np.squeeze(legal_indices_l, axis=1)
#                     indices_l = max_indices_l[legal_indices_l]
#                     indices.append(indices_l)
#
#         indices = np.sort(np.concatenate(indices))
#         print("batch index: {}, size of avaliable pesudo label: {}".format(idb, len(indices)))
#
#         if indices:
#             pesudo_labels = np.argmax(probs, axis=1)[indices]
#             feat_t = torch.index_select(feat_t, dim=0, index=torch.LongTensor(indices).cuda())
#
#             for l in range(4):
#                 _index = np.argwhere(pesudo_labels == l)
#                 if _index:
#                     counters[l] += len(_index)
#                     _index = np.squeeze(_index, axis=1)
#                     _feat = torch.index_select(feat_t, dim=0, index=torch.LongTensor(_index).cuda()).detach().cpu()
#                     _feat_sum = torch.sum(_feat, dim=0)
#                     features[l] += _feat_sum
#         torch.cuda.empty_cache()
#
#     for l in range(4):
#         features[l] = features[l] / counters[l]
#         features[l] = features[l].cuda()
#
#     return features


def init_source_centers(net, source, records, data_dict, batch_size, beat_num, fixed_len, num_workers, lead):

    dataset = MULTI_ECG_EVAL_DATASET(source,
                                     load_beat_with_rr,
                                     data_dict,
                                     test_records=records,
                                     beat_num=beat_num,
                                     fixed_len=fixed_len,
                                     lead=lead)

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers)

    features = {0: 0, 1: 0, 2: 0, 3: 0}
    counters = {0: 0, 1: 0, 2: 0, 3: 0}

    for idb, data_batch in enumerate(dataloader):

        s_batch, l_batch = data_batch
        s_batch = s_batch.cuda()

        l_batch = l_batch.numpy()

        feat, _ = net(s_batch)

        for l in range(4):

            _index = np.argwhere(l_batch == l)
            if len(_index) > 0:
                counters[l] += len(_index)
                _index = np.squeeze(_index, axis=1)
                _feat = torch.index_select(feat, dim=0, index=torch.LongTensor(_index).cuda()).detach().cpu()
                _feat_sum = torch.sum(_feat, dim=0)
                features[l] += _feat_sum
        torch.cuda.empty_cache()

    print('Procedure finished! Obtaining centers of source data!')
    for l in range(4):
        features[l] = features[l] / counters[l]
        features[l] = features[l].cuda()
        print(features[l].size())

    return features, counters


def init_target_centers(net, target, records, data_dict, batch_size, beat_num, fixed_len, num_workers, lead, thrs):

    dataset = MULTI_ECG_EVAL_DATASET(target,
                                     load_beat_with_rr,
                                     data_dict,
                                     test_records=records,
                                     beat_num=beat_num,
                                     fixed_len=fixed_len,
                                     lead=lead)

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers)

    features = {0: 0, 1: 0, 2: 0, 3: 0}
    counters = {0: 0, 1: 0, 2: 0, 3: 0}

    for idb, data_batch in enumerate(dataloader):

        s_batch, _ = data_batch
        s_batch = s_batch.cuda()

        feat, logits = net(s_batch)

        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
        max_probs = np.max(probs, axis=1)

        indices = []
        for l in range(4):
            max_indices_l = np.argwhere(np.argmax(probs, axis=1) == l)
            if len(max_indices_l) > 0:
                max_indices_l = np.squeeze(max_indices_l, axis=1)
                max_probs_l = max_probs[max_indices_l]
                legal_indices_l = np.argwhere(max_probs_l >= thrs[l])
                if len(legal_indices_l) > 0:
                    legal_indices_l = np.squeeze(legal_indices_l, axis=1)
                    indices_l = max_indices_l[legal_indices_l]
                    indices.append(indices_l)

        indices = np.sort(np.concatenate(indices))
        print("batch index: {}, size of avaliable pesudo label: {}".format(idb, len(indices)))

        if len(indices) > 0:
            pesudo_labels = np.argmax(probs, axis=1)[indices]
            feat = torch.index_select(feat, dim=0, index=torch.LongTensor(indices).cuda())

            for l in range(4):
                _index = np.argwhere(pesudo_labels == l)
                if len(_index) > 0:
                    counters[l] += len(_index)
                    _index = np.squeeze(_index, axis=1)
                    _feat = torch.index_select(feat, dim=0, index=torch.LongTensor(_index).cuda()).detach().cpu()
                    _feat_sum = torch.sum(_feat, dim=0)
                    features[l] += _feat_sum
        torch.cuda.empty_cache()

    print('Procedure finished! Obtaining centers of target data!')
    print("The numbers of available pesudo labels:")
    for l in range(4):
        if counters[l] > 0:
            print("{}: {}".format(l, counters[l]))
            # features[l] = torch.cat(features[l], dim=0)
            # features[l] = torch.mean(features[l], dim=0)
            features[l] = features[l] / counters[l]
            features[l] = features[l].cuda()
            print(features[l].size())
        else:
            del features[l]
            print('No avaliable centers')

    return features, counters


def init_entire_center(net, source, records, data_dict, batch_size, beat_num, fixed_len, num_workers, lead):
    dataset = MULTI_ECG_EVAL_DATASET(source,
                                     load_beat_with_rr,
                                     data_dict,
                                     test_records=records,
                                     beat_num=beat_num,
                                     fixed_len=fixed_len,
                                     lead=lead)

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers)

    features = 0
    counter = 0

    for idb, data_batch in enumerate(dataloader):

        s_batch, _ = data_batch
        s_batch = s_batch.cuda()

        feat, _ = net(s_batch)

        feat = feat.detach().cpu()
        counter += s_batch.size()[0]
        _feat_sum = torch.sum(feat, dim=0)
        features += _feat_sum

        torch.cuda.empty_cache()

    print('Procedure finished! Obtaining center of data!')

    features = features / counter
    features = features.cuda()
    print(features.size())

    return features

