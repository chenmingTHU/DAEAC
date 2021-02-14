import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.multibeat_dataset import *


def init_source_centers(net, source, records, data_dict, batch_size, beat_num, fixed_len, num_workers):

    dataset = MULTI_ECG_EVAL_DATASET(source,
                                     load_beat_with_rr,
                                     data_dict,
                                     beat_num=beat_num,
                                     fixed_len=fixed_len,
                                     test_records=records)

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers)

    features = {0: 0, 1: 0, 2: 0, 3: 0}
    counters = {0: 0, 1: 0, 2: 0, 3: 0}

    for idb, data_batch in enumerate(dataloader):

        s_batch, l_batch, rrs_batch, bin_batch = data_batch

        s_batch = s_batch.unsqueeze(dim=1)
        s_batch = s_batch.cuda()
        rrs_batch = rrs_batch.cuda()

        l_batch = l_batch.numpy()

        _, _, feat, _ = net(s_batch, rrs_batch)

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


def init_target_centers(net, target, records, data_dict, batch_size, beat_num, fixed_len, num_workers, thrs):

    dataset = MULTI_ECG_EVAL_DATASET(target,
                                     load_beat_with_rr,
                                     data_dict,
                                     beat_num=beat_num,
                                     fixed_len=fixed_len,
                                     test_records=records)

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers)

    # features = {0: [], 1: [], 2: [], 3: []}
    features = {0: 0, 1: 0, 2: 0, 3: 0}
    counters = {0: 0, 1: 0, 2: 0, 3: 0}

    for idb, data_batch in enumerate(dataloader):

        s_batch, _, rrs_batch, _ = data_batch

        s_batch = s_batch.unsqueeze(dim=1)
        s_batch = s_batch.cuda()
        rrs_batch = rrs_batch.cuda()

        _, _, feat, logits = net(s_batch, rrs_batch)

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
            print('No avaliable centers')
            raise ValueError

    return features, counters


def init_entire_center(net, source, records, data_dict, batch_size, beat_num, fixed_len, num_workers):
    dataset = MULTI_ECG_EVAL_DATASET(source,
                                     load_beat_with_rr,
                                     data_dict,
                                     beat_num=beat_num,
                                     fixed_len=fixed_len,
                                     test_records=records)

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers)

    features = 0
    counter = 0

    for idb, data_batch in enumerate(dataloader):

        s_batch, _, rrs_batch, _ = data_batch

        s_batch = s_batch.unsqueeze(dim=1)
        s_batch = s_batch.cuda()
        rrs_batch = rrs_batch.cuda()

        _, _, feat, _ = net(s_batch, rrs_batch)

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

