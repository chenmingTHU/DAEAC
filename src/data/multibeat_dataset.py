import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

import os.path as osp
import os
import numpy as np

import scipy.io as sio
import scipy.signal as ss
import pywt

import time
import random

from sklearn.cluster import DBSCAN


import matplotlib.pyplot as plt
plt.switch_backend('Agg')


DENOISE_DATA_DIRS = {
    'mitdb': "/home/workspace/mingchen/ECG_UDA/data/mitdb/",
    'incartdb': "/home/workspace/mingchen/ECG_UDA/data/incartdb",
    'svdb': "/home/workspace/mingchen/ECG_UDA/data/svdb"
}

SAMPLE_RATES = {
    'mitdb': 360,
    'incartdb': 257,
    'svdb': 128,
}

'''MITDB dataset division'''

DS1 = ['101', '106', '108', '109', '112',
       '114', '115', '116', '118', '119',
       '122', '124', '201', '203', '205',
       '207', '208', '209', '215', '220',
       '223', '230']

DS2 = ['100', '103', '105', '111', '113',
       '117', '121', '123', '200', '202',
       '210', '212', '213', '214', '219',
       '221', '222', '228', '231', '232',
       '233', '234']

DS_VEB = ['200', '202', '210', '213', '214',
          '219', '221', '228', '231', '233',
          '234']

DS_SVEB = ['200', '202', '210', '212', '213',
           '214', '219', '221', '222', '228',
           '231', '232', '233', '234']

DS_COM = ['201', '203', '205', '207', '208',
          '209', '215', '220', '223', '230',
          '200', '202', '210', '212', '213',
          '214', '219', '221', '222', '228',
          '231', '232', '233', '234']

DS_ENTIRE = DS1 + DS2


def _get_core_samples(dataset_name, records, root_index, cate, n, fs):

    data_dict = load_dataset_to_memory(dataset_name)

    load_path = osp.join(osp.join(root_index, dataset_name), 'entire')

    files = os.listdir(osp.join(load_path, cate))
    samples = [(file, cate) for file in files if file.split('_')[0] in records]
    samples = np.array(samples)

    beats = []
    for filename, _ in samples:
        file_path = osp.join(osp.join(load_path, cate), filename)
        beat, _, _ = load_beat_with_rr(file_path, data_dict, None, n=n, fs=fs)
        beat = np.expand_dims(ss.resample(beat, 200), axis=0)
        beats.append(beat)

    beats = np.concatenate(beats, axis=0)
    db = DBSCAN(eps=0.45, min_samples=10).fit(beats)
    core_sample_indices = db.core_sample_indices_
    print('The number of core samples of category {} is {}'.format(cate, len(core_sample_indices)))

    return samples[core_sample_indices]


def load_dataset_to_memory(dataset_name):

    load_path = DENOISE_DATA_DIRS[dataset_name]
    records = os.listdir(load_path)

    data_dict = {}

    for record in records:
        data = sio.loadmat(osp.join(load_path, record))
        signal = data['signal']
        category = data['category'][0]
        peaks = data['peaks'][0]

        data_dict[record] = {'signal': signal,
                             'category': category,
                             'peaks': peaks}

    return data_dict


def load_beat_with_rr(path, data_dir, half_beat_len, n, fs):
    information = np.load(path)
    record_id = str(information['record'])

    peak = information['peak']
    peaks = information['peaks']
    index = information['index']
    cls = information['cls']

    peak_prev = peaks[index - 1]
    peak_post = peaks[index + 1]

    low_bound = peaks[index - n - 1]
    high_bound = peaks[index + n]

    left_points = int(0.14 * fs)
    right_points = int(0.28 * fs)

    raw_segment = data_dir[record_id + '.mat']['signal'][0, low_bound + left_points: high_bound + right_points + 1]

    beat = raw_segment
    # beat = beat - np.mean(beat)
    max_v = np.max(beat)
    min_v = np.min(beat)
    beat = beat / (max_v - min_v)
    # beat = (beat - np.mean(beat, axis=1, keepdims=True)) / np.std(beat, axis=1, keepdims=True)

    peaks_10 = peaks[index - 10: index]
    rr_prev = peak - peak_prev
    rr_post = peak_post - peak
    rr_avg_10 = np.mean(np.array(peaks_10[1:] - peaks_10[:-1]))
    rr_avg_all = np.mean(peaks[1: index + 1] - peaks[0: index])

    rrs = np.array([rr_prev / rr_avg_10, rr_post / rr_avg_10,
                    rr_prev / rr_avg_all, rr_post / rr_avg_all]).astype(np.float32)

    return beat, cls, rrs


def augmentation_transform_with_rr(path, data_dir, half_beat_len, n, fs):
    information = np.load(path)
    record_id = str(information['record'])

    peak = information['peak']
    peaks = information['peaks']
    index = information['index']
    cls = information['cls']

    peak_prev = peaks[index - 1]
    peak_post = peaks[index + 1]

    left_bound = 0.14 + 0.01 * np.random.randn(1)
    right_bound = 0.28 + 0.01 * np.random.randn(1)

    left_points = int(left_bound * fs)
    right_points = int(right_bound * fs)

    low_bound = peaks[index - n - 1]
    high_bound = peaks[index + n]

    raw_segment = data_dir[record_id + '.mat']['signal'][0, low_bound + left_points: high_bound + right_points + 1]

    beat = raw_segment
    # beat = beat - np.mean(beat)
    max_v = np.max(beat)
    min_v = np.min(beat)
    beat = beat / (max_v - min_v)
    # beat = (beat - np.mean(beat, axis=1, keepdims=True)) / np.std(beat, axis=1, keepdims=True)

    peaks_10 = peaks[index - 10: index]
    rr_prev = peak - peak_prev
    rr_post = peak_post - peak
    rr_avg_10 = np.mean(np.array(peaks_10[1:] - peaks_10[:-1]))
    rr_avg_all = np.mean(peaks[1: index + 1] - peaks[0: index])

    rrs = np.array([rr_prev / rr_avg_10, rr_post / rr_avg_10,
                    rr_prev / rr_avg_all, rr_post / rr_avg_all]).astype(np.float32)

    return beat, cls, rrs


class MULTI_ECG_TRAIN_DATASET(Dataset):

    def __init__(self, dataset_name, data_loader, data_dict, train_records,
                 transform=None, target_transform=None, beat_num=1, fixed_len=200,
                 use_dbscan=False):
        super(MULTI_ECG_TRAIN_DATASET, self).__init__()

        '''
        The ECG dataset
        dataset_name: The dataset used for training |mitdb|svdb|incartdb|
        data_loader: The function that loads a sample
        data_dict: All records loaded into memory (dict)
        dataset_split: which subset to use for training |entrie(default)|DS1|DS2|
        transform: Transformation function for a sample
        target_transform: Transformation function for labels
        
        Notice: the parameter (use_dbscan) is only available for N
        '''

        root_index = '/home/workspace/mingchen/ECG_UDA/data_index'
        categories = ['N', 'V', 'S', 'F']

        self.n = beat_num

        self.dataset_name = dataset_name
        self.train_records = train_records
        self.loader = data_loader
        self.data = data_dict
        self.transform = transform
        self.target_transform = target_transform

        self.load_path = osp.join(osp.join(root_index, dataset_name), 'entire')
        self.data_path = DENOISE_DATA_DIRS[dataset_name]
        self.fs = SAMPLE_RATES[dataset_name]

        self.samples = []

        for cate in categories:
            files = os.listdir(osp.join(self.load_path, cate))
            if cate == 'N' and use_dbscan:
                samples = _get_core_samples(dataset_name, train_records,
                                            root_index, cate, self.n, self.fs)
            else:
                samples = [(file, cate) for file in files if file.split('_')[0] in train_records]
            self.samples.extend(samples)

        self.half_beat_len = int(self.fs * 0.7)
        self.fixed_len = fixed_len

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        filename, cate = self.samples[index]
        file_path = osp.join(osp.join(self.load_path, cate), filename)

        if self.transform is not None:
            beat, cls, rrs = self.transform(file_path, self.data, self.half_beat_len, n=self.n, fs=self.fs)
        else:
            beat, cls, rrs = self.loader(file_path, self.data, self.half_beat_len, n=self.n, fs=self.fs)

        beat = ss.resample(beat, self.fixed_len).astype(np.float32)

        if cls == 0:
            bin_cls = 0
        else:
            bin_cls = 1

        return beat, cls, rrs, bin_cls


class MULTI_ECG_EVAL_DATASET(Dataset):

    def __init__(self, dataset_name, data_loader, data_dict, test_records, beat_num=1, fixed_len=200):
        super(MULTI_ECG_EVAL_DATASET, self).__init__()

        root_index = '/home/workspace/mingchen/ECG_UDA/data_index'
        categories = ['N', 'V', 'S', 'F']
        self.n = beat_num

        self.dataset_name = dataset_name
        self.loader = data_loader
        self.data = data_dict

        self.load_path = osp.join(osp.join(root_index, dataset_name), 'entire')
        self.test_records = test_records

        self.samples = []

        for cate in categories:
            files = os.listdir(osp.join(self.load_path, cate))
            samples = [(file, cate) for file in files if file.split('_')[0] in self.test_records]
            self.samples.extend(samples)

        self.fs = SAMPLE_RATES[dataset_name]
        self.half_beat_len = int(self.fs * 0.7)
        self.fixed_len = fixed_len

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        filename, cate = self.samples[index]
        file_path = osp.join(osp.join(self.load_path, cate), filename)

        beat, cls, rrs = self.loader(file_path, self.data, self.half_beat_len, n=self.n, fs=self.fs)

        beat = ss.resample(beat, self.fixed_len).astype(np.float32)

        if cls == 0:
            bin_cls = 0
        else:
            bin_cls = 1

        return beat, cls, rrs, bin_cls


class UDA_DATASET(Dataset):

    def __init__(self,
                 source, target,
                 source_data, target_data,
                 source_records, target_records,
                 unlabel_num, data_loader,
                 transform=None, beat_num=1,
                 fixed_len=200, use_dbscan=False):
        super(UDA_DATASET, self).__init__()

        root_index = '/home/workspace/mingchen/ECG_UDA/data_index'
        categories = ['N', 'V', 'S', 'F']

        self.n = beat_num

        self.source = source
        self.target = target
        self.data_s = source_data
        self.data_t = target_data

        self.records_s = source_records
        self.records_t = target_records

        self.num_unlabeled = unlabel_num

        self.loadpath_s = osp.join(osp.join(root_index, source), 'entire')
        self.loadpath_t = osp.join(osp.join(root_index, target), 'entire')

        self.loader = data_loader
        self.transform = transform
        self.target_transform = transform

        self.fs_s = SAMPLE_RATES[source]
        self.half_beat_len_s = int(self.fs_s * 0.7)

        self.fs_t = SAMPLE_RATES[target]
        self.half_beat_len_t = int(self.fs_t * 0.7)

        self.fixed_len = fixed_len

        self.samples_s = []
        for cate in categories:
            files = os.listdir(osp.join(self.loadpath_s, cate))
            if use_dbscan and cate == 'N':
                samples = _get_core_samples(source, self.records_s,
                                            root_index, cate, n=self.n, fs=self.fs_s)
            else:
                samples = [(file, cate) for file in files if file.split('.')[0].split('_')[0] in self.records_s]
            self.samples_s.extend(samples)

        self.samples_t = []
        for cate in categories:
            files = os.listdir(osp.join(self.loadpath_t, cate))
            samples = [(file, cate) for file in files if (file.split('.')[0].split('_')[0] in self.records_t)
                       and (int(file.split('.')[0].split('_')[1]) < unlabel_num)]
            self.samples_t.extend(samples)

        self.len_s = len(self.samples_s)
        self.len_t = len(self.samples_t)

        random.shuffle(self.samples_t)

    def __len__(self):

        return self.len_s

    def __getitem__(self, index):
        '''get source data'''

        filename_s, cate_s = self.samples_s[index]
        file_path_s = osp.join(osp.join(self.loadpath_s, cate_s), filename_s)

        if self.transform is not None:
            pack = self.transform(file_path_s, self.data_s, self.half_beat_len_s, n=self.n, fs=self.fs_s)
        else:
            pack = self.loader(file_path_s, self.data_s, self.half_beat_len_s, n=self.n, fs=self.fs_s)
        beat_s = ss.resample(pack[0], self.fixed_len).astype(np.float32)

        '''get target data'''

        index_unlabel = index % self.len_t

        filename_t, cate_t = self.samples_t[index_unlabel]
        file_path_t = osp.join(osp.join(self.loadpath_t, cate_t), filename_t)

        # beat_u, _ = self.loader(file_path_t, self.data_t, self.half_beat_len_t)
        if self.transform is not None:
            pack_u = self.transform(file_path_t, self.data_t, self.half_beat_len_t, n=self.n, fs=self.fs_t)
        else:
            pack_u = self.loader(file_path_t, self.data_t, self.half_beat_len_t, n=self.n, fs=self.fs_t)
        beat_u = ss.resample(pack_u[0], self.fixed_len).astype(np.float32)

        cls_s = pack[1]
        rrs_s = pack[2]
        cls_bin_s = 0 if cls_s == 0 else 1

        cls_t = pack_u[1]
        rrs_t = pack_u[2]
        cls_bin_t = 0 if cls_t == 0 else 1

        return (beat_s, cls_s, rrs_s, cls_bin_s,
                beat_u, cls_t, rrs_t, cls_bin_t)

    def shuffle_target(self):

        random.shuffle(self.samples_t)
        random.shuffle(self.samples_s)


if __name__ == '__main__':

    # data_dict = load_dataset_to_memory(dataset_name='mitdb')
    pass
