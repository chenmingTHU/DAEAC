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

# FLAG_RESAMPLE = True
FLAG_RESAMPLE = False
if FLAG_RESAMPLE:
    DATA_PATH = '/home/workspace/mingchen/ECG_UDA/data'
    INDEX_ROOT_PATH = '/home/workspace/mingchen/ECG_UDA/data_index'
else:
    DATA_PATH = '/home/workspace/mingchen/ECG_UDA/raw_data'
    INDEX_ROOT_PATH = '/home/workspace/mingchen/ECG_UDA/raw_data_index'

DENOISE_DATA_DIRS = {
    'mitdb': DATA_PATH + "/mitdb/",
    'incartdb': DATA_PATH + "/incartdb",
    'svdb': DATA_PATH + "/svdb",
    'ltdb': DATA_PATH + "/ltdb"
}

SAMPLE_RATES = {
    'mitdb': 360,
    'incartdb': 257,
    'svdb': 128,
    'ltdb': 128
}
UNI_FS = 360

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


def load_dataset_to_memory(dataset_name):

    if dataset_name == 'fmitdb':
        dataset_name = 'mitdb'

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


def load_beat_with_rr(path, data_dir, n, fs, lead=1):
    information = np.load(path)
    record_id = str(information['record'])

    peak = information['peak']
    peaks = information['peaks']
    index = information['index']
    cls = information['cls']

    low_bound = peaks[index - n - 1]
    high_bound = peaks[index + n]

    left_points = int(0.14 * fs)
    right_points = int(0.28 * fs)

    if lead == 1:
        beat = data_dir[record_id + '.mat']['signal'][0, low_bound + left_points: high_bound + right_points + 1]
        beat = np.expand_dims(beat, axis=0)
    else:
        beat = data_dir[record_id + '.mat']['signal'][[0, 1], low_bound + left_points: high_bound + right_points + 1]

    max_v = np.max(beat, axis=1, keepdims=True)
    min_v = np.min(beat, axis=1, keepdims=True)
    beat = beat / (max_v - min_v)

    peak_prev = peaks[index - 1]
    peaks_10 = peaks[index - 10: index]
    rr_prev = peak - peak_prev
    rr_avg_10 = np.mean(np.array(peaks_10[1:] - peaks_10[:-1]))

    near_rr_prev_ratio = rr_prev / rr_avg_10

    peaks_prev_all = peaks[0: index]
    rr_avg_prev_all = np.mean(np.array(peaks_prev_all[1:] - peaks_prev_all[:-1]))

    rr_prev_ratio = rr_prev / rr_avg_prev_all

    return beat, cls, rr_prev_ratio, near_rr_prev_ratio


def augmentation_transform_with_rr(path, data_dir, n, fs, lead=1):
    information = np.load(path)
    record_id = str(information['record'])

    peak = information['peak']
    peaks = information['peaks']
    index = information['index']
    cls = information['cls']

    left_bound = 0.14 + 0.01 * np.random.rand(1)
    right_bound = 0.28 + 0.01 * np.random.rand(1)

    low_bound = peaks[index - n - 1]
    high_bound = peaks[index + n]

    left_points = int(left_bound * fs)
    right_points = int(right_bound * fs)

    if lead == 1:
        beat = data_dir[record_id + '.mat']['signal'][0, low_bound + left_points: high_bound + right_points + 1]
        beat = np.expand_dims(beat, axis=0)
    else:
        beat = data_dir[record_id + '.mat']['signal'][[0, 1], low_bound + left_points: high_bound + right_points + 1]
    beat = beat + 0.1 * np.max(beat) * np.random.randn(beat.shape[0], beat.shape[1])

    max_v = np.max(beat, axis=1, keepdims=True)
    min_v = np.min(beat, axis=1, keepdims=True)
    beat = beat / (max_v - min_v)

    peak_prev = peaks[index - 1]
    peaks_10 = peaks[index - 10: index]
    rr_prev = peak - peak_prev
    rr_avg_10 = np.mean(np.array(peaks_10[1:] - peaks_10[:-1]))

    near_rr_prev_ratio = rr_prev / rr_avg_10

    peaks_prev_all = peaks[0: index]
    rr_avg_prev_all = np.mean(np.array(peaks_prev_all[1:] - peaks_prev_all[:-1]))

    rr_prev_ratio = rr_prev / rr_avg_prev_all

    return beat, cls, rr_prev_ratio, near_rr_prev_ratio


class MULTI_ECG_TRAIN_DATASET(Dataset):

    def __init__(self, dataset_name, data_loader, data_dict, train_records,
                 transform=None, target_transform=None, beat_num=1, fixed_len=200,
                 lead=1):
        super(MULTI_ECG_TRAIN_DATASET, self).__init__()

        '''
        The ECG dataset
        dataset_name: The dataset used for training |mitdb|svdb|incartdb|
        data_loader: The function that loads a sample
        data_dict: All records loaded into memory (dict)
        dataset_split: which subset to use for training |entrie(default)|DS1|DS2|
        transform: Transformation function for a sample
        target_transform: Transformation function for labels
        '''

        root_index = INDEX_ROOT_PATH
        categories = ['N', 'V', 'S', 'F']

        self.n = beat_num
        if dataset_name == 'fmitdb':
            dataset_name = 'mitdb'

        self.dataset_name = dataset_name
        self.train_records = train_records
        self.loader = data_loader
        self.data = data_dict
        self.transform = transform
        self.target_transform = target_transform
        self.lead = lead

        self.load_path = osp.join(osp.join(root_index, dataset_name), 'entire')
        self.data_path = DENOISE_DATA_DIRS[dataset_name]

        self.fs = UNI_FS if FLAG_RESAMPLE else SAMPLE_RATES[dataset_name]

        self.samples = []

        for cate in categories:
            files = os.listdir(osp.join(self.load_path, cate))
            samples = [(file, cate) for file in files if file.split('_')[0] in train_records]
            self.samples.extend(samples)

        self.fixed_len = fixed_len

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        filename, cate = self.samples[index]
        file_path = osp.join(osp.join(self.load_path, cate), filename)

        transform = self.transform if (self.transform is not None) else self.loader
        beat, cls, rr_prev_ratio, near_rr_prev_ratio = transform(file_path, self.data,
                                                                 n=self.n, fs=self.fs, lead=self.lead)

        beat = ss.resample(beat, self.fixed_len, axis=1)

        beat = np.expand_dims(beat, axis=1)
        rr_prev_ratio = rr_prev_ratio * np.ones(shape=beat.shape)
        near_rr_prev_ratio = near_rr_prev_ratio * np.ones(shape=beat.shape)
        sample = np.concatenate([beat, rr_prev_ratio, near_rr_prev_ratio], axis=1).astype(np.float32)

        return sample, cls


class MULTI_ECG_EVAL_DATASET(Dataset):

    def __init__(self, dataset_name, data_loader, data_dict, test_records,
                 beat_num=1, fixed_len=200, lead=1, unlabel_num=300):
        super(MULTI_ECG_EVAL_DATASET, self).__init__()

        root_index = INDEX_ROOT_PATH
        categories = ['N', 'V', 'S', 'F']

        self.n = beat_num
        self.lead = lead
        if dataset_name == 'fmitdb':
            dataset_name = 'mitdb'

        self.dataset_name = dataset_name
        self.loader = data_loader
        self.data = data_dict

        self.load_path = osp.join(osp.join(root_index, dataset_name), 'entire')
        self.test_records = test_records

        self.samples = []

        for cate in categories:
            files = os.listdir(osp.join(self.load_path, cate))
            samples = [(file, cate) for file in files if file.split('_')[0] in self.test_records
                       and (int(file.split('.')[0].split('_')[1]) >= unlabel_num)]
            # samples = [(file, cate) for file in files if file.split('_')[0] in self.test_records]
            self.samples.extend(samples)

        self.fs = UNI_FS if FLAG_RESAMPLE else SAMPLE_RATES[dataset_name]
        self.fixed_len = fixed_len

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        filename, cate = self.samples[index]
        file_path = osp.join(osp.join(self.load_path, cate), filename)

        beat, cls, rr_prev_ratio, near_rr_prev_ratio = self.loader(file_path, self.data,
                                                                   n=self.n, fs=self.fs, lead=self.lead)

        beat = ss.resample(beat, self.fixed_len, axis=1)

        beat = np.expand_dims(beat, axis=1)
        rr_prev_ratio = rr_prev_ratio * np.ones(shape=beat.shape)
        near_rr_prev_ratio = near_rr_prev_ratio * np.ones(shape=beat.shape)
        sample = np.concatenate([beat, rr_prev_ratio, near_rr_prev_ratio], axis=1).astype(np.float32)

        return sample, cls


class UDA_DATASET(Dataset):

    def __init__(self,
                 source, target,
                 source_data, target_data,
                 source_records, target_records,
                 unlabel_num, data_loader,
                 transform=None, beat_num=1,
                 fixed_len=200, lead=1):
        super(UDA_DATASET, self).__init__()

        root_index = INDEX_ROOT_PATH
        categories = ['N', 'V', 'S', 'F']

        self.n = beat_num
        self.lead = lead

        if source == 'fmitdb':
            source = 'mitdb'

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

        self.fs_s = UNI_FS if FLAG_RESAMPLE else SAMPLE_RATES[source]
        self.fs_t = UNI_FS if FLAG_RESAMPLE else SAMPLE_RATES[target]

        self.fixed_len = fixed_len

        self.samples_s = []
        for cate in categories:
            files = os.listdir(osp.join(self.loadpath_s, cate))
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

        transform = self.transform if (self.transform is not None) else self.loader

        pack = transform(file_path_s, self.data_s, n=self.n, fs=self.fs_s, lead=self.lead)
        beat_s = ss.resample(pack[0], self.fixed_len, axis=1)
        beat_s = np.expand_dims(beat_s, axis=1)

        rr_ratio_s = np.ones(shape=beat_s.shape) * pack[2]
        near_rr_ratio_s = np.ones(shape=beat_s.shape) * pack[3]
        sample_s = np.concatenate([beat_s, rr_ratio_s, near_rr_ratio_s], axis=1).astype(np.float32)

        '''get target data'''

        index_unlabel = index % self.len_t

        filename_t, cate_t = self.samples_t[index_unlabel]
        file_path_t = osp.join(osp.join(self.loadpath_t, cate_t), filename_t)

        # beat_u, _ = self.loader(file_path_t, self.data_t, self.half_beat_len_t)
        pack_u = transform(file_path_t, self.data_t, n=self.n, fs=self.fs_t, lead=self.lead)
        beat_u = ss.resample(pack_u[0], self.fixed_len, axis=1)
        beat_u = np.expand_dims(beat_u, axis=1)

        rr_ratio_t = np.ones(shape=beat_u.shape) * pack_u[2]
        near_rr_ratio_t = np.ones(shape=beat_u.shape) * pack_u[3]
        sample_t = np.concatenate([beat_u, rr_ratio_t, near_rr_ratio_t], axis=1).astype(np.float32)

        cls_s = pack[1]
        cls_t = pack_u[1]

        return (sample_s, cls_s,
                sample_t, cls_t)

    def shuffle_target(self):

        random.shuffle(self.samples_t)
        random.shuffle(self.samples_s)


if __name__ == '__main__':

    # data_dict = load_dataset_to_memory(dataset_name='mitdb')
    pass
