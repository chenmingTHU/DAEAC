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


def load_beat(path, data_dir, half_beat_len):

    information = np.load(path)
    record_id = str(information['record'])
    peak = information['peak']
    cls = information['cls']

    Sigmoid_L = 1.0 / (1 + np.exp(
        -(np.array(np.arange(0, half_beat_len + 1)) - 0.6 * half_beat_len) * 0.1))

    Sigmoid_R = np.ones((half_beat_len,)) - 1.0 / (
            1 + np.exp(-(np.array(np.arange(0, half_beat_len)) - 0.6 * half_beat_len) * 0.1))

    Sigmoid_Win = np.concatenate([Sigmoid_L, Sigmoid_R])

    raw_segment = data_dir[record_id + '.mat']['signal'][0, peak - half_beat_len: peak + half_beat_len + 1]

    beat = np.multiply(raw_segment, Sigmoid_Win)
    max_v = np.max(beat)
    min_v = np.min(beat)
    beat = beat / (max_v - min_v)

    return beat, cls


def load_beat_with_rr(path, data_dir, half_beat_len):
    information = np.load(path)
    record_id = str(information['record'])

    peak = information['peak']
    peaks = information['peaks']
    index = information['index']
    cls = information['cls']

    Sigmoid_L = 1.0 / (1 + np.exp(
        -(np.array(np.arange(0, half_beat_len + 1)) - 0.6 * half_beat_len) * 0.1))

    Sigmoid_R = np.ones((half_beat_len,)) - 1.0 / (
            1 + np.exp(-(np.array(np.arange(0, half_beat_len)) - 0.6 * half_beat_len) * 0.1))

    Sigmoid_Win = np.concatenate([Sigmoid_L, Sigmoid_R])

    raw_segment = data_dir[record_id + '.mat']['signal'][0, peak - half_beat_len: peak + half_beat_len + 1]

    beat = np.multiply(raw_segment, Sigmoid_Win)
    max_v = np.max(beat)
    min_v = np.min(beat)
    beat = beat / (max_v - min_v)

    peak_prev = peaks[index - 1]
    peak_post = peaks[index + 1]
    peaks_10 = peaks[index - 10: index]
    rr_prev = peak - peak_prev
    rr_post = peak_post - peak
    rr_avg_10 = np.mean(np.array(peaks_10[1:] - peaks_10[:-1]))
    rrs = np.array([rr_prev, rr_post, rr_avg_10]).astype(np.float32)

    return beat, cls, rrs


def augmentation_transform(path, data_dir, half_beat_len):

    modes = np.array([0, 1, 2])
    mode = np.random.choice(modes, size=1)

    information = np.load(path)
    record_id = str(information['record'])
    peak = information['peak']
    cls = information['cls']

    if mode == 0:

        bias = np.random.randint(low=5, high=10, size=1).astype(np.float32)
        sign = np.random.choice(np.array([-1, 1]), size=1)

        half_beat_len_ = int(half_beat_len * (1 + sign * bias / 100.0))

        Sigmoid_L = 1.0 / (1 + np.exp(
            -(np.array(np.arange(0, half_beat_len_ + 1)) - 0.6 * half_beat_len_) * 0.1))

        Sigmoid_R = np.ones((half_beat_len_,)) - 1.0 / (
                1 + np.exp(-(np.array(np.arange(0, half_beat_len_)) - 0.6 * half_beat_len_) * 0.1))

        Sigmoid_Win = np.concatenate([Sigmoid_L, Sigmoid_R])

        raw_segment = data_dir[record_id + '.mat']['signal'][0, peak - half_beat_len_: peak + half_beat_len_ + 1]

        assert len(raw_segment) == len(Sigmoid_Win), "{}, {}".format(peak, half_beat_len_)

        beat = np.multiply(raw_segment, Sigmoid_Win)
        max_v = np.max(beat)
        min_v = np.min(beat)
        beat = beat / (max_v - min_v)

        return beat, cls

    elif mode == 1:

        bias_l = np.random.randint(low=5, high=10, size=1).astype(np.float32)
        sign_l = np.random.choice(np.array([-1, 1]), size=1)

        bias_r = np.random.randint(low=5, high=10, size=1).astype(np.float32)
        sign_r = np.random.choice(np.array([-1, 1]), size=1)

        half_beat_len_l = int(half_beat_len * (1 + sign_l * bias_l / 100.0))
        half_beat_len_r = int(half_beat_len * (1 + sign_r * bias_r / 100.0))

        Sigmoid_L = 1.0 / (1 + np.exp(
            -(np.array(np.arange(0, half_beat_len_l + 1)) - 0.6 * half_beat_len_l) * 0.1))

        Sigmoid_R = np.ones((half_beat_len_r,)) - 1.0 / (
                1 + np.exp(-(np.array(np.arange(0, half_beat_len_r)) - 0.6 * half_beat_len_r) * 0.1))

        Sigmoid_Win = np.concatenate([Sigmoid_L, Sigmoid_R])

        raw_segment = data_dir[record_id + '.mat']['signal'][0, peak - half_beat_len_l: peak + half_beat_len_r + 1]

        assert len(raw_segment) == len(Sigmoid_Win), "{}, {}, {}".format(peak,
                                                                         half_beat_len_l,
                                                                         half_beat_len_r)

        beat = np.multiply(raw_segment, Sigmoid_Win)
        max_v = np.max(beat)
        min_v = np.min(beat)
        beat = beat / (max_v - min_v)
        return beat, cls

    elif mode == 2:

        Sigmoid_L = 1.0 / (1 + np.exp(
            -(np.array(np.arange(0, half_beat_len + 1)) - 0.6 * half_beat_len) * 0.1))

        Sigmoid_R = np.ones((half_beat_len,)) - 1.0 / (
                1 + np.exp(-(np.array(np.arange(0, half_beat_len)) - 0.6 * half_beat_len) * 0.1))

        Sigmoid_Win = np.concatenate([Sigmoid_L, Sigmoid_R])

        wt_bases = pywt.wavelist(kind='discrete')
        wt_base_name = np.random.choice(wt_bases, size=1)
        # wt_base = pywt.Wavelet('coif2')
        wt_base = pywt.Wavelet(wt_base_name[0])
        # print(wt_base_name)

        record_signal = data_dir[record_id + '.mat']['signal'][0]
        coeffs = pywt.wavedec(record_signal, wt_base, level=11)
        disturb_level = [0, 1, 2, 3]
        for x in disturb_level:
            coeffs[x] = coeffs[x] + 0.5 * np.random.randn(len(coeffs[x]))
        disturbed_record_signal = pywt.waverec(coeffs, wt_base)

        raw_segment = disturbed_record_signal[peak - half_beat_len: peak + half_beat_len + 1]

        assert len(raw_segment) == len(Sigmoid_Win), "{}, {}".format(peak, half_beat_len)

        beat = np.multiply(raw_segment, Sigmoid_Win)
        max_v = np.max(beat)
        min_v = np.min(beat)
        beat = beat / (max_v - min_v)

        return beat, cls


def augmentation_transform_with_rr(path, data_dir, half_beat_len):

    # modes = np.array([0, 1])
    # mode = np.random.choice(modes, size=1)

    mode = 0

    information = np.load(path)
    record_id = str(information['record'])
    peak = information['peak']
    peaks = information['peaks']
    index = information['index']
    cls = information['cls']

    peak_prev = peaks[index - 1]
    peak_post = peaks[index + 1]
    peaks_10 = peaks[index - 10: index]
    rr_prev = peak - peak_prev
    rr_post = peak_post - peak
    rr_avg_10 = np.mean(np.array(peaks_10[1:] - peaks_10[:-1]))
    rrs = np.array([rr_prev, rr_post, rr_avg_10]).astype(np.float32)

    if mode == 0:

        Sigmoid_L = 1.0 / (1 + np.exp(
                    -(np.array(np.arange(0, half_beat_len + 1)) - 0.6 * half_beat_len) * 0.1))

        Sigmoid_R = np.ones((half_beat_len,)) - 1.0 / (
                    1 + np.exp(-(np.array(np.arange(0, half_beat_len)) - 0.6 * half_beat_len) * 0.1))

        Sigmoid_Win = np.concatenate([Sigmoid_L, Sigmoid_R])

        raw_segment = data_dir[record_id + '.mat']['signal'][0, peak - half_beat_len: peak + half_beat_len + 1]

        beat = np.multiply(raw_segment, Sigmoid_Win) + np.random.randn(len(raw_segment)) * 0.1
        max_v = np.max(beat)
        min_v = np.min(beat)
        beat = beat / (max_v - min_v)

        return beat, cls, rrs

    # elif mode == 1:
    #
    #     Sigmoid_L = 1.0 / (1 + np.exp(
    #         -(np.array(np.arange(0, half_beat_len + 1)) - 0.6 * half_beat_len) * 0.1))
    #
    #     Sigmoid_R = np.ones((half_beat_len,)) - 1.0 / (
    #             1 + np.exp(-(np.array(np.arange(0, half_beat_len)) - 0.6 * half_beat_len) * 0.1))
    #
    #     Sigmoid_Win = np.concatenate([Sigmoid_L, Sigmoid_R])
    #
    #     wt_bases = pywt.wavelist(kind='discrete')
    #     wt_base_name = np.random.choice(wt_bases, size=1)
    #     # wt_base = pywt.Wavelet('coif2')
    #     wt_base = pywt.Wavelet(wt_base_name[0])
    #
    #     record_signal = data_dir[record_id + '.mat']['signal'][0]
    #     coeffs = pywt.wavedec(record_signal, wt_base, level=11)
    #     disturb_level = [0, 1, 2, 3]
    #     for x in disturb_level:
    #         coeffs[x] = coeffs[x] + 0.5 * np.random.randn(len(coeffs[x]))
    #     disturbed_record_signal = pywt.waverec(coeffs, wt_base)
    #
    #     raw_segment = disturbed_record_signal[peak - half_beat_len: peak + half_beat_len + 1]
    #
    #     assert len(raw_segment) == len(Sigmoid_Win), "{}, {}".format(peak, half_beat_len)
    #
    #     beat = np.multiply(raw_segment, Sigmoid_Win)
    #     max_v = np.max(beat)
    #     min_v = np.min(beat)
    #     beat = beat / (max_v - min_v)
    #
    #     return beat, cls, rrs


class ECG_TRAIN_DATASET(Dataset):

    def __init__(self, dataset_name, data_loader, data_dict, train_records,
                 transform=None, target_transform=None):
        super(ECG_TRAIN_DATASET, self).__init__()

        '''
        The ECG dataset
        dataset_name: The dataset used for training |mitdb|svdb|incartdb|
        data_loader: The function that loads a sample
        data_dict: All records loaded into memory (dict)
        dataset_split: which subset to use for training |entrie(default)|DS1|DS2|
        transform: Transformation function for a sample
        target_transform: Transformation function for labels
        '''

        root_index = '/home/workspace/mingchen/ECG_UDA/data_index'
        categories = ['N', 'V', 'S', 'F']

        self.dataset_name = dataset_name
        self.train_records = train_records
        self.loader = data_loader
        self.data = data_dict
        self.transform = transform
        self.target_transform = target_transform

        self.load_path = osp.join(osp.join(root_index, dataset_name), 'entire')
        self.data_path = DENOISE_DATA_DIRS[dataset_name]

        self.samples = []

        for cate in categories:
            files = os.listdir(osp.join(self.load_path, cate))
            samples = [(file, cate) for file in files if file.split('_')[0] in train_records]
            self.samples.extend(samples)

        self.fs = SAMPLE_RATES[dataset_name]
        self.half_beat_len = int(self.fs * 0.7)
        self.fixed_len = 400

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        filename, cate = self.samples[index]
        file_path = osp.join(osp.join(self.load_path, cate), filename)

        if self.transform is not None:
            beat, cls = self.transform(file_path, self.data, self.half_beat_len)

        else:
            beat, cls = self.loader(file_path, self.data, self.half_beat_len)

        beat = ss.resample(beat, self.fixed_len).astype(np.float32)

        return beat, cls


class MULTI_ECG_TRAIN_DATASET(Dataset):

    def __init__(self, dataset_name, data_loader, data_dict, train_records,
                 transform=None, target_transform=None):
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

        root_index = '/home/workspace/mingchen/ECG_UDA/data_index'
        categories = ['N', 'V', 'S', 'F']

        self.dataset_name = dataset_name
        self.train_records = train_records
        self.loader = data_loader
        self.data = data_dict
        self.transform = transform
        self.target_transform = target_transform

        self.load_path = osp.join(osp.join(root_index, dataset_name), 'entire')
        self.data_path = DENOISE_DATA_DIRS[dataset_name]

        self.samples = []

        for cate in categories:
            files = os.listdir(osp.join(self.load_path, cate))
            samples = [(file, cate) for file in files if file.split('_')[0] in train_records]
            self.samples.extend(samples)

        self.fs = SAMPLE_RATES[dataset_name]
        self.half_beat_len = int(self.fs * 0.7)
        self.fixed_len = 400

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        filename, cate = self.samples[index]
        file_path = osp.join(osp.join(self.load_path, cate), filename)

        if self.transform is not None:
            beat, cls, rrs = self.transform(file_path, self.data, self.half_beat_len)

        else:
            beat, cls, rrs = self.loader(file_path, self.data, self.half_beat_len)

        beat = ss.resample(beat, self.fixed_len).astype(np.float32)

        if cls == 0:
            bin_cls = 0
        else:
            bin_cls = 1

        return beat, cls, rrs, bin_cls


class MULTI_ECG_EVAL_DATASET(Dataset):

    def __init__(self, dataset_name, data_loader, data_dict, test_records):
        super(MULTI_ECG_EVAL_DATASET, self).__init__()

        root_index = '/home/workspace/mingchen/ECG_UDA/data_index'
        categories = ['N', 'V', 'S', 'F']

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
        self.fixed_len = 400

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        filename, cate = self.samples[index]
        file_path = osp.join(osp.join(self.load_path, cate), filename)

        beat, cls, rrs = self.loader(file_path, self.data, self.half_beat_len)

        beat = ss.resample(beat, self.fixed_len).astype(np.float32)

        if cls == 0:
            bin_cls = 0
        else:
            bin_cls = 1

        return beat, cls, rrs, bin_cls


class ECG_EVAL_DATASET(Dataset):

    def __init__(self, dataset_name, data_loader, data_dict, test_records):
        super(ECG_EVAL_DATASET, self).__init__()

        root_index = '/home/workspace/mingchen/ECG_UDA/data_index'
        categories = ['N', 'V', 'S', 'F']

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
        self.fixed_len = 400

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        filename, cate = self.samples[index]
        file_path = osp.join(osp.join(self.load_path, cate), filename)

        beat, cls, rrs = self.loader(file_path, self.data, self.half_beat_len)

        beat = ss.resample(beat, self.fixed_len).astype(np.float32)

        return beat, cls, rrs


class UDA_DATASET(Dataset):

    def __init__(self,
                 source, target,
                 source_data, target_data,
                 source_records, target_records,
                 unlabel_num, data_loader,
                 transform=None):
        super(UDA_DATASET, self).__init__()

        root_index = '/home/workspace/mingchen/ECG_UDA/data_index'
        categories = ['N', 'V', 'S', 'F']

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

        self.fixed_len = 400

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

        if self.transform is not None:
            pack = self.transform(file_path_s, self.data_s, self.half_beat_len_s)
        else:
            pack = self.loader(file_path_s, self.data_s, self.half_beat_len_s)
        beat_s = ss.resample(pack[0], self.fixed_len).astype(np.float32)

        '''get target data'''

        index_unlabel = index % self.len_t

        filename_t, cate_t = self.samples_t[index_unlabel]
        file_path_t = osp.join(osp.join(self.loadpath_t, cate_t), filename_t)

        # beat_u, _ = self.loader(file_path_t, self.data_t, self.half_beat_len_t)
        if self.transform is not None:
            pack_u = self.transform(file_path_t, self.data_t, self.half_beat_len_t)
        else:
            pack_u = self.loader(file_path_t, self.data_t, self.half_beat_len_t)
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

    data_dict = load_dataset_to_memory(dataset_name='mitdb')
