import numpy as np
import scipy.io as sio
import os
import os.path as osp

from src.data.multibeat_dataset import *
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def L2(x1, x2):
    dist = np.sum(np.square(x1 - x2))
    return dist


def L1(x1, x2):
    dist = np.sum(np.abs(x1 - x2))
    return dist


class DensityBasedClustering(Dataset):

    def __init__(self, data_dict, data_loader, dist_func, dataset, records):
        super(DensityBasedClustering, self).__init__()
        
        root_index = '/home/workspace/mingchen/ECG_UDA/data_index'
        categories = ['N', 'V', 'S', 'F']

        self.dc = 0.08

        self.distance = dist_func
        self.dataset = dataset
        self.records = records

        self.loader = data_loader
        self.data = data_dict

        self.load_path = osp.join(osp.join(root_index, dataset), 'entire')
        self.data_path = DENOISE_DATA_DIRS[dataset]

        self.samples = {}

        for cate in categories:
            files = os.listdir(osp.join(self.load_path, cate))
            samples = [file for file in files if file.split('_')[0] in records]
            self.samples[cate] = samples

        self.fs = SAMPLE_RATES[dataset]
        self.half_beat_len = int(self.fs * 0.7)
        self.fixed_len = 400

        self.train_data = []
        for cate in categories:
            train_data_cate = self.clustering(cate)
            print('The size of {} is {}'.format(cate, train_data_cate.shape))
            self.train_data.append(train_data_cate)
        self.train_data = np.concatenate(self.train_data)

    def _get_samples(self, cate):

        files_cate = self.samples[cate]
        samples_cate = []

        for index in range(len(files_cate)):

            filename = files_cate[index]
            file_path = osp.join(osp.join(self.load_path, cate), filename)

            beat, cls, rrs = self.loader(file_path, self.data, self.half_beat_len)
            beat = ss.resample(beat, self.fixed_len).astype(np.float32)

            samples_cate.append((beat, cls, rrs))

        return samples_cate

    def _get_distance_matirx(self, samples_cate):

        num = len(samples_cate)
        dist_mat = np.zeros(shape=(num, num))

        for i in range(num):
            for j in range(num):
                beat_1 = samples_cate[i][0]
                beat_2 = samples_cate[j][0]
                dist_mat[i][j] = self.distance(beat_1, beat_2)

        return dist_mat

    def clustering(self, cate):

        samples_cate = self._get_samples(cate)

        dist_mat = self._get_distance_matirx(samples_cate)
        ros = np.zeros(shape=(len(samples_cate),))
        deltas = np.zeros(shape=(len(samples_cate),))

        for i in range(len(samples_cate)):
            Xs = np.where(dist_mat[i] - self.dc < 0, 1, 0)
            ros[i] = np.sum(Xs)

        for i in range(len(samples_cate)):
            ro = ros[i]
            indices = np.argwhere(ros > ro)

            if len(indices) > 0:
                indices = np.squeeze(indices)
                ros_j = dist_mat[i][indices]
                deltas[i] = np.min(ros_j)
            else:
                continue

        gamma = np.multiply(ros, deltas)
        sorted_gamma = np.flip(np.sort(gamma))
        sorted_index = np.flip(np.argsort(gamma))

        # fig = plt.figure(0)
        # fig.set_size_inches(22.5, 12.5)
        # plt.scatter(np.arange(len(sorted_gamma)), sorted_gamma)
        # plt.savefig('../figures/decision_graph_{}.png'.format(cate), bbox_inches='tight')
        # plt.close()

        center_indices = []
        cluster_indices = []

        max_gamma = sorted_gamma[0]
        for index in range(len(sorted_gamma)):
            gamma_index = sorted_gamma[index]
            if gamma_index > max_gamma * 0.3:
                center_indices.append(sorted_index[index])
                cluster_indices.append([sorted_index[index]])
            else:
                break

        cluster_num = len(center_indices)
        for index in sorted_index:
            distances = dist_mat[index][center_indices]
            min_index = np.argmin(distances)
            cluster_indices[min_index].append(index)

        cluster_core_indices = []

        for i in range(cluster_num):
            cluster = np.array(cluster_indices[i])
            gammas = gamma[cluster]

            max_gamma_c = np.max(gammas)
            cores = np.argwhere(gammas > max_gamma_c * 0.1)
            cores = np.squeeze(cores)

            core_indices = cluster[cores]
            cluster_core_indices.append(core_indices)

        cluster_core_indices = np.concatenate(cluster_core_indices)

        samples_cate = np.array(samples_cate)

        return samples_cate[cluster_core_indices]

    def __len__(self):

        return len(self.train_data)

    def __getitem__(self, item):

        beat, cls, rrs = self.train_data[item]

        if cls == 0:
            bin_cls = 0
        else:
            bin_cls = 1

        return beat, cls, rrs, bin_cls


if __name__ == '__main__':

    data_dict = load_dataset_to_memory('mitdb')

    db_cluster = DensityBasedClustering(data_dict, load_beat_with_rr,
                                        L2, 'mitdb', DS1)

    samples = db_cluster.clustering(cate='N')
    print(samples.shape)
