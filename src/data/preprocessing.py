import numpy as np
import os.path as osp
import os

import wfdb
import scipy.io as sio
import scipy.signal as ss

import pywt


FLAG_RESAMPLE = True
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


class DataPreprocessing(object):

    def __init__(self, load_path, save_path, dataset_name):

        self.load_path = load_path
        self.save_path = save_path
        self.dataset_name = dataset_name
        self.fs = SAMPLE_RATES[dataset_name]

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.records = [record.split('.')[0] for record in os.listdir(self.load_path)
                        if len(record.split('.')) > 1 and record.split('.')[1] == 'dat']

    def wt_denoise(self, signal):

        wt_base = pywt.Wavelet('coif2')

        if len(signal.shape) == 1:

            coeffs = pywt.wavedec(signal, wt_base, level=8)
            zero_level = [0, 1]
            for x in zero_level:
                coeffs[x] = np.array([0] * len(coeffs[x]))
            denoised_signal = pywt.waverec(coeffs, wt_base)
            return denoised_signal

        elif len(signal.shape) == 2:

            res = []

            for k in range(signal.shape[0]):
                sig = signal[k]

                coeffs = pywt.wavedec(sig, wt_base, level=8)
                zero_level = [0, 1]
                for x in zero_level:
                    coeffs[x] = np.array([0] * len(coeffs[x]))
                denoised_sig = pywt.waverec(coeffs, wt_base)
                denoised_sig = np.expand_dims(np.array(denoised_sig), 0)
                res.append(denoised_sig)

            return np.concatenate(res, axis=0)

        else:
            print("Invalid input shape of wt_denoise")
            raise ValueError

    def median_denoise(self, signal):

        p_win = int(self.fs * 100.0 / 1000.0)
        t_win = 3 * p_win

        if len(signal.shape) == 1:

            b0 = [np.median(signal[max(0, x - p_win): min(x + p_win, len(signal) - 1)]) for x in range(len(signal))]
            b1 = [np.median(b0[max(0, x - t_win): min(x + t_win, len(b0) - 1)]) for x in range(len(b0))]

            denoised_signal = signal - b1

            return np.array(denoised_signal)

        elif len(signal.shape) == 2:

            res = []

            for k in range(signal.shape[0]):
                signal_ = signal[k]

                b0 = [np.median(signal_[max(0, x - p_win): min(x + p_win, len(signal_) - 1)]) for x in range(len(signal_))]
                b1 = [np.median(b0[max(0, x - t_win): min(x + t_win, len(b0) - 1)]) for x in range(len(b0))]
                b1 = np.array(b1)
                denoised_signal = signal_ - b1

                res.append(np.expand_dims(denoised_signal, axis=0))

            res = np.concatenate(res, axis=0)

            return res

        else:
            print('Invalid input shape for median_denoise')
            raise ValueError

    def resample(self, signal):

        assert len(signal.shape) == 2, "The dimension of signal is wrong!"
        signal = ss.resample(signal, int(signal.shape[1] * UNI_FS / self.fs), axis=1)
        return signal

    def run(self):

        for record in self.records:

            signal, info = wfdb.rdsamp(osp.join(self.load_path, record))
            signal = np.transpose(signal)

            ann = wfdb.rdann(osp.join(self.load_path, record), extension='atr')

            '''
            symbol: the category of each hearbeat in a ECG record
            coords: the location (of QRS peak) of each hearbeat in a ECG record
            '''
            symbol = ann.symbol
            coords = ann.sample

            print(np.unique(symbol))
            # print(dir(ann))
            # print(info)

            # denoised_signal = self.wt_denoise(self.median_denoise(signal))
            denoised_signal = self.median_denoise(signal)
            if FLAG_RESAMPLE:
                denoised_signal = self.resample(denoised_signal)

            print("Saving {}".format(record))
            sio.savemat(osp.join(self.save_path, record + '.mat'), {'signal': denoised_signal,
                                                                    'peaks': coords,
                                                                    'category': symbol})


def save_indices_of_datasets(dataset_name):

    N = ['N', 'L', 'R', 'e', 'j', '.']
    V = ['V', 'E']
    S = ['A', 'a', 'J', 'S']
    F = ['F']

    root_path = osp.join(osp.join(INDEX_ROOT_PATH, dataset_name), 'entire')
    if not osp.exists(root_path):
        os.makedirs(root_path)

    load_path = osp.join(DATA_PATH, dataset_name)
    records = os.listdir(load_path)

    save_path_dict = {'N': osp.join(root_path, 'N'),
                      'V': osp.join(root_path, 'V'),
                      'S': osp.join(root_path, 'S'),
                      'F': osp.join(root_path, 'F')}

    for key in save_path_dict.keys():
        if not osp.exists(save_path_dict[key]):
            os.makedirs(save_path_dict[key])

    for record in records:

        print("processing {}".format(record))

        data = sio.loadmat(osp.join(load_path, record))
        peaks = data['peaks'][0]
        if FLAG_RESAMPLE:
            peaks = (peaks * UNI_FS / SAMPLE_RATES[dataset_name]).astype(np.int32)

        category = data['category']

        print("min peak: {}; max peak: {}".format(np.min(peaks), np.max(peaks)))

        assert len(peaks) == len(category), "Unequal lengths of peaks and category"

        for index in range(10, len(peaks) - 10):

            peak = peaks[index]
            cate = category[index]

            if cate in N:
                np.savez(osp.join(save_path_dict['N'], "{}_{}.npz".format(record.split('.')[0], index)),
                         peak=peak, cls=0, record=record.split('.')[0], peaks=peaks, index=index)
            elif cate in V:
                np.savez(osp.join(save_path_dict['V'], "{}_{}.npz".format(record.split('.')[0], index)),
                         peak=peak, cls=1, record=record.split('.')[0], peaks=peaks, index=index)
            elif cate in S:
                np.savez(osp.join(save_path_dict['S'], "{}_{}.npz".format(record.split('.')[0], index)),
                         peak=peak, cls=2, record=record.split('.')[0], peaks=peaks, index=index)
            elif cate in F:
                np.savez(osp.join(save_path_dict['F'], "{}_{}.npz".format(record.split('.')[0], index)),
                         peak=peak, cls=3, record=record.split('.')[0], peaks=peaks, index=index)


def split_dataset(dataset_name='mitdb'):

    DS1 = ['101', '106', '108', '109', '112',
           '114', '115', '116', '118', '119',
           '122', '124', '201', '203', '205',
           '207', '208', '209', '215', '220',
           '223',  '230']

    DS2 = ['100', '103', '105', '111', '113',
           '117', '121', '123', '200', '202',
           '210', '212', '213', '214', '219',
           '221', '222', '228', '231', '232',
           '233', '234']

    datasets = {'DS1': DS1,
                'DS2': DS2}

    N = ['N', 'L', 'R', 'e', 'j', '.']
    V = ['V', 'E']
    S = ['A', 'a', 'J', 'S']
    F = ['F']

    root_path = osp.join(INDEX_ROOT_PATH, dataset_name)
    if not osp.exists(root_path):
        os.makedirs(root_path)

    load_path = osp.join(DATA_PATH, dataset_name)

    for ds_id, DS in datasets.items():

        root_path_DS = osp.join(root_path, ds_id)
        save_path_dict_DS = {'N': osp.join(root_path_DS, 'N'),
                             'V': osp.join(root_path_DS, 'V'),
                             'S': osp.join(root_path_DS, 'S'),
                             'F': osp.join(root_path_DS, 'F')}

        for key in save_path_dict_DS.keys():
            if not osp.exists(save_path_dict_DS[key]):
                os.makedirs(save_path_dict_DS[key])

        for record in DS:

            print("processing {}".format(record))

            data = sio.loadmat(osp.join(load_path, record + '.mat'))
            peaks = data['peaks'][0]
            if FLAG_RESAMPLE:
                peaks = (peaks * UNI_FS / SAMPLE_RATES[dataset_name]).astype(np.int32)
            category = data['category']

            print("min peak: {}; max peak: {}".format(np.min(peaks), np.max(peaks)))

            assert len(peaks) == len(category), "Unequal lengths of peaks and category"

            for index in range(10, len(peaks) - 10):

                peak = peaks[index]
                cate = category[index]

                if cate in N:
                    np.savez(osp.join(save_path_dict_DS['N'], "{}_{}.npz".format(record.split('.')[0], index)),
                             peak=peak, cls=0, record=record.split('.')[0], peaks=peaks, index=index)
                elif cate in V:
                    np.savez(osp.join(save_path_dict_DS['V'], "{}_{}.npz".format(record.split('.')[0], index)),
                             peak=peak, cls=1, record=record.split('.')[0], peaks=peaks, index=index)
                elif cate in S:
                    np.savez(osp.join(save_path_dict_DS['S'], "{}_{}.npz".format(record.split('.')[0], index)),
                             peak=peak, cls=2, record=record.split('.')[0], peaks=peaks, index=index)
                elif cate in F:
                    np.savez(osp.join(save_path_dict_DS['F'], "{}_{}.npz".format(record.split('.')[0], index)),
                             peak=peak, cls=3, record=record.split('.')[0], peaks=peaks, index=index)


if __name__ == '__main__':

    pro_mitdb = DataPreprocessing('/home/workspace/mingchen/ECG_DATA/mitdb/',
                                  DENOISE_DATA_DIRS['mitdb'], 'mitdb')
    pro_mitdb.run()
    save_indices_of_datasets('mitdb')
    split_dataset('mitdb')
