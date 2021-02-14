import torch
import torch.utils.data
import torchvision


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
        super(ImbalancedDatasetSampler, self).__init__(dataset)

        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        label_to_count = {}

        for idx in self.indices:
            # self._get_label return a category
            label = self._get_label(dataset, idx)
            if label in label_to_count.keys():
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.samples[idx][1]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class UDAImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
        super(UDAImbalancedDatasetSampler, self).__init__(dataset)

        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        label_to_count = {}

        for idx in self.indices:
            # self._get_label return a category
            label = self._get_label(dataset, idx)
            if label in label_to_count.keys():
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.samples_s[idx][1]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

