import os
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
from torch.utils.data.sampler import Sampler
from torchvision.transforms import Compose
import logging


class PICAIDataset(Dataset):
    """ PICAI Dataset with multi-modal MRI and segmentation mask """

    def __init__(self, data_dir, list_dir, split, reverse=False, logging=logging):
        self.data_dir = data_dir
        self.list_dir = list_dir
        self.split = split
        self.reverse = reverse

        tr_transform = Compose([
            RandomCrop((256, 256, 20)),
            ToTensor()
        ])
        test_transform = Compose([
            CenterCrop((256, 256, 20)),
            ToTensor()
        ])

        if split == 'train_lab':
            data_path = os.path.join(list_dir, 'train_lab.txt')
            self.transform = tr_transform
        elif split == 'train_unlab':
            data_path = os.path.join(list_dir, 'train_unlab.txt')
            self.transform = tr_transform
            print("unlab transform")
        else:
            data_path = os.path.join(list_dir, 'test.txt')
            self.transform = test_transform

        with open(data_path, 'r') as f:
            self.image_list = f.read().splitlines()

        self.image_list = [os.path.join(self.data_dir, pid, f"{pid}.h5") for pid in self.image_list]

        logging.info("{} set: total {} samples".format(split, len(self.image_list)))
        logging.info("total {} samples".format(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx % len(self.image_list)]
        if self.reverse:
            image_path = self.image_list[len(self.image_list) - idx % len(self.image_list) - 1]

        with h5py.File(image_path, 'r') as h5f:
            t2w = h5f['image']['t2w'][:]
            adc = h5f['image']['adc'][:]
            hbv = h5f['image']['hbv'][:]
            seg = h5f['label']['seg'][:].astype(np.float32)

        image = np.stack([t2w, adc, hbv], axis=0)  # Shape: [3, H, W, D]
        samples = image, seg

        if self.transform:
            image_, label_ = self.transform(samples)
        else:
            image_, label_ = image, seg

        return image_.float(), label_.long()


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def _get_transform(self, label):
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 1, 0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = label.shape
        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        def do_transform(x):
            if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
                x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            x = x[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return x
        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[1])  # Use label to compute crop
        return [transform(samples[0]), transform(samples[1])]


class RandomCrop(object):
    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def _get_transform(self, x):
        if x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2] or x.shape[3] <= self.output_size[2]:
            pw = max((self.output_size[1] - x.shape[1]) // 2 + 1, 0)
            ph = max((self.output_size[2] - x.shape[2]) // 2 + 1, 0)
            pd = max((self.output_size[2] - x.shape[3]) // 2 + 1, 0)
            x = np.pad(x, [(0, 0), (pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        _, w, h, d = x.shape
        w1 = np.random.randint(0, w - self.output_size[1])
        h1 = np.random.randint(0, h - self.output_size[2])
        d1 = np.random.randint(0, d - self.output_size[2])

        def do_transform(x):
            return x[:, w1:w1 + self.output_size[1], h1:h1 + self.output_size[2], d1:d1 + self.output_size[2]]

        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0][np.newaxis, ...])  # Add batch channel
        return [transform(samples[0]), transform(samples[1][np.newaxis, ...])[0]]


class ToTensor(object):
    def __call__(self, sample):
        image = sample[0].astype(np.float32)  # [3, H, W, D]
        label = sample[1].astype(np.float32)  # [H, W, D]
        return [torch.from_numpy(image), torch.from_numpy(label)]


if __name__ == '__main__':
    data_dir = '/content/drive/MyDrive/SemiSL/Dataset/PICAI_dataset'
    list_dir = '/content/drive/MyDrive/SemiSL/Code/PICAI_SSL/Basecode/Datasets/picai/data_split'

    labset = PICAIDataset(data_dir, list_dir, split='lab')
    unlabset = PICAIDataset(data_dir, list_dir, split='unlab')
    trainset = PICAIDataset(data_dir, list_dir, split='train')
    testset = PICAIDataset(data_dir, list_dir, split='test')

    lab_sample = labset[0]
    unlab_sample = unlabset[0]
    train_sample = trainset[0]
    test_sample = testset[0]

    print(len(labset), lab_sample[0].shape, lab_sample[1].shape)
    print(len(unlabset), unlab_sample[0].shape, unlab_sample[1].shape)
    print(len(trainset), train_sample[0].shape, train_sample[1].shape)
    print(len(testset), test_sample[0].shape, test_sample[1].shape)
