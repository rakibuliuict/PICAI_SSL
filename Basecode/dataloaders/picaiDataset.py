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

    def __call__(self, samples):
        image, label = samples  # image: [C, H, W, D], label: [H, W, D]
        _, H, W, D = image.shape
        oH, oW, oD = self.output_size

        start_h = max((H - oH) // 2, 0)
        start_w = max((W - oW) // 2, 0)
        start_d = max((D - oD) // 2, 0)

        end_h = start_h + oH
        end_w = start_w + oW
        end_d = start_d + oD

        image_cropped = image[:, start_h:end_h, start_w:end_w, start_d:end_d]
        label_cropped = label[start_h:end_h, start_w:end_w, start_d:end_d]

        return image_cropped, label_cropped


class RandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, samples):
        image, label = samples  # image: [C, H, W, D], label: [H, W, D]
        _, H, W, D = image.shape
        oH, oW, oD = self.output_size

        if H < oH or W < oW or D < oD:
            pad_h = max(oH - H, 0)
            pad_w = max(oW - W, 0)
            pad_d = max(oD - D, 0)

            pad = (
                (0, 0),  # channel
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
                (pad_d // 2, pad_d - pad_d // 2)
            )
            image = np.pad(image, pad, mode='constant', constant_values=0)
            label = np.pad(label, (
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
                (pad_d // 2, pad_d - pad_d // 2)
            ), mode='constant', constant_values=0)

            _, H, W, D = image.shape  # Update new shape

        start_h = np.random.randint(0, H - oH + 1)
        start_w = np.random.randint(0, W - oW + 1)
        start_d = np.random.randint(0, D - oD + 1)

        end_h = start_h + oH
        end_w = start_w + oW
        end_d = start_d + oD

        image_cropped = image[:, start_h:end_h, start_w:end_w, start_d:end_d]
        label_cropped = label[start_h:end_h, start_w:end_w, start_d:end_d]

        return image_cropped, label_cropped



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
