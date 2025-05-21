from __future__ import print_function
import numpy as np
from PIL import Image, ImageOps
from torchvision import datasets
from utils import TransformTwice,Cifar10TransformTwice
from torchvision import transforms
import torch
import math

cifar10_mean, cifar10_std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
cifar100_mean, cifar100_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

def get_dataset_class(args):
    if args.dataset == 'cifar10':
        return cifar10_dataset(args)
    elif args.dataset == 'cifar100':
        return cifar100_dataset(args)

def x_u_split_seen_novel(labels, lbl_percent, num_classes, lbl_set, unlbl_set, imb_factor):
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        img_max = len(idx)
        num = img_max * ((1/imb_factor)**(i / (num_classes - 1.0)))
        idx = idx[:int(num)]
        n_lbl_sample = math.ceil(len(idx)*(lbl_percent/100))
        if i in lbl_set:
            labeled_idx.extend(idx[:n_lbl_sample])
            unlabeled_idx.extend(idx[n_lbl_sample:])
        elif i in unlbl_set:
            unlabeled_idx.extend(idx)
    return labeled_idx, unlabeled_idx

class cifar100_dataset():
    def __init__(self, args):
        self.transform_train = transforms.Compose([
            transforms.RandomChoice([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomResizedCrop(32, (0.5, 1.0)),
                ]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
            Solarize(p=0.1),
            Equalize(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
        ])

        self.transform_val = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
        ])

        base_dataset = datasets.CIFAR100(args.data_root, train=True, download=True)
        train_labeled_idxs, train_unlabeled_idxs = x_u_split_seen_novel(base_dataset.targets, args.lbl_percent, args.no_class, list(range(0,args.no_seen)), list(range(args.no_seen, args.no_class)), args.imb_factor)

        self.train_labeled_idxs = train_labeled_idxs
        self.train_unlabeled_idxs = train_unlabeled_idxs
        self.temperature = args.temperature
        self.data_root = args.data_root
        self.no_seen = args.no_seen
        self.no_class = args.no_class

    def get_dataset(self, temp_uncr=None):
        train_labeled_idxs = self.train_labeled_idxs.copy()
        train_unlabeled_idxs = self.train_unlabeled_idxs.copy()

        train_labeled_dataset = CIFAR100SSL(self.data_root, train_labeled_idxs, train=True,
                                            transform=TransformTwice(dict_transform['cifar_train']),temperature=self.temperature)
        train_unlabeled_dataset = CIFAR100SSL(self.data_root, train_unlabeled_idxs, train=True,
                                              transform=TransformTwice(dict_transform['cifar_train']),
                                                                       temperature=self.temperature,temp_uncr=temp_uncr)

        if temp_uncr is not None:
            return train_labeled_dataset, train_unlabeled_dataset

        test_dataset_seen = CIFAR100SSL_TEST(self.data_root, train=False, transform=TransformTwice(dict_transform['cifar_test'],test=1), download=False, labeled_set=list(range(0,self.no_seen)))
        test_dataset_novel = CIFAR100SSL_TEST(self.data_root, train=False, transform=TransformTwice(dict_transform['cifar_test'],test=1), download=False, labeled_set=list(range(self.no_seen, self.no_class)))
        test_dataset_all = CIFAR100SSL_TEST(self.data_root, train=False, transform=TransformTwice(dict_transform['cifar_test'],test=1), download=False)
        return train_labeled_dataset, train_unlabeled_dataset,test_dataset_all, test_dataset_seen, test_dataset_novel

class cifar10_dataset():
    def __init__(self, args):
        self.transform_train = transforms.Compose([
            transforms.RandomChoice([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomResizedCrop(32, (0.5, 1.0)),
                ]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
            Solarize(p=0.1),
            Equalize(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])

        self.transform_val = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        ])

        base_dataset = datasets.CIFAR10(args.data_root, train=True, download=True)
        train_labeled_idxs, train_unlabeled_idxs = x_u_split_seen_novel(base_dataset.targets, args.lbl_percent, args.no_class, list(range(0,args.no_seen)), list(range(args.no_seen, args.no_class)), args.imb_factor)

        self.train_labeled_idxs = train_labeled_idxs
        self.train_unlabeled_idxs = train_unlabeled_idxs
        self.temperature = args.temperature
        self.data_root = args.data_root
        self.no_seen = args.no_seen
        self.no_class = args.no_class

    def get_dataset(self, temp_uncr=None):
        train_labeled_idxs = self.train_labeled_idxs.copy()
        train_unlabeled_idxs = self.train_unlabeled_idxs.copy()

        train_labeled_dataset = CIFAR10SSL(self.data_root, train_labeled_idxs, train=True, transform=Cifar10TransformTwice(dict_transform['cifar10_train']), temperature=self.temperature)
        train_unlabeled_dataset = CIFAR10SSL(self.data_root, train_unlabeled_idxs, train=True, transform=Cifar10TransformTwice(dict_transform['cifar10_train']), temperature=self.temperature, temp_uncr=temp_uncr)

        if temp_uncr is not None:
            return train_labeled_dataset, train_unlabeled_dataset

        test_dataset_seen = CIFAR10SSL_TEST(self.data_root, train=False, transform=Cifar10TransformTwice(dict_transform['cifar10_test'],test=1), download=False, labeled_set=list(range(0,self.no_seen)))
        test_dataset_novel = CIFAR10SSL_TEST(self.data_root, train=False, transform=Cifar10TransformTwice(dict_transform['cifar10_test'],test=1), download=False, labeled_set=list(range(self.no_seen, self.no_class)))
        test_dataset_all = CIFAR10SSL_TEST(self.data_root, train=False, transform=Cifar10TransformTwice(dict_transform['cifar10_test'],test=1), download=False)
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset_all, test_dataset_seen, test_dataset_novel

dict_transform = {
    'cifar_train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]),

    'cifar_test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]),

    'cifar10_train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ]),

    'cifar10_test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ]),
}

class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, temperature=None, temp_uncr=None, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)

        if temperature is not None:
            self.temp = temperature*np.ones(len(self.targets))
        else:
            self.temp = np.ones(len(self.targets))

        if temp_uncr is not None:
            self.temp[temp_uncr['index']] = temp_uncr['uncr']

        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.temp = self.temp[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.indexs[index], self.temp[index]

class CIFAR10SSL_TEST(datasets.CIFAR10):
    def __init__(self, root, train=False,
                 transform=None, target_transform=None,
                 download=True, labeled_set=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)
        indexs = []
        if labeled_set is not None:
            for i in range(10):
                idx = np.where(self.targets == i)[0]
                if i in labeled_set:
                    indexs.extend(idx)
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, temperature=None, temp_uncr=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)

        if temperature is not None:
            self.temp = temperature*np.ones(len(self.targets))
        else:
            self.temp = np.ones(len(self.targets))

        if temp_uncr is not None:
            self.temp[temp_uncr['index']] = temp_uncr['uncr']

        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.temp = self.temp[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.indexs[index], self.temp[index]

class CIFAR100SSL_TEST(datasets.CIFAR100):
    def __init__(self, root, train=False,
                 transform=None, target_transform=None,
                 download=False, labeled_set=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)
        indexs = []
        if labeled_set is not None:
            for i in range(100):
                idx = np.where(self.targets == i)[0]
                if i in labeled_set:
                    indexs.extend(idx)
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class Solarize(object):
    def __init__(self, p=0.2):
        self.prob = p

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img
        v = torch.rand(1) * 256
        return ImageOps.solarize(img, v)

class Equalize(object):
    def __init__(self, p=0.2):
        self.prob = p

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img
        return ImageOps.equalize(img)