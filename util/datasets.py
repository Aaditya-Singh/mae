# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args, transform=None):
    if transform is None:
        transform = build_transform(is_train, args)
    # -- don't use suffix when dataset is imagenet-a/c/r/s
    in_variants = {'-a', '-c', '-r', '-s'}
    ood = any(var in args.data_path for var in in_variants)
    suffix = '' if ood else 'train/' if is_train else 'val/'    
    root = os.path.join(args.data_path, suffix)
    dataset = datasets.ImageFolder(root, transform=transform)
    if args.subset_file is not None and is_train:
        dataset = ImageNetSubset(dataset, args.subset_file)
    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class ImageNetSubset(object):

    def __init__(self, dataset, subset_file):
        """
        ImageNetSubset
        :param dataset: ImageNet dataset object
        :param subset_file: '.txt' file containing IDs of IN1K images to keep
        """
        self.dataset = dataset
        self.subset_file = subset_file
        self.filter_dataset_(subset_file)

    def filter_dataset_(self, subset_file):
        """ Filter self.dataset to a subset """
        root = self.dataset.root
        class_to_idx = self.dataset.class_to_idx
        # -- update samples to subset of IN1k targets/samples
        new_samples = []
        print(f'Using {subset_file}')
        with open(subset_file, 'r') as rfile:
            for line in rfile:
                class_name = line.split('_')[0]
                target = class_to_idx[class_name]
                img = line.split('\n')[0]
                new_samples.append(
                    (os.path.join(root, class_name, img), target)
                )
        self.samples = new_samples

    @property
    def classes(self):
        return self.dataset.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.dataset.loader(path)
        if self.dataset.transform is not None:
            img = self.dataset.transform(img)
        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)
        return img, target