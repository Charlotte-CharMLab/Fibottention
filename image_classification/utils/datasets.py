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
import torchvision
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.dataset == "c10":
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=is_train, download=True, transform=transform)
    elif args.dataset == "c100":
        dataset = torchvision.datasets.CIFAR100(
            root='./data', train=is_train, download=True, transform=transform)
    elif args.dataset == "tinyimagenet":
        root = os.path.join('../../imagenet-tiny', 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
    elif args.dataset == "imagenet":
        dataset = torchvision.datasets.ImageNet(
            root='./data', train=is_train, download=True, transform=transform)
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    if args.dataset == "c10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif args.dataset == "c100":
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2009, 0.1984, 0.2023)
    elif args.dataset == "tinyimagenet":
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
