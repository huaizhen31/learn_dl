#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: wanghuaizhen
# datetime:2023/1/4 21:22

import numpy as np
from struct import unpack
import gzip
import os.path as osp
import os
import cv2
from tqdm import tqdm


def read_image(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)
    return img


def read_label(path):
    with gzip.open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.frombuffer(f.read(), dtype=np.uint8)
        # print(lab[1])
    return lab


def normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img


def one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab


def load_mnist(x_train_path, y_train_path, x_test_path, y_test_path, normalize=True, one_hot=True):
    '''读入MNIST数据集
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label :
        one_hot为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    Returns
    ----------
    (训练图像, 训练标签), (测试图像, 测试标签)
    '''
    image = {
        'train': read_image(x_train_path),
        'test': read_image(x_test_path)
    }

    label = {
        'train': read_label(y_train_path),
        'test': read_label(y_test_path)
    }

    if normalize:
        for key in ('train', 'test'):
            image[key] = normalize_image(image[key])

    if one_hot:
        for key in ('train', 'test'):
            label[key] = one_hot_label(label[key])

    return (image['train'], label['train']), (image['test'], label['test'])


def save_mnist_imgs(x_train_path, y_train_path, x_test_path, y_test_path, save_dir):
    '''读入MNIST数据集
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label :
        one_hot为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    Returns
    ----------
    (训练图像, 训练标签), (测试图像, 测试标签)
    '''
    image = {
        'train': read_image(x_train_path, is_img=True),
        'test': read_image(x_test_path, is_img=True)
    }

    label = {
        'train': read_label(y_train_path),
        'test': read_label(y_test_path)
    }

    for key in ['train', 'test']:
        x = image[key]
        y = label[key]

        num = x.shape[0]
        for i in tqdm(range(num)):
            img = x[i, :]
            img = img.reshape(28, 28)
            this_label = y[i]
            save_this = osp.join(save_dir, key, str(this_label))
            if not osp.isdir(save_this):
                os.makedirs(save_this)
            save_path = osp.join(save_this, f"{this_label}-{i}.png")
            cv2.imwrite(save_path, img)

    return (image['train'], label['train']), (image['test'], label['test'])
