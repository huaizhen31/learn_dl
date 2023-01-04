#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: wanghuaizhen
# datetime:2023/1/4 21:52

import cv2
import glob
import os.path  as osp
from m_utils import load_mnist
import torch.utils.data.dataset
def dataset(boot_dir):
    imgs = glob.glob(osp.join(boot_dir,"*","*.png"))
    return imgs


if __name__ == "__main__":
    train_dir = r"D:\Dataset\MNIST\imgs\train"
    imgs = dataset(train_dir)
    print()