#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: wanghuaizhen
# datetime:2023/1/4 21:19

from m_utils import save_mnist_imgs

if __name__ == "__main__":
    x_train_path = r"D:\Dataset\MNIST\train-images-idx3-ubyte.gz"
    y_train_path = r"D:\Dataset\MNIST\train-labels-idx1-ubyte.gz"
    x_test_path = r"D:\Dataset\MNIST\t10k-images-idx3-ubyte.gz"
    y_test_path = r"D:\Dataset\MNIST\t10k-labels-idx1-ubyte.gz"
    save_dir = r"D:\Dataset\MNIST\imgs"

    result = save_mnist_imgs(x_train_path, y_train_path, x_test_path, y_test_path, save_dir)
