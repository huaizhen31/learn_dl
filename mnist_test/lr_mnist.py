#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: wanghuaizhen
# datetime:2023/1/4 21:52

import glob
import os.path  as osp
from m_utils import load_mnist
import numpy as np
import datetime


def dataset(boot_dir):
    imgs = glob.glob(osp.join(boot_dir, "*", "*.png"))
    return imgs


if __name__ == "__main__":
    # 设置数据文件路径
    x_train_path = r"D:\Dataset\MNIST\train-images-idx3-ubyte.gz"
    y_train_path = r"D:\Dataset\MNIST\train-labels-idx1-ubyte.gz"
    x_test_path = r"D:\Dataset\MNIST\t10k-images-idx3-ubyte.gz"
    y_test_path = r"D:\Dataset\MNIST\t10k-labels-idx1-ubyte.gz"

    # 设置学习率，Batch Size，训练的Epoch数量
    alpha = 1e-4
    batch_size = 128
    num_epoch = 100

    # 从文件中加载数据集
    ((train_x, train_y), (test_x, test_y)) = load_mnist(x_train_path, y_train_path, x_test_path, y_test_path,
                                                        normalize=True, one_hot=False)
    # x矩阵添加1个维度，单条数据形成 [1,x1,x2,x3...] 的向量
    train_x = np.insert(train_x, 0, values=1, axis=1)
    test_x = np.insert(test_x, 0, values=1, axis=1)
    # 标签的数据形状由(1)转为(1,1)，例如：[1]转为 [[1]]，再对标签进行归一化，0~10转为0~1
    train_y = train_y[:, np.newaxis]
    train_y = train_y / 10
    test_y = test_y[:, np.newaxis]
    test_y = test_y / 10
    # n是单条数据的维度，785
    n = train_x.shape[1]
    # w是待训练的初始化参数
    w = np.random.rand(n, 1)

    # 训练集和测试集的数量
    num_train = train_x.shape[0]
    num_test = test_x.shape[0]

    # 每个Epoch需要训练多少轮，每轮训练一个Batch Size的数据
    iter_per_train_ep = int(num_train / batch_size)
    iter_per_test_ep = int(num_test / batch_size)

    # 线性回归函数
    def f(x, w):
        return np.dot(x.T, w)  # size_x=(785,m), size_w=(785,1)

    # 损失函数
    def loss(y, y_):
        val = 0.5 * np.dot((y_ - y).T, (y_ - y))
        n = y.shape[0]
        return val / n

    # 根据损失函数计算梯度
    def grad_loss(x, y, w):  # x * (x_T * w - y)
        num = x.shape[1]
        item1 = np.dot(x.T, w)
        item2 = item1 - y
        grad = np.dot(x, item2)
        return grad / num


    # debug
    x_T = train_x[0:16, :]
    x = x_T.T
    y = train_y[0:16, :]
    y_ = f(x, w)
    loss_val = loss(y, y_)
    grad = grad_loss(x, y, w)

    # 训练和验证
    for i_ep in range(num_epoch):
        # 用于选择数据的矩阵
        select_train = np.arange(0, num_train, dtype=int)
        select_test = np.arange(0, num_test, dtype=int)

        # 开始训练
        print(f"{datetime.datetime.now()}: EP {i_ep}, Start Training model")
        train_loss = 0.
        for i_iter in range(iter_per_train_ep):
            # 随机选择Batch Size数量的数据
            index = np.random.choice(select_train, batch_size, replace=False)
            # 取数据和标签，对x形状进行处理
            x_T = train_x[index]
            x = x_T.T
            y = train_y[index]

            # 用包含参数w的模型f(x)对x进行预测，得到预测标签y_
            y_ = f(x, w)
            # 用loss函数计算实际标签y和模型预测标签y_之间的损失
            loss_val = loss(y, y_)[0][0]
            train_loss += loss_val # 累计损失
            # 计算损失函数的梯度
            grad = grad_loss(x, y, w)
            # BP梯度反传算法
            w = w - alpha * grad
            # 打印
            if i_iter % 10 == 0 or i_iter == iter_per_train_ep - 1:
                print(
                    f"{datetime.datetime.now()}:  -- EP {i_ep}, Train, ITER {i_iter}/{iter_per_train_ep}, Loss={loss_val}")
        print(f"{datetime.datetime.now()}: EP {i_ep}, End Training model, Loss={train_loss / iter_per_train_ep}")
        print()

        # 开始验证
        print(f"{datetime.datetime.now()}: EP {i_ep}, Start validating model")
        val_loss = 0.
        for i_iter in range(iter_per_test_ep):
            index = np.random.choice(select_test, batch_size, replace=False)
            x_T = test_x[index]
            x = x_T.T
            y = test_y[index]

            y_ = f(x, w)
            loss_val = loss(y, y_)[0][0]
            val_loss += loss_val
            if i_iter % 10 == 0 or i_iter == iter_per_test_ep - 1:
                print(
                    f"{datetime.datetime.now()}:  -- EP {i_ep}, Val ITER {i_iter}/{iter_per_test_ep}, Loss={loss_val}")
        print(f"{datetime.datetime.now()}: EP {i_ep}, End validating model, Loss={val_loss / iter_per_test_ep}")
        print()
