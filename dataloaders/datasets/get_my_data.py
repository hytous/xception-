from keras.preprocessing.image import ImageDataGenerator
from torch.utils.data import Dataset
# 用于分训练集和验证集
from sklearn.model_selection import train_test_split

import tensorflow as tf

import cv2
import os

import numpy as np
import pandas as pd

# 读取mat文件
import scipy.io as scio

# 图片大小
img_chang = 434
img_kuan = 636


def get_data():  # 获取数据
    data = []
    # 可以用matlab查看数据形式
    allmatdatas = scio.loadmat(
        'D:\S\study\大四\毕设\代码\pytorch-deeplab-xception-master\dataloaders'
        '\dataset_liver_bmodes_steatosis_assessment_IJCARS.mat')
    matdatas = allmatdatas['data']
    matdatas = matdatas[0]
    for matdata in matdatas:  # 总共55组图片
        fatclass = matdata['class'][0, 0]  # 是否是脂肪肝，用0或1表示 [0,0]是因为数据格式是：[[0]]这样的，可能是tensor吧
        img_batch = matdata['images']  # 每组图片为10张434*636的灰度图
        try:
            data.append([img_batch, fatclass])
        except Exception as e:
            print(e)
    return np.array(data)


# 定义FattyLiver类，继承Dataset方法，并重写__getitem__()和__len__()方法
class FattyLiver(Dataset):
    def __init__(self, data_root, data_label):
        # 初始化函数，得到数据
        self.data = data_root
        self.label = data_label

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    """
    关于pytorch加载数据的流程
    https://blog.csdn.net/weixin_42469843/article/details/126044188
    """
    from torch.utils.data import DataLoader

    # 把数据从mat文件中取出
    data = get_data()
    img_batch = data[:, 0]
    img_batch_label = data[:, 1]

    # 通过FattyLiver将数据进行加载，返回Dataset对象，包含data和labels
    torch_data = FattyLiver(img_batch, img_batch_label)

    # 通过上述的程序，我们构造了一个数据加载器torch_data，但是还是不能直接
    # 传入网络中。接下来需要构造数据装载器，产生可迭代的数据，再传入网络中。DataLoader类完成这个工作。
    # num_workers : 表示加载的时候子进程数
    dataloader = DataLoader(torch_data, batch_size=4, shuffle=True, drop_last=False, num_workers=0)

