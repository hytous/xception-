from torch.utils.data import Dataset
import torch
import numpy as np

# 读取mat文件
import scipy.io as scio

# 图片大小
img_chang = 434
img_kuan = 636


def get_data():  # 获取数据
    val_index = [34, 42, 23, 2, 12]  # 验证集的下表
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    # 可以用matlab查看数据形式
    # allmatdatas = scio.loadmat(
    #     r'../dataset_liver_bmodes_steatosis_assessment_IJCARS.mat')  # 在本文件内运行时的相对路径
    allmatdatas = scio.loadmat(
        r'./dataloaders/dataset_liver_bmodes_steatosis_assessment_IJCARS.mat')  # 在train文件内运行时的相对路径
    matdatas = allmatdatas['data']
    matdatas = matdatas[0]
    for i, matdata in enumerate(matdatas):  # 总共55组图片  enumerate可以在for里多加一个计数器i
        # fatclass = matdata['class'][0, 0]  # 是否是脂肪肝，用0或1表示 [0,0]是因为数据格式是：[[0]]这样的，可能是tensor吧
        fat = matdata['fat'][0, 0]
        # 先做四分类
        if fat < 5:
            fatclass = 0  # 无脂肪肝
        elif 5 <= fat <= 27:
            fatclass = 1  # 轻度脂肪肝
        elif 28 <= fat <= 45:  # 55
            fatclass = 2  # 中度脂肪肝
        elif 46 <= fat:  # 56
            fatclass = 3  # 重度脂肪肝

        img_batch = matdata['images']  # 每组图片为10张434*636的灰度图
        # print('编号为%d的病人fat程度%d  fat种类%d' % (i, fat, fatclass))
        try:
            # if i in val_index:  # 隔11个数据抽一组作为验证集，验证机中总共5组
            if i % 11 == 0:
                # print('验证集fat程度%d  fat种类%d' % (fat, fatclass))
                for img in img_batch:
                    img = np.expand_dims(img, axis=0)  # 增维
                    val_data.append(img.astype(np.float32))  # 原始图片数据是int类型的，之后的处理需要float类型
                    val_label.append(fatclass)
            else:
                for img in img_batch:
                    img = np.expand_dims(img, axis=0)  # 增维
                    train_data.append(img.astype(np.float32))  # 原始图片数据是int类型的，之后的处理需要float类型
                    train_label.append(fatclass)
        except Exception as e:
            print(e)
    return np.array(train_data), np.array(train_label), np.array(val_data), np.array(val_label)


def get_all_data():  # 获取数据
    all_data = []
    all_label = []
    # allmatdatas = scio.loadmat(
    #     r'../dataset_liver_bmodes_steatosis_assessment_IJCARS.mat')  # 在本文件内运行时的相对路径
    allmatdatas = scio.loadmat(
        r'./dataloaders/dataset_liver_bmodes_steatosis_assessment_IJCARS.mat')  # 在train文件内运行时的相对路径
    matdatas = allmatdatas['data']
    matdatas = matdatas[0]
    for i, matdata in enumerate(matdatas):  # 总共55组图片  enumerate可以在for里多加一个计数器i
        # fatclass = matdata['class'][0, 0]  # 是否是脂肪肝，用0或1表示 [0,0]是因为数据格式是：[[0]]这样的，可能是tensor吧
        fat = matdata['fat'][0, 0]
        # 先做四分类
        if fat < 5:
            fatclass = 0  # 无脂肪肝
        elif 5 <= fat <= 27:
            fatclass = 1  # 轻度脂肪肝
        elif 28 <= fat <= 45:  # 55
            fatclass = 2  # 中度脂肪肝
        elif 46 <= fat:  # 56
            fatclass = 3  # 重度脂肪肝

        img_batch = matdata['images']  # 每组图片为10张434*636的灰度图
        # print('编号为%d的病人fat程度%d  fat种类%d' % (i, fat, fatclass))
        try:
            for img in img_batch:
                img = np.expand_dims(img, axis=0)  # 增维
                all_data.append(img.astype(np.float32))  # 原始图片数据是int类型的，之后的处理需要float类型
                all_label.append(fatclass)
        except Exception as e:
            print(e)
    return np.array(all_data), np.array(all_label)


# 定义FattyLiver类，继承Dataset方法，并重写__getitem__()和__len__()方法
class FattyLiver(Dataset):
    # 总共四个类
    NUM_CLASSES = 4

    def __init__(self, args, split='train'):
        super().__init__()

        # 初始化函数，得到数据
        if split == 'train':
            train_data, train_label, val_data, val_label = get_data()
            img_batch = train_data
            img_batch_label = train_label
            self.data = img_batch
            self.label = img_batch_label
        elif split == 'val':
            train_data, train_label, val_data, val_label = get_data()
            img_batch = val_data
            img_batch_label = val_label
            self.data = img_batch
            self.label = img_batch_label
        elif split == 'all':
            all_data, all_label = get_all_data()
            img_batch = all_data
            img_batch_label = all_label
            self.data = img_batch
            self.label = img_batch_label
        self.split = split
        self.args = args

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        img_data = self.data[index]
        labels = self.label[index]
        sample = {'image': img_data, 'label': labels}
        return sample

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    """
    关于pytorch加载数据的流程
    https://blog.csdn.net/weixin_42469843/article/details/126044188
    """
    from torch.utils.data import DataLoader

    train_data, train_label, val_data, val_label = get_data()
    print("train_data的shape", train_data.shape)
    print("train_label的shape", train_label.shape)
    print("val_data的shape", val_data.shape)
    print("val_label的shape", val_label.shape)
    # 通过FattyLiver将数据进行加载，返回Dataset对象，包含data和labels
    # torch_data = FattyLiver('train')

    # 通过上述的程序，我们构造了一个数据加载器torch_data，但是还是不能直接
    # 传入网络中。接下来需要构造数据装载器，产生可迭代的数据，再传入网络中。DataLoader类完成这个工作。
    # num_workers : 表示加载的时候子进程数
    # dataloader = DataLoader(torch_data, batch_size=5, shuffle=True, drop_last=False, num_workers=0)
"""
fat程度为0的共2人
fat程度为1的共3人
fat程度为2的共8人
fat程度为3的共3人
fat程度为4的共1人
fat程度为5的共1人
fat程度为7的共1人
fat程度为10的共4人
fat程度为15的共3人
fat程度为20的共7人
fat程度为25的共3人
fat程度为30的共1人
fat程度为40的共3人
fat程度为50的共3人
fat程度为55的共2人
fat程度为70的共5人
fat程度为75的共2人
fat程度为80的共2人
fat程度为85的共1人
"""
