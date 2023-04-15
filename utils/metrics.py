import numpy as np


class Evaluator(object):
    def __init__(self, num_class=4):  # 默认4分类
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    # 分类准确率
    def Accuracy(self):
        # 混淆矩阵的对角线元素为预测为真的
        # 混淆矩阵的对角线元素求和就是所有预测为真的求和，再除以矩阵中全部元素求和(全部预测结果)得到的就是准确率了
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()  # np.diag()取矩阵的对角线元素
        return Acc

    # 不同类各自的准确率
    def Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)  # nanmean忽略掉空值进行求均值，不忽略掉空值计算会报错
        return Acc

    # 生成混淆矩阵
    def _generate_matrix(self, real_type, pre_type):
        confusion_matrix = np.zeros((self.num_class,) * 2)  # 创建一个C*C大小的0矩阵
        for p, t in zip(pre_type, real_type):
            confusion_matrix[p, t] += 1
        return confusion_matrix

    # 输入预测结果和真实值
    def add_batch(self, real_type, pre_type):
        # assert（断言）用于判断一个表达式
        # 在表达式条件为 false 的时候触发异常。
        # 断言可以在条件不满足程序运行的情况下直接返回错误，而不必等待程序运行后出现崩溃的情况
        assert real_type.shape == pre_type.shape
        self.confusion_matrix += self._generate_matrix(real_type, pre_type)  # 生成混淆矩阵

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)




