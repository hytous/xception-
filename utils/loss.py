import numpy as np
import torch
import torch.nn as nn

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        """
        CrossEntropyLoss交叉熵用的是，图片属于某类的概率
        也就是说在已知的数据上，某个图片属于它自身的类的概率为1，属于其他类的概率为0
        预测时某图片属于各个类的概率不同，取概率最大的作为预测值

        在多分类里，如果有0，1，2，3总共4个类，且batch为1
        某图片为2类，那么它的target的格式为[0, 0, 1, 0]
        然后预测这个图片的output的格式可能是[0.1, 0.3, 0.6, 0.0]

        当batch大于1时，target和output就成二维的了
        比如batch=2，其中第一个图片为0类，第二个图片为3类。那target就是
        [1, 0, 0, 0]
        [0, 0, 0, 1]
        """
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        # 图片没有为图片分割做好标注，所以先随便填充一下，之后换算法或者给图片弄上标注
        changed_target = target
        changed_target = np.zeros((5, 434, 636))
        for i in range(5):
            changed_target[i][0][0] = target[i]
        changed_target = torch.tensor(changed_target)
        loss = criterion(logit, changed_target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




