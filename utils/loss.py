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

        # n, c, h, w = logit.size()
        n, c = logit.size()  # n为batch数，c为类数
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        # batchsize现在为5，到时候可能要改
        # 构建一下target的概率数组
        # changed_target = np.zeros((n, c))
        # for i in range(n):
        #     changed_target[i][target[i]] = 1  # 已知属于该类，概率为1
        # changed_target = torch.tensor(changed_target).cuda()
        # print(changed_target)
        # torch.nn.Softmax(dim=1)(changed_target)
        # print('softmax后', changed_target)
        # loss = criterion(logit, changed_target)
        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        # n, c, h, w = logit.size()
        n, c = logit.size()  # n为batch数，c为类数
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
    a = torch.tensor([[0.2263, 0.2201, 0.0747, 0.4571]]).cuda()
    b = torch.tensor([0]).cuda()
    # b = torch.randn(1, 4).softmax(dim=1).cuda()
    # b = torch.tensor([1, 0, 0, 0]).cuda()

    # a = torch.rand(1, 3, 7, 7).cuda()
    # b = torch.rand(1, 7, 7).cuda()
    print('输出的loss值为%.4f' % loss.CrossEntropyLoss(a, b).item())
    # print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    # print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
"""
output为 : tensor([[0.2263, 0.2201, 0.0747, 0.4571]])
真实值为 : tensor([0])
这次的loss为 : tensor(1.4141)


output为 : tensor([[0.2243, 0.2199, 0.0779, 0.4522]])
真实值为 : tensor([1])
这次的loss为 : tensor(1.4192)


output为 : tensor([[0.2292, 0.2115, 0.0808, 0.4563]])
真实值为 : tensor([1])
这次的loss为 : tensor(1.4286)
"""