##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import math

class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0):
        self.mode = mode  # 这里的model是学习率的调整方式
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr  # 一开始指定的学习率
        if mode == 'step':
            assert lr_step  # 选择学习率调整模式为step时不能为0
        self.lr_step = lr_step  # 选择学习率调整模式为step时，是一个固定值，人为规定
        self.iters_per_epoch = iters_per_epoch  # 训练集的长度
        self.N = num_epochs * iters_per_epoch  # 整个训练中总共使用了N组数据(每代iters_per_epoch组，训练了num_epochs代)
        self.epoch = -1
        # 规定一个热身时间，在开始学习时先热身，不管选择什么学习率调整手段都先不用，热身时使用一个固定的简单调整手段
        # 热身完才使用选定的学习率调整手段
        self.warmup_iters = warmup_epochs * iters_per_epoch

    """
    学习率调整机制，详细解释见lr_scheduler.md
    """
    def __call__(self, optimizer, i, epoch, best_pred):
        # 第n代的第i组数据集,每代有iters_per_epoch组数据集所以现在的位置是:T = epoch * self.iters_per_epoch + i
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            # Cosine Annealing
            # 1.0 * T / self.N会在训练中不断趋近于1
            # math.cos(1.0 * T / self.N * math.pi)会先向1趋近然后最终趋近于-1
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            # Polynomial learning rate decay
            # p=0.9
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            # Step learning rate decay
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        # warm up lr schedule
        # 热身阶段
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        # 每代开始时先输出一下代数、学习率
        if epoch > self.epoch:  # self.epoch还没更新，所以会比epoch小1
            print('\n=>Epoches %i, learning rate = %.4f, \
                previous best = %.4f' % (epoch, lr, best_pred))
            self.epoch = epoch  # 更新self.epoch
        assert lr >= 0  # 学习率不能小于0
        # 调整学习率
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        """
        当优化器只有一个参数组时，直接将该参数组的学习率设置为指定的值 lr。
        如果存在多个参数组，则将第一个参数组的学习率设为指定的值 lr。然后将后
        面每个参数组的学习率都设为第一个参数组的学习率乘以10，这样就实现了不同
        参数组具有不同学习率的效果。
        """
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10
