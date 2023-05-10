import torch
import torchvision
from torch import nn
import argparse
import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.buildnet import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator


class TrainedModleValidator(object):
    def __init__(self, args):
        self.args = args
        # Define Tensorboard Summary 可视化
        self.summary = TensorboardSummary(r'/root/tf-logs/experiment_val')
        self.writer = self.summary.create_summary()
        # 载入数据
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.all_loader, self.nclass, self.class_names\
            = make_data_loader(args, **kwargs)
        # 定义网络
        model = Builder(num_classes=self.nclass,
                        backbone=args.backbone,
                        pretrained=False)
        self.model = model
        # 读取训练好的模型
        # checkpoint = torch.load(r'/root/tf-logs/experimentxception_13/checkpoint.pth.tar')
        checkpoint = torch.load(r'/root/tf-logs/bestpredict_ever/model_best.pth.tar')
        # 读取出来的键值最前面多了个module.所以删掉
        checkpoint_dict = {}
        for k, v in checkpoint['state_dict'].items():
            new_k = k.replace('module.', '') if 'module' in k else k
            checkpoint_dict[new_k] = v
        # 载入模型数据
        self.model.load_state_dict(checkpoint_dict)
        # Using cuda
        if args.cuda:
            # 使用nn.DataParallel函数来用多个GPU来加速训练
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
        # 定义混淆矩阵生成器、准确率计算器
        self.evaluator = Evaluator(self.nclass)
        # 定义loss计算器
        self.criterion = SegmentationLosses(cuda=args.cuda).build_loss(mode=args.loss_type)

    # 模型评估
    def validation(self, valortrain):
        self.model.eval()
        self.evaluator.reset()  # 清空混淆矩阵
        if valortrain == 'train':
            tbar = tqdm(self.train_loader, desc='\r')  # 选择评估数据集并塞进进度条
            epoch = 0
        elif valortrain == 'val':
            tbar = tqdm(self.val_loader, desc='\r')  # 选择评估数据集并塞进进度条
            epoch = 1
        elif valortrain == 'all':
            tbar = tqdm(self.all_loader, desc='\r')  # 选择评估数据集并塞进进度条
            epoch = 2
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()  # 预测出的各类的概率
            target = target.cpu().numpy()  # 真实值
            pred = np.argmax(pred, axis=1)  # 选出概率最大的作为预测结果
            for j, tr in enumerate(zip(target, pred)):
                tru, pre = tr
                if tru != pre:
                    print('第%d组数据的第%d张图,预测值为%d,真实值为%d' % (i, j, pre, tru))
            # for j, tru, pre in enumerate(zip(target, pred)):
            #     if tru != pre:
            #         print('第%d组数据的第%d张图,预测值为%d,真实值为%d' % (i, j, pre, tru))
            # 将一个batch的预测结果和真实值传入评估器，并在其内部生成混淆矩阵
            self.evaluator.add_batch(target, pred)
        # 获得数据
        Acc = self.evaluator.Accuracy()  # 准确率
        Acc_class = self.evaluator.Accuracy_Class()  # 不同类各自的准确率加起来求平均
        confusion_matrix = self.evaluator.back_matrix()  # 获得混淆矩阵
        # 绘制混淆矩阵
        self.summary.visualize_confusion_matrix(writer=self.writer,
                                                confusion_matrix=confusion_matrix,
                                                num_classes=self.nclass,
                                                class_names=self.class_names,
                                                global_step=epoch)
        # self.writer.add_scalar('验证/total_loss_epoch', test_loss, epoch)
        # self.writer.add_scalar('验证/准确率', Acc, epoch)
        # self.writer.add_scalar('验证/类准确率', Acc_class, epoch)
        if valortrain == 'train':
            print("train数据集的测试结果:")
        else:
            print("val数据集的测试结果:")
        print("准确率:{}, 类准确率:{}".format(Acc, Acc_class))
        print('Loss: %.3f' % test_loss)


def main():
    parser = argparse.ArgumentParser(description="PyTorch ValidateTrainedModel")
    parser.add_argument('--backbone', type=str, default='xception',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--dataset', type=str, default='fattyliver',
                        choices=['pascal', 'coco', 'fattyliver'],  # 加入自己的数据集选项fattyliver
                        help='dataset name (default: fattyliver)')
    # 数据加载线程
    """
    help是整个命令的参数信息，metavar是此命令的某个参数的help信息
    """
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    # 原图的大小
    parser.add_argument('--base-size', type=int, default=636,
                        help='base image size')
    # 裁剪图像大小
    parser.add_argument('--crop-size', type=int, default=636,
                        help='crop image size')
    # 选择使用的loss函数， 默认是ce
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # batch的大小
    parser.add_argument('--batch-size', type=int, default=10,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    # 测试时batch的大小
    parser.add_argument('--test-batch-size', type=int, default=10,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    # 用cuda
    parser.add_argument('--no-cuda', action='store_false', default=False,
                        help='disables CUDA training')
    # gpu号 选择用哪个gpu训练 ，输入必须是逗号分隔的整数列表
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    # 随机种子
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    # 设置检查点的名字
    parser.add_argument('--checkname', type=str, default=r'/root/tf-logs/',
                        help='set the checkpoint name')
    args = parser.parse_args()  # 解析出这些参数
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]  # 取出gpu号
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    print(args)
    torch.manual_seed(args.seed)
    valier = TrainedModleValidator(args)

    # valier.validation('train')  # 评估一下
    # valier.validation('val')  # 评估一下
    valier.validation('all')  # 评估一下
    valier.writer.close()


if __name__ == "__main__":
    main()
"""
有问题的数据
第0组数据的第2张图,预测值为1,真实值为0
第0组数据的第4张图,预测值为1,真实值为0
第2组数据的第9张图,预测值为1,真实值为0
第6组数据的第1张图,预测值为1,真实值为0
第6组数据的第2张图,预测值为1,真实值为0
第6组数据的第8张图,预测值为1,真实值为0
第17组数据的第4张图,预测值为1,真实值为2
第17组数据的第5张图,预测值为1,真实值为2
第17组数据的第8张图,预测值为1,真实值为2
第49组数据的第6张图,预测值为3,真实值为1
第52组数据的第0张图,预测值为1,真实值为3
第52组数据的第7张图,预测值为1,真实值为3
"""