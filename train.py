import argparse
import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary 可视化
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)  # 随机梯度下降

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler  学习率调整器
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)  # 进度条，使python进度可视化



        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
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
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

def main():
    # parser 解析器
    # argparse 模块是 Python 内置的用于命令项选项与参数解析的模块，
    # argparse 模块可以让人轻松编写用户友好的命令行接口，能够帮助程序员为模型定义参数。
    """
    https://blog.csdn.net/feichangyanse/article/details/128559542
    导入argparse包 ——import argparse
    创建一个命令行解析器对象 ——创建 ArgumentParser() 对象
    给解析器添加命令行参数 ——调用add_argument() 方法添加参数
    解析命令行的参数 ——使用 parse_args() 解析添加的参数
    """

    """
    首先我们导入argparse这个包，然后包中的ArgumentParser类生成一
    个parser对象（其中的description对参数解析器的作用进行描述）
    ，当我们在命令行显示帮助信息的时候会看到description描述的信息。
    """
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")

    """
    接着我们通过对象的add_argument函数来增加参数。
    """
    # 网络骨架 有：残差神经网络、xception神经网络、深度残差网络（drn）、mobilenet网络可以选，默认是残差神经网络
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    # 输出步长 默认是8。是用于图像分割的参数，应该是和ASPP和deeplabc3plus算法相关
    # deeplabv3plus里有一段说明
    # out stride这个值是输入图像的空间分辨率和输出特征图的空间分辨率的比值。
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    # 选择数据集，应该都是谷歌数据集
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    # 是否使用sbd数据集，默认是用的
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    # 数据加载线程
    """
    help是整个命令的参数信息，metavar是此命令的某个参数的help信息
    """
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    # 原图的大小
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    # 裁剪图像大小
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    # 是否使用 同步Batch Normalization 默认不用
    """sync bn ，Cross-GPU Synchronized Batch Normalization，
    是因为多显卡训练中每张卡上的batch size过小，不同步会造成BN层失效。
    单卡直接用原始就好了，不需要开sync—bn。
    """
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    # 是否将bn层冻结 默认不冻结
    """
    随着网络深度的加深，函数变的越来越复杂，每一层的输出的数据分布变化
    越来越大。BN的作用就是把数据强行拉回我们想要的比较好的正态分布下。这样
    可以在一定程度上避免梯度爆炸或者梯度消失的问题，加快收敛的速度。
    如果batch_size太小，计算一个小batch_size的均值和方差，肯定没有计
    算大的batch_size的均值和方差稳定和有意义，这个时候，还不如不使用bn层，因此可以将bn层冻结。
    """
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    # 选择使用的loss函数， 默认是ce
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    # 迭代次数
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    # 开始迭代 默认从第0代开始
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    # 训练时batch的大小
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    # 测试时batch的大小
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    # 训练数据样本不均衡（比如某类样本占80%）时使用
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    # 学习率
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    # 调整学习率的机制
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    # 惯量 使学习更加平滑
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    # 权重衰减 防止过拟合用的
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    # Nesterov动量优化算法是Momentum优化算法的一种改进。
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    # 不用cuda训练
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    # gpu号 选择用哪个gpu训练 ，输入必须是逗号分隔的整数列表
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    # 随机种子
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    # 如果需要，放置恢复文件的路径
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    # 设置检查点的名字
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models 微调预训练模型
    # 因数据集不同而进行微调
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    # 隔几代评估一次
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    # 在训练中跳过验证操作
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    """
    最后采用对象的parse_args获取解析的参数。
    """
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]  # 取出gpu号
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:  # 如果多gpu训练，就打开sync_bn
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr 设置迭代次数、batch大小、学习率的默认值
    if args.epochs is None:  # 设置在不同数据集上迭代次数的默认值
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]  # lower()方法转换字符串中所有大写字符为小写

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)  # 只有一个gpu的话，默认batch大小为4

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:  # 设置在不同数据集上学习率的默认值
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)  # deeplab+算法骨架
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)  # 开始迭代 默认从第0代开始
    print('Total Epoches:', trainer.args.epochs)  # 总迭代次数
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        # 如果不跳过评估阶段，并且到该评估的epoch了
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)  # 评估一下

    trainer.writer.close()

if __name__ == "__main__":
   main()
