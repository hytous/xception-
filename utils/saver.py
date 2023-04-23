import os
import shutil
import torch
from collections import OrderedDict
import glob


class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.checkname)
        # 现有checkpoint的文件名
        # sorted() 作为 Python 内置函数之一，其功能是对序列（列表、元组、字典、集合、还包括字符串）进行排序
        # glob.glob https://blog.csdn.net/shary_cao/article/details/122050756
        # 其功能是返回一个与pathname匹配的路径名列表（该列表可以为空，必须是符合路径规范的字符串）
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment{}_*'.format(str(args.backbone)))))
        # 找到之前存的最后一个checkpoint的文件名，把文件名的序号拿出来加一就是新checkpoint的序号
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment{}_{}'.format(str(args.backbone), str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        print("存储了模型，模型的存储路径是 :", filename)
        torch.save(state, filename)
        if is_best:
            print("模型是目前最优的,保存")
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    # 存储args，也就是此次训练的各种选项（在main函数里）
    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['datset'] = self.args.dataset
        p['backbone'] = self.args.backbone
        p['out_stride'] = self.args.out_stride
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()