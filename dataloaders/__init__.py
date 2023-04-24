# 声明一下自己的数据类
from dataloaders.datasets import fattyliver
from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):

    if args.dataset == 'fattyliver':
        class_names = ['health', 'mild', 'moderate', 'severe']  # 健康 轻度 中度 重度
        train_set = fattyliver.FattyLiver(args, split='train')
        val_set = fattyliver.FattyLiver(args, split='val')
        all_set = fattyliver.FattyLiver(args, split='all')
        num_class = train_set.NUM_CLASSES
        # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        all_loader = DataLoader(all_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, val_loader, all_loader, num_class, class_names

    else:
        raise NotImplementedError

