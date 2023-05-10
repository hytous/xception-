import os

import numpy as np
import torch
import tensorboard
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from tqdm.contrib import itertools
from dataloaders.utils import decode_seg_map_sequence


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, image, global_step):
        # grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)  # 一次显示多个图
        grid_image = make_grid(image, 5, normalize=True)  # 一次显示多个图
        writer.add_image('Image', grid_image, global_step)

    # 绘制混淆矩阵
    def visualize_confusion_matrix(
            self,
            writer,
            confusion_matrix,
            num_classes=4,  # 默认分4类
            class_names=None,  # 类名
            subset_ids=None,  # 只显示部分混淆矩阵
            global_step=None
    ):
        tag = "confusion matrix"

        if subset_ids is None or len(subset_ids) != 0:
            # If class names are not provided, use class indices as class names.
            if class_names is None:
                class_names = [str(i) for i in range(num_classes)]
            # If subset is not provided, take every classes.
            if subset_ids is None:
                subset_ids = list(range(num_classes))

            sub_cmtx = confusion_matrix[subset_ids, :][:, subset_ids]
            sub_names = [class_names[j] for j in subset_ids]

            sub_cmtx = self.plot_confusion_matrix(
                sub_cmtx,
                num_classes=len(subset_ids),
                class_names=sub_names,
            )

            # Add the confusion matrix image to writer.
            writer.add_figure(tag=tag, figure=sub_cmtx, global_step=global_step)

    def plot_confusion_matrix(self, confusion_matrix, num_classes, class_names=None, figsize=None):
        if class_names is None or type(class_names) != list:
            class_names = [str(i) for i in range(num_classes)]

        figure = plt.figure(figsize=figsize)
        plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # 往混淆矩阵的方块上写数字
        # Use white text if squares are dark; otherwise black.
        # 高于中间值字就是白色,低于中间值字就是黑色
        threshold = confusion_matrix.max() / 2.0
        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            color = "white" if confusion_matrix[i, j] > threshold else "black"
            plt.text(
                j,
                i,
                format(confusion_matrix[i, j], ".2f") if confusion_matrix[i, j] != 0 else ".",
                horizontalalignment="center",
                color=color,
            )

        plt.tight_layout()
        plt.ylabel("Predicted label")
        plt.xlabel("True label")

        return figure
