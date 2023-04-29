import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable


# 绘制热力图
class GradCam:
    def __init__(self, model):
        self.model = model.eval()
        self.feature = None
        self.gradient = None

    def save_gradient(self, grad):
        self.gradient = grad
        # return grad

    def __call__(self, x):
        image_size = (x.size(-1), x.size(-2))
        datas = Variable(x)
        heat_maps = []
        for i in range(datas.size(0)):
            img = datas[i].data.cpu().numpy()

            feature = datas[i].unsqueeze(0)
            output, feature = self.model(feature)
            feature.register_hook(self.save_gradient)  # 等backword时，这个钩子会记录梯度
            self.feature = feature

            # 预测得分最高的那一类对应的输出score
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)  # 选出计算出的概率最大的作为预测结果
            # print('***********pred:',pred)
            pred_class = output[:, pred]
            pred_class.backward()  # 计算梯度
            grads = self.gradient  # 获取梯度
            # print(grads.shape)
            # pooled_grads = paddle.nn.functional.adaptive_avg_pool2d( x = grads, output_size=[1, 1])
            pooled_grads = grads

            # 此处batch size默认为1，所以去掉了第0维（batch size维）
            # print('pooled_grads:', pooled_grads.shape)  # [1, 2048, 28, 40])
            pooled_grads = pooled_grads[0]
            # print('pooled_grads:', pooled_grads.shape)  # [2048, 28, 40])
            feature = feature[0]
            # print(feature.shape)
            # 最后一层feature的通道数
            for j in range(2048):
                feature[j, ...] *= pooled_grads[j, ...]

            heatmap = feature.detach().cpu().numpy()
            heatmap = np.mean(heatmap, axis=0)
            # print(heatmap)
            heatmap = np.maximum(heatmap, 0)
            # print('+++++++++',heatmap)
            heatmap /= np.max(heatmap)
            # print('+++++++++',heatmap)
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]))  # 将热力图的大小调整为与原始图像相同
            heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图变成伪彩色图像
            # img = np.tile(img, (3, 1, 1))  # 灰度图重复三次变成rgb形式
            # print("img的shape", img.shape)
            img = np.swapaxes(img, 0, 2)  # 交换第一维和第三维
            superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
            superimposed_img = np.swapaxes(superimposed_img, 0, 2)  # 交换第一维和第三维
            heat_maps.append(superimposed_img)
        return torch.tensor(heat_maps)
        # return heat_maps
