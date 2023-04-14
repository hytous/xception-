import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone


class Builder(nn.Module):
    def __init__(self, backbone='xception', output_stride=16, num_classes=4,
                 sync_bn=True, freeze_bn=False):
        super(Builder, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        # 多gpu和单gpu的batchnorm不同
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)

        self.freeze_bn = freeze_bn

        # self.flat = nn.Flatten()
        # self.fc = nn.Linear(5418, 4, bias=False)

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    # 返回每层需要被训练的参数
    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            # modules[0]里存了所有层的名称、参数等信息
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:  # 要被训练
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


if __name__ == "__main__":
    model = Builder(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())
