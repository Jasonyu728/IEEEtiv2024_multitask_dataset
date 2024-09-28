# *-* coding: utf-8 *-*
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
import itertools
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import numpy as np
import os
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import torchvision.models as models


def cluster_acc(Y_pred, Y):
    from linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, w


class Model(nn.Module):
    def __init__(self, input_dim=784, inter_dims=[500, 500, 2000], hid_dim=10):
        super(Model, self).__init__()

        self.encoder = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            #
            # # conv5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),
        )

        self.classifier1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 20),
        )

        self.classifier2 = nn.Sequential(
            nn.Linear(20, 2),
        )

        self.classifier3 = nn.Sequential(
            nn.Linear(20, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 512 * 7 * 7),
        )

        self.decoder = nn.Sequential(
            # deconv1
            nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(512, 512, 3, padding=1),  # 10
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, padding=1),  # 12
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, padding=1),  # 14
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # deconv2
            nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(512, 512, 3, padding=1),  # 17
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, padding=1),  # 19
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, padding=1),  # 21
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # deconv3
            nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(256, 256, 3, padding=1),  # 24
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, padding=1),  # 26
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, padding=1),  # 28
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # deconv4
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, padding=1),  # 32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, padding=1),  # 34
            nn.BatchNorm2d(64),

            # deconv5
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, padding=1),  # 37
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.ConvTranspose2d(64, 3, 3, padding=1),
            # nn.BatchNorm2d(3),
        )

        self.conv2deconv_indices = {
            2: 28, 5: 25, 7: 23,
            10: 19, 12: 17, 14: 15, 17: 12,
            19: 10, 21: 8, 24: 5, 26: 3, 28: 1
        }

        self.restruction = nn.Sequential(nn.ConvTranspose2d(64, 3, 3, padding=1))

        self.init_weights()

    def init_weights(self):
        """
        initial weights from preptrained model by vgg16

        """
        vgg16_pretrained = models.vgg16(pretrained=True)
        num = 1
        for idx, layer in enumerate(vgg16_pretrained.features):
            if idx <= 30:
                if isinstance(layer, nn.Conv2d) and idx == 0:
                    self.encoder[idx].weight.data = layer.weight.data
                    self.encoder[idx].bias.data = layer.bias.data
                if isinstance(layer, nn.Conv2d) and idx != 0:
                    self.encoder[idx + num].weight.data = layer.weight.data
                    self.encoder[idx + num].bias.data = layer.bias.data
                    num += 1
                # if isinstance(layer, nn.Conv2d) and idx != 0:
                #    self.decoder[self.conv2deconv_indices[idx]].weight.data = layer.weight.data
                # if idx == 0:
                #  self.restruction[0].weight.data = layer.weight.data

    def forward(self, x):
        # x=self.encoder(x)
        locations = []
        # 遍历编码器部分，如果是最大池化层，则记录最大池化的位置信息，并应用最大池化操作
        for idx, layer in enumerate(self.encoder):
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
                # print(x.shape)
                locations.append(location)
            else:
                x = layer(x)
                # print(x.shape)

        e = x.view(x.size(0), -1)   # 将编码器部分输出的特征x展平为一维向量
        id = self.classifier1(e)    # 生成20维的向量id，用于后续任务
        sex = self.classifier2(id)  # 生成2维的sex，用于粗粒度分类

        # for i in range(80):
        #  fine[0][i] = 0
        # 进行特征重建：将id重新映射为一个高维的特征图z(batch_size, 512, 7, 7)，用于后续解码部分
        z = self.classifier3(id).view(id.size(0), 512, 7, 7)

        # 遍历解码器部分
        num = 4
        for idx, layer in enumerate(self.decoder):
            # 如果是最大解池层，则通过位置信息locations执行最大解池操作，以还原最大池化的效果
            if isinstance(self.decoder[idx], nn.MaxUnpool2d):
                z = layer(z, locations[num])
                num -= 1
            # 对z进行反向还原
            else:
                z = layer(z)

        restruction = self.restruction(z)

        return id, sex, restruction
