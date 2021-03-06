import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils import model_zoo
from torchvision import models




class UNetEnc(nn.Module):

    def __init__(self, in_channels, features, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_channels, features, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features, out_channels, 2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class UNetDec(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers += [nn.Dropout(.5)]
        layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)


class UNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.dec1 = UNetDec(2, 16)
        self.dec2 = UNetDec(16, 32, dropout=True)
        self.center = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.enc2 = UNetEnc(64, 32, 16)
        self.enc1 = nn.Sequential(
            nn.Conv2d(32, 16, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(16, num_classes, 1)
    

    def forward(self, x):
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        center = self.center(dec2)
        enc2 = self.enc2(torch.cat([
            center, F.upsample_bilinear(dec2, center.size()[2:])], 1))
        enc1 = self.enc1(torch.cat([
            enc2, F.upsample_bilinear(dec1, enc2.size()[2:])], 1))
        return F.log_softmax(F.upsample_bilinear(self.final(enc1), x.size()[2:]))


