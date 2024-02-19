# change from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LogSoftmax
from torch.nn.init import kaiming_normal_, zeros_
from models.modules import LDM, GDM, SAM, SELayer, NonLocalBlock

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class down_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_conv, self).__init__()
        self.go = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.go(x)

class up_concate_conv(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_concate_conv, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

        self.conv = double_conv(in_ch * 2, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        if x1.size(2) != x2.size(2) or x1.size(3) != x2.size(3):
            x1 = F.interpolate(x1, size=(x2.size(2), x2.size(3)), mode='bilinear', align_corners=True)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class unet_slg(nn.Module):
    def __init__(self, num_classes, input_channel=3):
        super(unet_slg, self).__init__()
        self.pre_conv = double_conv(input_channel, 64)  # 1
        self.down1 = down_conv(64, 128)  # 1/2
        self.ldm1 = LDM(128)
        self.gdm1 = GDM(128)
        self.sam1 = SAM(128)


        self.down2 = down_conv(128, 256)  # 1/4
        self.ldm2 = LDM(256)
        self.gdm2 = GDM(256)
        self.sam2 = SAM(256)

        self.down3 = down_conv(256, 512)  # 1/8
        self.ldm3 = LDM(512)
        self.gdm3 = GDM(512)
        self.sam3 = SAM(512)
        self.down4 = down_conv(512, 512)  # 1/16

        self.up1 = up_concate_conv(512, 256)
        self.up2 = up_concate_conv(256, 128)
        self.up3 = up_concate_conv(128, 64)
        self.up4 = up_concate_conv(64, 64)
        self.end_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x1 = self.pre_conv(x)
        x2 = self.down1(x1)
        l1 = self.ldm1(x2)
        g1 = self.gdm1(x2)
        x2 = self.sam1(x2,l1,g1)
        x3 = self.down2(x2)
        l2 = self.ldm2(x3)
        g2 = self.gdm2(x3)
        x3 = self.sam2(x3,l2,g2)
        x4 = self.down3(x3)
        l3 = self.ldm3(x4)
        g3 = self.gdm3(x4)
        x4 = self.sam3(x4,l3,g3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.end_conv(x)
        return x





