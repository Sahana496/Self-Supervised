import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import spectral_norm

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))


def snlinear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)

        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)

        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)

        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)

        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)

        # Out
        out = x + self.sigma*attn_g
        return out

class Residual_G(nn.Module):
    def __init__(self, in_channels, out_channels = 256, kernel_size = 3, stride = 1, up_sampling = False):
        super(Residual_G, self).__init__()
        self.up_sampling = up_sampling
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.conv1 = snconv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = 1)
        self.conv2 = snconv2d(out_channels, out_channels,kernel_size = kernel_size, stride = stride, padding = 1)
        self.learnable_sc = in_channels != out_channels or up_sampling
        if self.learnable_sc:
            self.c_sc = snconv2d(in_channels,out_channels,kernel_size = 1, stride = 1, padding=0)

    def forward(self, x):
        input = x
        x = self.relu(self.batch_norm1(x))
        if self.up_sampling:
            x = self.upsample(x)
        x = self.conv1(x)
        x = self.batch_norm2(x)
        x = self.conv2(self.relu(x))
        return x + self.shortcut(input)

    def shortcut(self, x):
        if self.upsample:
            x = self.upsample(x)
        if self.learnable_sc:
            x = self.c_sc(x)
        return x


class Residual_D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel = 3, stride = 1, down_sampling = False, is_start = False):
        super(Residual_D, self).__init__()
        self.down_sampling = down_sampling
        self.is_start = is_start
        self.conv1 = snconv2d(in_channels, out_channels,kernel_size = kernel, stride = stride, padding = 1)
        self.conv2 = snconv2d(out_channels, out_channels,kernel_size = kernel, stride = stride, padding = 1)
        self.avgpool2 = nn.AvgPool2d(2, 2, padding = 0)
        self.relu = nn.ReLU()
        self.learnable_sc = (in_channels != out_channels) or down_sampling

        if self.learnable_sc:
            self.c_sc = snconv2d(in_channels,out_channels,kernel_size=1, stride=1,padding=0)

    def forward(self, x):
        input = x
        if self.is_start:
            x = self.relu(self.conv1(x))
            x = self.conv2(x)
            x = self.avgpool2(x)
            return x + self.shortcut(input)
        else:
            x = self.conv1(self.relu(x))
            x = self.conv2(self.relu(x))
            if self.down_sampling:
                x = self.avgpool2(x)

        return x + self.shortcut(input) 

    def shortcut(self,x):
        if self.is_start:
            return self.c_sc(self.avgpool2(x))
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.down_sampling:
               x = self.avgpool2(x)

        return x