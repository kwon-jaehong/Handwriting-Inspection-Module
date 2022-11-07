## reference code is https://github.com/pytorch/examples/blob/master/dcgan/main.py

from numpy import outer
import torch
import torch.nn as nn
import torch.nn.functional as F
import os



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netD(nn.Module):
    def __init__(self, nc, ndf):
        super(_netD, self).__init__()

        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.leak_relu = nn.LeakyReLU(0.2, inplace=True)


        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)

        
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        
        
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        
        
        # self.conv5 = nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False)
        # self.bn5 = nn.BatchNorm2d(ndf * 32)

        
        self.conv6 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.leak_relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leak_relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leak_relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leak_relu(x)
        
        # x = self.conv5(x)
        # x = self.bn5(x)
        # x = self.leak_relu(x)
        
        
        x = self.conv6(x)
        output = self.sigmoid(x)

        return output.view(-1, 1)

class _netG(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(_netG, self).__init__()
        
        # nz, ngf, nc
        # (100, 64, 1)
        # nc - 입력 이미지의 색 채널개수입니다.
        # nz - 잠재공간 벡터의 원소들 개수입니다.
        # ngf - 생성자를 통과할때 만들어질 특징 데이터의 채널개수입니다.
        # ndf - 구분자를 통과할때 만들어질 특징 데이터의 채널개수입니다.

        
        # input is Z, going into a convolution
        ## 입력 채널, 아웃풋 채널수, 커널사이즈, 스트라이드, 패딩
        self.conv1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf * 8)
        self.relu = nn.ReLU(True)
        # state size. (ngf*8) x 4 x 4
        
        self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 4)
        
        # state size. (ngf*4) x 8 x 8
        self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf *2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 2)
        
        self.conv4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        
        
        self.conv6 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)        
        self.Tanh = nn.Tanh()
        
    def forward(self, x):
        
        # 입력전 x torch.Size([256, 400, 1, 1])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
         # 입력후 x torch.Size([256, 2048, 4, 4])
        
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # 통과후 torch.Size([256, 1024, 8, 8])
        
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # torch.Size([256, 512, 16, 16])
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        # torch.Size([256, 256, 32, 32])
        
        # x = self.conv5(x)
        # x = self.bn5(x)
        # x = self.relu(x)
        # torch.Size([256, 128, 64, 64])
        
        x = self.conv6(x)
        # torch.Size([256, 1, 128, 128])
        output = self.Tanh(x)
        
        return output

def Generator(nz, ngf, nc):
    model = _netG(nz, ngf, nc)
    model.apply(weights_init)
    return model

def Discriminator(nc, ndf):
    model = _netD(nc, ndf)
    model.apply(weights_init)
    return model

