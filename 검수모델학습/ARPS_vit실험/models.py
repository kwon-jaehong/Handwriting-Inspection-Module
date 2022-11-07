
import torch
import torch.nn as nn
from torch.nn import functional as F
from ABN import MultiBatchNorm

def weights_init_ABN(m):
    classname = m.__class__.__name__
    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('MultiBatchNorm') != -1:
        m.bns[0].weight.data.normal_(1.0, 0.02)
        m.bns[0].bias.data.fill_(0)
        m.bns[1].weight.data.normal_(1.0, 0.02)
        m.bns[1].bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels,identity_downsample=None, stride=1,num_ABN=2):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = MultiBatchNorm(intermediate_channels,num_ABN)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = MultiBatchNorm(intermediate_channels,num_ABN)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = MultiBatchNorm(intermediate_channels * self.expansion,num_ABN)
        self.leakerelu = nn.LeakyReLU(0.2)
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self,device, x,bn_label=None):
        if bn_label is None:
            bn_label = 0 * torch.ones(x.shape[0], dtype=torch.long).to(device=device)
            
        identity = x.clone()

        x = self.conv1(x)
        x,_ = self.bn1(x,bn_label)
        x = self.leakerelu(x)
        x = self.conv2(x)
        x,_ = self.bn2(x,bn_label)
        x = self.leakerelu(x)
        x = self.conv3(x)
        x,_ = self.bn3(x,bn_label)

        if self.identity_downsample is not None:
            identity = self.identity_downsample[0](identity)
            identity,_ = self.identity_downsample[1](identity,bn_label)

        x += identity
        x = self.leakerelu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes,feat_dim,num_ABN=2):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = MultiBatchNorm(64,num_ABN)
        self.lakerelu = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 
        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=64, stride=1,num_ABN=num_ABN)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=128, stride=2,num_ABN=num_ABN)
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=256, stride=2,num_ABN=num_ABN)
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels=512, stride=2,num_ABN=num_ABN)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(512 * 4, feat_dim)
        
        self.fc_last = nn.Linear(feat_dim, num_classes)
        
        self.apply(weights_init_ABN)

    # net(device,fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).to(device=device))
    def forward(self,device, x,return_feature=False, bn_label=None):
        if bn_label is None:
            bn_label = 0 * torch.ones(x.shape[0], dtype=torch.long).to(device=device)
        
        # 앞단
        x = self.conv1(x)
        x,_ = self.bn1(x,bn_label)
        x = self.lakerelu(x)
        x = self.maxpool(x)
        
        ## 보틀넥 구조 통과
        for i in range(0,len(self.layer1)):
            x = self.layer1[i](device,x,bn_label)
        
        for i in range(0,len(self.layer2)):
            x = self.layer2[i](device,x,bn_label)
        
        for i in range(0,len(self.layer3)):
            x = self.layer3[i](device,x,bn_label)
            
        for i in range(0,len(self.layer4)):
            x = self.layer4[i](device,x,bn_label)
            

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        y = self.fc_last(x)
        
        if return_feature:
            return x, y
        else:
            return y


    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride,num_ABN):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                MultiBatchNorm(intermediate_channels * 4,num_ABN),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )


        self.in_channels = intermediate_channels * 4


        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channel, num_classes,feat_dim):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes,feat_dim)
def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)
def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)

