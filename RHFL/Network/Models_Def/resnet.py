'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.backbone = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(),
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            nn.AvgPool2d(kernel_size=4)
        )

        # projector 2 layer
        sizes = [512*block.expansion,1024,1024,1024]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        layers.append(nn.BatchNorm1d(sizes[-1], affine=False))
        self.bn_projector = nn.Sequential(*layers)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.backbone(x)
        out = out.view(out.size(0), -1)
        linear_output = self.linear(out)
        embedding_output = self.bn_projector(out)
        return linear_output, embedding_output


def ResNet10(num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1, 1],num_classes)

def ResNet12(num_classes=10):
    return ResNet(BasicBlock, [2, 1, 1, 1],num_classes)

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

#from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ =='__main__':
    net = ResNet10()
    print(net.parameters('linear'))
    print(net.named_parameters())
    for i in net.named_parameters():
        print(i)
    for name, param in net.named_parameters():
        print(name)
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    # # device_ids = [0,1,2,3,4,5,6,7]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # device_ids = [0,1,2,3]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.backends.cudnn.benchmark = True
    # network = nn.DataParallel(net, device_ids=device_ids).to(device)
    # for para in network.module.bn_projector.parameters():
    #     para
    #     para.requires_grad = False
    # print(network.parameters('bn_projector'))
    # for para in network.parameters('bn_projector'):
    #     print(para)
        # if "fc1" in name:
        #     param.requires_grad = False
    # print(net.state_dict)
    # y = net(torch.randn(2, 3, 32, 32),private=True)
    # summary(net.to(device), input_size=(3, 32, 32))
    # print(y.size())
    # net = ResNet12()
    # y = net(torch.randn(2, 3, 32, 32),private=False)
    # print(y.size())