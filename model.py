import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np


class ResBlock(nn.Module):

    def __init__(self):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(num_parameters=64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)

        nn.init.constant_(self.conv1.bias, 0.01)
        nn.init.constant_(self.conv2.bias, 0.01)

    def forward(self, x):
        
        y = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        x += y

        return x


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

        self.conv1   = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.prelu1  = nn.PReLU(num_parameters=64)
        
        self.resbl1  = ResBlock()
        self.resbl2  = ResBlock()
        self.resbl3  = ResBlock()
        self.resbl4  = ResBlock()
        self.resbl5  = ResBlock()
        self.resbl6  = ResBlock()
        self.resbl7  = ResBlock()
        self.resbl8  = ResBlock()
        self.resbl9  = ResBlock()
        self.resbl10 = ResBlock()
        self.resbl11 = ResBlock()
        self.resbl12 = ResBlock()
        self.resbl13 = ResBlock()
        self.resbl14 = ResBlock()
        self.resbl15 = ResBlock()
        self.resbl16 = ResBlock()

        self.conv2   = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2     = nn.BatchNorm2d(64)

        self.conv3   = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1)
        self.pixsf3  = nn.PixelShuffle(upscale_factor=2)
        self.prelu3  = nn.PReLU(num_parameters=64)

        self.conv4   = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1)
        self.pixsf4  = nn.PixelShuffle(upscale_factor=2)
        self.prelu4  = nn.PReLU(num_parameters=64)

        self.conv5   = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, padding=4)

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.xavier_normal_(self.conv4.weight)
        nn.init.xavier_normal_(self.conv5.weight)
        
        nn.init.constant_(self.conv1.bias, 0.01)
        nn.init.constant_(self.conv2.bias, 0.01)
        nn.init.constant_(self.conv3.bias, 0.01)
        nn.init.constant_(self.conv4.bias, 0.01)
        nn.init.constant_(self.conv5.bias, 0.01)

    def forward(self, x):

        x = self.conv1(x)
        x = self.prelu1(x)

        y = x

        x = self.resbl1(x)
        x = self.resbl2(x)
        x = self.resbl3(x)
        x = self.resbl4(x)
        x = self.resbl5(x)
        x = self.resbl6(x)
        x = self.resbl7(x)
        x = self.resbl8(x)
        x = self.resbl9(x)
        x = self.resbl10(x)
        x = self.resbl11(x)
        x = self.resbl12(x)
        x = self.resbl13(x)
        x = self.resbl14(x)
        x = self.resbl15(x)
        x = self.resbl16(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += y

        x = self.conv3(x)
        x = self.pixsf3(x)
        x = self.prelu3(x)

        x = self.conv4(x)
        x = self.pixsf4(x)
        x = self.prelu4(x)

        x = self.conv5(x)

        return x


class VGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(VGGBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride
                              )
        
        self.bn   = nn.BatchNorm2d(num_features=out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        nn.init.xavier_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.01)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)

        return x


class VGG13(nn.Module):

    def __init__(self):
        super(VGG13, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.leak0 = nn.LeakyReLU(0.2, inplace=True)

        self.vggb1 = VGGBlock(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        self.vggb2 = VGGBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.vggb3 = VGGBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2)
        self.vggb4 = VGGBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.vggb5 = VGGBlock(in_channels=256, out_channels=256, kernel_size=3, stride=2)
        self.vggb6 = VGGBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1)
        self.vggb7 = VGGBlock(in_channels=512, out_channels=512, kernel_size=3, stride=2)

        self.dens1 = nn.Linear(in_features=3 * 3 * 512, out_features=1024)
        self.leak1 = nn.LeakyReLU(0.2, inplace=True)
        self.dens2 = nn.Linear(in_features=1024, out_features=1)
        self.sigmd = nn.Sigmoid()

        nn.init.xavier_normal_(self.conv0.weight)
        nn.init.xavier_normal_(self.dens1.weight)
        nn.init.xavier_normal_(self.dens2.weight)

        nn.init.constant_(self.conv0.bias, 0.01)
        nn.init.constant_(self.dens1.bias, 0.01)
        nn.init.constant_(self.dens2.bias, 0.01)

    def forward(self, x):

        x = self.conv0(x)
        x = self.leak0(x)

        x = self.vggb1(x)
        x = self.vggb2(x)
        x = self.vggb3(x)
        x = self.vggb4(x)
        x = self.vggb5(x)
        x = self.vggb6(x)
        x = self.vggb7(x)

        x = x.view(-1, 3 * 3 * 512)

        x = self.dens1(x)
        x = self.leak1(x)
        x = self.dens2(x)
        x = self.sigmd(x)

        return x


class VGG19(nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()

        vgg19 = torchvision.models.vgg19_bn(pretrained=True)
        for param in vgg19.parameters():
            param.requires_grad = False
        self.features = nn.Sequential(*list(vgg19.features.children())[:-1])

    def forward(self, x):
        
        x = self.features(x)

        return x

        
class Edge(nn.Module):
    
    def __init__(self, mode='RGB', s=1):
        super(Edge, self).__init__()

        if mode == 'RGB' or mode == 'L':
            pass
        else:
            raise NotImplementedError('mode must be RGB or L')

        self.mode = mode

        self.s = s

        self.g_kernel = np.zeros([5, 5], dtype=np.float32)

        for i in range(5):
            for j in range(5):
                self.g_kernel[i, j] = (1 / (2 * np.pi * s * s)) * np.exp(-((i - 2) ** 2 + (j - 2) ** 2) / (2 * s * s))

        self.g_filter = torch.from_numpy(self.g_kernel).unsqueeze(0).unsqueeze(0)
        self.g_filter = self.g_filter.cuda()
        
        self.zero_filter = torch.FloatTensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        self.h_filter = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        self.h_filter_r = torch.stack((self.h_filter, self.zero_filter, self.zero_filter))
        self.h_filter_g = torch.stack((self.zero_filter, self.h_filter, self.zero_filter))
        self.h_filter_b = torch.stack((self.zero_filter, self.zero_filter, self.h_filter))

        if mode == 'RGB':
            self.h_filter = torch.stack((self.h_filter_r, self.h_filter_g, self.h_filter_b))
        else:
            self.h_filter = self.h_filter.unsqueeze(0).unsqueeze(0)
        self.h_filter = self.h_filter.cuda()
        
        self.v_filter = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.v_filter_r = torch.stack((self.v_filter, self.zero_filter, self.zero_filter))
        self.v_filter_g = torch.stack((self.zero_filter, self.v_filter, self.zero_filter))
        self.v_filter_b = torch.stack((self.zero_filter, self.zero_filter, self.v_filter))

        if mode == 'RGB':
            self.v_filter = torch.stack((self.v_filter_r, self.v_filter_g, self.v_filter_b))
        else:
            self.v_filter = self.v_filter.unsqueeze(0).unsqueeze(0)
        self.v_filter = self.v_filter.cuda()
        
    def forward(self, x):

        gx = F.conv2d(x, self.g_filter, padding=2)
        
        dx = F.conv2d(gx, self.h_filter, padding=1)
        dy = F.conv2d(gx, self.v_filter, padding=1)
        
        d = torch.sqrt(dx ** 2 + dy ** 2)
        
        # d = torch.div(d, 5.6568542495)
        d = torch.div(d, d.max())

        # Transfer function
        d = d ** 3
        # d = torch.add(torch.mul(d, 0.9), 0.1)
        
        return d


class SRResNet(nn.Module):

    class ResBlock(nn.Module):

        def __init__(self):
            super().__init__()

            self.block = nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.PReLU(64),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
            )

        def forward(self, x):
            y = x
            return y + self.block(x)

    def __init__(self):
        super().__init__()

        self.checkpoint = './weights/ckpt'

        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1, 4),
            nn.PReLU(64),
        )
        self.res_blocks = nn.Sequential(*nn.ModuleList([self.ResBlock() for _ in range(16)]))
        self.post_res = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(64),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(64),
            nn.Conv2d(64, 3, 9, 1, 4),
        )

    def forward(self, x):
        x = self.init_conv(x)
        y = x
        x = y + self.post_res(self.res_blocks(x))
        return self.upsample(x)

    def init_conv_layers(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
                with torch.no_grad():
                    m.weight *= 0.1

    def restoreCheckpoint(self):
        self.load_state_dict(torch.load(self.checkpoint))

    def saveCheckpoint(self):
        torch.save(self.state_dict(), self.checkpoint)
































