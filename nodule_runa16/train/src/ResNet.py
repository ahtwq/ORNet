import torch
from torch import nn
import torch.backends.cudnn
import torch.nn.functional as F


config = {}
config['anchors'] = [5., 10., 20.]  # [ 10.0, 30.0, 60.]
config['chanel'] = 1
config['crop_size'] = [96, 96, 96]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1
config['num_hard'] = 2
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 2.5  # 3 #6. #mm
config['sizelim2'] = 10  # 30
config['sizelim3'] = 20  # 40
config['aug_scale'] = True
config['r_rand_crop'] = 0.3
config['pad_value'] = 170
config['augtype'] = {'flip': True, 'swap': False, 'scale': True, 'rotate': False}

config['augtype'] = {'flip': True, 'swap': False, 'scale': True, 'rotate': False}
config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38', '990fbe3f0a1b53878669967b9afd1441',
                       'adc3bbc63d40f8761c59be10f1e504c3']


class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride=1, bias=True):
        super(PostRes, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm3d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm3d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm3d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class PostResBottleneck(nn.Module):
    def __init__(self, n_channel, stride=1, bias=True):
        super(PostResBottleneck, self).__init__()
        self.conv1 = nn.Conv3d(n_channel[0], n_channel[1], kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm3d(n_channel[1])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(n_channel[1], n_channel[1], kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm3d(n_channel[1])
        self.conv3 = nn.Conv3d(n_channel[1], n_channel[2], kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm3d(n_channel[2])

        if stride != 1 or n_channel[0] == n_channel[1]:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_channel[0], n_channel[2], kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm3d(n_channel[2]))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)
        return out


class ResNet3D(nn.Module):
    def __init__(self):
        super(ResNet3D, self).__init__()

        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Conv3d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True)
        )

        self.forw1 = nn.Sequential(
            PostRes(24, 32),
            PostRes(32, 32),
        )
        self.forw2 = nn.Sequential(
            PostRes(32, 64),
            PostRes(64, 64),
        )
        self.forw3 = nn.Sequential(
            PostRes(64, 64),
            PostRes(64, 64),
            PostRes(64, 64),
        )
        self.forw4 = nn.Sequential(
            PostRes(64, 64),
            PostRes(64, 64),
            PostRes(64, 64),
        )

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        # (N, 1, 64, 64)
        out = self.preBlock(x)                       # (N, 1, H, H) -> (N, 24, H, H)
        out_pool, indices0 = self.maxpool1(out)      # (N, 24, H, H) -> (N, 24, H/2, H/2)
        out1 = self.forw1(out_pool)                  # (N, 24, H/2, H/2) -> (N, 32, H/2, H/2)
        out1_pool, indices1 = self.maxpool2(out1)    # (N, 32, H/2, H/2) -> (N, 32, H/4, H/4)
        out2 = self.forw2(out1_pool)                 # (N, 32, H/4, H/4) -> (N, 64, H/4, H/4)
        out2_pool, indices2 = self.maxpool3(out2)    # (N, 64, H/4, H/4) -> (N, 64, H/8, H/8)
        out3 = self.forw3(out2_pool)                 # (N, 64, H/8, H/8) -> (N, 64, H/8, H/8)
        out3_pool, indices3 = self.maxpool4(out3)    # (N, 64, H/8, H/8) -> (N, 64, H/16, H/16)
        out4 = self.forw4(out3_pool)                 # (N, 64, H/16, H/16) -> (N, 64, H/16, H/16)
        # (N, 64, 4, 4)
        return out4


class ResNet3D18(nn.Module):
    def __init__(self):
        super(ResNet3D18, self).__init__()

        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.forw1 = nn.Sequential(
            PostRes(64, 64, bias=False),
            PostRes(64, 64, bias=False),
        )
        self.forw2 = nn.Sequential(
            PostRes(64, 128, stride=2, bias=False),
            PostRes(128, 128, bias=False),
        )
        self.forw3 = nn.Sequential(
            PostRes(128, 256, stride=2, bias=False),
            PostRes(256, 256, bias=False),
        )
        self.forw4 = nn.Sequential(
            PostRes(256, 512, stride=2, bias=False),
            PostRes(512, 512, bias=False),
        )
        # self.avgpool = nn.AvgPool3d((2, 2, 2))

    def forward(self, x):
        # (N, 1, 64, 64)
        out = self.preBlock(x)                       # (N, 1, H, H) -> (N, 24, H, H)
        out1 = self.forw1(out)                  # (N, 24, H/2, H/2) -> (N, 32, H/2, H/2)
        out2 = self.forw2(out1)                 # (N, 32, H/4, H/4) -> (N, 64, H/4, H/4)
        out3 = self.forw3(out2)                 # (N, 64, H/8, H/8) -> (N, 64, H/8, H/8)
        out4 = self.forw4(out3)                 # (N, 64, H/16, H/16) -> (N, 64, H/16, H/16)
        # (N, 64, 4, 4)
        return out4


class ResNet3D34ThreeChannel(nn.Module):
    def __init__(self):
        super(ResNet3D34ThreeChannel, self).__init__()

        self.preBlock = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.forw1 = nn.Sequential(
            PostRes(64, 64, bias=False),
            PostRes(64, 64, bias=False),
            PostRes(64, 64, bias=False),
        )
        self.forw2 = nn.Sequential(
            PostRes(64, 128, stride=2, bias=False),
            PostRes(128, 128, bias=False),
            PostRes(128, 128, bias=False),
            PostRes(128, 128, bias=False),
        )
        self.forw3 = nn.Sequential(
            PostRes(128, 256, stride=2, bias=False),
            PostRes(256, 256, bias=False),
            PostRes(256, 256, bias=False),
            PostRes(256, 256, bias=False),
            PostRes(256, 256, bias=False),
            PostRes(256, 256, bias=False),
        )
        self.forw4 = nn.Sequential(
            PostRes(256, 512, stride=2, bias=False),
            PostRes(512, 512, bias=False),
            PostRes(512, 512, bias=False),
        )
        # self.avgpool = nn.AvgPool3d((2, 2, 2))

    def forward(self, x):
        # (N, 1, 64, 64)
        out = self.preBlock(x)                       # (N, 1, H, H) -> (N, 24, H, H)
        out1 = self.forw1(out)                  # (N, 24, H/2, H/2) -> (N, 32, H/2, H/2)
        out2 = self.forw2(out1)                 # (N, 32, H/4, H/4) -> (N, 64, H/4, H/4)
        out3 = self.forw3(out2)                 # (N, 64, H/8, H/8) -> (N, 64, H/8, H/8)
        out4 = self.forw4(out3)                 # (N, 64, H/16, H/16) -> (N, 64, H/16, H/16)
        # (N, 64, 4, 4)
        return out4


class ResNet3D34(nn.Module):
    def __init__(self):
        super(ResNet3D34, self).__init__()

        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.forw1 = nn.Sequential(
            PostRes(64, 64, bias=False),
            PostRes(64, 64, bias=False),
            PostRes(64, 64, bias=False),
        )
        self.forw2 = nn.Sequential(
            PostRes(64, 128, stride=2, bias=False),
            PostRes(128, 128, bias=False),
            PostRes(128, 128, bias=False),
            PostRes(128, 128, bias=False),
        )
        self.forw3 = nn.Sequential(
            PostRes(128, 256, stride=2, bias=False),
            PostRes(256, 256, bias=False),
            PostRes(256, 256, bias=False),
            PostRes(256, 256, bias=False),
            PostRes(256, 256, bias=False),
            PostRes(256, 256, bias=False),
        )
        self.forw4 = nn.Sequential(
            PostRes(256, 512, stride=2, bias=False),
            PostRes(512, 512, bias=False),
            PostRes(512, 512, bias=False),
        )
        self.avgpool = nn.AvgPool3d((2, 2, 2))
        self.fc = nn.Linear(512, 400)

    def forward(self, x):
        # (N, 1, 64, 64)
        out = self.preBlock(x)                       # (N, 1, H, H) -> (N, 24, H, H)
        out1 = self.forw1(out)                  # (N, 24, H/2, H/2) -> (N, 32, H/2, H/2)
        out2 = self.forw2(out1)                 # (N, 32, H/4, H/4) -> (N, 64, H/4, H/4)
        out3 = self.forw3(out2)                 # (N, 64, H/8, H/8) -> (N, 64, H/8, H/8)
        out4 = self.forw4(out3)                 # (N, 64, H/16, H/16) -> (N, 64, H/16, H/16)
        # (N, 64, 4, 4)
        out4 = self.avgpool(out4)
        # print(out4.shape)
        out4 = out4.view(out4.size(0), -1)
        # print(out4.shape)
        out4 = self.fc(out4)
        return out4


class ResNet3D34Diameter(nn.Module):
    def __init__(self):
        super(ResNet3D34Diameter, self).__init__()

        self.diameterBlock = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool3d((2, 1, 1)),
        )

        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.forw1 = nn.Sequential(
            PostRes(64, 64, bias=False),
            PostRes(64, 64, bias=False),
            PostRes(64, 64, bias=False),
        )
        self.forw2 = nn.Sequential(
            PostRes(64, 128, stride=2, bias=False),
            PostRes(128, 128, bias=False),
            PostRes(128, 128, bias=False),
            PostRes(128, 128, bias=False),
        )
        self.forw3 = nn.Sequential(
            PostRes(128, 256, stride=2, bias=False),
            PostRes(256, 256, bias=False),
            PostRes(256, 256, bias=False),
            PostRes(256, 256, bias=False),
            PostRes(256, 256, bias=False),
            PostRes(256, 256, bias=False),
        )
        self.forw4 = nn.Sequential(
            PostRes(256, 512, stride=2, bias=False),
            PostRes(512, 512, bias=False),
            PostRes(512, 512, bias=False),
        )
        # self.avgpool = nn.AvgPool3d((2, 2, 2))

    def forward(self, x):
        # (N, 1, 64, 64)
        out = self.preBlock(x)                       # (N, 1, H, H) -> (N, 24, H, H)
        out1 = self.forw1(out)                  # (N, 24, H/2, H/2) -> (N, 32, H/2, H/2)
        out2 = self.forw2(out1)                 # (N, 32, H/4, H/4) -> (N, 64, H/4, H/4)
        out3 = self.forw3(out2)                 # (N, 64, H/8, H/8) -> (N, 64, H/8, H/8)
        out4 = self.forw4(out3)                 # (N, 64, H/16, H/16) -> (N, 64, H/16, H/16)
        # (N, 64, 4, 4)
        return out4, self.diameterBlock(out)


# res3d18 = ResNet3D18()
# print(res3d18)


class ResNet3D50(nn.Module):
    def __init__(self):
        super(ResNet3D50, self).__init__()

        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.forw1 = nn.Sequential(
            PostResBottleneck([64, 64, 256], stride=1, bias=False),
            PostResBottleneck([256, 64, 256], stride=1, bias=False),
            PostResBottleneck([256, 64, 256], stride=1, bias=False),
        )
        self.forw2 = nn.Sequential(
            PostResBottleneck([256, 128, 512], stride=2, bias=False),
            PostResBottleneck([512, 128, 512], stride=1, bias=False),
            PostResBottleneck([512, 128, 512], stride=1, bias=False),
            PostResBottleneck([512, 128, 512], stride=1, bias=False),
        )
        self.forw3 = nn.Sequential(
            PostResBottleneck([512, 256, 1024], stride=2, bias=False),
            PostResBottleneck([1024, 256, 1024], stride=1, bias=False),
            PostResBottleneck([1024, 256, 1024], stride=1, bias=False),
            PostResBottleneck([1024, 256, 1024], stride=1, bias=False),
            PostResBottleneck([1024, 256, 1024], stride=1, bias=False),
            PostResBottleneck([1024, 256, 1024], stride=1, bias=False),
        )
        self.forw4 = nn.Sequential(
            PostResBottleneck([1024, 512, 2048], stride=2, bias=False),
            PostResBottleneck([2048, 512, 2048], stride=1, bias=False),
            PostResBottleneck([2048, 512, 2048], stride=1, bias=False),
        )
        # self.avgpool = nn.AvgPool3d((2, 2, 2))

    def forward(self, x):
        # (N, 1, 64, 64)
        out = self.preBlock(x)                       # (N, 1, H, H) -> (N, 24, H, H)
        out1 = self.forw1(out)                  # (N, 24, H/2, H/2) -> (N, 32, H/2, H/2)
        out2 = self.forw2(out1)                 # (N, 32, H/4, H/4) -> (N, 64, H/4, H/4)
        out3 = self.forw3(out2)                 # (N, 64, H/8, H/8) -> (N, 64, H/8, H/8)
        out4 = self.forw4(out3)                 # (N, 64, H/16, H/16) -> (N, 64, H/16, H/16)
        # (N, 64, 4, 4)
        return out4


# f = ResNet3D50()
# x = torch.rand(2,1,32,32,32)
# print(f(x).shape)

# import torchvision
# resnet = torchvision.models.resnet50(False)
# t = transforms.ToTensor()


'''Dual Path Networks in PyTorch.'''

debug = True #True


class DPNBottleneck(nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer):
        super(DPNBottleneck, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth
        self.last_planes = last_planes
        self.in_planes = in_planes

        self.conv1 = nn.Conv3d(last_planes, in_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.conv2 = nn.Conv3d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm3d(in_planes)
        self.conv3 = nn.Conv3d(in_planes, out_planes+dense_depth, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_planes+dense_depth)

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv3d(last_planes, out_planes+dense_depth, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_planes+dense_depth)
            )

    def forward(self, x):
        # print 'bottleneck_0', x.size(), self.last_planes, self.in_planes, 1
        out = F.relu(self.bn1(self.conv1(x)))
        # print 'bottleneck_1', out.size(), self.in_planes, self.in_planes, 3
        out = F.relu(self.bn2(self.conv2(out)))
        # print 'bottleneck_2', out.size(), self.in_planes, self.out_planes+self.dense_depth, 1
        out = self.bn3(self.conv3(out))
        # print 'bottleneck_3', out.size()
        x = self.shortcut(x)
        d = self.out_planes
        # print 'bottleneck_4', x.size(), self.last_planes, self.out_planes+self.dense_depth, d
        out = torch.cat([x[:, :d, :, :]+out[:, :d, :, :], x[:, d:, :, :], out[:, d:, :, :]], 1)
        # print 'bottleneck_5', out.size()
        out = F.relu(out)
        return out


class DPN(nn.Module):
    def __init__(self, cfg, n_classes=5):
        super(DPN, self).__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']

        # self.in_planes = in_planes
        # self.out_planes = out_planes
        # self.num_blocks = num_blocks
        # self.dense_depth = dense_depth
        self.n_classes = n_classes

        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.last_planes = 64
        self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2)
        self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)
        self.linear = nn.Linear(out_planes[3]+(num_blocks[3]+1)*dense_depth[3], self.n_classes)  # 10)

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i,stride in enumerate(strides):
            layers.append(DPNBottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i==0))
            self.last_planes = out_planes + (i+2) * dense_depth
            # print '_make_layer', i, layers[-1].size()
        return nn.Sequential(*layers)

    def forward(self, x):
        #if debug: print('0', x.size(), 64)
        out = self.maxpool(self.bn1(self.conv1(x)))
        #out = self.bn1(self.conv1(x))
        out = F.relu(out)
        #if debug: print('1', out.size())
        out = self.layer1(out)
        #if debug: print('2', out.size())
        out = self.layer2(out)
        #if debug: print('3', out.size())
        out = self.layer3(out)
        #if debug: print('4', out.size())
        out = self.layer4(out)
        #if debug: print('5', out.size())
        out = self.avgpool(out)
        #if debug: print('6', out.size())
        out_1 = out.view(out.size(0), -1)
        #if debug: print('7', out_1.size())
        out = self.linear(out_1)
        #if debug: print('8', out.size())
        return out


def DPN26():
    cfg = {
        'in_planes': (96, 192, 384, 768),
        'out_planes': (256, 512, 1024, 2048),
        'num_blocks': (2, 2, 2, 2),
        'dense_depth': (16, 32, 24, 128)
    }
    return DPN(cfg)


def DPN92():
    cfg = {
        'in_planes': (96, 192, 384, 768),
        'out_planes': (256, 512, 1024, 2048),
        'num_blocks': (3, 4, 20, 3),
        'dense_depth': (16, 32, 24, 128)
    }
    return DPN(cfg)


if __name__ == "__main__":
    debug = True
    net = DPN92()
    x = torch.rand(2, 1, 64, 64, 64)
    y = net(x)
    print(y.shape)


