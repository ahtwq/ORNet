import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn
import random, os
def setup_seed(seed=100):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
setup_seed()

class ResNet3DNodule64x64x64TwoRegularBeta(nn.Module):
    def __init__(self, cnn, n_classes=6):
        super(ResNet3DNodule64x64x64TwoRegularBeta, self).__init__()

        self.cnn = cnn
        self.n_classes = n_classes
        self.avgpool = nn.AvgPool3d((2, 2, 1))
        self.fc1 = nn.Linear(1024, self.n_classes, bias=True)
        self.fc2 = nn.Linear(1024, 1, bias=True)
        self.fc3 = nn.Linear(1024, 1, bias=True)

    def forward(self, x):
        feat, diameter = self.cnn(x)
        x = self.avgpool(feat).view(-1, 1024)
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        y3 = self.fc3(diameter.view(-1, 1024))
        return y1, y2, y3, feat.view(-1, 4096)


class ResNet3DNodule64x64x64TwoRegularAlpha(nn.Module):
    def __init__(self, cnn, n_classes=6):
        super(ResNet3DNodule64x64x64TwoRegularAlpha, self).__init__()

        self.cnn = cnn
        self.n_classes = n_classes
        self.avgpool = nn.AvgPool3d((2, 2, 1))
        self.fc1 = nn.Linear(1024, self.n_classes, bias=True)
        self.fc2 = nn.Linear(1024, 1, bias=True)
        self.fc3 = nn.Linear(1024, 1, bias=True)
        self.fc4 = nn.Linear(1024, 1, bias=True)

    def forward(self, x):
        feat = self.cnn(x)
        x = self.avgpool(feat).view(-1, 1024)
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        y3 = self.fc3(x)
        y4 = self.fc4(x)
        return y1, y2, y3, y4, feat.view(-1, 4096)


class ResNet3DNodule64x64RegressRegular(nn.Module):
    def __init__(self, cnn, n_classes=5):
        super(ResNet3DNodule64x64RegressRegular, self).__init__()

        self.cnn = cnn
        self.n_classes = n_classes
        self.avgpool = nn.AvgPool3d((2, 2, 1))
        self.fc1 = nn.Linear(1024, self.n_classes)
        self.fc2 = nn.Linear(self.n_classes, 1)

    def forward(self, x):
        feat = self.cnn(x)
        x = self.avgpool(feat).view(-1, 1024)
        y1 = self.fc1(x)
        y2 = self.fc2(y1)
        return y1, y2, feat.view(-1, 4096)


class ResNet3DNodule64x64Fusion(nn.Module):
    def __init__(self, cnn, n_classes=6):
        super(ResNet3DNodule64x64Fusion, self).__init__()

        self.cnn = cnn
        self.n_classes = n_classes
        self.avgpool = nn.AvgPool3d((2, 2, 1))

        self.fusion = nn.Sequential(
            nn.Linear(1024 + 48, 512, bias=True),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(512, self.n_classes, bias=True)

    def forward(self, x, glcm_feature):
        x_feature = self.avgpool(self.cnn(x)).view(-1, 1024)
        x_cat_glcm = torch.cat([x_feature, glcm_feature.view(-1, 48)], dim=1)
        # print(glcm_feature.view(-1, 48)[0])
        x_fusion = self.fusion(x_cat_glcm)
        return self.fc(x_fusion)


class ResNet3DNodule64x64Regress(nn.Module):
    def __init__(self, cnn):
        super(ResNet3DNodule64x64Regress, self).__init__()

        self.cnn = cnn
        self.avgpool = nn.AvgPool3d((2, 2, 1))
        self.fc = nn.Linear(1024, 1, bias=True)

    def forward(self, x):
        x = self.avgpool(self.cnn(x))
        return self.fc(x.view(-1, 1024))


class ResNet3DNodule64x64CatThreeChannel(nn.Module):
    def __init__(self, cnn, n_classes=6):
        super(ResNet3DNodule64x64CatThreeChannel, self).__init__()

        self.cnn = cnn
        self.n_classes = n_classes
        self.avgpool = nn.AvgPool3d((2, 2, 1))
        self.fc = nn.Linear(1024, self.n_classes, bias=True)

    def forward(self, x):
        x_1 = x
        x_2 = torch.transpose(x, 2, 4)
        x_3 = torch.transpose(x, 2, 3)
        x_123 = torch.cat([x_1, x_2, x_3], dim=1)
        x_123 = self.avgpool(self.cnn(x_123))
        return self.fc(x_123.view(-1, 1024))


class ResNet3DNodule64x64CatThreeView(nn.Module):
    def __init__(self, cnn, n_classes=6):
        super(ResNet3DNodule64x64CatThreeView, self).__init__()

        self.cnn = cnn
        self.n_classes = n_classes
        self.avgpool = nn.AvgPool3d((2, 2, 3))
        self.fc = nn.Linear(1024, self.n_classes, bias=True)

    def forward(self, x):
        x_1 = x
        x_2 = torch.transpose(x, 2, 4)
        x_3 = torch.transpose(x, 2, 3)
        x_123 = torch.cat([x_1, x_2, x_3], dim=-1)
        x_123 = self.avgpool(self.cnn(x_123))
        return self.fc(x_123.view(-1, 1024))


class ResNet3DNodule64x64(nn.Module):
    def __init__(self, cnn, n_classes=6):
        super(ResNet3DNodule64x64, self).__init__()

        self.cnn = cnn
        self.n_classes = n_classes
        self.avgpool = nn.AvgPool3d((2, 2, 1))
        self.fc = nn.Linear(1024, self.n_classes, bias=True)

    def forward(self, x):
        x = self.avgpool(self.cnn(x))
        return self.fc(x.view(-1, 1024))


class Nodule3D18(nn.Module):
    def __init__(self, cnn, n_classes=6):
        super(Nodule3D18, self).__init__()

        self.cnn = cnn
        self.n_classes = n_classes
        self.fc = nn.Linear(512, self.n_classes, bias=True)

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x.view(-1, 512))


class Nodule3D50(nn.Module):
    def __init__(self, cnn, n_classes=6):
        super(Nodule3D50, self).__init__()

        self.cnn = cnn
        self.n_classes = n_classes
        self.fc = nn.Linear(2048, self.n_classes, bias=True)

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x.view(-1, 2048))


class NoduleDeepLung(nn.Module):
    def __init__(self, cnn, n_classes=6):
        super(NoduleDeepLung, self).__init__()

        self.cnn = cnn
        self.n_classes = n_classes
        # self.avgpool3d = nn.AvgPool3d((2, 2, 2))
        self.fc = nn.Linear(512, self.n_classes, bias=True)

    def forward(self, x):
        # x = self.avgpool3d(self.cnn(x))
        x = self.cnn(x)
        return self.fc(x.view(-1, 512))


class ClassifierForNodule(nn.Module):
    def __init__(self, n_classes=6):
        super(ClassifierForNodule, self).__init__()

        # self.cnn = cnn
        self.n_classes = n_classes
        self.avgpool3d = nn.AvgPool3d((2, 2, 2))
        self.fc = nn.Linear(512, self.n_classes, bias=True)

    def forward(self, x):
        # x_feature = self.cnn(x)
        x = self.avgpool3d(x)
        return self.fc(x.view(-1, 512))


class ClassifierForNodule3D18(nn.Module):
    def __init__(self, n_classes=6):
        super(ClassifierForNodule3D18, self).__init__()

        # self.cnn = cnn
        self.n_classes = n_classes
        self.fc = nn.Linear(512, self.n_classes, bias=True)

    def forward(self, x):
        return self.fc(x.view(-1, 512))
