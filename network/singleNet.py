import torch
import torch.nn as nn
import timm
from collections import OrderedDict
import torchvision.models as models
import os


class ORNet1(nn.Module):
    def __init__(self, args):
        super(ORNet1, self).__init__()
        self.alpha = args.alpha
        assert self.alpha < 1, "0 <= alpha < 1"
        self.bname = args.bname
        self.base, self.fea_dim = self.get_base(args)
        num_classes = args.num_classes
        if self.bname in ['resnet18', 'resnet34', 'resnet50']:
            self.conv1 = self.base.conv1
            self.bn1 = self.base.bn1
            self.relu = self.base.relu
            self.maxpool = self.base.maxpool
            self.layer1 = self.base.layer1
            self.layer2 = self.base.layer2
            self.layer3 = self.base.layer3
            self.layer4 = self.base.layer4
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        if self.alpha == 0.0:
            if args.datasetName == 'idrid':
                self.fc = nn.Sequential(
                    nn.Linear(self.fea_dim, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, num_classes))
            else:
                self.fc = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(self.fea_dim, num_classes))

        else:
            if args.datasetName == 'idrid':
                self.fc = nn.Sequential(
                    nn.Linear(self.fea_dim, 512),
                    nn.Dropout(),
                    nn.ReLU(True),
                    nn.Linear(512, num_classes))
                # self.reg = nn.Sequential(
                #     nn.Linear(num_classes, num_classes * 4),
                #     nn.ReLU(True),
                #     nn.Linear(num_classes*4, 1),
                # )
                self.reg = nn.Linear(num_classes, 1)

            else:
                self.fc = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(self.fea_dim, num_classes))
                # self.reg = nn.Sequential(
                #     nn.Linear(num_classes, num_classes * 4),
                #     nn.ReLU(True),
                #     nn.Linear(num_classes*4, 1),
                # )
                self.reg = nn.Linear(num_classes, 1)


    def forward(self, x):
        if self.bname in ['resnet18', 'resnet34', 'resnet50']:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        
        elif self.bname == 'inception_v3':
            x = self.base(x)
            x = x.view(x.size(0), -1)

        if self.alpha == 0.0:
            x = self.fc(x)
            return x, None

        xc = self.fc(x)
        xr = self.reg(xc)
        return xc, xr


    def get_base(self, args):
        cfg = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'inception_v3': 2048
        }
        bname = args.bname
        if bname == 'resnet18':
            base = models.resnet18(pretrained=True)
    
        elif bname == 'resnet34':
            base = models.resnet34(pretrained=True)

        elif bname == 'resnet50':
            base = models.resnet50(pretrained=True)
            base.fc = nn.Sequential(
                    nn.Linear(2048, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 5))
            n = 4 if args.datasetName == 'idrid' else 2
            path = './network/preweight/r50_pretrainOnKgdr_{}.pkl'.format(224 * n)
            if os.path.exists(path):
                state_dict = torch.load(path)
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict[k[7:]] = v
                base.load_state_dict(new_state_dict)
            else:
                print('do not load pretrained model, please download.')
            
        elif bname == 'inception_v3':
            base = timm.create_model(bname, pretrained=True, num_classes=0)
        return base, cfg[bname]

