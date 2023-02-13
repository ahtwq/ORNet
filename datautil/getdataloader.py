import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from datautil.imgutil import get_transform


# for single
class MyDataset1(Dataset):
    def __init__(self, args, img_root, label_file, img_transform=None):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        k = 1 if args.disease == 'dr' else 2
        if args.datasetName == 'idrid':
            self.img_list = [os.path.join(img_root, i.split(',')[0] + '.png') for i in lines]
        else:
            self.img_list = [os.path.join(img_root, i.split(',')[0][:-3] + 'png') for i in lines]
        self.label_list = [int(i.split(',')[k]) for i in lines] # DR/DME
        scoremat = self.cls2rlabel(args)
        self.rlabel_list = [float(scoremat[i].item()) for i in self.label_list]
        self.img_transform = img_transform

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path)
        label = self.label_list[index]
        rlabel = self.rlabel_list[index]
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, label, rlabel

    def cls2rlabel(self, args):
        cuts = args.cuts
        num_classes = args.num_classes
        assert len(cuts) == num_classes - 1
        if not isinstance(cuts, torch.Tensor):
            cuts = torch.FloatTensor(cuts)
        dist = (max(cuts) - min(cuts)) / (len(cuts) - 1)
        score = torch.cat([cuts[0:1] - dist, cuts, cuts[-1:] + dist], dim=0)
        score = 0.5 * (score[1:] + score[:-1])
        score = score.view(-1,1)
        return score

    def __len__(self):
        return len(self.img_list)


# for joint 
class MyDataset2(Dataset):
    def __init__(self, args, img_root, label_file, img_transform=None):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        if args.datasetName == 'idrid':
            self.img_list = [os.path.join(img_root, i.split(',')[0] + '.png') for i in lines]
        else:
            self.img_list = [os.path.join(img_root, i.split(',')[0][:-3] + 'png') for i in lines]
        self.label1_list = [int(i.split(',')[1]) for i in lines] # DR
        self.label2_list = [int(i.split(',')[2]) for i in lines] # DME
        scoremat1 = self.cls2rlabel(args, k=0)
        scoremat2 = self.cls2rlabel(args, k=1)
        self.rlabel1_list = [float(scoremat1[i].item()) for i in self.label1_list]
        self.rlabel2_list = [float(scoremat2[i].item()) for i in self.label2_list]
        self.img_transform = img_transform

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path)
        label1 = self.label1_list[index]
        rlabel1 = self.rlabel1_list[index]
        label2 = self.label2_list[index]
        rlabel2 = self.rlabel2_list[index]
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, label1, rlabel1, label2, rlabel2 

    def cls2rlabel(self, args, k=0):
        if k == 0:
            cuts = args.cuts1
            num_classes = args.num_classes1
            assert len(cuts) == num_classes - 1
        elif k == 1:
            cuts = args.cuts2
            num_classes = args.num_classes2
            assert len(cuts) == num_classes - 1
        
        if not isinstance(cuts, torch.Tensor):
            cuts = torch.FloatTensor(cuts)
        dist = (max(cuts) - min(cuts)) / (len(cuts) - 1)
        score = torch.cat([cuts[0:1] - dist, cuts, cuts[-1:] + dist], dim=0)
        score = 0.5 * (score[1:] + score[:-1])
        score = score.view(-1,1)
        return score

    def __len__(self):
        return len(self.img_list)


def get_imgdatalaoder(args):
    # img_root_train = 'Data/train_resize{}'.format(256*n)
    # img_root_test = 'Data/test_resize{}'.format(256*n)
    # train_txt = 'ground/train.txt'
    # valid_txt = 'ground/valid.txt'
    # test_txt = 'ground/test.txt'
    if args.task == 'single':
        train_transform, val_transform = get_transform(args.datasetName)
        train_set = MyDataset1(args, img_root=args.img_root_train, label_file=args.train_txtpath, img_transform=train_transform)
        valid_set = MyDataset1(args, img_root=args.img_root_train, label_file=args.valid_txtpath, img_transform=val_transform)
        test_set = MyDataset1(args, img_root=args.img_root_test, label_file=args.test_txtpath, img_transform=val_transform)

    elif args.task == 'joint':
        train_transform, val_transform = get_transform(args.datasetName)
        train_set = MyDataset2(args, img_root=args.img_root_train, label_file=args.train_txtpath, img_transform=train_transform)
        valid_set = MyDataset2(args, img_root=args.img_root_train, label_file=args.valid_txtpath, img_transform=val_transform)
        test_set = MyDataset2(args, img_root=args.img_root_test, label_file=args.test_txtpath, img_transform=val_transform)


    loaders = {
        'train': DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        ),
        'valid': DataLoader(
            valid_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
        ),
        'test': DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
        )
    }

    return loaders
