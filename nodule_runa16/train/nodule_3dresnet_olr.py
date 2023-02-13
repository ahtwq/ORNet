import random
import os
from cv2 import threshold
import numpy as np
import torch
from scipy.ndimage.interpolation import rotate
from torch.nn.functional import fold
from torch.utils.data.dataset import Dataset
from src.ResNet import *
from src.models import *
from src.train_olr import *
from src import videoresnet
import argparse


print(os.path.abspath(__file__))
seed = 100
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

##########################################
root_path = '../ten_fold_new_NotExtend_balance_S0_S1_S2_S4_S5_ver0'
parser = argparse.ArgumentParser()
parser.add_argument('--DataRoot', type=str, default=root_path, help='the root of data set')
parser.add_argument('--test_fold', type=int, default=0, help='the data set for val')
parser.add_argument('--batchSize_train', type=int, default=32, help='the number of neighbourhood to consider')
parser.add_argument('--batchSize_val', type=int, default=1, help='the number of neighbourhood to consider')
parser.add_argument('--batchSize_test', type=int, default=1, help='the number of neighbourhood to consider')
parser.add_argument('--checkpoints', type=str, default='./checkpoints_NotExtend_Balance_S_012_45', help='folder to output model checkpoints')
parser.add_argument('--checkpoints_weight', type=str, default='', help='folder to load model weight')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--nEpochs', type=int, default=180, help='number of epochs to train for')
parser.add_argument('--LR', type=float, default=0.0008, help='learning rate')
parser.add_argument('--lambda', type=float, default=0, help='L1 regularity')
parser.add_argument('--momentum', type=float, default=0.95, help='momentum')
parser.add_argument('--wd', type=float, default=1e-4, help='wd')
parser.add_argument('--num_classes', type=int, default=5, help='number of classes')
parser.add_argument('--cuts', nargs='+', type=float, default=None)
parser.add_argument('--regfn', type=str, default='sl1', help='regression function')
parser.add_argument('--threshold', type=str, default='fix', choices=['learnable', 'fix'])

listdir = lambda f: sorted(os.listdir(f))

class ReadDataSetFromNpy2Tensor(Dataset):
    def __init__(self, data_list, mu, sig, training, num_classes):
        super(ReadDataSetFromNpy2Tensor, self).__init__()
        self.data_list = data_list  # ./subset%d/patient_id
        self.mu = mu
        self.sig = sig
        self.training = training
        self.num_classes = num_classes

    def augment(self, npy_data):
        CROP_SIZE = 64
        angle = [90, 180, 270]

        if random.random() < 0.5:
            alpha = random.randint(0, 2)
            npy_data = rotate(npy_data, angle=angle[alpha], axes=(1, 2), reshape=False)

        if random.random() < 0.5:
            alpha = random.randint(0, 2)
            npy_data = rotate(npy_data, angle=angle[alpha], axes=(-1, 0), reshape=False)

        if random.random() < 0.5:
            alpha = random.randint(0, 2)
            npy_data = rotate(npy_data, angle=angle[alpha], axes=(-1, 1), reshape=False)

        if random.random() < 0.5:
            axisorder = np.random.permutation(3)
            npy_data = np.transpose(npy_data, axisorder)

        if random.random() < 0.5:
            npy_data = npy_data[:, :, ::-1]

        if random.random() < 0.5:
            npy_data = npy_data[::-1, :, :]

        if random.random() < 0.5:
            npy_data = npy_data[:, ::-1, :]

        # feat = npy_data[28:28 + 16, 28:28 + 16, 28:28 + 16]
        x_crop = random.randint(0, 7)
        y_crop = random.randint(0, 7)
        z_crop = random.randint(0, 7)
        x_crop = int(x_crop)
        y_crop = int(y_crop)
        z_crop = int(z_crop)

        npy_data = npy_data[x_crop:x_crop + CROP_SIZE, y_crop:y_crop + CROP_SIZE, z_crop:z_crop + CROP_SIZE]
        # return npy_data, feat
        return npy_data

    def crop(self, npy_data):
        # feat = npy_data[28:28 + 16, 28:28 + 16, 28:28 + 16]
        CROP_SIZE = 64
        x_crop = y_crop = z_crop = 4
        npy_data = npy_data[x_crop:x_crop + CROP_SIZE, y_crop:y_crop + CROP_SIZE, z_crop:z_crop + CROP_SIZE]
        # return npy_data, feat
        return npy_data

    def label_tf(self, label):
        if self.num_classes == 5:
            if label > 3:
                label = label - 1
        elif self.num_classes == 2:
            if label < 3:
                label = 0
            elif label > 3:
                label = 1
        elif self.num_classes == 3:
            if 0 < label < 3:
                label = 1
            elif label > 3:
                label = 2
        return label
    
    def __getitem__(self, index):

        npy_name = self.data_list[index]
        npy_data = np.load(npy_name).astype(np.float32)

        if self.training:
            npy_data = self.augment(npy_data)
        else:
            npy_data = self.crop(npy_data)

        npy_data = (npy_data.astype(np.float32) - self.mu) / self.sig
        tensor_data = torch.from_numpy(npy_data).view(1, 64, 64, 64)

        try:
            label = int(npy_name[-11])
        except ValueError:
            label = 0
        real_label = label
        label = self.label_tf(label)

        # feat = np.hstack((np.reshape(feat, (-1,)) / 255, float(diameter)))
        # feat = np.reshape(feat, (-1,)) / 255
        # feat_tensor = torch.from_numpy(feat)

        return tensor_data, label, real_label

    def __len__(self):
        return len(self.data_list)


def cls2score(cuts, num_classes):
    assert len(cuts) == num_classes - 1
    flag = False
    if cuts == [0.5,1.5,3,4.5]:
        flag = True
    if not isinstance(cuts, torch.Tensor):
        cuts = torch.FloatTensor(cuts).cuda()
    dist = (max(cuts) - min(cuts)) / (len(cuts) - 1)
    score = torch.cat([cuts[0:1] - dist, cuts, cuts[-1:] + dist], dim=0)
    score = 0.5 * (score[1:] + score[:-1])
    score = score.view(-1,1)
    if flag:
        score = torch.round(score)
    return score

def get_interval_score(cuts, num_classes):
    assert len(cuts) == num_classes - 1
    n = len(cuts)
    dist = (max(cuts) - min(cuts)) / (len(cuts) - 1)
    intervals = [(cuts[0]-dist, cuts[0])] + [(cuts[i], cuts[i+1]) for i in range(n - 1)] + [(cuts[-1], cuts[-1]+dist)]
    scores = cls2score(cuts, num_classes).view(-1).tolist()
    assert len(scores) == num_classes
    print('cuts:', cuts)
    print('cls2score:', dict(zip(range(num_classes), scores)))
    print('interval:', intervals)
# if __name__ == '__main__':


print('='*50)
opt = parser.parse_args()
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    opt.cuda = True

opts = {k: v for k, v in opt._get_kwargs()}

if opts["cuts"] == None:
    opts["cuts"] = [0.5, 1.5, 2.5, 3.5]
get_interval_score(opts["cuts"], 5)
print(opts)

opts['checkpoints'] = opts['checkpoints'] + '/Res18_LR_%s_LAMBDA_%s_test_%d_ver%s' % (str(opts['LR']).replace('.', '_'),
                                                                                    str(opts['lambda']).replace('.', '_'),
                                                                                    opts['test_fold'],
                                                                                    opts['DataRoot'][-1])
try:
    os.makedirs(opts['checkpoints'])
except OSError:
    pass

root_list = listdir(opts['DataRoot'])
root_list = [i for i in root_list if 'subset' in i]
print('===> train data preparing:')
num_state0 = [0] * 6
mu, sig = 94.58499300498612, 80.25472376202508
test_order = opts['test_fold']
val_order = test_order + 1 if test_order < 9 else 0
subset_for_train = [subset for subset in root_list if int(subset[-1]) not in (test_order, val_order)]
train_nodule_list = []
for subset in subset_for_train:
    nodule_subset = listdir(opts['DataRoot'] + '/' + subset)
    for nodule in nodule_subset:
        num_state0[int(nodule[-11])] += 1
        train_nodule_list.append(opts['DataRoot'] + '/' + subset + '/' + nodule)
print(subset_for_train, num_state0)

print('===> train data balance:')
class_num = [0] * opts['num_classes']
if opts['num_classes'] == 2:
    item_dict = {0: [0, 1, 2], 1: [4, 5]}
elif opts['num_classes'] == 3:
    item_dict = {0: [0], 1: [1, 2], 2: [4, 5]}
else:
    item_dict = {0: [0], 1: [1], 2: [2], 3: [4], 4: [5]}

# class_list = [[sample1, sample2, ...,], [], ...., []]
class_list = [[data for data in train_nodule_list if int(data[-11]) in item_dict[label]]
                for label in range(opts['num_classes'])]
for i in range(opts['num_classes']):
    class_num[i] = len(class_list[i])
print('initial class distribution:', num_state0)

# augment (copy) sample
expand_list = []
maxx = np.max(class_num)
max_id = np.argmax(class_num)
for label in range(opts['num_classes']):
    class_label = class_list[label]
    expand_list.extend(class_label)
    if label == max_id:
        continue
    lable_num = len(class_label)
    rate = maxx // lable_num
    res = maxx % lable_num
    # print(maxx, lable_num, rate, res, len(class_label))
    for it in class_label:
        expand_list.extend([it] * int(rate-1))
    samples = random.sample(class_label, res)
    expand_list.extend(samples)

print('shuffle training samples')
random.shuffle(expand_list)
class_state = [0] * 6
for it in expand_list:
    class_state[int(it[-11])] += 1
print('final class distribution:', class_state)

print('\n===> valid data preparing:')
subset_for_val = [subset for subset in root_list if int(subset[-1]) == val_order]
print('subset_for_val', subset_for_val)
opts['val_set'] = subset_for_val[0]
val_nodule_list = []
class_state = [0] * 6
for subset in subset_for_val:
    nodule_subset = listdir(opts['DataRoot'] + '/' + subset)
    for nodule in nodule_subset:
        class_state[int(nodule[-11])] += 1
        val_nodule_list.append(opts['DataRoot'] + '/' + subset + '/' + nodule)
print('inital class distribution:', class_state)

print('\n===> test data preparing:')
subset_for_test = [subset for subset in root_list if int(subset[-1]) == test_order]
print('subset_for_test', subset_for_test)
test_nodule_list = []
class_state = [0] * 6
for subset in subset_for_test:
    nodule_subset = listdir(opts['DataRoot'] + '/' + subset)
    for nodule in nodule_subset:
        class_state[int(nodule[-11])] += 1
        test_nodule_list.append(opts['DataRoot'] + '/' + subset + '/' + nodule)
print('inital class distribution:', class_state)


print(f'train_num:{len(train_nodule_list)}/{len(expand_list)}, val_num:{len(val_nodule_list)}, test_num:{len(test_nodule_list)}')
train_set = ReadDataSetFromNpy2Tensor(expand_list, mu, sig, True, opts['num_classes'])
train_loader = torch.utils.data.DataLoader(dataset=train_set, num_workers=int(opt.workers),
                                            batch_size=opts['batchSize_train'], shuffle=True)
val_set = ReadDataSetFromNpy2Tensor(val_nodule_list, mu, sig, False, opts['num_classes'])
val_loader = torch.utils.data.DataLoader(dataset=val_set, num_workers=int(opt.workers),
                                            batch_size=opts['batchSize_val'], shuffle=False)

FP_set = ReadDataSetFromNpy2Tensor(test_nodule_list, mu, sig, False, opts['num_classes'])
FP_loader = torch.utils.data.DataLoader(dataset=FP_set, num_workers=int(opt.workers),
                                            batch_size=opts['batchSize_test'], shuffle=False)


print('===> model preparing')
r3d = videoresnet.r3d_18(pretrained=True)
basicstem = nn.Sequential(
    nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
    nn.BatchNorm3d(64),
    nn.ReLU(inplace=True),
    nn.MaxPool3d(kernel_size=2, stride=2)
)
r3d.stem = basicstem
r3d.fc = nn.Linear(512, 1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ResNet3DNodule64x64OLR(nn.Module):
    def __init__(self, cnn, n_classes=5):
        super(ResNet3DNodule64x64OLR, self).__init__()
        self.cnn = cnn
        self.cnn.fc =  nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 1)
        )

        if opts["threshold"] == 'fix':
            cutpoints = opts["cuts"]
            self.cutpoints = torch.FloatTensor(cutpoints).to(device)
            print(f'fixed init cutpoints: {self.cutpoints}')
        elif opts["threshold"] == 'learnable':
            cutpoints = opts["cuts"]
            self.cutpoints = torch.FloatTensor(cutpoints).to(device)
            self.cutpoints = nn.Parameter(self.cutpoints)
            print(f'ordered cutpoints is learnable!! Init cutpoints: {cutpoints}')
        else:
            raise ValueError('cutpoints is not valid.')

    def forward(self, x):
        x = self.cnn(x)
        # transfored class prob based on score using LogisticCumulativeLink
        sigmoids = torch.sigmoid(self.cutpoints - x)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat((
                sigmoids[:, [0]],
                link_mat,
                (1 - sigmoids[:, [-1]])
            ),
            dim=1
        )
        return link_mat


nodule_net = ResNet3DNodule64x64OLR(r3d)
# cutpoints = torch.FloatTensor(opt.cuts).to(device)
# opt.cuts = cutpoints
trainer = Trainer(nodule_net, train_loader, val_loader, FP_loader, opts['num_classes'], opts)
print('===> start training ......')
# torch.cuda.empty_cache()
trainer.run()
# trainer.test_model()
