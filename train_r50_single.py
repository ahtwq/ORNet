import os
import numpy as np
from network import ORNet1
from datautil import get_imgdatalaoder
from utils import qwk, FusionLoss, _score2cls, labelsmoothingLoss
import torch
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from functools import partial
import torch.nn as nn
import random
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import time
import argparse

np.set_printoptions(suppress=True)
print(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='lab', required=False, help='training directory')
parser.add_argument('--bname', type=str, default='resnet50', help='selecting backbone')
parser.add_argument('--task', type=str, default='single', help='single or joint task')
parser.add_argument('--datasetName', type=str, default='idrid', help='datsset name')
parser.add_argument('--disease', type=str, default='dr', help='disease name')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--epochs', type=int, default=180, help='epochs of training')
parser.add_argument('--lr', type=float, default=5e-4, help='init lr')
parser.add_argument('--momentum', type=float, default=0.95, help='momentum')
parser.add_argument('--gamma', type=float, default=0.5, help='lr decay')
parser.add_argument('--wd', type=float, default=1e-4, help='init wd')
parser.add_argument("--nesterov", action="store_true", help='whether use nesterov')
parser.add_argument("--lsce", action="store_true", help='whether use label smoothing loss')
parser.add_argument('--seed', type=int, default=100, metavar='S', help='random seed')
parser.add_argument('--alpha', type=float, default=0.0, help='weight for ordial regularation')
parser.add_argument('--num_classes', type=int, default=5, help='number of classes')
parser.add_argument('--cuts', nargs='+', type=float, default=None, help='threshold')
parser.add_argument('--reglossfn', type=str, default=None, help='loss function')
parser.add_argument('--opt', type=str, default='sgd', help='optimizer')
# for datalaoder
parser.add_argument('--img_root_train', type=str, default='./Data/idrid/train_resize1024', help='img_root')
parser.add_argument('--img_root_test', type=str, default='./Data/idrid/test_resize1024', help='img_root')
parser.add_argument('--train_txtpath', type=str, default='ground/idrid/train.txt', help='train path')
parser.add_argument('--valid_txtpath', type=str, default='ground/idrid/valid.txt', help='valid path')
parser.add_argument('--test_txtpath', type=str, default='ground/idrid/test.txt', help='test path')
args = parser.parse_args()
if args.cuts == None:
    args.cuts = list(np.arange(args.num_classes-1) + 0.5)
if args.reglossfn is None:
    rce = nn.SmoothL1Loss()
    args.reglossfn = 'SL1'
elif args.reglossfn == 'L1':
    rce = nn.L1Loss()
elif args.reglossfn == 'L2':
    rce = nn.MSELoss()
args.dir = 'outputs/' + args.dir
def check_args(args):
    if args.datasetName == 'idrid':
        if args.disease == 'dr':
            assert args.num_classes == 5, 'num_classes != 5'
        elif args.disease == 'dme':
            assert args.num_classes == 3, 'num_classes != 3'
        
    elif args.datasetName == 'messidor':
        if args.disease == 'dr':
            assert args.num_classes == 4, 'num_classes != 4'
        elif args.disease == 'dme':
            assert args.num_classes == 3, 'num_classes != 3'
    assert len(args.cuts) == args.num_classes - 1, 'cut not match num_classes'
check_args(args)
print(args)


# train model
def train(model, loss_func, optimizer, scheduler, num_classes, num_epochs):
    # import csv
    # csvfile = open("loss_epoch.csv", "w+")
    # writer = csv.writer(csvfile)
    dic = {}
    best_epoch = 0
    best_val_acc = 0
    best_tra_acc = 0
    threshold = 0.80
    score2cls = partial(_score2cls, cuts=args.cuts, num_classes=args.num_classes)
    n_samples_train = len(loaders['train'].dataset)
    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        train_running_loss = 0.0
        train_running_corrects = 0
        curlr = optimizer.param_groups[0]['lr']
        for data in loaders['train']:
            inputs, labels, rlabels = data
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
            rlabels = Variable(rlabels).to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, labels, rlabels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.max(outputs[0].detach(), 1)[1]
            train_running_loss += loss * inputs.size(0)
            train_running_corrects += torch.sum(preds == labels.detach())
        train_loss = train_running_loss.item() / n_samples_train
        train_acc = train_running_corrects.item() / n_samples_train

        # writer.writerow([avg, maxval, minval])

        scheduler.step()
        # print training info every 5 epoches
        if epoch % 5 == 0 or epoch == 0 or epoch + 1 == num_epochs:
            train_info = 'epoch:[{:<3d}/{}], lr:{:.6f}, train loss:{:.4f}, acc:{:.2%}'.format(epoch, num_epochs, curlr, train_loss, train_acc)
            print(train_info, end='| ')

        ## eval model on validation set
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_conf_matrix = np.zeros((num_classes, num_classes))
        label_list = list(np.arange(num_classes))
        if args.datasetName == 'messidor' and args.disease == 'dr':
            val_conf_matrix = np.zeros((num_classes-2, num_classes-2))
            label_list = list(np.arange(num_classes-2))
        n_samples_val = len(loaders['valid'].dataset)
        with torch.no_grad():
            for data in loaders['valid']:
                inputs, labels, rlabels = data
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)
                rlabels = Variable(rlabels).to(device)
                outputs = model(inputs)
                preds = torch.max(outputs[0].detach(), 1)[1]
                loss = loss_func(outputs, labels, rlabels)
                # 4class 转化为 2class
                if args.datasetName == 'messidor' and args.disease == 'dr':
                    preds = torch.clamp(preds - 1, 0, 1)
                    labels = torch.clamp(labels - 1, 0, 1)
                val_running_loss += loss * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels)
                val_conf_matrix += confusion_matrix(list(labels.cpu().detach().numpy()), list(preds.cpu().detach().numpy()), labels=label_list)

        valid_loss = val_running_loss.item() / n_samples_val
        valid_acc = val_running_corrects.item() / n_samples_val
        end = time.time()
        epoch_time = (end - start) / 60
        if epoch % 5 == 0 or epoch == 0 or epoch + 1 == num_epochs:
            info = 'valid loss:{:.4f}, acc:{:.2%}, time:{:.2f}m'.format(valid_loss, valid_acc, epoch_time)
            print(info)

        # log the accuracy and save the best acc
        if valid_acc >= best_val_acc and train_acc > threshold:
            best_val_acc = valid_acc
            best_tra_acc = train_acc
            best_epoch = epoch
            path = args.dir + '/checkpoint_best.pth'
            torch.save(model.state_dict(), path)

    torch.save(model.state_dict(), args.dir + '/checkpoint_last.pth')
    dic['best'] = [best_epoch, round(best_tra_acc, 6), round(best_val_acc, 6)]
    dic['last'] = [epoch, round(train_acc, 6), round(valid_acc, 6)]

    return dic


def eval(loaders, model, loss_func, num_classes):
    model.eval()
    score2cls = partial(_score2cls, cuts=args.cuts, num_classes=args.num_classes)
    val_running_loss = 0.0
    val_running_corrects = 0
    val_conf_matrix = np.zeros((num_classes, num_classes))
    label_list = list(np.arange(num_classes))
    if args.datasetName == 'messidor' and args.disease == 'dr':
        val_conf_matrix = np.zeros((num_classes-2, num_classes-2))
        label_list = list(np.arange(num_classes-2))
    n_samples = len(loaders.dataset)
    y_true = []
    y_pred = []
    y_prob = []
    with torch.no_grad():
        for data in loaders:
            inputs, labels, rlabels = data
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
            rlabels = Variable(rlabels).to(device)
            outputs = model(inputs)
            preds = torch.max(outputs[0].detach(), 1)[1]
            loss = loss_func(outputs, labels, rlabels)
            # 4class 转化为 2class
            if args.datasetName == 'messidor' and args.disease == 'dr':
                preds = torch.clamp(preds - 1, 0, 1)
                labels = torch.clamp(labels - 1, 0, 1)
            val_running_loss += loss * inputs.size(0)

            val_running_corrects += torch.sum(preds == labels)
            val_conf_matrix += confusion_matrix(list(labels.cpu().detach().numpy()), list(preds.cpu().detach().numpy()), labels=label_list)

            if args.datasetName == 'messidor':
                y_true += list(labels.cpu().detach().numpy())
                y_pred += list(preds.cpu().detach().numpy())
                out1 = torch.softmax(outputs[0], 1)
                out1 = out1.cpu().detach().numpy()
                y_prob.append(out1)

    valid_loss = val_running_loss.item() / n_samples
    valid_acc = val_running_corrects.item() / n_samples
    valid_qwk = qwk(val_conf_matrix)
    print('test {} acc:{:.2%}, qwk:{:.4f}'.format(args.disease, valid_acc, valid_qwk))
    print(val_conf_matrix)

    # 4class to 2class
    if args.datasetName == 'messidor' and args.disease == 'dr':
        y_prob = np.concatenate(y_prob, axis=0)
        y_prob_1 = np.sum(y_prob[:,0:2], axis=1, keepdims=True)
        y_prob_2 = np.sum(y_prob[:,2: ], axis=1, keepdims=True)
        y_prob = np.concatenate([y_prob_1, y_prob_2], axis=1)
        f1 = metrics.f1_score(y_true, y_pred)
        p = metrics.precision_score(y_true, y_pred)
        r = metrics.recall_score(y_true, y_pred)
        auc = metrics.roc_auc_score(y_true, y_prob[:,1])
        print('DR: auc:{:.4f}, ac:{:.4f}, Pre:{:.4f}, Re:{:.4f}, F1:{:.4f}'.format(auc, valid_acc, p, r, f1))

    elif args.datasetName == 'messidor' and args.disease == 'dme':
        y_prob = np.concatenate(y_prob, axis=0)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        p = metrics.precision_score(y_true, y_pred, average='macro')
        r = metrics.recall_score(y_true, y_pred, average='macro')
        auc = metrics.roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')
        print('DME: auc:{:.4f}, ac:{:.4f}, Pre:{:.4f}, Re:{:.4f}, F1:{:.4f}'.format(auc, valid_acc, p, r, f1))


##----------------------- Main ------------------------------##
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Preparing directory {}'.format(args.dir))
    os.makedirs(args.dir, exist_ok=True)

    # load model
    model = ORNet1(args)
    model.to(device)

    loaders = get_imgdatalaoder(args)

    # loss func
    ce = nn.CrossEntropyLoss().to(device)
    lsce = labelsmoothingLoss(smoothing=0.1).to(device)

    criterion = FusionLoss(lsce, rce, args.alpha) if args.lsce else FusionLoss(ce, rce, args.alpha)
    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=args.nesterov)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    print(f'using optimizer: {args.opt}')
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.epochs//3, gamma=args.gamma)

    # training
    dicout = train(model, criterion, optimizer, scheduler, args.num_classes, args.epochs)
    keys = ['best', 'last'] if dicout['best'][0] > 0 else ['last']

    # Tesing
    print()
    if len(keys) == 1:
        print('network accuray on the train dataset is low, strongly recommend retraining the network!!!')
    print('Train over. Testing ......')
    print('**alpha={}'.format(args.alpha))
    for key in keys:
        res = dicout[key]
        print('epochs:{}, train acc:{:.2%}, valid acc:{:.2%}'.format(res[0], res[1], res[2]))
        print('-'*50)
        weight_path = os.path.join(args.dir, 'checkpoint_{}.pth'.format(str(key)))
        model.load_state_dict(torch.load(weight_path))
        eval(loaders['test'], model, criterion, args.num_classes)
        print()
    print()
