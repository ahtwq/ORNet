import os
import numpy as np
from network import ORNet2, CANetResNet50
from datautil import get_imgdatalaoder
from utils import qwk, JointFusionLoss, _score2cls, labelsmoothingLoss, num_consistent
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
parser.add_argument('--datasetName', type=str, default='idrid', help='dataset')
parser.add_argument('--task', type=str, default='joint', help='single or joint task')
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
parser.add_argument('--canet_lambda', type=float, default=0.25, help='lambda coef. for CANet')
parser.add_argument('--num_classes1', type=int, default=5, help='number of classes for DR')
parser.add_argument('--num_classes2', type=int, default=3, help='number of classes for DME')
parser.add_argument('--cuts1', nargs='+', type=float, default=None, help='1th threshold')
parser.add_argument('--cuts2', nargs='+', type=float, default=None, help='2nd threshold')
parser.add_argument('--reglossfn', type=str, default=None, help='loss function')
parser.add_argument('--opt', type=str, default='sgd', help='optimizer')
# for datalaoder
parser.add_argument('--img_root_train', type=str, default='./Data/idrid/train_resize1024', help='img_root')
parser.add_argument('--img_root_test', type=str, default='./Data/idrid/test_resize1024', help='img_root')
parser.add_argument('--train_txtpath', type=str, default='ground/idrid/train.txt', help='train path')
parser.add_argument('--valid_txtpath', type=str, default='ground/idrid/valid.txt', help='valid path')
parser.add_argument('--test_txtpath', type=str, default='ground/idrid/test.txt', help='test path')
args = parser.parse_args()
if args.cuts1 == None:
    args.cuts1 = list(np.arange(args.num_classes1-1) + 0.5)
if args.cuts2 == None:
    args.cuts2 = list(np.arange(args.num_classes2-1) + 0.5)
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
        assert args.num_classes1 == 5, 'num_classes1 != 5'
        assert args.num_classes2 == 3, 'num_classes2 != 3'
        assert len(args.cuts1) == 5 - 1, 'cut1 not match num_classes'
        assert len(args.cuts2) == 3 - 1, 'cut1 not match num_classes'
    elif args.datasetName == 'messidor':
        assert args.num_classes1 == 4, 'num_classes1 != 4'
        assert args.num_classes2 == 3, 'num_classes2 != 3'
        assert len(args.cuts1) == 4 - 1, 'cut1 not match num_classes'
        assert len(args.cuts2) == 3 - 1, 'cut1 not match num_classes'
check_args(args)
print(args)


# train model
def train(model, loss_func, optimizer, scheduler, num_epochs):
    # import csv
    # csvfile = open("loss_epoch.csv", "w+")
    # writer = csv.writer(csvfile)
    dic = {}
    best_epoch = 0
    best_val_acc = 0
    best_tra_acc = 0
    threshold = 0.70 if args.datasetName == 'idrid' else 0.80
    n_samples_train = len(loaders['train'].dataset)
    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        train_running_loss = 0.0
        train_running_corrects = 0
        curlr = optimizer.param_groups[0]['lr']
        for data in loaders['train']:
            inputs, labels1, rlabels1, labels2, rlabels2 = data
            inputs = Variable(inputs).to(device)
            labels1 = Variable(labels1).to(device)
            rlabels1 = Variable(rlabels1).to(device)
            labels2 = Variable(labels2).to(device)
            rlabels2 = Variable(rlabels2).to(device)
            outputs = model(inputs)
            if args.bname == 'canet':
                loss1 = loss_func(outputs[0], labels1) + args.canet_lambda * loss_func(outputs[1], labels1)
                loss2 = loss_func(outputs[2], labels2) + args.canet_lambda * loss_func(outputs[3], labels2)
                loss = loss1 + loss2
            else:
                loss = loss_func(outputs, labels1, rlabels1, labels2, rlabels2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds1 = torch.max(outputs[0].detach(), 1)[1]
            preds2 = torch.max(outputs[2].detach(), 1)[1]
            train_running_loss += loss * inputs.size(0)
            train_running_corrects += num_consistent(preds1, preds2, labels1.detach(), labels2.detach())
        train_loss = train_running_loss.item() / n_samples_train
        train_acc = train_running_corrects / n_samples_train

        # writer.writerow([avg, maxval, minval])
        # print training info every 5 epoches
        if epoch % 5 == 0 or epoch == 0 or epoch + 1 == num_epochs:
            train_info = 'epoch:[{:<3d}/{}], lr:{:.6f}, train loss:{:.4f}, acc:{:.2%}'.format(epoch, num_epochs, curlr, train_loss, train_acc)
            print(train_info, end='| ')
        scheduler.step()

        ## eval model on validation set
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_running_corrects1 = 0
        val_running_corrects2 = 0
        n_samples_val = len(loaders['valid'].dataset)
        with torch.no_grad():
            for data in loaders['valid']:
                inputs, labels1, rlabels1, labels2, rlabels2 = data
                inputs = Variable(inputs).to(device)
                labels1 = Variable(labels1).to(device)
                rlabels1 = Variable(rlabels1).to(device)
                labels2 = Variable(labels2).to(device)
                rlabels2 = Variable(rlabels2).to(device)
                outputs = model(inputs)
                if args.bname == 'canet':
                    loss1 = loss_func(outputs[0], labels1) + args.canet_lambda * loss_func(outputs[1], labels1)
                    loss2 = loss_func(outputs[2], labels2) + args.canet_lambda * loss_func(outputs[3], labels2)
                    loss = loss1 + loss2
                else:
                    loss = loss_func(outputs, labels1, rlabels1, labels2, rlabels2)
                preds1 = torch.max(outputs[0].detach(), 1)[1]
                preds2 = torch.max(outputs[2].detach(), 1)[1]
                # 4class 转化为 2class
                if args.datasetName == 'messidor':
                    preds1 = torch.clamp(preds1 - 1, 0, 1)
                    labels1 = torch.clamp(labels1 - 1, 0, 1)
                val_running_loss += loss * inputs.size(0)
                val_running_corrects += num_consistent(preds1, preds2, labels1.detach(), labels2.detach())
                val_running_corrects1 += torch.sum(preds1 == labels1)
                val_running_corrects2 += torch.sum(preds2 == labels2)

        valid_loss = val_running_loss.item() / n_samples_val
        valid_acc = val_running_corrects / n_samples_val
        valid_acc1 = val_running_corrects1.item() / n_samples_val
        valid_acc2 = val_running_corrects2.item() / n_samples_val

        end = time.time()
        epoch_time = (end - start) / 60
        if epoch % 5 == 0 or epoch == 0 or epoch + 1 == num_epochs:
            valid_info = 'valid loss:{:.4f}, acc:{:.2%}(t1:{:.2%}, t2:{:.2%}), time:{:.2f}m'.format(valid_loss, valid_acc, valid_acc1, valid_acc2, epoch_time)
            print(valid_info)

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


def eval(loaders, model, loss_func):
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    val_running_corrects1 = 0
    val_running_corrects2 = 0
    n_samples = len(loaders.dataset)
    y_true_dr = []
    y_pred_dr = []
    y_prob_dr = []
    y_true_dme = []
    y_pred_dme = []
    y_prob_dme = []
    with torch.no_grad():
        for data in loaders:
            inputs, labels1, rlabels1, labels2, rlabels2 = data
            inputs = Variable(inputs).to(device)
            labels1 = Variable(labels1).to(device)
            rlabels1 = Variable(rlabels1).to(device)
            labels2 = Variable(labels2).to(device)
            rlabels2 = Variable(rlabels2).to(device)
            outputs = model(inputs)
            if args.bname == 'canet':
                loss1 = loss_func(outputs[0], labels1) + args.canet_lambda * loss_func(outputs[1], labels1)
                loss2 = loss_func(outputs[2], labels2) + args.canet_lambda * loss_func(outputs[3], labels2)
                loss = loss1 + loss2
            else:
                loss = loss_func(outputs, labels1, rlabels1, labels2, rlabels2)
            preds1 = torch.max(outputs[0].detach(), 1)[1]
            preds2 = torch.max(outputs[2].detach(), 1)[1]
            # 4class 转化为 2class
            if args.datasetName == 'messidor':
                preds1 = torch.clamp(preds1 - 1, 0, 1)
                labels1 = torch.clamp(labels1 - 1, 0, 1)

            val_running_loss += loss * inputs.size(0)
            val_running_corrects += num_consistent(preds1, preds2, labels1.detach(), labels2.detach())
            val_running_corrects1 += torch.sum(preds1 == labels1)
            val_running_corrects2 += torch.sum(preds2 == labels2)

            if args.datasetName == 'messidor':
                y_true_dr += list(labels1.cpu().detach().numpy())
                y_pred_dr += list(preds1.cpu().detach().numpy())
                out1 = torch.softmax(outputs[0], 1)
                out1 = out1.cpu().detach().numpy()
                y_prob_dr.append(out1)

                y_true_dme += list(labels2.cpu().detach().numpy())
                y_pred_dme += list(preds2.cpu().detach().numpy())
                out2 = torch.softmax(outputs[2], 1)
                out2 = out2.cpu().detach().numpy()
                y_prob_dme.append(out2)

    valid_loss = val_running_loss.item() / n_samples
    valid_acc = val_running_corrects / n_samples
    valid_acc1 = val_running_corrects1.item() / n_samples
    valid_acc2 = val_running_corrects2.item() / n_samples
    print('test consistent acc:{:.2%}, loss:{:.6f}'.format(valid_acc, valid_loss))
    print('test dr  acc:{:.2%}'.format(valid_acc1))
    print('test dme acc:{:.2%}'.format(valid_acc2))

    # 4class to 2class
    if args.datasetName == 'messidor':
        y_prob_dr = np.concatenate(y_prob_dr, axis=0)
        y_prob_dr_1 = np.sum(y_prob_dr[:,0:2], axis=1, keepdims=True)
        y_prob_dr_2 = np.sum(y_prob_dr[:,2: ], axis=1, keepdims=True)
        y_prob_dr = np.concatenate([y_prob_dr_1, y_prob_dr_2], axis=1)
        y_prob_dme = np.concatenate(y_prob_dme, axis=0)

        f1 = metrics.f1_score(y_true_dr, y_pred_dr)
        p = metrics.precision_score(y_true_dr, y_pred_dr)
        r = metrics.recall_score(y_true_dr, y_pred_dr)
        auc = metrics.roc_auc_score(y_true_dr, y_prob_dr[:,1])
        print('DR: auc:{:.4f}, ac:{:.4f}, Pre:{:.4f}, Re:{:.4f}, F1:{:.4f}'.format(auc, valid_acc, p, r, f1))

        f1 = metrics.f1_score(y_true_dme, y_pred_dme, average='macro')
        p = metrics.precision_score(y_true_dme, y_pred_dme, average='macro')
        r = metrics.recall_score(y_true_dme, y_pred_dme, average='macro')
        auc = metrics.roc_auc_score(y_true_dme, y_prob_dme, average='macro', multi_class='ovr')
        print('DME: auc:{:.4f}, ac:{:.4f}, Pre:{:.4f}, Re:{:.4f}, F1:{:.4f}'.format(auc, valid_acc2, p, r, f1))


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
    if args.bname == 'canet':
        model = CANetResNet50()
    else:
        model = ORNet2(args)
    model.to(device)

    loaders = get_imgdatalaoder(args)

    # loss func
    ce = nn.CrossEntropyLoss().to(device)
    lsce = labelsmoothingLoss(smoothing=0.1).to(device)
    if args.bname == 'canet':
        criterion = lsce if args.lsce else ce
    else:
        criterion = JointFusionLoss(lsce, rce, args.alpha) if args.lsce else JointFusionLoss(ce, rce, args.alpha)
    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=args.nesterov)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    print(f'using optimizer: {args.opt}')
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.epochs//3, gamma=args.gamma)

    # training
    dicout = train(model, criterion, optimizer, scheduler, args.epochs)

    # Tesing
    print()
    print('Train over. Testing ......')
    print('**alpha={}'.format(args.alpha))
    for key in ['best', 'last']:
        res = dicout[key]
        print('epochs:{}, train acc:{:.2%}, valid acc:{:.2%}'.format(res[0], res[1], res[2]))
        print('-'*50)
        weight_path = os.path.join(args.dir, 'checkpoint_{}.pth'.format(str(key)))
        model.load_state_dict(torch.load(weight_path))
        eval(loaders['test'], model, criterion)
        print()
    print()
