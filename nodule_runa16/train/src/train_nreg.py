import torch
import torch.nn as nn
import random
import os
import numpy as np
import torch.backends.cudnn
import torch.cuda
from datetime import datetime
from sklearn import metrics
import time
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from src.ResNet import *
from src.models import *
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from xgboost import XGBClassifier as xgb
import pickle
from interval import Interval
from functools import partial
from copy import deepcopy


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


def _score2cls(score, cuts, num_classes):
    assert len(cuts) == num_classes - 1
    M = 1e6
    n = len(cuts)
    intervals = [Interval(-M, cuts[0], lower_closed=False)]
    intervals += [Interval(cuts[i], cuts[i+1], lower_closed=False) for i in range(n - 1)]
    intervals += [Interval(cuts[-1], M, lower_closed=False)]
    for index, item in enumerate(intervals):
        if score in item:
            return index
    return None


class Trainer:
    def __init__(self, net, train_loader, val_loader, FP_loader, num_classes, opt):
        super(Trainer, self).__init__()
        self.scoremat = cls2score(opt["cuts"], num_classes)
        self.score2cls = partial(_score2cls, cuts=opt["cuts"], num_classes=num_classes)

        self.net = net
        self.num_classes = num_classes
        self.opt = opt

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.FP_loader = FP_loader

        self.optimizer = optim.SGD(self.net.parameters(), lr=self.opt['LR'], momentum=self.opt['momentum'], weight_decay=self.opt['wd'], nesterov=True)

        self.criterion = nn.CrossEntropyLoss()
        if self.opt["regfn"] == 'sl1':
            self.regular = nn.SmoothL1Loss()
        elif self.opt["regfn"] == 'l1':
            self.regular = nn.L1Loss()
        elif self.opt["regfn"] == 'l2':
            self.regular = nn.MSELoss()
        self.softmax = nn.Softmax(dim=1)

        self.train_loss = []
        self.train_acc_net = []

        self.val_loss = []
        self.val_acc_net = []

        self.target = torch.from_numpy(np.arange(self.num_classes).astype(np.float32))

        self.register_on_gpu()
        self.set_optimizer()

        self.best_mat_net = np.zeros((self.num_classes, self.num_classes))
        self.best_acc_net = 0.0
        self.best_epoch_net = 0


    def register_on_gpu(self):
        if self.opt['checkpoints_weight'] != '':
            check = torch.load('%s/net_classifier_best_xgb.tar' % self.opt['checkpoints_weight'])
            self.net.load_state_dict(check['net_state_dict'])
            print('load model parameter.')

        if self.opt['cuda']:
            self.target = self.target.cuda()
            self.net.cuda()

    def set_optimizer(self):
        if self.opt['checkpoints_weight'] != '':
            check = torch.load('%s/net_classifier_best_xgb.tar' % self.opt['checkpoints_weight'])
            self.optimizer.load_state_dict(check['optimizer_net_state_dict'])
            print('load optimizer parameter.')

    def get_lr(self, epoch):
        if epoch < self.opt['nEpochs'] * 1 / 3:
            lr = self.opt['LR']
        elif epoch < self.opt['nEpochs'] * 2 / 3:
            lr = 0.5 * self.opt['LR']
        elif epoch < self.opt['nEpochs'] * 0.8:
            lr = 0.5 ** 2 * self.opt['LR']
        else:
            lr = 0.5 ** 2 * self.opt['LR']
        return lr


    def change_lr(self, epoch):
        lr = self.get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_epoch(self, epoch):
        if self.opt["threshold"] == 'fix':
            cuts = deepcopy(self.net.cutpoints.data).tolist()
            score2cls = partial(_score2cls, cuts=cuts, num_classes=5)
            score = cls2score(cuts=cuts, num_classes=5)
            
        mean_loss = 0.0
        mean_acc_net = 0.0
        self.net.train()
        idx = 0
        for train_data, train_label, train_rlabel in self.train_loader:
            # print(train_data[0,0,0,0,:])
            if self.opt["threshold"] == 'learnable':
                cuts = deepcopy(self.net.cutpoints.data).tolist()
                score2cls = partial(_score2cls, cuts=cuts, num_classes=5)
                # get score which can updated
                res = (max(self.net.cutpoints) - min(self.net.cutpoints)) / (len(self.net.cutpoints) - 1)
                score = torch.cat([self.net.cutpoints[0:1] - res, self.net.cutpoints, self.net.cutpoints[-1:] + res], dim=0)
                score = 0.5 * (score[1:] + score[:-1])
                score = score.view(-1, 1)

            if self.opt['cuda']:
                train_label = Variable(train_label).cuda()
                float_scores = score[train_label].type_as(train_label)
                float_scores = Variable(float_scores).cuda()
                train_data = Variable(train_data).cuda()
            score2 = self.net(train_data)
            # loss1 = self.criterion(score1, train_label)
            # _, pred_net = torch.max(score1.detach(), 1)
            # loss2 = self.regular(score2, train_rlabel.view(-1, 1).float())
            loss = self.regular(score2, float_scores.view(-1, 1).float())
            # loss = (1 - self.opt['lambda']) * loss1 + self.opt['lambda'] * loss2
            
            preds = [score2cls(s) for s in score2.view(-1).tolist()]
            preds = torch.LongTensor(preds)
            pred_net = preds.type_as(train_label).view_as(train_label)

            acc_temp1 = int(torch.sum(pred_net == train_label.data)) / self.opt['batchSize_train']
            mean_acc_net += acc_temp1


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mean_loss += loss.item()
            idx += len(train_label)

            # clip cutpoints after each batch
            if self.opt["threshold"] == "learnable":
                margin = 0.5
                min_val = -1.0e6
                cutpoints = self.net.cutpoints.data
                for i in range(cutpoints.shape[0] - 1):
                    cutpoints[i].clamp_(min_val, cutpoints[i + 1] - margin)

        return mean_loss / len(self.train_loader), mean_acc_net / len(self.train_loader)


    def val_epoch(self, epoch):
        cuts = deepcopy(self.net.cutpoints.data).tolist()
        score2cls = partial(_score2cls, cuts=cuts, num_classes=5)
        score = cls2score(cuts=cuts, num_classes=5)

        confusion_val_mat_net = np.zeros((self.num_classes, self.num_classes))
        prob_list = []
        label_list = []
        with torch.no_grad():
            mean_loss = 0.0
            mean_acc_net = 0.0

            self.net.eval()

            for val_data, val_label, val_rlabel in self.val_loader:
                if self.opt['cuda']:
                    val_label = Variable(val_label.cuda())
                    val_rlabel = Variable(val_rlabel.cuda())
                    float_scores = score[val_label].type_as(val_label)
                    float_scores = Variable(float_scores).cuda()
                    val_data = Variable(val_data.cuda())

                score2 = self.net(val_data)
                # loss1 = self.criterion(score1, val_label)
                # _, pred_net = torch.max(score1.detach(), 1)
                # loss2 = self.regular(score2, val_rlabel.view(-1, 1).float())
                preds = [score2cls(s) for s in score2.view(-1).tolist()]
                preds = torch.LongTensor(preds)
                pred_net = preds.type_as(val_label).view_as(val_label)

                loss = self.regular(score2, float_scores.view(-1, 1).float())              
                # loss = (1 - self.opt['lambda']) * loss1 + self.opt['lambda'] * loss2
                mean_loss += loss.item()
                label_tmp = list(val_label.cpu().detach().numpy())
                label_list += [0 if l < 3 else 1 for l in label_tmp]
                acc_temp1 = int(torch.sum(pred_net == val_label.data)) / self.opt['batchSize_val']
                mean_acc_net += acc_temp1
                confusion_val_mat_net += confusion_matrix(val_label.cpu().data, pred_net.cpu().data,
                                                          labels=list(range(self.num_classes)))
        auc = 0.0
        return mean_loss / len(self.val_loader), mean_acc_net / len(self.val_loader), confusion_val_mat_net, auc


    def test_model(self):
        cuts = deepcopy(self.net.cutpoints.data).tolist()
        score2cls = partial(_score2cls, cuts=cuts, num_classes=5)
        score = cls2score(cuts=cuts, num_classes=5)

        confusion_val_mat_net = np.zeros((self.num_classes, self.num_classes))
        prob_list = []
        label_list = []
        with torch.no_grad():
            mean_loss = 0.0
            mean_acc_net = 0.0

            self.net.eval()

            for val_data, val_label, val_rlabel in self.FP_loader:
                if self.opt['cuda']:
                    val_label = Variable(val_label.cuda())
                    val_rlabel = Variable(val_rlabel.cuda())
                    float_scores = score[val_label].type_as(val_label)
                    float_scores = Variable(float_scores).cuda()
                    val_data = Variable(val_data.cuda())

                score2 = self.net(val_data)
                # loss1 = self.criterion(score1, val_label)
                # _, pred_net = torch.max(score1.detach(), 1)
                # loss2 = self.regular(score2, val_rlabel.view(-1, 1).float())
                preds = [score2cls(s) for s in score2.view(-1).tolist()]
                preds = torch.LongTensor(preds)
                pred_net = preds.type_as(val_label).view_as(val_label)

                loss = self.regular(score2, float_scores.view(-1, 1).float())
                # loss = (1 - self.opt['lambda']) * loss1 + self.opt['lambda'] * loss2
                mean_loss += loss.item()
                label_tmp = list(val_label.cpu().detach().numpy())
                label_list += [0 if l < 3 else 1 for l in label_tmp]
                acc_temp1 = int(torch.sum(pred_net == val_label.data)) / self.opt['batchSize_val']
                mean_acc_net += acc_temp1
                confusion_val_mat_net += confusion_matrix(val_label.cpu().data, pred_net.cpu().data,
                                                          labels=list(range(self.num_classes)))
        auc = 0.0
        return mean_loss / len(self.FP_loader), mean_acc_net / len(self.FP_loader), confusion_val_mat_net, auc

    @staticmethod
    def merge(A, dim):  # dim = [(0,), (1,2),(3,4)]
        B = np.zeros((len(dim), A.shape[1]))
        for i in range(len(dim)):
            B[i, :] = np.sum(A[dim[i], :].reshape(len(dim[i]), -1), axis=0)
        C = np.zeros((len(dim), len(dim)))
        for i in range(len(dim)):
            C[:, i] = np.sum(B[:, dim[i]].reshape(-1, len(dim[i])), axis=1)
        return C

    @staticmethod
    def choose(A, dim):
        return A[dim, :][:, dim]

    @staticmethod
    def cal_index(A):
        acc = np.sum(np.diag(A)) / np.sum(A)
        precision = np.diag(A) / np.sum(A, 0)
        recall = np.diag(A) / np.sum(A, 1)
        f1_score = 2 / ((1 / precision[1]) + (1 / recall[1]))
        tp = A[1, 1]
        fp = A[0, 1]
        fn = A[1, 0]
        tn = A[0, 0]
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return acc, precision, recall, f1_score, sensitivity, specificity

    def analysis_model(self, best_mat, flag='net'):
        dict_analysis = dict()
        temp_mat = best_mat
        dict_analysis['cm'] = temp_mat
        dict_analysis['acc_val_raw'] = round(np.sum(np.diag(temp_mat)) / np.sum(temp_mat), 6)
        # {0, 1, 2}, {4, 5};
        if self.num_classes == 5:
            mat1 = self.merge(temp_mat, [(0, 1, 2), (3, 4)])
        elif self.num_classes == 3:
            mat1 = self.merge(temp_mat, [(0, 1), (2,)])
        else:
            mat1 = temp_mat
        acc, precision, recall, f1_score, sensitivity, specificity = self.cal_index(mat1)
        dict_analysis['acc_val_012_45'] = round(acc, 6)
        dict_analysis['prec_val_012_45'] = round(precision[1], 6)
        dict_analysis['recall_val_012_45'] = round(recall[1], 6)
        dict_analysis['f1_val_012_45'] = round(f1_score, 6)
        dict_analysis['sens_val_012_45'] = round(sensitivity, 6)
        dict_analysis['spec_val_012_45'] = round(specificity, 6)
        return dict_analysis
        # np.save('%s/analysis_best_%s.npy' % (self.opt['checkpoints'], flag), dict_analysis)


    def run(self):
        print('--# Res18 not extend (val in %s)  '
                'for %d classification with lambda=%.2f #--'
                % (self.opt['val_set'],
                    self.num_classes, self.opt['lambda']))
        for epoch in range(self.opt['nEpochs']):
            start = time.time()
            tra_loss, tra_acc = self.train_epoch(epoch)
            self.train_loss.append(tra_loss)
            self.train_acc_net.append(tra_acc)
            lr = self.optimizer.param_groups[0]['lr']
            nEpochs = self.opt['nEpochs']
            if epoch % 5 == 0 or epoch == 0 or epoch == self.opt['nEpochs'] - 1:
                print(f'epoch:[{epoch:<3d}/{nEpochs}], lr:{lr:.6f}, tra_loss:{tra_loss:.5f}, acc:{tra_acc:.2%}', end='|')
            val_loss, val_acc, val_cm, auc = self.val_epoch(epoch)
            if epoch > 10:
                dic = self.analysis_model(val_cm)
                val_acc = dic['acc_val_012_45']
                dic['auc'] = round(auc, 6)
            self.val_loss.append(val_loss)
            self.val_acc_net.append(val_acc)
            self.change_lr(epoch)

            end = time.time()
            if epoch % 5 == 0 or epoch == 0 or epoch == self.opt['nEpochs'] - 1:
                print(f'val_loss:{val_loss:.5f}, acc:{val_acc:.2%}, time:{(end-start)/60:.2f}m' )

            if self.best_acc_net <= val_acc:
                self.best_acc_net = val_acc
                self.best_mat_net = val_cm
                self.best_epoch_net = epoch
                try:
                    checkpoint_best = {
                        'epoch': self.best_epoch_net,
                        'best_acc': self.best_acc_net,
                        'net_state_dict': self.net.module.state_dict(),
                        'optimizer_net_state_dict': self.optimizer.state_dict(),
                    }
                    torch.save(checkpoint_best, '%s/net_classifier_best_net.tar' % self.opt['checkpoints'])
                except AttributeError:
                    checkpoint_best = {
                        'epoch': self.best_epoch_net,
                        'best_acc': self.best_acc_net,
                        'net_state_dict': self.net.state_dict(),
                        'optimizer_net_state_dict': self.optimizer.state_dict(),
                    }
                    torch.save(checkpoint_best, '%s/net_classifier_best_net.tar' % self.opt['checkpoints'])
                print('best epoch:{}, best acc:{:.2%}'.format(self.best_epoch_net, self.best_acc_net))
        print('best epoch:{}, best acc:{:.2%}'.format(self.best_epoch_net, self.best_acc_net))
        print(self.best_mat_net)
        print('----------------------------')
        print('testing ......')
        test_loss, test_acc, test_cm, auc = self.test_model()
        dic = self.analysis_model(test_cm)
        dic['auc'] = round(auc, 6)
        print(dic)
        print()
        print()
