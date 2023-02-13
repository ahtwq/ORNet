import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from interval import Interval


class FusionLoss(nn.Module):
    def __init__(self, claloss, regloss, alpha):
        super(FusionLoss, self).__init__()
        self.claloss = claloss
        self.regloss = regloss
        self.alpha = alpha

    def forward(self, preds, targets1, targets2):
        output1, output2 = preds
        if self.alpha == 0.0:
            loss1 = self.claloss(output1, targets1)
            return loss1

        loss1 = self.claloss(output1, targets1)
        targets2 = targets2.type_as(output2).view_as(output2)
        loss2 = self.regloss(output2, targets2)
        loss_sum = (1 - self.alpha) * loss1 + self.alpha * loss2
        return loss_sum

class JointFusionLoss(nn.Module):
    def __init__(self, claloss, regloss, alpha):
        super(JointFusionLoss, self).__init__()
        self.claloss = claloss
        self.regloss = regloss
        self.alpha = alpha

    def forward(self, preds, target1, rtarget1, target2, rtarget2):
        x1, x1_r, x2, x2_r = preds

        loss1 = self.claloss(x1, target1)
        rtarget1 = rtarget1.type_as(x1_r).view_as(x1_r)
        loss1_r = self.regloss(x1_r, rtarget1)

        loss2 = self.claloss(x2, target2)
        rtarget2 = rtarget2.type_as(x2_r).view_as(x2_r)
        loss2_r = self.regloss(x2_r, rtarget2)

        loss1_sum = (1 - self.alpha) * loss1 + self.alpha * loss1_r
        loss2_sum = (1 - self.alpha) * loss2 + self.alpha * loss2_r
        return loss1_sum + loss2_sum


def cls2score(cuts, num_classes):
    assert len(cuts) == num_classes - 1
    if not isinstance(cuts, torch.Tensor):
        cuts = torch.FloatTensor(cuts)
    dist = (max(cuts) - min(cuts)) / (len(cuts) - 1)
    score = torch.cat([cuts[0:1] - dist, cuts, cuts[-1:] + dist], dim=0)
    score = 0.5 * (score[1:] + score[:-1])
    score = score.view(-1,1)
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


# compute ac with offset <= 1
def cal_acc_oneoff(m):
    corrects = np.diag(m, -1).sum() + np.diag(m, 0).sum() + np.diag(m, 1).sum()
    samples = m.sum()
    return corrects / samples


# compute joint ac
def num_consistent(preds1, preds2, labels1, labels2):
    assert len(preds1) == len(preds2) == len(labels1) == len(labels2)
    n = len(preds1)
    corrects = 0
    for i in range(n):
        if preds1[i] == labels1[i] and preds2[i] == labels2[i]:
            corrects += 1
    return corrects


def qwk(conf_mat):
    num_ratings = len(conf_mat)
    num_scored_items = float(np.sum(conf_mat))

    hist_rater_a = np.sum(conf_mat, axis=1)
    hist_rater_b = np.sum(conf_mat, axis=0)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator


class labelsmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(labelsmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(-1)
        num_classes = pred.size(-1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (num_classes - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, -1))

