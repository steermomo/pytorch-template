import torch.nn.functional as F
import torch
import torch.nn as nn
from model.lovasz_losses import lovasz_softmax


def nll_loss(output, target):
    return F.nll_loss(output, target)


class ClassificationLosses(object):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, batch_average=True, cuda=False):
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.ignore_index = ignore_index

    def build_loss(self, mode='ce'):
        if mode == 'ce':
            return self.CrossEntropyLoss

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def lovasz_loss(probas, labels, classes='present', per_image=False, ignore=None):
        """
        Multi-class Lovasz-Softmax loss
        probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        per_image: compute the loss per image instead of per batch
        ignore: void class labels
        """  
        probas = torch.softmax(probas, dim=1)
        loss = lovasz_softmax(probas, labels)
        return loss
    

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        print(f'Load loss {mode}')
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'lovaz':
            return self.lovasz_loss
        elif mode == 'bce':
            return self.bce_loss
        else:
            raise NotImplementedError

    def bce_loss(self, probas, labels):
        n, c, h, w = probas.shape
        # target = torch.zeros(n, c, h, w)
        target = torch.zeros_like(probas, requires_grad=False)
        one = torch.tensor(1, device=probas.device)
        zero = torch.tensor(0, device=probas.device)
        for c_idx in range(c):
            target[:, c_idx, :, :] = torch.where(labels == c_idx, one, zero)
        # probas_sigmoid = torch.sigmoid(probas)        
        loss = nn.functional.binary_cross_entropy_with_logits(probas, target, reduction="none")
        loss = torch.mean(loss, dim=0)
        if self.weight is not None:
            weight = self.weight.view(-1, c, 1, 1).to(probas.device)
            weighted_loss = loss * weight
        else:
            weighted_loss = loss
        weighted_loss = torch.mean(weighted_loss)
        return weighted_loss

    def lovasz_loss(self, probas, labels, classes='present', per_image=True, ignore=None):
        """
        Multi-class Lovasz-Softmax loss
        probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        per_image: compute the loss per image instead of per batch
        ignore: void class labels
        """  
        probas = torch.softmax(probas, dim=1)
        loss = lovasz_softmax(probas, labels, weights=self.weight, classes='all', per_image=per_image)
        return loss

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        # criterion = nn.CrossEntropyLoss
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss
