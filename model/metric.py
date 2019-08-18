import torch
import numpy as np


class ClasEvaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.metric = [
            self.Accuracy,
            self.Accuracy_Class
        ]

    def __len__(self):
        return 2

    def Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def _generate_matrix(self, gt_label, pre_label):
        mask = (gt_label >= 0) & (gt_label < self.num_class)
        label = self.num_class * gt_label[mask].astype('int') + pre_label[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_label, pre_label):
        # print(gt_label.shape, pre_label.shape)
        assert gt_label.shape == pre_label.shape
        self.confusion_matrix += self._generate_matrix(gt_label, pre_label)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def __iter__(self):
        for fuc in self.metric:
            return fuc


class SegEvaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.metric = [
            self.Pixel_Accuracy,
            self.Pixel_Accuracy_Class,
            self.MIoU,
            self.Dice,
            self.FWIoU
        ]

    def __len__(self):
        return len(self.metric)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def MIoU(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Dice(self):
        label = np.sum(self.confusion_matrix, axis=1).reshape(-1)  # ground true
        pred = np.sum(self.confusion_matrix, axis=0).reshape(-1)
        insert = np.diag(self.confusion_matrix).reshape(-1)

        dice = 2 * insert / (label + pred)
        return np.mean(dice[1:])  # background不算

    def FWIoU(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        # print(gt_image.shape, pre_image.shape)
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def __getitem__(self, idx):
        return self.metric[idx]
    # def __iter__(self):
    #     return self 
    # def __next__(self):
    #     for fuc in self.metric:
    #         return fuc


def my_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
