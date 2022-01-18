from collections import OrderedDict

import numpy as np

from .base import Base
from .registry import METRICS


@METRICS.register_module
class ConfusionMatrix(Base):
    def __init__(self, num_classes, topk=(1,)):
        super(ConfusionMatrix, self).__init__()

        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

        self.topk = topk
        self.maxk = max(topk)

    def add(self, pred, gt):
        pd = np.argsort(pred)[:, -self.maxk:][:, ::-1]

        gt = gt.reshape(-1)
        pd = pd.reshape(-1)

        current_state = np.bincount(
            self.num_classes * gt + pd, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        self.confusion_matrix += current_state

    def reset(self):
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes))

    def result(self):
        res = OrderedDict()
        norm_cm = self.confusion_matrix / (
            self.confusion_matrix.sum(axis=1).reshape(-1, 1) + 1e-6)
        acc = norm_cm[np.arange(self.num_classes), np.arange(self.num_classes)].mean()
        print(self.confusion_matrix.astype(np.uint32))
        res['acc'] = acc

        return res


@METRICS.register_module
class MultiConfusionMatrix(Base):
    def __init__(self, num_classes, fbg_thresh=0.5, topk=(1,)):
        super(MultiConfusionMatrix, self).__init__()

        self.num_classes = num_classes
        self.fbg_thresh = fbg_thresh
        self.confusion_matrix = np.zeros((num_classes, num_classes))

        self.topk = topk
        self.maxk = max(topk)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def add(self, pred, gt):

        pred = self.sigmoid(pred)
        pred *= (1 - pred[:, 0])[:, None]

        pd = np.argsort(pred)[:, -self.maxk:][:, ::-1]

        gt = gt.reshape(-1)
        pd = pd.reshape(-1)

        current_state = np.bincount(
            self.num_classes * gt + pd, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        self.confusion_matrix += current_state

    def reset(self):
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes))

    def result(self):
        res = OrderedDict()
        norm_cm = self.confusion_matrix / (
            self.confusion_matrix.sum(axis=1).reshape(-1, 1) + 1e-6)
        acc = norm_cm[np.arange(self.num_classes), np.arange(self.num_classes)].mean()
        res['acc'] = acc

        return res
