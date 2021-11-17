from collections import OrderedDict

import numpy as np

from .base import Base
from .registry import METRICS


@METRICS.register_module
class MultiAccuracy(Base):
    def __init__(self, num_classes, topk=(1,)):
        super(MultiAccuracy, self).__init__()

        self.num_classes = num_classes
        self.topk = topk
        self.maxk = max(topk)
        self.count = {k: 0 for k in range(num_classes)}
        self.tp = {k: 0 for k in range(num_classes)}

    def add(self, pred, gt):
        if gt.ndim == 1:
            gt = gt[:, None]

        pd = np.argsort(pred)[:, -self.maxk:][:, ::-1]

        for k in range(self.num_classes):
            mask = gt == k
            self.count[k] += np.sum(mask)
            self.tp[k] += np.sum(pd[mask] == gt[mask])

        print(np.sum([self.count[k] for k in range(self.num_classes)]))

    def reset(self):
        self.count = {k: 0 for k in range(self.num_classes)}
        self.tp = {k: 0 for k in range(self.num_classes)}

    def result(self):
        res = OrderedDict()
        for k in range(self.num_classes):
            res['top1_cls{}'.format(k)] = self.tp[k] / (self.count[k] + 1e-5)

        return res
