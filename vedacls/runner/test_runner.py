import torch
import numpy as np
import cv2

from .inference_runner import InferenceRunner


class TestRunner(InferenceRunner):
    def __init__(self, test_cfg, inference_cfg, common_cfg=None):
        super(TestRunner, self).__init__(inference_cfg, common_cfg)

        self.test_dataloader = self._build_dataloader(test_cfg['data'])
        self._save = False

    @property
    def save(self):
        return self._save

    @save.setter
    def save(self, v):
        self._save = v

    def __call__(self):
        self.metric.reset()
        self.model.eval()

        res = {}

        self.logger.info('Start testing')
        count = 0
        with torch.no_grad():
            for idx, (img, label, path) in enumerate(self.test_dataloader):
                if self.use_gpu:
                    img = img.cuda()

                out = self.model(img)
                self.metric.add(out.cpu().numpy(), label.cpu().numpy())
                res = self.metric.result()

                if self.save:
                    pd = np.argsort(out.cpu().numpy())[:, -1]
                    gt = label.cpu().numpy()
                    idxs = np.where(pd != gt)[0]
                    for i in idxs:
                        _img = img[i].permute(1, 2, 0).cpu().numpy()
                        p = pd[i]
                        g = gt[i]
                        mean = np.array(0.5, dtype=np.float32) * 255
                        std = np.array(0.5, dtype=np.float32) * 255
                        denominator = np.reciprocal(std, dtype=np.float32)
                        _img /= denominator
                        _img += mean
                        _img = _img.astype(np.uint8)
                        count += 1
                        cv2.imwrite(f'output/g{g}_p{p}.jpg', _img[:, :, (2, 1, 0)])

                self.logger.info('Test, Iter {}, {}'.format(
                    idx+1,
                    ', '.join(['{}: {:.4f}'.format(name, res[name]) for name in
                               res])))
        self.logger.info(', '.join(['{}: {:.4f}'.format(k, v) for k, v in
                   res.items()]))

        return res
