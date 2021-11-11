import cv2
import torch
import numpy as np

from ..models import build_model
from ..utils import load_checkpoint
from .base import Common


class InferenceRunner(Common):
    def __init__(self, inference_cfg, common_cfg=None):
        inference_cfg = inference_cfg.copy()
        common_cfg = {} if common_cfg is None else common_cfg.copy()

        common_cfg['gpu_id'] = inference_cfg.pop('gpu_id')
        super(InferenceRunner, self).__init__(common_cfg)

        # common cfg
        self.batch = inference_cfg.get('batch', 1)
        self.fps = inference_cfg.get('fps', -1)
        # build test transform
        self.transform = self._build_transform(inference_cfg['transform'])
        # build model
        self.model = self._build_model(inference_cfg['model'])
        self.model.eval()

    def _build_model(self, cfg):
        self.logger.info('Build model')

        model = build_model(cfg)
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model.cuda()

        return model

    def load_checkpoint(self, filename, map_location='default', strict=True):
        self.logger.info('Load checkpoint from {}'.format(filename))

        if map_location == 'default':
            if self.use_gpu:
                device_id = torch.cuda.current_device()
                map_location = lambda storage, loc: storage.cuda(device_id)
            else:
                map_location = 'cpu'

        return load_checkpoint(self.model, filename, map_location, strict)

    def __call__(self, image):
        with torch.no_grad():
            image = self.transform(image=image)['image']
            image = image.unsqueeze(0)

            if self.use_gpu:
                image = image.cuda()

            output = self.model(image)
            output = torch.softmax(output, dim=-1)[0]
        output = output.cpu().numpy()

        return output

    def _load_video(self, path):

        cap = cv2.VideoCapture(path)
        if self.fps < 0:
            self.fps = int(np.round(cap.get(cv2.CAP_PROP_FPS)))
            interval = 1
        else:
            if cap.set(cv2.CAP_PROP_FPS, self.fps):
                interval = 1
            else:
                interval = int(cap.get(cv2.CAP_PROP_FPS) / self.fps)

        frames = []
        timestamps = []
        while cap.isOpened():
            ret, frame = cap.read()
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not ret:
                break
            frames.append(frame)
            timestamps.append(timestamp)

        return frames[::interval], timestamps[::interval]

    def _preprocess(self, frames):
        images = []
        for f in frames:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            image = self.transform(image=f)['image']
            images.append(image.unsqueeze(0))
        return torch.cat(images)

    def inference(self, video_path):
        frames, timestamps = self._load_video(video_path)
        images = self._preprocess(frames)
        results = []
        with torch.no_grad():
            clips = int(np.ceil(images.shape[0] / self.batch))
            for i in range(clips):
                frames = images[i * self.batch:(i + 1) * self.batch]

                if self.use_gpu:
                    frames = frames.cuda()

                output = self.model(frames)
                output = torch.softmax(output, dim=-1)
                results.append(output)

        results = torch.cat(results).cpu().numpy()
        labels = np.argmax(results, axis=1)

        return labels

