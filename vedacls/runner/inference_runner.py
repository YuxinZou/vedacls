import cv2
import torch
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..models import build_model
from ..utils import load_checkpoint
from .base import Common


def parse_json(json_path):
    data = json.load(open(json_path, 'r'))['video']
    return data


def xywh2xyxy(box):
    return np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]])


def cal_iou(box1, box2):
    box1_ = xywh2xyxy(box1)
    box2_ = xywh2xyxy(box2)
    bxmin = max(box1_[0], box2_[0])
    bymin = max(box1_[1], box2_[1])
    bxmax = min(box1_[2], box2_[2])
    bymax = min(box1_[3], box2_[3])

    bbxmin = min(box1_[0], box2_[0])
    bbymin = min(box1_[1], box2_[1])
    bbxmax = max(box1_[2], box2_[2])
    bbymax = max(box1_[3], box2_[3])

    bwidth = bxmax - bxmin
    bhight = bymax - bymin
    if bwidth < 0 or bhight < 0:
        return 0, [bbxmin, bbymin, bbxmax, bbymax]
    inter = bwidth * bhight
    union = (box1_[2] - box1_[0]) * (box1_[3] - box1_[1]) + (
            box2_[2] - box2_[0]) * (
                    box2_[3] - box2_[1]) - inter
    return inter / union, [bbxmin, bbymin, bbxmax, bbymax]


def unclip(box, shape, unclip_ratio=1.5, keep_ratio=True):
    H, W = shape
    w = box[2] - box[0]
    h = box[3] - box[1]
    if keep_ratio:
        half_size = int(max(w, h) * (unclip_ratio - 1) / 2)
        new_x1 = max((box[0] - half_size), 0)
        new_y1 = max((box[1] - half_size), 0)
        new_x2 = min((box[2] + half_size), W)
        new_y2 = min((box[3] + half_size), H)
    else:
        center_w = int((box[2] + box[0]) / 2)
        center_h = int((box[3] + box[1]) / 2)
        half_size = int(max(w, h) * unclip_ratio / 2)
        new_x1 = max((center_w - half_size), 0)
        new_y1 = max((center_h - half_size), 0)
        new_x2 = min((center_w + half_size), W)
        new_y2 = min((center_h + half_size), H)

    return [new_x1, new_y1, new_x2, new_y2]


class InferenceRunner(Common):
    def __init__(self, inference_cfg, common_cfg=None):
        inference_cfg = inference_cfg.copy()
        common_cfg = {} if common_cfg is None else common_cfg.copy()

        common_cfg['gpu_id'] = inference_cfg.pop('gpu_id')
        super(InferenceRunner, self).__init__(common_cfg)

        # common cfg
        self.batch = inference_cfg.get('batch', 1)
        self.fps = inference_cfg.get('fps', -1)
        self.class_name = inference_cfg['class_name']
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
        frames, _ = self._load_video(video_path)
        images = self._preprocess(frames)
        results = []
        with torch.no_grad():
            clips = int(np.ceil(images.shape[0] / self.batch))
            for i in range(clips):
                batch = images[i * self.batch:(i + 1) * self.batch]

                if self.use_gpu:
                    batch = batch.cuda()

                output = self.model(batch)
                output = torch.softmax(output, dim=-1)
                results.append(output)

        results = torch.cat(results).cpu().numpy()
        labels = np.argmax(results, axis=1)
        scores = np.max(results, axis=1)

        return labels, scores, frames

    def plot_v3(self, video_path, json_path, save_pth):
        frames, _ = self._load_video(video_path)
        size = frames[0].shape[:2][::-1]

        video = cv2.VideoWriter(
            save_pth, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps,
            size)
        font_text = ImageFont.truetype("./fonts/SimHei.ttf", min(size) // 20)

        gt = parse_json(json_path)
        assert len(frames) == len(gt)
        texts = []

        for idx, (img, g) in enumerate(zip(frames, gt)):
            names = [box[0] for box in g['bbox_vector']]
            if 'funnel_paper' not in names or 'glass_rod' not in names:
                continue
            for bbox in g['bbox_vector']:
                if bbox[0] == 'funnel_paper':
                    funnel_paper_box = bbox[-1]
                elif bbox[0] == 'glass_rod':
                    glass_rod_box = bbox[-1]
            iou, max_box = cal_iou(funnel_paper_box, glass_rod_box)

            if iou > 0:
                max_box = unclip(max_box, img.shape[:2], keep_ratio=False)
                crop_img = img[max_box[1]:max_box[3], max_box[0]:max_box[2], :]
                probs = self.__call__(crop_img)
                label = self.class_name[np.argmax(probs, axis=-1)]
                score = np.max(probs)
                texts.append(['进行识别', f"{label}, {score:.4f}"])
            else:
                texts.append(['不进行识别', f""])

        if len(texts) >= 5:
            for i in range(2, len(texts)-2):
                if texts[i] == '不进行识别':
                    continue
                text = [text[1].split(',')[0] for text in texts[i-2:i+2]]
                text = [x for x in text if x != '']
                maxlabel = max(text, key=text.count)
                texts[i] = maxlabel

        for idx, (img, text) in enumerate(zip(frames, texts)):
            frame = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame)

            draw.text((10, 10), text[0], (255, 0, 0), font=font_text)
            draw.text((10, 15 + min(size) // 20), text[1], (255, 0, 0),
                      font=font_text)

            frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
            video.write(frame)
        video.release()

    def plot_v2(self, video_path, json_path, save_pth):
        frames, _ = self._load_video(video_path)
        size = frames[0].shape[:2][::-1]

        video = cv2.VideoWriter(
            save_pth, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps,
            size)
        font_text = ImageFont.truetype("./fonts/SimHei.ttf", min(size) // 20)

        gt = parse_json(json_path)
        assert len(frames) == len(gt)

        for idx, (img, g) in enumerate(zip(frames, gt)):
            if isinstance(img, np.ndarray):
                frame = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame)

            names = [box[0] for box in g['bbox_vector']]
            if 'funnel_paper' not in names or 'glass_rod' not in names:
                continue
            for bbox in g['bbox_vector']:
                if bbox[0] == 'funnel_paper':
                    funnel_paper_box = bbox[-1]
                elif bbox[0] == 'glass_rod':
                    glass_rod_box = bbox[-1]
            iou, max_box = cal_iou(funnel_paper_box, glass_rod_box)

            if iou > 0:
                max_box = unclip(max_box, img.shape[:2], keep_ratio=False)
                crop_img = img[max_box[1]:max_box[3], max_box[0]:max_box[2], :]
                probs = self.__call__(crop_img)
                label = self.class_name[np.argmax(probs, axis=-1)]
                score = np.max(probs)

                draw.text((10, 10), f"进行识别",
                          (0, 0, 255), font=font_text)
                draw.text((10, 15 + min(size) // 20), f"{label}, {score:.4f}",
                          (0, 0, 255), font=font_text)
            else:
                draw.text((10, 10), f"不进行识别",
                          (255, 0, 0), font=font_text)

            frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
            video.write(frame)
        video.release()

    def plot(self, video_path, save_pth):
        assert save_pth is not None
        labels, scores, frames = self.inference(video_path)
        size = frames[0].shape[:2][::-1]

        video = cv2.VideoWriter(
            save_pth, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps,
            size)
        font_text = ImageFont.truetype("./fonts/SimHei.ttf", min(size) // 20)
        for i, frame in enumerate(frames):
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame)

            label = labels[i]
            score = scores[i]
            draw.text((10, 10), f"{self.class_name[label]}, {score:.4f}",
                      (255, 0, 0), font=font_text)

            frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
            video.write(frame)

        video.release()
