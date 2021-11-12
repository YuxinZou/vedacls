import os
import sys
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2

from vedacls.runner import TrainRunner
from vedacls.utils import Config
from vedacls.datasets import build_dataset
from vedacls.transforms import build_transform


def parse_args():
    parser = argparse.ArgumentParser(description='Train a classification model')
    parser.add_argument('config', type=str, help='config file path')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    train_cfg = cfg['train']['data']['train']

    transform = build_transform(train_cfg['transform'])
    dataset = build_dataset(train_cfg['dataset'], dict(transform=transform))

    for i in dataset:
        img = i[0]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('f', img)
        cv2.waitKey()


if __name__ == '__main__':
    main()
