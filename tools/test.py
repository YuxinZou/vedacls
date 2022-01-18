import os
import sys
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from vedacls.runner import TestRunner
from vedacls.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a classification model')
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file path')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    _, fullname = os.path.split(cfg_path)
    fname, ext = os.path.splitext(fullname)

    root_workdir = cfg.pop('root_workdir')
    workdir = os.path.join(root_workdir, fname)
    os.makedirs(workdir, exist_ok=True)

    test_cfg = cfg['test']
    inference_cfg = cfg['inference']
    common_cfg = cfg['common']
    common_cfg['workdir'] = workdir

    runner = TestRunner(test_cfg, inference_cfg, common_cfg)
    runner.save = True
    runner.load_checkpoint(args.checkpoint)
    runner()


if __name__ == '__main__':
    main()
