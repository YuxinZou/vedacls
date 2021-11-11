import os
import sys
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from vedacls.runner import InferenceRunner
from vedacls.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file path')
    parser.add_argument('--inp', type=str, help='input video path')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    inference_cfg = cfg['inference']
    common_cfg = cfg.get('common')

    runner = InferenceRunner(inference_cfg, common_cfg)
    runner.load_checkpoint(args.checkpoint)

    output = runner.inference(args.inp)
    print(output)


if __name__ == '__main__':
    main()
