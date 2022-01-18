import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a classification model')
    parser.add_argument('txt', type=str, help='absolute image folder path')
    parser.add_argument('vars', type=str, nargs='*')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    file_write_obj = open(args.txt, 'w')

    for folder in args.vars:
        for root, dirs, files in os.walk(folder, topdown=False):
            for name in files:
                name = os.path.join(root, name)
                label = name.split('/')[-2]
                file_write_obj.writelines(f'{name} {label}')
                file_write_obj.write('\n')


if __name__ == '__main__':
    main()