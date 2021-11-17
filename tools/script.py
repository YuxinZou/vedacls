import cv2
import os
import random
import shutil
import imgaug.augmenters as iaa

from tqdm import tqdm


def resize_img():
    src = '/DATA8_DB12/home/yuxinzou/zhongke/vedacls/data/val/1'
    dst = '/DATA8_DB12/home/yuxinzou/zhongke/vedacls/data_resized/val/1'
    for i in tqdm(os.listdir(src)):
        fname = os.path.join(src, i)
        img = cv2.imread(fname)
        img = cv2.resize(img, (480, 270))
        cv2.imwrite(os.path.join(dst, i), img)


def split_trainval():
    random.seed(0)
    src = '/DATA8_DB12/home/yuxinzou/zhongke/vedacls/data_resize/train/1'
    dst = '/DATA8_DB12/home/yuxinzou/zhongke/vedacls/data_resize/val/1'
    img_list = os.listdir(src)
    num = len(img_list)
    print(num)
    val_num = int(num * 0.2)
    val_sample = random.sample(img_list, val_num)
    print(val_sample)
    print(len(val_sample))
    for i in val_sample:
        fname = os.path.join(src, i)
        shutil.move(fname, dst)


def test_imgaug():
    import numpy as np
    trans = iaa.Resize((120, 140), interpolation="linear")
    images = np.zeros((100, 112, 112, 1), dtype=np.uint8)  # two example images
    # images[:, 64, 64, :] = 255
    images_aug = trans(images=images)
    print(type(images_aug))
    for i in range(50):
        print(images_aug[i].shape)


if __name__ == '__main__':
    test_imgaug()
