import cv2
import os
from tqdm import tqdm

def resize_img():
    src = '/DATA8_DB12/home/yuxinzou/zhongke/vedacls/data/train/1'
    dst = '/DATA8_DB12/home/yuxinzou/zhongke/vedacls/data_resize/train/1'
    for i in tqdm(os.listdir(src)):
        fname = os.path.join(src, i)
        img = cv2.imread(fname)
        img = cv2.resize(img, (480,270))
        cv2.imwrite(os.path.join(dst, i), img)


if __name__ == '__main__':
    resize_img()
