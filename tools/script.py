import cv2
import os
import random
import shutil
import imgaug.augmenters as iaa
import numpy as np
import json
import pyclipper
import subprocess
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
    t = iaa.MultiplyBrightness()
    trans = iaa.Resize((120, 140), interpolation="linear")
    images = np.zeros((100, 112, 112, 1), dtype=np.uint8)  # two example images
    # images[:, 64, 64, :] = 255
    images_aug = trans(images=images)
    print(type(images_aug))
    for i in range(50):
        print(images_aug[i].shape)


def get_duration(path):
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        rate = cap.get(5)
        print(rate)
        FrameNumber = cap.get(7)
        print(FrameNumber)
        duration = FrameNumber / rate
        print(duration)


def get_duration(path):
    cameraCapture = cv2.VideoCapture(path)
    success, frame = cameraCapture.read()
    while success:
        if cv2.waitKey(1) == 27:
            break
        success, frame = cameraCapture.read()
        milliseconds = cameraCapture.get(cv2.CAP_PROP_POS_MSEC)

        # seconds = milliseconds // 1000
        # milliseconds = milliseconds % 1000
        # minutes = 0
        # hours = 0
        # if seconds >= 60:
        #     minutes = seconds // 60
        #     seconds = seconds % 60
        #
        # if minutes >= 60:
        #     hours = minutes // 60
        #     minutes = minutes % 60

        print(int(milliseconds))


def test_json():
    import json
    video_ori = f'/home/admin123/PycharmProjects/DATA/中科/按压v2/top/291501/291501-top-orig-ch-073-2021-10-26_15_37_27.479.avi'
    video_after = f'/home/admin123/PycharmProjects/DATA/中科/按压v2/top/291501/291501-top-ch-073-2021-10-26_15_37_27.479.avi'
    json_file = f'/home/admin123/PycharmProjects/DATA/中科/按压v2/top/291501/video_291501-ch-073-01top2021-10-26_15_37_27.479.json'
    gt = json.load(open(json_file, 'r'))['video']
    print(len(gt))
    # for g in gt:
    #     print(g['current_time'])
    # get_duration(video_ori)
    get_duration(video_after)


def test_folder():
    import json
    path = '/home/admin123/PycharmProjects/DATA/中科/按压/按压顶视'
    for i in os.listdir(path):
        if i == '291516':
            print(f'folder: {i}')
            subdpath = os.path.join(path, i)
            for j in os.listdir(subdpath):
                if j.endswith('.mp4'):
                    get_duration(os.path.join(subdpath, j))
                if j.endswith('.avi'):
                    get_duration(os.path.join(subdpath, j))
                if j.endswith('.json'):
                    gt = json.load(open(os.path.join(subdpath, j), 'r'))[
                        'video']
                    print(len(gt))
                    # for g in gt:
                    #     print(g)


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


def show_gt():
    import json
    image_path = '/home/admin123/PycharmProjects/DATA/中科/按压v2/top/291501/images'
    json_file = f'/home/admin123/PycharmProjects/DATA/中科/按压v2/top/291501/video_291501-ch-073-01top2021-10-26_15_37_27.479.json'
    gt = json.load(open(json_file, 'r'))['video']
    images = sorted(os.listdir(image_path))
    for i, (image, g) in enumerate(zip(images, gt)):
        img = cv2.imread(os.path.join(image_path, image))
        names = [box[0] for box in g['bbox_vector']]
        if 'funnel_paper' not in names or 'glass_rod' not in names:
            continue
        for bbox in g['bbox_vector']:
            if bbox[0] == 'funnel_paper':
                funnel_paper_box = bbox[-1]
            elif bbox[0] == 'glass_rod':
                glass_rod_box = bbox[-1]
        iou, max_box = cal_iou(funnel_paper_box, glass_rod_box)
        max_box = unclip(max_box, img.shape[:2])
        if iou > 0:
            for bbox in g['bbox_vector']:
                bbox = bbox[-1]
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + \
                                 bbox[3]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.rectangle(img, (max_box[0], max_box[1]),
                          (max_box[2], max_box[3]), (0, 0, 255), 1)
            cv2.imshow(f's', img)
            cv2.waitKey()


def check_data(mode='val'):
    path = f'/home/admin123/PycharmProjects/DATA/中科/按压v2/top/{mode}'
    # check file name
    folders = os.listdir(path)
    for f in sorted(folders):
        print(f)
        sufolder = os.path.join(path, f)
        chs = [fname for fname in os.listdir(sufolder) if 'orig' not in fname]
        origs = [fname for fname in os.listdir(sufolder) if 'orig' in fname]
        print(chs)
        assert len(chs) == 2
        assert len(origs) == 1


def split_images(mode='val'):
    path = f'/home/admin123/PycharmProjects/DATA/中科/按压v2/top/{mode}'
    # check file name
    folders = os.listdir(path)
    for f in sorted(folders):
        print(f)
        sufolder = os.path.join(path, f)
        ori_video = [fname for fname in os.listdir(sufolder) if 'orig' in fname]
        json_file = [fname for fname in os.listdir(sufolder) if
                     fname.endswith('json')]
        os.makedirs(os.path.join(sufolder, 'images'), exist_ok=True)
        cmd = f'ffmpeg -i {os.path.join(sufolder, ori_video[0])} {os.path.join(sufolder, "images")}/%05d.png'
        print(cmd)
        subprocess.call(cmd, shell=True)


def check_imageslen(mode='val'):
    path = f'/home/admin123/PycharmProjects/DATA/中科/按压v2/top/{mode}'
    # check file name
    folders = os.listdir(path)
    for f in sorted(folders):
        sufolder = os.path.join(path, f)
        json_file = [fname for fname in os.listdir(sufolder) if
                     fname.endswith('json')]
        images_path = os.path.join(sufolder, 'images')
        gt = json.load(open(os.path.join(sufolder, json_file[0]), 'r'))['video']
        assert len(gt) == len(os.listdir(images_path))


def show_img(mode='train'):
    path = f'/home/admin123/PycharmProjects/DATA/中科/按压v2/top/{mode}'
    # check file name
    folders = os.listdir(path)
    for f in sorted(folders):
        sufolder = os.path.join(path, f)
        json_file = [fname for fname in os.listdir(sufolder) if
                     fname.endswith('json')]
        images_path = os.path.join(sufolder, 'images')
        gt = json.load(open(os.path.join(sufolder, json_file[0]), 'r'))['video']
        images = sorted(os.listdir(images_path))
        for i, (image, g) in enumerate(zip(images, gt)):
            img = cv2.imread(os.path.join(images_path, image))
            names = [box[0] for box in g['bbox_vector']]
            if 'funnel_paper' not in names or 'glass_rod' not in names:
                continue
            for bbox in g['bbox_vector']:
                if bbox[0] == 'funnel_paper':
                    funnel_paper_box = bbox[-1]
                elif bbox[0] == 'glass_rod':
                    glass_rod_box = bbox[-1]
            iou, max_box = cal_iou(funnel_paper_box, glass_rod_box)

            max_box = unclip(max_box, img.shape[:2], keep_ratio=False)
            if iou > 0:
                for bbox in g['bbox_vector']:
                    if bbox[0] not in ['funnel_paper', 'glass_rod']:
                        continue
                    bbox = bbox[-1]
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[
                        1] + bbox[3]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                print(
                    f'box shape:{max_box[2] - max_box[0]} {max_box[3] - max_box[1]}')
                cv2.rectangle(img, (max_box[0], max_box[1]),
                              (max_box[2], max_box[3]), (0, 0, 255), 1)
                cv2.imshow(f's', img)
                cv2.waitKey()


def crop_img(mode='train'):
    path = f'/home/admin123/PycharmProjects/DATA/中科/按压v2/top/{mode}'
    # check file name
    folders = os.listdir(path)
    for f in sorted(folders):
        sufolder = os.path.join(path, f)
        json_file = [fname for fname in os.listdir(sufolder) if
                     fname.endswith('json')]
        images_path = os.path.join(sufolder, 'images')
        os.makedirs(os.path.join(sufolder, 'crop_images'), exist_ok=True)
        gt = json.load(open(os.path.join(sufolder, json_file[0]), 'r'))['video']
        images = sorted(os.listdir(images_path))
        for i, (image, g) in enumerate(zip(images, gt)):
            img = cv2.imread(os.path.join(images_path, image))
            names = [box[0] for box in g['bbox_vector']]
            if 'funnel_paper' not in names or 'glass_rod' not in names:
                continue
            for bbox in g['bbox_vector']:
                if bbox[0] == 'funnel_paper':
                    funnel_paper_box = bbox[-1]
                elif bbox[0] == 'glass_rod':
                    glass_rod_box = bbox[-1]
            iou, max_box = cal_iou(funnel_paper_box, glass_rod_box)

            max_box = unclip(max_box, img.shape[:2], keep_ratio=False)
            if iou > 0:
                # for bbox in g['bbox_vector']:
                #     if bbox[0] not in ['funnel_paper', 'glass_rod']:
                #         continue
                #     bbox = bbox[-1]
                #     x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[
                #         1] + bbox[3]
                #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                print(
                    f'box shape:{max_box[2] - max_box[0]} {max_box[3] - max_box[1]}')
                crop_img = img[max_box[1]:max_box[3], max_box[0]:max_box[2], :]
                cv2.imwrite(os.path.join(sufolder, 'crop_images', image),
                            crop_img)
                # cv2.rectangle(img, (max_box[0], max_box[1]),
                #               (max_box[2], max_box[3]), (0, 0, 255), 1)


def split_dataset():
    import random
    random.seed(0)
    print(random.random())
    path = '/home/admin123/PycharmProjects/DATA/中科/按压v2/top/'
    data = os.listdir(path)
    train_folders = random.sample(data, int(len(data) * 0.8))
    val_folders = [d for d in data if d not in train_folders]
    print(train_folders)
    print(len(train_folders))
    print(val_folders)
    print(len(val_folders))
    for v in val_folders:
        cmd = f'mv /home/admin123/PycharmProjects/DATA/中科/按压v2/top/{v} /home/admin123/PycharmProjects/DATA/中科/按压v2/val'
        subprocess.call(cmd, shell=True)


def move_data(mode='val'):
    src = f'/home/admin123/PycharmProjects/DATA/中科/按压v2/top/{mode}'
    for f in os.listdir(src):
        print(f)
        subdolder = os.path.join(src, f, 'crop_images')
        for fname in os.listdir(subdolder):
            shutil.copy(os.path.join(subdolder, fname),
                        os.path.join(f'/home/admin123/PycharmProjects/DATA/中科/按压v2/top/cls_data/{mode}/0', f'top_{f}_{fname}'))


if __name__ == '__main__':
    # move_data()
