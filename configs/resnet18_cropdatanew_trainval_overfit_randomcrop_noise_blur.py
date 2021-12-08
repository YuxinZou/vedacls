import cv2

# 1. configuration for inference
size = 256
crop_size = 224
padding_value = 127.5
img_norm_cfg = dict(mean=0.5, std=0.5, max_pixel_value=255.0)

inference = dict(
    gpu_id='2',
    transform=[
        dict(type='Resize', height=size, width=size),
        dict(type='CenterCrop', height=crop_size, width=crop_size),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='ToTensor'),
    ],
    model=dict(
        arch='resnet18',
        num_classes=2,
        pretrained=True,
    ),
    batch=32,
    fsp=-1,
    class_name=['按压滤纸', '未按压滤纸'],
)

# 2. configuration for train/test
root_workdir = 'workdir'
data_root = 'data_crop_new'
dataset_type = 'ImageFolder'

common = dict(
    seed=1111,
    logger=dict(
        handlers=(
            dict(type='StreamHandler', level='INFO'),
            dict(type='FileHandler', level='INFO'),
        ),
    ),
    cudnn_deterministic=False,
    cudnn_benchmark=True,
    metric=dict(type='Accuracy', topk=(1,)),
)

## 2.1 configuration for test
test = dict(
    data=dict(
        dataloader=dict(
            type='DataLoader',
            batch_size=128,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        ),
        dataset=dict(
            type=dataset_type,
            root=data_root + '/' + 'val',
        ),
        transform=inference['transform'],
    ),
)

## 2.2 configuration for train
train = dict(
    data=dict(
        train=dict(
            dataloader=dict(
                type='DataLoader',
                batch_size=256,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
            ),
            dataset=dict(
                type=dataset_type,
                root=data_root + '/' + 'train',
            ),
            transform=[
                dict(type='Resize', height=size, width=size),
                # dict(type='CenterCrop', height=crop_size, width=crop_size),
                dict(type='RandomResizedCrop',
                     height=crop_size, width=crop_size, scale=(0.6, 1.0),
                     p=1),
                dict(type='GaussNoise', var_limit=(0.2 * 255), p=0.5),
                dict(type='MotionBlur', blur_limit=(3, 7), p =0.5),
                dict(type='HorizontalFlip'),
                dict(type='VerticalFlip'),
                # dict(type='ColorJitter'),
                # dict(type='Rotate', limit=(-30, 30), p=0.5, value=padding_value,
                #      border_mode=cv2.BORDER_CONSTANT),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ToTensor'),
            ],
        ),
        val=dict(
            dataloader=dict(
                type='DataLoader',
                batch_size=128,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                drop_last=False,
            ),
            dataset=dict(
                type=dataset_type,
                root=data_root + '/' + 'val',
            ),
            transform=inference['transform'],
        ),
    ),
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    criterion=dict(type='CrossEntropyLoss'),
    lr_scheduler=dict(type='StepLR', step_size=30, gamma=0.1),
    max_epochs=90,
    log_interval=10,
    trainval_ratio=1,
    snapshot_interval=-1,
    save_best=True,
    resume=None,
)
