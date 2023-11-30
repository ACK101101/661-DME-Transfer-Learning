# Check Pytorch installation
import torch, torchvision

print(torch.__version__, torch.cuda.is_available())

# Check MMSegmentation installation
import mmseg

print(mmseg.__version__)

import mmcv
import mmengine
import matplotlib.pyplot as plt
from mmengine.runner import Runner
from mmseg.apis import init_model, inference_model, show_result_pyplot
from mmengine import Config
from PIL import Image
import os.path as osp
import numpy as np
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

classes = ('background', 'object')
class_weights = [.1, 1.0]
palette = [[255, 0, 0], [0, 0, 255]]

data_root = 'Dataset'
img_dir = 'images'
ann_dir = 'labels'

@DATASETS.register_module()
class OctDmeDataset(BaseSegDataset):
    METAINFO = dict(classes=classes, palette=palette)

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)


if __name__ == '__main__':
    # mean = 0
    # std = 0
    # total = 0
    # for file in mmengine.scandir(osp.join(data_root, img_dir), suffix='.png'):
    #     img = np.array(Image.open(osp.join(data_root, img_dir, file)))
    #     mean = np.mean(img)
    #     std += np.std(img)
    #     total += 1
    # mean /= total
    # std /= total
    # print('mean ', mean)
    # print('std ', std)
    # mean 0.5191644901637341
    # std 54.68139636232457

    img = mmcv.imread('./Dataset/images/10_32.png')
    plt.imshow(mmcv.bgr2rgb(img))
    plt.show()

    img = mmcv.imread('./Dataset/labels/10_32.png')
    plt.imshow(mmcv.bgr2rgb(img))
    plt.show()

    data_root = 'Dataset'
    img_dir = 'images'
    ann_dir = 'labels'

    cfg = Config.fromfile('configs/unet/unet-s5-d16_fcn_4xb4-40k_stare-128x128.py')
    print(f'Config:\n{cfg.pretty_text}')

    # Since we use only one GPU, BN is used instead of SyncBN
    cfg.norm_cfg = dict(type='BN', requires_grad=True)
    cfg.crop_size = (256, 256)
    cfg.model.data_preprocessor.size = cfg.crop_size
    cfg.model.data_preprocessor.mean = [
        0.519164, 0.519164, 0.519164
    ]
    cfg.model.data_preprocessor.std = [
        54.681, 54.681, 54.681
    ]
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg

    # Modify loss
    cfg.model.auxiliary_head.out_channels = 1
    cfg.model.auxiliary_head.loss_decode.use_sigmoid = True
    cfg.model.decode_head.out_channels = 1
    cfg.model.decode_head.loss_decode.use_sigmoid = True
    # cfg.model.auxiliary_head.loss_decode['class_weight'] = class_weights
    # cfg.model.decode_head.loss_decode['class_weight'] = class_weights

    # modify num classes of the model in decode/auxiliary head
    # cfg.model.decode_head.num_classes = 2 # already 2
    # cfg.model.auxiliary_head.num_classes = 2

    # Modify dataset type and path
    cfg.dataset_type = 'OctDmeDataset'
    cfg.data_root = data_root

    cfg.train_dataloader.batch_size = 1

    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='RandomResize', scale=(768, 496), ratio_range=(0.5, 2.0), keep_ratio=True),
        dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PackSegInputs')
    ]

    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(768, 496), keep_ratio=True),
        # add loading annotation after ``Resize`` because ground truth
        # does not need to do resize data transform
        dict(type='LoadAnnotations'),
        dict(type='PackSegInputs')
    ]

    cfg.train_dataloader.dataset = dict(
        type=cfg.dataset_type,
        data_root=cfg.data_root,
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
        pipeline=cfg.train_pipeline,
        ann_file='splits/train.txt')

    cfg.val_dataloader.dataset = dict(
        type=cfg.dataset_type,
        data_root=cfg.data_root,
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
        pipeline=cfg.test_pipeline,
        ann_file='splits/val.txt')

    cfg.test_dataloader = cfg.val_dataloader

    # Load the pretrained weights
    cfg.load_from = 'checkpoints/fcn_unet_s5-d16_128x128_40k_stare_20201223_191051-7d77e78b.pth'

    # Set up working dir to save files and logs.
    cfg.work_dir = './work_dirs/unet_fcn_stare'

    cfg.train_cfg.max_iters = 500
    cfg.train_cfg.val_interval = 500
    cfg.default_hooks.logger.interval = 50
    cfg.default_hooks.checkpoint.interval = 500

    # Set seed to facilitate reproducing the result
    cfg['randomness'] = dict(seed=0)

    # Remove times from dataset loader
    cfg.val_evaluator.iou_metrics = ['mIoU']

    # Let's have a look at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')

    runner = Runner.from_cfg(cfg)

    runner.val()

    runner.train()

    checkpoint_path = './work_dirs/unet_fcn_stare/iter_200.pth'
    model = init_model(cfg, checkpoint_path, 'cuda:0')

    img = mmcv.imread('./Dataset/images/10_32.png')
    result = inference_model(model, img)
    plt.figure(figsize=(8, 6))
    vis_result = show_result_pyplot(model, img, result)
    plt.imshow(mmcv.bgr2rgb(vis_result))

    exit(0)
