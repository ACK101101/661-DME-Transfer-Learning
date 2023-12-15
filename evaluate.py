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

classes = ('background', 'fluid')
palette = [[0, 0, 0], [255, 255, 255]]

@DATASETS.register_module()
class OctDmeDataset(BaseSegDataset):
    METAINFO = dict(classes=classes, palette=palette)

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)


if __name__ == '__main__':
    data_root = 'Dataset'
    img_dir = 'images'
    ann_dir = 'labels'
    work_dir = './work_dirs/unet_fcn_stare_stare'
    checkpoint = work_dir+'/20231214_202828/iter_2000.pth'  # Changes this to direct to loaded checkpoint

    # Change this to direct to checkpoint configuration
    cfg = Config.fromfile(work_dir + '/20231214_202828/vis_data/config.py')

    cfg.load_from = checkpoint

    # runner = Runner.from_cfg(cfg)
    # runner.val()

    model = init_model(cfg, checkpoint, 'cuda:0')

    for img_num in [10, 15, 20, 25, 28, 30, 32, 35, 40, 45, 50]:
        img_name = f'10_{img_num}'
        img_path = './Dataset/images/' + f'{img_name}.png'
        label_path = './Dataset/labels/' + f'{img_name}.png'

        img = mmcv.imread(img_path)
        result = inference_model(model, img)
        fig, ax = plt.subplots(1, 2, figsize=(4 * 2, 3))
        for axis in ax:
            axis.axis('off')
        vis_result = show_result_pyplot(model, img, result, show=False)
        ax[-1].imshow(vis_result)    # show result
        ax[-1].set_title('Prediction')

        ax[0].imshow(mmcv.bgr2rgb(mmcv.imread(img_path)))    # show orig image
        ax[0].imshow(mmcv.bgr2rgb(mmcv.imread(label_path)), alpha=0.5)  # show label
        ax[0].set_title('Truth')
        fig.suptitle(img_name)

        fig.tight_layout()
        plt.show()
        # exit(0)
