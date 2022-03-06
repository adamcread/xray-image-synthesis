from mmcv import Config, mkdir_or_exist
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import os.path as osp

from mmcv import Config
from mmdet.apis import set_random_seed
import os.path as osp
import torch


def resize_classes(model, num_classes):
    model_name = osp.splitext(model)[0]
    pretrained_weights  = torch.load(model)

    pretrained_weights['state_dict']['roi_head.bbox_head.0.fc_cls.weight'].resize_(num_classes+1, 1024)
    pretrained_weights['state_dict']['roi_head.bbox_head.0.fc_cls.bias'].resize_(num_classes+1)
    pretrained_weights['state_dict']['roi_head.bbox_head.1.fc_cls.weight'].resize_(num_classes+1, 1024)
    pretrained_weights['state_dict']['roi_head.bbox_head.1.fc_cls.bias'].resize_(num_classes+1)
    pretrained_weights['state_dict']['roi_head.bbox_head.2.fc_cls.weight'].resize_(num_classes+1, 1024)
    pretrained_weights['state_dict']['roi_head.bbox_head.2.fc_cls.bias'].resize_(num_classes+1)

    torch.save(pretrained_weights, f'{model_name}_{num_classes}_classes.pth')
    return f'{model_name}_{num_classes}_classes.pth'


PREFIX = '../dataset/xray/unpaired/'
cfg = Config.fromfile('./configs/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.py')
cfg.dataset_type = 'CocoDataset'
cfg.classes = ('FIREARM','KNIFE') 

cfg.data.train.img_prefix = PREFIX + 'resized/images/' 
cfg.data.train.classes = cfg.classes
cfg.data.train.ann_file = PREFIX + '/helper/annotation/dbf3_train.json'
cfg.data.train.type = 'CocoDataset'

cfg.data.val.img_prefix = PREFIX + 'resized/images/' 
cfg.data.val.classes = cfg.classes
cfg.data.val.ann_file = PREFIX + '/helper/annotation/dbf3_test.json'
cfg.data.val.type = 'CocoDataset'

cfg.data.test.img_prefix = PREFIX + 'resized/images/' 
cfg.data.test.classes = cfg.classes
cfg.data.test.ann_file = PREFIX + '/helper/annotation/dbf3_test.json'
cfg.data.test.type = 'CocoDataset'

for i, model in enumerate(cfg.model.roi_head.bbox_head):
    cfg.model.roi_head.bbox_head[i].num_classes = len(cfg.classes)

cfg.optimizer.lr = 0.02/8
cfg.lr_config.warmup = None
cfg.log_config.interval = 1

cfg.evaluation.metric = 'bbox'
cfg.evaluation.interval = 3
cfg.checkpoint_config.interval = 3

cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

cfg.load_from = 'checkpoints/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf_2_classes.pth'
cfg.work_dir = './outs'

datasets = [build_dataset(cfg.data.train)]
print(datasets)
model = build_detector(cfg.model)

mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets[0], cfg, distributed=False, validate=False)