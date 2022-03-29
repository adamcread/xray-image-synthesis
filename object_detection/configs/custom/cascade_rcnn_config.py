_base_ = '../cascade_rcnn/cascade_mask_rcnn_r101_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=2,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=2,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=2,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=2,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
    )
)

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('FIREARM', 'KNIFE')
data = dict(
    train=dict(
        img_prefix='../dataset/xray/synthetic_composed/unpaired_combined_dice_100/',
        classes=classes,
        ann_file='../dataset/xray/synthetic_composed/helper/annotation/unpaired_combined_dice_100.json'),
    val=dict(
        img_prefix='../dataset/xray/synthetic_composed/unpaired_combined_dice_100/',
        classes=classes,
        ann_file='../dataset/xray/synthetic_composed/helper/annotation/unpaired_combined_dice_100.json'),
    test=dict(
        img_prefix='../dataset/xray/synthetic_composed/unpaired_combined_dice_100/',
        classes=classes,
        ann_file='../dataset/xray/synthetic_composed/helper/annotation/unpaired_combined_dice_100.json')
)

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = './checkpoints/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf_2_classes.pth'

