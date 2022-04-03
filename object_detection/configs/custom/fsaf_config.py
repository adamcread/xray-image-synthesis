_base_ = '../fsaf/fsaf_r101_fpn_1x_coco.py'

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = './checkpoints/fsaf_r101_fpn_1x_coco-9e71098f.pth'
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    bbox_head=dict(
        type='FSAFHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        reg_decoded_bbox=True,
        # Only anchor-free branch is implemented. The anchor generator only
        #  generates 1 anchor at each feature point, as a substitute of the
        #  grid of features.
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=1,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(_delete_=True, type='TBLRBBoxCoder', normalizer=4.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
            reduction='none'),
        loss_bbox=dict(
            _delete_=True,
            type='IoULoss',
            eps=1e-6,
            loss_weight=1.0,
            reduction='none'))
)


# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('FIREARM', 'KNIFE')
data = dict(
    train=dict(
        img_prefix='../dataset/xray/unpaired/resized_128x128/composed_images/',
        classes=classes,
        ann_file='../dataset/xray/unpaired/helper/annotation/dbf3_train_resized.json'),
    val=dict(
        img_prefix='../dataset/xray/unpaired/resized_128x128/composed_images/',
        classes=classes,
        ann_file='../dataset/xray/unpaired/helper/annotation/dbf3_test_resized.json'),
    test=dict(
        img_prefix='../dataset/xray/unpaired/resized_128x128/composed_images/',
        classes=classes,
        ann_file='../dataset/xray/unpaired/helper/annotation/dbf3_test_resized.json')
)

runner = dict(type='EpochBasedRunner', max_epochs=24)
