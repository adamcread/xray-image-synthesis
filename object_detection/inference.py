from mmcv import Config, mkdir_or_exist
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

PREFIX = './inference/original/'
cfg = Config.fromfile('./configs/custom/crcnn_config.py')

model = init_detector(cfg, checkpoint='./work_dirs/dbf3/crcnn/best.pth', device='cpu')
model.CLASSES = cfg.classes

img_firearm = PREFIX + 'firearm.jpg'
result_firearm = inference_detector(model, img_firearm)
model.show_result(
    img_firearm, 
    result_firearm, out_file='./inference/result/firearm.jpg',
    bbox_color=(255, 0, 0), 
    text_color=(255, 0, 0), 
    thickness=3, 
    font_size=0
)


img_knife = PREFIX + 'knife.jpg'
result_knife = inference_detector(model, img_knife)
model.show_result(
    img_knife, 
    result_knife, out_file='inference/result/knife.jpg',
    bbox_color=(255, 0, 0), 
    text_color=(255, 0, 0), 
    thickness=3, 
    font_size=0
)

