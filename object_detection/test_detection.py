from mmcv import Config, mkdir_or_exist
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

PREFIX = '../dataset/xray/unpaired/'
cfg = Config.fromfile('./configs/custom/cascade_rcnn_config.py')

model = init_detector(cfg, checkpoint='checkpoints/epoch_6.pth', device='cpu')
model.CLASSES = cfg.classes

img = PREFIX + 'original_size/composed_images/BAGGAGE_20140123_103927_68622_D.jpg'
result = inference_detector(model, img)
model.show_result(img, result, out_file='result.jpg')

