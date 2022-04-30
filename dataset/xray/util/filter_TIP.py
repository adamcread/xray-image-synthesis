import os
from resize_images import resize_image
from pycocotools.coco import COCO

coco = COCO(annotation_file="../TIP/helper/annotation/dbf3_TIP100_train.json")

firearm_img_ids = coco.getImgIds(catIds=[1])
knife_img_ids = coco.getImgIds(catIds=[3])

filtered_imgs = list(map(lambda x: x['file_name'], coco.loadImgs(ids=firearm_img_ids+knife_img_ids)))
# print(filtered_imgs)

with open('../TIP/helper/file_lists/firearm_knife_TIP.txt', "w+") as fp:
    fp.write('\n'.join(filtered_imgs))

resize_image(
    root = "../TIP/original_size/",
    image_list = "../TIP/helper/file_lists/firearm_knife_TIP.txt",
    dest = "../TIP/resized_128x128/",
    modifier = 1
)


