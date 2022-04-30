from pycocotools.coco import COCO
import cv2 as cv
import numpy as np
import imutils
import os
import json


def resize_coco(json_roots, image_list, image_root, out_file, res, mask_root=None):
    coco_dataset = {
        "info": {
            "description": "X-ray image Dataset",
            "url": "https://www.durham.ac.uk",
            "version": "0.1",
            "year": 2018,
            "contributor": "ICG",
            "date_created": "20/11/2018"
        },
        "licenses": [
            {
                "url": "https://www.durham.ac.uk",
                "id": 0,
                "name": "Durham ICG, Research work"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "supercategory": "xrayimage",
                "id": 1,
                "name": "FIREARM"
            },
            {
                "supercategory": "xrayimage",
                "id": 2,
                "name": "KNIFE"
            }
        ]
    }

    for json_file in json_roots:
        coco = COCO(annotation_file=json_file)

        file_names = open(image_list).readlines()
        imgs = coco.loadImgs(ids=coco.getImgIds())

        img_ids = {}
        for img in imgs:
            img_ids[img['file_name']] = img['id']

        for i, fname in enumerate(file_names):
            fname = fname.rstrip()

            if fname in img_ids:
                print(f'{i+1} out of {len(file_names)}')
                file_annotation = coco.loadAnns(coco.getAnnIds(img_ids[fname]))[0]
                polygon = file_annotation['segmentation'][0]

                polygon_points = np.array([[polygon[i], polygon[i+1]] for i in range(0, len(polygon), 2)], np.int32)

                img = cv.imread(image_root+fname)

                blank = np.zeros(img.shape, np.uint8)
                polygon_mask = cv.fillPoly(blank, [polygon_points], color=(255, 255, 255))
                resized_mask = cv.resize(polygon_mask, (res, res))

                _, thresh1 = cv.threshold(resized_mask.copy(), 254, 255, cv.THRESH_BINARY)
                thresh1 = cv.cvtColor(thresh1, cv.COLOR_BGR2GRAY)

                contours = cv.findContours(thresh1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                contours = imutils.grab_contours(contours)
                max_contour = max(contours, key=cv.contourArea)

                x,y,w,h = cv.boundingRect(max_contour)
                coco_dataset["images"].append({
                    "license": 0,
                    "file_name": fname,
                    "width": res,
                    "height": res,
                    "id": file_annotation['id']
                })

                coco_dataset["annotations"].append({
                    "segmentation": [x,y, x,(y+h), (x+w),(y+h), (x+w),y, x,y],
                    "iscrowd": 0,
                    "image_id": file_annotation['id'],
                    "id": file_annotation['id'],
                    "category_id": file_annotation['category_id'],
                    "bbox": [x, y, w, h],
                    "area": w*h
                })

    with open(out_file, 'w+') as fp:
        json.dump(coco_dataset, fp, indent=4)



resize_coco(
    json_roots = ['../TIP/helper/annotation/dbf3_TIP100_train.json'], 
    image_list = '../TIP/helper/file_lists/firearm_knife_TIP.txt', 
    image_root = '../TIP/original_size/',
    out_file = '../TIP/helper/annotation/dbf3_TIP_resized.json',
    res=128
)