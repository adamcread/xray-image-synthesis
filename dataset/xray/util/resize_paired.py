import cv2 as cv
from pycocotools.coco import COCO
import numpy as np
import imutils
import json

def convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError





def resize_paired(in_file, out_file):
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
                "name": "XRAY"
            }
        ]
    }

    coco = COCO(annotation_file=in_file)
    ids = coco.getImgIds()
    for id in ids:
        print(f'{id} out of {len(ids)}')
        annotation = coco.loadAnns(ids=[id])[0]
        image = coco.loadImgs(ids=[id])[0]

        pts = annotation['segmentation']
        polygon_pts = np.array([[pts[i], pts[i+1]] for i in range(0, len(pts), 2)])
        polygon_pts = polygon_pts.reshape((-1, 1, 2))

        blank = np.zeros((image['height'], image['width'], 3), np.uint8)

        poly_image = cv.polylines(blank.copy(), [polygon_pts], True, (255, 255, 255), 2)
        poly_image = cv.cvtColor(poly_image, cv.COLOR_BGR2GRAY)

        _, thresh = cv.threshold(poly_image.copy(), 1, 255, cv.THRESH_BINARY)
        contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        max_contour = max(contours, key=cv.contourArea)
        poly_mask = cv.drawContours(blank, [max_contour], -1, (255, 255, 255), -1)

        polygon_resized = cv.resize(poly_mask, (128, 128))
        polygon_resized = cv.cvtColor(polygon_resized, cv.COLOR_BGR2GRAY)

        _, thresh_resized = cv.threshold(polygon_resized.copy(), 1, 255, cv.THRESH_BINARY)
        resized_contours = cv.findContours(thresh_resized.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        resized_contours = imutils.grab_contours(resized_contours)
        max_resized_contour = max(resized_contours, key=cv.contourArea)

        segmentation = []
        for position in max_resized_contour:
            segmentation.extend(list(map(lambda x: int(x), position[0])))
        segmentation.extend([segmentation[0], segmentation[1]])
        
        coco_dataset["images"].append({
                    "license": 0,
                    "file_name": image['file_name'],
                    "width": 128,
                    "height": 128,
                    "id": id
        })

        coco_dataset["annotations"].append({
            "segmentation": segmentation,
            "iscrowd": 0,
            "image_id": id,
            "id": id,
            "category_id": 1,
            "bbox": [],
            "area": 0
        })

    with open(out_file, 'w+') as fp:
        json.dump(coco_dataset, fp, indent=4)


resize_paired(
    in_file='../paired/annotations/paired.json',
    out_file="../paired/annotations/paired_resized.json"
)
