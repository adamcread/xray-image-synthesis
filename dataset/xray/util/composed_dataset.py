import os
import cv2 as cv
import imutils
import numpy as np
import json
import shutil

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

def get_bbox(img_path):
    transformed_img = cv.imread(img_path)
    grey_img = cv.cvtColor(transformed_img, cv.COLOR_BGR2GRAY)
    _, thresh1 = cv.threshold(transformed_img, 254, 255, cv.THRESH_BINARY_INV)
    thresh1 = cv.cvtColor(thresh1, cv.COLOR_BGR2GRAY)

    contours = cv.findContours(thresh1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    max_contour = max(contours, key=cv.contourArea)

    x,y,w,h = cv.boundingRect(max_contour)
    img_h = transformed_img.shape[0]
    img_w = transformed_img.shape[1]

    return img_h, img_w, x, y, w, h

 
def create_coco(root, composed_images, out_file):
    images = open(composed_images, 'r').readlines()
    for id, image in enumerate(images):
        print(f'making dataset: {id+1} out of {len(images)}')
        image = image.rstrip()
        image = image.split('/')[-1].split('.')[0]

        img_h, img_w, x, y, w, h = get_bbox(root+image+'_fake_A2_T.png')

        coco_dataset["images"].append({
            "license": 0,
            "file_name": image+'_fake_B.png',
            "width": img_w,
            "height": img_h,
            "id": id 
        })

        coco_dataset["annotations"].append({
            "segmentation": [x,y, x,(y+h), (x+w),(y+h), (x+w),y, x,y],
            "iscrowd": 0,
            "image_id": id,
            "id": id,
            "category_id": 1 if 'firearm' in image else 2,
            "bbox": [x, y, w, h],
            "area": h*w
        })

    with open(out_file, 'w+') as fp:
        json.dump(coco_dataset, fp, indent=4)


def move_images(root, composed_images, dest):
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.mkdir(dest)

    images = open(composed_images, 'r').readlines()
    for id, image in enumerate(images):
        print(f'moving images: {id+1} out of {len(images)}')
        image = image.rstrip()

        image = image.split('/')[-1].split('.')[0]
        B_image = image+'_fake_B.png'

        shutil.copyfile(root+B_image, dest+B_image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    parser.add_argument('--test')

    args = parser.parse_args()

    root = f'../../../compositionGAN/results/test_{args.test}/test_{args.file}/images/'

    create_coco(
        root = root,
        composed_images = '../../../compositionGAN/scripts/paths_test.txt',
        out_file = f'../composed/0_real_20000_fake/helper/annotation/{args.test}.json'
    )

    move_images(
        root = root,
        composed_images = '../../../compositionGAN/scripts/paths_test.txt',
        dest = f'../composed/0_real_20000_fake/{args.test}/'
    )