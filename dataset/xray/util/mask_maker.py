from pycocotools.coco import COCO
import cv2 as cv
import numpy as np
import imutils
import os

# morphologyEx
def bag_masks(root, image_list, dest):
    file_names = open(image_list, 'r').readlines()

    for i in range(0, len(file_names)):
        file = file_names[i].rstrip()
        print(f'{i} out of {len(file_names)}')

        if not os.path.isfile(''+file):
            img = cv.imread(root + file)
            img_gray = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
            img_gray = cv.bitwise_not(img_gray)

            blurred = cv.GaussianBlur(img_gray.copy(), (7, 7), 0)
            _, thresh = cv.threshold(blurred.copy(), 1, 255, cv.THRESH_BINARY)

            contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            contours = imutils.grab_contours(contours)
            max_contour = max(contours, key=cv.contourArea)

            blank = np.zeros(img.shape, np.uint8)
            mask = cv.drawContours(blank, [max_contour], -1, (255, 255, 255), -1)
            cv.imwrite(dest + file, mask)


def threat_polygons(json_roots, image_list, image_root, dest, mask_root=None):
    for json in json_roots:
        coco = COCO(annotation_file=json)

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
                ones = np.ones(img.shape, np.uint8)*255

                polygon_mask = cv.fillPoly(blank, [polygon_points], color=(255, 255, 255))
                polygon_mask_inv = cv.fillPoly(ones, [polygon_points], color=(0, 0, 0))


                if mask_root:
                    mask = cv.imread(mask_root+fname)
                    mask_threat_removed = cv.bitwise_and(mask, polygon_mask_inv)
                    polygon_concat = cv.hconcat([mask_threat_removed, polygon_mask])
                else:
                    polygon_concat = cv.hconcat([polygon_mask_inv, polygon_mask])

                cv.imwrite(dest+fname, polygon_concat)


# bag_masks(
#     root='../unpaired/original_size/composed_images/',
#     image_list="../unpaired/helper/file_lists/composed_firearm_knife.txt",
#     dest="../unpaired/original_size/masks/bags/"
# )


threat_polygons(
    json_roots = ['../unpaired/helper/annotation/dbf3_train.json', '../unpaired/helper/annotation/dbf3_test.json'], 
    image_list = '../unpaired/helper/file_lists/composed_firearm_knife.txt', 
    image_root = '../unpaired/original_size/composed_images/',
    dest = '../unpaired/original_size/masks/threat_mask/',
    mask_root = '../unpaired/original_size/masks/bags/'
)