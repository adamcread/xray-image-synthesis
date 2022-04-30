# 5 images
# original
# original without threat object
# only threat object
# mask of threat object
# mask of bag

import cv2 as cv
import imutils
import numpy as np
import random


def unpaired_images(image_list, image_root, mask_root, dest, rotate):
    file_names = open(image_list, 'r').readlines()

    for i, f in enumerate(file_names):
        f = f.rstrip()
        print(f'{i+1} out of {len(file_names)}')
        all_mask = cv.imread(mask_root + f)
        img = cv.imread(image_root + f)

        mask = all_mask[:, all_mask.shape[1]//2:]

        masked_img_gray = cv.cvtColor(mask.copy(), cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(masked_img_gray, 70, 255, cv.THRESH_BINARY)

        contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        max_contour = max(contours, key=cv.contourArea)

        threat_removed = cv.drawContours(img.copy(), [max_contour], -1, (255, 255, 255), -1)
        clean_inv_mask = cv.drawContours(np.ones(img.shape, np.uint8)*255, [max_contour], -1, (0, 0, 0), -1)
        
        x,y,w,h = cv.boundingRect(max_contour)

        threat_item = cv.bitwise_or(clean_inv_mask, img)
        threat_item = threat_item[
            max(0, y-10): min(img.shape[0], y+h+10), 
            max(0, x-10): min(img.shape[1], x+w+10), 
            :
        ]
        threat_item = cv.resize(threat_item, img.shape[:2])

        if rotate:
            image_center = (64, 64)
            rot_mat = cv.getRotationMatrix2D(image_center, random.randint(0, 359), 1.0)
            threat_item = cv.warpAffine(threat_item, rot_mat, threat_item.shape[1::-1], borderValue=(255,255,255), flags=cv.INTER_LINEAR)

        blank = np.ones(img.shape, np.uint8)*255
        h_img = cv.hconcat([img, threat_removed, threat_item, blank, blank])

        cv.imwrite(dest + f, h_img)

unpaired_images(
    image_list = '../unpaired/helper/file_lists/composed_firearm_knife.txt',
    image_root = '../unpaired/resized_512x512/composed_images/',
    mask_root = '../unpaired/resized_512x512/masks/threat_mask/',
    dest = '../unpaired/resized_512x512/unpaired_images/threat_mask/',
    rotate = False
)