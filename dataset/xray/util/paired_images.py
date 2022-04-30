import cv2 as cv
import os
import random
from pycocotools.coco import COCO
import numpy as np
import imutils

def crop_image(image, border=10, threat=False, threat_image=None):
    image_gray = cv.cvtColor(image.copy(), cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(image_gray, 30, 255, cv.THRESH_BINARY)

    contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    max_contour = max(contours, key=cv.contourArea)
    
    x,y,w,h = cv.boundingRect(max_contour)

    image = image[
        max(0, y-border): min(image.shape[0], y+h+border), 
        max(0, x-border): min(image.shape[1], x+w+border), 
        :
    ]

    if threat:
        threat_image = threat_image[
            max(0, y-border): min(threat_image.shape[0], y+h+border), 
            max(0, x-border): min(threat_image.shape[1], x+w+border), 
            :
        ]

        return image, threat_image

    return image


def paired_grouping(res, root, dest):
    groupings = os.listdir(root)

    for i, group in enumerate(groupings):
        print(f'{i+1} out of {len(groupings)} {group}')
        composed = cv.imread(f'{root}/{group}/composed.png')
        decomposed_bag = cv.imread(f'{root}/{group}/decomposed_bag.png')
        decomposed_threat = cv.imread(f'{root}/{group}/decomposed_threat.png')
        decomposed_threat_T = cv.imread(f'{root}/{group}/decomposed_threat_T.png')

        composed, decomposed_threat_T = crop_image(composed, border=30, threat=True, threat_image=decomposed_threat_T)
        decomposed_bag = crop_image(decomposed_bag, border=30)
        decomposed_threat = crop_image(decomposed_threat)
        
        composed = cv.resize(composed, (res, res))
        decomposed_bag = cv.resize(decomposed_bag, (res, res))
        decomposed_threat = cv.resize(decomposed_threat, (res, res))
        decomposed_threat_T = cv.resize(decomposed_threat_T, (res, res))

        h_img = cv.hconcat([composed, decomposed_bag, decomposed_threat, decomposed_bag, decomposed_threat_T])

        cv.imwrite(f'{dest}/{group}.png', h_img)


def get_transpose(image, segmentation, t=False):
    # draw polygon
    polygon_pts = np.array([[segmentation[i], segmentation[i+1]] for i in range(0, len(segmentation), 2)])
    polygon_pts = polygon_pts.reshape((-1, 1, 2))

    blank = np.zeros(image.shape, np.uint8)
    poly_image = cv.polylines(blank.copy(), [polygon_pts], True, (255, 255, 255), 2)
    poly_image = cv.cvtColor(poly_image, cv.COLOR_BGR2GRAY)
    if t:
        cv.imwrite('test.png', poly_image)

    _, thresh = cv.threshold(poly_image.copy(), 1, 255, cv.THRESH_BINARY)
    contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    max_contour = max(contours, key=cv.contourArea)

    clean_inv_mask = cv.drawContours(np.ones(image.shape, np.uint8)*255, [max_contour], -1, (0, 0, 0), -1)

    threat_item = cv.bitwise_or(clean_inv_mask, image)
    return threat_item

def paired_images(root, dest, threat_ids, bag_ids, bag_subviews, in_json, in_threat_json):
    coco = COCO(annotation_file=in_json)
    threat_coco = COCO(annotation_file=in_threat_json)

    counter = 0
    for threat_id in threat_ids:
        # for each bag
        for bag_id in bag_ids:
            # for each view
            for bag_view in ['side-view', 'top-view']:
                # for each rotation
                for bag_subview in bag_subviews[bag_view]:
                    # for each threat view
                    decomposed_bag = os.listdir(f'{root}/negative-bags/{bag_id}/{bag_view}/{bag_subview}')[0]

                    for threat_view in ['side', 'top']:
                        threat_images = os.listdir(f'{root}/{threat_id}/threat/{threat_view}-view')
                        images = os.listdir(f'{root}/{threat_id}/real/{bag_id}/{bag_view}/{bag_subview}-{threat_view}/')

                        # for each image in file
                        for image in images:
                            print(bag_id, bag_view, bag_subview, threat_view, threat_id)
                            test_dir = f'{dest}/{threat_id}-{bag_id}-{counter}'
                            if not os.path.exists(test_dir):
                                os.mkdir(test_dir)
                            
                            composed_image = cv.imread(f'{root}/{threat_id}/real/{bag_id}/{bag_view}/{bag_subview}-{threat_view}/{image}')
                            cv.imwrite(f'{test_dir}/composed.png', composed_image)

                            bag_image = cv.imread(f'{root}/negative-bags/{bag_id}/{bag_view}/{bag_subview}/{decomposed_bag}')
                            cv.imwrite(f'{test_dir}/decomposed_bag.png', bag_image)

                            threat_name = random.choice(threat_images)
                            threat_image = cv.imread(f'{root}/{threat_id}/threat/{threat_view}-view/{threat_name}')    
                            cv.imwrite(f'{test_dir}/decomposed_threat.png', threat_image)

                            for coco_img in coco.loadImgs(ids=coco.getImgIds()):
                                if coco_img['file_name'] == image:
                                    segmentation = coco.loadAnns(ids=[coco_img['id']])[0]['segmentation']
                                    transposed_threat = get_transpose(composed_image, segmentation)
                            
                            cv.imwrite(f'{test_dir}/decomposed_threat_T.png', transposed_threat)
                        
                            counter = counter + 1


# paired_images(
#     root='../paired/sorted',
#     dest='../paired/test',
#     threat_ids = [
#         'firearm-5', 'firearm-6', 'firearm-11', 'firearm-16', 'firearm-19', 
#         'knife-1', 'knife-2', 'knife-3', 'knife-4', 'knife-5'
#     ],
#     bag_ids = [
#         'bag-2', 'bag-16', 'bag-28', 'bag-32', 'bag-36', 'bag-47'
#     ],
#     bag_subviews = {
#         'side-view': ['end', 'left', 'right'],
#         'top-view': ['down', 'left', 'right', 'up']
#     },
#     in_json = '../paired/annotations/paired.json',
#     in_threat_json = '../paired/annotations/threat.json'
# )

paired_grouping(
    res = 128,
    root='../paired/original_size/grouped_scans',
    dest='../paired/resized_128x128/paired_images'
)