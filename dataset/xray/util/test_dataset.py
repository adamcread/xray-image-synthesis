import cv2 as cv
import numpy as np
import random

def test_dataset(size, cls_props, cls_files, cls_roots, img_dest, flist_dest):
    prefix = '../dataset/xray/'
    cls_amount = {cls: (size*cls_props[cls])//100 for cls in cls_props.keys()}
    print(cls_amount)
    files = []

    for cls, amount in cls_amount.items():
        for n in range(0, amount):
            print(n, amount)
            cls_instance = cv.imread(cls_roots[cls] + random.choice(cls_files[cls]).rstrip())
            bag_instance = cv.imread(cls_roots['bag'] + random.choice(cls_files['bag']).rstrip())

            blank = np.ones(cls_instance.shape, np.uint8)*255

            test_img = cv.hconcat([bag_instance, bag_instance, cls_instance, blank, blank])

            f_name = f'{img_dest}{cls}_{n}.jpg'
            files.append(prefix + f_name[3:])
            cv.imwrite(f_name, test_img)
    
    with open(flist_dest, 'w+') as fp:
        fp.write('\n'.join(files))
    

# take in dataset size
# take in dataset proportions
# take in file_lists
# take in roots
# take in output folder

test_dataset(
    size = 20_000,
    cls_props = {
        'firearm': 50,
        'knife': 50
    },
    cls_files = {
        'bag': open('../unpaired/helper/file_lists/decomposed_bags.txt', 'r').readlines(),
        'firearm': open('../unpaired/helper/file_lists/decomposed_firearm.txt', 'r').readlines(),
        'knife': open('../unpaired/helper/file_lists/decomposed_knife.txt', 'r').readlines()
    },
    cls_roots = {
        'bag': '../unpaired/resized_128x128/decomposed_images/bags/',
        'firearm': '../unpaired/resized_128x128/decomposed_images/threat_items/firearm/',
        'knife': '../unpaired/resized_128x128/decomposed_images/threat_items/knife/'
    },
    img_dest = '../test/',
    flist_dest = 'test.txt'
)