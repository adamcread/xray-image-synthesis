import glob 
import random 
import os

def train_unpaired_paths(classes, test_name, image_lists, image_roots, dest):
    bag_files = open(image_lists['bag'], 'r').readlines()

    groups = []
    for class_name in classes:
        decomp_images = open(image_lists['decomp_' + class_name], 'r').readlines()
        for comp_image in open(image_lists[class_name], 'r').readlines():
            bag_image = image_roots['bag'] + random.choice(bag_files).rstrip()
            decomp_image = image_roots['decomp_' + class_name] + random.choice(decomp_images).rstrip()
            unpaired_image = image_roots['unpaired'] + comp_image.rstrip()

            groups.append(f'{bag_image} {decomp_image} {unpaired_image}')

    with open(f'{dest}paths_train_{test_name}.txt', 'w+') as fp:
        print(len(groups))
        fp.write('\n'.join(groups))

def train_paired_paths(root, prefix, dest):
    images = os.listdir(root)
    images_prefix = list(map(lambda x: prefix+x, images))
    shuffled_images = random.sample(images_prefix, k=len(images_prefix))

    with open(dest, 'w+') as fp:
        fp.write('\n'.join(shuffled_images))


# train_unpaired_paths(
#     classes = ['knife', 'firearm'],
#     test_name='combined',
#     image_lists = {
#         'firearm': '../unpaired/helper/file_lists/composed_firearm.txt',
#         'knife': '../unpaired/helper/file_lists/composed_knife.txt',
#         'bag': '../unpaired/helper/file_lists/decomposed_bags.txt',
#         'decomp_firearm': '../unpaired/helper/file_lists/decomposed_firearm.txt',
#         'decomp_knife': '../unpaired/helper/file_lists/decomposed_knife.txt'
#     },
#     image_roots = {
#         'unpaired': '../dataset/xray/unpaired/resized_512x512/unpaired_images/threat_mask/',
#         'bag': '../dataset/xray/unpaired/resized_512x512/decomposed_images/bags/',
#         'decomp_firearm': '../dataset/xray/unpaired/resized_512x512/decomposed_images/threat_items/firearm/',
#         'decomp_knife': '../dataset/xray/unpaired/resized_512x512/decomposed_images/threat_items/knife/'
#     },
#     dest='../../../compositionGAN/scripts/',
# )

train_paired_paths(
    root='../paired/resized_512x512/paired_images/',
    prefix='../dataset/xray/paired/resized_512x512/paired_images/',
    dest='paths_train_paired_512x512.txt'
)
