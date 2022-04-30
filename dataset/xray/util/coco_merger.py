"""
Created on Thu May 2 2019
@author: Brian Isaac-Medina
"""
import json
import sys
def check_if_image_exists(original_data, image_name):
    for im in original_data['images']:
        if im['file_name'] == image_name:
            return True
    return False

def search_image_name_by_id(data, image_id):
    for im in data['images']:
        if im['id'] == image_id:
            return im['file_name']
    return None

def search_image_id_by_name(data, image_name):
    for im in data['images']:
        if im['file_name'] == image_name:
            return int(im['id'])
    return None


def merger(json_file_1, json_file_2, OUTPUT_FILENAME):
    with open(json_file_1) as f:
        cvpr = json.load(f)

    with open(json_file_2) as f:
        additional = json.load(f)

    for img in additional['images']:
        print ("Processing image ", img['file_name'])
        if not check_if_image_exists(cvpr, img['file_name']):
            current_images = cvpr["images"]
            last_img = current_images[-1]
            last_id = int(last_img['id'])
            new_img = dict(img)
            new_img['id'] = last_id + 1
            cvpr['images'].append(new_img)

    for an in additional['annotations']:
        print ("Processing annotation ", an['id'])
        ad_img_id = an['image_id']
        img_name = search_image_name_by_id(additional, ad_img_id)
        img_new_id = search_image_id_by_name(cvpr, img_name)
        current_ann = cvpr['annotations']
        last_ann = current_ann[-1]
        last_id = int(last_ann['id'])
        an['id'] = last_id + 1
        an['image_id'] = img_new_id
        cvpr['annotations'].append(an)

    json_output = json.dumps(cvpr)
    with open(OUTPUT_FILENAME, 'w') as f:
        f.write(json_output)


# argc = int(len(sys.argv))
# if argc != 4:
#     print ('')
#     print ('USAGE: python ' + sys.argv[0] + ' <json file1> <json file2> <o/p json file>')
#     print (' json file      [i] Input json file1')
#     print (' json file      [i] Input json file2')
#     print (' json file      [o] Output json file path')
#     print ('')
#     quit()

# json_file_1 = sys.argv[1]
# json_file_2 = sys.argv[2]
# OUTPUT_FILENAME = sys.argv[3]

# with open(json_file_1) as f:
#     cvpr = json.load(f)

# with open(json_file_2) as f:
#     additional = json.load(f)

# for img in additional['images']:
#     print ("Processing image ", img['file_name'])
#     if not check_if_image_exists(cvpr, img['file_name']):
#         current_images = cvpr["images"]
#         last_img = current_images[-1]
#         last_id = int(last_img['id'])
#         new_img = dict(img)
#         new_img['id'] = last_id + 1
#         cvpr['images'].append(new_img)

# for an in additional['annotations']:
#     print ("Processing annotation ", an['id'])
#     ad_img_id = an['image_id']
#     img_name = search_image_name_by_id(additional, ad_img_id)
#     img_new_id = search_image_id_by_name(cvpr, img_name)
#     current_ann = cvpr['annotations']
#     last_ann = current_ann[-1]
#     last_id = int(last_ann['id'])
#     an['id'] = last_id + 1
#     an['image_id'] = img_new_id
#     cvpr['annotations'].append(an)

# json_output = json.dumps(cvpr)
# with open(OUTPUT_FILENAME, 'w') as f:
#     f.write(json_output)

# print(f'++input/output++\n{json_file_1}\n{json_file_2}\n{OUTPUT_FILENAME}')