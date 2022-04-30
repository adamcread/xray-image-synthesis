"""
Created on Thu Apr 25 2019
@author: Brian Isaac-Medina

@ref: coco_json_create_2.0.py by Neel
"""
import sys
import json
import os
from PIL import Image
from shapely import geometry
from tqdm import tqdm
from colorama import Fore
from datetime import date


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


argc = int(len(sys.argv))

if argc <= 2 or (argc - 2) % 4 != 0:
    print('USAGE: python %s <output path> <img dir> <vgg json file> <category id> <category name> ' % sys.argv[0] +
          '[<img dir> <category id> ..., [<img...]]]')
    print(' output path                 [i] Output file to save coco format json file')
    print(' img dir                     [i] Image directory path')
    print(' vgg json file               [i] vgg json file')
    print(' category id                 [i] Category id {1 | 2 | 3...}')
    print(' category name               [i] Category name {BOTTLE | LAPTOP...}')
    print('')
    quit()

output_path = sys.argv[1]
data_sets = []
today = date.today()
for i in range(int((argc - 2) / 4)):
    data_sets.append({
        'img_dir': sys.argv[(i * 4 + 2)],
        'vgg': sys.argv[(i * 4 + 2) + 1],
        'category_id': sys.argv[(i * 4 + 2) + 2],
        'category_name': sys.argv[(i * 4 + 2) + 3]
    })

output = {
    "info": {
        "description": "Dataset",
        "url": "https://www.durham.ac.uk",
        "version": "0.1",
        "year": 2020,
        "contributor": "ICG",
        "date_created": today.strftime("%d/%m/%Y")
    },
    "licenses": [{
        "url": "https://www.durham.ac.uk",
        "id": 0,
        "name": "Durham ICG, Research work"
    }],
    "images": [],
    "annotations": [],
    "categories": []
}

img_id = 1
annotation_id = 1
for data_set in data_sets:
    with open(data_set['vgg']) as f:
        data = json.load(f)
    print(f'{data_set["category_name"]}---->>')
    for data_id, data_info in tqdm(data.items()):
        img_name = data_info['filename']
        #print(img_name)
        filename = find(img_name, data_set['img_dir'])

        img = Image.open(filename)
        width, height = img.size
        img_info = {
            "license": 0,
            "file_name": img_name,
            "width": width,
            "height": height,
            "id": img_id
        }
        output["images"].append(img_info)

        regions = data_info['regions']
        polygons = []
        bbox_list = []
        bbox = None
        area = 0
        for _, region in regions.items():
            ptx = region['shape_attributes']['all_points_x']
            pty = region['shape_attributes']['all_points_y']
            pts_zip = list(zip(ptx, pty))
            pts = [p for point in pts_zip for p in point]

            polygon = geometry.Polygon([[x, y] for x, y in pts_zip])
            x, y, max_x, max_y = polygon.bounds
            box_width = max_x - x
            box_height = max_y - y
            bbox = (x, y, box_width, box_height)
            area = area + polygon.area
            polygons.append(pts)
            bbox_list.append(bbox)

            output['annotations'].append({
                "segmentation": pts,
                "iscrowd": 0,
                "image_id": img_id,
                "category_id": int(data_set["category_id"]),
                "id": annotation_id,
                "bbox": bbox,
                "area": area
            })
            annotation_id = annotation_id + 1
        img_id = img_id + 1
    output["categories"].append({
        "supercategory": "xrayimage",
        "id": int(data_set["category_id"]),
        "name": data_set["category_name"]
    })

json_output = json.dumps(output)
with open(output_path, 'w') as output_file:
    output_file.write(json_output)
