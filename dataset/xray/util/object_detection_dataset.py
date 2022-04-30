import os
from pycocotools.coco import COCO
from file_list import file_list
from filter_coco import CocoFilter
import random
import shutil
from coco_merger import merger

def move_files(fake_root, real_root, real_json, classes, fake_count, real_count, dest):
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.mkdir(dest)

    coco = COCO(annotation_file=real_json)
    fake_images = os.listdir(fake_root)

    for class_name in classes:
        class_filtered = [image for image in fake_images if class_name in image]
        class_filtered = random.sample(class_filtered, k=len(class_filtered))
        for i in range(0, fake_count//len(classes)):
            print(class_name, i+1)
            random_file = class_filtered.pop()
            shutil.copy(fake_root+random_file, dest+random_file)
    
    real_images = [x['file_name'] for x in coco.loadImgs(coco.getImgIds())]
    print(len(real_images))
    real_shuffled = random.sample(real_images, k=real_count)
    for i, random_file in enumerate(real_shuffled):
        print('real', i+1)
        shutil.copy(real_root+random_file, dest+random_file)


def main(total_real, test_name, fake_count, real_count):
    fake_amount = total_real*fake_count // 100
    real_amount = total_real*real_count // 100

    move_files(
        fake_root = f'../composed/0_real_20000_fake/{test_name}/',
        real_root = '../unpaired/resized_128x128/composed_images/',
        real_json = '../unpaired/helper/annotation/dbf3_train_resized.json',
        classes = ['knife', 'firearm'],
        fake_count=fake_amount,
        real_count=real_amount,
        dest= f'../composed/{real_count}_real_{fake_count}_fake/{test_name}/'
    )

    file_list(
        root = f'../composed/{real_count}_real_{fake_count}_fake/{test_name}/',
        dest = f'../composed/{real_count}_real_{fake_count}_fake/helper/file_lists/{test_name}.txt',
        classes = []
    )

    class FilterObj:
        def __init__(self, input_json, output_json, categories, filter_list):
            self.input_json = input_json
            self.output_json = output_json
            self.categories = categories
            self.filter_list = filter_list

    fake_args = FilterObj(
        input_json = f'../composed/0_real_20000_fake/helper/annotation/{test_name}.json',
        output_json = f'../composed/{real_count}_real_{fake_count}_fake/helper/annotation/{test_name}_fake.json',
        categories = ['FIREARM', 'KNIFE'],
        filter_list = f'../composed/{real_count}_real_{fake_count}_fake/helper/file_lists/{test_name}.txt'
    )

    real_args = FilterObj(
        input_json = '../unpaired/helper/annotation/dbf3_train_resized.json',
        output_json = f'../composed/{real_count}_real_{fake_count}_fake/helper/annotation/{test_name}_real.json',
        categories = ['FIREARM', 'KNIFE'],
        filter_list = f'../composed/{real_count}_real_{fake_count}_fake/helper/file_lists/{test_name}.txt'
    )

    cf = CocoFilter()
    cf.main(fake_args)
    cf.main(real_args)

    if real_count > 0:
        merger(
            json_file_1 =f'../composed/{real_count}_real_{fake_count}_fake/helper/annotation/{test_name}_fake.json', 
            json_file_2 =f'../composed/{real_count}_real_{fake_count}_fake/helper/annotation/{test_name}_real.json', 
            OUTPUT_FILENAME = f'../composed/{real_count}_real_{fake_count}_fake/helper/annotation/{test_name}.json'
        )

        os.remove(f'../composed/{real_count}_real_{fake_count}_fake/helper/annotation/{test_name}_fake.json')
        os.remove(f'../composed/{real_count}_real_{fake_count}_fake/helper/annotation/{test_name}_real.json')
    else:
        os.remove(f'../composed/{real_count}_real_{fake_count}_fake/helper/annotation/{test_name}_real.json')
        os.rename(
            src = f'../composed/{real_count}_real_{fake_count}_fake/helper/annotation/{test_name}_fake.json',
            dst = f'../composed/{real_count}_real_{fake_count}_fake/helper/annotation/{test_name}.json'
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name')
    parser.add_argument('--real_count')
    parser.add_argument('--fake_count')

    args = parser.parse_args()

    main(
        total_real = 4433,
        test_name = args.test_name,
        real_count = int(args.real_count),
        fake_count = int(args.fake_count)
    )
