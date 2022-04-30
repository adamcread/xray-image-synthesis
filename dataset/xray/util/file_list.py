# composed_firearm.txt
# composed_knife.txt
# composed_firearm_knife.txt

# decomposed_firearm.txt
# decomposed_knife.txt
# decomposed_bags.txt

import os

def file_list(root, classes, dest):
    if classes:
        root_dirs = os.listdir(root)
        files = []
        for dir in root_dirs:
            if dir in classes:
                files.extend(os.listdir(root+dir))
                # files.extend(list(map(lambda x: dir+'/'+x, os.listdir(root+dir))))
    else:
        files = os.listdir(root)

    with open(dest, 'w+') as fp:
        fp.write('\n'.join(files))


file_list(
    root = '../unpaired/resized_128x128/composed_images/',
    dest = '../unpaired/helper/file_lists/composed_firearm_knife.txt',
    classes = []
)

