from PIL import Image


def resize_image(root, image_list, dest, modifier):
    files = open(image_list, 'r').readlines()

    for i, f in enumerate(files):
        f = f.rstrip()
        print(f'{i+1} out of {len(files)}')
        img = Image.open(root + f)

        width, height = 512*modifier, 512
        img = img.resize((width, height), Image.ANTIALIAS)
        img.save(dest + f)


resize_image(
    root = '../unpaired/original_size/masks/threat_mask/',
    image_list = '../unpaired/helper/file_lists/composed_firearm_knife.txt',
    dest = '../unpaired/resized_512x512/masks/threat_mask/',
    modifier = 2
)
