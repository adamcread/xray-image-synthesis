import os
import shutil

bags = [
    'bag-2/',
    'bag-16/',
    'bag-28/',
    'bag-32/',
    'bag-36/',
    'bag-47/'
]


category_files = {
    'side-view/': [
        'left-top',
        'left-side',
        'right-top',
        'right-side',
        'end-top',
        'end-side'
    ],
    'top-view/': [
        'up-top',
        'up-side',
        'down-top',
        'down-side',
        'left-top',
        'left-side',
        'right-top',
        'right-side'
    ]
}

root = "../paired/negative-bags/"
for bag in bags:
    if not os.path.exists(root+bag):
        os.mkdir(root+bag)
    

    for view in ['side-view/', 'top-view/']:
        if not os.path.exists(root+bag+view):
            os.mkdir(root+bag+view)
        
        for category_file in category_files[view]:
            if not os.path.exists(root+bag+view+category_file):
                os.mkdir(root+bag+view+category_file)

        for file in os.listdir(root+view+bag):
            shutil.move(root+view+bag+file, root+bag+view+file)
        
        
