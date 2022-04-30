import cv2 as cv
import os
import numpy as np

root = '../unpaired/original_size/composed_images/'

heights = []
widths = []

files = os.listdir(root)
for i, file in enumerate(files):
    print(f'{i} out of {len(files)}')
    img = cv.imread(root+file)

    height, width, _ = img.shape

    heights.append(height)
    widths.append(width)


heights = np.array(heights)
widths = np.array(widths)

print(f'height avg: {heights.mean()} max: {max(heights)} mean: {np.median(heights)}')
print(f'width avg: {widths.mean()} max: {max(widths)} file: {np.median(widths)}')