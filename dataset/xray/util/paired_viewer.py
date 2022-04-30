import os
import cv2 as cv


root = '../paired/test/'
files = os.listdir(root)
for i, file in enumerate(files):
    print(f'{i} out of {len(files)} - {file}')

    composed_image = cv.imread(root+file+'/composed.png')
    bag_image = cv.imread(root+file+'/decomposed_bag.png')
    threat_image = cv.imread(root+file+'/decomposed_threat.png')
    threat_image_T = cv.imread(root+file+'/decomposed_threat_T.png')

    cv.imwrite('current_composed.png', composed_image)
    cv.imwrite('current_decomposed_bag.png', bag_image)
    cv.imwrite('current_decomposed_threat.png',  threat_image)
    cv.imwrite('current_decomposed_threat_T.png', threat_image_T)

    input()
