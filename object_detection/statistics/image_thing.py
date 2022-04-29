import cv2 as cv

img = cv.imread('./test.png')
h = cv.hconcat([img, img])
v = cv.vconcat([h, h])

cv.imwrite('./test_vconcat.png', v)