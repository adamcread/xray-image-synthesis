import cv2 as cv

root_file = "../test/knife_2261.jpg"

test_image = cv.imread(root_file)
bag = test_image[:128, :128]
firearm = test_image[:128, 256:384]

cv.imwrite("bag.png", bag)
cv.imwrite("firearm.png", firearm)

