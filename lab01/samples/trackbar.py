import cv2
import numpy as np
import sys

def do_nothing(x):
    pass

cv2.namedWindow('imagem')
cv2.createTrackbar('Lower Hue', 'imagem', 0, 180, do_nothing)
cv2.createTrackbar('Upper Hue', 'imagem', 0, 180, do_nothing)

img = cv2.imread(sys.argv[1])
im_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

while(True):

    lower_hue_trackbar = cv2.getTrackbarPos('Lower Hue', 'imagem')
    upper_hue_trackbar = cv2.getTrackbarPos('Upper Hue', 'imagem')

    #https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#object-tracking
    lower_val = np.array([lower_hue_trackbar, 50, 50])
    upper_val = np.array([lower_hue_trackbar, 255, 255])

    mask = cv2.inRange(im_hsv, lower_val, upper_val)
    res = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('imagem', res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()