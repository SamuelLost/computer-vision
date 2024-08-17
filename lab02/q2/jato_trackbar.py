import sys
import cv2
import numpy as np

from cv_utils import waitKey

# vermelho e verde
def gamma_correction(img, gamma,c=1.0):
   i = img.copy()
   i[0,:,:] = 255*(c*(img[0,:,:]/255.0)**(1.0 / gamma))
   return i


def gamma_correction_LUT(img, gamma,c=1.0):
	GAMMA_LUT = np.array([c*((i / 255.0) ** (1.0 / gamma)) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 	
	# aplica a transformação usando LUT
	return cv2.LUT(img, GAMMA_LUT)
	
def callback_trackbar(x):
    try:
        blue = cv2.getTrackbarPos('blue','image')
        green = cv2.getTrackbarPos('green','image')
        red = cv2.getTrackbarPos('red','image')
        
        im_gamma_b = gamma_correction_LUT(im_b, blue*0.01)
        im_gamma_g = gamma_correction_LUT(im_g, green*0.01)
        im_gamma_r = gamma_correction_LUT(im_r, red*0.01)
        
        im_gamma = cv2.merge([im_gamma_b,im_gamma_g,im_gamma_r])
        cv2.imshow('image',im_gamma)
    except:
        return


#abre imagem
filename = sys.argv[1]
im = cv2.imread(filename)

im_b,im_g,im_r = cv2.split(im)

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.createTrackbar('blue','image',0,100,callback_trackbar)
cv2.createTrackbar('green','image',0,100,callback_trackbar)
cv2.createTrackbar('red','image',0,100,callback_trackbar)


cv2.imshow('image',im)
waitKey('image', 27) #27 = ESC	