import sys
import cv2
import numpy as np

# Carrega a imagem
img = cv2.imread(sys.argv[1])

# Converte BGR para HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define a faixa para as cores azul e verde
# Referência: usando o trackbar mostrado em sala
lower_blue = np.array([80, 50, 50])
upper_blue = np.array([112, 255, 255])
lower_green = np.array([10, 50, 50])
upper_green = np.array([75, 255, 255])

# Criando máscaras para as cores azul e verde
mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

# Trocando azul para verde (Hue para verde é em torno de 60)
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#how-to-find-hsv-values-to-track
img_hsv[:,:,0] = np.where(mask_blue > 0, 60, img_hsv[:,:,0])

# Trocando verde para azul (Hue para azul é em torno de 120)
img_hsv[:,:,0] = np.where(mask_green > 0, 120, img_hsv[:,:,0])

# Converte HSV para BGR
img_swap = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

cv2.imwrite('lab1_mask_result.jpg', img_swap)

# Mostre a imagem
cv2.imshow('Imagem nova', img_swap)
cv2.waitKey(0)
cv2.destroyAllWindows()