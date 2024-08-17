import cv2 
import numpy as np
import sys

# Lab 1: Trocar a cor da pel de Gamora (verde) com a cor da Nebulosa (azul)

img = cv2.imread(sys.argv[1])
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img_h, img_s, img_v = cv2.split(img_hsv)

width = img.shape[1]
height = img.shape[0]

# Modificando o pixel
for c in range(0, width-1):
    for l in range(0, height-1):
        if img_h[l][c] >= 10 and img_h[l][c] <= 75:
            img_h[l][c] += 65
        elif img_h[l][c] >= 80 and img_h[l][c] <= 112:
            img_h[l][c] -= 40


img_hsv = cv2.merge([img_h, img_s, img_v])
img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

cv2.imwrite('lab1_matrix_result.jpg', img)

cv2.imshow("Imagem nova", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Verde: entre 15 e 75
# Azul: entre 80 e 112




