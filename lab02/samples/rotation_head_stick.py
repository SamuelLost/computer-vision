import cv2
import numpy as np
import cv_utils as waitKey
import sys

# Carrega a imagem da cabeça do "boneco palito"
circle = cv2.imread(sys.argv[1])

# Calcula a largura e altura da imagem
width_circle = circle.shape[1]
height_circle = circle.shape[0]

# Define o fator de escala
scale_factor = 300 / max(width_circle, height_circle)

# Calcula a matriz de rotação usando cv2.getRotationMatrix2D
rotation_matrix = cv2.getRotationMatrix2D((width_circle / 2, height_circle / 2), 0, 1)

print(rotation_matrix)
# Aplica a matriz de rotação à imagem
head_scaled = cv2.warpAffine(circle, rotation_matrix, (300, 300))

# Exibe a imagem resultante
cv2.imshow('Scaled Head', head_scaled)
waitKey.waitKey('Scaled Head', 27)
