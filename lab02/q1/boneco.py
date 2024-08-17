import sys
import cv2
import numpy as np
import cv_utils as waitKey

# Lab 2 - Questão 2 -
# Utilizando como base as figuras ‘circle.jpg’ e ‘line.jpg’, forme o desenho de um “boneco
# palito” aplicando uma sequência de transformações geométricas e operações lógicas nas
# imagens, seguindo as regras abaixo.
# – A figura resultante deve ter um tamanho de 300x300.
# – Use cópias da figura ‘line.jpg’ para os braços, pernas e tronco do boneco.
# – Não redimensione as imagens para criar o tronco e a cabeça.
# – Cada braço deve ter 75% do tamanho do tronco.
# – As pernas devem estar em um ângulo de 90º entre si e devem ter o dobro do tamanho dos
# braços.
# – Posicione o boneco no centro da imagem.

circle = cv2.imread(sys.argv[1])
line = cv2.imread(sys.argv[2])

height_circle, width_circle = circle.shape[:2]
height_line, width_line = line.shape[:2]

dimensions = (300, 300)

# Escalas
scale_head = np.float32([[1, 0, 0], [0, 1, 0]])  # Escala do círculo. Mantendo o tamanho original
scale_body = np.float32([[1, 0, 0], [0, 1, 0]])  # Escala da linha. Mantendo o tamanho original
scale_arm = np.float32([[0.75, 0, 0], [0, 1, 0]])  # Escala do braço. 75% do tamanho do tronco. Largura do braço foi mantida, o comprimento do braço contém 75% do comprimento do tronco
scale_leg = np.float32([[1.5, 0, 0], [0, 1.5, 0]])  # Escala da perna. Dobro do tamanho dos braços

# Cabeça
head = cv2.bitwise_not(circle) # Invertendo as cores da imagem
head = cv2.warpAffine(head, scale_head, dimensions)  # Não modifica a imagem, apenas redimensiona para 300x300
M_translation_c = np.float32([[1, 0, 100], [0, 1, 10]]) # Posicionando a cabeça no centro da imagem, X=100, Y=10.
# A imagem do circulo é um quadrado 100x100. Para centralizar a imagem dentro de um quadrado 300x300, é necessário deslocar 100 pixels para a direita.

im_translated_c = cv2.warpAffine(head, M_translation_c, dimensions) # Deslocando a imagem para a posição correta
head = cv2.bitwise_not(im_translated_c) # Invertendo as cores da imagem

# Tronco
body = cv2.bitwise_not(line) # Invertendo as cores da imagem
body = cv2.warpAffine(body, scale_body, dimensions) # Não modifica a imagem, apenas redimensiona para 300x300
M_rotation_l = cv2.getRotationMatrix2D((width_line / 2, height_line / 2), 90, 1) # Rotacionando a linha em 90 graus para ficar na vertical
body = cv2.warpAffine(body, M_rotation_l, dimensions) # Rotacionando a linha
M_translation_l = np.float32([[1, 0, 100], [0, 1, 72]]) # Posicinando o tronco abaixo da cabeça, X=100, Y=72
body = cv2.warpAffine(body, M_translation_l, dimensions) # Deslocando a imagem para a posição correta
body = cv2.bitwise_not(body) # Invertendo as cores da imagem

# Braço esquerdo - 75% do tamanho do tronco
left_arm = cv2.bitwise_not(line) # Invertendo as cores da imagem
left_arm = cv2.warpAffine(left_arm, scale_arm, dimensions) # Redimensionando a imagem para 300x300
# M_rotation_arm_left = cv2.getRotationMatrix2D((width_line / 2, height_line / 2), 0, 1) # Rotacionando a linha em 0 graus
# left_arm = cv2.warpAffine(left_arm, M_rotation_arm_left, dimensions) # Rotacionando a linha
# Não é necessário rotacionar o braço, pois a imagem da linha já está na posição correta

M_translation_arm_left = np.float32([[1, 0, 79], [0, 1, 65]]) # Posicinando o braço esquerdo abaixo da cabeça, X=79, Y=65
left_arm = cv2.warpAffine(left_arm, M_translation_arm_left, dimensions) # Deslocando a imagem para a posição correta
left_arm = cv2.bitwise_not(left_arm) # Invertendo as cores da imagem

# Braço direito - 75% do tamanho do tronco
right_arm = cv2.bitwise_not(line) # Invertendo as cores da imagem 
right_arm = cv2.warpAffine(right_arm, scale_arm, dimensions) # Redimensionando a imagem para 300x300
# M_rotation_arm_right = cv2.getRotationMatrix2D((width_line / 2, height_line / 2), 0, 1) # Rotacionando a linha em 0 graus
# right_arm = cv2.warpAffine(right_arm, M_rotation_arm_right, dimensions) # Rotacionando a linha
# Não é necessário rotacionar o braço, pois a imagem da linha já está na posição correta

M_translation_arm_right = np.float32([[1, 0, 145], [0, 1, 65]]) # Posicinando o braço direito abaixo da cabeça, X=145, Y=65
right_arm = cv2.warpAffine(right_arm, M_translation_arm_right, dimensions) # Deslocando a imagem para a posição correta
right_arm = cv2.bitwise_not(right_arm) # Invertendo as cores da imagem

# Perna esquerda - Dobro do tamanho dos braços
left_leg = cv2.bitwise_not(line) # Invertendo as cores da imagem
left_leg = cv2.warpAffine(left_leg, scale_leg, dimensions) # Redimensionando a imagem para 300x300
M_rotation_leg_left = cv2.getRotationMatrix2D((width_line / 2, height_line / 2), 45, 1) # Rotacionando a linha em 45 graus
left_leg = cv2.warpAffine(left_leg, M_rotation_leg_left, dimensions) # Rotacionando a linha
M_translation_leg_left = np.float32([[1, 0, 22], [0, 1, 156]]) # Posicinando a perna esquerda abaixo do tronco, X=22, Y=156
left_leg = cv2.warpAffine(left_leg, M_translation_leg_left, dimensions) # Deslocando a imagem para a posição correta
left_leg = cv2.bitwise_not(left_leg)

# Perna direita - Dobro do tamanho dos braços
right_leg = cv2.bitwise_not(line) # Invertendo as cores da imagem
right_leg = cv2.warpAffine(right_leg, scale_leg, dimensions) # Redimensionando a imagem para 300x300
M_rotation_leg_right = cv2.getRotationMatrix2D((width_line / 2, height_line / 2), -45, 1) # Rotacionando a linha em -45 graus
right_leg = cv2.warpAffine(right_leg, M_rotation_leg_right, dimensions) # Rotacionando a linha
M_translation_leg_right = np.float32([[1, 0, 142], [0, 1, 122]]) # Posicinando a perna direita abaixo do tronco, X=142, Y=122
right_leg = cv2.warpAffine(right_leg, M_translation_leg_right, dimensions) # Deslocando a imagem para a posição correta
right_leg = cv2.bitwise_not(right_leg)

img = cv2.bitwise_and(head, body) # Cabeça e tronco
img = cv2.bitwise_and(img, left_arm) # Cabeça, tronco e braço esquerdo
img = cv2.bitwise_and(img, right_arm) # Cabeça, tronco, braço esquerdo e braço direito
img = cv2.bitwise_and(img, left_leg) # Cabeça, tronco, braço esquerdo, braço direito e perna esquerda 
img = cv2.bitwise_and(img, right_leg) # Cabeça, tronco, braço esquerdo, braço direito, perna esquerda e perna direita

cv2.imwrite('boneco_resultado.jpg', img)

cv2.imshow('Boneco de Palito', img)
waitKey.waitKey('Boneco de Palito', 27)  # 27 = ESC
