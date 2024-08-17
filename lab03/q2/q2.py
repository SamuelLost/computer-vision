import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

"""
Escolha uma imagem qualquer colorida e aplique um ou mais filtros convolucionais, de
forma a resultar em uma imagem em tons de cinza.
Considere a fórmula abaixo, onde Y é o valor do pixel em tons de cinza e R, G e B
correspondem aos valores dos pixels nos canais Vermelho, Verde e Azul, respectivamente.

Y = (0,3*R + 0,59*G + 0,11*B)
"""

# Carrega a imagem
img = cv2.imread(sys.argv[1])

# Kernel de convolução
kernel_r = np.array([[0, 0, 0],
                     [0, 0.3, 0],
                     [0, 0, 0]])

kernel_g = np.array([[0, 0, 0],
                     [0, 0.59, 0],
                     [0, 0, 0]])

kernel_b = np.array([[0, 0, 0],
                     [0, 0.11, 0],
                     [0, 0, 0]])

conv_r = cv2.filter2D(img[:, :, 2], -1, kernel_r) # Convolução no canal R
conv_g = cv2.filter2D(img[:, :, 1], -1, kernel_g) # Convolução no canal G
conv_b = cv2.filter2D(img[:, :, 0], -1, kernel_b) # Convolução no canal B

conv_result = conv_r + conv_g + conv_b # Soma das convoluções

fig, axes = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)

# Primeiro gráfico
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Imagem Original')
axes[0].axis('off')

# Segundo gráfico
axes[1].imshow(conv_result, cmap='gray')
axes[1].set_title('Convolução com filter2D')
axes[1].axis('off')

# Ajusta o layout para evitar sobreposição de títulos e imagens
plt.tight_layout()

plt.show()
