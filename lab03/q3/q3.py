import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

"""
3) Repita a questão anterior, mas dessa vez a imagem resultante deve ter uma coloração
sépia.
"""

original = cv2.imread(sys.argv[1])
img = original.copy()

# Kernel Sépia
sepia_kernel = np.array([[0.272, 0.534, 0.131],
		                [0.349, 0.686, 0.168],
		                [0.393, 0.769, 0.189]])

# Convolução da imagem com o kernel
sepia = cv2.transform(img, sepia_kernel)

_, axes = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)

# Primeiro gráfico (linha 1, coluna 1)
axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
axes[0].set_title('Imagem Original')

# Segundo gráfico (linha 1, coluna 2)
axes[1].imshow(cv2.cvtColor(sepia, cv2.COLOR_BGR2RGB))
axes[1].set_title('Imagem Sépia')

# Ajusta o layout para evitar sobreposição de títulos e imagens
plt.tight_layout()
plt.show()

