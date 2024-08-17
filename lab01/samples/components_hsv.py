import numpy as np
import matplotlib.pyplot as plt
import cv2

# Cria uma grade de cores para o componente H (matiz)
h_values = np.arange(0, 180, 1)
s_value = 255
v_value = 255
hsv_colors_h = np.zeros((len(h_values), 1, 3), dtype=np.uint8)
hsv_colors_h[:, 0, 0] = h_values
hsv_colors_h[:, 0, 1] = s_value
hsv_colors_h[:, 0, 2] = v_value
rgb_colors_h = cv2.cvtColor(hsv_colors_h, cv2.COLOR_HSV2RGB)

# Cria uma grade de cores para o componente S (saturação)
h_value = 0
s_values = np.arange(0, 256, 1)
hsv_colors_s = np.zeros((len(s_values), 1, 3), dtype=np.uint8)
hsv_colors_s[:, 0, 0] = h_value
hsv_colors_s[:, 0, 1] = s_values
hsv_colors_s[:, 0, 2] = v_value
rgb_colors_s = cv2.cvtColor(hsv_colors_s, cv2.COLOR_HSV2RGB)

# Cria uma grade de cores para o componente V (valor)
h_value = 0
s_value = 255
v_values = np.arange(0, 256, 1)
hsv_colors_v = np.zeros((len(v_values), 1, 3), dtype=np.uint8)
hsv_colors_v[:, 0, 0] = h_value
hsv_colors_v[:, 0, 1] = s_value
hsv_colors_v[:, 0, 2] = v_values
rgb_colors_v = cv2.cvtColor(hsv_colors_v, cv2.COLOR_HSV2RGB)

# Plotagem
plt.figure(figsize=(12, 4))

# Componente H
plt.subplot(1, 3, 1)
plt.imshow(rgb_colors_h)
plt.title('Matiz (H)')
plt.axis('off')

# Componente S
plt.subplot(1, 3, 2)
plt.imshow(rgb_colors_s)
plt.title('Saturação (S)')
plt.axis('off')

# Componente V
plt.subplot(1, 3, 3)
plt.imshow(rgb_colors_v)
plt.title('Valor (V)')
plt.axis('off')

plt.show()
