import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from skimage import filters
from skimage import color

def calculate_energy(image, orientation, mask=None):
    """
    Calcula o mapa de energia da imagem com base na orientação escolhida.

    Parâmetros:
    - image: Imagem de entrada (em RGB).
    - orientation: Pode ser 'horizontal', 'vertical' ou 'both' para calcular a energia.
    - mask: Máscara opcional para diminuir a energia em certas regiões.
    Retorno:
    - energy: O mapa de energia da imagem.
    """
    gray_image = color.rgb2gray(image)
    
    # Calcular o gradiente horizontal e vertical
    if orientation == 'vertical':
        # Sobel horizontal detecta bordas verticais (mudanças no eixo Y)
        energy = np.abs(filters.sobel_h(gray_image))
    elif orientation == 'horizontal':
        # Sobel vertical detecta bordas horizontais (mudanças no eixo X)
        energy = np.abs(filters.sobel_v(gray_image))
    elif orientation == 'both':
        # Combinação de gradientes horizontais e verticais
        sobel_h = np.abs(filters.sobel_h(gray_image))
        sobel_v = np.abs(filters.sobel_v(gray_image))
        energy = sobel_h + sobel_v
    else:
        raise ValueError("Orientation must be 'horizontal', 'vertical', or 'both'")
    
    # Diminuir a energia dentro da máscara
    if mask is not None:
        energy[mask == 255] -= 1000

    return energy

def find_seam(energy, orientation):
    """
    Encontra o seam de menor energia na direção especificada (horizontal ou vertical).
    
    Parâmetros:
    - energy: O mapa de energia da imagem.
    - orientation: 'vertical' para encontrar seams verticais, 'horizontal' para seams horizontais.
    
    Retorno:
    - M: Matriz de energia acumulada.
    - backtrack: Matriz para rastrear o caminho de menor energia.
    """
    if orientation == 'vertical':
        r, c = energy.shape
        M = energy.copy()
        backtrack = np.zeros_like(M, dtype=int)

        # Preenchendo a matriz de energia acumulada (verticalmente)
        for i in range(1, r):
            for j in range(c):
                # Bordas são tratadas separadamente
                if j == 0:
                    idx = np.argmin(M[i-1, j:j+2])
                    backtrack[i, j] = idx + j
                    min_energy = M[i-1, idx + j]
                else:
                    idx = np.argmin(M[i-1, j-1:j+2])
                    backtrack[i, j] = idx + j - 1
                    min_energy = M[i-1, idx + j - 1]
                M[i, j] += min_energy

    elif orientation == 'horizontal':
        r, c = energy.shape
        M = energy.copy()
        backtrack = np.zeros_like(M, dtype=int)

        # Transpor a matriz para tratar horizontalmente
        for j in range(1, c):
            for i in range(r):
                # Bordas são tratadas separadamente
                if i == 0:
                    idx = np.argmin(M[i:i+2, j-1])
                    backtrack[i, j] = idx + i
                    min_energy = M[idx + i, j-1]
                else:
                    idx = np.argmin(M[i-1:i+2, j-1])
                    backtrack[i, j] = idx + i - 1
                    min_energy = M[idx + i - 1, j-1]
                M[i, j] += min_energy

    else:
        raise ValueError("Orientation must be 'vertical' or 'horizontal'")

    return M, backtrack

def remove_seam(image, backtrack, orientation, mask=None):
    """
    Remove o seam de menor energia da imagem na direção especificada (horizontal ou vertical).

    Parâmetros:
    - image: Imagem de entrada (em RGB).
    - backtrack: Matriz de rastreamento do caminho de menor energia.
    - orientation: 'vertical' para remover seams verticais, 'horizontal' para seams horizontais.
    - mask: Máscara opcional para diminuir a energia em certas regiões.
    Retorno:
    - output: Imagem com o seam removido.
    """
    if orientation == 'vertical':
        r, c, _ = image.shape
        output = np.zeros((r, c - 1, 3), dtype=image.dtype)
        new_mask = np.zeros((r, c - 1), dtype=mask.dtype) if mask is not None else None
        j = np.argmin(backtrack[-1])
        for i in reversed(range(r)):
            output[i, :, 0] = np.delete(image[i, :, 0], [j])
            output[i, :, 1] = np.delete(image[i, :, 1], [j])
            output[i, :, 2] = np.delete(image[i, :, 2], [j])
            if mask is not None:
                new_mask[i, :] = np.delete(mask[i, :], [j])
            j = backtrack[i, j]

    elif orientation == 'horizontal':
        r, c, _ = image.shape
        output = np.zeros((r - 1, c, 3), dtype=image.dtype)
        new_mask = np.zeros((r - 1, c), dtype=mask.dtype) if mask is not None else None
        i = np.argmin(backtrack[:, -1])
        for j in reversed(range(c)):
            output[:, j, 0] = np.delete(image[:, j, 0], [i])
            output[:, j, 1] = np.delete(image[:, j, 1], [i])
            output[:, j, 2] = np.delete(image[:, j, 2], [i])
            if mask is not None:
                new_mask[:, j] = np.delete(mask[:, j], [i])
            i = backtrack[i, j]

    else:
        raise ValueError("Orientation must be 'vertical' or 'horizontal'")
    return output, new_mask

def seam_carving(image, num_seams, orientation, mask=None): 
    """
    Realiza o Seam Carving para reduzir a imagem de acordo com a orientação.

    Parâmetros:
    - image: Imagem de entrada.
    - num_seams: Número de seams a serem removidos.
    - orientation: 'vertical', 'horizontal' ou 'both'.
    - mask: Máscara opcional para diminuir a energia em certas regiões.
    
    Retorno:
    - A imagem redimensionada após a remoção dos seams.
    """
    for _ in range(num_seams):
        if orientation == 'vertical':
            energy = calculate_energy(image, 'vertical', mask)
            M, backtrack = find_seam(energy, 'vertical')
            image, mask = remove_seam(image, backtrack, 'vertical', mask)
        elif orientation == 'horizontal':
            energy = calculate_energy(image, 'horizontal', mask)
            M, backtrack = find_seam(energy, 'horizontal')
            image, mask = remove_seam(image, backtrack, 'horizontal', mask)
        elif orientation == 'both':
            # Alternar entre remover seams verticais e horizontais
            if _ % 2 == 0:
                energy = calculate_energy(image, 'vertical', mask)
                M, backtrack = find_seam(energy, 'vertical')
                image, mask = remove_seam(image, backtrack, 'vertical', mask)
            else:
                energy = calculate_energy(image, 'horizontal', mask)
                M, backtrack = find_seam(energy, 'horizontal')
                image, mask = remove_seam(image, backtrack, 'horizontal', mask)
        
        else:
            raise ValueError("Orientation must be 'vertical', 'horizontal', or 'both'")

    return image

if __name__ == '__main__':
    
    # Carregar a imagem
    img = io.imread('../imagens/bolas.jpg')

    num_seams = 60

    vertical_seams = seam_carving(img, num_seams, 'vertical')
    horizontal_seams = seam_carving(img, num_seams, 'horizontal')
    both_seams = seam_carving(img, num_seams+40, 'both')

    # Mostrar as imagens resultantes
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Imagem Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(vertical_seams)
    axes[0, 1].set_title('Imagem Redimensionada (Vertical)')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(horizontal_seams)
    axes[1, 0].set_title('Imagem Redimensionada (Horizontal)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(both_seams)
    axes[1, 1].set_title('Imagem Redimensionada (Ambas)')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()