import cv2
import numpy as np

# Variável global para armazenar os pontos e o status da seleção
start_point = None
end_point = None
drawing = False

# Função para capturar o clique e arrastar
def cb_select_area(event, x, y, flags, param):
    global start_point, end_point, drawing
    img_original = param.copy()
    # Quando o botão esquerdo do mouse é pressionado, o desenho começa
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        drawing = True

    # Quando o mouse é movido e o desenho está ativo
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Desenhar retângulo temporário na imagem
            copy = img_original.copy()
            end_point = (x, y)
            cv2.rectangle(copy, start_point, end_point, (0, 255, 0), 2)
            cv2.imshow('image', copy)

    # Quando o botão esquerdo é solto, o desenho termina
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        # Desenhar o retângulo final
        cv2.rectangle(img_original, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow('image', img_original)

# Carregar a imagem e fazer uma cópia para restaurar durante a interação
img = cv2.imread('../imagens/bolas.jpg')
img_copy = img.copy()

# Exibir a imagem e esperar a interação
cv2.imshow('image', img)
cv2.setMouseCallback('image', cb_select_area, img_copy)
# Aguarda a tecla ESC para sair
cv2.waitKey(0)
cv2.destroyAllWindows()

# Criar a máscara da área selecionada
mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.rectangle(mask, start_point, end_point, 255, -1)  # Preenche a área do retângulo
cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
# print(f"Área selecionada de {start_point} até {end_point}. Máscara salva como 'mask.jpg'.")
