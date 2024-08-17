import sys
import cv2
import numpy as np

from cv_utils import waitKey

# Deixar a imagem jato.jpg mais amarela com correção de gamma

def gamma_correction(img, gamma,c=1.0):
    i = 255*(c*(img/255.0)**(1.0 / gamma))
    return i

def gamma_correction_yellow(img, gamma):
    i = img.copy()
    i_g, i_b, i_r = cv2.split(i)
    # i_r[:,:] = 255*(1.2*(i_r[:,:]/255.0)**(1.0 / gamma)) # Aplica a correção de gamma no vermelho
    # i_g[:,:] = 255*(1.2*(i_g[:,:]/255.0)**(1.0 / gamma)) # Aplica a correção de gamma no verde
    # i_b[:,:] = 255*(0.4*(i_b[:,:]/255.0)**(1.0 / gamma)) # Aplica a correção de gamma no azul
    i_r[:,:] = gamma_correction(i_r[:,:], gamma, 1.2) # Aplica a correção de gamma no vermelho
    i_g[:,:] = gamma_correction(i_g[:,:], gamma, 1.2) # Aplica a correção de gamma no verde
    i_b[:,:] = gamma_correction(i_b[:,:], gamma, 0.4) # Aplica a correção de gamma no azul
    i = cv2.merge((i_b, i_g, i_r))
    return i

im = cv2.imread(sys.argv[1])

im_amarelada = gamma_correction_yellow(im, 0.5)

cv2.imwrite('jato_resultado.jpg', im_amarelada)

cv2.imshow('Jato Amarelado',im_amarelada)
waitKey('Jato Amarelado', 27) #27 = ESC