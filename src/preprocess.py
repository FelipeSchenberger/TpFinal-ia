import cv2
import numpy as np
import os

def image_to_vector(path, size=(63, 63), visualizar=True):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró la imagen: {path}")

    img_color = cv2.imread(path)
    if img_color is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    img_resized = cv2.resize(img_gray, size)

    _, img_bin = cv2.threshold(img_resized, 127, 255, cv2.THRESH_BINARY)

    if visualizar:
        cv2.imshow("1 - Original", img_color)
        cv2.imshow("2 - Escala de grises", img_gray)
        cv2.imshow(f"3 - Redimensionada {size}", img_resized)
        cv2.imshow("4 - Binarizada (0/1)", img_bin)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    img_bin = (img_bin / 255).astype(np.uint8)
    return img_bin.flatten()
