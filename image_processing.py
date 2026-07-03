import cv2
import numpy as np


def image_information(image):
    # Cek jumlah dimensi matriks gambar
    shape = image.shape
    height = shape[0]
    width = shape[1]
    
    # Jika gambar grayscale, jumlah dimensi hanya 2 (tidak ada channel)
    if len(shape) == 2:
        channel = 1
    else:
        channel = shape[2]

    return {
        "width": width,
        "height": height,
        "channel": channel,
        "pixel": width * height
    }


def rgb_to_grayscale(image):
    # Sudah benar karena input dari Streamlit berbasis RGB
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
