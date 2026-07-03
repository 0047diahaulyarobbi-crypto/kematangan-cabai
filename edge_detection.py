import cv2
import numpy as np


def sobel_edge(image):
    """Deteksi tepi menggunakan algoritma Sobel."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    return sobel


def prewitt_edge(image):
    """Deteksi tepi menggunakan algoritma Prewitt."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float64)
    prewitt_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
    prewitt_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
    prewitt_x = np.abs(prewitt_x)
    prewitt_y = np.abs(prewitt_y)
    prewitt = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)
    prewitt = np.uint8(np.clip(prewitt, 0, 255))
    return prewitt
