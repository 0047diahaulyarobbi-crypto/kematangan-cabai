import cv2
import numpy as np


def segment_chili(image):
    """
    Segmentasi warna cabai menggunakan HSV
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Merah bagian bawah
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])

    # Merah bagian atas
    lower_red2 = np.array([170, 100, 50])
    upper_red2 = np.array([180, 255, 255])

    # Hijau
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])

    # Kuning / Oranye
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([34, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2

    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask = mask_red + mask_green + mask_yellow

    result = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    return result, mask