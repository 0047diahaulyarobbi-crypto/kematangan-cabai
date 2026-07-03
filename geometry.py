import cv2


def rotate_image(image, angle):
    """
    Rotasi gambar
    """

    height, width = image.shape[:2]

    center = (width // 2, height // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height)
    )

    return rotated


def flip_horizontal(image):
    """
    Flip Horizontal
    """

    return cv2.flip(image, 1)


def flip_vertical(image):
    """
    Flip Vertical
    """

    return cv2.flip(image, 0)


def negative_image(image):
    """
    Invers warna
    """

    return 255 - image


def crop_center(image):
    """
    Crop bagian tengah gambar
    """

    h, w = image.shape[:2]

    x1 = int(w * 0.25)
    x2 = int(w * 0.75)

    y1 = int(h * 0.25)
    y2 = int(h * 0.75)

    return image[y1:y2, x1:x2]