import cv2
import numpy as np


def image_format(uploaded_file):
    # Mengembalikan tipe MIME file (contoh: image/png, image/jpeg)
    return uploaded_file.type


def resize_image(image, scale_percent):
    # Menghitung dimensi baru
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)

    # Melakukan resizing citra
    resized = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    return resized


def color_quantization(image, k=8):
    # ANTISIPASI BUG: Jika gambar memiliki saluran Alpha/Transparansi (RGBA)
    # Kita hanya ambil 3 saluran pertama (RGB) agar tidak error saat reshape (-1, 3)
    if image.shape[2] == 4:
        image = image[:, :, :3]

    # Mengubah matriks gambar menjadi baris data piksel
    data = image.reshape((-1, 3))
    data = np.float32(data)

    # Kriteria penghentian iterasi algoritma K-Means
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        10,
        1.0
    )

    # Menjalankan Algoritma K-Means Clustering
    _, label, center = cv2.kmeans(
        data,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    # Mengembalikan format data ke bentuk gambar semula
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(image.shape)

    return result
