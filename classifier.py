import cv2
import numpy as np


def classify_chili(image):
    """Mengklasifikasikan tingkat kematangan cabai berdasarkan dominasi warna

    pada ruang warna HSV.
    """
    # Konversi dari RGB (Streamlit) ke HSV untuk segmentasi warna yang stabil
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # 1. Threshold Batasan Warna Hijau
    green_lower = np.array([35, 50, 50])
    green_upper = np.array([90, 255, 255])

    # 2. Threshold Batasan Warna Kuning
    yellow_lower = np.array([15, 80, 80])
    yellow_upper = np.array([34, 255, 255])

    # 3. Threshold Batasan Warna Merah (Memiliki 2 rentang di ujung hue)
    red_lower1 = np.array([0, 100, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 100, 50])
    red_upper2 = np.array([180, 255, 255])

    # Pembuatan Masking Piksel untuk masing-masing warna
    mask_green = cv2.inRange(hsv, green_lower, green_upper)
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.add(mask_red1, mask_red2)

    # Hitung jumlah piksel yang aktif (berwarna putih pada mask)
    green_pixel = cv2.countNonZero(mask_green)
    yellow_pixel = cv2.countNonZero(mask_yellow)
    red_pixel = cv2.countNonZero(mask_red)

    # Total piksel cabai yang terdeteksi warna dasarnya
    total = green_pixel + yellow_pixel + red_pixel

    # Antisipasi jika tidak ada piksel cabai (misal gambar latar belakang polos)
    if total == 0:
        return "Tidak Terdeteksi", 0.0, 0.0, 0.0

    # Kalkulasi persentase masing-masing warna
    green_percent = (green_pixel / total) * 100
    yellow_percent = (yellow_pixel / total) * 100
    red_percent = (red_pixel / total) * 100

    # Logika Penentuan Status Kematangan yang Lebih Akurat (Mencari Nilai Terbesar)
    max_percent = max(red_percent, yellow_percent, green_percent)

    if max_percent == red_percent:
        status = "🔴 Matang"
    elif max_percent == yellow_percent:
        status = "🟡 Setengah Matang"
    else:
        status = "🟢 Belum Matang"

    return status, green_percent, yellow_percent, red_percent
