import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Import fungsi dari modul lokal Anda
from digitalization import color_quantization, image_format, resize_image
from edge_detection import prewitt_edge, sobel_edge
from geometry import (
    crop_center,
    flip_horizontal,
    flip_vertical,
    negative_image,
    rotate_image,
)
from image_processing import image_information, rgb_to_grayscale
from segmentation import segment_chili
from classifier import classify_chili

# 1. KONFIGURASI HALAMAN STREAMLIT
st.set_page_config(
    page_title="Identifikasi Kematangan Cabai", page_icon="🌶️", layout="wide"
)

# 2. INJEKSI CUSTOM CSS (TEMA BIRU PASTEL & UI MODERN)
st.markdown(
    """
    <style>
        .main {
            background-color: #f4f7f6;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }
        [data-testid="stSidebar"] {
            background-color: #e3edfd !important;
        }
        h1 {
            color: #4a7cbf !important;
            font-weight: 700;
        }
        h2, h3, h4 {
            color: #5c8ecb !important;
            font-weight: 600;
        }
        .pastel-card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.03);
            border-left: 5px solid #a2c2ec;
            margin-bottom: 20px;
        }
        hr {
            margin-top: 1rem;
            margin-bottom: 1rem;
            border: 0;
            border-top: 2px solid #d0e1f9;
        }
        .stProgress > div > div > div > div {
            background-color: #a2c2ec;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# SIDEBAR MENU & KONTROL NAVIGASI
st.sidebar.markdown(
    "<div style='text-align: center;'><img src='https://cdn-icons-png.flaticon.com/512/2909/2909763.png' width='100'></div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<h2 style='text-align: center; color: #3b5c85;'>Menu Aplikasi</h2>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<p style='text-align: center; font-size: 14px; color: #555;'>Laboratorium Pengolahan Citra</p>",
    unsafe_allow_html=True,
)
st.sidebar.write("---")

st.sidebar.markdown(
    """
    <div style='background-color: #ffffff; padding: 15px; border-radius: 10px;'>
        <h5 style='margin-top:0; color:#3b5c85;'>Alur Proses Sistem:</h5>
        <p style='margin-bottom:5px;'>🔹 1. Upload Gambar</p>
        <p style='margin-bottom:5px;'>🔹 2. Citra Asli & Representasi</p>
        <p style='margin-bottom:5px;'>🔹 3. Grayscale</p>
        <p style='margin-bottom:5px;'>🔹 4. Digitalisasi Citra</p>
        <p style='margin-bottom:5px;'>🔹 5. Operasi Geometri</p>
        <p style='margin-bottom:5px;'>🔹 6. Deteksi Tepi</p>
        <p style='margin-bottom:5px;'>🔹 7. Segmentasi Citra</p>
        <p style='margin-bottom:0px;'>🔹 8. Hasil Identifikasi</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# HEADER APLIKASI UTAMA
st.title("🌶️ Implementasi Pengolahan Citra Digital")

# =========================================================================
# PERINTAH 1: UPLOAD GAMBAR
# =========================================================================
st.markdown("### 📥 1. Upload Gambar")
uploaded_file = st.file_uploader(
    "Pilih berkas citra cabai untuk dianalisis", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Konversi awal berkas
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Proteksi saluran Alpha (RGBA to RGB)
    if len(image_np.shape) == 3 and image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]

    # =========================================================================
    # PERINTAH 2: CITRA ASLI & REPRESENTASI CITRA
    # =========================================================================
    st.write("---")
    col_asli, col_rep = st.columns([4, 3])

    with col_asli:
        st.markdown("### 📸 2. Citra Asli")
        st.image(image, caption="Sampel Citra Utama (Original)", use_container_width=True)

    with col_rep:
        st.markdown("### 📊 3. Representasi Citra")
        info = image_information(image_np)
        st.markdown(
            f"""
            <div style='background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #d0e1f9;'>
                <p style='margin-bottom:8px;'><b>Format Data:</b> <code style='color:#3b5c85;'>{image_format(uploaded_file)}</code></p>
                <p style='margin-bottom:8px;'><b>Lebar (Width):</b> <code>{info['width']} px</code></p>
                <p style='margin-bottom:8px;'><b>Tinggi (Height):</b> <code>{info['height']} px</code></p>
                <p style='margin-bottom:8px;'><b>Saluran Warna:</b> <code>{info['channel']} (RGB)</code></p>
                <p style='margin-bottom:0px;'><b>Total Piksel Matriks:</b> <code>{info['pixel']:,} dots</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # PERINTAH 3: GRAYSCALE
    # =========================================================================
    st.write("---")
    st.markdown("### 🔘 4. Transformasi Ruang Warna Monokrom (Grayscale)")
    col_gray1, col_gray2 = st.columns([1, 2])

    with col_gray1:
        st.write(
            "Mengeliminasi elemen warna citra untuk mendapatkan nilai intensitas tunggal. "
            "Tahapan ini mereduksi beban komputasi struktur kontur citra."
        )
        aktifkan_gray = st.checkbox("Aktifkan Pratinjau Grayscale", value=True)

    with col_gray2:
        if aktifkan_gray:
            gray_image = rgb_to_grayscale(image_np)
            st.image(
                gray_image,
                caption="Matriks Citra Keabuan (Grayscale)",
                use_container_width=True,
            )
        else:
            st.info("Aktifkan penanda centang di sisi kiri untuk memuat citra.")

    # =========================================================================
    # PERINTAH 4: DIGITALISASI
    # =========================================================================
    st.write("---")
    st.markdown("### ⚙️ 5. Digitalisasi Citra")
    col_dig1, col_dig2 = st.columns(2)

    with col_dig1:
        st.markdown("#### A. Implementasi Kriteria Sampling")
        scale = st.slider("Ubah Skala Dimensi Spatial (%)", 10, 200, 100, step=10)
        resized = resize_image(image_np, scale)
        st.image(resized, caption=f"Resolusi Setelah Resizing ({scale}%)", use_container_width=True)

    with col_dig2:
        st.markdown("#### B. Kuantisasi Tingkat Warna")
        k = st.slider("Tentukan Klaster Level Warna (K-Means)", 2, 32, 8)
        quantized = color_quantization(image_np, k)
        st.image(quantized, caption=f"Hasil Kompresi Menjadi {k} Warna Dominan", use_container_width=True)

    # =========================================================================
    # PERINTAH 5: OPERASI GEOMETRI
    # =========================================================================
    st.write("---")
    st.markdown("### 📐 6. Eksperimen Operasi Geometri Interaktif")
    col_ctrl, col_view = st.columns([1, 2])
    
    with col_ctrl:
        st.markdown("#### Pengaturan Parameter")
        angle = st.slider("Sudut Rotasi (Derajat)", min_value=0, max_value=360, value=0, step=5)
        opsi_flip = st.selectbox("Jenis Pencerminan (Flip)", ["Tanpa Flip", "Horizontal (Sumbu-Y)", "Vertikal (Sumbu-X)", "Keduanya"])
        aktifkan_negatif = st.checkbox("Terapkan Efek Citra Negatif")
        st.markdown("**Area Pemotongan (Crop %)**")
        crop_range_h = st.slider("Rentang Tinggi (Vertikal)", 0, 100, (0, 100))
        crop_range_w = st.slider("Rentang Lebar (Horizontal)", 0, 100, (0, 100))

    with col_view:
        st.markdown("#### Hasil Manipulasi Geometri")
        img_transformed = image_np.copy()
        
        if angle > 0:
            img_transformed = rotate_image(img_transformed, angle)
            
        if opsi_flip == "Horizontal (Sumbu-Y)":
            img_transformed = flip_horizontal(img_transformed)
        elif opsi_flip == "Vertikal (Sumbu-X)":
            img_transformed = flip_vertical(img_transformed)
        elif opsi_flip == "Keduanya":
            img_transformed = flip_horizontal(img_transformed)
            img_transformed = flip_vertical(img_transformed)
            
        if aktifkan_negatif:
            img_transformed = negative_image(img_transformed)
            
        h_current, w_current = img_transformed.shape[:2]
        start_h = int(crop_range_h[0] * h_current / 100)
        end_h = int(crop_range_h[1] * h_current / 100)
        start_w = int(crop_range_w[0] * w_current / 100)
        end_w = int(crop_range_w[1] * w_current / 100)
        
        if end_h > start_h and end_w > start_w:
            img_transformed = img_transformed[start_h:end_h, start_w:end_w]
            
        st.image(img_transformed, caption="Hasil Kombinasi Operasi Geometri", use_container_width=True)

    # =========================================================================
    # PERINTAH 6: DETEKSI TEPI
    # =========================================================================
    st.write("---")
    st.markdown("### 🔍 7. Deteksi Tepi Objek")
    col_edge1, col_edge2 = st.columns(2)

    with col_edge1:
        sobel = sobel_edge(image_np)
        st.subheader("Metode Sobel")
        st.image(sobel, use_container_width=True)

    with col_edge2:
        prewitt = prewitt_edge(image_np)
        st.subheader("Metode Prewitt")
        st.image(prewitt, use_container_width=True)

    # =========================================================================
    # PERINTAH 7: SEGMENTASI
    # =========================================================================
    st.write("---")
    st.markdown("### 🎯 8. Segmentasi Citra Cabai")
    segmented, mask = segment_chili(image_np)
    col_seg1, col_seg2 = st.columns(2)

    with col_seg1:
        st.subheader("Masking Hasil Segmentasi")
        st.image(mask, use_container_width=True)

    with col_seg2:
        st.subheader("Ekstraksi Objek Tersegmentasi")
        st.image(segmented, use_container_width=True)

    # =========================================================================
    # PERINTAH 8: HASIL IDENTIFIKASI (KLASIFIKASI)
    # =========================================================================
    st.write("---")
    st.markdown("### 📊 9. Hasil Identifikasi & Klasifikasi Akhir")
    status, green, yellow, red = classify_chili(image_np)

    # Alert Box Utama berdasarkan klasifikasi
    if "Matang" in status and "Setengah" not in status and "Belum" not in status:
        st.markdown(f"<div style='background-color: #ffebe6; padding: 20px; border-radius: 10px; border: 2px solid #ff4d4d; text-align: center;'><h2 style='color: #cc0000; margin: 0;'>Status: {status}</h2></div>", unsafe_allow_html=True)
    elif "Setengah" in status:
        st.markdown(f"<div style='background-color: #fff9e6; padding: 20px; border-radius: 10px; border: 2px solid #ffcc00; text-align: center;'><h2 style='color: #b38600; margin: 0;'>Status: {status}</h2></div>", unsafe_allow_html=True)
    elif "Belum" in status:
        st.markdown(f"<div style='background-color: #e6f7ed; padding: 20px; border-radius: 10px; border: 2px solid #29a3a3; text-align: center;'><h2 style='color: #1f6666; margin: 0;'>Status: {status}</h2></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background-color: #f4f7f6; padding: 20px; border-radius: 10px; border: 2px solid #4a7cbf; text-align: center;'><h2 style='color: #4a7cbf; margin: 0;'>Status: {status}</h2></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Komposisi Nilai Spektrum Warna")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric(label="Rasio Hijau", value=f"{green:.2f}%")
    col_m2.metric(label="Rasio Kuning", value=f"{yellow:.2f}%")
    col_m3.metric(label="Rasio Merah", value=f"{red:.2f}%")

    st.markdown("<br>**Grafik Kepadatan Nilai Hijau**")
    st.progress(min(int(green), 100))
    st.markdown("**Grafik Kepadatan Nilai Kuning**")
    st.progress(min(int(yellow), 100))
    st.markdown("**Grafik Kepadatan Nilai Merah**")
    st.progress(min(int(red), 100))

    # Interpretasi logika ilmiah di bagian paling akhir halaman
    st.markdown("<br>#### 📖 Interpretasi Logika Ilmiah")
    if "Matang" in status and "Setengah" not in status and "Belum" not in status:
        penjelasan_status = f"Objek diklasifikasikan sebagai <b>🔴 Matang</b> karena spektrum warna <b>Merah ({red:.2f}%)</b> mendominasi secara mutlak. Secara biologis, klorofil telah terdegradasi sepenuhnya digantikan pigmen karotenoid yang merefleksikan gelombang cahaya merah pada ruang warna HSV."
    elif "Setengah" in status:
        penjelasan_status = f"Objek diklasifikasikan sebagai <b>🟡 Setengah Matang</b> karena spektrum warna <b>Kuning/Oranye ({yellow:.2f}%)</b> memiliki intensitas tertinggi, menandakan fase transisi pematangan kloronema pada model ruang warna HSV."
    elif "Belum" in status:
        penjelasan_status = f"Objek diklasifikasikan sebagai <b>🟢 Belum Matang</b> karena spektrum warna <b>Hijau ({green:.2f}%)</b> terdeteksi paling dominan, menandakan jaringan luar kulit cabai masih kaya akan pigmen klorofil."
    else:
        penjelasan_status = "Sistem tidak berhasil mengidentifikasi piksel yang masuk ke dalam parameter ambang batas komponen warna."

    st.markdown(f"<div class='pastel-card'><p style='line-height: 1.6; margin: 0;'>{penjelasan_status}</p></div>", unsafe_allow_html=True)