import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
import time
import io
import pandas as pd

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="Sistem Deteksi Lubang Jalan",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS - PROFESSIONAL DESIGN WITH BLACK BACKGROUND ==========
st.markdown("""
    <style>
    /* Root colors */
    :root {
        --dark-bg: #1a1a1a;
        --darker-bg: #0d0d0d;
        --card-bg: #262626;
        --accent: #c69749;
        --accent-dark: #b38638;
        --text-primary: #ffffff;
        --text-secondary: #d4d4d4;
        --border-dark: #3a3a3a;
    }
    
    /* Main styling */
    .main {
        padding: 2rem;
        background-color: #1a1a1a;
    }
    
    /* App background */
    [data-testid="stAppViewContainer"] {
        background-color: #1a1a1a;
    }
    
    [data-testid="stSidebar"] {
        background-color: #262626;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #0d0d0d 0%, #1a1a1a 100%);
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        border: 2px solid #c69749;
    }
    
    .header-container h1 {
        margin: 0;
        color: #c69749;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
        text-align: center;
    }
    
    .header-container p {
        margin: 0.8rem 0 0 0;
        font-size: 1.1rem;
        color: #d4d4d4;
        text-align: center;
    }
    
    /* Metric styling */
    .stMetric {
        background-color: #262626;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #c69749;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    .stMetric > div {
        color: #ffffff;
    }
    
    /* Section title */
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #c69749;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #c69749;
        padding-bottom: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #c69749;
        color: #1a1a1a;
        border-radius: 8px;
        border: none;
        padding: 0.8rem 1.5rem;
        font-weight: 700;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #d4a574;
        box-shadow: 0 4px 12px rgba(198, 151, 73, 0.4);
        transform: translateY(-2px);
    }
    
    /* Info box */
    .info-box {
        background-color: #1e3a4d;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 5px solid #c69749;
        color: #d4d4d4;
        font-weight: 500;
    }
    
    /* Success box */
    .success-box {
        background-color: #1e3a2a;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 5px solid #10b981;
        color: #86efac;
        font-weight: 500;
    }
    
    /* Warning box */
    .warning-box {
        background-color: #3a2d1a;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 5px solid #c69749;
        color: #fcd34d;
        font-weight: 500;
    }
    
    /* Table styling */
    .stDataFrame {
        background-color: #262626;
    }
    
    .dataframe {
        border-collapse: collapse;
        width: 100%;
        background-color: #262626;
    }
    
    .dataframe th {
        background-color: #0d0d0d;
        color: #c69749;
        padding: 1rem;
        text-align: left;
        font-weight: 700;
        border-bottom: 2px solid #c69749;
    }
    
    .dataframe td {
        padding: 0.8rem 1rem;
        border-bottom: 1px solid #3a3a3a;
        color: #d4d4d4;
    }
    
    .dataframe tr:hover {
        background-color: #323232;
    }
    
    /* Sidebar styling */
    .sidebar-card {
        background-color: #262626;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        border-left: 4px solid #c69749;
        border: 1px solid #3a3a3a;
    }
    
    .sidebar-card h3 {
        color: #c69749;
        margin-top: 0;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        text-align: center;
    }
    
    .sidebar-card p, .sidebar-card li {
        color: #d4d4d4;
    }
    
    /* Text styling */
    body {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background-color: #3a3a3a;
    }
    
    .stSlider > div > div > div > div {
        color: #c69749;
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        padding: 2rem;
        color: #808080;
        font-size: 0.9rem;
        border-top: 1px solid #3a3a3a;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 2px solid #c69749;
        margin: 1.5rem 0;
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: #262626;
    }
    
    /* Spinner */
    .stSpinner {
        color: #c69749;
    }
    
    /* All text */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
    }
    
    p, span, label {
        color: #d4d4d4;
    }
    
    </style>
    """, unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown("""
    <div class="header-container">
        <h1>Sistem Deteksi Lubang Jalan</h1>
        <p>Deteksi Otomatis Lubang Jalan Menggunakan Pengolahan Citra Digital dan Deep Learning (YOLOv8)</p>
    </div>
    """, unsafe_allow_html=True)

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    """Load YOLO model dengan caching"""
    try:
        model = YOLO('best.pt')
        return model
    except FileNotFoundError:
        st.error("Error: File 'best.pt' tidak ditemukan di folder GUI!")
        st.stop()

model = load_model()

# ========== SIDEBAR CONFIGURATION ==========
with st.sidebar:
    st.markdown('<div class="sidebar-card"><h3>Pengaturan Deteksi</h3>', unsafe_allow_html=True)
    
    conf_threshold = st.slider(
        "Threshold Model",
        min_value=0.1,
        max_value=0.95,
        value=0.5,
        step=0.05,
        help="Nilai lebih tinggi = deteksi lebih ketat, Nilai lebih rendah = lebih banyak deteksi"
    )
    
    st.caption(f"Nilai saat ini: {conf_threshold:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Informasi Model
    st.markdown('<div class="sidebar-card"><h3>Informasi Model</h3>', unsafe_allow_html=True)
    st.markdown(f"""
    **Model:** YOLOv8
    
    **Dataset:** Roboflow Pothole Dataset
    
    **GUI:** Streamlit
    
    **Akurasi:** ~70% mAP
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Petunjuk Penggunaan
    st.markdown('<div class="sidebar-card"><h3>Cara Penggunaan</h3>', unsafe_allow_html=True)
    st.markdown("""
    1. Unggah gambar jalan
    2. Klik tombol DETEKSI
    3. Lihat hasil deteksi
    4. Unduh hasil jika diperlukan
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ========== MAIN CONTENT ==========
st.markdown("---")

# Upload section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h3 class="section-title">Unggah Gambar Jalan</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Pilih file gambar",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        label_visibility="collapsed"
    )

with col2:
    st.markdown('<h3 class="section-title">Format yang Didukung</h3>', unsafe_allow_html=True)
    st.markdown("""
    - JPG/JPEG
    - PNG
    - BMP
    
    Maksimal: 200MB
    
    Min Resolusi: 320x320
    """)

# ========== DETECTION PROCESS ==========
if uploaded_file is not None:
    st.markdown("---")
    
    # Load image
    image = Image.open(uploaded_file)
    
    # Display original image with info
    col_preview1, col_preview2 = st.columns([3, 1])
    
    with col_preview1:
        st.markdown('<h3 class="section-title">Gambar Original</h3>', unsafe_allow_html=True)
        st.image(image, use_column_width=True)
    
    with col_preview2:
        st.markdown('<h3 class="section-title">Informasi Gambar</h3>', unsafe_allow_html=True)
        st.metric("Lebar", f"{image.size[0]} px")
        st.metric("Tinggi", f"{image.size[1]} px")
        st.metric("Format", image.format)
    
    st.markdown("---")
    
    # Detection buttons
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        detect_button = st.button(
            "DETEKSI",
            use_container_width=True,
            type="primary"
        )
    
    with col_btn2:
        reset_button = st.button(
            "RESET",
            use_container_width=True
        )
    
    if reset_button:
        st.rerun()
    
    # ========== DETECTION & RESULTS ==========
    if detect_button:
        # Convert image to OpenCV format
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Run inference
        with st.spinner("Memproses gambar..."):
            start_time = time.time()
            results = model.predict(img_bgr, conf=conf_threshold, verbose=False)
            inference_time = time.time() - start_time
        
        # Get annotated image
        annotated_img = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Extract detections
        detections = []
        if results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes, 1):
                detection = {
                    'No': i,
                    'Confidence': float(box.conf[0]),
                    'X1': int(box.xyxy[0][0]),
                    'Y1': int(box.xyxy[0][1]),
                    'X2': int(box.xyxy[0][2]),
                    'Y2': int(box.xyxy[0][3]),
                    'Lebar': int(box.xyxy[0][2]) - int(box.xyxy[0][0]),
                    'Tinggi': int(box.xyxy[0][3]) - int(box.xyxy[0][1])
                }
                detections.append(detection)
            
            # Sort by confidence
            detections = sorted(detections, key=lambda x: x['Confidence'], reverse=True)
        
        # Results and statistics
        col_result, col_stat = st.columns([2, 1])
        
        with col_result:
            st.markdown('<h3 class="section-title">Hasil Deteksi</h3>', unsafe_allow_html=True)
            st.image(annotated_rgb, use_column_width=True)
        
        with col_stat:
            st.markdown('<h3 class="section-title">Statistik</h3>', unsafe_allow_html=True)
            
            # Total detections
            st.metric(
                "Total Lubang Terdeteksi",
                len(detections)
            )
            
            # Accuracy dari detections yang ditemukan
            if detections:
                avg_conf = np.mean([d['Confidence'] for d in detections])
                st.metric("Akurasi Deteksi", f"{avg_conf:.1%}")
            else:
                st.metric("Akurasi Deteksi", "0%")
            
            # Processing time
            st.metric("Waktu Pemrosesan", f"{inference_time:.3f}s")
        
        # Detection details table
        st.markdown("---")
        st.markdown('<h3 class="section-title">Detail Deteksi</h3>', unsafe_allow_html=True)
        
        if detections:
            # Create display dataframe
            df_display = pd.DataFrame([
                {
                    'No': d['No'],
                    'Akurasi': f"{d['Confidence']:.1%}",
                    'X1': d['X1'],
                    'Y1': d['Y1'],
                    'X2': d['X2'],
                    'Y2': d['Y2'],
                    'Lebar': d['Lebar'],
                    'Tinggi': d['Tinggi']
                }
                for d in detections
            ])
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # Download section
            st.markdown("---")
            st.markdown('<h3 class="section-title">Unduh Hasil</h3>', unsafe_allow_html=True)
            
            col_down1, col_down2, col_down3 = st.columns(3)
            
            # Download image
            with col_down1:
                annotated_pil = Image.fromarray(annotated_rgb)
                buf_img = io.BytesIO()
                annotated_pil.save(buf_img, format='PNG')
                buf_img.seek(0)
                
                st.download_button(
                    label="Unduh Gambar (PNG)",
                    data=buf_img,
                    file_name="deteksi_lubang.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Download report
            with col_down2:
                report = f"""LAPORAN DETEKSI LUBANG JALAN
=============================
Tanggal: {time.strftime('%Y-%m-%d %H:%M:%S')}
Model: YOLOv8

RINGKASAN:
----------
Total Lubang Terdeteksi: {len(detections)}
Treshold: {conf_threshold}
Waktu Pemrosesan: {inference_time:.3f} detik
Rata-rata Akurasi: {np.mean([d['Confidence'] for d in detections]):.1%}

DETAIL LUBANG:
---------------
"""
                for det in detections:
                    report += f"\nLubang #{det['No']}:\n"
                    report += f"  Akurasi: {det['Confidence']:.1%}\n"
                    report += f"  Posisi: ({det['X1']}, {det['Y1']}) ke ({det['X2']}, {det['Y2']})\n"
                    report += f"  Ukuran: {det['Lebar']}x{det['Tinggi']} px\n"
                
                st.download_button(
                    label="Unduh Laporan (TXT)",
                    data=report,
                    file_name="laporan_deteksi.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
        else:
            st.markdown('<div class="warning-box"><strong>Tidak Ada Lubang Terdeteksi</strong> pada gambar ini. Coba sesuaikan Treshold </div>', unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown("""
    <div class="footer-text">
        <strong>Sistem Deteksi Lubang Jalan Otomaris | Aspalt</strong> 
    """, unsafe_allow_html=True)