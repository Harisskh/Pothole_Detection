"""
POTHOLE DETECTION STREAMLIT GUI - ENHANCED VERSION
===================================================
Aplikasi GUI untuk deteksi lubang jalan menggunakan YOLO

Cara menjalankan:
    streamlit run app.py
"""

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
    page_title="Pothole Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B6B;
    }
    </style>
    """, unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white;">
        <h1 style="margin: 0; color: white;">🚨 Pothole Detection System</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            Deteksi lubang jalan otomatis menggunakan Deep Learning (YOLOv8)
        </p>
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
        st.error("❌ File 'best.pt' tidak ditemukan!")
        st.stop()

model = load_model()

# ========== SIDEBAR CONFIGURATION ==========
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Deteksi")
    
    conf_threshold = st.slider(
        "**Confidence Threshold**",
        min_value=0.1,
        max_value=0.95,
        value=0.5,
        step=0.05,
        help="Semakin tinggi = deteksi lebih ketat, semakin rendah = lebih banyak deteksi"
    )
    
    st.markdown(f"**Nilai saat ini:** `{conf_threshold:.2f}`")
    
    st.divider()
    
    st.markdown("### 📊 Model Info")
    st.info("""
    **Model:** YOLOv8  
    **Accuracy:** ~70% mAP  
    **Dataset:** Roboflow Pothole  
    **Framework:** PyTorch
    """)
    
    st.divider()
    
    st.markdown("### 📝 Petunjuk Penggunaan")
    st.markdown("""
    1. Upload gambar jalan
    2. Klik tombol **DETECT**
    3. Lihat hasil deteksi
    4. Download hasil jika perlu
    """)

# ========== MAIN CONTENT ==========
st.markdown("---")

# Upload section
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 📤 Upload Gambar Jalan")
    uploaded_file = st.file_uploader(
        "Pilih gambar (JPG, PNG, BMP)",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        label_visibility="collapsed"
    )

with col2:
    st.markdown("### ℹ️ Format")
    st.markdown("""
    - JPG/PNG/BMP
    - Max 200MB
    - Resolusi minimum 320x320
    """, help="Gambar dengan kualitas tinggi akan memberikan hasil deteksi yang lebih baik")

# ========== DETECTION PROCESS ==========
if uploaded_file is not None:
    st.markdown("---")
    
    # Load and display original image
    image = Image.open(uploaded_file)
    
    col_preview1, col_preview2 = st.columns([3, 1])
    
    with col_preview1:
        st.markdown("### 🖼️ Gambar Original")
        st.image(image, use_column_width=True)
    
    with col_preview2:
        st.markdown("### 📐 Informasi Gambar")
        st.metric("Lebar", f"{image.size[0]} px")
        st.metric("Tinggi", f"{image.size[1]} px")
        st.metric("Format", image.format)
    
    st.markdown("---")
    
    # Detect button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        detect_button = st.button(
            "🔍 DETEKSI POTHOLE",
            use_container_width=True,
            type="primary"
        )
    
    with col_btn2:
        reset_button = st.button(
            "🔄 Reset",
            use_container_width=True
        )
    
    if reset_button:
        st.rerun()
    
    # ========== DETECTION & RESULTS ==========
    if detect_button:
        # Convert to OpenCV format
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Run inference
        with st.spinner("⏳ Sedang memproses gambar..."):
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
                    'Width': int(box.xyxy[0][2]) - int(box.xyxy[0][0]),
                    'Height': int(box.xyxy[0][3]) - int(box.xyxy[0][1])
                }
                detections.append(detection)
            
            # Sort by confidence descending
            detections = sorted(detections, key=lambda x: x['Confidence'], reverse=True)
        
        # Display results
        st.markdown("---")
        st.success("✅ **Deteksi Selesai!**")
        st.markdown("---")
        
        # Results and statistics in columns
        col_result, col_stat = st.columns([2, 1])
        
        with col_result:
            st.markdown("### 🎯 Hasil Deteksi")
            st.image(annotated_rgb, use_column_width=True)
        
        with col_stat:
            st.markdown("### 📊 Statistik")
            
            # Total detections
            st.metric(
                "🔍 Total Pothole",
                len(detections),
                delta="detected" if len(detections) > 0 else "none"
            )
            
            # Average confidence
            if detections:
                avg_conf = np.mean([d['Confidence'] for d in detections])
                max_conf = max([d['Confidence'] for d in detections])
                min_conf = min([d['Confidence'] for d in detections])
                
                st.metric("📈 Avg Confidence", f"{avg_conf:.1%}")
                st.metric("⭐ Max Confidence", f"{max_conf:.1%}")
                st.metric("📉 Min Confidence", f"{min_conf:.1%}")
            
            # Processing time
            st.metric("⚡ Processing Time", f"{inference_time:.3f}s")
        
        # Detection details
        st.markdown("---")
        st.markdown("### 📋 Detail Deteksi")
        
        if detections:
            # Create dataframe
            df_display = pd.DataFrame([
                {
                    'No': d['No'],
                    'Confidence': f"{d['Confidence']:.1%}",
                    'X1': d['X1'],
                    'Y1': d['Y1'],
                    'X2': d['X2'],
                    'Y2': d['Y2'],
                    'Width': d['Width'],
                    'Height': d['Height']
                }
                for d in detections
            ])
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # Download options
            st.markdown("---")
            st.markdown("### 💾 Download Hasil")
            
            col_down1, col_down2, col_down3 = st.columns(3)
            
            # Download annotated image
            with col_down1:
                annotated_pil = Image.fromarray(annotated_rgb)
                buf_img = io.BytesIO()
                annotated_pil.save(buf_img, format='PNG')
                buf_img.seek(0)
                
                st.download_button(
                    label="📥 Gambar (PNG)",
                    data=buf_img,
                    file_name=f"pothole_detection.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Download report
            with col_down2:
                report = f"""LAPORAN DETEKSI POTHOLE
==========================
Tanggal: {time.strftime('%Y-%m-%d %H:%M:%S')}
Model: YOLOv8

RINGKASAN:
----------
Total Pothole Terdeteksi: {len(detections)}
Confidence Threshold: {conf_threshold}
Waktu Pemrosesan: {inference_time:.3f} detik
Rata-rata Confidence: {np.mean([d['Confidence'] for d in detections]):.1%}

DETAIL POTHOLE:
---------------
"""
                for det in detections:
                    report += f"\nPothole #{det['No']}:\n"
                    report += f"  - Confidence: {det['Confidence']:.1%}\n"
                    report += f"  - Posisi: ({det['X1']}, {det['Y1']}) ke ({det['X2']}, {det['Y2']})\n"
                    report += f"  - Ukuran: {det['Width']}x{det['Height']} px\n"
                
                st.download_button(
                    label="📥 Report (TXT)",
                    data=report,
                    file_name=f"pothole_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # Download CSV
            with col_down3:
                csv_data = "No,Confidence,X1,Y1,X2,Y2,Width,Height\n"
                for det in detections:
                    csv_data += f"{det['No']},{det['Confidence']:.4f},{det['X1']},{det['Y1']},{det['X2']},{det['Y2']},{det['Width']},{det['Height']}\n"
                
                st.download_button(
                    label="📥 Data (CSV)",
                    data=csv_data,
                    file_name=f"pothole_detections.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        else:
            st.info("❌ Tidak ada pothole yang terdeteksi pada gambar ini")
            st.markdown("**Saran:** Coba ubah confidence threshold ke nilai yang lebih rendah")

else:
    st.info("👈 Silakan upload gambar jalan untuk memulai deteksi")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #666;">
        <small>Pothole Detection System v1.0 | Powered by YOLOv8</small>
    </div>
    """, unsafe_allow_html=True)