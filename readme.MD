# Deteksi Lubang Jalan dengan YOLO dan Preprocessing PCD Kelompok Ashpalt

### Anggota Kelompok:
- Nydia Renli Sinaga (122140007)
- Muhammad Daffa Rafif Wibowo (122140036)
- Adin Adry Tjindarbumi (122140024)
- Rafki Haykhal Alif (122140035)
- Naufal Haris N (122140040)

Project ini mengimplementasikan sistem deteksi lubang jalan menggunakan YOLOv8 dengan penerapan 4 konsep preprocessing Pengolahan Citra Digital (PCD) untuk meningkatkan kualitas dataset dan akurasi deteksi.

## Konsep PCD yang Diterapkan

1. **Image Resizing**
   - Normalisasi ukuran gambar ke 640x640 piksel
   - Mempertahankan aspect ratio untuk menghindari distorsi

2. **CLAHE Enhancement**
   - Contrast Limited Adaptive Histogram Equalization
   - Meningkatkan kontras lokal untuk memperjelas detail lubang
   - Diterapkan pada channel L dalam LAB color space

3. **Bilateral Filter**
   - Pengurangan noise dengan mempertahankan tepi objek
   - Edge-preserving smoothing filter
   - Mengurangi noise dari tekstur jalan tanpa mengaburkan batas lubang

4. **Data Augmentation**
   - Transformasi geometrik: horizontal flip, rotation, shift
   - Transformasi fotometrik: brightness, contrast, blur, noise
   - Memperbanyak dataset training untuk meningkatkan generalisasi model

## Struktur Dataset

```
POTHOLE_DETECTION/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Format label menggunakan YOLO format (file .txt):
```
class_id x_center y_center width height
```
Semua nilai dinormalisasi ke range [0, 1].

## Instalasi

### Requirements

- Python 3.8 atau lebih tinggi
- CUDA (opsional, untuk GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Cara Penggunaan

### 1. Persiapan Dataset

Letakkan dataset dalam format YOLO di direktori `POTHOLE_DETECTION/` dengan struktur seperti di atas.

### 2. Jalankan Notebook

Buka dan jalankan notebook `pothole_preprocessing_yolo.ipynb` secara berurutan:

```bash
jupyter notebook pothole_preprocessing_yolo.ipynb
```

### 3. Konfigurasi Path

Sesuaikan path dataset di cell konfigurasi:

```python
ORIGINAL_DATASET = "dataset"  # Sesuaikan dengan lokasi dataset Anda
```

### 4. Proses Training

Notebook akan menjalankan pipeline lengkap:

1. Inspeksi dataset original
2. Preprocessing (resize, CLAHE, bilateral filter)
3. Augmentasi dataset training
4. Training model YOLO
5. Evaluasi pada validation dan test set
6. Visualisasi hasil prediksi

### 5. Output

Setelah training selesai, hasil akan tersimpan di:

```
yolo_training_results/
├── pothole_detection/
│   ├── weights/
│   │   ├── ashpalt.pt      # Model terbaik
│   │   └── last.pt      # Model terakhir
│   ├── results.png      # Grafik training metrics
│   ├── confusion_matrix.png
│   └── ...
├── original_samples.png
├── preprocessing_*.png
├── augmentation_examples.png
└── prediction_samples.png
```

## Evaluasi Model

Model dievaluasi menggunakan metrics:

- **mAP50**: Mean Average Precision pada IoU threshold 0.5
- **mAP50-95**: Mean Average Precision pada IoU threshold 0.5-0.95
- **Precision**: Ketepatan prediksi positif
- **Recall**: Kemampuan mendeteksi semua objek positif

## Kontributor

Tugas Besar Pengolahan Citra Digital Kelompok Ashpalt