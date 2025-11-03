"""
Configuration file untuk sistem deteksi lubang jalan.
Berisi semua parameter yang dapat di-tuning untuk optimasi hasil deteksi.

Author: Naufal (Image Processing Engineer)
Project: Tugas Besar Pengolahan Citra Digital - ITERA
"""

class Config:
    """
    Kelas konfigurasi yang berisi semua parameter sistem deteksi pothole.
    Parameter ini berdasarkan penelitian dari referensi paper.
    """
    
    # ========== PREPROCESSING PARAMETERS ==========
    
    # Target size untuk resize image (width, height) dalam piksel
    # Reasoning: 640x480 cukup untuk deteksi, lebih cepat diproses
    # Aspect ratio akan dipertahankan
    TARGET_SIZE = (640, 480)
    
    # Interpolation method untuk resizing
    # INTER_NEAREST = paling cepat, cukup untuk deteksi
    # INTER_LINEAR = lebih smooth tapi lebih lambat
    RESIZE_INTERPOLATION = 'INTER_NEAREST'
    
    
    # ========== GRAYSCALE CONVERSION ==========
    
    # Formula: Gray = 0.2989*R + 0.5870*G + 0.0721*B
    # Sudah built-in di OpenCV (cv2.COLOR_RGB2GRAY)
    # Tidak perlu parameter tambahan
    
    
    # ========== ADAPTIVE THRESHOLD PARAMETERS ==========
    
    # Block size untuk adaptive threshold (harus bilangan ganjil)
    # Reasoning dari paper: 101x101 optimal untuk jalan kampus
    # Terlalu kecil: sensitif noise
    # Terlalu besar: detail hilang
    ADAPTIVE_BLOCK_SIZE = 101
    
    # Constant C untuk adaptive threshold
    # Nilai threshold lokal = weighted_average - C
    # Reasoning: C=10 optimal berdasarkan eksperimen
    ADAPTIVE_CONSTANT = 10
    
    # Method adaptive threshold
    # ADAPTIVE_THRESH_GAUSSIAN_C = weighted sum (lebih smooth)
    # ADAPTIVE_THRESH_MEAN_C = arithmetic mean (lebih cepat)
    ADAPTIVE_METHOD = 'GAUSSIAN'
    
    
    # ========== CANNY EDGE DETECTION PARAMETERS ==========
    
    # Gaussian sigma untuk smoothing sebelum Canny
    # Reasoning dari paper: sigma=3 optimal
    # Lebih besar = lebih smooth tapi tepi kurang detail
    CANNY_SIGMA = 3
    
    # Low threshold untuk hysteresis thresholding
    # Piksel dengan gradient < LOW akan di-reject
    CANNY_LOW_THRESHOLD = 0
    
    # High threshold untuk hysteresis thresholding
    # Piksel dengan gradient > HIGH pasti tepi
    # Reasoning dari paper: 150 optimal
    CANNY_HIGH_THRESHOLD = 150
    
    
    # ========== MORPHOLOGICAL OPERATIONS ==========
    
    # Ukuran structure element untuk dilasi
    # Square 3x3 untuk menebalkan dan menyambungkan tepi
    STREL_SIZE = (3, 3)
    
    # Tipe structure element
    # MORPH_RECT = rectangular (paling umum)
    # MORPH_ELLIPSE = elliptical (lebih smooth)
    # MORPH_CROSS = cross-shaped
    STREL_SHAPE = 'RECT'
    
    # Jumlah iterasi dilasi
    # 1 iterasi biasanya cukup
    DILATION_ITERATIONS = 1
    
    
    # ========== OBJECT FILTERING PARAMETERS ==========
    
    # Ukuran MINIMUM objek yang dianggap lubang (width, height) dalam piksel
    # Reasoning dari paper: 15x15 piksel
    # Lebih kecil = noise (batu, kotoran)
    MIN_POTHOLE_SIZE = 15
    
    # Ukuran MAKSIMUM objek yang dianggap lubang (width, height) dalam piksel
    # Reasoning dari paper: 290x540 piksel
    # Lebih besar = bukan lubang (motor, bayangan besar, dll)
    MAX_POTHOLE_WIDTH = 290
    MAX_POTHOLE_HEIGHT = 540
    
    # Aspect Ratio (AR = width/height) filter
    # Lubang cenderung bulat/oval, AR mendekati 1.0
    # Range valid: 0.3 <= AR <= 3.0
    # Tujuan: buang objek terlalu lonjong (marka, bayangan motor)
    AR_MIN = 0.3
    AR_MAX = 3.0
    
    # Solidity filter (Solidity = area/convex_hull_area)
    # Lubang cenderung solid/padat
    # Threshold: Solidity > 0.6
    # Tujuan: buang objek tidak solid (kotoran, ranting)
    SOLIDITY_THRESHOLD = 0.6
    
    
    # ========== VIDEO PROCESSING PARAMETERS ==========
    
    # Frame sampling rate (berapa frame per detik yang akan diproses)
    # 1 fps = 1 frame per detik
    # Reasoning: 1 fps cukup untuk deteksi, efisien komputasi
    # Video 30fps, 10 detik = 300 frame → sampling 1fps = 10 frame saja
    VIDEO_FPS_SAMPLE = 1
    
    # Euclidean distance threshold untuk deduplication (dalam piksel)
    # Jika jarak antar bounding box < 50 piksel → dianggap lubang yang sama
    # Reasoning: kamera bergerak sedikit, koordinat berubah sedikit
    DEDUPLICATION_THRESHOLD = 50
    
    
    # ========== OUTPUT VISUALIZATION PARAMETERS ==========
    
    # Warna bounding box untuk marking lubang (BGR format)
    # Merah = (0, 0, 255) dalam BGR
    BBOX_COLOR = (0, 0, 255)  # RED
    
    # Ketebalan garis bounding box
    BBOX_THICKNESS = 2
    
    # Ukuran font untuk label
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    
    
    # ========== PATH CONFIGURATION ==========
    
    # Base directory paths
    DATASET_DIR = 'dataset'
    OUTPUT_DIR = 'output'
    GROUND_TRUTH_DIR = 'ground_truth'
    
    # Output subdirectories
    OUTPUT_IMAGES_DIR = 'output/images'
    OUTPUT_VIDEOS_DIR = 'output/videos'
    
    
    # ========== EVALUATION PARAMETERS ==========
    
    # Minimum IoU (Intersection over Union) untuk True Positive
    # IoU > 0.5 biasanya dianggap deteksi benar
    MIN_IOU_THRESHOLD = 0.5
    
    
    # ========== DEBUG & LOGGING ==========
    
    # Enable debug mode (print intermediate results)
    DEBUG_MODE = False
    
    # Save intermediate images (grayscale, threshold, canny, dll)
    SAVE_INTERMEDIATE = False
    
    
    @classmethod
    def print_config(cls):
        """
        Print semua konfigurasi yang sedang digunakan.
        Berguna untuk debugging dan dokumentasi.
        """
        print("=" * 60)
        print("KONFIGURASI SISTEM DETEKSI LUBANG JALAN")
        print("=" * 60)
        print(f"\n📐 PREPROCESSING:")
        print(f"  Target Size: {cls.TARGET_SIZE}")
        print(f"  Resize Method: {cls.RESIZE_INTERPOLATION}")
        
        print(f"\n🎯 ADAPTIVE THRESHOLD:")
        print(f"  Block Size: {cls.ADAPTIVE_BLOCK_SIZE}")
        print(f"  Constant C: {cls.ADAPTIVE_CONSTANT}")
        print(f"  Method: {cls.ADAPTIVE_METHOD}")
        
        print(f"\n🔍 CANNY EDGE DETECTION:")
        print(f"  Sigma: {cls.CANNY_SIGMA}")
        print(f"  Low Threshold: {cls.CANNY_LOW_THRESHOLD}")
        print(f"  High Threshold: {cls.CANNY_HIGH_THRESHOLD}")
        
        print(f"\n🔧 MORPHOLOGICAL:")
        print(f"  Strel Size: {cls.STREL_SIZE}")
        print(f"  Strel Shape: {cls.STREL_SHAPE}")
        
        print(f"\n📏 FILTERING:")
        print(f"  Min Size: {cls.MIN_POTHOLE_SIZE}px")
        print(f"  Max Size: {cls.MAX_POTHOLE_WIDTH}x{cls.MAX_POTHOLE_HEIGHT}px")
        print(f"  Aspect Ratio: {cls.AR_MIN} - {cls.AR_MAX}")
        print(f"  Solidity: > {cls.SOLIDITY_THRESHOLD}")
        
        print(f"\n🎥 VIDEO:")
        print(f"  FPS Sampling: {cls.VIDEO_FPS_SAMPLE}")
        print(f"  Deduplication Threshold: {cls.DEDUPLICATION_THRESHOLD}px")
        
        print("=" * 60)


# Test konfigurasi jika file ini dijalankan langsung
if __name__ == "__main__":
    Config.print_config()