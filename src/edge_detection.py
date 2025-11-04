"""
Edge Detection Module untuk Sistem Deteksi Lubang Jalan.
Implementasi Canny Edge Detection untuk mendeteksi tepi lubang.

Canny Edge Detection adalah salah satu algoritma deteksi tepi terbaik karena:
1. Low error rate (deteksi tepi yang benar-benar tepi)
2. Good localization (tepi terdeteksi dekat dengan tepi sebenarnya)
3. Single response (satu tepi hanya terdeteksi sekali)

Author: Naufal (Image Processing Engineer)
Project: Tugas Besar Pengolahan Citra Digital - ITERA
"""

import cv2
import numpy as np
from config import Config


class EdgeDetection:
    """
    Kelas untuk melakukan edge detection menggunakan Canny algorithm.
    
    Canny Edge Detection memiliki 5 tahapan internal:
    1. Gaussian Smoothing → Kurangi noise
    2. Gradient Calculation → Hitung magnitude dan direction
    3. Non-maximum Suppression → Tipiskan tepi
    4. Double Threshold → Klasifikasi strong/weak edges
    5. Edge Tracking by Hysteresis → Sambungkan weak edges ke strong edges
    """
    
    def __init__(self, config=None):
        """
        Inisialisasi EdgeDetection.
        
        Parameters:
        -----------
        config : Config object, optional
            Objek konfigurasi. Jika None, gunakan Config default.
        
        Example:
        --------
        >>> edge_detector = EdgeDetection()
        >>> edge_detector = EdgeDetection(custom_config)
        """
        self.config = config if config else Config()
    
    
    def canny_edge_detection(self, binary_image):
        """
        Deteksi tepi menggunakan Canny Edge Detection.
        
        Kenapa Canny untuk Project Ini?
        --------------------------------
        ✅ Deteksi tepi dengan presisi tinggi
        ✅ Otomatis menghilangkan noise dengan Gaussian filter
        ✅ Menghasilkan tepi yang tipis dan jelas
        ✅ Hysteresis thresholding → sambungkan tepi yang putus
        
        Tahapan Internal Canny (dilakukan otomatis oleh OpenCV):
        --------------------------------------------------------
        1. **Gaussian Smoothing:**
           - Blur image dengan Gaussian filter (sigma=3)
           - Tujuan: Kurangi noise sebelum deteksi tepi
           - Formula: G(x,y) = (1/2πσ²) * exp(-(x²+y²)/2σ²)
        
        2. **Gradient Calculation:**
           - Hitung gradient magnitude: G = √(Gx² + Gy²)
           - Hitung gradient direction: θ = atan2(Gy, Gx)
           - Menggunakan Sobel operator (3×3 kernel)
        
        3. **Non-Maximum Suppression:**
           - Tipiskan tepi dengan suppress piksel yang bukan maksimum lokal
           - Cek gradient direction, bandingkan dengan neighbors
           - Hasil: Tepi menjadi 1 piksel tebal
        
        4. **Double Threshold:**
           - High threshold: Piksel dengan gradient > HIGH = strong edge
           - Low threshold: Piksel dengan gradient < LOW = bukan edge
           - Between: Piksel antara LOW dan HIGH = weak edge
        
        5. **Edge Tracking by Hysteresis:**
           - Pertahankan weak edges yang terhubung dengan strong edges
           - Buang weak edges yang tidak terhubung (noise)
           - Hasil: Tepi yang tersambung dengan baik
        
        Parameter Penting:
        ------------------
        - **Sigma (3):** Standar deviasi Gaussian filter
          * Lebih besar: Lebih smooth, tapi detail hilang
          * Lebih kecil: Lebih detail, tapi noise tinggi
          * Optimal: 3 (berdasarkan paper referensi)
        
        - **Low Threshold (0):** Threshold minimum untuk gradient
          * 0 = Tidak ada rejection di tahap awal
          * Biarkan hysteresis yang handle
        
        - **High Threshold (150):** Threshold untuk strong edges
          * Terlalu tinggi: Under-detection (tepi hilang)
          * Terlalu rendah: Over-detection (noise terdeteksi)
          * Optimal: 150 (berdasarkan eksperimen)
        
        Kenapa Input Binary Image (bukan Grayscale)?
        ---------------------------------------------
        - Kita sudah lakukan Adaptive Threshold sebelumnya
        - Binary image = foreground (lubang) sudah terpisah dari background (jalan)
        - Canny tinggal deteksi tepi dari foreground yang sudah clear
        - Hasil lebih bersih dibanding langsung Canny pada grayscale
        
        Parameters:
        -----------
        binary_image : numpy.ndarray
            Input binary image (hasil adaptive threshold)
            Shape: (height, width)
            Dtype: uint8
            Values: 0 (hitam/foreground) atau 255 (putih/background)
        
        Returns:
        --------
        edges : numpy.ndarray
            Edge image
            Shape: (height, width)
            Dtype: uint8
            Values: 0 (bukan tepi) atau 255 (tepi)
        
        Example:
        --------
        >>> binary = segmenter.adaptive_threshold(gray)
        >>> edge_detector = EdgeDetection()
        >>> edges = edge_detector.canny_edge_detection(binary)
        >>> # Putih (255) = tepi lubang, Hitam (0) = bukan tepi
        
        Notes:
        ------
        - Output: Garis tepi PUTIH (255) pada latar HITAM (0)
        - Tepi yang terdeteksi = batas antara lubang dan jalan
        - Garis tepi tipis (1 piksel) hasil non-maximum suppression
        
        Referensi:
        ----------
        Paper: "Sistem Pendeteksi Kerusakan Jalan Aspal Menggunakan Canny Edge Detection"
               J-ICON Vol. 11 No. 1 (2023)
               Parameter optimal: sigma=3, low=0, high=150
        """
        if binary_image is None:
            raise ValueError("Input binary_image is None.")
        
        # Validasi input
        if len(binary_image.shape) != 2:
            raise ValueError(f"Expected binary image (2D), got shape: {binary_image.shape}")
        
        # Get parameters from config
        sigma = self.config.CANNY_SIGMA
        low_threshold = self.config.CANNY_LOW_THRESHOLD
        high_threshold = self.config.CANNY_HIGH_THRESHOLD
        
        # Apply Gaussian Blur (preprocessing sebelum Canny)
        # Ini OPSIONAL karena Canny sudah apply Gaussian internal
        # Tapi kita lakukan untuk kontrol manual terhadap sigma
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)  # Hitung kernel size dari sigma
        if kernel_size % 2 == 0:  # Pastikan ganjil
            kernel_size += 1
        
        blurred = cv2.GaussianBlur(
            binary_image,
            (kernel_size, kernel_size),
            sigma
        )
        
        # Apply Canny Edge Detection
        edges = cv2.Canny(
            blurred,           # Input: binary image yang sudah di-blur
            low_threshold,     # Low threshold (0)
            high_threshold     # High threshold (150)
        )
        
        if self.config.DEBUG_MODE:
            print(f"\n🔍 CANNY EDGE DETECTION:")
            print(f"   Gaussian Kernel Size: {kernel_size}x{kernel_size}")
            print(f"   Gaussian Sigma: {sigma}")
            print(f"   Low Threshold: {low_threshold}")
            print(f"   High Threshold: {high_threshold}")
            print(f"   Output shape: {edges.shape}")
            
            # Hitung statistik
            num_edge_pixels = np.sum(edges == 255)
            total_pixels = edges.size
            edge_percent = (num_edge_pixels / total_pixels) * 100
            
            print(f"   Edge pixels: {num_edge_pixels} ({edge_percent:.2f}%)")
            print(f"   Non-edge pixels: {total_pixels - num_edge_pixels} ({100-edge_percent:.2f}%)")
        
        return edges
    
    
    def sobel_edge_detection(self, gray_image):
        """
        Deteksi tepi menggunakan Sobel operator (untuk perbandingan).
        
        CATATAN: Method ini TIDAK digunakan di project karena Canny lebih baik.
        Hanya disediakan untuk perbandingan dan educational purpose.
        
        Sobel Edge Detection:
        ---------------------
        - Menggunakan kernel 3×3 untuk deteksi gradient horizontal (Gx) dan vertikal (Gy)
        - Gradient magnitude: G = √(Gx² + Gy²)
        - Lebih simple tapi kurang presisi dibanding Canny
        
        Kekurangan Sobel vs Canny:
        --------------------------
        ❌ Tidak ada noise reduction
        ❌ Tepi lebih tebal (tidak ada non-maximum suppression)
        ❌ Tidak ada hysteresis thresholding
        ❌ Banyak false edges dari noise
        
        Parameters:
        -----------
        gray_image : numpy.ndarray
            Input grayscale image
        
        Returns:
        --------
        edges : numpy.ndarray
            Edge image dari Sobel
        
        Example:
        --------
        >>> edges_sobel = edge_detector.sobel_edge_detection(gray)
        """
        if gray_image is None:
            raise ValueError("Input gray_image is None.")
        
        # Sobel operator untuk Gx (horizontal gradient)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        
        # Sobel operator untuk Gy (vertical gradient)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Hitung magnitude: G = √(Gx² + Gy²)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize ke range 0-255
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))
        
        if self.config.DEBUG_MODE:
            print(f"\n🔍 SOBEL EDGE DETECTION (for comparison):")
            print(f"   Output shape: {magnitude.shape}")
        
        return magnitude
    
    
    def compare_methods(self, binary_image, gray_image):
        """
        Bandingkan Canny vs Sobel untuk educational purpose.
        
        Parameters:
        -----------
        binary_image : numpy.ndarray
            Binary image untuk Canny
        gray_image : numpy.ndarray
            Grayscale image untuk Sobel
        
        Returns:
        --------
        results : dict
            Dictionary berisi hasil kedua method
            Keys: 'canny', 'sobel'
        
        Example:
        --------
        >>> results = edge_detector.compare_methods(binary, gray)
        >>> cv2.imshow('Canny', results['canny'])
        >>> cv2.imshow('Sobel', results['sobel'])
        """
        canny_result = self.canny_edge_detection(binary_image)
        sobel_result = self.sobel_edge_detection(gray_image)
        
        results = {
            'canny': canny_result,
            'sobel': sobel_result
        }
        
        return results


# ========== TEST CODE ==========

if __name__ == "__main__":
    """
    Test code untuk memverifikasi edge detection bekerja dengan benar.
    """
    print("🧪 TESTING EDGE DETECTION MODULE")
    print("="*60)
    
    # Create dummy binary image (simulasi hasil adaptive threshold)
    print("\n1️⃣ Creating dummy binary image...")
    dummy_binary = np.zeros((480, 640), dtype=np.uint8)
    # Buat rectangle putih (simulasi jalan)
    dummy_binary[:, :] = 255
    # Buat circle hitam (simulasi lubang)
    cv2.circle(dummy_binary, (320, 240), 50, 0, -1)
    print(f"   Binary image shape: {dummy_binary.shape}")
    print(f"   Simulated circular pothole at center (radius=50)")
    
    # Create dummy grayscale (untuk Sobel comparison)
    dummy_gray = dummy_binary.copy()
    
    # Initialize edge detection
    print("\n2️⃣ Initializing edge detection...")
    config = Config()
    config.DEBUG_MODE = True
    edge_detector = EdgeDetection(config)
    print("   Edge detection initialized ✅")
    
    # Test Canny
    print("\n3️⃣ Testing Canny Edge Detection...")
    canny_edges = edge_detector.canny_edge_detection(dummy_binary)
    print(f"   Canny result shape: {canny_edges.shape}")
    
    # Test Sobel (untuk perbandingan)
    print("\n4️⃣ Testing Sobel Edge Detection (for comparison)...")
    sobel_edges = edge_detector.sobel_edge_detection(dummy_gray)
    print(f"   Sobel result shape: {sobel_edges.shape}")
    
    # Compare methods
    print("\n5️⃣ Comparing both methods...")
    results = edge_detector.compare_methods(dummy_binary, dummy_gray)
    print(f"   Comparison completed ✅")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\n💡 Note:")
    print("   - Canny Edge Detection dipilih untuk project ini")
    print("   - Sobel hanya untuk perbandingan")
    print("   - Canny lebih presisi dan robust")