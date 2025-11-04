"""
Post-Processing Module untuk Sistem Deteksi Lubang Jalan.
Berisi operasi morfologi (dilasi) dan multi-level filtering untuk
membersihkan hasil deteksi dan filter objek yang bukan lubang.

Author: Naufal (Image Processing Engineer)
Project: Tugas Besar Pengolahan Citra Digital - ITERA
"""

import cv2
import numpy as np
from config import Config


class PostProcessing:
    """
    Kelas untuk post-processing hasil edge detection.
    
    Tahapan:
    1. Dilasi → Tebalkan dan sambungkan tepi yang putus
    2. Find Contours → Deteksi semua objek (connected components)
    3. Multi-level Filtering → Filter objek yang bukan lubang
       - Size Filter (ukuran)
       - Aspect Ratio Filter (rasio lebar/tinggi)
       - Solidity Filter (kepadatan objek)
    """
    
    def __init__(self, config=None):
        """
        Inisialisasi PostProcessing.
        
        Parameters:
        -----------
        config : Config object, optional
            Objek konfigurasi. Jika None, gunakan Config default.
        
        Example:
        --------
        >>> postprocessor = PostProcessing()
        >>> postprocessor = PostProcessing(custom_config)
        """
        self.config = config if config else Config()
        
        # Mapping string ke OpenCV morphological shape
        self.strel_shapes = {
            'RECT': cv2.MORPH_RECT,
            'ELLIPSE': cv2.MORPH_ELLIPSE,
            'CROSS': cv2.MORPH_CROSS
        }
    
    
    def dilate(self, edge_image):
        """
        Dilasi (morphological dilation) untuk menebalkan tepi.
        
        Kenapa Dilasi?
        --------------
        - Tepi dari Canny sangat tipis (1 piksel)
        - Tepi sering putus-putus karena noise
        - Dilasi menebalkan dan menyambungkan tepi yang berdekatan
        - Hasil: Area lubang lebih solid dan mudah dideteksi
        
        Konsep Dilasi:
        --------------
        Dilasi = Ekspansi objek dengan structure element (strel)
        
        Formula: D = A ⊕ B
        - A = input image
        - B = structure element
        - ⊕ = dilasi operator
        
        Cara kerja:
        1. Slide strel di atas image
        2. Jika strel overlap dengan objek (putih/255), set center ke putih
        3. Hasil: Objek membesar sesuai shape strel
        
        Structure Element (Strel):
        --------------------------
        Strel = Kernel yang menentukan shape dilasi
        
        Square 3×3:
        ```
        1 1 1
        1 1 1
        1 1 1
        ```
        - Expand uniformly ke 8 neighbors
        - Paling umum digunakan
        - Baik untuk menebalkan tepi tipis
        
        Parameter Penting:
        ------------------
        - **Strel Size (3×3):** Ukuran kernel dilasi
          * Terlalu kecil (1×1): Tidak ada efek
          * Terlalu besar (7×7): Over-dilation, detail hilang
          * Optimal (3×3): Balance antara sambung tepi dan preserve detail
        
        - **Strel Shape (RECT):** Bentuk kernel
          * RECT: Rectangular, expand uniform
          * ELLIPSE: Elliptical, lebih smooth
          * CROSS: Cross-shaped, expand diagonal
        
        - **Iterations (1):** Jumlah kali dilasi diulang
          * Lebih banyak iterasi = tepi lebih tebal
          * 1 iterasi biasanya cukup
        
        Parameters:
        -----------
        edge_image : numpy.ndarray
            Input edge image (hasil Canny)
            Shape: (height, width)
            Dtype: uint8
            Values: 0 (bukan tepi) atau 255 (tepi)
        
        Returns:
        --------
        dilated : numpy.ndarray
            Dilated edge image
            Shape: (height, width)
            Dtype: uint8
            Values: 0 atau 255
        
        Example:
        --------
        >>> edges = edge_detector.canny_edge_detection(binary)
        >>> postprocessor = PostProcessing()
        >>> dilated = postprocessor.dilate(edges)
        
        Notes:
        ------
        - Dilasi menebalkan tepi PUTIH (255)
        - Latar HITAM (0) mengecil
        - Tepi yang putus akan tersambung jika jarak < strel size
        
        Referensi:
        ----------
        Paper: "Sistem Pendeteksi Kerusakan Jalan Aspal Menggunakan Canny Edge Detection"
               J-ICON Vol. 11 No. 1 (2023)
               Strel: Square 3×3, Iterations: 1
        """
        if edge_image is None:
            raise ValueError("Input edge_image is None.")
        
        # Validasi input
        if len(edge_image.shape) != 2:
            raise ValueError(f"Expected edge image (2D), got shape: {edge_image.shape}")
        
        # Get strel shape
        strel_shape = self.strel_shapes.get(
            self.config.STREL_SHAPE,
            cv2.MORPH_RECT
        )
        
        # Create structure element
        strel = cv2.getStructuringElement(
            strel_shape,
            self.config.STREL_SIZE
        )
        
        # Apply dilation
        dilated = cv2.dilate(
            edge_image,
            strel,
            iterations=self.config.DILATION_ITERATIONS
        )
        
        if self.config.DEBUG_MODE:
            print(f"\n🔧 MORPHOLOGICAL DILATION:")
            print(f"   Strel Shape: {self.config.STREL_SHAPE}")
            print(f"   Strel Size: {self.config.STREL_SIZE}")
            print(f"   Iterations: {self.config.DILATION_ITERATIONS}")
            print(f"   Output shape: {dilated.shape}")
            
            # Hitung perubahan
            before = np.sum(edge_image == 255)
            after = np.sum(dilated == 255)
            increase = after - before
            
            print(f"   Edge pixels before: {before}")
            print(f"   Edge pixels after: {after}")
            print(f"   Increase: {increase} pixels ({(increase/before)*100:.1f}%)")
        
        return dilated
    
    
    def find_contours(self, dilated_image):
        """
        Find contours (connected components) dari dilated image.
        
        Konsep Contours:
        ----------------
        Contour = Kurva yang menghubungkan semua piksel kontinu dengan warna sama
        
        Dalam konteks project ini:
        - Contour = Batas area lubang (setelah dilasi)
        - Setiap lubang = 1 contour
        - Bisa ada multiple contours jika ada multiple lubang
        
        Connectivity:
        -------------
        8-neighbors connectivity:
        ```
        X X X
        X O X  ← O connected to all 8 X
        X X X
        ```
        
        Piksel dianggap connected jika:
        - Horizontal neighbor (kiri/kanan)
        - Vertical neighbor (atas/bawah)
        - Diagonal neighbor (4 corners)
        
        Retrieval Mode:
        ---------------
        RETR_EXTERNAL = Hanya ambil contour terluar
        - Abaikan contour di dalam contour lain (hole dalam hole)
        - Cocok untuk pothole detection (lubang tidak punya hole dalam)
        
        Approximation Method:
        ---------------------
        CHAIN_APPROX_SIMPLE = Compress contour untuk efisiensi
        - Hanya simpan endpoint dari garis lurus
        - Mengurangi jumlah points tanpa kehilangan shape
        - Contoh: Rectangle hanya butuh 4 points (bukan ratusan)
        
        Parameters:
        -----------
        dilated_image : numpy.ndarray
            Input dilated edge image
            Shape: (height, width)
            Dtype: uint8
        
        Returns:
        --------
        contours : list of numpy.ndarray
            List of contours, setiap contour adalah array of points
            Format: [contour1, contour2, ...]
            contour shape: (n_points, 1, 2) dimana 2 = (x, y)
        
        Example:
        --------
        >>> contours = postprocessor.find_contours(dilated)
        >>> print(f"Found {len(contours)} objects")
        
        Notes:
        ------
        - OpenCV findContours memodifikasi input image (deprecated behavior)
        - Kita pass copy agar original tidak berubah
        - Contour coordinate: (x, y) dengan origin di top-left
        """
        if dilated_image is None:
            raise ValueError("Input dilated_image is None.")
        
        # Find contours
        # OpenCV >= 4.x returns (contours, hierarchy)
        # OpenCV < 4.x returns (image, contours, hierarchy)
        contours, hierarchy = cv2.findContours(
            dilated_image.copy(),           # Make copy to avoid modifying original
            cv2.RETR_EXTERNAL,              # Hanya external contours
            cv2.CHAIN_APPROX_SIMPLE         # Compress contours
        )
        
        if self.config.DEBUG_MODE:
            print(f"\n🔍 CONTOUR DETECTION:")
            print(f"   Total contours found: {len(contours)}")
            if len(contours) > 0:
                areas = [cv2.contourArea(c) for c in contours]
                print(f"   Contour areas: min={min(areas):.0f}, max={max(areas):.0f}, mean={np.mean(areas):.0f}")
        
        return contours
    
    
    def filter_contours(self, contours):
        """
        Multi-level filtering untuk membuang objek yang bukan lubang.
        
        Kenapa Perlu Filtering?
        -----------------------
        Setelah find contours, kita punya SEMUA objek terdeteksi:
        ✅ Lubang (yang kita mau)
        ❌ Batu kecil (noise)
        ❌ Kotoran (noise)
        ❌ Bayangan motor (false positive)
        ❌ Marka jalan (false positive)
        ❌ Genangan air (false positive)
        
        Goal filtering: HANYA pertahankan lubang, buang yang lain!
        
        Multi-Level Filtering (3 Filter Sequential):
        ---------------------------------------------
        
        **FILTER 1: SIZE FILTER** 🔍
        
        Kriteria: MIN_SIZE ≤ width, height ≤ MAX_SIZE
        
        Reasoning:
        - Lubang nyata: 15×15 sampai 290×540 piksel (dari paper)
        - Terlalu kecil (< 15×15): Batu, kotoran, noise
        - Terlalu besar (> 290×540): Motor, bayangan besar, bukan lubang
        
        Formula:
        ```
        x, y, w, h = cv2.boundingRect(contour)
        
        if w < MIN_SIZE or h < MIN_SIZE:
            REJECT (terlalu kecil)
        
        if w > MAX_WIDTH or h > MAX_HEIGHT:
            REJECT (terlalu besar)
        
        PASS ✅
        ```
        
        **FILTER 2: ASPECT RATIO FILTER** 📐
        
        Kriteria: AR_MIN ≤ aspect_ratio ≤ AR_MAX
        
        Aspect Ratio (AR) = width / height
        
        Reasoning:
        - Lubang cenderung bulat/oval → AR ≈ 1.0
        - Marka jalan sangat lonjong → AR >> 3.0 atau AR << 0.3
        - Bayangan motor sangat tinggi → AR << 0.3
        
        Range valid: 0.3 ≤ AR ≤ 3.0
        
        Contoh:
        ```
        Square (50×50): AR = 1.0 → PASS ✅
        Circle (diameter=50): AR ≈ 1.0 → PASS ✅
        Oval (80×40): AR = 2.0 → PASS ✅
        Marka (200×10): AR = 20.0 → REJECT ❌
        Motor shadow (30×150): AR = 0.2 → REJECT ❌
        ```
        
        **FILTER 3: SOLIDITY FILTER** 💎
        
        Kriteria: solidity > SOLIDITY_THRESHOLD
        
        Solidity = contour_area / convex_hull_area
        
        Reasoning:
        - Lubang cenderung solid/padat → Solidity > 0.8
        - Ranting, daun, kotoran tidak solid → Solidity < 0.6
        
        Convex Hull = "Rubber band" di sekitar objek
        
        Contoh:
        ```
        Circle: 
          area = πr², hull_area ≈ πr²
          solidity ≈ 1.0 → PASS ✅
        
        L-shape (ranting):
          area = kecil, hull_area = besar
          solidity < 0.5 → REJECT ❌
        ```
        
        Parameters:
        -----------
        contours : list of numpy.ndarray
            List of all contours dari find_contours
        
        Returns:
        --------
        filtered_contours : list of numpy.ndarray
            List of valid contours (lubang saja)
        bounding_boxes : list of tuple
            List of bounding boxes: (x, y, w, h)
        
        Example:
        --------
        >>> contours = postprocessor.find_contours(dilated)
        >>> valid_contours, boxes = postprocessor.filter_contours(contours)
        >>> print(f"{len(contours)} → {len(valid_contours)} after filtering")
        
        Notes:
        ------
        - Filtering sequential: Filter 1 → Filter 2 → Filter 3
        - Objek harus PASS semua filter untuk dianggap lubang
        - Parameter filter dapat di-tuning di config.py
        """
        if not contours:
            if self.config.DEBUG_MODE:
                print("\n⚠️ No contours to filter")
            return [], []
        
        filtered_contours = []
        bounding_boxes = []
        
        # Counters untuk debugging
        rejected_by_size = 0
        rejected_by_ar = 0
        rejected_by_solidity = 0
        
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # ========== FILTER 1: SIZE FILTER ==========
            if w < self.config.MIN_POTHOLE_SIZE or h < self.config.MIN_POTHOLE_SIZE:
                rejected_by_size += 1
                continue  # REJECT: Too small
            
            if w > self.config.MAX_POTHOLE_WIDTH or h > self.config.MAX_POTHOLE_HEIGHT:
                rejected_by_size += 1
                continue  # REJECT: Too large
            
            # ========== FILTER 2: ASPECT RATIO FILTER ==========
            aspect_ratio = w / h if h > 0 else 0
            
            if aspect_ratio < self.config.AR_MIN or aspect_ratio > self.config.AR_MAX:
                rejected_by_ar += 1
                continue  # REJECT: AR out of range
            
            # ========== FILTER 3: SOLIDITY FILTER ==========
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            
            solidity = area / hull_area if hull_area > 0 else 0
            
            if solidity < self.config.SOLIDITY_THRESHOLD:
                rejected_by_solidity += 1
                continue  # REJECT: Not solid enough
            
            # ========== PASSED ALL FILTERS ✅ ==========
            filtered_contours.append(contour)
            bounding_boxes.append((x, y, w, h))
        
        if self.config.DEBUG_MODE:
            print(f"\n📏 MULTI-LEVEL FILTERING:")
            print(f"   Input contours: {len(contours)}")
            print(f"   Rejected by SIZE: {rejected_by_size}")
            print(f"   Rejected by ASPECT RATIO: {rejected_by_ar}")
            print(f"   Rejected by SOLIDITY: {rejected_by_solidity}")
            print(f"   Valid potholes: {len(filtered_contours)} ✅")
        
        return filtered_contours, bounding_boxes
    
    
    def process(self, edge_image):
        """
        Jalankan seluruh pipeline post-processing.
        
        Pipeline:
        1. Dilasi → Tebalkan tepi
        2. Find Contours → Deteksi objek
        3. Multi-level Filtering → Filter lubang valid
        
        Parameters:
        -----------
        edge_image : numpy.ndarray
            Input edge image (hasil Canny)
        
        Returns:
        --------
        filtered_contours : list
            List of valid contours (lubang)
        bounding_boxes : list
            List of bounding boxes: (x, y, w, h)
        dilated_image : numpy.ndarray
            Dilated image (untuk visualisasi)
        
        Example:
        --------
        >>> contours, boxes, dilated = postprocessor.process(edges)
        >>> print(f"Detected {len(boxes)} potholes")
        """
        if self.config.DEBUG_MODE:
            print("\n" + "="*60)
            print("🔧 POST-PROCESSING PIPELINE")
            print("="*60)
        
        # Step 1: Dilasi
        dilated = self.dilate(edge_image)
        
        # Step 2: Find Contours
        contours = self.find_contours(dilated)
        
        # Step 3: Filter Contours
        filtered_contours, bounding_boxes = self.filter_contours(contours)
        
        if self.config.DEBUG_MODE:
            print("="*60 + "\n")
        
        return filtered_contours, bounding_boxes, dilated


# ========== TEST CODE ==========

if __name__ == "__main__":
    """
    Test code untuk memverifikasi post-processing bekerja dengan benar.
    """
    print("🧪 TESTING POST-PROCESSING MODULE")
    print("="*60)
    
    # Create dummy edge image (simulasi hasil Canny)
    print("\n1️⃣ Creating dummy edge image...")
    dummy_edges = np.zeros((480, 640), dtype=np.uint8)
    
    # Buat beberapa objek dengan ukuran berbeda
    # Objek 1: Lubang valid (circle, radius=30)
    cv2.circle(dummy_edges, (150, 150), 30, 255, 2)
    
    # Objek 2: Lubang valid (rectangle, 40×60)
    cv2.rectangle(dummy_edges, (400, 100), (440, 160), 255, 2)
    
    # Objek 3: Noise kecil (circle, radius=5)
    cv2.circle(dummy_edges, (200, 400), 5, 255, -1)
    
    # Objek 4: Objek terlalu besar (rectangle, 300×400)
    cv2.rectangle(dummy_edges, (50, 300), (350, 470), 255, 2)
    
    print(f"   Created 4 objects: 2 valid potholes + 1 noise + 1 too large")
    
    # Initialize post-processing
    print("\n2️⃣ Initializing post-processing...")
    config = Config()
    config.DEBUG_MODE = True
    postprocessor = PostProcessing(config)
    print("   Post-processing initialized ✅")
    
    # Test full pipeline
    print("\n3️⃣ Running full post-processing pipeline...")
    contours, boxes, dilated = postprocessor.process(dummy_edges)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Total objects detected: 4")
    print(f"   Valid potholes after filtering: {len(boxes)}")
    print(f"   Expected: 2 valid potholes")
    
    if len(boxes) == 2:
        print("\n✅ Test result: CORRECT!")
    else:
        print(f"\n⚠️ Test result: Expected 2, got {len(boxes)}")
    
    # Print bounding boxes
    if boxes:
        print(f"\n📦 Bounding Boxes:")
        for i, (x, y, w, h) in enumerate(boxes):
            print(f"   Pothole #{i+1}: (x={x}, y={y}, w={w}, h={h})")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED!")
    print("="*60)