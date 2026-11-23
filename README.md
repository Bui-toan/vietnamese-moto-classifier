# Đề Tài: Phân Loại Dòng Xe Máy Phổ Biến Tại Việt Nam (Phiên Bản v4)

Hệ thống Thị giác Máy tính và Học sâu nhận diện 6 dòng xe máy phổ biến nhất tại Việt Nam (Honda Wave, Honda Vision, Honda Air Blade, Yamaha Exciter, Piaggio Vespa, Honda SH) được xây dựng theo tiêu chuẩn kỹ thuật phần mềm sạch (Clean Architecture & Modular Design):
- **Tự code 100% các thuật toán cốt lõi bằng NumPy** (không dùng `cv2.warpAffine`, `cv2.convertScaleAbs`, `cv2.img_hash` hay `sklearn.svm`).
- **Khóa dữ liệu & Kiểm tra rò rỉ** bằng mã băm SHA-256 trên Windows và Average Hash $8 \times 8$.
- **Bộ phát hiện phụ trợ Detector** đóng khung và đánh số thứ tự theo hàng ($1, 2, 3...$) cho ảnh nhiều xe.
- **3 mô hình phân loại** (M1 HOG+LinearSVM, M2 CNN-1 Baseline, M3 CNN-2 ResNet-style).
- **Quy trình kiểm thử nghiêm ngặt** với trọn bộ 17 Test Cases (TC-01 đến TC-17) tự động.
- **Giao diện Web Streamlit 5 tab** trực quan, chuyên nghiệp.

---

## 📁 Cấu Trúc Mã Nguồn Module Hóa Chi Tiết

```
vietnamese-moto-classifier/
├── data/
│   ├── raw/                      # Thư mục chứa ảnh gốc
│   ├── manifest.csv              # File manifest quản lý: image_path, label, group_id, bbox
│   └── test_frozen/              # Bản sao niêm phong của tập Test chính thức
├── splits/
│   ├── train.csv, val.csv, test.csv
│   ├── ood_val.csv, ood_test.csv # Bộ ảnh ngoài phân phối (OOD)
│   ├── lock.json                 # Khóa SHA-256 niêm phong tập test kèm locked_at UTC
│   └── split_warnings.txt        # Cảnh báo độ lệch tỷ lệ nếu có
│
├── src/
│   ├── preprocess/               # 1. TIỀN XỬ LÝ (TỰ CODE NUMPY)
│   │   ├── bbox_utils.py         # Cắt ảnh, tính IoU dùng chung cho Detector và Jitter bbox
│   │   ├── letterbox.py          # Resize giữ tỉ lệ, đệm viền bằng giá trị trung bình (TC-01)
│   │   ├── transforms.py         # Hàm tự code: xoay song tuyến (TC-03), che, sáng/tối, blur
│   │   ├── jitter.py             # Augmentation jitter bbox (mở rộng 0-20%, dịch tâm 10%)
│   │   ├── normalize.py          # Chuẩn hóa Z-score mean/std, lưu/nạp file stats .npz
│   │   └── hog.py                # Trích xuất vector đặc trưng HOG tự code 6.084 chiều (TC-02)
│   │
│   ├── models/                   # 2. KIẾN TRÚC MÔ HÌNH
│   │   ├── svm.py                # LinearSVM tự code bằng NumPy (Multiclass Hinge Loss + SGD momentum)
│   │   ├── cnn1.py               # M2 CNN-1 Baseline (3 conv + GAP + FC 128 -> 6)
│   │   └── cnn2.py               # M3 CNN-2 ResNet-style (Stem + ResBlocks + Dropout 0.3 + Grad-CAM)
│   │
│   ├── detector/                 # 3. BỘ PHÁT HIỆN PHỤ TRỢ (DETECTOR)
│   │   └── detector.py           # Sliding window đa tỉ lệ, HOG 64x64, SVM nhị phân, NMS, sắp xếp theo hàng
│   │
│   ├── training/                 # 4. HUẤN LUYỆN & GRID SEARCH
│   │   ├── losses.py             # Tự code MulticlassHingeLoss (SVM) và LabelSmoothingCrossEntropy (CNN)
│   │   ├── calibration.py        # Temperature Scaling tối ưu NLL và xác định 3 ngưỡng (tau_conf, tau_ood, reject)
│   │   ├── grid_search.py        # Duyệt tổ hợp siêu tham số ở seed 42
│   │   ├── trainer.py            # Class TorchTrainer: epoch loop, early stopping, history.csv, learning_curve
│   │   ├── train_m1.py           # Script train M1: HOG -> grid search -> 3 seeds -> calibration
│   │   ├── train_m2.py           # Script train M2: CNN-1 -> Trainer -> calibration
│   │   ├── train_m3.py           # Script train M3: CNN-2 -> Trainer -> CosineAnnealing -> ablation
│   │   └── train_detector.py     # Script train Sliding Window SVM (mẫu âm + hard negative)
│   │
│   └── evaluation/               # 5. ĐÁNH GIÁ & KIỂM ĐỊNH NGHIÊM NGẶT
│       ├── metrics.py            # Top-1/2/3, Macro-F1, Confusion matrix 6x6 + cột Từ chối, ECE, AUROC
│       ├── statistics.py         # Bootstrap 1.000 lần tính CI 95%, kiểm định McNemar chính xác
│       ├── preflight_check.py    # Cổng kiểm tra: hash verify_lock, timestamp locked_at < created_at, git
│       ├── ledger.py             # Ghi sổ io_ledger_<model>.csv chi tiết cho từng ảnh test
│       ├── robustness.py         # Chạy các bài test độ bền: Blur, Xoay, Che, BBox rộng (TC-10 đến TC-13)
│       ├── evaluate_test.py      # Script đánh giá tập test chính thức (chỉ chạy 1 lần, xuất acceptance.csv)
│       └── run_testcases.py      # Thực thi trọn bộ 17 Test Cases tự động (xuất test_case_report.csv)
│
├── reports/                      # Gói báo cáo tự sinh cho từng model
│   ├── m1_hog_svm/               # config, env, history, learning_curve, seeds, complexity, val_metrics, model_card
│   ├── m2_cnn1/
│   ├── m3_cnn2/
│   ├── ablation/                 # Ablation M2 vs M3 (residual), M3 có vs không augment
│   ├── training_summary.csv      # Bảng tổng hợp so sánh các mô hình
│   └── training_summary.md
├── evidence/                     # Bằng chứng kiểm thử chính thức
│   ├── OFFICIAL_RUN.lock         # Khóa niêm phong chạy 1 lần
│   ├── io_ledger_*.csv           # Sổ ghi log từng ảnh đầu vào/đầu ra
│   ├── metrics_*.csv, confusion_*.png
│   ├── mcnemar.csv, coverage_accuracy.csv, latency.json
│   ├── test_case_report.csv      # Bảng kết quả 17 Test Cases (PASS/FAIL)
│   ├── acceptance.csv            # Bảng nghiệm thu tiêu chí đề tài
│   └── screenshots/              # Ảnh chụp màn hình bằng chứng cho UI
├── tests/                        # Hệ thống Unit Test tự động (17 bài test độc lập)
├── app/                          # Giao diện Streamlit Demo 5 tab
│   ├── app.py
│   └── ui/                       # theme.py, draw.py (vẽ khung đánh số), explain.py (Grad-CAM)
└── best_config.json              # Cấu hình mô hình tối ưu nhất phục vụ suy luận
```

---

## 🚀 Hướng Dẫn Tái Lập Toàn Bộ Dự Án Từ Đầu (Reproducibility)

### Bước 1: Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### Bước 2: Chuẩn bị Dữ liệu và Phân chia Split (70/15/15)
Đặt danh sách ảnh vào `data/manifest.csv` (gồm các cột: `image_path, label, group_id, bbox`). Sau đó chạy:
```bash
# Phân chia dữ liệu theo deficit (ràng buộc group không rò rỉ)
python src/data/split.py --manifest data/manifest.csv --seed 42

# Kiểm tra rò rỉ và tìm ảnh gần trùng (aHash 8x8)
python src/data/check_leak.py

# Niêm phong và khóa dữ liệu tập test
python src/data/lock.py
```

### Bước 3: Huấn luyện Mô hình
Chạy huấn luyện lần lượt các mô hình qua các script modular trong `src/training/`:
```bash
# Huấn luyện M1: HOG + LinearSVM
python src/training/train_m1.py

# Huấn luyện M2: CNN-1 Baseline
python src/training/train_m2.py

# Huấn luyện M3: CNN-2 ResNet-style
python src/training/train_m3.py

# Huấn luyện Detector D (phụ trợ cho ảnh nhiều xe)
python src/training/train_detector.py

# Tổng hợp kết quả và chọn model tối ưu nhất
python reports/build_training_summary.py
```

### Bước 4: Đánh giá Kiểm thử Chính thức (Test Evaluation)
```bash
# Kiểm tra pre-flight gate và đánh giá tập test chính thức (chỉ chạy 1 lần)
python src/evaluation/evaluate_test.py

# Chạy trọn bộ 17 Test Cases tự động (xuất test_case_report.csv)
python src/evaluation/run_testcases.py
```

### Bước 5: Khởi chạy Giao diện Demo Streamlit
```bash
streamlit run app/app.py
```

---

## 🎯 Bảng Nghiệm Thu 17 Test Cases (Mục 6.3)

| ID | Loại | Nội dung kiểm tra | Kỳ vọng | Trạng thái |
| :---: | :---: | :--- | :--- | :---: |
| **TC-01** | Unit | `letterbox` ảnh $640 \times 480$ | Shape $(112, 112, 3)$, giữ tỉ lệ, viền đệm giá trị trung bình | **PASS** |
| **TC-02** | Unit | HOG $8 \times 8, 16 \times 16, 112 \times 112$ | $8 \times 8$ báo lỗi; $16 \times 16 \to 36$; $112 \times 112 \to 6.084$ không NaN | **PASS** |
| **TC-03** | Unit | Xoay $0^\circ$, rồi $+\theta, -\theta$ | PSNR $> 30\text{ dB}$ vùng lõi bằng nội suy song tuyến tự code | **PASS** |
| **TC-04** | Unit | Metrics trên ma trận tính tay | Accuracy, Macro-F1, Top-k khớp tuyệt đối | **PASS** |
| **TC-05** | Data | `check_leak`, near-duplicate | Không rò rỉ group; aHash $8 \times 8$ phát hiện near-duplicate | **PASS** |
| **TC-06** | Data | Sửa 1 byte trong bản sao ảnh test | Pre-flight `verify_lock` bắt lỗi sai hash lập tức (FAIL) | **PASS** |
| **TC-07** | Model | Chạy M1, M2, M3 trên test | Đủ metric, sổ I/O ledger cho 3 model | **PASS** |
| **TC-08** | Model | So sánh 3 model | Bảng + McNemar + Delta Macro-F1 có CI bootstrap cặp | **PASS** |
| **TC-09** | Model | Ma trận nhầm lẫn $6 \times 6$ + Từ chối | File ảnh và CSV cho cả 3 model | **PASS** |
| **TC-10** | Robust | Làm mờ $\sigma \in \{1, 2, 3\}$ | Giảm Top-1 $\le 8$ điểm % tại $\sigma=2$ | **PASS** |
| **TC-11** | Robust | Xoay $\pm 5, 10, 15^\circ$ | Giảm $\le 10$ điểm % tại $\pm 10^\circ$ | **PASS** |
| **TC-12** | Robust | Che phủ $10, 20, 30\%$ | Giảm $\le 15$ điểm % tại $20\%$ | **PASS** |
| **TC-13** | Robust | BBox rộng thêm $15, 30\%$ | BBox rộng $15\%$ giảm $\le 10$ điểm % | **PASS** |
| **TC-14** | OOD | Bộ OOD-test (xe ngoài 6 lớp) | Chỉ tiêu tham khảo: $\ge 80\%$ từ chối | **PASS** |
| **TC-15** | Detector | Bộ ảnh nhiều xe (test) | Tối thiểu Recall $\ge 0.50$, Precision $\ge 0.40$ | **PASS** |
| **TC-16** | UI | Streamlit AppTest: ảnh 1 xe, ảnh 2 xe | Không crash, tiếng Việt, ảnh 2 xe vẽ đúng 2 khung đánh số | **PASS** |
| **TC-17** | Hiệu năng | 100 lần suy luận trên CPU | $p50$ và $p95 \le 1.0\text{ s}$ | **PASS** |
