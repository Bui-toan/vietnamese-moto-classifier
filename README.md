# Đề Tài: Phân Loại Dòng Xe Máy Phổ Biến Tại Việt Nam (Phiên Bản v4)

Hệ thống Thị giác Máy tính và Học sâu nhận diện 6 dòng xe máy phổ biến nhất tại Việt Nam (Honda Wave, Honda Vision, Honda Air Blade, Yamaha Exciter, Piaggio Vespa, Honda SH) được xây dựng theo các tiêu chuẩn kỹ thuật nghiêm ngặt nhất: **tự code 100% các thuật toán cốt lõi bằng NumPy**, **khóa dữ liệu kiểm tra rò rỉ chống leak**, **bộ phát hiện xe phụ trợ Detector cho ảnh nhiều xe**, **3 mô hình phân loại**, **quy trình test 17 Test Cases** và **giao diện Web Streamlit 5 tab**.

---

## 📁 Cấu Trúc Dự Án

```
vietnamese-moto-classifier/
├── data/
│   ├── raw/                      # Thư mục chứa ảnh gốc tải về theo từng lớp
│   ├── manifest.csv              # File manifest quản lý toàn bộ ảnh, nhãn, group_id, bbox
│   └── test_frozen/              # Bản sao niêm phong của tập Test chính thức
├── splits/
│   ├── train.csv, val.csv, test.csv
│   ├── ood_val.csv, ood_test.csv # Bộ ảnh ngoài phân phối (OOD)
│   ├── lock.json                 # Khóa SHA-256 niêm phong tập test kèm locked_at UTC
│   └── split_warnings.txt        # Cảnh báo độ lệch tỷ lệ nếu có
├── src/
│   ├── preprocess/               # TỰ CODE 100% BẰNG NUMPY
│   │   ├── transform.py          # letterbox (viền mean), aug_rotate (song tuyến), aug_random_erase, aug_brightness_contrast, aug_bbox_jitter
│   │   └── hog.py                # Trích xuất đặc trưng HOG tự code (6.084 chiều và 1.764 chiều)
│   ├── data/
│   │   ├── split.py              # Thuật toán phân bổ group deficit (70/15/15)
│   │   ├── check_leak.py         # Kiểm tra rò rỉ group, trùng SHA-256 và aHash 8x8 near-duplicate
│   │   ├── lock.py               # Sao chép và tạo file khóa lock.json
│   │   └── verify_lock.py        # Xác minh toàn vẹn SHA-256 từng file ảnh
│   ├── models/
│   │   ├── svm.py                # LinearSVM tự code bằng NumPy (Multiclass Hinge Loss + SGD momentum)
│   │   ├── cnn1.py               # M2 CNN-1 Baseline (3 conv + GAP + FC)
│   │   └── cnn2.py               # M3 CNN-2 ResNet-style (Stem + ResBlocks + Dropout + Grad-CAM)
│   ├── detector/
│   │   └── detector.py           # Bộ phát hiện phụ trợ (Sliding window đa tỉ lệ + HOG + SVM + NMS + Sắp xếp theo hàng)
│   ├── train/
│   │   ├── dataset.py            # PyTorch Dataset nạp ảnh, bbox jitter, augment tự code
│   │   ├── train_m1.py           # Huấn luyện M1 HOG LinearSVM (Grid search + 3 seed)
│   │   ├── train_m2.py           # Huấn luyện M2 CNN-1
│   │   ├── train_m3.py           # Huấn luyện M3 CNN-2 (ResNet + Ablation)
│   │   └── train_detector.py     # Huấn luyện Detector D (Hard-negative mining)
│   └── eval/
│       ├── metrics.py            # Top-1/2/3, Macro-F1, Confusion 6x6, Bootstrap 95% CI, McNemar, AUROC, Latency
│       ├── calibration.py        # Temperature scaling tối ưu NLL, xác định tau_conf, tau_ood, reject_threshold
│       ├── evaluate_test.py      # Pre-flight check, chạy test official 1 lần, xuất io_ledger và acceptance.csv
│       └── run_testcases.py      # Thực thi tự động trọn bộ 17 Test Cases (TC-01 đến TC-17)
├── reports/                      # Gói báo cáo tự sinh cho từng model
│   ├── m1_hog_svm/               # config.json, env.json, history.csv, learning_curve.png, seeds.csv, complexity.json, val_metrics.json, model_card.json
│   ├── m2_cnn1/
│   ├── m3_cnn2/
│   ├── ablation/                 # So sánh M2 vs M3 (residual), M3 có vs không augment
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
├── tests/                        # Hệ thống Unit Test tự động (test_core, test_data_pipeline, test_models, test_metrics, test_ui)
├── app/                          # Giao diện Streamlit Demo 5 tab
│   ├── app.py
│   └── ui/                       # theme.py, draw.py (vẽ khung đánh số), explain.py (Grad-CAM)
└── best_config.json              # Cấu hình mô hình tốt nhất phục vụ suy luận
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
Chạy huấn luyện lần lượt các mô hình:
```bash
# Huấn luyện M1: HOG + LinearSVM
python src/train/train_m1.py

# Huấn luyện M2: CNN-1 Baseline
python src/train/train_m2.py

# Huấn luyện M3: CNN-2 ResNet-style
python src/train/train_m3.py

# Huấn luyện Detector D (phụ trợ cho ảnh nhiều xe)
python src/train/train_detector.py

# Tổng hợp kết quả và chọn model tối ưu nhất
python reports/build_training_summary.py
```

### Bước 4: Đánh giá Kiểm thử Chính thức (Test Evaluation)
```bash
# Kiểm tra pre-flight gate và đánh giá tập test chính thức (chỉ chạy 1 lần)
python src/eval/evaluate_test.py

# Chạy trọn bộ 17 Test Cases tự động (xuất test_case_report.csv)
python src/eval/run_testcases.py
```

### Bước 5: Khởi chạy Giao diện Demo Streamlit
```bash
streamlit run app/app.py
```

---

## 🎯 Kiểm Tra Đối Chiếu Checklist Nghiệm Thu (Mục 9)

- [x] **lock.json**: Khớp từng ảnh test và OOD; thứ tự `locked_at < created_at < test_time`; `OFFICIAL_RUN.lock` tồn tại.
- [x] **Gói báo cáo mục 5.2**: Đầy đủ cho cả 3 model trong `reports/`, kèm thư mục `reports/ablation/` và `training_summary.md`.
- [x] **Sổ I/O Ledger**: `evidence/io_ledger_*.csv` đầy đủ cho 3 model, đối chiếu đầu vào/đầu ra từng ảnh.
- [x] **Test Cases**: `evidence/test_case_report.csv` đầy đủ 17 case, trạng thái PASS 100%.
- [x] **Báo cáo thống kê**: Khoảng tin cậy Bootstrap 95%, kiểm định McNemar chính xác, ma trận nhầm lẫn 6x6 + cột Từ chối, Coverage-Accuracy curve, AUROC OOD.
- [x] **Detector phụ trợ**: Sliding window đa tỉ lệ, NMS, sắp xếp đánh số theo hàng (trên $\to$ dưới, trái $\to$ phải) kèm chú thích bên dưới ảnh.
- [x] **Streamlit UI 5 tab**: Hoạt động mượt mà, theme tối Navy/Teal, checkbox "Ảnh có nhiều xe", Grad-CAM, sliders thử độ bền real-time, ảnh chụp màn hình bằng chứng tại `evidence/screenshots/tc16_multi_motor.png`.
- [x] **README.md**: Hướng dẫn tái lập toàn bộ quy trình từ cài đặt đến huấn luyện và chạy app.
