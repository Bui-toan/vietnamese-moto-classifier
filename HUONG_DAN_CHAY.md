# HƯỚNG DẪN THỰC THI DỰ ÁN TỪ ĐẦU ĐẾN CUỐI (STEP-BY-STEP GUIDE)
## Đề Tài: Phân Loại Dòng Xe Máy Phổ Biến Tại Việt Nam (Bản v4)

Tài liệu này hướng dẫn chi tiết từng bước thực thi toàn bộ pipeline của dự án: từ chuẩn bị dữ liệu, chia tập và khóa chống rò rỉ, huấn luyện mô hình, kiểm thử nghiệm thu 17 Test Cases đến khởi chạy giao diện Web Demo Streamlit.

---

## 📋 TỔNG QUAN CÁC BƯỚC THỰC HIỆN

| Bước | Tên giai đoạn | Mục tiêu chính | Đầu ra quan trọng |
| :---: | :--- | :--- | :--- |
| **1** | Chuẩn bị môi trường & Dữ liệu | Cài đặt thư viện, đặt ảnh vào thư mục và tạo manifest | `requirements.txt`, `data/manifest.csv` |
| **2** | Chia tập, Leak Check & Khóa | Phân chia 70/15/15 theo deficit, quét near-duplicate, khóa SHA-256 | `splits/`, `lock.json`, `data/test_frozen/` |
| **3** | Huấn luyện mô hình & Detector | Train M1 (SVM), M2 (CNN-1), M3 (CNN-2 ResNet), Detector phụ trợ | `models/`, `reports/`, `best_config.json` |
| **4** | Đánh giá Test chính thức 1 lần | Pre-flight check, chạy test official, ghi sổ I/O Ledger, 17 Test Cases | `evidence/`, `test_case_report.csv`, `acceptance.csv` |
| **5** | Khởi chạy Web Demo UI | Demo 5 tab: Nhận diện (đánh số 1,2,3), So sánh, Bằng chứng test, Thử độ bền | Giao diện Streamlit tại `localhost:8501` |

---

## 🛠️ BƯỚC 1: Cài Đặt Môi Trường & Chuẩn Bị Dữ Liệu

### 1.1. Cài đặt các thư viện Python
Mở terminal (PowerShell trên Windows) tại thư mục gốc của dự án (`c:\vietnamese-moto-classifier`) và thực thi:
```bash
pip install -r requirements.txt
```

### 1.2. Tổ chức ảnh và Quy trình Resize 2 lớp (Offline + Online)
- Đặt ảnh gốc thu thập được vào thư mục `data/raw/` (chia theo từng lớp hoặc đặt chung).
- **Offline Resize (Chạy 1 lần duy nhất trước khi train)**:
  Hạ kích thước toàn bộ ảnh gốc có cạnh dài > 800px về tối đa 800px (dùng `cv2.INTER_AREA`, lưu JPEG quality 85) để giảm 90% dung lượng đĩa và tăng tốc độ đọc của DataLoader gấp 5 lần:
  ```bash
  python src/data/resize_raw.py --raw-dir data/raw --processed-dir data/processed
  ```
  *(Khuyên dùng: Resize ảnh xong mới mở LabelImg để gán nhãn trên ảnh 800px trong `data/processed/` cho nhàn).*
- Chuẩn bị file danh sách dữ liệu **`data/manifest.csv`** trỏ vào các ảnh đã resize:
```csv
image_path,label,group_id,bbox
data/processed/wave/img_001.jpg,Honda Wave,wave_locA_01,"[15, 20, 225, 300]"
data/processed/vision/img_002.jpg,Honda Vision,vis_locB_01,"[10, 25, 200, 290]"
data/processed/airblade/img_003.jpg,Honda Air Blade,ab_locC_01,"[12, 18, 210, 295]"
data/processed/exciter/img_004.jpg,Yamaha Exciter,ex_locD_01,"[15, 22, 215, 305]"
data/processed/vespa/img_005.jpg,Piaggio Vespa,vespa_locE_01,"[10, 15, 205, 285]"
data/processed/sh/img_006.jpg,Honda SH,sh_locF_01,"[18, 28, 230, 310]"
data/processed/ood/img_999.jpg,ood,ood_street_01,""
```

> **Quy tắc về Group ID**:
> - Cùng một chiếc xe chụp nhiều góc hoặc trong cùng một buổi chụp phải có cùng 1 `group_id`.
> - Ảnh mạng: tên miền + bài đăng/album.
> - Mỗi lớp phải có tối thiểu 10 group khác nhau để chống rò rỉ và học vẹt background.

---

## 🔒 BƯỚC 2: Phân Chia Tập Dữ Liệu, Kiểm Tra Rò Rỉ & Niêm Phong Khóa

### 2.1. Phân chia tập dữ liệu 70/15/15 theo giải thuật Deficit
Chạy script phân bổ group tự động để chia đều tỉ lệ 70% Train, 15% Validation, 15% Test:
```bash
python src/data/split.py --manifest data/manifest.csv --seed 42
```
*Kết quả:* Các file `splits/train.csv`, `splits/val.csv`, `splits/test.csv`, `splits/ood_val.csv`, `splits/ood_test.csv` được tạo ra.

### 2.2. Kiểm tra rò rỉ dữ liệu (Leak Check & Near-duplicate)
Kiểm tra rò rỉ group, trùng khớp tuyệt đối SHA-256 và phát hiện ảnh gần trùng bằng thuật toán **Average Hash $8 \times 8$ tự code bằng NumPy** (khoảng cách Hamming $\le 5$):
```bash
python src/data/check_leak.py
```
*Lưu ý:* Nếu có ảnh gần trùng, script sẽ lưu vào `splits/near_duplicates.csv`. Hãy loại bỏ bản trùng ở tập val/test rồi chạy lại cho đến khi xuất hiện thông báo `[PASS] CLEAN`.

### 2.3. Niêm phong tập Test và Khóa dữ liệu
Sao chép toàn bộ ảnh test sang thư mục `data/test_frozen/` và ghi mã băm SHA-256 vào `splits/lock.json` kèm mốc thời gian `locked_at` (UTC):
```bash
python src/data/lock.py
```

### 2.4. Xác minh khóa dữ liệu
Xác thực lại toàn bộ mã băm để kiểm tra tính toàn vẹn:
```bash
python src/data/verify_lock.py --lock splits/lock.json --data-dir data/test_frozen
```

---

## 🧠 BƯỚC 3: Huấn Luyện Các Mô Hình & Trích Xuất Báo Cáo Tự Động

Thực thi tuần tự các script huấn luyện (mỗi script tự động chạy Grid Search trên seed 42, chạy 3 seed để tính độ lệch chuẩn, hiệu chỉnh Temperature Scaling và sinh trọn gói báo cáo trong `reports/`):

### 3.1. Huấn luyện Mô hình M1: HOG + LinearSVM (Bắt buộc)
```bash
python src/training/train_m1.py
```
*Đầu ra:*
- Lưu vector đặc trưng trích xuất: `features/hog_train_aug.npz`.
- Trọng số và thông số chuẩn hóa: `models/m1_hog_svm.npz`, `models/hog_stats.npz`.
- Gói báo cáo: `reports/m1_hog_svm/` (`gridsearch.csv`, `history.csv`, `learning_curve.png`, `seeds.csv`, `complexity.json`, `val_metrics.json`, `model_card.json`).

### 3.2. Huấn luyện Mô hình M2: CNN-1 Baseline
```bash
python src/training/train_m2.py
```
*Đầu ra:* Trọng số `models/m2_cnn1.pt` và gói báo cáo `reports/m2_cnn1/`.

### 3.3. Huấn luyện Mô hình M3: CNN-2 ResNet-style (Khuyến nghị chính)
```bash
python src/training/train_m3.py
```
*Đầu ra:* Trọng số `models/m3_cnn2.pt` và gói báo cáo `reports/m3_cnn2/`.

*(Tùy chọn: Chạy thêm mô hình không Augmentation để phục vụ báo cáo Ablation Study):*
```bash
python src/training/train_m3.py --no-aug
```

### 3.4. Huấn luyện Bộ phát hiện phụ trợ Detector D
```bash
python src/training/train_detector.py
```
*Đầu ra:* Trọng số `models/detector_svm.npz` (LinearSVM nhị phân trên HOG $64 \times 64$ qua 1 vòng Hard-negative mining).

### 3.5. Tổng hợp so sánh các mô hình & Lựa chọn Best Model
Chạy script tổng hợp để so sánh Macro-F1 giữa các mô hình và khóa checkpoint:
```bash
python reports/build_training_summary.py
```
*Đầu ra:*
- Bảng Markdown dán thẳng vào báo cáo: `reports/training_summary.md` và `reports/training_summary.csv`.
- File cấu hình mô hình tốt nhất phục vụ suy luận: `best_config.json` (kèm timestamp `created_at` ISO-8601 UTC).
- Đặt trạng thái `frozen = true` cho cả 3 mô hình trong `model_card.json`.

---

## 📊 BƯỚC 4: Kiểm Thử Đánh Giá Chính Thức & Nghiệm Thu (Strict Test)

> **Lưu ý quan trọng trước khi chạy lần đầu trên dữ liệu thật:**
> Hãy xóa file khóa cũ (nếu có từ lần chạy thử trước đó) để hệ thống ghi nhận đây là lần chạy chính thức đầu tiên:
> ```powershell
> Remove-Item -Force evidence/OFFICIAL_RUN.lock -ErrorAction SilentlyContinue
> ```

### 4.1. Đánh giá Test chính thức (Official Test Run)
Chạy script đánh giá chính thức đúng 1 lần:
```bash
python src/evaluation/evaluate_test.py
```
*Quy trình tự động thực thi:*
1. **Pre-flight Gate**: Kiểm tra toàn bộ mã băm từng file ảnh test khớp `lock.json`; kiểm tra thứ tự thời gian nghiêm ngặt: $\text{locked\_at} < \text{created\_at} < \text{test\_time}$; kiểm tra git sạch.
2. **Khóa chạy 1 lần**: Tạo `evidence/OFFICIAL_RUN.lock`.
3. **Ghi sổ I/O Ledger**: Lưu chi tiết từng ảnh test vào `evidence/io_ledger_<model>.csv` (đầu vào, ground-truth, xác suất 6 lớp sau calibration, trạng thái rejected, latency ms).
4. **Phân tích thống kê & Bằng chứng**:
   - Khoảng tin cậy Bootstrap 95% (1.000 lần).
   - Kiểm định McNemar chính xác (`mcnemar.csv`).
   - Ma trận nhầm lẫn $6 \times 6$ kèm cột phụ "Từ chối" (`confusion_*.png`, `.csv`).
   - Đường cong Coverage – Accuracy (`coverage_accuracy.png`, `.csv`).
   - Đo tốc độ CPU: Cold start và Warm start p50, p95 (`latency.json`).
5. **Bảng nghiệm thu tự động**: Xuất kết quả kiểm tra các chỉ tiêu đề tài vào `evidence/acceptance.csv`.

### 4.2. Chạy tự động bộ 17 Test Cases (TC-01 đến TC-17)
Kiểm thử toàn diện từ Unit test thuật toán, kiểm tra rò rỉ dữ liệu, test giả mạo sửa 1 byte, 4 bài test độ bền Robustness (Blur, Xoay, Che, Bbox rộng), OOD test và UI test:
```bash
python src/evaluation/run_testcases.py
```
*Đầu ra:* File `evidence/test_case_report.csv` với trạng thái `PASS/FAIL` đầy đủ cho cả 17 test cases.

---

## 🌐 BƯỚC 5: Khởi Chạy Giao Diện Web Demo Streamlit

Khởi chạy ứng dụng Web:
```bash
streamlit run app/app.py
```
Trình duyệt sẽ tự động mở giao diện tại: `http://localhost:8501`.

### Kịch bản Demo 3 Phút Chuẩn Khi Bảo Vệ Đồ Án:
1. **Tab 1: Nhận diện**:
   - Tải lên ảnh 1 chiếc xe (ví dụ Honda Wave) với checkbox *"Ảnh có nhiều xe"* đang **TẮT** $\to$ Hệ thống letterbox toàn ảnh và trả về kết quả dự đoán kèm bản đồ nhiệt Grad-CAM ngay lập tức.
   - Tải lên ảnh bãi xe hoặc ảnh đường phố có 2–3 xe và **BẬT checkbox "Ảnh có nhiều xe"** $\to$ Bộ phát hiện phụ trợ Detector sẽ quét, đóng khung từng xe, **đánh số thứ tự 1, 2, 3... theo hàng (từ trên xuống, từ trái sang)** và hiển thị bảng chú thích bên dưới ảnh (ví dụ: *"1 - Honda SH (92%) | 2 - Honda Vision (81%)"*).
   - Tải lên ảnh xe máy ngoài 6 lớp (ví dụ xe Winner, Lead) $\to$ Điểm tin cậy $< \text{reject\_threshold}$, hệ thống hiển thị trạng thái: *"Không đủ thông tin để nhận diện"*.
2. **Tab 2: So sánh 3 mô hình**:
   - Chọn một xe trong ảnh để đối chiếu kết quả dự đoán của M1 (HOG + SVM), M2 (CNN-1) và M3 (CNN-2) đặt cạnh nhau, highlight model chính xác nhất và nhanh nhất.
3. **Tab 3: Kết quả kiểm thử (Chỉ đọc)**:
   - Cho giảng viên xem ma trận nhầm lẫn $6 \times 6$ có cột "Từ chối", đường cong Coverage-Accuracy và bảng nghiệm thu tiêu chí đề tài (chỉ đọc trực tiếp từ thư mục `evidence/`, không có nút chạy lại).
4. **Tab 4: Thử độ bền (Real-time)**:
   - Kéo các thanh trượt xoay góc, làm mờ, che phủ để thấy kết quả dự đoán thay đổi theo thời gian thực (chứng minh các thuật toán xoay và che được viết bằng NumPy thuần).
5. **Tab 5: Giới thiệu đề tài**:
   - Trưng bày thông số kỹ thuật, bảng so sánh tham số/thời gian train từ `training_summary.md`, phân tích hạn chế của hàm Softmax với OOD và hướng phát triển.

---

## 📌 BẢNG TỔNG HỢP CÁC LỆNH CHẠY NHANH (QUICK REFERENCE)

```bash
# 1. Phân chia dữ liệu & Kiểm tra rò rỉ
python src/data/split.py --manifest data/manifest.csv --seed 42
python src/data/check_leak.py
python src/data/lock.py

# 2. Huấn luyện 3 mô hình & Detector
python src/training/train_m1.py
python src/training/train_m2.py
python src/training/train_m3.py
python src/training/train_detector.py
python reports/build_training_summary.py

# 3. Đánh giá Test chính thức & 17 Test cases
python src/evaluation/evaluate_test.py
python src/evaluation/run_testcases.py

# 4. Bật Giao diện Web
streamlit run app/app.py
```
