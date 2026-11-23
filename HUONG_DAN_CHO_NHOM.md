# HƯỚNG DẪN DÀNH CHO THÀNH VIÊN KHI PULL DỰ ÁN VỀ (TEAM ONBOARDING)
## Đề Tài: Phân Loại Dòng Xe Máy Phổ Biến Tại Việt Nam (Bản v4)

Chào các thành viên trong nhóm! Tài liệu này hướng dẫn cách chạy và sử dụng mã nguồn của dự án sau khi bạn `git pull` hoặc `git clone` từ kho Git về máy tính của mình.

Hướng dẫn được chia làm **2 PHẦN**:
- **PHẦN 1: Giai đoạn HIỆN TẠI (Chưa có dữ liệu ảnh thật, chạy thử nghiệm pipeline)**.
- **PHẦN 2: Giai đoạn ĐÃ CÓ DỮ LIỆU & TRAIN XONG (Chạy đánh giá chính thức và nộp bài)**.

---

## 🛠️ CÀI ĐẶT MÔI TRƯỜNG CHUNG (Áp dụng cho cả 2 phần)

Sau khi kéo code về máy, mở Terminal/PowerShell tại thư mục gốc của dự án (`vietnamese-moto-classifier`) và chạy:

```bash
# 1. Tạo môi trường ảo (khuyến nghị, nếu chưa có)
python -m venv .venv
# Trên Windows:
.venv\Scripts\activate

# 2. Cài đặt toàn bộ thư viện cần thiết
pip install -r requirements.txt
```

---

# PHẦN 1: KHI CHƯA CÓ DATA ẢNH THẬT (GIAI ĐOẠN HIỆN TẠI)
> **Mục tiêu**: Làm quen với hệ thống, kiểm tra tính toàn vẹn của mã nguồn, chạy thử nghiệm toàn bộ thuật toán tự code bằng NumPy, và xem giao diện Web Demo hoạt động.

Trong code đã được tích hợp sẵn cơ chế **Mock/Dummy Data** tự sinh các ma trận ảnh và vector toán học để chạy thông suốt toàn bộ quy trình mà không bị báo lỗi thiếu file.

### 1.1. Chạy kiểm tra Unit Test các thuật toán cốt lõi
Chạy bộ 17 Unit Tests tự động để xác nhận toàn bộ các thuật toán tự code (xoay ảnh song tuyến, HOG 6.084 chiều, letterbox đệm trung bình, SVM, CNN, aHash 8x8) hoạt động chuẩn xác:
```bash
python -m unittest discover tests
```
*Kết quả kỳ vọng:* Xuất hiện thông báo `Ran 17 tests ... OK (ALL PASS)`.

### 1.2. Chạy thử nghiệm bộ 17 Test Cases nghiệm thu
```bash
python src/evaluation/run_testcases.py
```
*Kết quả:* Hệ thống tự sinh bảng `evidence/test_case_report.csv` với 17/17 test cases đạt trạng thái `PASS`.

### 1.3. Khởi chạy Giao diện Web Streamlit Demo (Xem trước)
Bạn có thể mở ngay ứng dụng Web để trải nghiệm các tính năng:
```bash
streamlit run app/app.py
```
- Trình duyệt sẽ mở tại: `http://localhost:8501`.
- **Trải nghiệm ngay**:
  - Chọn một bức ảnh bất kỳ từ máy hoặc chọn mục **"Ảnh mẫu có sẵn"** ở cột bên trái.
  - Bật/tắt checkbox **"Ảnh có nhiều xe"** để xem bộ phát hiện phụ trợ Detector đóng khung và đánh số 1, 2, 3...
  - Sang **Tab 4 (Thử độ bền)**: Kéo thử các thanh trượt xoay, làm mờ, che phủ để thấy kết quả cập nhật thời gian thực.

---

# PHẦN 2: KHI ĐÃ CÓ ẢNH THẬT & TIẾN HÀNH TRAIN ĐỂ ĐẨY LÊN
> **Mục tiêu**: Thực hiện đúng quy trình 6 cổng kiểm soát (G0 - G5) của bản kế hoạch v4 để huấn luyện mô hình trên ảnh thật, niêm phong khóa dữ liệu, đánh giá chính thức đúng 1 lần và đẩy kết quả lên Git.

Khi nhóm đã thu thập xong ảnh (khoảng 200 - 700 ảnh/lớp), hãy làm theo các bước chuẩn sau:

### 2.1. Đặt ảnh và chuẩn bị file `data/manifest.csv`
1. Đặt ảnh gốc thu thập được vào thư mục `data/raw/` (chia theo các lớp: `wave/`, `vision/`, `airblade/`, `exciter/`, `vespa/`, `sh/`, và `ood/`).
2. **Quy trình Resize 2 lớp (Bắt buộc để chống tràn RAM và tăng tốc DataLoader gấp 5 lần)**:
   - Chạy script hạ kích thước ảnh gốc về tối đa 800px:
   ```bash
   python src/data/resize_raw.py --raw-dir data/raw --processed-dir data/processed
   ```
   *Lưu ý:* Nên resize ảnh trước rồi mới dùng LabelImg gán nhãn trên ảnh `data/processed/` để tọa độ bbox khít và nhàn nhất. Nếu đã lỡ gán nhãn trên ảnh gốc, script `resize_raw.py` sẽ tự động nhân tỉ lệ để cập nhật lại tọa độ bbox sang `data/manifest_processed.csv`.
3. Điền thông tin vào file `data/manifest.csv` (hoặc dùng `data/manifest_processed.csv`):
```csv
image_path,label,group_id,bbox
data/processed/wave/img_001.jpg,Honda Wave,wave_locA_01,"[15, 20, 225, 300]"
data/processed/vision/img_002.jpg,Honda Vision,vis_locB_01,"[10, 25, 200, 290]"
...
data/processed/ood/img_999.jpg,ood,ood_street_01,""
```
*(Lưu ý: Bắt buộc điền đúng `group_id` cho từng chiếc xe cụ thể để tránh rò rỉ dữ liệu giữa train và test).*

### 2.2. Phân chia tập dữ liệu & Kiểm tra rò rỉ
```bash
# Chia tập 70% Train, 15% Val, 15% Test theo thuật toán deficit
python src/data/split.py --manifest data/manifest.csv --seed 42

# Quét rò rỉ group, trùng SHA-256 và phát hiện ảnh gần trùng (Hamming <= 5)
python src/data/check_leak.py
```
*Lưu ý:* Nếu có ảnh bị báo gần trùng, xử lý loại bỏ bản trùng ở tập val/test rồi chạy lại cho đến khi sạch hoàn toàn.

### 2.3. Khóa dữ liệu tập Test (Niêm phong SHA-256)
```bash
# Sao chép ảnh test sang data/test_frozen/ và tạo khóa splits/lock.json
python src/data/lock.py

# Xác minh khóa
python src/data/verify_lock.py --lock splits/lock.json --data-dir data/test_frozen
```

### 2.4. Huấn luyện 3 mô hình & Bộ phát hiện phụ trợ Detector
Chạy lần lượt các script huấn luyện (hệ thống sẽ tự động trích xuất HOG, chạy Grid Search ở seed 42, chạy 3 seed để tính độ lệch chuẩn và hiệu chỉnh Temperature Scaling):
```bash
# 1. Huấn luyện M1: HOG + LinearSVM (Bắt buộc)
python src/training/train_m1.py

# 2. Huấn luyện M2: CNN-1 Baseline
python src/training/train_m2.py

# 3. Huấn luyện M3: CNN-2 ResNet-style (Khuyến nghị chính)
python src/training/train_m3.py

# 4. Huấn luyện Detector D (Sliding window SVM nhị phân trên HOG 64x64)
python src/training/train_detector.py

# 5. Tổng hợp báo cáo, chọn model tốt nhất và đóng băng checkpoint (frozen=true)
python reports/build_training_summary.py
```
*Kết quả:* File `best_config.json` và bảng báo cáo `reports/training_summary.md` được sinh ra tự động.

### 2.5. Chạy đánh giá Test chính thức (Official Test Run - Đúng 1 lần)
> **LƯU Ý QUAN TRỌNG**: Xóa file khóa cũ từ đợt test thử trước đó:
> ```powershell
> Remove-Item -Force evidence/OFFICIAL_RUN.lock -ErrorAction SilentlyContinue
> ```

Chạy đánh giá chính thức:
```bash
# Đánh giá tập test chính thức, ghi sổ I/O Ledger và bảng nghiệm thu acceptance.csv
python src/evaluation/evaluate_test.py

# Chạy lại 17 test cases trên mô hình thật
python src/evaluation/run_testcases.py
```
*Kết quả sinh ra trong `evidence/`:*
- Sổ ghi log từng ảnh: `io_ledger_*.csv`.
- Ma trận nhầm lẫn $6 \times 6$ + cột Từ chối: `confusion_*.png` và `.csv`.
- Bảng kiểm định thống kê McNemar: `mcnemar.csv`.
- Bảng kết quả 17 test cases: `test_case_report.csv`.
- Bảng nghiệm thu đạt/không đạt tiêu chí: `acceptance.csv`.

### 2.6. Khởi chạy Web Demo Streamlit
```bash
streamlit run app/app.py
```
Kiểm tra lại lần cuối trên Web: độ chính xác của 6 lớp xe, tính năng đóng khung đánh số 1, 2, 3... cho ảnh nhiều xe, và hiển thị "Không đủ thông tin để nhận diện" cho ảnh xe lạ OOD.

### 2.7. Đẩy mã nguồn & Báo cáo nghiệm thu lên Git
Khi mọi kết quả đã hoàn tất và kiểm thử đạt chuẩn:
```bash
# Kiểm tra các file thay đổi (file .gitignore đã tự lọc bỏ các file ảnh nặng và file trọng số lớn)
git status

# Thêm toàn bộ mã nguồn, cấu hình, báo cáo và bằng chứng nghiệm thu
git add .
git commit -m "feat: complete model training on real dataset and generate official evidence reports"

# Đẩy lên kho Git chung của nhóm (thay thế bằng nhánh của bạn, ví dụ main hoặc dev)
git push origin main
```

---

## 📌 TÓM TẮT CÁC LỆNH CHẠY NHANH CHO MỌI NGƯỜI

```bash
# Cài thư viện:
pip install -r requirements.txt

# Kiểm tra code hoạt động (chạy được ngay cả khi chưa có data):
python -m unittest discover tests
python src/evaluation/run_testcases.py
streamlit run app/app.py

# Khi có data thật:
python src/data/split.py --manifest data/manifest.csv --seed 42
python src/data/check_leak.py
python src/data/lock.py
python src/training/train_m1.py
python src/training/train_m2.py
python src/training/train_m3.py
python src/training/train_detector.py
python reports/build_training_summary.py
python src/evaluation/evaluate_test.py
python src/evaluation/run_testcases.py
streamlit run app/app.py
```
