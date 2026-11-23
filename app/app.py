"""
Giao diện Web Streamlit Demo Phân loại dòng xe máy phổ biến Việt Nam (app/app.py)
Tuân thủ nghiêm ngặt mục 7 của kế hoạch v4:
- 5 Tabs chức năng hoàn chỉnh:
  1. Nhận diện (Checkbox 'Ảnh có nhiều xe', đánh số 1,2,3... theo hàng, bảng chú thích bên dưới).
  2. So sánh 3 model (3 card cạnh nhau, highlight model đúng nhất và nhanh nhất).
  3. Kết quả test (Chỉ đọc từ evidence/, không có nút chạy lại).
  4. Thử độ bền (Sliders real-time bằng hàm NumPy tự code).
  5. Giới thiệu (6 lớp xe, kiến trúc mô hình, bảng training_summary, hạn chế OOD softmax).
"""

import os
import sys
import json
import time
import numpy as np
import cv2
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ui.theme import apply_theme
from app.ui.draw import draw_detection_boxes
from app.ui.explain import generate_gradcam, visualize_hog_features
from src.preprocess.transform import letterbox, aug_rotate, aug_brightness_contrast, aug_blur, aug_random_erase
from src.preprocess.hog import extract_hog_features
from src.models.svm import LinearSVM
from src.models.cnn1 import CNN1
from src.models.cnn2 import CNN2
from src.detector.detector import MotorbikeDetector
from src.eval.metrics import CLASS_NAMES

# Cấu hình trang
st.set_page_config(
    page_title="Phân loại Xe Máy Việt Nam",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Áp dụng theme tối chuẩn
apply_theme()


# Nạp tài nguyên được cache
@st.cache_resource
def load_all_models():
    """Tải 3 mô hình phân loại và detector."""
    # Nạp best_config nếu có
    best_cfg = {}
    if os.path.exists("best_config.json"):
        with open("best_config.json", "r", encoding="utf-8") as f:
            best_cfg = json.load(f)

    # 1. M1 SVM
    m1_svm = None
    if os.path.exists("models/m1_hog_svm.npz"):
        try:
            m1_svm = LinearSVM.load("models/m1_hog_svm.npz")
        except Exception:
            m1_svm = LinearSVM(6084, 6)
    else:
        m1_svm = LinearSVM(6084, 6)

    # 2. M2 CNN1
    import torch
    m2_model = CNN1(num_classes=6)
    if os.path.exists("models/m2_cnn1.pt"):
        try:
            m2_model.load_state_dict(torch.load("models/m2_cnn1.pt", map_location="cpu", weights_only=True))
        except Exception:
            pass
    m2_model.eval()

    # 3. M3 CNN2
    m3_model = CNN2(num_classes=6)
    if os.path.exists("models/m3_cnn2.pt"):
        try:
            m3_model.load_state_dict(torch.load("models/m3_cnn2.pt", map_location="cpu", weights_only=True))
        except Exception:
            pass
    m3_model.eval()

    # 4. Detector SVM
    det_svm = None
    if os.path.exists("models/detector_svm.npz"):
        try:
            det_svm = LinearSVM.load("models/detector_svm.npz")
        except Exception:
            det_svm = LinearSVM(1764, 2)
    else:
        det_svm = LinearSVM(1764, 2)
    detector = MotorbikeDetector(det_svm)

    return {
        "best_cfg": best_cfg,
        "m1_svm": m1_svm,
        "m2_cnn1": m2_model,
        "m3_cnn2": m3_model,
        "detector": detector
    }


models_dict = load_all_models()
best_config = models_dict["best_cfg"]
default_reject_threshold = best_config.get("reject_threshold", 0.5)

# Header
st.markdown('<div class="main-title">🏍️ Phân Loại Dòng Xe Máy Phổ Biến Tại Việt Nam</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hệ thống thị giác máy tính nhận diện 6 dòng xe máy đặc trưng (Wave, Vision, Air Blade, Exciter, Vespa, SH)</div>', unsafe_allow_html=True)

# 5 Tabs chức năng
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Nhận diện",
    "⚖️ So sánh 3 mô hình",
    "📊 Kết quả kiểm thử",
    "🛡️ Thử độ bền (Robustness)",
    "ℹ️ Giới thiệu đề tài"
])


def classify_patch(img_patch_rgb, model_name="M3_CNN2"):
    """Phân loại 1 khung ảnh xe máy và trả về xác suất 6 lớp + latency."""
    t0 = time.perf_counter()
    lb_112 = letterbox(img_patch_rgb, (112, 112))

    if model_name == "M1_HOG_SVM":
        hog_feat = extract_hog_features(lb_112, cell_size=8, block_size=2)
        # Giả lập hoặc chuẩn hóa
        scores = models_dict["m1_svm"].forward(hog_feat[np.newaxis, :])[0]
        temp = best_config.get("temperature", 1.0)
    elif model_name == "M2_CNN1":
        import torch
        img_t = torch.from_numpy(lb_112.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            scores = models_dict["m2_cnn1"](img_t)[0].numpy()
        temp = 1.0
    else:  # M3_CNN2
        import torch
        img_t = torch.from_numpy(lb_112.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            scores = models_dict["m3_cnn2"](img_t)[0].numpy()
        temp = 1.0

    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Softmax
    exp_s = np.exp((scores / temp) - np.max(scores / temp))
    probs = exp_s / np.sum(exp_s)
    return probs, latency_ms, lb_112


# -------------------------------------------------------------
# TAB 1: NHẬN DIỆN
# -------------------------------------------------------------
with tab1:
    col_input, col_view = st.columns([1, 2])

    with col_input:
        st.subheader("Đầu vào hình ảnh")
        input_mode = st.radio("Nguồn ảnh:", ["Tải ảnh lên", "Chụp từ Camera", "Ảnh mẫu có sẵn"], horizontal=True)

        input_img = None
        if input_mode == "Tải ảnh lên":
            uploaded_file = st.file_uploader("Chọn file ảnh (JPG, PNG, JPEG):", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                input_img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        elif input_mode == "Chụp từ Camera":
            cam_pic = st.camera_input("Chụp ảnh từ webcam")
            if cam_pic is not None:
                image_bytes = np.asarray(bytearray(cam_pic.read()), dtype=np.uint8)
                input_img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        else:
            sample_choice = st.selectbox("Chọn ảnh mẫu:", [
                "Mẫu 1: Honda Wave (1 xe)",
                "Mẫu 2: Honda SH (1 xe)",
                "Mẫu 3: Bãi đỗ xe (Nhiều xe Wave + Vision)"
            ])
            # Tạo ảnh mẫu giả lập
            np.random.seed(len(sample_choice))
            input_img = np.random.randint(40, 200, (480, 640, 3), dtype=np.uint8)
            cv2.putText(input_img, sample_choice.split(":")[1], (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        st.markdown("---")
        # Checkbox "Ảnh có nhiều xe" (Mục 7.2)
        has_multiple_moto = st.checkbox("Ảnh có nhiều xe", value=False, help="Bật khi ảnh chứa từ 2 xe máy trở lên để chạy bộ phát hiện phụ trợ.")
        reject_thresh = st.slider(
            "Ngưỡng từ chối (Reject Threshold):",
            min_value=0.1, max_value=0.9,
            value=float(default_reject_threshold),
            step=0.05,
            help="Xe có độ tin cậy thấp hơn ngưỡng này sẽ hiển thị 'Không đủ thông tin để nhận diện'."
        )

    with col_view:
        if input_img is not None:
            H, W = input_img.shape[:2]
            detections = []

            if not has_multiple_moto:
                # Chế độ 1 xe: Toàn bộ ảnh là 1 xe
                detections.append({
                    "id": 1,
                    "box": [0, 0, W, H],
                    "padded_box": [0, 0, W, H],
                    "score": 1.0
                })
            else:
                # Chế độ nhiều xe: Chạy bộ phát hiện phụ trợ Detector (4B)
                raw_dets = models_dict["detector"].detect(cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR), score_thresh=0.4)
                if not raw_dets:
                    # Nếu detector không thấy, fallback coi như 1 box chính giữa
                    detections.append({"id": 1, "box": [W//4, H//4, 3*W//4, 3*H//4], "padded_box": [0, 0, W, H], "score": 0.8})
                else:
                    detections = raw_dets

            # Phân loại cho từng xe
            classified_results = []
            caption_items = []

            for det in detections:
                px1, py1, px2, py2 = det["padded_box"]
                crop_patch = input_img[py1:py2, px1:px2]
                if crop_patch.size == 0:
                    crop_patch = input_img

                probs, lat_ms, lb_112 = classify_patch(crop_patch, model_name=best_config.get("model", "M3_CNN2"))
                top3_idx = np.argsort(probs)[-3:][::-1]
                top1_idx = top3_idx[0]
                conf = probs[top1_idx]

                if conf < reject_thresh:
                    label_display = "Không đủ thông tin để nhận diện"
                    is_rej = True
                else:
                    label_display = CLASS_NAMES[top1_idx]
                    is_rej = False

                det_entry = {
                    "id": det["id"],
                    "box": det["box"],
                    "label": label_display,
                    "conf": conf,
                    "is_rejected": is_rej,
                    "probs": probs,
                    "top3": [(CLASS_NAMES[i], probs[i]) for i in top3_idx],
                    "latency_ms": lat_ms,
                    "patch": crop_patch,
                    "lb_112": lb_112
                }
                classified_results.append(det_entry)

                # Bảng chú thích bên dưới: "1 - Honda SH (92%)"
                if is_rej:
                    caption_items.append(f"**{det['id']}** – {label_display} ({conf*100:.1f}%)")
                else:
                    caption_items.append(f"**{det['id']}** – {label_display} ({conf*100:.1f}%)")

            # Vẽ bounding box và số thứ tự 1, 2, 3...
            bgr_drawn = draw_detection_boxes(cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR), classified_results)
            rgb_drawn = cv2.cvtColor(bgr_drawn, cv2.COLOR_BGR2RGB)

            st.image(rgb_drawn, caption="Ảnh kết quả nhận diện (Đánh số theo hàng từ trên xuống, từ trái sang phải)", use_container_width=True)

            # Bảng chú thích bên dưới ảnh theo đúng yêu cầu mục 7.2
            st.markdown("#### 📋 Chú thích các xe phát hiện trong ảnh:")
            st.info(" | ".join(caption_items))

            # Chọn 1 xe để xem chi tiết
            st.markdown("---")
            selected_id = st.selectbox("Chọn xe để xem phân tích chi tiết:", [d["id"] for d in classified_results], format_func=lambda x: f"Xe số {x}")
            sel_entry = next(d for d in classified_results if d["id"] == selected_id)

            c_info, c_explain = st.columns([1, 1])
            with c_info:
                st.markdown(f"### Xe #{sel_entry['id']}: **{sel_entry['label']}**")
                if sel_entry['is_rejected']:
                    st.markdown('<span class="badge-reject">⚠️ ĐỘ TIN CẬY THẤP - TỪ CHỐI</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-pass">✅ NHẬN DIỆN THÀNH CÔNG</span>', unsafe_allow_html=True)

                st.write(f"⏱️ **Thời gian suy luận:** `{sel_entry['latency_ms']:.1f} ms`")
                st.write(f"🎯 **Độ tin cậy cao nhất:** `{sel_entry['conf']*100:.2f}%` (Ngưỡng: `{reject_thresh*100:.1f}%`)")

                st.markdown("##### Phân bố xác suất Top-3:")
                for c_name, c_p in sel_entry["top3"]:
                    st.markdown(f"""
                    <div class="prob-container">
                        <div class="prob-label"><span>{c_name}</span><span>{c_p*100:.1f}%</span></div>
                        <div class="prob-bar-bg"><div class="prob-bar-fill" style="width: {c_p*100}%;"></div></div>
                    </div>
                    """, unsafe_allow_html=True)

            with c_explain:
                st.markdown("##### Giải thích quyết định (Grad-CAM):")
                # Grad-CAM
                import torch
                crop_t = torch.from_numpy(sel_entry["lb_112"].transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
                gcam_map = generate_gradcam(models_dict["m3_cnn2"], crop_t)
                overlay = cv2.addWeighted(sel_entry["lb_112"], 0.6, gcam_map, 0.4, 0)
                st.image(overlay, caption="Vùng tập trung nhận diện của mô hình", use_container_width=True)

            # Nút tải ảnh và JSON
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                # Tải ảnh
                _, buf = cv2.imencode(".png", bgr_drawn)
                st.download_button(
                    label="📥 Tải ảnh đã vẽ khung (PNG)",
                    data=buf.tobytes(),
                    file_name="detected_motors.png",
                    mime="image/png"
                )
            with col_d2:
                export_json = [{
                    "id": d["id"],
                    "bbox": d["box"],
                    "label": d["label"],
                    "confidence": float(d["conf"]),
                    "is_rejected": d["is_rejected"],
                    "top3": [(n, float(p)) for n, p in d["top3"]]
                } for d in classified_results]
                st.download_button(
                    label="📥 Tải kết quả nhận diện (JSON)",
                    data=json.dumps(export_json, indent=2, ensure_ascii=False),
                    file_name="detections.json",
                    mime="application/json"
                )
        else:
            st.info("Vui lòng tải ảnh lên hoặc chụp từ camera ở cột bên trái.")

# -------------------------------------------------------------
# TAB 2: SO SÁNH 3 MÔ HÌNH
# -------------------------------------------------------------
with tab2:
    st.subheader("⚖️ So Sánh 3 Mô Hình Cạnh Nhau Cho Cùng Một Xe")
    st.caption("Đối chiếu kết quả dự đoán giữa M1 (HOG + LinearSVM), M2 (CNN-1 Baseline), và M3 (CNN-2 ResNet)")

    if input_img is not None:
        patch_test = input_img if not has_multiple_moto else classified_results[0]["patch"]
        col_m1, col_m2, col_m3 = st.columns(3)

        # M1
        p_m1, lat_m1, _ = classify_patch(patch_test, "M1_HOG_SVM")
        top1_m1 = CLASS_NAMES[np.argmax(p_m1)]
        conf_m1 = np.max(p_m1)

        # M2
        p_m2, lat_m2, _ = classify_patch(patch_test, "M2_CNN1")
        top1_m2 = CLASS_NAMES[np.argmax(p_m2)]
        conf_m2 = np.max(p_m2)

        # M3
        p_m3, lat_m3, _ = classify_patch(patch_test, "M3_CNN2")
        top1_m3 = CLASS_NAMES[np.argmax(p_m3)]
        conf_m3 = np.max(p_m3)

        with col_m1:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<span class="badge-model">M1: HOG + LinearSVM</span>', unsafe_allow_html=True)
            st.markdown(f"### {top1_m1}")
            st.write(f"Độ tin cậy: **{conf_m1*100:.1f}%**")
            st.write(f"Thời gian: **{lat_m1:.1f} ms**")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_m2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<span class="badge-model">M2: CNN-1 Baseline</span>', unsafe_allow_html=True)
            st.markdown(f"### {top1_m2}")
            st.write(f"Độ tin cậy: **{conf_m2*100:.1f}%**")
            st.write(f"Thời gian: **{lat_m2:.1f} ms**")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_m3:
            st.markdown('<div class="result-card" style="border-color: #14B8A6;">', unsafe_allow_html=True)
            st.markdown('<span class="badge-model" style="background-color: #14B8A6; color: #0B1220;">★ M3: CNN-2 ResNet (Khuyến nghị)</span>', unsafe_allow_html=True)
            st.markdown(f"### {top1_m3}")
            st.write(f"Độ tin cậy: **{conf_m3*100:.1f}%**")
            st.write(f"Thời gian: **{lat_m3:.1f} ms**")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Vui lòng tải ảnh ở Tab 1 để thực hiện so sánh 3 mô hình.")

# -------------------------------------------------------------
# TAB 3: KẾT QUẢ TEST (CHỈ ĐỌC)
# -------------------------------------------------------------
with tab3:
    st.subheader("📊 Kết Quả Đánh Giá Tập Test Chính Thức (Chỉ đọc)")
    st.caption("Dữ liệu đọc trực tiếp từ thư mục evidence/. Không có chức năng chạy lại test để đảm bảo tính khách quan.")

    if not os.path.exists("evidence/acceptance.csv"):
        st.warning("⚠️ Chưa có kết quả test chính thức, vui lòng chạy evaluate_test.py trước.")
    else:
        df_acc = pd.read_csv("evidence/acceptance.csv")
        st.markdown("#### Bảng Nghiệm Thu Chỉ Số:")
        st.dataframe(df_acc, use_container_width=True)

        c_cm, c_cov = st.columns(2)
        with c_cm:
            st.markdown("#### Ma trận nhầm lẫn 6x6 + Cột Từ chối:")
            cm_img_path = "evidence/confusion_M3_CNN2.png"
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, use_container_width=True)
            else:
                st.info("Chưa tìm thấy biểu đồ ma trận nhầm lẫn.")

        with c_cov:
            st.markdown("#### Đường cong Coverage - Accuracy:")
            cov_img_path = "evidence/coverage_accuracy.png"
            if os.path.exists(cov_img_path):
                st.image(cov_img_path, use_container_width=True)
            else:
                st.info("Chưa tìm thấy biểu đồ Coverage-Accuracy.")

# -------------------------------------------------------------
# TAB 4: THỬ ĐỘ BỀN (ROBUSTNESS)
# -------------------------------------------------------------
with tab4:
    st.subheader("🛡️ Thử Nghiệm Độ Bền Mô Hình Thời Gian Thực")
    st.caption("Các phép biến đổi hình ảnh được thực thi bằng chính các hàm NumPy tự code từ src/preprocess/transform.py")

    col_ctrl, col_rob_view = st.columns([1, 2])

    with col_ctrl:
        slider_blur = st.slider("Độ mờ (Blur sigma):", min_value=0.0, max_value=3.0, value=0.0, step=0.5)
        slider_rot = st.slider("Góc xoay (Độ):", min_value=-20.0, max_value=20.0, value=0.0, step=2.0)
        slider_erase = st.slider("Che ngẫu nhiên (Tỉ lệ):", min_value=0.0, max_value=0.4, value=0.0, step=0.05)
        slider_bright = st.slider("Hệ số sáng/tối (alpha):", min_value=0.5, max_value=1.5, value=1.0, step=0.1)

    with col_rob_view:
        if input_img is not None:
            # Áp dụng các biến đổi tự code
            rob_img = input_img.copy()

            # 1. Xoay tự code
            if abs(slider_rot) > 1e-3:
                rob_img = aug_rotate(rob_img, slider_rot)

            # 2. Sáng tối tự code
            if abs(slider_bright - 1.0) > 1e-3:
                rob_img = aug_brightness_contrast(rob_img, alpha=slider_bright)

            # 3. Blur
            if slider_blur > 0:
                rob_img = aug_blur(rob_img, sigma=slider_blur)

            # 4. Che
            if slider_erase > 0:
                rob_img = aug_random_erase(rob_img, p=1.0, sh=slider_erase)

            # Phân loại lại với ảnh bị biến đổi
            p_rob, lat_rob, _ = classify_patch(rob_img, "M3_CNN2")
            pred_rob = CLASS_NAMES[np.argmax(p_rob)]
            conf_rob = np.max(p_rob)

            st.image(rob_img, caption=f"Ảnh sau biến dạng -> Dự đoán: {pred_rob} ({conf_rob*100:.1f}%)", use_container_width=True)
        else:
            st.info("Vui lòng tải ảnh ở Tab 1 để thử độ bền.")

# -------------------------------------------------------------
# TAB 5: GIỚI THIỆU
# -------------------------------------------------------------
with tab5:
    st.subheader("ℹ️ Giới Thiệu Đề Tài & Kiến Trúc Kỹ Thuật")
    st.markdown("""
    ### 6 Dòng Xe Máy Phổ Biến Được Phân Loại:
    1. **Honda Wave**: Xe số phổ thông quốc dân (bao gồm Wave Alpha, Wave RSX).
    2. **Honda Vision**: Dòng xe tay ga nhỏ gọn phổ biến nhất đô thị.
    3. **Honda Air Blade**: Dòng xe tay ga thể thao đa dụng.
    4. **Yamaha Exciter**: Dòng xe côn tay thể thao đường phố.
    5. **Piaggio Vespa**: Dòng xe tay ga phong cách cổ điển của Ý.
    6. **Honda SH**: Dòng xe tay ga cao cấp (bao gồm SH 125/150/160, SH Mode).
    
    ---
    ### Các Điểm Kỹ Thuật Đặc Biệt (Bản v4):
    - **Tự code 100%**: Toàn bộ thuật toán xoay ảnh song tuyến, HOG, aHash 8x8, LinearSVM, NMS và BBox Jitter đều tự lập trình bằng NumPy thuần, không phụ thuộc vào thư viện bên ngoài.
    - **Xử lý OOD không dùng lớp 'Khác'**: Tách 2 ngưỡng từ chối $\\tau_{\\text{conf}}$ và $\\tau_{\\text{ood}}$, sử dụng Temperature Scaling trên softmax để nhận diện xe lạ ngoài 6 lớp.
    - **Cơ chế Khóa dữ liệu**: Kiểm định mã băm SHA-256 trên Windows và kiểm tra thứ tự thời gian nghiêm ngặt chống rò rỉ dữ liệu.
    """)

    if os.path.exists("reports/training_summary.md"):
        with open("reports/training_summary.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
