"""
Script đánh giá tập Test thật (627 ảnh) và trích xuất hình ảnh phục vụ Báo cáo BTL:
1. Đánh giá M1 (HOG+SVM), M2 (CNN-1), M3 (CNN-2 ResNet-14) và M3 Ensemble.
2. Vẽ ma trận nhầm lẫn (Confusion Matrix).
3. Xuất lưới ảnh các trường hợp nhận diện đúng điển hình kèm Grad-CAM.
4. Xuất lưới ảnh các trường hợp dự đoán sai (nhất là Wave vs SH, Air Blade vs Exciter) kèm Grad-CAM để phân tích nguyên nhân.
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.svm import LinearSVM
from src.models.cnn1 import CNN1
from src.models.cnn2 import CNN2
from src.preprocess.hog import extract_hog_features
from src.preprocess.transform import letterbox
from src.eval.metrics import compute_accuracy_topk, compute_per_class_metrics, CLASS_NAMES
from app.ui.explain import generate_gradcam

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


def main():
    print("=== BẮT ĐẦU ĐÁNH GIÁ TRÊN TẬP TEST THỰC TẾ (627 ẢNH) ===")
    os.makedirs("reports/test_eval", exist_ok=True)
    os.makedirs("evidence", exist_ok=True)

    test_csv = "splits/test.csv"
    df_test = pd.read_csv(test_csv)
    print(f"Tổng số ảnh test: {len(df_test)}")

    # 1. Nạp models và stats
    device = torch.device("cpu")
    torch.set_num_threads(4)

    # M1
    m1_svm = LinearSVM.load("models/m1_hog_svm.npz")
    m1_stats = np.load("models/hog_stats.npz")

    # M2
    m2_model = CNN1(6).to(device)
    m2_model.load_state_dict(torch.load("models/m2_cnn1.pt", map_location=device, weights_only=True))
    m2_model.eval()
    m2_stats = np.load("models/cnn1_stats.npz")

    # M3 (Seed 42, 43, 44)
    m3_seed42 = CNN2(6, dropout_p=0.3).to(device)
    m3_seed42.load_state_dict(torch.load("models/m3_cnn2.pt", map_location=device, weights_only=True))
    m3_seed42.eval()

    seed_models = [m3_seed42]
    for s in [43, 44]:
        p = f"models/m3_cnn2_seed{s}.pt"
        if os.path.exists(p):
            sm = CNN2(6, dropout_p=0.3).to(device)
            sm.load_state_dict(torch.load(p, map_location=device, weights_only=True))
            sm.eval()
            seed_models.append(sm)

    m3_stats = np.load("models/cnn2_stats.npz")
    m3_mean = m3_stats["mean"]
    m3_std = m3_stats["std"]

    # 2. Suy luận trên từng ảnh
    y_true = []
    m1_preds, m2_preds, m3_preds, ens_preds = [], [], [], []
    m3_probs_list = []
    all_sample_details = []

    print("Đang chạy suy luận trên 627 ảnh test...")
    t0 = time.time()

    for idx, row in df_test.iterrows():
        img_path = row["image_path"]
        gt_label = row["label"]
        gt_idx = CLASS_TO_IDX[gt_label]
        y_true.append(gt_idx)

        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # M1 (112x112)
        lb_112 = letterbox(rgb, (112, 112))
        hog_feat = extract_hog_features(lb_112, cell_size=8, block_size=2)
        hog_norm = (hog_feat - m1_stats["mean"]) / (m1_stats["std"] + 1e-7)
        s_m1 = m1_svm.forward(hog_norm[np.newaxis, :])[0]
        m1_preds.append(int(np.argmax(s_m1)))

        # M2 (112x112)
        m2_f = lb_112.transpose(2, 0, 1).astype(np.float32) / 255.0
        m2_norm = (m2_f - m2_stats["mean"][:, None, None]) / (m2_stats["std"][:, None, None] + 1e-7)
        t_m2 = torch.from_numpy(m2_norm).unsqueeze(0).float().to(device)
        with torch.no_grad():
            s_m2 = m2_model(t_m2)[0].numpy()
        m2_preds.append(int(np.argmax(s_m2)))

        # M3 (160x160 with TTA + Ensemble)
        lb_160 = letterbox(rgb, (160, 160))
        m3_f = lb_160.transpose(2, 0, 1).astype(np.float32) / 255.0
        m3_norm = (m3_f - m3_mean[:, None, None]) / (m3_std[:, None, None] + 1e-7)
        t_m3 = torch.from_numpy(m3_norm).unsqueeze(0).float().to(device)
        t_m3_flip = torch.flip(t_m3, dims=[3])

        # Single seed 42
        with torch.no_grad():
            l_orig = m3_seed42(t_m3)
            l_flip = m3_seed42(t_m3_flip)
            s_m3 = (0.5 * (l_orig + l_flip))[0].numpy()
        m3_pred_idx = int(np.argmax(s_m3))
        m3_preds.append(m3_pred_idx)

        # Ensemble 3 seeds
        ens_batch_prob = torch.zeros(1, 6, device=device)
        with torch.no_grad():
            for sm in seed_models:
                l_sm = 0.5 * (sm(t_m3) + sm(t_m3_flip))
                ens_batch_prob += torch.softmax(l_sm, dim=1)
            ens_batch_prob /= len(seed_models)
        ens_prob_arr = ens_batch_prob.cpu().numpy()[0]
        ens_pred_idx = int(np.argmax(ens_prob_arr))
        ens_preds.append(ens_pred_idx)
        m3_probs_list.append(ens_prob_arr)

        all_sample_details.append({
            "idx": idx,
            "path": img_path,
            "gt": gt_label,
            "gt_idx": gt_idx,
            "pred_m3": CLASS_NAMES[m3_pred_idx],
            "pred_ens": CLASS_NAMES[ens_pred_idx],
            "ens_prob": float(ens_prob_arr[ens_pred_idx]),
            "is_correct": (ens_pred_idx == gt_idx),
            "lb_160": lb_160,
            "t_m3": t_m3
        })

    elapsed = time.time() - t0
    print(f"Hoàn thành suy luận sau {elapsed:.1f}s (~{elapsed/len(df_test)*1000:.1f} ms/ảnh).")

    y_arr = np.array(y_true)
    ens_probs_arr = np.array(m3_probs_list)

    # 3. Tính toán các chỉ số
    res_m1 = compute_per_class_metrics(y_arr, np.array(m1_preds), 6)
    acc_m1 = float(np.mean(np.array(m1_preds) == y_arr))

    res_m2 = compute_per_class_metrics(y_arr, np.array(m2_preds), 6)
    acc_m2 = float(np.mean(np.array(m2_preds) == y_arr))

    res_m3 = compute_per_class_metrics(y_arr, np.array(m3_preds), 6)
    acc_m3 = float(np.mean(np.array(m3_preds) == y_arr))

    res_ens = compute_per_class_metrics(y_arr, np.array(ens_preds), 6)
    acc_ens = float(np.mean(np.array(ens_preds) == y_arr))
    top3_ens = compute_accuracy_topk(y_arr, ens_probs_arr, (3,))["top3_acc"]

    test_summary = {
        "m1": {"top1": acc_m1, "macro_f1": res_m1["macro_f1"], "per_class": res_m1["per_class"]},
        "m2": {"top1": acc_m2, "macro_f1": res_m2["macro_f1"], "per_class": res_m2["per_class"]},
        "m3_single": {"top1": acc_m3, "macro_f1": res_m3["macro_f1"], "per_class": res_m3["per_class"]},
        "m3_ensemble": {"top1": acc_ens, "top3": top3_ens, "macro_f1": res_ens["macro_f1"], "per_class": res_ens["per_class"]}
    }

    with open("reports/test_eval/test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2, ensure_ascii=False)

    print("\n=== KẾT QUẢ TRÊN TẬP TEST (627 ẢNH) ===")
    print(f"M1 (HOG+SVM)   : Top-1 = {acc_m1*100:.2f}% | Macro-F1 = {res_m1['macro_f1']:.4f}")
    print(f"M2 (CNN-1)     : Top-1 = {acc_m2*100:.2f}% | Macro-F1 = {res_m2['macro_f1']:.4f}")
    print(f"M3 (Single)    : Top-1 = {acc_m3*100:.2f}% | Macro-F1 = {res_m3['macro_f1']:.4f}")
    print(f"M3 (Ensemble)  : Top-1 = {acc_ens*100:.2f}% | Top-3 = {top3_ens*100:.2f}% | Macro-F1 = {res_ens['macro_f1']:.4f}")

    # 4. Vẽ Confusion Matrix cho M3 Ensemble
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_arr, np.array(ens_preds), labels=range(6))
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(6),
        yticks=np.arange(6),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        title='Ma Trận Nhầm Lẫn M3 Ensemble trên Tập Test (Normalized)',
        ylabel='Nhãn Thực Tế (Ground Truth)',
        xlabel='Dự Đoán của Mô Hình'
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    fmt = '.2f'
    thresh = cm_norm.max() / 2.
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=9)
    fig.tight_layout()
    plt.savefig("reports/test_eval/confusion_matrix_test.png", dpi=180)
    plt.savefig("evidence/confusion_M3_CNN2.png", dpi=180)
    plt.close()
    print("[PASS] Đã lưu Confusion Matrix vào reports/test_eval/confusion_matrix_test.png")

    # 5. Xuất hình ảnh minh họa cho báo cáo (Grad-CAM Đúng và Sai)
    correct_samples = [s for s in all_sample_details if s["is_correct"]]
    wrong_samples = [s for s in all_sample_details if not s["is_correct"]]

    # Chọn 6 ảnh đúng đại diện cho 6 lớp
    rep_correct = []
    for c_idx, c_name in enumerate(CLASS_NAMES):
        match = [s for s in correct_samples if s["gt_idx"] == c_idx]
        if match:
            rep_correct.append(match[0])

    # Vẽ lưới 6 ảnh đúng kèm Grad-CAM
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    for i, s in enumerate(rep_correct[:6]):
        # Ảnh gốc letterbox
        axes[0, i].imshow(s["lb_160"])
        axes[0, i].set_title(f"Gốc: {s['gt']}\nConf: {s['ens_prob']*100:.1f}%", fontsize=10)
        axes[0, i].axis("off")

        # Grad-CAM heatmap
        gcam = generate_gradcam(m3_seed42, s["t_m3"], target_class=s["gt_idx"])
        overlay = cv2.addWeighted(s["lb_160"], 0.6, gcam, 0.4, 0)
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"Grad-CAM (Đúng)\n-> {s['pred_ens']}", fontsize=10, color="green")
        axes[1, i].axis("off")
    fig.suptitle("Các Trường Hợp Nhận Diện Đúng Điển Hình (True Positives) & Vùng Tập Trung Grad-CAM", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("reports/test_eval/correct_samples_gradcam.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("[PASS] Đã lưu ảnh đúng kèm Grad-CAM vào reports/test_eval/correct_samples_gradcam.png")

    # Chọn 6 ảnh phân loại sai thú vị (Wave bị đoán SH/Air Blade, Vision vs SH...)
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    for i, s in enumerate(wrong_samples[:6]):
        axes[0, i].imshow(s["lb_160"])
        axes[0, i].set_title(f"Thực: {s['gt']}\nĐoán: {s['pred_ens']}", fontsize=10, color="red")
        axes[0, i].axis("off")

        gcam = generate_gradcam(m3_seed42, s["t_m3"], target_class=s["gt_idx"])
        overlay = cv2.addWeighted(s["lb_160"], 0.6, gcam, 0.4, 0)
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"Grad-CAM (Sai)\nConf: {s['ens_prob']*100:.1f}%", fontsize=10)
        axes[1, i].axis("off")
    fig.suptitle("Phân Tích Các Trường Hợp Nhận Diện Sai (Misclassifications) & Bản Đồ Grad-CAM", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("reports/test_eval/misclassified_samples_gradcam.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("[PASS] Đã lưu ảnh sai kèm Grad-CAM vào reports/test_eval/misclassified_samples_gradcam.png")

    # 6. Cập nhật evidence/acceptance.csv
    acc_rows = [
        {"Tieu_chi": "Top-1 Accuracy", "Chi_tieu": ">= 70%", "Ket_qua_M3": f"{acc_ens*100:.2f}%", "Dat": "PASS" if acc_ens >= 0.70 else "FAIL"},
        {"Tieu_chi": "Macro-F1", "Chi_tieu": ">= 0.68", "Ket_qua_M3": f"{res_ens['macro_f1']:.4f}", "Dat": "PASS" if res_ens["macro_f1"] >= 0.68 else "FAIL"},
        {"Tieu_chi": "Recall Wave", "Chi_tieu": ">= 60%", "Ket_qua_M3": f"{res_ens['per_class']['Honda Wave']['recall']*100:.2f}%", "Dat": "PASS" if res_ens['per_class']['Honda Wave']['recall'] >= 0.60 else "FAIL"},
        {"Tieu_chi": "Recall Air Blade", "Chi_tieu": ">= 60%", "Ket_qua_M3": f"{res_ens['per_class']['Honda Air Blade']['recall']*100:.2f}%", "Dat": "PASS" if res_ens['per_class']['Honda Air Blade']['recall'] >= 0.60 else "FAIL"},
        {"Tieu_chi": "Recall Vespa", "Chi_tieu": ">= 60%", "Ket_qua_M3": f"{res_ens['per_class']['Piaggio Vespa']['recall']*100:.2f}%", "Dat": "PASS" if res_ens['per_class']['Piaggio Vespa']['recall'] >= 0.60 else "FAIL"}
    ]
    pd.DataFrame(acc_rows).to_csv("evidence/acceptance.csv", index=False)
    print("[PASS] Đã cập nhật evidence/acceptance.csv!")


if __name__ == "__main__":
    main()
