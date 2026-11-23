"""
Module ghi sổ đầu vào/đầu ra từng ảnh (src/evaluation/ledger.py)
Tuân thủ nghiêm ngặt mục 6.2 của kế hoạch v4:
- Sổ evidence/io_ledger_<model>.csv ghi nhận chi tiết từng ảnh test.
- Đảm bảo toàn bộ ma trận nhầm lẫn và per-class metrics đều tính lại được từ sổ này.
"""

from typing import List, Dict, Any, Optional
import os
import pandas as pd


class IOLedger:
    """Quản lý việc ghi sổ I/O Ledger cho từng mô hình."""
    def __init__(self, model_name: str, output_csv: str):
        self.model_name = model_name
        self.output_csv = output_csv
        self.rows: List[Dict[str, Any]] = []

    def record(
        self,
        image_path: str,
        sha256: str,
        group_id: str,
        gt_label: str,
        pred_top1: str,
        pred_top2: str,
        pred_top3: str,
        prob_top1: float,
        prob_top2: float,
        prob_top3: float,
        probs_all: List[float],
        is_rejected: bool,
        is_correct_top1: bool,
        is_correct_top2: bool,
        is_correct_top3: bool,
        latency_ms: float,
        bbox: Optional[str] = None
    ):
        self.rows.append({
            "path": image_path,
            "sha256": sha256,
            "group_id": group_id,
            "bbox": bbox or "",
            "gt_label": gt_label,
            "pred_top1": pred_top1,
            "pred_top2": pred_top2,
            "pred_top3": pred_top3,
            "prob_top1": prob_top1,
            "prob_top2": prob_top2,
            "prob_top3": prob_top3,
            "probs_all": str(probs_all),
            "rejected": is_rejected,
            "correct_top1": is_correct_top1,
            "correct_top2": is_correct_top2,
            "correct_top3": is_correct_top3,
            "handled_ok": is_correct_top1 if gt_label.lower() != "ood" else is_rejected,
            "latency_ms": latency_ms
        })

    def save(self):
        """Xuất ra file CSV."""
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        df = pd.DataFrame(self.rows)
        df.to_csv(self.output_csv, index=False)
        return df
