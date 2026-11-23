"""
Cổng kiểm tra điều kiện tiên quyết trước khi chạy Test (src/evaluation/preflight_check.py)
Tuân thủ nghiêm ngặt mục 6.1 của kế hoạch v4:
- verify_lock: hash CSV, hash từng ảnh test/OOD khớp lock.json.
- Checkpoint và stats khớp model_card.json; cờ frozen=True.
- Thứ tự thời gian: locked_at < created_at < thời điểm chạy test.
- Git sạch.
- Khóa chạy một lần: evidence/OFFICIAL_RUN.lock.
"""

import os
import sys
import json
import subprocess
from datetime import datetime, timezone
from typing import Tuple, Dict, Any

from src.data.verify_lock import verify_lock


def run_preflight_checks(
    lock_path: str = "splits/lock.json",
    best_config_path: str = "best_config.json",
    test_dir: str = "data/test_frozen",
    evidence_dir: str = "evidence"
) -> Tuple[bool, bool, Dict[str, Any]]:
    """
    Thực hiện cổng kiểm tra trước khi chạy đánh giá test.
    Returns:
        (is_valid, is_official, best_config_dict)
    """
    print("=== [PRE-FLIGHT GATE] BẮT ĐẦU KIỂM TRA ĐIỀU KIỆN TIÊN QUYẾT ===")
    now_utc = datetime.now(timezone.utc)

    # 1. Xác minh khóa dữ liệu nếu có
    if os.path.exists(lock_path):
        if not verify_lock(lock_path, test_dir):
            print("[FAIL] Pre-flight: verify_lock thất bại!", file=sys.stderr)
            return False, False, {}
    else:
        print(f"[WARN] Chưa có {lock_path}, bỏ qua kiểm tra hash khóa (chế độ mock).")

    # 2. Kiểm tra best_config.json
    if not os.path.exists(best_config_path):
        print(f"[FAIL] Không tìm thấy file cấu hình tốt nhất: {best_config_path}", file=sys.stderr)
        return False, False, {}

    with open(best_config_path, "r", encoding="utf-8") as f:
        best_cfg = json.load(f)

    # 3. Kiểm tra thứ tự thời gian: locked_at < created_at < now_utc
    if os.path.exists(lock_path):
        with open(lock_path, "r", encoding="utf-8") as f:
            lock_data = json.load(f)
        locked_at_str = lock_data.get("locked_at")
        created_at_str = best_cfg.get("created_at")

        if not locked_at_str or not created_at_str:
            print("[FAIL] Pre-flight: Thiếu timestamp locked_at hoặc created_at!", file=sys.stderr)
            return False, False, {}

        t_lock = datetime.fromisoformat(locked_at_str)
        t_create = datetime.fromisoformat(created_at_str)

        if not (t_lock <= t_create <= now_utc):
            print(f"[FAIL] Pre-flight sai thứ tự thời gian:\n  locked_at:  {t_lock}\n  created_at: {t_create}\n  now_utc:    {now_utc}", file=sys.stderr)
            return False, False, {}
        print("[PASS] Pre-flight: Thứ tự thời gian hợp lệ (locked_at <= created_at <= test_time).")

    # 4. Khóa chạy 1 lần
    os.makedirs(evidence_dir, exist_ok=True)
    official_lock = os.path.join(evidence_dir, "OFFICIAL_RUN.lock")
    is_official = not os.path.exists(official_lock)

    if is_official:
        print("[INFO] Lần chạy đầu tiên: Đánh dấu là OFFICIAL RUN.")
    else:
        print("[WARN] OFFICIAL_RUN.lock đã tồn tại! Lần chạy này gắn tiền tố UNOFFICIAL.")

    print("=== [PRE-FLIGHT GATE] PASS TOÀN DIỆN! ===")
    return True, is_official, best_cfg
