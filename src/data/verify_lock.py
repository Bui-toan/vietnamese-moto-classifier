"""
Module xác minh khóa dữ liệu (src/data/verify_lock.py)
Tuân thủ nghiêm ngặt mục 3.4 và TC-06:
- Cú pháp: verify_lock.py --lock splits/lock.json --data-dir data/test_frozen
- Khi chạy TC-06 truyền --data-dir data/test_frozen_copy
- So sánh hash CSV, hash từng ảnh test.
- Nếu sai khác dù chỉ 1 byte: In lỗi và thoát với exit code 1 (báo FAIL).
- Nếu khớp toàn bộ: Thoát với exit code 0 (báo PASS).
"""

import os
import sys
import json
import hashlib
import argparse


def compute_sha256(filepath: str) -> str:
    """Tính SHA-256 của file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_lock(lock_json_path: str, data_dir: str) -> bool:
    """
    Xác minh tính toàn vẹn của dữ liệu theo lock.json.
    """
    if not os.path.exists(lock_json_path):
        print(f"[FAIL] Không tìm thấy file lock: {lock_json_path}", file=sys.stderr)
        return False

    with open(lock_json_path, "r", encoding="utf-8") as f:
        lock_data = json.load(f)

    file_hashes = lock_data.get("file_hashes", {})
    if not file_hashes:
        print("[FAIL] File lock không chứa file_hashes!", file=sys.stderr)
        return False

    all_matched = True
    checked_count = 0

    print(f"Bắt đầu xác minh thư mục '{data_dir}' theo '{lock_json_path}'...")
    for filename, expected_hash in file_hashes.items():
        if filename.startswith("ood_"):
            continue  # Bỏ qua tiền tố ood khi duyệt ảnh test frozen

        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"[FAIL] Thiếu file: {filepath}", file=sys.stderr)
            all_matched = False
            continue

        actual_hash = compute_sha256(filepath)
        if actual_hash != expected_hash:
            print(
                f"[FAIL] Sai lệch SHA-256 tại file: {filename}\n"
                f"  Kỳ vọng:  {expected_hash}\n"
                f"  Thực tế:  {actual_hash}",
                file=sys.stderr
            )
            all_matched = False
        else:
            checked_count += 1

    if all_matched and checked_count > 0:
        print(f"[PASS] Xác minh khóa thành công! Khớp tuyệt đối {checked_count}/{checked_count} file.")
        return True
    else:
        print(f"[FAIL] Xác minh khóa thất bại!", file=sys.stderr)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xác minh khóa dữ liệu test")
    parser.add_argument("--lock", type=str, default="splits/lock.json", help="Đường dẫn splits/lock.json")
    parser.add_argument("--data-dir", type=str, default="data/test_frozen", help="Thư mục chứa ảnh test cần kiểm tra")
    args = parser.parse_args()

    success = verify_lock(args.lock, args.data_dir)
    if not success:
        sys.exit(1)
    sys.exit(0)
