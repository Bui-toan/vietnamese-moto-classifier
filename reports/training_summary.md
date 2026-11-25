# Bảng Tổng Hợp Kết Quả Huấn Luyện (Training Summary)

| model | params | size_mb | train_time_s | val_top1 | val_top3 | val_macro_f1 | seeds_f1_mean | seeds_f1_std | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1_HOG_SVM | 36510 | 0.1308 | 8.5653 | 0.4609 | 0.8006 | 0.4408 | 0.4339 | 0.0092 | 0.4022 |
| M2_CNN1 | 375878 | 1.4433 | 10997.3180 | 0.5088 | 0.8389 | 0.5083 | 0.5058 | 0.0098 | 0.0377 |
| M3_CNN2 | 703270 | 2.7241 | 17221.8124 | 0.7161 | 0.9553 | 0.7266 | 0.7065 | 0.0197 | 0.0361 |

*Ghi chú: Điểm số ghi nhận trên tập Validation sau khi hiệu chỉnh Temperature Scaling.*
