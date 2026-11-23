# Bảng Tổng Hợp Kết Quả Huấn Luyện (Training Summary)

| model | params | size_mb | train_time_s | val_top1 | val_top3 | val_macro_f1 | seeds_f1_mean | seeds_f1_std | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1_HOG_SVM | 36510 | 0.1309 | 1.0968 | 0.1333 | 0.5833 | 0.1235 | 0.1200 | 0.0170 | 0.7340 |
| M2_CNN1 | 94246 | 0.3672 | 88.5157 | 0.1667 | 0.5167 | 0.0476 | 0.0483 | 0.0012 | 0.1524 |
| M3_CNN2 | 308134 | 1.1973 | 133.6539 | 0.1833 | 0.4333 | 0.1721 | 0.1968 | 0.0393 | 0.0039 |

*Ghi chú: Điểm số ghi nhận trên tập Validation sau khi hiệu chỉnh Temperature Scaling.*
