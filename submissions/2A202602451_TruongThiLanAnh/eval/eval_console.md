### F01 (3 seed: [0, 1, 2]; 3507 ảnh)

| Chỉ số | mean ± std |
|---|---|
| top-1 accuracy | 0.9623 ± 0.0012 |
| macro-F1 | 0.9518 ± 0.0017 |
| balanced accuracy | 0.9541 ± 0.0040 |
| ECE (15 bin) | 0.0068 ± 0.0018 |
| NLL | 0.1240 ± 0.0061 |

| Lớp | Precision | Recall | F1 | Số ảnh |
|---|---|---|---|---|
| Chinee apple | 0.966 ± 0.012 | 0.866 ± 0.024 | 0.913 ± 0.010 | 226 |
| Lantana | 0.934 ± 0.017 | 0.977 ± 0.008 | 0.955 ± 0.008 | 213 |
| Parkinsonia | 0.958 ± 0.008 | 0.984 ± 0.007 | 0.971 ± 0.007 | 207 |
| Parthenium | 0.952 ± 0.030 | 0.961 ± 0.005 | 0.956 ± 0.013 | 205 |
| Prickly acacia | 0.915 ± 0.014 | 0.939 ± 0.038 | 0.926 ± 0.013 | 213 |
| Rubber vine | 0.965 ± 0.009 | 0.964 ± 0.011 | 0.964 ± 0.003 | 202 |
| Siam weed | 0.948 ± 0.009 | 0.986 ± 0.008 | 0.967 ± 0.001 | 215 |
| Snake weed | 0.940 ± 0.014 | 0.938 ± 0.006 | 0.939 ± 0.008 | 204 |
| Negative | 0.977 ± 0.005 | 0.973 ± 0.006 | 0.975 ± 0.001 | 1822 |

Đã lưu kết quả vào /content/drive/MyDrive/DeepWeeds_Day2_2A202602451/eval_out/F01_*


---

### T00 (3 seed: [0, 1, 2]; 3507 ảnh)

| Chỉ số | mean ± std |
|---|---|
| top-1 accuracy | 0.9589 ± 0.0014 |
| macro-F1 | 0.9474 ± 0.0015 |
| balanced accuracy | 0.9469 ± 0.0015 |
| ECE (15 bin) | 0.0197 ± 0.0026 |
| NLL | 0.1374 ± 0.0098 |

| Lớp | Precision | Recall | F1 | Số ảnh |
|---|---|---|---|---|
| Chinee apple | 0.959 ± 0.008 | 0.867 ± 0.016 | 0.911 ± 0.012 | 226 |
| Lantana | 0.909 ± 0.022 | 0.978 ± 0.005 | 0.942 ± 0.014 | 213 |
| Parkinsonia | 0.968 ± 0.011 | 0.981 ± 0.005 | 0.974 ± 0.004 | 207 |
| Parthenium | 0.958 ± 0.035 | 0.943 ± 0.015 | 0.950 ± 0.010 | 205 |
| Prickly acacia | 0.904 ± 0.015 | 0.941 ± 0.023 | 0.922 ± 0.004 | 213 |
| Rubber vine | 0.971 ± 0.006 | 0.950 ± 0.005 | 0.961 ± 0.001 | 202 |
| Siam weed | 0.966 ± 0.009 | 0.975 ± 0.010 | 0.971 ± 0.003 | 215 |
| Snake weed | 0.933 ± 0.007 | 0.913 ± 0.010 | 0.923 ± 0.005 | 204 |
| Negative | 0.972 ± 0.001 | 0.974 ± 0.002 | 0.973 ± 0.001 | 1822 |

Đã lưu kết quả vào /content/drive/MyDrive/DeepWeeds_Day2_2A202602451/eval_out/T00_*


---

## Tự chấm RUBRIC mục I (đề xuất; giảng viên xác nhận)

| Mã | Tiêu chí | Điểm | Tối đa | Chi tiết |
|---|---|---|---|---|
| I1 | Top-1 accuracy test | 7 | 7 | 96.23% (mean 3 seed) |
| I2 | Macro-F1 cải thiện so với mốc | 4 | 5 | final 0.9518, mốc 0.9474, Δ=+0.0043, s=0.0017 |
| I3 | Recall hai lớp khó | 3 | 4 | Chinee Apple 86.6% (mốc 88.5%), Snake Weed 93.8% (mốc 88.8%) |
| I4a | ECE sau TS < ECE trước | 1 | 1 | trước 0.0152, sau 0.0068 |
| I4b | Chênh macro-F1 val/test <= 0.02 | 1 | 1 | val 0.9555, test 0.9518, chênh 0.0038 |
| I5 | Cấu hình thời gian thực | 2 | 2 | p95 = 10.3 ms (ngân sách 100 ms), đo đúng cách |

**Tổng các ý đã chấm: 18 / 20** (phần I tối đa 20).

Ngưỡng điểm là TẠM THỜI (xem khối hằng số đầu file eval.py và RUBRIC.md mục I).
