# Kế hoạch thực nghiệm và ngân sách GPU

## Nguyên tắc

- Fold 0 cố định; không chia lại dữ liệu.
- Mọi lựa chọn được thực hiện trên validation.
- Test chỉ chạy ở Bước 4, đúng một lần cho mỗi seed và cấu hình đã chốt.
- Một thí nghiệm chỉ thay đổi một yếu tố so với mốc tương ứng.
- Lưu `config.json`, `history.csv`, checkpoint tốt nhất, val logits/predictions và biểu đồ sau mỗi lần chạy.
- GPU local hiện có: NVIDIA GeForce GTX 1650 Max-Q 4 GB. Dùng máy này cho smoke test; ưu tiên Kaggle/Colab T4 cho các lượt huấn luyện chính.

## Công thức nền dự kiến

| Thuộc tính | Giá trị |
|---|---|
| Fold | 0 |
| Seed sàng lọc/ablation | 0 |
| Epoch | 10 (chỉ tăng nếu ngân sách sau phép đo cho phép) |
| Kích thước ảnh | 224 |
| Optimizer | AdamW |
| LR backbone/head | `1e-4` / `1e-3` |
| Weight decay | `0.05`, không áp dụng cho norm/bias |
| Scheduler | warmup 1 epoch + cosine |
| Loss | cross-entropy |
| Augmentation | RandomResizedCrop + horizontal flip |
| AMP | bật trên CUDA |
| Checkpoint | macro-F1 validation cao nhất; hòa lấy epoch sớm hơn |

Batch vật lý được chọn sau smoke test. Nếu phải giảm batch vì VRAM, dùng gradient accumulation để giữ cùng effective batch size giữa các backbone và ghi rõ trong báo cáo.

## Bước 0 — Pipeline

- Kiểm tra split: `10501 / 3501 / 3507`, ba giao rỗng, hợp đủ `17509`, không thiếu ảnh.
- EDA: phân bố lớp, tỷ lệ mất cân bằng, ít nhất ba ảnh mỗi lớp, kích thước và số kênh.
- Kiểm tra focal loss `gamma=0` bằng CE.
- Kiểm tra CutMix thay đổi cả ảnh và nhãn, lambda theo diện tích thật.
- Kiểm tra CE ban đầu gần `ln(9) = 2.197`.
- Overfit một batch nhỏ và kiểm tra ảnh sau augmentation.
- Chỉ chuyển sang Bước 1 khi `run(Config(...))` chạy hết và lưu đủ artifact validation.

## Bước 1 — Backbone screening

Các cấu hình dùng cùng seed và công thức nền:

| exp_id | Backbone | Vai trò |
|---|---|---|
| B01 | `resnet50` | CNN mốc |
| B02 | `convnext_tiny` | CNN hiện đại |
| B03 | `efficientnet_b0` | mạng nhẹ |
| B04 | `mobilenetv3_large_100` | mạng nhẹ thứ hai |
| B05 | `deit_tiny_patch16_224` | transformer, phù hợp ngân sách hơn DeiT-S |

Với mỗi backbone, trước tiên chạy đúng một epoch và ghi thời gian vào `gpu_budget.csv`. Sau đó mới quyết định chạy đủ 10 epoch. Chọn một backbone đi tiếp dựa trên macro-F1 validation, params, GMAC và latency; không chỉ dựa trên accuracy.

## Bước 2 — Công thức huấn luyện

Chạy trên một backbone được chọn ở Bước 1, seed 0:

| exp_id | Trục | Khác T00 |
|---|---|---|
| T00 | mốc | công thức nền |
| T01 | khởi tạo | frozen backbone |
| T02 | khởi tạo | scratch |
| T03 | augmentation | ColorJitter |
| T04 | augmentation | RandAugment |
| T05 | loss | label smoothing `0.1` |
| T06 | loss | focal loss `gamma=2` |
| T07 | loss | weighted CE, trọng số chỉ từ train |
| T08 | kết hợp | kết hợp các yếu tố thắng, sau khi ablation đơn hoàn tất |

Như vậy có ba trục, mỗi trục gồm mốc và ít nhất hai giá trị thay thế. T08 dùng để kiểm tra các cải thiện có cộng dồn hay không.

## Bước 3 — Suy luận

Không huấn luyện lại, chỉ dùng checkpoint đã chọn trên validation:

| exp_id | Phương pháp |
|---|---|
| I00 | 1-view, FP32 — mốc |
| I01 | TTA identity + horizontal flip |
| I02 | multi-crop hoặc multi-scale |
| I03 | so sánh gộp probability và logit |
| I04 | test resolution 224/256/288 |
| I07 | temperature scaling, fit T trên validation |
| I08 | FP16/AMP và gộp BatchNorm nếu backbone hỗ trợ |

Đo p50/p95/p99 ở batch 1, tối thiểu 10 warmup và 100 lượt đo, đồng bộ CUDA trước/sau. Ghi GPU, dtype, độ phân giải và số view.

## Bước 4 — Chung kết

1. Khóa cấu hình tốt nhất hoàn toàn từ validation.
2. Chạy `F01` với seed `0, 1, 2`.
3. Chạy mốc `T00 + I00` với cùng seed `0, 1, 2`; có thể tái sử dụng seed 0 nếu artifact đầy đủ.
4. Chạy test đúng một lần cho mỗi seed.
5. Lưu prediction chuẩn bằng `eval.save_predictions` và chạy `eval.py score/grade`.

## Tổng số lượt train dự kiến

| Nhóm | Lượt |
|---|---:|
| Backbone screening | 5 |
| Training baseline + ablation + combination | 9 |
| Chung kết F01, 3 seed | 3 |
| Bổ sung baseline T00 seed 1 và 2 | 2 |
| **Tổng chính thức** | **19** |

Thêm khoảng 10–20% dự phòng cho lượt lỗi hoặc bị ngắt. Các phương pháp suy luận không tính là lượt train.

## Công thức ngân sách

Sau khi đo thời gian epoch đầu tiên:

```text
thời gian_run_giờ = giây_mỗi_epoch × số_epoch / 3600
tổng_giờ = tổng(thời_gian_run_giờ × số_lượt_cấu_hình)
ngân_sách_có_dự_phòng = tổng_giờ × 1.2
```

Nếu vượt hạn mức: giữ 10 epoch, chỉ ablation trên một backbone, chỉ dùng một seed ở Bước 1–3, giữ ba seed cho Bước 4, rồi mới cân nhắc giảm độ phân giải.
