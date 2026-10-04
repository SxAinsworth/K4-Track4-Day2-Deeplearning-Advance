# Bước 0 — EDA và kiểm tra pipeline

## Kiểm tra dữ liệu

Đã chạy trên fold 0 chính chủ, không sửa hoặc chia lại CSV.

| Split | Số ảnh | Tỷ lệ |
|---|---:|---:|
| Train | 10.501 | 59,975% |
| Validation | 3.501 | 19,995% |
| Test | 3.507 | 20,030% |
| Tổng | 17.509 | 100% |

- `train ∩ val = 0`, `train ∩ test = 0`, `val ∩ test = 0`.
- Hợp ba tập có đúng 17.509 tên file duy nhất.
- Không thiếu ảnh được tham chiếu trong CSV.
- Ảnh mẫu đều là RGB, kích thước `256 × 256`, ba kênh.

## Phân bố lớp

| Label | Lớp | Toàn bộ |
|---:|---|---:|
| 0 | Chinee Apple | 1.126 |
| 1 | Lantana | 1.063 |
| 2 | Parkinsonia | 1.031 |
| 3 | Parthenium | 1.022 |
| 4 | Prickly Acacia | 1.062 |
| 5 | Rubber Vine | 1.009 |
| 6 | Siam Weed | 1.074 |
| 7 | Snake Weed | 1.016 |
| 8 | Negatives | 9.106 |

Tỷ lệ lớp lớn nhất/nhỏ nhất là `9106 / 1009 = 9,025`. Negative chiếm khoảng 52%, vì vậy accuracy có thể bị kéo cao; macro-F1 được dùng làm chỉ số lựa chọn chính.

CSV hiện tại lệch một ảnh ở Chinee Apple/Lantana so với Table 1 được trích trong README (1.126/1.063 thay vì 1.125/1.064), nhưng tổng vẫn đúng 17.509 và đây là file fold chính chủ không bị chỉnh sửa. Sự khác biệt này được ghi nhận thay vì sửa CSV.

## Nhận xét ảnh mẫu

- Chinee Apple và Snake Weed đều có thể xuất hiện dưới dạng các cụm lá bầu dục dày, màu sắc thay đổi mạnh theo ánh sáng; đây là cặp dễ nhầm bằng mắt và cũng là cặp khó trong bài báo.
- Một số loài xuất hiện rất nhỏ giữa đất, cỏ khô hoặc cây khác. Random crop quá mạnh có nguy cơ cắt mất đối tượng chính.
- Negative rất không đồng nhất: cỏ nền, đất trống, cành khô, bóng cây và các thực vật không thuộc tám loài mục tiêu.
- Dùng chuẩn hóa ImageNet là phù hợp với backbone pretrained; biểu đồ augmentation đã được giải chuẩn hóa để kiểm tra màu và nhãn.

Artifact:

- `eda/class_distribution.png`
- `eda/samples_3_per_class.png`
- `eda/augmented_samples.png`
- `eda/eda_report.json`

## Kiểm tra toán học và pipeline

| Kiểm tra | Kết quả |
|---|---:|
| CE của logit đều cho 9 lớp (kiểm tra công thức) | 2,1972244 |
| `ln(9)` | 2,1972246 |
| Sai số focal `gamma=0` so với CE | 2,38e-7 |
| CutMix thay đổi pixel | Đạt |
| CutMix trộn nhãn | Đạt |
| Loss đầu khi overfit batch nhỏ | 2,07367 |
| Loss cuối sau 100 bước | 2,97e-5 |

### Loss ban đầu của model thật

Kiểm tra `initial_loss_check`: backbone pretrained + head 9 lớp mới, `model.eval()`, 64 ảnh val (transform val), trước bất kỳ bước cập nhật nào. Kỳ vọng ≈ `ln 9 = 2,197`.

| Backbone | Tag timm | Loss ban đầu (head mặc định của timm) | Loss ban đầu (sau `init_head`) |
|---|---|---:|---:|
| `resnet50` | `a1_in1k` | 2,192 | 2,255 |
| `convnext_tiny` | `in12k_ft_in1k` | 2,608 | 2,182 |
| `efficientnet_b0` | `ra_in1k` | **5,923** | 2,254 |
| `mobilenetv3_large_100` | `ra_in1k` | **4,713** | 2,244 |
| `deit_tiny_patch16_224` | `fb_in1k` | 2,270 | 2,276 |

Lỗi phát hiện: với EfficientNet-B0 và MobileNetV3, khởi tạo mặc định của `nn.Linear` trên đặc trưng 1280 chiều cho logit có độ lệch chuẩn ≈ 2,8, nên loss ban đầu 4,7–5,9 thay vì 2,197. Đã sửa bằng `model.init_head`: mọi head mới được khởi tạo giống nhau (`trunc_normal`, std 0,01, bias 0) cho **mọi** backbone, để công thức nền vẫn giống nhau. Sau khi sửa, loss ban đầu nằm trong 2,18–2,28.

### Chế độ train/eval

`mode_checks` gắn hook vào BN đầu tiên và head của ResNet-18:

| Trường hợp | BN trong `train_one_epoch` | Head trong `train_one_epoch` | BN/head trong `evaluate` |
|---|---|---|---|
| Tinh chỉnh toàn bộ | train | train | eval / eval |
| Đóng băng backbone | **eval** | train | eval / eval |

### Unit test

- `code/test_step0.py`: 11 test đạt: focal `γ=0` ≡ CE, CutMix trộn ảnh và nhãn, `lam` khớp đúng diện tích dán thật sau khi cắt ở biên, `mixed_loss`, `class_weights` chỉ từ số đếm train, `param_groups` đủ 3 nhóm (norm/bias không weight decay, head LR ×10), backbone đóng băng chỉ còn head, `evaluate` giữ thứ tự file và chuyển sang eval, warmup + cosine, `parse_overrides`, `set_seed`.
- `python -m unittest discover -s tests` (từ thư mục gốc): 38 test đạt. Trên Windows cần đặt `PYTHONUTF8=1`, vì `eval.py` ghi JSON tiếng Việt bằng `write_text()` không chỉ định encoding (mặc định cp1252 sẽ gây `UnicodeEncodeError` và mã thoát 2). Không sửa `eval.py`.

### Smoke run

Smoke run dùng ResNet-18 scratch, 64 ảnh train, 64 ảnh validation, một epoch, ảnh 128px. Lượt này chỉ kiểm tra pipeline và không được dùng làm kết quả thí nghiệm:

- Hoàn thành `run(Config(...))`.
- `best_epoch = 1`.
- Đã tạo `history.csv`, checkpoint, curve và prediction validation.
- `model.eval()` và `torch.inference_mode()` được gọi trong `evaluate()`.
- Test loader không được tạo khi `save_test_predictions=False`.

## Môi trường kiểm tra

- Python 3.11 trong `.venv`, PyTorch 2.7.1+cu118, timm 1.0.30.
- GPU local: NVIDIA GeForce GTX 1650 Max-Q 4 GB (CUDA dùng được).
- Lượt đo thử `efficientnet_b0` (batch 8 × tích luỹ 8 = 64, AMP): 232 s/epoch trên GTX 1650. Chỉ dùng để ước lượng ngân sách, không phải kết quả thí nghiệm.
