# Phân tích backbone (seed 0)

- Macro-F1 validation cao nhất: **B03 — deit_small_patch16_224.fb_in1k** (0.9508).
- Cấu hình cân bằng theo quy tắc đã khai báo: **B02 — convnext_tiny.fb_in1k**; đây là model nhanh nhất trong phạm vi 0,02 macro-F1 so với model tốt nhất.
- Thời gian train/epoch thấp nhất: **B05** (38.9 giây).
- Checkpoint tốt nhất xuất hiện sớm nhất: **B03**, epoch 7.
- Tương quan Pearson GMAC–latency p50: -0.981; GMAC–thời gian train/epoch: 0.626. FLOPs chỉ là chỉ báo; kernel, memory access và kiến trúc phần cứng cũng ảnh hưởng tốc độ.
- Model có dấu hiệu overfit theo quy tắc val loss cuối > min val loss 5%: B05, B01.

Kết quả chỉ dùng một seed, nên chênh lệch nhỏ chưa đủ để kết luận chắc chắn. Thứ hạng ImageNet và DeepWeeds không được coi là tương đương; bảng này chỉ xếp hạng theo validation DeepWeeds.
