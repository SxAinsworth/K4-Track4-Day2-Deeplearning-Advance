# Phân tích công thức huấn luyện

Backbone: `deit_small_patch16_224.fb_in1k`. T00 qua ba seed có macro-F1 std = 0.0017.

- Trục A: tốt nhất `T01` (frozen backbone), Δ=-0.1883; phân biệt được.
- Trục B: tốt nhất `T06` (CutMix alpha=1), Δ=+0.0061; phân biệt được.
- Trục C: tốt nhất `T08` (focal gamma=2), Δ=+0.0050; phân biệt được.
- Trục F: tốt nhất `T10` (EMA decay=0.999), Δ=-0.0015; không phân biệt được.
- Kết hợp T11: Δ=+0.0023; hiệu ứng cộng dồn một phần / có tương tác.
- Công thức chuyển sang Bước 3: `T06`, macro-F1 val=0.9570.

Các ablation dùng seed 0; std của ba lượt T00 là ngưỡng nhiễu tham chiếu, không phải std riêng của từng kỹ thuật.