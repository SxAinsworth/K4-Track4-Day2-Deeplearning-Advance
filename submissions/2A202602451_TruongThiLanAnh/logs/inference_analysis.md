# Phân tích suy luận

- ECE trước/sau temperature scaling: 0.0159 → 0.0094, T=0.870875.
- Phương pháp macro-F1 cao nhất trên val: I05 (0.9615), p95=18.89 ms.
- Cấu hình thời gian thực p95≤100 ms: I05.
- TTA/ensemble được đánh giá cùng chi phí tương đối; phương pháp offline chỉ được chọn khi mức tăng chất lượng biện minh cho độ trễ.
- Gộp Conv-BN: {"applicable": false, "bn_before": 0}.
