# Phân tích chung kết

Cấu hình đã được khóa bằng `final_lock.json` trước khi test được mở. Mỗi seed chỉ forward test một lần; temperature scaling tái sử dụng logit đã lưu.

- Cặp nhầm nhiều nhất: **Chinee Apple → Negatives**, 16 lần trong ma trận cộng gộp. Giả thuyết: hình thái lá, nền thực địa và mức che khuất tương tự làm giảm tín hiệu phân biệt.
- Hai lớp khó: .
- Xem `eval_out/eval_console.md` để lấy mean ± std chính thức từ `eval.py`; không dùng điểm test để đổi cấu hình.
