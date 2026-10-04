# Lab Day 2 — Trương Thị Lan Anh

- MSSV: `2A202602451`
- Repo fork: `https://github.com/SxAinsworth/K4-Track4-Day2-Deeplearning-Advance`
- Notebook: `code/lab_day2.ipynb`
- Kế hoạch thực nghiệm: `experiment_plan.md`
- Bảng ngân sách GPU: `gpu_budget.csv`

## Môi trường

Khuyến nghị chạy trên Kaggle hoặc Google Colab với GPU T4 trở lên. Cài dependencies:

```bash
python -m pip install -r submissions/2A202602451_TruongThiLanAnh/code/requirements.txt
```

Trên Windows local, repo hiện dùng môi trường Python 3.11 tại `.venv`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r submissions\2A202602451_TruongThiLanAnh\code\requirements.txt
```

Notebook phải in được phiên bản Python, PyTorch, timm và tên GPU trước khi chạy thí nghiệm.

## Dữ liệu

Notebook tải `images.zip` từ Zenodo, kiểm tra MD5
`b7b30f96d466fba86016aa5a26606e0f`, sau đó tải nhãn và fold 0 từ repo DeepWeeds gốc.

```text
data/
├── images/
└── labels/
    ├── labels.csv
    ├── train_subset0.csv
    ├── val_subset0.csv
    └── test_subset0.csv
```

Không commit dataset hoặc checkpoint vào Git. Sau khi giải nén, cần xác nhận `IMAGES_DIR` trỏ trực tiếp đến thư mục chứa các file `.jpg`.

## Thứ tự chạy

1. Chạy ô cài đặt và xác nhận có GPU.
2. Tải dữ liệu, kiểm tra MD5 và đường dẫn ảnh.
3. Chạy kiểm tra split và EDA.
4. Chạy kiểm tra pipeline trước khi huấn luyện thật.
5. Thực hiện backbone screening, training ablation và inference ablation chỉ trên validation.
6. Chốt cấu hình rồi mới chạy test một lần cho mỗi seed.
7. Chạy `eval.py score` và `eval.py grade` từ notebook.

Seed chung kết dự kiến: `0, 1, 2`. Kết quả và phiên bản thư viện thực tế sẽ được cập nhật sau khi chạy thí nghiệm.
