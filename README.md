# Are Language Models Actually Useful for Time Series Forecasting? (PAttn with DINOv3)

[![arXiv](https://img.shields.ch/badge/cs.LG-arXiv%3A2406.16964-b31b1b.svg)](https://arxiv.org/abs/2406.16964)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-blue)](https://huggingface.co/)

Dự án nghiên cứu, tái hiện và cải tiến mô hình **PAttn** (*NeurIPS 2024 Spotlight*) cho bài toán dự báo chuỗi thời gian dài hạn (Long-term Time Series Forecasting). Mô hình đã được nâng cấp bằng cách tích hợp **DINOv3 Backbone** (Pretrained Self-Supervised Vision Transformer) làm bộ trích xuất đặc trưng (Encoder).

---

## 📌 Tổng quan Cập nhật & Cải tiến Code

Repository này giữ nguyên luồng xử lý chính của **PAttn** và tiến hành nâng cấp khối Encoder:

### 1. Tích hợp DINOv3 Backbone (`models/PAttn.py`)
* **Khối Encoder cũ:** Sử dụng Multi-Head Self-Attention / Transformer Encoder tiêu chuẩn.
* **Cập nhật mới:** Tải trực tiếp Pretrained **DINOv3** (`facebook/dinov3-base-p16`) từ Hugging Face để làm bộ trích xuất đặc trưng chính.
* **Freeze Weights:** Đóng băng trọng số DINOv3 (`requires_grad = False`) nhằm tận dụng các biểu diễn ngữ cảnh tự giám sát ổn định.

### 2. Tự động đồng bộ Kích thước Feature Vector
* Tự động cấu hình `d_model` theo chiều ẩn `hidden_size` của DINOv3 (`768` với bản Base, `384` với bản Small).
* Cập nhật các lớp `input_projection` và `head` để đảm bảo tương thích hoàn toàn về mặt số chiều dữ liệu.

---

## 🏗️ Luồng xử lý dữ liệu (Architecture Pipeline)
