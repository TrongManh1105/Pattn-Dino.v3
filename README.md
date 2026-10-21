# Are Language Models Actually Useful for Time Series Forecasting? (PAttn with DINOv3 & Combined Loss)

[![arXiv](https://img.shields.io/badge/cs.LG-arXiv%3A2406.16964-b31b1b.svg)](https://arxiv.org/abs/2406.16964)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-blue)](https://huggingface.co/)

Dự án nghiên cứu, tái hiện và cải tiến mô hình **PAttn** (*NeurIPS 2024 Spotlight*) cho bài toán dự báo chuỗi thời gian dài hạn (Long-term Time Series Forecasting). Mô hình được nâng cấp toàn diện bằng cách tích hợp **DINOv3 Backbone** (Pretrained Self-Supervised Vision Transformer) làm Encoder trích xuất đặc trưng và áp dụng **Combined Loss Function** nâng cao hiệu năng huấn luyện.

---

## 📖 Giới thiệu Bài toán & Động lực (Introduction)

Trong các nghiên cứu gần đây, việc áp dụng các Mô hình Ngôn ngữ Dài (LLM) như GPT-2 hay LLaMA vào dự báo chuỗi thời gian trở nên rất phổ biến. Tuy nhiên, bài báo **PAttn (NeurIPS 2024)** đã chứng minh rằng: **Các thành phần phức tạp của LLM không nhất thiết phải có để đạt hiệu năng tốt nhất.**

Một mô hình đơn giản sử dụng **Patching + Multi-Head Attention** (PAttn) có thể đạt kết quả vượt trội so với các mô hình LLM-based nặng nề. Trong dự án này, chúng tôi mở rộng ý tưởng trên bằng hai cải tiến cốt lõi:
1. **Thay thế Attention gốc bằng DINOv3 Backbone** để đánh giá khả năng trích xuất đặc trưng tự giám sát (Self-Supervised Learning) trên dữ liệu chuỗi thời gian.
2. **Cập nhật Hàm tổn thất kết hợp (Combined Loss Function)** giúp tối ưu hóa cân bằng giữa sai số trung bình bình phương và sai số tuyệt đối.

---

## 📌 Tổng quan Cập nhật & Cải tiến Code

Repository này giữ nguyên luồng xử lý chính của **PAttn** và tiến hành nâng cấp hai khối quan trọng:

### 1. Tích hợp DINOv3 Backbone (`models/PAttn.py`)
- **Khối Encoder cũ:** Sử dụng Multi-Head Self-Attention / Transformer Encoder tiêu chuẩn (`Encoder_LLaTA`).
- **Cập nhật mới:** Tải trực tiếp Pretrained **DINOv3** (`facebook/dinov3-base-p16`) từ Hugging Face làm bộ trích xuất đặc trưng chính.
- **Freeze Weights:** Đóng băng trọng số DINOv3 (`requires_grad = False`) nhằm tận dụng các biểu diễn ngữ cảnh tự giám sát ổn định.
- **Đồng bộ Kích thước Feature Vector:** Tự động cấu hình `d_model` theo chiều ẩn `hidden_size` của DINOv3 (`768` với bản Base, `384` với bản Small) và cập nhật các lớp `input_projection`, `head` tương thích.

### 2. Cập nhật Combined Loss Function (`main.py`) *(Mới)*
- **Cấu hình cũ:** Sử dụng hàm Loss đơn lẻ (`MSELoss` hoặc `L1Loss`).
- **Cập nhật mới:** Tích hợp lớp `CombinedLoss(nn.Module)` kết hợp giữa **MSE** (Mean Squared Error) và **MAE** (Mean Absolute Error) với hệ số $\alpha = 0.5$:
  $$\mathcal{L}_{\text{total}} = 0.5 \times \mathcal{L}_{\text{MSE}} + 0.5 \times \mathcal{L}_{\text{MAE}}$$
- **Ưu điểm:** Giúp mô hình vừa tối ưu hóa tốc độ hội tụ (nhờ đặc tính mượt của MSE), vừa tăng độ bền vững (robustness) trước các điểm dữ liệu dị biệt (outliers) nhờ MAE.

---

## 🏗️ Luồng xử lý dữ liệu (Architecture Pipeline)

```text
[ Input Data: B x S x N ]
          │
          ▼
 [ Instance Normalization (RevIN) ]
          │
          ▼
 [ Channel Independence: (B * N) x S ]
          │
          ▼
 [ Patching & Linear Projection ] ──► (B * N) x Patch_Num x d_model
          │
          ▼
 [ DINOv3 Pretrained Backbone ] ──► Extracted Features: (B * N) x Patch_Num x d_model
          │
          ▼
 [ Forecast Head & Denormalize ] ──► Output Prediction: B x P x N
          │
          ▼
 [ Combined Loss Optimization ] ──► Loss = 0.5 * MSE + 0.5 * MAE
