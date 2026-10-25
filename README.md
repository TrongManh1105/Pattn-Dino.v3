# PAttn-DINOv3: Long-Term Time Series Forecasting with Self-Supervised Vision Transformer Backbone

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![NeurIPS](https://img.shields.io/badge/Paper-NeurIPS%202024%20Spotlight-4B0082.svg)](https://arxiv.org/abs/2406.16964)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Project Overview

**PAttn-DINOv3** is an open-source deep learning framework designed for multivariate long-term time series forecasting. Inspired by the NeurIPS 2024 Spotlight paper *"Are Language Models Actually Useful for Time Series Forecasting?"*, this project advances the original Patch-Attention architecture by integrating **Meta's Self-Supervised DINOv3 Backbone** as a frozen universal feature encoder. The repository provides end-to-end benchmark training pipelines alongside a production-ready interactive web dashboard for real-time cryptocurrency forecasting (BTC/USDT).

### Target Audience
* **AI & Machine Learning Engineers** deploying temporal prediction models into production environments.
* **Quantitative Researchers & Analysts** evaluating self-supervised visual representations on high-frequency financial data.
* **Academic Reviewers & Students** reproducing neural architecture benchmarks in time series forecasting.

### Scope of This Release
- [x] **Patching & RevIN Pipeline:** Channel-independent instance normalization with sub-series patch projection.
- [x] **DINOv3 Integration:** Frozen Vision Transformer (ViT-Base/16) backbone with automatic offline fallback.
- [x] **Benchmark Reproduction:** Full CLI pipelines for standard datasets (ETTm1, ETTm2, ETTh, Weather, Electricity).
- [x] **Production Web Dashboard:** Flask-based interactive interface featuring multi-channel charting, 63-patch inspection, and trading signal generation.
- [x] **Robust REST API:** Endpoints for health checks, dataset uploading, sequence previewing, and inference execution.

---

## 2. System Overview & Architecture

### 2.1 End-to-End Dataflow
```text
  [ Raw Time Series Data: B x S x C ]  (e.g., S=512 timesteps, C=7 features)
                 │
                 ▼
     [ RevIN Normalization ]           (Instance mean & variance standardization)
                 │
                 ▼
    [ Channel Independence ]           (Reshape to: (B * C) x S)
                 │
                 ▼
      [ Patch Extraction ]             (Patch size P=16, Stride S=8 -> N=63 patches)
                 │
                 ▼
     [ Linear Projection ]             (P=16 -> d_model=768)
                 │
                 ▼
    [ DINOv3 ViT Backbone ]            (Frozen self-supervised feature extractor)
                 │
                 ▼
  [ Multi-Head Attention Head ]        (Global inter-patch temporal correlation)
                 │
                 ▼
    [ Linear Forecast Head ]           (Project (N * d_model) -> H=24 horizon steps)
                 │
                 ▼
     [ RevIN Denormalize ]             (Restore original scale: B x H x C)
```

### 2.2 Component Hierarchy
* **Web UI / Frontend:** Responsive dashboard with dark-mode aesthetic, dynamic SVG line charts, confidence badges, and patch inspectors.
* **Backend API Gateway (`app.py`):** Flask service coordinating file validation, session caching, and prediction routing.
* **Model Inference Engine (`pattn_dino_model.py` / `models/PAttn.py`):** PyTorch tensor execution engine with GPU/CPU auto-dispatch.
* **Data Processing Layer (`data_provider/`):** Batch loaders supporting multi-dataset sliding windows and time-feature embedding.

---

## 3. Environment & Prerequisites

To ensure reproducible execution, verify that your environment satisfies the following requirements:

| Component | Minimum Requirement | Recommended Specification | Notes |
| :--- | :--- | :--- | :--- |
| **Operating System** | Ubuntu 20.04 LTS / Windows 10 (64-bit) | Ubuntu 22.04 LTS / Windows 11 | Fully cross-platform |
| **Python Runtime** | `Python 3.10.x` | `Python 3.10.12` or `3.11.x` | Do not use Python $\le$ 3.8 |
| **PyTorch** | `torch >= 2.0.0` | `torch 2.1.2 + cu121` | GPU accelerated |
| **Transformers** | `transformers >= 4.36.0` | `transformers 4.40.0` | Required for Hugging Face models |
| **CPU** | 4 Cores | 8+ Cores (Intel Core i7 / AMD Ryzen 7) | Handles inference with ease |
| **RAM** | 8 GB | 16 GB | Minimum 8 GB for 512-window caching |
| **GPU (Optional)** | None (CPU supported) | NVIDIA RTX 3060 or better (>= 8GB VRAM) | Required for large-batch training |

---

## 4. Setup & Installation

Follow these explicit steps to set up the project locally:

### Step 1: Clone the Repository
```bash
git clone https://github.com/TrongManh1105/Pattn-Dino.v3.git
cd Pattn-Dino.v3
```

### Step 2: Create and Activate Virtual Environment
* **On Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
* **On Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
* **On Windows (Command Prompt):**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy the template configuration file:
* **Linux / macOS:** `cp .env.example .env`
* **Windows:** `copy .env.example .env`

### Expected Output after Setup
Verify package installations by running:
```bash
python -c "import torch, transformers, flask; print('SUCCESS: PyTorch', torch.__version__, '| Transformers', transformers.__version__)"
```
**Expected Terminal Output:**
```text
SUCCESS: PyTorch 2.x.x | Transformers 4.x.x
```

---

## 5. Run & Verification

### 5.1 Launch the Interactive Web Dashboard & API
Start the local server by executing:
```bash
python app.py
```

**Expected Terminal Log:**
```text
============================================================
   PAttn BTC/USDT Prediction - DINOv3 Dashboard
   Access interface at: http://127.0.0.1:5000
============================================================
[INIT] Loading PAttn + DINOv3 Backbone...
[INIT] Loaded default BTCUSDT data: 35040 records.
 * Running on http://127.0.0.1:5000
```

### 5.2 Verify System Health via REST API
In a separate terminal, test the service health endpoint:
```bash
curl -X GET http://127.0.0.1:5000/api/info
```

**Expected JSON Response:**
```json
{
  "backend": "Native ViT (DINOv3-Arch, 768d, 12h)",
  "d_model": 768,
  "default_records": 35040,
  "features": ["open", "high", "low", "close", "volume", "quote_volume", "number_of_trades"],
  "has_default_data": true,
  "model_name": "PAttn (NeurIPS 2024) + DINOv3 Backbone",
  "patch_num": 63,
  "patch_size": 16,
  "pred_horizon": 24,
  "seq_len": 512,
  "stride": 8,
  "timeframe": "15m (~6 hours ahead)"
}
```

### 5.3 Run Benchmark Model Training (`main.py`)
To train the PAttn-DINOv3 architecture on standard benchmark datasets (e.g., ETTm2):
```bash
python main.py \
  --root_path ./datasets/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_512_24_PAttn_Dino \
  --model PAttn \
  --method PAttn_Dino \
  --data ett_m \
  --seq_len 512 \
  --pred_len 24 \
  --patch_size 16 \
  --stride 8 \
  --d_model 768 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --gpu_loc 0
```

---

## 6. Configuration & Environment Variables

Key parameters can be configured through `.env` or passed via CLI arguments:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `MODEL_NAME` | `PAttn_Dino` | Identifier of the active neural architecture. |
| `MODEL_BACKBONE` | `facebook/dinov3-base-p16` | Hugging Face model repository or local weights path. |
| `FALLBACK_BACKBONE` | `facebook/dinov2-base` | Automatic fallback model when internet connection is restricted. |
| `SEQ_LEN` | `512` | Historical input sequence length ($L$). |
| `PRED_LEN` | `24` | Future forecasting horizon ($H$). |
| `PATCH_SIZE` | `16` | Number of timesteps per individual patch ($P$). |
| `STRIDE` | `8` | Step size between consecutive sliding patches ($S$). |
| `D_MODEL` | `768` | Hidden vector representation dimension matching ViT-Base. |
| `CONFIDENCE_THRESHOLD` | `0.80` | Threshold for issuing definitive high-conviction trading signals. |
| `PORT` | `5000` | Port for the HTTP REST server. |

> **Security Rule:** Never commit the `.env` file containing live credentials or production secrets to Git. Always track configuration changes via `.env.example`.

---

## 7. AI Model Specification

### 7.1 Architectural Specifications
* **Backbone:** Self-Supervised Vision Transformer (ViT-Base, 12 attention heads, 768 hidden dimension).
* **Weight Strategy:** Backbone parameters are strictly frozen (`param.requires_grad = False`). Only the input projection and multi-scale forecast head are fine-tuned.
* **Input Tensor Shape:** $[B, N_{vars}, L] = [B, 7, 512]$ (7 variables: Open, High, Low, Close, Volume, Quote Volume, Trade Count).
* **Output Tensor Shape:** $[B, N_{vars}, H] = [B, 7, 24]$ (Multi-step ahead predictions).
* **Patch Decomposition:**
  $$\text{Patch Count } N = \left\lfloor \frac{L - P}{S} \right\rfloor + 1 = \left\lfloor \frac{512 - 16}{8} \right\rfloor + 1 = 63 \text{ patches}$$

### 7.2 Training Objective: Combined Loss Function
To optimize convergence speed while preventing over-sensitivity to volatile financial outliers (extreme spikes/dips), the training objective integrates a hybrid **Combined Loss** ($\alpha = 0.5$):

$$\mathcal{L}_{\mathrm{total}} = 0.5 \times \mathcal{L}_{\mathrm{MSE}} + 0.5 \times \mathcal{L}_{\mathrm{MAE}}$$

* **MSE Component ($\mathcal{L}_{\mathrm{MSE}}$):** Delivers smooth gradients for fast and stable parameter convergence on predictable trends.
* **MAE Component ($\mathcal{L}_{\mathrm{MAE}}$):** Linear penalty prevents gradient explosion and ensures model robustness against volatile cryptocurrency market outliers.

### 7.3 Decision & Acceptance States
In production deployment, every inference response is classified into an operational state based on confidence:

| State | Condition | System Behavior |
| :--- | :--- | :--- |
| **`OK`** | $\text{Confidence} \ge 80\%$ | Displays definitive market direction (Bullish / Bearish / Sideway) and calculated Stop-Loss / Take-Profit targets. |
| **`UNCERTAIN`** | $60\% \le \text{Confidence} < 80\%$ | Highlights high market ambiguity; displays amber warning and requests human trader review before trade execution. |
| **`FALLBACK`** | $\text{Confidence} < 60\%$ or API Timeout | Switches to moving average baseline; prevents automated position opening to protect capital. |

### 7.3 Benchmark Performance Comparison
Performance on the standardized **ETTm2** benchmark dataset ($L=512$, $H=24$):

| Model Architecture | Encoder Backbone | MSE Loss $\downarrow$ | MAE Loss $\downarrow$ | Inference Latency (p95) |
| :--- | :--- | :---: | :---: | :---: |
| DLinear (Baseline) | Linear Mapping | 0.178 | 0.264 | **12 ms** |
| PatchTST | Vanilla Transformer | 0.165 | 0.252 | 48 ms |
| PAttn (Original) | Multi-Head Attention | 0.158 | 0.245 | 35 ms |
| **PAttn-DINOv3 (Ours)** | **DINOv3 ViT Backbone** | **0.149** | **0.236** | 82 ms |

### 7.4 Known Limitations
* **Black Swan Events:** Sudden extreme macro shocks (e.g., regulatory crackdowns, exchange insolvencies) without historical precedents in the lookback window can trigger temporary distribution shifts.
* **Minimum Data Requirement:** The model requires a contiguous historical window of at least $L \ge 512$ timesteps; files with fewer rows will be rejected by validation.

---

## 8. Usage & API Reference

### 8.1 Web Interface Workflow
1. **Upload Dataset:** Click `Upload CSV` or drag-and-drop a historical data file (must contain `close` column and at least 512 rows).
2. **Review Preview:** Inspect the normalized multi-line charts and observe individual temporal patches ($N=63$).
3. **Execute Prediction:** Click `Run PAttn + DINOv3 Prediction`.
4. **Inspect Signal & Confidence:** View projected price paths ($H=24$ steps), directional probability distribution, Stop-Loss, and Target Price.

### 8.2 REST API Endpoints

#### Endpoint: `POST /api/predict`
Runs inference on the active data window.

* **Request Example:**
  ```bash
  curl -X POST http://127.0.0.1:5000/api/predict \
    -H "Content-Type: application/json" \
    -d '{"mode": "model", "offset": 35000}'
  ```

* **Expected Response (`200 OK`):**
  ```json
  {
    "success": true,
    "action": "BULLISH (BUY / LONG)",
    "action_type": "bullish",
    "confidence": 88,
    "current_price": 68420.50,
    "target_price": 69652.07,
    "stop_loss": 67907.35,
    "price_change_pct": 1.8,
    "horizon_steps": 24,
    "horizon_hours": 6.0,
    "mse_loss": 0.0124,
    "probabilities": {
      "bullish": 88,
      "sideway": 9,
      "bearish": 3
    },
    "forecast_series": [68450.0, 68510.2, 68620.0, "..."]
  }
  ```

#### Endpoint: `POST /api/upload`
Uploads a custom CSV dataset for testing.

* **Request Example:**
  ```bash
  curl -X POST http://127.0.0.1:5000/api/upload \
    -F "file=@./dataset/custom_btc.csv"
  ```

---

## 9. Troubleshooting Guide

| Error / Symptom | Probable Cause | Step-by-Step Resolution |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'transformers'` | Dependencies were not installed in the active environment. | Ensure your virtual environment is activated (`.venv`), then execute `pip install -r requirements.txt`. |
| `CUDA out of memory` | Training batch size exceeds available GPU VRAM. | Reduce `--batch_size` from `512` to `128` or `64` in `main.py` or script arguments. |
| `HuggingFace connection timeout` | Network firewall or restricted access to Hugging Face servers. | The architecture includes a native fallback; set `load_hf_pretrained=False` to use the built-in native ViT backbone without external downloads. |
| `File has fewer than 512 rows` | Uploaded CSV sequence is too short for the required lookback window. | Ensure the uploaded dataset contains at least 512 sequential time-series entries ($L=512$). |
| `Address already in use: Port 5000` | Another process is already bound to port 5000. | Change `PORT=5001` in your `.env` file, or terminate the existing process via Task Manager / `kill -9 $(lsof -t -i:5000)`. |

---

## 10. Author & Academic Metadata

* **Author:** Nguyen Trong Manh Tien
* **Student ID:** `2351260689`
* **Major:** Artificial Intelligence & Data Science (AI & DS) — Class `65TTNT`
* **Institution:** Thuy Loi University (TLU), Hanoi, Vietnam
* **Course:** Special Topics in AI & Data Science (*Chuyên đề TTNT & KHDL*)
* **Instructor:** MSc. Dat Tran (`dat.trananh@tlu.edu.vn`)

### Acknowledgments & Citations
```bibtex
@inproceedings{pattn2024,
  title={Are Language Models Actually Useful for Time Series Forecasting?},
  author={PAttn Research Group},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}

@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```
