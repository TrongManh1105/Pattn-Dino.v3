import os
import io
import json
import webbrowser
import threading
import pandas as pd
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify

import pattn_dino_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload

# Global state
# Global state & Primary Dataset Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CANDIDATES = [
    os.path.join(BASE_DIR, "BTCUSDT_1m_5M_rows.csv"),
    os.path.join(os.path.dirname(BASE_DIR), "BTCUSDT_1m_5M_rows.csv"),
    os.path.join(BASE_DIR, "BTCUSDT_15m_private.csv"),
    os.path.join(os.path.dirname(BASE_DIR), "BTCUSDT_15m_private.csv")
]

DATA_FILE = next((p for p in DATA_CANDIDATES if os.path.exists(p)), os.path.join(BASE_DIR, "BTCUSDT_1m_5M_rows.csv"))

# Initialize model
print("[INIT] Loading PAttn + DINOv3 Backbone...")
model = pattn_dino_model.PAttnDino(
    seq_len=512,
    pred_len=24,
    patch_size=16,
    stride=8,
    d_model=768,
    num_heads=12,
    load_hf_pretrained=False
)
model.eval()

# Cache default master dataset (BTCUSDT_1m_5M_rows.csv)
default_df = None
if DATA_FILE and os.path.exists(DATA_FILE):
    try:
        default_df, _ = pattn_dino_model.load_dataset(DATA_FILE)
        print(f"[INIT] Loaded primary master data ({os.path.basename(DATA_FILE)}): {len(default_df):,} records.")
    except Exception as e:
        print(f"[WARN] Failed to load default data: {e}")

# In-memory uploaded dataset storage
uploaded_data = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/info", methods=["GET"])
def get_info():
    has_default = default_df is not None
    info = {
        "model_name": "PAttn (NeurIPS 2024) + DINOv3 Backbone",
        "dataset_name": os.path.basename(DATA_FILE) if DATA_FILE and os.path.exists(DATA_FILE) else "BTCUSDT_1m_5M_rows.csv",
        "d_model": 768,
        "seq_len": 512,
        "patch_size": 16,
        "stride": 8,
        "patch_num": 63,
        "pred_horizon": 24,
        "timeframe": "1m (5M Rows Master Dataset)",
        "features": ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'number_of_trades'],
        "has_default_data": has_default,
        "default_records": len(default_df) if has_default else 0,
        "date_start": str(default_df['date'].iloc[0]) if has_default and 'date' in default_df.columns else None,
        "date_end": str(default_df['date'].iloc[-1]) if has_default and 'date' in default_df.columns else None,
        "backend": model.dino_backbone.backend_type
    }
    return jsonify(info)

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "Không tìm thấy file tải lên"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "Chưa chọn file"}), 400

    try:
        filename = file.filename
        content = file.read().decode('utf-8', errors='ignore')
        df = pd.read_csv(io.StringIO(content))
        
        if len(df) < 512:
            return jsonify({
                "success": False, 
                "error": f"File chỉ có {len(df)} dòng. Cần tối thiểu 512 dòng cho chuỗi thời gian (L=512)."
            }), 400

        # Save to session
        uploaded_data['active'] = df
        uploaded_data['filename'] = filename
        
        return jsonify({
            "success": True,
            "filename": filename,
            "rows": len(df),
            "columns": df.columns.tolist(),
            "start_date": df['date'].iloc[0] if 'date' in df.columns else "N/A",
            "end_date": df['date'].iloc[-1] if 'date' in df.columns else "N/A"
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Lỗi đọc file: {str(e)}"}), 500

@app.route("/api/preview", methods=["GET"])
def get_preview():
    """
    Returns time series preview data (L=512) and patch metadata (N=63)
    """
    df = uploaded_data.get('active', default_df)
    if df is None:
        return jsonify({"success": False, "error": "Chưa có dữ liệu"}), 400

    offset = request.args.get('offset', default=None, type=int)
    mode = request.args.get('mode', default='latest', type=str)

    total = len(df)
    if mode == 'mockup':
        # Select an index that demonstrates Bullish trend with high fidelity matching mockup
        # Search for a 512-window with steady upward close
        idx = min(35000, total)
    elif offset is not None:
        idx = min(max(512, offset), total)
    else:
        idx = total

    window_df = df.iloc[idx-512:idx].copy().reset_index(drop=True)
    
    # Feature columns
    feat_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'number_of_trades']
    for c in feat_cols:
        if c not in window_df.columns:
            window_df[c] = window_df['close'] if 'close' in window_df.columns else 0.0

    close_vals = window_df['close'].values
    vol_vals = window_df['volume'].values

    # Sample 128 points for smooth line rendering
    sample_rate = 4
    sampled_indices = list(range(0, 512, sample_rate))
    if sampled_indices[-1] != 511:
        sampled_indices.append(511)

    # 4 normalized curves for multi-line visual preview
    def norm_arr(arr):
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return [0.5] * len(arr)
        return ((arr - mn) / (mx - mn)).tolist()

    sampled_close = close_vals[sampled_indices]
    sampled_vol = vol_vals[sampled_indices]
    sampled_high = window_df['high'].values[sampled_indices]
    sampled_low = window_df['low'].values[sampled_indices]

    # Patches preview (N=63 patches, P=16, S=8)
    patches = []
    for p_id in range(63):
        p_start = p_id * 8
        p_end = p_start + 16
        p_slice = close_vals[p_start:p_end]
        p_norm = norm_arr(p_slice)
        patches.append({
            "id": p_id + 1,
            "start": p_start,
            "end": p_end,
            "points": p_norm
        })

    return jsonify({
        "success": True,
        "total_rows": total,
        "current_window_end": idx,
        "window_start_date": str(window_df['date'].iloc[0]) if 'date' in window_df.columns else "t-512",
        "window_end_date": str(window_df['date'].iloc[-1]) if 'date' in window_df.columns else "t-0",
        "current_close": float(close_vals[-1]),
        "time_series": {
            "indices": sampled_indices,
            "close_raw": sampled_close.tolist(),
            "close_norm": norm_arr(sampled_close),
            "vol_norm": norm_arr(sampled_vol),
            "high_norm": norm_arr(sampled_high),
            "low_norm": norm_arr(sampled_low)
        },
        "patches_summary": {
            "count": 63,
            "patch_size": 16,
            "stride": 8,
            "patches": patches[:15] # first 15 patches for UI card display
        }
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Executes PAttn + DINOv3 model on the current window.
    """
    df = uploaded_data.get('active', default_df)
    if df is None:
        return jsonify({"success": False, "error": "Chưa có dữ liệu BTC/USDT"}), 400

    data = request.get_json(silent=True) or {}
    mode = data.get('mode', 'model')
    window_offset = data.get('offset', None)

    if mode == 'mockup':
        # Exact numbers matching user's mockup slide
        df_len = len(df)
        window_df = df.iloc[-512:].copy().reset_index(drop=True)
        cur_price = float(window_df['close'].iloc[-1])
        target_price = round(cur_price * 1.018, 2)
        
        # Build 24-step forecast array trending upward
        steps = 24
        trend = np.linspace(cur_price, target_price, steps)
        noise = np.random.normal(0, cur_price * 0.001, steps)
        noise[0] = 0
        noise[-1] = 0
        forecast_vals = (trend + noise).round(2).tolist()

        return jsonify({
            "success": True,
            "action": "BULLISH (BUY / LONG)",
            "action_type": "bullish",
            "confidence": 88,
            "probabilities": {
                "bullish": 88,
                "sideway": 9,
                "bearish": 3
            },
            "mse_loss": 0.0124,
            "current_price": cur_price,
            "target_price": target_price,
            "price_change_pct": 1.8,
            "stop_loss": round(cur_price * 0.9925, 2),
            "horizon_steps": 24,
            "horizon_hours": 6.0,
            "forecast_series": forecast_vals,
            "backend": "DINOv3-Base-p16 Backbone (NeurIPS 2024 PAttn)",
            "patch_count": 63,
            "patch_size": 16,
            "patch_stride": 8,
            "is_mockup_preset": True
        })

    try:
        res = pattn_dino_model.run_inference(
            model=model,
            input_df=df,
            window_idx=window_offset,
            seq_len=512,
            pred_len=24
        )
        return jsonify(res)
    except Exception as e:
        return jsonify({"success": False, "error": f"Lỗi chạy dự đoán: {str(e)}"}), 500

def open_browser():
    try:
        webbrowser.open_new("http://127.0.0.1:5000")
    except Exception:
        pass

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    is_prod = os.environ.get("FLASK_ENV") == "production"

    print("\n" + "="*60)
    print("   PAttn BTC/USDT Prediction - DINOv3 Dashboard")
    print(f"   Listening on http://{host}:{port}")
    print("="*60 + "\n")
    
    # Auto open browser only in non-production local mode
    if not is_prod and host in ["127.0.0.1", "0.0.0.0", "localhost"]:
        threading.Timer(1.2, open_browser).start()

    app.run(host=host, port=port, debug=False)
