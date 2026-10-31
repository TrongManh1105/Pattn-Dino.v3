import os
import sys
import numpy as np
import pandas as pd
import pytest
import torch

# Add parent directory to path so tests can import app and pattn_dino_model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pattn_dino_model  # noqa: E402
import app as app_module  # noqa: E402
from app import app  # noqa: E402


class TestPAttnModel:
    """Unit tests for PAttn-DINOv3 Architecture & Tensor Operations"""

    @pytest.fixture
    def model(self):
        # Native ViT backbone (lightweight, deterministic, offline)
        m = pattn_dino_model.PAttnDino(
            seq_len=512,
            pred_len=24,
            patch_size=16,
            stride=8,
            d_model=768,
            num_heads=12,
            load_hf_pretrained=False
        )
        m.eval()
        return m

    def test_patch_calculation(self, model):
        """Verify patch count formula N = (L - P) / S + 1"""
        expected_patches = int((512 - 16) / 8 + 1)
        assert model.patch_num == 63
        assert expected_patches == 63

    def test_forward_pass_shape(self, model):
        """Verify tensor output shape is exactly [B, pred_len=24, channels=7]"""
        batch_size = 2
        seq_len = 512
        channels = 7
        dummy_input = torch.randn(batch_size, seq_len, channels)

        with torch.no_grad():
            pred, feats, (means, stdev) = model(dummy_input)

        assert pred.shape == (batch_size, 24, channels), f"Expected shape (2, 24, 7), got {pred.shape}"
        assert feats.shape == (batch_size * channels, 63, 768), f"Expected features shape (14, 63, 768), got {feats.shape}"
        assert not torch.isnan(pred).any(), "Prediction tensor contains NaN values!"
        assert not torch.isinf(pred).any(), "Prediction tensor contains Inf values!"

    def test_revin_denormalization(self, model):
        """Verify RevIN scales back correctly to raw price domain"""
        # Simulate realistic BTC price level (~65,000)
        base_price = 65000.0
        dummy_input = base_price + torch.randn(1, 512, 7) * 500.0

        with torch.no_grad():
            pred, _, (means, stdev) = model(dummy_input)

        # Output predictions should remain in the range of ~65,000, not normalized around 0
        mean_pred = pred[0, :, 3].mean().item()  # close price
        assert 50000.0 < mean_pred < 80000.0, f"Predicted price {mean_pred} is out of expected scale"


class TestFlaskAPI:
    """Integration tests for Model Serving REST API"""

    @pytest.fixture
    def client(self):
        app.config['TESTING'] = True
        # In CI environment without large CSV, generate mock BTC window
        if app_module.default_df is None and 'active' not in app_module.uploaded_data:
            dates = pd.date_range('2024-01-01', periods=600, freq='15min')
            mock_df = pd.DataFrame({
                'date': dates.astype(str),
                'open': np.linspace(60000, 65000, 600),
                'high': np.linspace(60500, 65500, 600),
                'low': np.linspace(59500, 64500, 600),
                'close': np.linspace(60200, 65200, 600),
                'volume': np.random.uniform(100, 500, 600),
                'quote_volume': np.random.uniform(6000000, 30000000, 600),
                'number_of_trades': np.random.randint(1000, 5000, 600)
            })
            app_module.uploaded_data['active'] = mock_df

        with app.test_client() as client:
            yield client

    def test_api_info_endpoint(self, client):
        """Test /api/info returns model metadata correctly"""
        response = client.get('/api/info')
        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        assert data['seq_len'] == 512
        assert data['pred_horizon'] == 24
        assert data['patch_num'] == 63
        assert len(data['features']) == 7

    def test_api_predict_mockup_mode(self, client):
        """Test /api/predict in mockup mode returns trading signals & forecast"""
        response = client.post('/api/predict', json={'mode': 'mockup'})
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'action' in data
        assert 'forecast_series' in data
        assert len(data['forecast_series']) == 24
        assert 'confidence' in data

    def test_api_preview_endpoint(self, client):
        """Test /api/preview returns 63 patches metadata"""
        response = client.get('/api/preview?mode=mockup')
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'time_series' in data
        assert 'patches_summary' in data
        assert data['patches_summary']['count'] == 63
