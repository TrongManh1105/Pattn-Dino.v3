# =========================================================================
# Dockerfile for PAttn-DINOv3 Time Series Forecasting Web Service
# Model: Patch-Attention (NeurIPS 2024) + DINOv3 Vision Transformer Backbone
# =========================================================================

FROM python:3.10-slim

# Set environment variables to optimize Python runtime in container
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    FLASK_ENV=production

# Set working directory inside container
WORKDIR /app

# Install minimal OS dependencies needed for PyTorch, math libraries, and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt /app/

# Install Python packages (using CPU PyTorch for lightweight & fast deployment)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and template assets
COPY pattn_dino_model.py app.py embed.py main.py /app/
COPY templates /app/templates/
COPY models /app/models/
COPY utils /app/utils/
COPY data_provider /app/data_provider/
# Copy primary master dataset (BTCUSDT_1m_5M_rows.csv) and fallback dataset
COPY BTCUSDT_*.csv /app/

# Create a dedicated non-root user for security best practices
RUN useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose web service port
EXPOSE 5000

# Container healthcheck testing Flask API
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/api/info || exit 1

# Launch production server
CMD ["python", "app.py"]
