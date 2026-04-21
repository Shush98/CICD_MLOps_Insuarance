FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependency required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY src/ ./src/
COPY label_encoder.pkl /app/label_encoder.pkl
COPY lgbm_model.pkl /app/lgbm_model.pkl

# Expose FastAPI port
EXPOSE 7005

# Run FastAPI app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7005"]