FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY api.py .
COPY dashboard.py .
COPY predictions.parquet .
COPY plots/ ./plots/

# Expose ports
EXPOSE 8000 8501

# Start both services
COPY start.sh .
RUN chmod +x start.sh
CMD ["./start.sh"]
