# 1. Base Image (Lightweight Python)
FROM python:3.9-slim

# 2. Set Environment Variables
# Prevents Python from writing pyc files to disc (equivalent to python -B option)
ENV PYTHONDONTWRITEBYTECODE 1
# Prevents Python from buffering stdout and stderr (equivalent to python -u option)
ENV PYTHONUNBUFFERED 1

# 3. Set Working Directory
WORKDIR /app

# 4. Install System Dependencies
# libgomp1 is required for XGBoost to run on Linux containers
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 5. Install Python Dependencies
# We copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code
COPY . .

# 7. Expose Ports
# 8000 for FastAPI (The Brain)
# 8501 for Streamlit (The Face)
EXPOSE 8000
EXPOSE 8501

# 8. Command to Run Both Services
# We use a shell command to launch API in background & Dashboard in foreground
CMD uvicorn app.api:app --host 0.0.0.0 --port 8000 & \
    streamlit run app/dashboard.py --server.port 8501 --server.address 0.0.0.0