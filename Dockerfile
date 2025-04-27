FROM python:3.10

# Install system packages needed by TensorFlow
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libzmq3-dev \
    pkg-config \
    python3-dev \
    software-properties-common \
    unzip \
    && apt-get clean

# Create user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY --chown=user requirements.txt . 
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY --chown=user . /app

# 🛠️ Correct way: use sh to inject $PORT
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]
