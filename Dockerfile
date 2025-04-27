# Use lightweight Python 3.10 image
FROM python:3.10-slim

# Install system dependencies (for audio processing like ffmpeg)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create a user for security
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY --chown=user . .

# Expose port 8000 (this is correct for Railway)
EXPOSE 8000

# Start the app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
