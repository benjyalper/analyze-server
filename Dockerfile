# Use Python 3.10
FROM python:3.10-slim

# Create user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt tensorflow

# Copy app
COPY --chown=user . /app

# Expose the app on port 7860 (HuggingFace / Railway expects it)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
