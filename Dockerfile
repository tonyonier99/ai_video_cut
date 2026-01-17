# Use an official Python runtime as a parent image
# python:3.10-slug is often smaller, but we need build tools for some libs
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Force ImageIO to use system ffmpeg if needed (optional, but good backup)
# ENV IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg

# Set work directory
WORKDIR /app

# Install system dependencies (FFmpeg is CRITICAL here)
# libgl1-mesa-glx and libglib2.0-0 are often needed for OpenCV/MediaPipe
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . .

# Create uploads and exports directories
RUN mkdir -p uploads exports

# Expose port 8000
EXPOSE 8000

# Run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
