# Use Python 3.11 with CUDA support for ML workloads
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy English model
RUN python -m spacy download en_core_web_sm

# Install Jupyter and additional useful packages for ML development
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipywidgets \
    matplotlib \
    seaborn \
    plotly

# Copy project files
COPY . .

# Make download scripts executable
RUN chmod +x /app/download_enron_data.sh && \
    chmod +x /app/download_dataset.py

# Create necessary directories
RUN mkdir -p /app/data /app/outputs /app/configs /app/utils

# Set up Jupyter configuration
RUN jupyter lab --generate-config

# Expose ports for Jupyter Lab and any potential web interfaces
EXPOSE 8888
EXPOSE 8000

# Create a non-root user for security
RUN useradd -m -u 1000 mluser && \
    chown -R mluser:mluser /app

USER mluser

# Download Enron dataset during build (CMU source - no authentication required)
RUN /app/download_enron_data.sh

# Default command to start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"] 