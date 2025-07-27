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

# Create a non-root user for security
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN if ! getent group ${GROUP_ID} >/dev/null; then groupadd -g ${GROUP_ID} mlgroup; fi && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} mluser

# Copy project files and set ownership
COPY . .
RUN chown -R ${USER_ID}:${GROUP_ID} /app

# Create necessary directories with proper ownership
RUN mkdir -p /app/data /app/outputs /app/configs /app/utils && \
    chown -R ${USER_ID}:${GROUP_ID} /app

# Switch to non-root user
USER mluser

# Set up Jupyter configuration
RUN jupyter lab --generate-config

# Expose ports for Jupyter Lab and any potential web interfaces
EXPOSE 8888
EXPOSE 8000

# Default command to start Jupyter Lab with modern configuration
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--IdentityProvider.token=''", "--ServerApp.password=''", "--ServerApp.allow_origin='*'"] 