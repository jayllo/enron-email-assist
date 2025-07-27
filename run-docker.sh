#!/bin/bash

# Get current user and group IDs
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)

echo "Running Docker container with USER_ID=$USER_ID and GROUP_ID=$GROUP_ID"

# Build the image with correct user IDs
docker build --build-arg USER_ID=$USER_ID --build-arg GROUP_ID=$GROUP_ID -t enron-email-assist .

# Run the container with volume mounts
docker run --rm \
  -p 8888:8888 \
  -p 8000:8000 \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/utils:/app/utils \
  enron-email-assist

echo "Container stopped" 