#!/bin/bash

# Download Enron Email Dataset from CMU (Official Source)
# This script downloads the official Enron dataset from Carnegie Mellon University

set -e  # Exit on error

DATA_DIR="/app/data"
DOWNLOAD_DIR="/tmp/enron_download"

echo "🔄 Starting Enron email dataset download from CMU..."

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$DOWNLOAD_DIR"

# Function to download from CMU (official source)
download_from_cmu() {
    echo "Downloading from CMU (official source)..."
    cd "$DOWNLOAD_DIR"
    
    # Download the May 7, 2015 version (latest official release - about 1.7GB)
    echo "This will download approximately 1.7GB of data..."
    wget --progress=bar:force -O enron_mail_20150507.tar.gz "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
    
    if [ $? -eq 0 ] && [ -f "enron_mail_20150507.tar.gz" ]; then
        echo "Downloaded from CMU successfully"
        
        # Verify file size (should be around 1.7GB)
        file_size=$(stat -c%s "enron_mail_20150507.tar.gz" 2>/dev/null || stat -f%z "enron_mail_20150507.tar.gz" 2>/dev/null || echo "0")
        echo "Downloaded file size: $(($file_size / 1024 / 1024)) MB"
        
        if [ "$file_size" -lt 100000000 ]; then  # Less than 100MB indicates likely failure
            echo "Downloaded file seems too small, may be incomplete"
            return 1
        fi
        
        # Extract the dataset
        echo "Extracting dataset..."
        tar -xzf enron_mail_20150507.tar.gz
        
        # Move to data directory
        if [ -d "maildir" ]; then
            mv maildir "$DATA_DIR/enron_maildir"
            echo "Dataset extracted to $DATA_DIR/enron_maildir"
        else
            echo "Warning: Expected 'maildir' folder not found after extraction"
            echo "Contents of download directory:"
            ls -la
            
            # Try to find any directory that might contain the emails
            for dir in */; do
                if [ -d "$dir" ]; then
                    echo "🔍 Found directory: $dir"
                    mv "$dir" "$DATA_DIR/enron_maildir"
                    echo "Moved $dir to $DATA_DIR/enron_maildir"
                    break
                fi
            done
        fi
        
        # Clean up
        rm -f enron_mail_20150507.tar.gz
        return 0
    else
        echo "Failed to download from CMU"
        return 1
    fi
}

# Main download logic - CMU official source
echo "Downloading from CMU official source..."

if download_from_cmu; then
    echo "Successfully downloaded from CMU!"
else
    echo "Download from CMU failed!"
    echo "Manual download option:"
    echo "Download manually from: https://www.cs.cmu.edu/~enron/"
    echo "Place the extracted 'maildir' folder in /app/data/enron_maildir"
    exit 1
fi

# Verify the download
if [ -d "$DATA_DIR/enron_maildir" ]; then
    file_count=$(find "$DATA_DIR/enron_maildir" -type f | wc -l)
    echo "Dataset verification:"
    echo "Location: $DATA_DIR/enron_maildir"
    echo "Files found: $file_count"
    
    if [ "$file_count" -gt 1000 ]; then
        echo "Dataset appears complete!"
    else
        echo "Warning: Dataset may be incomplete (found $file_count files)"
    fi
else
    echo "Warning: Dataset directory not found at expected location"
fi

# Clean up download directory
rm -rf "$DOWNLOAD_DIR"

echo "Dataset download process completed!"