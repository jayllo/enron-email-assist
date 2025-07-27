#!/usr/bin/env python3
"""
Enron Dataset Downloader

A Python script to download the Enron email dataset from multiple sources.
Can be run from Jupyter notebooks or command line.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_download_script():
    """Run the bash download script."""
    print("Starting Enron dataset download...")
    print("This may take several minutes depending on your internet connection.")
    
    # Path to the download script
    script_path = "/app/download_enron_data.sh"
    
    if not os.path.exists(script_path):
        print("Download script not found!")
        return False
    
    try:
        # Run the bash script
        result = subprocess.run(
            ["/bin/bash", script_path],
            capture_output=False,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            print("\nDataset download completed successfully!")
            return True
        else:
            print(f"\nDownload failed with exit code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"Error running download script: {e}")
        return False


def check_dataset_status():
    """Check if the dataset is already downloaded."""
    data_dir = Path("/app/data")
    enron_dir = data_dir / "enron_maildir"
    
    if enron_dir.exists():
        file_count = len(list(enron_dir.rglob("*")))
        print(f"Dataset found: {enron_dir}")
        print(f"Total files: {file_count}")
        
        if file_count > 1000:
            print("Dataset appears complete!")
            return True
        else:
            print("Dataset may be incomplete")
            return False
    else:
        print("Dataset not found")
        return False


def show_dataset_info():
    """Show information about the Enron dataset."""
    print("\nAbout the Enron Email Dataset:")
    print("Source: Carnegie Mellon University")
    print(" Size: ~1.7GB compressed")
    print("Contains: ~500,000 emails from 150 users")
    print("Format: Individual email files in directory structure")
    print("Official URL: https://www.cs.cmu.edu/~enron/")


def main():
    """Main function."""
    print("Enron Email Dataset Downloader")
    print("=" * 40)
    
    # Show dataset information
    show_dataset_info()
    
    # Check current status
    if check_dataset_status():
        print("\nDataset is already available!")
        response = input("\nDo you want to re-download? (y/N): ").lower().strip()
        if response not in ['y', 'yes']:
            print("Exiting...")
            return
    
    print("\nProceeding with download from CMU...")
    print("⏳ This will take several minutes depending on your internet connection.")
    
    # Run download
    success = run_download_script()
    
    if success:
        print("Dataset location: /app/data/enron_maildir")
        print("The dataset contains email files organized in user directories.")
    else:
        print("\nDownload failed. Please check the logs above.")
        print("You can also download manually from: https://www.cs.cmu.edu/~enron/")
        print("   Extract the 'maildir' folder to /app/data/enron_maildir")


if __name__ == "__main__":
    main() 