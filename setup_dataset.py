r"""
Dataset Setup Utility for ASL Alphabet Recognition

This script helps download and organize the ASL Alphabet Dataset from Kaggle.

Installation Requirements:
pip install kaggle

Setup Instructions:
1. Create a Kaggle account at https://www.kaggle.com
2. Go to Account settings and create an API token (download kaggle.json)
3. Place kaggle.json in your user directory:
   - Windows: C:\Users\{username}\.kaggle\kaggle.json
   - Mac/Linux: ~/.kaggle/kaggle.json
4. Run this script to download and organize the dataset
"""

import os
import zipfile
import shutil
from pathlib import Path

def setup_kaggle_api():
    """
    Check if Kaggle API is properly configured
    """
    try:
        import kaggle
        print("Kaggle API is available.")
        return True
    except ImportError:
        print("Kaggle library not found. Please install it:")
        print("pip install kaggle")
        return False
    except OSError as e:
        print(f"Kaggle API configuration error: {e}")
        print("\nPlease ensure you have:")
        print("1. Created a Kaggle account")
        print("2. Downloaded your API token (kaggle.json)")
        print("3. Placed it in the correct location:")
        print("   - Windows: C:\\Users\\{username}\\.kaggle\\kaggle.json")
        print("   - Mac/Linux: ~/.kaggle/kaggle.json")
        return False

def download_asl_dataset():
    """
    Download the ASL Alphabet Dataset from Kaggle
    """
    try:
        import kaggle
        
        print("Downloading ASL Alphabet Dataset from Kaggle...")
        
        # Download the dataset
        kaggle.api.dataset_download_files(
            'grassknoted/asl-alphabet',
            path='.',
            unzip=True
        )
        
        print("Dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def organize_dataset():
    """
    Organize the downloaded dataset into the expected structure
    """
    print("Organizing dataset structure...")
    
    # Expected download folder name (may vary)
    possible_folders = ['asl-alphabet', 'asl_alphabet_train', 'ASL_Alphabet_Dataset']
    
    source_folder = None
    for folder in possible_folders:
        if os.path.exists(folder):
            source_folder = folder
            break
    
    if source_folder is None:
        print("Could not find downloaded dataset folder.")
        print("Please check if the download was successful.")
        return False
    
    # Create target folder
    target_folder = 'asl_alphabet_train'
    
    if os.path.exists(target_folder):
        print(f"Target folder '{target_folder}' already exists.")
        response = input("Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            return False
        shutil.rmtree(target_folder)
    
    # Check if source folder contains alphabet folders directly
    alphabet_folders = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    
    # Look for the actual training folder
    train_folder = None
    if os.path.exists(os.path.join(source_folder, 'asl_alphabet_train')):
        train_folder = os.path.join(source_folder, 'asl_alphabet_train')
    elif all(os.path.exists(os.path.join(source_folder, letter)) for letter in alphabet_folders[:3]):
        train_folder = source_folder
    else:
        # Look for subfolder containing alphabet folders
        for subfolder in os.listdir(source_folder):
            subfolder_path = os.path.join(source_folder, subfolder)
            if os.path.isdir(subfolder_path):
                if all(os.path.exists(os.path.join(subfolder_path, letter)) for letter in alphabet_folders[:3]):
                    train_folder = subfolder_path
                    break
    
    if train_folder is None:
        print("Could not find alphabet folders (A, B, C, ..., Z) in the dataset.")
        return False
    
    # Copy or move the training folder
    if train_folder != target_folder:
        print(f"Copying dataset from '{train_folder}' to '{target_folder}'...")
        shutil.copytree(train_folder, target_folder)
    
    # Verify the structure
    missing_letters = []
    for letter in alphabet_folders:
        letter_path = os.path.join(target_folder, letter)
        if not os.path.exists(letter_path):
            missing_letters.append(letter)
        else:
            # Count images in each folder
            image_count = len([f for f in os.listdir(letter_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"Class '{letter}': {image_count} images")
    
    if missing_letters:
        print(f"Warning: Missing alphabet folders: {missing_letters}")
        return False
    
    print("Dataset organized successfully!")
    return True

def cleanup_downloads():
    """
    Clean up downloaded zip files and temporary folders
    """
    print("Cleaning up temporary files...")
    
    # Remove zip files
    zip_files = [f for f in os.listdir('.') if f.endswith('.zip') and 'asl' in f.lower()]
    for zip_file in zip_files:
        os.remove(zip_file)
        print(f"Removed {zip_file}")
    
    # Remove temporary folders (but keep our organized folder)
    temp_folders = ['asl-alphabet']
    for folder in temp_folders:
        if os.path.exists(folder) and folder != 'asl_alphabet_train':
            shutil.rmtree(folder)
            print(f"Removed temporary folder {folder}")

def main():
    """
    Main function to setup the ASL dataset
    """
    print("ASL Alphabet Dataset Setup")
    print("=" * 30)
    
    # Check if dataset already exists
    if os.path.exists('asl_alphabet_train'):
        print("Dataset folder 'asl_alphabet_train' already exists.")
        
        # Count total images
        total_images = 0
        for letter in [chr(i) for i in range(ord('A'), ord('Z') + 1)]:
            letter_path = os.path.join('asl_alphabet_train', letter)
            if os.path.exists(letter_path):
                image_count = len([f for f in os.listdir(letter_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                total_images += image_count
        
        print(f"Found {total_images} images in existing dataset.")
        
        if total_images > 1000:  # Reasonable number for ASL dataset
            print("Dataset appears to be complete. You can proceed with training.")
            return
        else:
            print("Dataset appears incomplete. Proceeding with download...")
    
    # Setup Kaggle API
    if not setup_kaggle_api():
        print("\nAlternative: Manual Download")
        print("1. Go to https://www.kaggle.com/grassknoted/asl-alphabet")
        print("2. Download the dataset manually")
        print("3. Extract it to a folder named 'asl_alphabet_train'")
        print("4. Ensure the folder contains subfolders A, B, C, ..., Z")
        return
    
    # Download dataset
    if not download_asl_dataset():
        return
    
    # Organize dataset
    if not organize_dataset():
        return
    
    # Cleanup
    cleanup_downloads()
    
    print("\nDataset setup completed successfully!")
    print("You can now run 'train_model.py' to train the ASL recognition model.")

if __name__ == "__main__":
    main()
