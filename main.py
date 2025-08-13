"""
ASL Alphabet Recognition - Main Launcher

This script provides a simple menu interface to run different components
of the ASL alphabet recognition system.
"""

import os
import sys
import subprocess

def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("           ASL Alphabet Recognition System")
    print("=" * 60)
    print("Real-time American Sign Language Alphabet Recognition")
    print("Using Computer Vision and Deep Learning")
    print("=" * 60)

def check_file_exists(filepath, description):
    """Check if a required file exists"""
    if os.path.exists(filepath):
        print(f"✓ {description} found")
        return True
    else:
        print(f"✗ {description} not found")
        return False

def check_requirements():
    """Check if all required files exist"""
    print("\nChecking system requirements...")
    
    requirements = [
        ("asl_alphabet_train", "Dataset folder"),
        ("asl_model.h5", "Trained model"),
        ("class_names.npy", "Class labels")
    ]
    
    all_exist = True
    for filepath, description in requirements:
        if not check_file_exists(filepath, description):
            all_exist = False
    
    return all_exist

def run_script(script_name, description):
    """Run a Python script"""
    print(f"\n{description}...")
    print("-" * 40)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, text=True)
        if result.returncode == 0:
            print(f"\n✓ {description} completed successfully!")
        else:
            print(f"\n✗ {description} failed with return code {result.returncode}")
    except FileNotFoundError:
        print(f"✗ Script '{script_name}' not found!")
    except KeyboardInterrupt:
        print(f"\n⚠ {description} interrupted by user")
    except Exception as e:
        print(f"✗ Error running {description}: {e}")

def show_menu():
    """Display main menu"""
    print("\nMain Menu:")
    print("1. Setup Dataset (Download ASL alphabet images)")
    print("2. Train Model (Train CNN for ASL recognition)")
    print("3. Run Real-time Detection (Basic webcam recognition)")
    print("4. Run Enhanced Detection (With hand tracking)")
    print("5. Check System Status")
    print("6. Install Requirements")
    print("7. Exit")
    print("-" * 40)

def install_requirements():
    """Install Python requirements"""
    print("\nInstalling Python requirements...")
    print("-" * 40)
    
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                              capture_output=False, text=True)
        if result.returncode == 0:
            print("\n✓ Requirements installed successfully!")
        else:
            print(f"\n✗ Installation failed with return code {result.returncode}")
    except Exception as e:
        print(f"✗ Error installing requirements: {e}")

def main():
    """Main application loop"""
    print_banner()
    
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (1-7): ").strip()
            
            if choice == "1":
                # Setup Dataset
                if os.path.exists("asl_alphabet_train"):
                    print("\nDataset folder already exists!")
                    overwrite = input("Do you want to re-download? (y/n): ").lower()
                    if overwrite != 'y':
                        continue
                
                run_script("setup_dataset.py", "Setting up ASL dataset")
                
            elif choice == "2":
                # Train Model
                if not os.path.exists("asl_alphabet_train"):
                    print("\n✗ Dataset not found! Please run option 1 first.")
                    continue
                
                if os.path.exists("asl_model.h5"):
                    print("\nTrained model already exists!")
                    retrain = input("Do you want to retrain? (y/n): ").lower()
                    if retrain != 'y':
                        continue
                
                run_script("train_model.py", "Training ASL recognition model")
                
            elif choice == "3":
                # Run Basic Detection
                if not os.path.exists("asl_model.h5"):
                    print("\n✗ Trained model not found! Please run option 2 first.")
                    continue
                
                print("\nStarting basic real-time ASL detection...")
                print("Make sure your webcam is connected!")
                input("Press Enter to continue...")
                
                run_script("detect_asl.py", "Running basic real-time ASL detection")
                
            elif choice == "4":
                # Run Enhanced Detection
                if not os.path.exists("asl_model.h5"):
                    print("\n✗ Trained model not found! Please run option 2 first.")
                    continue
                
                print("\nStarting enhanced real-time ASL detection...")
                print("This version includes hand tracking for better accuracy.")
                print("Make sure your webcam is connected!")
                
                # Check if MediaPipe is available
                try:
                    import mediapipe
                    print("✓ MediaPipe detected - hand tracking enabled")
                except ImportError:
                    print("⚠ MediaPipe not found - using basic detection")
                    print("Install with: pip install mediapipe")
                
                input("Press Enter to continue...")
                
                run_script("detect_asl_enhanced.py", "Running enhanced real-time ASL detection")
                
            elif choice == "5":
                # Check Status
                dataset_exists = os.path.exists("asl_alphabet_train")
                model_exists = os.path.exists("asl_model.h5")
                labels_exist = os.path.exists("class_names.npy")
                
                print("\nSystem Status:")
                print("-" * 20)
                print(f"Dataset: {'✓ Ready' if dataset_exists else '✗ Not found'}")
                print(f"Model: {'✓ Ready' if model_exists else '✗ Not trained'}")
                print(f"Labels: {'✓ Ready' if labels_exist else '✗ Not found'}")
                
                # Check optional dependencies
                try:
                    import mediapipe
                    print("MediaPipe: ✓ Available (enhanced hand tracking)")
                except ImportError:
                    print("MediaPipe: ✗ Not installed (optional for enhanced detection)")
                
                if dataset_exists and model_exists and labels_exist:
                    print("\n✓ System is ready for ASL detection!")
                elif dataset_exists:
                    print("\n⚠ Dataset ready. Please train the model (option 2).")
                else:
                    print("\n⚠ Please setup dataset first (option 1).")
                
                # Count dataset images if available
                if dataset_exists:
                    total_images = 0
                    for letter in [chr(i) for i in range(ord('A'), ord('Z') + 1)]:
                        letter_path = os.path.join('asl_alphabet_train', letter)
                        if os.path.exists(letter_path):
                            image_count = len([f for f in os.listdir(letter_path) 
                                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                            total_images += image_count
                    print(f"Dataset contains {total_images} images across 26 letters")
                
            elif choice == "6":
                # Install Requirements
                install_requirements()
                
            elif choice == "7":
                # Exit
                print("\nThank you for using ASL Alphabet Recognition!")
                print("Goodbye! 👋")
                break
                
            else:
                print("\n⚠ Invalid choice! Please enter 1-7.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n✗ An error occurred: {e}")
        
        print("\nPress Enter to continue...")
        input()

if __name__ == "__main__":
    main()
