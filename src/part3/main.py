

import sys
import os
import torch


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

try:
    from config import print_config, verify_paths
    
    from training import run_complete_training_pipeline
    from prediction import prediction_main, check_trained_models
except ImportError as e:
    print(f" Critical Import Error: {e}\n. Please ensure all required files exist and you are running from the correct directory.")
    sys.exit(1)

def print_header():
    """Prints the main header for the application."""
    print("=" * 60)
    print("  Part 3: Expression Recognition - End-to-End Pipeline")
    print("=" * 60)

def main():
    """Main entry point for the entire application."""
    print_header()
    print_config()
    
    if not verify_paths():
        sys.exit(1)
        
    
    models_exist, _ = check_trained_models()
    
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("\n--- Execution Mode ---")
        if models_exist:
            print("  Trained models found. What would you like to do?")
            print("  1. Inference (Use existing models for prediction/validation)")
            print("  2. Re-train (Delete existing models and train from scratch)")
            choice = input("Enter your choice (1/2): ").strip()
            mode = "--inference" if choice == '1' else "--train"
        else:
            print("  No trained models found. Training is required.")
            mode = "--train"
    
    
    if mode == "--train":
        print("\n" + "="*50)
        print(" STARTING FULL TRAINING PIPELINE")
        print("="*50)
        
        run_complete_training_pipeline()
        print("\n Training complete. Proceeding to inference...")
        prediction_main()
    elif mode == "--inference":
        if not models_exist:
            print(" Error: --inference mode selected, but no models found.")
            sys.exit(1)
        print("\n Models found. Proceeding to inference...")
        prediction_main()
    else:
        print(f"Unknown argument: {mode}. Use --train or --inference.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 User interrupted. Exiting.")
        sys.exit(0)