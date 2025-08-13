import os
import sys
import torch
import numpy as np
import random

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  

BATCH_SIZE = 64  
NUM_WORKERS = 2 
AUGMENTATION_MULTIPLIER = 30 
SSL_CONFIDENCE_THRESHOLD = 0.96  
SSL_CONFIDENCE_GAP = 0.35 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_project_root():
    
    current_dir = os.path.abspath(os.path.dirname(__file__))
    
    for _ in range(4):
        if os.path.exists(os.path.join(current_dir, "dataset")):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

PROJECT_ROOT = get_project_root()
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "part3")


TRAIN_IMAGES = os.path.join(DATASET_ROOT, "train", "images")
TRAIN_LABELS = os.path.join(DATASET_ROOT, "train", "labels")
VALID_IMAGES = os.path.join(DATASET_ROOT, "valid", "images")
VALID_LABELS = os.path.join(DATASET_ROOT, "valid", "labels")
TEST_IMAGES = os.path.join(DATASET_ROOT, "test", "images")


os.makedirs(RESULTS_ROOT, exist_ok=True)


CHARACTERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              '+', '-', '*', '/', '(', ')']

char_to_idx = {char: idx for idx, char in enumerate(CHARACTERS)}
idx_to_char = {idx: char for idx, char in enumerate(CHARACTERS)}
NUM_CLASSES = len(CHARACTERS)


def print_config():
    
    print("--- Configuration ---")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Batch Size: {BATCH_SIZE}, Workers: {NUM_WORKERS}")
    print(f"  Classes: {NUM_CLASSES} ('{''.join(CHARACTERS)}')")
    print("-" * 21)

def verify_paths():
    
    print("Verifying essential paths...")
    required_dirs = [DATASET_ROOT, TRAIN_IMAGES, TRAIN_LABELS, VALID_IMAGES, VALID_LABELS, TEST_IMAGES]
    missing_dirs = [p for p in required_dirs if not os.path.exists(p)]
    
    if not missing_dirs:
        print(" All paths verified successfully.")
        return True
    else:
        print(" Error: Missing required directories:")
        for path in missing_dirs:
            print(f"  - {path}")
        return False