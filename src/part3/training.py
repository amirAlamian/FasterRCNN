

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import cv2
import os
import random
from collections import Counter
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from config import (
    BATCH_SIZE, NUM_WORKERS, AUGMENTATION_MULTIPLIER,
    SSL_CONFIDENCE_THRESHOLD, SSL_CONFIDENCE_GAP, device,
    TRAIN_IMAGES, TRAIN_LABELS, RESULTS_ROOT, CHARACTERS, idx_to_char
)
from data_preprocessing import process_labeled_data_for_resnet
from dataset import ResNetCharacterDataset, AugmentedResNetDataset

from model import ResNet18Classifier

def run_training_phase(model, train_loader, val_loader, epochs, learning_rate, phase_name="Training"):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, verbose=True)
    
    best_val_acc = 0.0
    best_model_path = os.path.join(RESULTS_ROOT, f'resnet18_{phase_name.lower()}.pth')

    print(f"\n Starting Phase: {phase_name} | Epochs: {epochs} | LR: {learning_rate}")
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [{phase_name}]")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix(loss=f"{total_loss / (total / BATCH_SIZE):.4f}", acc=f"{100. * correct / total:.2f}%")

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100. * val_correct / val_total
        print(f"  Validation Accuracy: {val_acc:.2f}%")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"   New best validation accuracy: {best_val_acc:.2f}%. Model saved.")
            
        if optimizer.param_groups[0]['lr'] < 1e-7:
            print("  Learning rate too low, stopping training early.")
            break
            
    model.load_state_dict(torch.load(best_model_path))
    return model, best_val_acc

def perform_semi_supervised_learning(model):
    
    print("\n Starting Semi-Supervised Learning Phase")
    model.eval()
    pseudo_labeled_samples = []

    unlabeled_image_indices = range(8, 300) 
    
    for img_id in tqdm(unlabeled_image_indices, desc="Scanning Unlabeled Data"):
        img_path = os.path.join(TRAIN_IMAGES, f"{img_id}.png")
        json_path = os.path.join(TRAIN_LABELS, f"{img_id}.json")

        if not all(os.path.exists(p) for p in [img_path, json_path]): continue
            
        try:
            image = cv2.imread(img_path)
            if image is None: continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            annotation = load_json_annotation(json_path)
            if not annotation or 'annotations' not in annotation: continue

            for bbox in annotation['annotations']:
                validated = validate_bbox_ultra_robust(bbox, image.shape)
                if not validated: continue

                x, y, w, h = validated['x'], validated['y'], validated['width'], validated['height']
                char_img = image[y:y+h, x:x+w]

                if char_img.size == 0: continue
                
                processed_char = preprocess_for_resnet(char_img)
                if processed_char is None: continue

                char_tensor = torch.from_numpy(processed_char).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(char_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                    confidence, predicted_idx = torch.max(probabilities, 0)

                    if confidence.item() > SSL_CONFIDENCE_THRESHOLD:
                        sorted_probs, _ = torch.sort(probabilities, descending=True)
                        if len(sorted_probs) > 1 and (sorted_probs[0] - sorted_probs[1]) > SSL_CONFIDENCE_GAP:
                            pseudo_labeled_samples.append({
                                'image': processed_char,
                                'label': predicted_idx.item(),
                                'char': idx_to_char[predicted_idx.item()]
                            })
        except Exception:
            continue

    print(f"  > Generated {len(pseudo_labeled_samples)} high-confidence pseudo-labels.")
    if pseudo_labeled_samples:
        char_counts = Counter(s['char'] for s in pseudo_labeled_samples)
        print("  > Pseudo-label distribution:", dict(sorted(char_counts.items())))
            
    return pseudo_labeled_samples

def run_complete_training_pipeline():
    
    
    print("--- Stage 1: Initial Training on Labeled Data ---")
    labeled_data = process_labeled_data_for_resnet()
    if not labeled_data:
        print(" Fatal Error: No labeled data found. Aborting training.")
        return

    model = ResNet18Classifier(num_classes=len(CHARACTERS)).to(device)
    
    random.shuffle(labeled_data)
    split_idx = int(0.8 * len(labeled_data))
    train_samples, val_samples = labeled_data[:split_idx], labeled_data[split_idx:]
    
    train_dataset = AugmentedResNetDataset(ResNetCharacterDataset(train_samples), multiplier=AUGMENTATION_MULTIPLIER)
    val_dataset = ResNetCharacterDataset(val_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    
    model, best_acc = run_training_phase(model, train_loader, val_loader, epochs=40, learning_rate=1e-3, phase_name="best")
    print(f"\n Initial training finished with best validation accuracy: {best_acc:.2f}%")

    
    pseudo_labels = perform_semi_supervised_learning(model)
    if not pseudo_labels or len(pseudo_labels) < 50:
        print("\nInsufficient pseudo-labels generated. Training complete.")
        return

    
    print("\n--- Stage 2: Fine-tuning with Pseudo-Labels ---")
    combined_data = labeled_data + pseudo_labels
    random.shuffle(combined_data)
    
    split_idx_ssl = int(0.8 * len(combined_data))
    train_samples_ssl, val_samples_ssl = combined_data[:split_idx_ssl], combined_data[split_idx_ssl:]
    
    train_dataset_ssl = AugmentedResNetDataset(ResNetCharacterDataset(train_samples_ssl), multiplier=15)
    val_dataset_ssl = ResNetCharacterDataset(val_samples_ssl)
    
    train_loader_ssl = DataLoader(train_dataset_ssl, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader_ssl = DataLoader(val_dataset_ssl, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    
    model, best_ssl_acc = run_training_phase(model, train_loader_ssl, val_loader_ssl, epochs=20, learning_rate=5e-4, phase_name="ssl")
    print(f"\n SSL fine-tuning finished with best validation accuracy: {best_ssl_acc:.2f}%")