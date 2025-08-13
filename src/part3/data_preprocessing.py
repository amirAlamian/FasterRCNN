import cv2
import numpy as np
import json
import os
import sys
from collections import Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from config import (TRAIN_IMAGES, TRAIN_LABELS, CHARACTERS, char_to_idx)

def load_json_annotation(json_path):
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        
        return None

def validate_bbox_ultra_robust(bbox, image_shape):
    
    if not isinstance(bbox, dict): return None
    
    try:
        
        if 'boundingBox' in bbox: bb = bbox['boundingBox']
        else: bb = bbox
        
        x, y, w, h = int(bb.get('x', 0)), int(bb.get('y', 0)), int(bb.get('width', 0)), int(bb.get('height', 0))

        
        if w < 3 or h < 3: return None

        img_h, img_w = image_shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        if w <= 0 or h <= 0: return None
        
        return {'x': x, 'y': y, 'width': w, 'height': h}
    except (TypeError, ValueError):
        return None

def preprocess_for_resnet(image):
    
    if image is None or image.size == 0:
        return None
    
    try:
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        resized = cv2.resize(thresh, (224, 224), interpolation=cv2.INTER_AREA)
        
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        
        normalized = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        return normalized.transpose(2, 0, 1).astype(np.float32)
        
    except cv2.error as e:
        
        return None
    except Exception as e:
        
        return None

def process_labeled_data_for_resnet():
    
    print(" Processing initial labeled data (images 0-7)...")
    labeled_samples = []

    for img_id in range(8): 
        img_path = os.path.join(TRAIN_IMAGES, f"{img_id}.png")
        json_path = os.path.join(TRAIN_LABELS, f"{img_id}.json")

        if not all(os.path.exists(p) for p in [img_path, json_path]): continue

        annotation = load_json_annotation(json_path)
        if not annotation: continue

        try:
            image = cv2.imread(img_path)
            if image is None: continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            continue

        expression = annotation.get('expression', '')
        bboxes = annotation.get('annotations', [])
        
        
        sorted_bboxes = sorted([b for b in bboxes if validate_bbox_ultra_robust(b, image.shape)], key=lambda b: b['boundingBox']['x'])

        if not expression or not sorted_bboxes or len(expression) != len(sorted_bboxes):
            continue

        for char_label, bbox in zip(expression, sorted_bboxes):
            if char_label in char_to_idx:
                v_bbox = validate_bbox_ultra_robust(bbox, image.shape)
                if v_bbox:
                    x, y, w, h = v_bbox['x'], v_bbox['y'], v_bbox['width'], v_bbox['height']
                    char_img = image[y:y+h, x:x+w]
                    
                    processed_char = preprocess_for_resnet(char_img)
                    if processed_char is not None:
                        labeled_samples.append({
                            'image': processed_char,
                            'label': char_to_idx[char_label],
                            'char': char_label
                        })

    print(f"  > Found {len(labeled_samples)} valid characters.")
    if labeled_samples:
        char_counts = Counter(s['char'] for s in labeled_samples)
        print("  > Character distribution:", dict(sorted(char_counts.items())))
        
    return labeled_samples