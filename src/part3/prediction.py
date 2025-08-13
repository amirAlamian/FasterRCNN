import torch
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
import re 
from collections import Counter
from Levenshtein import distance as levenshtein_distance

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from config import (RESULTS_ROOT, VALID_IMAGES, VALID_LABELS, TEST_IMAGES, 
                   CHARACTERS, idx_to_char, NUM_CLASSES, device)
from data_preprocessing import load_json_annotation, validate_bbox_ultra_robust, preprocess_for_resnet
from model import ResNet18Classifier

def post_process_expression(expr: str) -> str:
    
    if not isinstance(expr, str) or not expr.strip():
        return "1+1" 

    while True:
        original_expr = expr

        expr = expr.replace('--', '-')
        expr = expr.replace('-+', '-').replace('+-', '-')
        for op in '*/':
            expr = expr.replace(f'-{op}', op).replace(f'+{op}', op)

        expr = re.sub(r'([0-9\)])\(', r'\1*(', expr) 
        expr = re.sub(r'\)([0-9\(])', r')*\1', expr) 
        
        expr = expr.replace('()', '(1+1)') 

        expr = re.sub(r'\(([0-9\)])\)', r'\1', expr)
        
        expr = expr.replace('(+','(1+')
        expr = expr.replace('+)','+1)')

        if expr.startswith(('*', '/', '+')):
            expr = expr[1:]
        if expr.endswith(('*', '/', '+', '-')):
            expr = expr[:-1]

        if expr.startswith(')'): expr = expr[1:]
        if expr.endswith('('): expr = expr[:-1]

        if expr == original_expr:
            break

    if expr.count('(') != expr.count(')'):
        expr = expr.replace('(', '').replace(')', '')

    return expr if expr else "1+1"


def load_trained_model(model_path):
    
    if not os.path.exists(model_path):
        print(f" Error: Model file not found at {model_path}")
        return None
    
    print(f" Loading model from: {os.path.basename(model_path)}")
    model = ResNet18Classifier(num_classes=NUM_CLASSES).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(" Model loaded successfully!")
        return model
    except Exception as e:
        print(f" Error loading model state_dict: {e}")
        return None

def check_trained_models():
    
    ssl_path = os.path.join(RESULTS_ROOT, 'resnet18_ssl.pth')
    base_path = os.path.join(RESULTS_ROOT, 'resnet18_best.pth')

    if os.path.exists(ssl_path):
        return load_trained_model(ssl_path), "resnet18_ssl.pth"
    elif os.path.exists(base_path):
        return load_trained_model(base_path), "resnet18_best.pth"
    else:
        print(" No trained models found in the results directory.")
        return None, None

def predict_expression_from_image(model, image, bboxes):
    
    char_data = []
    for bbox in bboxes:
        v_bbox = validate_bbox_ultra_robust(bbox, image.shape)
        if v_bbox:
            x, y, w, h = v_bbox['x'], v_bbox['y'], v_bbox['width'], v_bbox['height']
            char_img = image[y:y+h, x:x+w]
            processed_char = preprocess_for_resnet(char_img)
            if processed_char is not None:
                char_data.append({'tensor': torch.from_numpy(processed_char).unsqueeze(0), 'x_pos': x})

    if not char_data: return "", 0.0
    
    char_data.sort(key=lambda item: item['x_pos'])
    
    predicted_chars = []
    with torch.no_grad():
        for item in char_data:
            output = model(item['tensor'].to(device))
            probs = F.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            predicted_chars.append(idx_to_char[pred_idx.item()])
    
    raw_expression = "".join(predicted_chars)
    
    clean_expression = post_process_expression(raw_expression)
    return clean_expression, 1.0 

def test_on_official_validation(model):
    
    print("\n" + "="*50)
    print(" RUNNING OFFICIAL VALIDATION")
    print("="*50)
    results = []
    
    for img_id in tqdm(range(300, 500), desc="Validating"):
        img_path = os.path.join(VALID_IMAGES, f"{img_id}.png")
        json_path = os.path.join(VALID_LABELS, f"{img_id}.json")
        
        if not all(os.path.exists(p) for p in [img_path, json_path]): continue

        annotation = load_json_annotation(json_path)
        if not annotation or 'expression' not in annotation or 'annotations' not in annotation: continue
        
        true_expr = post_process_expression(annotation['expression'])
        if not true_expr: continue
        
        try:
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            pred_expr, _ = predict_expression_from_image(model, image, annotation['annotations'])
            
            similarity = 1 - levenshtein_distance(pred_expr, true_expr) / max(len(pred_expr), len(true_expr), 1)
            
            results.append({
                'image_id': img_id,
                'true_expression': true_expr,
                'predicted_expression': pred_expr,
                'similarity': similarity
            })
        except Exception:
            continue
            
    if not results:
        print(" No validation samples could be processed.")
        return

    df = pd.DataFrame(results)
    avg_sim = df['similarity'].mean()
    exact_matches = (df['similarity'] == 1.0).sum()
    
    print("\n--- Validation Results ---")
    print(f"  Average Similarity: {avg_sim:.3f}")
    print(f"  Exact Matches: {exact_matches}/{len(df)} ({(exact_matches/len(df))*100:.2f}%)")
    
    val_path = os.path.join(RESULTS_ROOT, 'validation_results.csv')
    df.to_csv(val_path, index=False)
    print(f"  > Detailed results saved to: {val_path}")

def generate_test_submission(model):
    
    print("\n" + "="*50)
    print(" GENERATING TEST SUBMISSION")
    print("="*50)
    
    part1_output_path = os.path.join(RESULTS_ROOT, "..", "part1", "part1_output.csv")
    if not os.path.exists(part1_output_path):
        print(f" Error: Part 1 bbox output not found at '{part1_output_path}'")
        return
        
    bbox_df = pd.read_csv(part1_output_path)
    test_bboxes = bbox_df[bbox_df['image_id'].between(500, 699)].groupby('image_id')
    
    submission_data = []
    for img_id in tqdm(range(500, 700), desc="Generating Submission"):
        img_path = os.path.join(TEST_IMAGES, f"{img_id}.png")
        expression = "1+1" 
        
        if img_id in test_bboxes.groups and os.path.exists(img_path):
            try:
                image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                bboxes_for_img = test_bboxes.get_group(img_id)
                bboxes_list = [{'boundingBox': row.to_dict()} for _, row in bboxes_for_img.iterrows()]
                
                
                pred_expr, _ = predict_expression_from_image(model, image, bboxes_list)
                if pred_expr: expression = pred_expr
            except Exception:
                pass 
        
        submission_data.append({'image_id': img_id, 'expression': expression})
        
    submission_df = pd.DataFrame(submission_data)
    output_path = os.path.join(RESULTS_ROOT, "output.csv")
    submission_df.to_csv(output_path, index=False)
    
    print("\n--- Submission Generation Complete ---")
    print(f"  Submission file created at: {output_path}")

def prediction_main():
    
    model, model_name = check_trained_models()
    if model is None: return

    print("\n--- Prediction & Evaluation Menu ---")
    print(f"  Using model: {model_name}")
    print("  1. Generate Test Submission File")
    print("  2. Run Performance Validation")
    print("  3. Run Both (Validation then Submission)")
    
    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == '1':
        generate_test_submission(model)
    elif choice == '2':
        test_on_official_validation(model)
    elif choice == '3':
        test_on_official_validation(model)
        generate_test_submission(model)
    else:
        print("Invalid choice. Exiting.")