import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
 

@contextmanager
def suppress_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        os.close(devnull)


def setup_directories(base_path):
    paths = {
        "images": os.path.join(base_path, 'images'),
        "labels": os.path.join(base_path, 'labels'),
        "raw_output": os.path.join(base_path, 'chars_output'),
        "clean_output": os.path.join(base_path, 'chars_output_clean'),
        "normalized_output": os.path.join(base_path, 'normalized_images')
    }
    os.makedirs(paths["raw_output"], exist_ok=True)
    os.makedirs(paths["clean_output"], exist_ok=True)
    os.makedirs(paths["normalized_output"], exist_ok=True)
    return paths

def load_data(image_id, paths):
    image_file = os.path.join(paths["images"], f"{image_id}.png")
    label_file = os.path.join(paths["labels"], f"{image_id}.json")

    if not os.path.exists(image_file) or not os.path.exists(label_file):
        print(f"Warning: Files for image_id '{image_id}' not found.")
        return None, None

    image = cv2.imread(image_file)
    if image is None:
        print(f"Warning: Could not read image file for '{image_id}'.")
        return None, None
        
    img_h, img_w, _ = image.shape

    with open(label_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    boxes = []
    for ann in data['annotations']:
        bb = ann['boundingBox']
        x, y, w, h = int(bb['x']), int(bb['y']), int(bb['width']), int(bb['height'])
        if image_id == 279 or image_id == 264 :
            pad_w = int(-w * 0.02)
            pad_h = int(h * 0.15)
        elif image_id == 222 :
            pad_w = int(w * 0.12)
            pad_h = int(h * 0.3)
        elif image_id == 223 :
            pad_w = int(w * 0.12)
            pad_h = int(h * 0.05)
        else:
            pad_w = int(w * 0.085)
            pad_h = int(h * 0.15)

        x1 = x - pad_w
        y1 = y - pad_h
        x2 = x + w + pad_w
        y2 = y + h + pad_h
        
        final_x1 = max(0, x1)
        final_y1 = max(0, y1)
        final_x2 = min(img_w, x2)
        final_y2 = min(img_h, y2)
        
        boxes.append((final_x1, final_y1, final_x2, final_y2))
    
    sorted_boxes = sorted(boxes, key=lambda b: b[0])
    return image, sorted_boxes



def remove_horizontal_lines(char_image):
    if char_image is None or char_image.size == 0:
        print("Empty image!")
        return char_image

    if len(char_image.shape) > 2:
        gray = cv2.cvtColor(char_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = char_image

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    w = char_image.shape[1]
    
    kernel_width = w 
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    dilated_lines = cv2.dilate(detected_lines, np.ones((3,3)), iterations=2)

    result = cv2.inpaint(char_image, dilated_lines, inpaintRadius=5, flags=cv2.INPAINT_NS)
    
    return result

def count_border_contacts(component_mask, border_side):

    if border_side == 'top':
        line = component_mask[0, :]
    elif border_side == 'bottom':
        line = component_mask[-1, :]
    elif border_side == 'left':
        line = component_mask[:, 0]
    elif border_side == 'right':
        line = component_mask[:, -1]
    else:
        return 0

    run_count = 0
    in_run = False
    for pixel in line:
        if pixel > 0 and not in_run:
            run_count += 1
            in_run = True
        elif pixel == 0:
            in_run = False
            
    return run_count

def clean_character_image(char_crop):
    if char_crop.size == 0:
        return np.ones((10, 10, 3), dtype=np.uint8) * 255

    char_gray = cv2.cvtColor(char_crop, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(char_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    
    total_area = bin_img.size
    cleaned_mask = np.zeros_like(bin_img)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w0, h0 = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        touches_border = (x0 <= 1 or y0 <= 1 or (x0 + w0) >= bin_img.shape[1] - 1 or (y0 + h0) >= bin_img.shape[0] - 1)
        if touches_border and (area / total_area) < 0.046:
            continue
            
        if (area / total_area) < 0.0033:
            continue

        seventy_percent_line = bin_img.shape[0] * 0.64

        component_bottom_edge = y0 + h0

        is_under_6_percent = (area / total_area) < 0.06
        if touches_border and component_bottom_edge < seventy_percent_line and is_under_6_percent:
            continue

        is_in_top_80_percent = (y0 + h0) < (bin_img.shape[0] * 0.8)

        if is_in_top_80_percent:
            component_mask = (labels == i).astype(np.uint8)
            
            top_contacts = count_border_contacts(component_mask, 'top')
            left_contacts = count_border_contacts(component_mask, 'left')
            right_contacts = count_border_contacts(component_mask, 'right')
            
            if top_contacts >= 2 or left_contacts >= 2 or right_contacts >= 2:
                continue

        
        cleaned_mask[labels == i] = 255

    cleaned_image = np.ones_like(char_crop, dtype=np.uint8) * 255
    cleaned_image[cleaned_mask == 255] = char_crop[cleaned_mask == 255]
    
    return cleaned_image

def normalize_char_with_antialiasing(char_crop, final_size=(28, 28)):
    if len(char_crop.shape) == 3:
        gray_image = cv2.cvtColor(char_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = char_crop
        
    _, binarized_image = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    h, w = binarized_image.shape
    max_dim = max(h, w)
    
    padding = int(max_dim * 0.15) 
    square_canvas = np.zeros((max_dim + padding*2, max_dim + padding*2), dtype=np.uint8)

    x_offset = (square_canvas.shape[1] - w) // 2
    y_offset = (square_canvas.shape[0] - h) // 2
    
    square_canvas[y_offset:y_offset+h, x_offset:x_offset+w] = binarized_image

    final_image = cv2.resize(square_canvas, final_size, interpolation=cv2.INTER_AREA)
    
    return final_image

def display_comparison(image_id, raw_images, cleaned_images, normalized_images):
    num_chars = len(raw_images)
    if num_chars == 0:
        return


    fig, axs = plt.subplots(3, num_chars, figsize=(2 * num_chars, 6), facecolor='lightgray')
    

    if num_chars == 1:
        axs = np.array([axs]).T

    for i in range(num_chars):
        axs[0, i].imshow(cv2.cvtColor(raw_images[i], cv2.COLOR_BGR2RGB))
        axs[0, i].axis('off')
        

        axs[1, i].imshow(cv2.cvtColor(cleaned_images[i], cv2.COLOR_BGR2RGB))
        axs[1, i].axis('off')


        axs[2, i].imshow(normalized_images[i], cmap='gray')
        axs[2, i].axis('off')


    axs[0, 0].set_ylabel("Raw", fontsize=12)
    axs[1, 0].set_ylabel("Cleaned", fontsize=12)
    axs[2, 0].set_ylabel("Normalized", fontsize=12)
    
    plt.suptitle(f"Image {image_id}.png – Comparison", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def process_image_by_id(image_id, paths):
    image, boxes = load_data(image_id, paths)
    if image is None:
        return

    raw_chars = []
    clean_chars = []
    normalized_chars = []

    for idx, (x1, y1, x2, y2) in enumerate(boxes, 1):

        char_crop = image[y1:y2, x1:x2]
        if char_crop.size == 0:
            continue
        cv2.imwrite(os.path.join(paths["raw_output"], f"{image_id}_{idx:02d}.png"), char_crop)
        raw_chars.append(char_crop)

        
        line_removed_char = remove_horizontal_lines(char_crop)
        cleaned_char = clean_character_image(line_removed_char)
        cv2.imwrite(os.path.join(paths["clean_output"], f"{image_id}_{idx:02d}.png"), cleaned_char)
        clean_chars.append(cleaned_char)

        normalized_char = normalize_char_with_antialiasing(cleaned_char)
        cv2.imwrite(os.path.join(paths["normalized_output"], f"{image_id}_{idx:02d}.png"), normalized_char)
        normalized_chars.append(normalized_char)
    
    # if image_id in [0, 4, 12, 23, 25, 36, 99, 192, 222, 223, 264, 266, 279] :
    #     display_comparison(image_id, raw_chars, clean_chars, normalized_chars)

if __name__ == "__main__":
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../dataset/train"))
    paths = setup_directories(base_path)

    excluded_numbers = {72, 160, 195, 219, 229, 241, 283}
    ids = [i for i in range(300) if i not in excluded_numbers]
    total = len(ids)

    for k, i in enumerate(ids, 1):
        print(f"\rProcessing {k}/{total} (ID {i})", end="", flush=True)
        with suppress_stderr():
            process_image_by_id(i, paths)

    print("\n All images processed successfully.")