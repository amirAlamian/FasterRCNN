import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
 

# Ignore libpng warnings
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
    """
    Create required output folders.
    """
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
    """
    تصویر و اطلاعات کادرهای محاطی را بارگذاری می‌کند.
    در این نسخه، کادرها قبل از بازگردانده شدن به اندازه 5% بزرگ‌تر می‌شوند.
    """
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
            # --- مرحله جدید: محاسبه و اعمال حاشیه  0.1 - 15 درصدی ---
            pad_w = int(-w * 0.02)
            pad_h = int(h * 0.15)
        elif image_id == 222 :
            # --- مرحله جدید: محاسبه و اعمال حاشیه  30 - 12 درصدی ---
            pad_w = int(w * 0.12)
            pad_h = int(h * 0.3)
        elif image_id == 223 :
            # --- مرحله جدید: محاسبه و اعمال حاشیه  5 - 12 درصدی ---
            pad_w = int(w * 0.12)
            pad_h = int(h * 0.05)
        else:
            # --- مرحله جدید: محاسبه و اعمال حاشیه  8.5 - 15 درصدی ---
            pad_w = int(w * 0.085)
            pad_h = int(h * 0.15)

        # محاسبه مختصات جدید با احتساب حاشیه
        x1 = x - pad_w
        y1 = y - pad_h
        x2 = x + w + pad_w
        y2 = y + h + pad_h
        
        # اطمینان از اینکه کادر جدید از ابعاد تصویر اصلی بیرون نمی‌زند
        final_x1 = max(0, x1)
        final_y1 = max(0, y1)
        final_x2 = min(img_w, x2)
        final_y2 = min(img_h, y2)
        
        boxes.append((final_x1, final_y1, final_x2, final_y2))
    
    sorted_boxes = sorted(boxes, key=lambda b: b[0])
    return image, sorted_boxes



def remove_horizontal_lines(char_image):
    """
    این تابع فقط خطوط افقی را حذف می‌کند که تمام عرض تصویر را پوشش می‌دهند.
    """
    # بررسی ورودی
    if char_image is None or char_image.size == 0:
        print("خطا: تصویر ورودی خالی است.")
        return char_image

    # ۱. تبدیل تصویر به سیاه و سفید (اگر رنگی بود)
    if len(char_image.shape) > 2:
        gray = cv2.cvtColor(char_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = char_image

    # ۲. آستانه‌گذاری برای جدا کردن پیش‌زمینه
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # ۳. شناسایی خطوط افقی یکسره
    w = char_image.shape[1]
    
    # --- تغییر کلیدی در این خط است ---
    # عرض هسته را برابر با عرض کامل تصویر قرار می‌دهیم تا فقط خطوط یکسره پیدا شوند.
    kernel_width = w 
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # ۴. ضخیم کردن خطوط شناسایی‌شده
    dilated_lines = cv2.dilate(detected_lines, np.ones((3,3)), iterations=2)

    # ۵. ترمیم تصویر اصلی
    result = cv2.inpaint(char_image, dilated_lines, inpaintRadius=5, flags=cv2.INPAINT_NS)
    
    return result

def count_border_contacts(component_mask, border_side):
    """
    تعداد تماس‌های مجزای یک جزء با یک لبه مشخص را می‌شمارد.
    """
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

    # شمارش تعداد گروه‌های پیوسته از پیکسل‌های غیرصفر
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
    """
    الگوریتم پاکسازی را اجرا می‌کند.
    این نسخه شامل سه شرط فیلترینگ است.
    """
    if char_crop.size == 0:
        return np.ones((10, 10, 3), dtype=np.uint8) * 255

    # --- مرحله موقت: تحلیل روی نسخه سیاه و سفید ---
    char_gray = cv2.cvtColor(char_crop, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(char_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # -----------------------------------------------

    # پیدا کردن تمام اجزای متصل در نسخه سیاه و سفید
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    
    total_area = bin_img.size
    # ایجاد یک "ماسک" برای نگهداری اجزای معتبر
    cleaned_mask = np.zeros_like(bin_img)

    # --- فیلتر کردن نویزها با منطق جدید ---
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # استخراج مشخصات کادر دور جزء
        x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w0, h0 = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        # شرط اول: حذف نویزهای کوچک متصل به لبه
        touches_border = (x0 <= 1 or y0 <= 1 or (x0 + w0) >= bin_img.shape[1] - 1 or (y0 + h0) >= bin_img.shape[0] - 1)
        if touches_border and (area / total_area) < 0.046:
            continue
            
        # شرط دوم: حذف تمام نویزهای بسیار کوچک (با مساحت کمتر از 0.7%)
        if (area / total_area) < 0.0033:
            continue

        # محاسبه خطی که ۷۰٪ بالای تصویر را از ۳۰٪ پایین جدا می‌کند
        seventy_percent_line = bin_img.shape[0] * 0.64

        # پیدا کردن مختصات y پایین‌ترین نقطه جزء فعلی
        component_bottom_edge = y0 + h0

        is_under_6_percent = (area / total_area) < 0.06
        # شرط اصلی: بررسی اینکه آیا مرکز جزء بالاتر از خط ۷۰٪ قرار دارد یا خیر
        if touches_border and component_bottom_edge < seventy_percent_line and is_under_6_percent:
            # این کد فقط برای اجزایی اجرا می‌شود که در ۳۰٪ پایینی تصویر نیستند.
            # به عنوان مثال، می‌توانید دستور continue را اینجا قرار دهید.
            continue

        # --- شرط جدید شما در اینجا اضافه شده است ---
        # ۱. بررسی موقعیت: آیا جزء کاملا در ۷۰٪ بالایی تصویر است؟
        is_in_top_80_percent = (y0 + h0) < (bin_img.shape[0] * 0.8)

        if is_in_top_80_percent:
            # ۲. بررسی تماس‌های چندگانه با لبه‌ها
            component_mask = (labels == i).astype(np.uint8)
            
            top_contacts = count_border_contacts(component_mask, 'top')
            left_contacts = count_border_contacts(component_mask, 'left')
            right_contacts = count_border_contacts(component_mask, 'right')
            
            # ۳. اعمال شرط: اگر با هر یک از لبه‌ها ۲ بار یا بیشتر تماس داشت، حذف کن
            if top_contacts >= 2 or left_contacts >= 2 or right_contacts >= 2:
                continue

        
        # فقط اجزای اصلی و معتبر در ماسک علامت‌گذاری می‌شوند
        cleaned_mask[labels == i] = 255

    # --- مرحله نهایی: بازسازی تصویر با رنگ‌های اصلی ---
    cleaned_image = np.ones_like(char_crop, dtype=np.uint8) * 255
    cleaned_image[cleaned_mask == 255] = char_crop[cleaned_mask == 255]
    
    return cleaned_image

def normalize_char_with_antialiasing(char_crop, final_size=(28, 28)):
    """
    ابتدا تصویر را باینری کرده و سپس با تغییر اندازه و استفاده از anti-aliasing،
    سطوح خاکستری را در لبه‌ها ایجاد می‌کند (رویکرد استاندارد MNIST).
    """
    # --- مرحله 1: سیاه و سفید کردن مطلق (Binarization) ---
    
    # ابتدا به خاکستری تبدیل می‌کنیم
    if len(char_crop.shape) == 3:
        gray_image = cv2.cvtColor(char_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = char_crop
        
    # با استفاده از اتسو، بهترین آستانه را پیدا کرده و تصویر را باینری می‌کنیم.
    # THRESH_BINARY_INV کاراکتر را سفید و پس‌زمینه را سیاه می‌کند که استاندارد است.
    _, binarized_image = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # --- مرحله 2: نرمال‌سازی سایز و ایجاد لبه‌های خاکستری ---

    # برای حفظ نسبت ابعاد، تصویر باینری را در مرکز یک بوم مربع قرار می‌دهیم
    h, w = binarized_image.shape
    max_dim = max(h, w)
    
    # حاشیه اضافه می‌کنیم تا کاراکتر به لبه‌ها نچسبد
    padding = int(max_dim * 0.15) # 15% حاشیه
    square_canvas = np.zeros((max_dim + padding*2, max_dim + padding*2), dtype=np.uint8)

    x_offset = (square_canvas.shape[1] - w) // 2
    y_offset = (square_canvas.shape[0] - h) // 2
    
    square_canvas[y_offset:y_offset+h, x_offset:x_offset+w] = binarized_image
    
    # تغییر اندازه نهایی. cv2.INTER_AREA به طور خودکار anti-aliasing را اعمال می‌کند
    # و سطوح خاکستری مورد نظر شما را در لبه‌ها ایجاد می‌کند.
    final_image = cv2.resize(square_canvas, final_size, interpolation=cv2.INTER_AREA)
    
    return final_image

def display_comparison(image_id, raw_images, cleaned_images, normalized_images):
    """
    نتایج را در سه مرحله مقایسه‌ای (خام، پاک‌شده و نرمال‌شده) نمایش می‌دهد.
    """
    num_chars = len(raw_images)
    if num_chars == 0:
        return

    # تعداد ردیف‌ها به 3 افزایش یافت
    fig, axs = plt.subplots(3, num_chars, figsize=(2 * num_chars, 6), facecolor='lightgray')
    
    # اگر فقط یک کاراکتر وجود داشته باشد، axs را برای اندیس‌دهی دو بعدی آماده می‌کنیم
    if num_chars == 1:
        axs = np.array([axs]).T

    for i in range(num_chars):
        # ردیف اول: تصاویر خام
        axs[0, i].imshow(cv2.cvtColor(raw_images[i], cv2.COLOR_BGR2RGB))
        axs[0, i].axis('off')
        
        # ردیف دوم: تصاویر پاک‌شده
        axs[1, i].imshow(cv2.cvtColor(cleaned_images[i], cv2.COLOR_BGR2RGB))
        axs[1, i].axis('off')

        # --- ردیف جدید: تصاویر نرمال‌شده ---
        # این تصاویر خاکستری هستند، بنابراین از cmap='gray' استفاده می‌کنیم
        axs[2, i].imshow(normalized_images[i], cmap='gray')
        axs[2, i].axis('off')

    # تنظیم برچسب برای هر ردیف
    axs[0, 0].set_ylabel("Raw", fontsize=12)
    axs[1, 0].set_ylabel("Cleaned", fontsize=12)
    axs[2, 0].set_ylabel("Normalized", fontsize=12) # <-- برچسب ردیف جدید
    
    plt.suptitle(f"Image {image_id}.png – Comparison", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

def process_image_by_id(image_id, paths):
    """
    تابع اصلی که فرآیند را برای یک تصویر مدیریت می‌کند.
    """
    # 1. بارگذاری داده‌ها
    image, boxes = load_data(image_id, paths)
    if image is None:
        return

    raw_chars = []
    clean_chars = []
    normalized_chars = []

    # 2. حلقه روی هر کادر برای پردازش کاراکترها
    for idx, (x1, y1, x2, y2) in enumerate(boxes, 1):

        # برش کاراکتر خام
        char_crop = image[y1:y2, x1:x2]
        if char_crop.size == 0:
            continue # به سراغ کادر بعدی در همین تصویر می‌رود
        cv2.imwrite(os.path.join(paths["raw_output"], f"{image_id}_{idx:02d}.png"), char_crop)
        raw_chars.append(char_crop)

        

        # پاکسازی کاراکتر
        line_removed_char = remove_horizontal_lines(char_crop)
        cleaned_char = clean_character_image(line_removed_char)
        cv2.imwrite(os.path.join(paths["clean_output"], f"{image_id}_{idx:02d}.png"), cleaned_char)
        clean_chars.append(cleaned_char)

        # پاکسازی نهایی
        normalized_char = normalize_char_with_antialiasing(cleaned_char)
        cv2.imwrite(os.path.join(paths["normalized_output"], f"{image_id}_{idx:02d}.png"), normalized_char)
        normalized_chars.append(normalized_char)
    
    # 3. نمایش نتایج
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