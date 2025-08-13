import torchvision.transforms.functional as TF
from PIL import ImageOps, Image, ImageFilter
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import torch
import json
import cv2
import os
import re


def remove_horizontal_lines_pil(img, line_length=200, line_thickness=3):
    img_cv = np.array(img)
    if img_cv.ndim == 3:
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_cv

    _, img_bin = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_length, line_thickness))
    detected_lines = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=1)

    img_no_lines = cv2.bitwise_and(img_bin, cv2.bitwise_not(detected_lines))

    img_result = cv2.bitwise_not(img_no_lines)

    img_result_pil = Image.fromarray(img_result)
    return img_result_pil


def remove_shadows_pil(pil_img):
    img = np.array(pil_img.convert("L"))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 70))
    background = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    shadow_removed = cv2.subtract(background, img)
    norm = cv2.normalize(
        shadow_removed, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    return Image.fromarray(norm.astype(np.uint8))


def binarize_pil_image(threshold=128):
    def transform(image):
        image = ImageOps.grayscale(image)
        image = image.point(lambda x: 255 if x > threshold else 0, mode="1")
        return image.convert("RGB")

    return transform


class ResizeWithBoxes:
    def __init__(self, max_size=1500):
        self.max_size = max_size

    def __call__(self, image, target={}):
        orig_width, orig_height = image.size
        scale = self.max_size / orig_width
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        image = TF.resize(
            image, (new_height, new_width), interpolation=TF.InterpolationMode.BILINEAR
        )

        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= scale
            boxes[:, [1, 3]] *= scale
            target["boxes"] = boxes
            target["scale"] = torch.tensor(scale)
            target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            target["scale"] = scale

        return image, target


class ConvertToOriginalSize:
    def convert(self, image, target):
        scale = 1 / target["scale"]
        _, height, width = image.shape

        org_width = int(width * scale)
        org_height = int(height * scale)

        image = TF.resize(
            image, (org_height, org_width), interpolation=TF.InterpolationMode.BILINEAR
            )

        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= scale
            boxes[:, [1, 3]] *= scale
            target["boxes"] = boxes
            target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        return image, target


class ForceHorizontal:
    def __call__(self, img):
        if isinstance(img, Image.Image):
            width, height = img.size
            if height > width:
                img = img.rotate(-90, expand=True)
        else:
            _, height, width = img.shape
            if height > width:
                img = img.permute(0, 2, 1).flip(-2)

        return img


def light_blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=0.5))


images_with_lines = [509, 522, 574, 600, 676, 25, 309, 459, 475, 477]


class RemoveHorizontalLines:
    def __call__(self, img_pil, target):
        image_id = int(target["image_id"])
        if image_id in images_with_lines:
            img_np = np.array(img_pil)

            image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )[1]

            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))

            detected_lines = cv2.morphologyEx(
                thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
            )

            dilated_lines = cv2.dilate(detected_lines, np.ones((3, 3)), iterations=2)

            result = cv2.inpaint(
                image, dilated_lines, inpaintRadius=5, flags=cv2.INPAINT_NS
            )
            img_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            return img_pil, target
        else:
            return img_pil, target


class AdaptiveThresholdPIL:
    def __init__(self, block_size=15, C=10):
        self.block_size = block_size
        self.C = C

    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image.convert("L"))  # Grayscale
        th = cv2.adaptiveThreshold(
            img_np,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.block_size,
            self.C,
        )
        return Image.fromarray(th)


class CustomCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = (
                t(image, target)
                if isinstance(t, ResizeWithBoxes)
                or isinstance(t, RemoveHorizontalLines)
                else (t(image), target)
            )
        return image, target


def natural_key(filename):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", filename)]


def get_transform(train, max_size=512):
    t = []
    t.append(RemoveHorizontalLines())
    t.append(T.Lambda(remove_shadows_pil))
    t.append(T.Lambda(light_blur))
    t.append(binarize_pil_image(threshold=80))
    t.append(T.Lambda(lambda img: TF.adjust_gamma(img, gamma=2)))
    t.append(T.ToTensor())
    t.append(ForceHorizontal())
    if train:
        pass
    return CustomCompose(t)


exclude_ids = {
    72,
    219,
    229,
    241,
    283,
    344,
}  # these images are excluded because they are not correct


class ExpressionDataset(Dataset):
    def __init__(self, img_dir, ann_dir, dataset, transforms=None):
        self.scales = [512, 640, 768, 896, 1024, 1500] if dataset == "train" else [1024]
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.offset = 300 if dataset == "valid" else 500 if dataset == "test" else 0
        self.img_files = sorted(
            [
                f
                for f in os.listdir(img_dir)
                if f.endswith(".png")
                and int(re.search(r"\d+", f).group()) not in exclude_ids
            ],
            key=natural_key,
        )

    def __len__(self):
        return len(self.img_files) * len(self.scales)

    def __getitem__(self, idx):
        scale_index = idx % len(self.scales)
        idx = int(idx / len(self.scales))
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.ann_dir, img_name.replace(".png", ".json"))

        img = Image.open(img_path).convert("RGB")
        boxes = []
        labels = []
        image_id = torch.tensor(int(img_name.replace(".png", "")))
        area = 0
        iscrowd = []
        if os.path.exists(ann_path):
            with open(ann_path, "r", encoding="utf-8") as f:
                annotation_data = json.load(f)
            for i, ann in enumerate(annotation_data["annotations"]):
                bb = ann["boundingBox"]
                x, y, w, h = bb["x"], bb["y"], bb["width"], bb["height"]
                boxes.append([x, y, x + w, y + h])
                labels = [1] * len(annotation_data["annotations"])

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.long)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        scale = self.scales[scale_index]
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }
        img, target = ResizeWithBoxes(scale).__call__(img, target)

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target
