import argparse, json
from pathlib import Path
import numpy as np
import cv2, joblib

from sklearn.preprocessing import normalize

from .config import ROOT, RESULTS_DIR
from .io_utils import ensure_dirs
from .features_shape import extract_shape_features
from .features_lbp import extract_lbp_features
from .features_hog import extract_hog_features

from .preprocess import (
    remove_horizontal_lines,
    clean_character_image,
    normalize_char_with_antialiasing
)


def _artifact_dir(art_suffix: str | None):
    if art_suffix:
        d = RESULTS_DIR / f"artifacts_{art_suffix}"
        if d.exists():
            return d
    return RESULTS_DIR  # fallback

def _load_artifacts(art_dir: Path):
    shp = joblib.load(art_dir / "shape_scaler.pkl")
    lbp = joblib.load(art_dir / "lbp_scaler.pkl")
    hog = joblib.load(art_dir / "hog_scaler.pkl")
    pca = joblib.load(art_dir / "hog_pca.pkl")
    km  = joblib.load(art_dir / "kmeans.pkl")
    with open(art_dir / "refined_features_info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    return shp, lbp, hog, pca, km, info


def _prep_char(crop_bgr, mode="mnist"):
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros((28, 28), dtype=np.uint8) + 255

    if mode == "mnist":
        no_line = remove_horizontal_lines(crop_bgr)
        cleaned = clean_character_image(no_line)
        norm28  = normalize_char_with_antialiasing(cleaned)
        if norm28.ndim == 3:
            norm28 = cv2.cvtColor(norm28, cv2.COLOR_BGR2GRAY)
        return norm28

    if crop_bgr.ndim == 3:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_bgr
    return cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

def _to_white_bg_black_text(img28: np.ndarray) -> np.ndarray:

    if img28.ndim == 3:
        img28 = cv2.cvtColor(img28, cv2.COLOR_BGR2GRAY)

    vis = img28.copy()
    if np.mean(vis) < 127:  
        vis = cv2.bitwise_not(vis)

    vis = cv2.normalize(vis, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return vis


def _vectorize_28(gray28, info, shp_scaler, lbp_scaler, hog_scaler, hog_pca):
    feats = {}
    feats.update(extract_shape_features(gray28))
    feats.update(extract_lbp_features(gray28))
    feats.update(extract_hog_features(gray28))

    parts = []
    if info.get("shape_cols"):
        Xs = np.array([[feats[c] for c in info["shape_cols"]]], dtype="float32")
        parts.append(shp_scaler.transform(Xs))
    if info.get("lbp_cols"):
        Xl = np.array([[feats[c] for c in info["lbp_cols"]]], dtype="float32")
        parts.append(lbp_scaler.transform(Xl))
    if info.get("hog_cols"):
        Xh = np.array([[feats[c] for c in info["hog_cols"]]], dtype="float32")
        Xh = hog_scaler.transform(Xh)
        if hog_pca is not None and getattr(hog_pca, "n_components_", 0) > 0:
            Xh = hog_pca.transform(Xh)
        parts.append(Xh)

    X = np.hstack(parts).astype("float32")
    X = normalize(X, norm="l2", axis=1)
    return X

def run_infer(split="test", image_id=None, image_path=None, prep="mnist",
              art_suffix: str | None = None, save_vis=True):
    """
    Inference on a RAW image with label json:
      - Loads artifacts from results/part2 (or results/part2/artifacts_<suffix>)
      - Predicts cluster per bbox
      - Saves annotated RAW and per-char 28x28 VIS (white bg, black text)
    """
    ensure_dirs()
    art_dir = _artifact_dir(art_suffix)
    shp_scaler, lbp_scaler, hog_scaler, hog_pca, kmeans, info = _load_artifacts(art_dir)

    if image_path:
        img_path = Path(image_path)
        lab_path = img_path.parent.parent / "labels" / (img_path.stem + ".json")
    else:
        base = ROOT / f"/dataset/{split}"
        img_path = (base / "images" / f"{image_id}.png").resolve()
        lab_path = (base / "labels" / f"{image_id}.json").resolve()

    if not img_path.exists():  raise FileNotFoundError(f"Image not found: {img_path}")
    if not lab_path.exists():  raise FileNotFoundError(f"Label json not found: {lab_path}")

    image = cv2.imread(str(img_path))
    with open(lab_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    out_dir = RESULTS_DIR / "infer"
    crop_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    boxes = []
    for ann in meta["annotations"]:
        bb = ann["boundingBox"]
        x, y, w, h = int(bb["x"]), int(bb["y"]), int(bb["width"]), int(bb["height"])
        boxes.append((x, y, x + w, y + h))
    boxes.sort(key=lambda b: b[0])

    preds = []
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        gray28 = _prep_char(crop, mode=prep)
        if gray28.ndim == 3:
            gray28 = cv2.cvtColor(gray28, cv2.COLOR_BGR2GRAY)

        X = _vectorize_28(gray28, info, shp_scaler, lbp_scaler, hog_scaler, hog_pca)
        c = int(kmeans.predict(X)[0])
        preds.append({"idx": i, "bbox": [x1, y1, x2, y2], "cluster": c})

        if save_vis:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, f"C{c}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 180, 60), 2, cv2.LINE_AA)

            vis28 = _to_white_bg_black_text(gray28)
            cv2.imwrite(str(crop_dir / f"{img_path.stem}_{i:02d}.png"), vis28)

    out_img  = out_dir / f"{img_path.stem}_pred.png"
    out_json = out_dir / f"{img_path.stem}_pred.json"
    if save_vis:
        cv2.imwrite(str(out_img), image)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"image": str(img_path), "predictions": preds}, f, ensure_ascii=False, indent=2)

    print(f"[infer] saved -> {out_img.name} , {out_json.name}")
    print(f"[infer] crops -> {crop_dir}")
    return str(out_img), str(out_json)


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Infer clusters on RAW image (prep-consistent with TRAIN).")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--image-id", type=int, help="e.g., 0 -> dataset/<split>/images/0.png")
    ap.add_argument("--image-path", type=str, help="direct path to a raw image")
    ap.add_argument("--prep", type=str, default="mnist",
                    choices=["mnist", "resize", "none"],
                    help="character crop preparation; must match the TRAIN setting for reliable results.")
    ap.add_argument("--art-suffix", type=str, default=None,
                    help="if artifacts were saved under results/part2/artifacts_<suffix>, pass this suffix (e.g., mnist).")
    args = ap.parse_args()

    if args.image_path is None and args.image_id is None:
        raise SystemExit("Provide --image-id or --image-path")

    run_infer(split=args.split, image_id=args.image_id, image_path=args.image_path,
              prep=args.prep, art_suffix=args.art_suffix, save_vis=True)
