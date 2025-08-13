# src/part2/samples_viz.py
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

# --- سازگاری با اجرای ماژولی و مستقیم ---
try:
    from .config import RESULTS_DIR
except Exception:
    # اگر مستقیم اجرا شد، مسیر src را به sys.path اضافه کن
    here = Path(__file__).resolve()
    sys.path.append(str(here.parents[1]))   # .../src را اضافه می‌کند
    from part2.config import RESULTS_DIR

from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
import joblib

def _load_artifacts(res_dir: Path):
    shp = joblib.load(res_dir / "shape_scaler.pkl")
    lbp = joblib.load(res_dir / "lbp_scaler.pkl")
    hog = joblib.load(res_dir / "hog_scaler.pkl")
    pca = joblib.load(res_dir / "hog_pca.pkl")
    with open(res_dir / "refined_features_info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    return shp, lbp, hog, pca, info

def _vectorize_df(df: pd.DataFrame, info, shp_scaler, lbp_scaler, hog_scaler, hog_pca):
    parts = []
    if info.get("shape_cols"):
        Xs = df[info["shape_cols"]].values.astype("float32")
        parts.append(shp_scaler.transform(Xs))
    if info.get("lbp_cols"):
        Xl = df[info["lbp_cols"]].values.astype("float32")
        parts.append(lbp_scaler.transform(Xl))
    if info.get("hog_cols"):
        Xh = df[info["hog_cols"]].values.astype("float32")
        Xh = hog_scaler.transform(Xh)
        if hog_pca is not None and getattr(hog_pca, "n_components_", 0) > 0:
            Xh = hog_pca.transform(Xh)
        parts.append(Xh)
    X = np.hstack(parts).astype("float32")
    X = normalize(X, norm="l2", axis=1)
    return X

def main(k_top=5, k_far=5):
    res_dir = RESULTS_DIR
    out_dir = res_dir / "samples_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV نهاییِ پارت ۲ (refined)؛ اگر نبود، به features.csv برگرد
    csv_refined = res_dir / "features_refined.csv"
    csv_basic   = res_dir / "features.csv"
    if csv_refined.exists():
        df = pd.read_csv(csv_refined)
    elif csv_basic.exists():
        df = pd.read_csv(csv_basic)
    else:
        raise FileNotFoundError("هیچ‌کدام از features_refined.csv / features.csv پیدا نشد. اول پارت ۲ را اجرا کنید.")

    # حذف ستون‌های غیر ویژگی
    drop_cols = {"image", "path"}
    feat_cols = [c for c in df.columns if c not in drop_cols and not c == "cluster"]

    if "cluster" not in df.columns:
        raise RuntimeError("ستون 'cluster' در CSV وجود ندارد؛ ابتدا کلاسترینگ را اجرا کنید.")

    # آرتیفکت‌ها را برای بازسازی دقیق فضای ویژگی‌ها لود می‌کنیم
    shp_sc, lbp_sc, hog_sc, hog_pca, info = _load_artifacts(res_dir)

    # برخی ستون‌های لازم ممکن است در df نباشند (به‌علت افتادن نمونه‌ها). فیلتر ایمن:
    for fam_key in ["shape_cols", "lbp_cols", "hog_cols"]:
        if info.get(fam_key):
            info[fam_key] = [c for c in info[fam_key] if c in df.columns]

    # آماده‌سازی ماتریس ویژگی در همان فضای آموزشی (اسکیل+PCA+L2)
    X = _vectorize_df(df, info, shp_sc, lbp_sc, hog_sc, hog_pca)
    labels = df["cluster"].values
    paths = df["path"].values

    for cid in sorted(np.unique(labels)):
        idxs = np.where(labels == cid)[0]
        if idxs.size == 0:
            continue
        Xc = X[idxs]
        centroid = Xc.mean(axis=0, keepdims=True)
        dists = pairwise_distances(Xc, centroid, metric="euclidean").ravel()
        order = np.argsort(dists)

        close_idx = idxs[order[:min(k_top, len(order))]]
        far_idx   = idxs[order[-min(k_far, len(order)):]]

        cdir = out_dir / f"cluster_{int(cid)}"
        cdir.mkdir(parents=True, exist_ok=True)

        # ذخیره نزدیک‌ترین‌ها
        for i, ridx in enumerate(close_idx, 1):
            img_path = Path(paths[ridx])
            img = cv2.imread(str(img_path))
            if img is not None:
                cv2.imwrite(str(cdir / f"close_{i:02d}.png"), img)

        # ذخیره دورافتاده‌ها
        for i, ridx in enumerate(far_idx, 1):
            img_path = Path(paths[ridx])
            img = cv2.imread(str(img_path))
            if img is not None:
                cv2.imwrite(str(cdir / f"far_{i:02d}.png"), img)

    print(f"[OK] Samples in: {out_dir}")

if __name__ == "__main__":
    main()
