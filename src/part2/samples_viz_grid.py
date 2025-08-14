import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
import joblib


try:
    from .config import RESULTS_DIR, PLOTS_DIR
except Exception:
    here = Path(__file__).resolve()
    sys.path.append(str(here.parents[1]))
    from part2.config import RESULTS_DIR, PLOTS_DIR

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

def _imread_rgb(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def _tile_row(axs_row, img_paths, title_text=None):
    for j, (ax, p) in enumerate(zip(axs_row, img_paths), start=1):
        if p is None:
            ax.axis("off"); continue
        img = _imread_rgb(Path(p))
        if img is None:
            ax.axis("off"); continue
        ax.imshow(img); ax.axis("off")
        if title_text and j == 1:
            ax.set_title(title_text, fontsize=11)

def make_cluster_grids(k_top=6, k_far=6, figsize=(16, 6)):
    res_dir = RESULTS_DIR
    plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_refined = res_dir / "features_refined.csv"
    csv_basic   = res_dir / "features.csv"
    if csv_refined.exists():
        df = pd.read_csv(csv_refined)
    elif csv_basic.exists():
        df = pd.read_csv(csv_basic)
    else:
        raise FileNotFoundError("Missing features_refined.csv / features.csv. Run Part 2 first.")

    if "cluster" not in df.columns:
        raise RuntimeError("Missing 'cluster' column. Run clustering first.")

    shp_sc, lbp_sc, hog_sc, hog_pca, info = _load_artifacts(res_dir)
    for fam in ["shape_cols", "lbp_cols", "hog_cols"]:
        if info.get(fam):
            info[fam] = [c for c in info[fam] if c in df.columns]

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

        close_paths = [paths[i] for i in close_idx]
        far_paths   = [paths[i] for i in far_idx]

        cols = max(len(close_paths), len(far_paths))
        fig, axs = plt.subplots(2, cols, figsize=figsize)
        if cols == 1:
            axs = np.array([[axs[0]], [axs[1]]])

        close_paths += [None] * (cols - len(close_paths))
        far_paths   += [None] * (cols - len(far_paths))

        _tile_row(axs[0], close_paths, title_text=f"Cluster {int(cid)} — Nearest to centroid")
        _tile_row(axs[1], far_paths,   title_text="Farthest (Outliers)")

        plt.tight_layout()
        out_path = plots_dir / f"cluster_{int(cid)}_nearest_farthest.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[saved] {out_path}")

if __name__ == "__main__":
    make_cluster_grids(k_top=6, k_far=6, figsize=(16, 6))
