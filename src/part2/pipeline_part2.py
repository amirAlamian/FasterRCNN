from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

from .config import (
    set_seed, DATA_DIR, RESULTS_DIR, PLOTS_DIR, CLUSTERS_DIR,
    SEED, N_CLUSTERS
)

from .io_utils import list_images, read_gray, ensure_dirs, copy_to
from .features_shape import extract_shape_features
from .features_lbp import extract_lbp_features
from .features_hog import extract_hog_features



def _collect_features_from_dir(data_dir: Path, feature_sets=("shape", "lbp", "hog")) -> pd.DataFrame:
    """Read 28x28 gray images from data_dir and build a features DataFrame."""
    image_paths = list_images(data_dir)
    rows = []
    for p in image_paths:
        gray = read_gray(p)
        feats = {}
        if "shape" in feature_sets:
            feats.update(extract_shape_features(gray))
        if "lbp" in feature_sets:
            feats.update(extract_lbp_features(gray))
        if "hog" in feature_sets:
            feats.update(extract_hog_features(gray))
        feats["image"] = p.name
        feats["path"] = str(p)
        rows.append(feats)
    df = pd.DataFrame(rows)
    return df


def _separate_families(df: pd.DataFrame):
    """Split columns by family; drop NaNs/Infs."""
    feat_cols = [c for c in df.columns if c not in ("image", "path")]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols).reset_index(drop=True)

    shape_cols = [c for c in feat_cols if c in {
        "area", "perimeter", "bbox_w", "bbox_h", "aspect_ratio", "edge_count",
        "edge_density", "circularity", "hu1", "hu2", "hu3", "v_mass_ratio", "h_mass_ratio"
    }]
    lbp_cols = [c for c in feat_cols if c.startswith("lbp_")]
    hog_cols = [c for c in feat_cols if c.startswith("hog_")]
    return df, shape_cols, lbp_cols, hog_cols



def _fit_transforms_on_train(df: pd.DataFrame,
                             shape_cols, lbp_cols, hog_cols,
                             hog_pca_components: int = 48):
    """
    Fit scalers and HOG PCA on TRAIN and return:
    (shape_scaler, lbp_scaler, hog_scaler, hog_pca, X_concat)
    """
    parts = []
    shape_scaler = lbp_scaler = hog_scaler = hog_pca = None

    if shape_cols:
        shape_scaler = StandardScaler().fit(df[shape_cols].values.astype("float32"))
        parts.append(shape_scaler.transform(df[shape_cols].values.astype("float32")))

    if lbp_cols:
        lbp_scaler = StandardScaler().fit(df[lbp_cols].values.astype("float32"))
        parts.append(lbp_scaler.transform(df[lbp_cols].values.astype("float32")))

    if hog_cols:
        hog_scaler = StandardScaler().fit(df[hog_cols].values.astype("float32"))
        Xh_std = hog_scaler.transform(df[hog_cols].values.astype("float32"))
        n_comp = min(hog_pca_components, Xh_std.shape[1], max(2, Xh_std.shape[0] - 1))
        hog_pca = PCA(n_components=n_comp, random_state=SEED).fit(Xh_std)
        parts.append(hog_pca.transform(Xh_std))

    X = np.hstack(parts).astype("float32")
    X = normalize(X, norm="l2", axis=1)
    return shape_scaler, lbp_scaler, hog_scaler, hog_pca, X



def _choose_k_by_elbow_silhouette(X, k_min=12, k_max=24, save_plots=True):
    inertias, sils, ks = [], [], list(range(k_min, k_max + 1))
    best_k, best_s = None, -1.0

    for k in ks:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=20)
        labels = km.fit_predict(X)
        inertias.append(float(km.inertia_))
        sil = float("nan")
        if len(np.unique(labels)) > 1:
            sil = float(silhouette_score(X, labels))
            if sil > best_s:
                best_s, best_k = sil, k
        sils.append(sil)

    diffs = np.diff(inertias)
    curv = np.diff(diffs) * -1.0
    k_elbow = ks[2 + int(np.argmax(curv))] if len(curv) else ks[0]
    if best_k is None:
        best_k = k_elbow

    if save_plots:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        # Elbow
        plt.figure(figsize=(6, 4))
        plt.plot(ks, inertias, marker="o")
        plt.xlabel("K"); plt.ylabel("Inertia"); plt.title("Elbow curve")
        plt.tight_layout(); plt.savefig(PLOTS_DIR / "elbow.png", dpi=150); plt.close()
        # Silhouette
        plt.figure(figsize=(6, 4))
        plt.plot(ks, sils, marker="o")
        plt.xlabel("K"); plt.ylabel("Silhouette"); plt.title("Silhouette vs K")
        plt.tight_layout(); plt.savefig(PLOTS_DIR / "silhouette_vs_k.png", dpi=150); plt.close()

        with open(RESULTS_DIR / "k_selection.txt", "w", encoding="utf-8") as f:
            f.write(f"Ks: {ks}\nInertia: {inertias}\nSilhouette: {sils}\n")
            f.write(f"Chosen K (silhouette): {best_k}\nChosen K (elbow): {k_elbow}\nBest K: {best_k}\n")

    return best_k



def _cluster_fit_and_export(X, df, image_paths, K):
    km = KMeans(n_clusters=K, random_state=SEED, n_init=20).fit(X)
    labels = km.labels_

    if len(np.unique(labels)) > 1:
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
    else:
        sil = db = ch = float("nan")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "features_refined.csv"
    df_out = df.copy()
    df_out["cluster"] = labels
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    p2 = PCA(n_components=2, random_state=SEED).fit_transform(X)
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(p2[:, 0], p2[:, 1], c=labels, cmap="tab20", s=14, alpha=0.9)
    plt.title(f"PCA(2D) | KMeans({K})")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.colorbar(sc, label="Cluster")
    plot_path = PLOTS_DIR / "pca_scatter_refined.png"
    plt.tight_layout(); plt.savefig(plot_path, dpi=150); plt.close()

    uniq = np.unique(labels)
    for c in uniq:
        (CLUSTERS_DIR / f"cluster_{c}").mkdir(parents=True, exist_ok=True)
    for img_path, lab in zip(image_paths, labels):
        dst = CLUSTERS_DIR / f"cluster_{lab}" / Path(img_path).name
        copy_to(Path(img_path), dst)

    with open(RESULTS_DIR / "metrics_refined.txt", "w", encoding="utf-8") as f:
        f.write(f"Silhouette: {sil:.4f}\nDavies-Bouldin: {db:.4f}\nCalinski-Harabasz: {ch:.2f}\nK: {K}\n")

    return {
        "csv": str(out_csv),
        "pca_plot": str(plot_path),
        "metrics": str(RESULTS_DIR / "metrics_refined.txt"),
        "scores": {
            "silhouette": float(sil),
            "davies_bouldin": float(db),
            "calinski_harabasz": float(ch)
        },
        "K": int(K),
        "labels": labels,
        "kmeans": km
    }



def _save_artifacts(shape_scaler, lbp_scaler, hog_scaler, hog_pca,
                    shape_cols, lbp_cols, hog_cols, kmeans):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(shape_scaler, RESULTS_DIR / "shape_scaler.pkl")
    joblib.dump(lbp_scaler,   RESULTS_DIR / "lbp_scaler.pkl")
    joblib.dump(hog_scaler,   RESULTS_DIR / "hog_scaler.pkl")
    joblib.dump(hog_pca,      RESULTS_DIR / "hog_pca.pkl")
    joblib.dump(kmeans,       RESULTS_DIR / "kmeans.pkl")

    with open(RESULTS_DIR / "refined_features_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "shape_cols": shape_cols,
            "lbp_cols": lbp_cols,
            "hog_cols": hog_cols,
            "hog_pca_components": (
                getattr(hog_pca, "n_components_", 0) if hog_pca is not None else 0
            )
        }, f, ensure_ascii=False, indent=2)

    print(f"[save] Artifacts saved into: {RESULTS_DIR}")



def run(feature_sets=("shape", "lbp", "hog"),
        hog_pca_components=48,
        k_min=12, k_max=24):
    """
    Train-only pipeline for Part 2 with:
    - per-family standardization
    - PCA on HOG
    - global L2 normalization
    - K selection by Silhouette/Elbow (or fixed if N_CLUSTERS is set)
    - artifact saving for inference on RAW images
    """
    set_seed()
    ensure_dirs()

    train_dir = DATA_DIR
    if not train_dir.exists():
        raise FileNotFoundError(f"normalized_images (train) not found at: {train_dir}")


    df = _collect_features_from_dir(train_dir, feature_sets=feature_sets)
    df, shape_cols, lbp_cols, hog_cols = _separate_families(df)


    shape_scaler, lbp_scaler, hog_scaler, hog_pca, X = _fit_transforms_on_train(
        df, shape_cols, lbp_cols, hog_cols, hog_pca_components=hog_pca_components
    )

    if isinstance(N_CLUSTERS, int) and N_CLUSTERS > 1:
        best_k = int(N_CLUSTERS)  # قفل روی K مشخص در config.py
    else:
        best_k = _choose_k_by_elbow_silhouette(X, k_min=k_min, k_max=k_max, save_plots=True)

    report = _cluster_fit_and_export(X, df, df["path"].tolist(), best_k)

    _save_artifacts(shape_scaler, lbp_scaler, hog_scaler, hog_pca,
                    shape_cols, lbp_cols, hog_cols, report["kmeans"])

    return {
        "csv": report["csv"],
        "pca_plot": report["pca_plot"],
        "metrics": report["metrics"],
        "scores": report["scores"],
        "K": report["K"]
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Part-2 pipeline (TRAIN only, refined)")
    parser.add_argument("--features", type=str, default="shape,lbp,hog",
                        help="comma-separated subset of: shape,lbp,hog")
    parser.add_argument("--hog-pca", type=int, default=48, help="HOG PCA components")
    parser.add_argument("--k-min", type=int, default=12)
    parser.add_argument("--k-max", type=int, default=24)
    args = parser.parse_args()

    feats = tuple(s.strip() for s in args.features.split(",") if s.strip())
    summary = run(
        feature_sets=feats,
        hog_pca_components=args.hog_pca,
        k_min=args.k_min, k_max=args.k_max
    )
    print("== Summary ==")
    for k, v in summary.items():
        print(k, ":", v)
