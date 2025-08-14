import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from .config import RESULTS_DIR, CLUSTERS_DIR, PLOTS_DIR, N_CLUSTERS, SEED
from .io_utils import copy_to

def run_clustering(df: pd.DataFrame, image_paths):
    feature_cols = [c for c in df.columns if c not in ["image", "path"]]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    X = df[feature_cols].values.astype("float32")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=20)
    labels = km.fit_predict(Xs)
    df["cluster"] = labels

    if len(np.unique(labels)) > 1:
        sil = silhouette_score(Xs, labels)
        db  = davies_bouldin_score(Xs, labels)
        ch  = calinski_harabasz_score(Xs, labels)
    else:
        sil = db = ch = float("nan")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "features.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    pca = PCA(n_components=2, random_state=SEED)
    X2 = pca.fit_transform(Xs)
    plt.figure(figsize=(8,6))
    sc = plt.scatter(X2[:,0], X2[:,1], c=labels, cmap="tab20", s=14, alpha=0.9)
    plt.title(f"PCA(2D) | KMeans({N_CLUSTERS}) | Var: {pca.explained_variance_ratio_[0]*100:.1f}% + {pca.explained_variance_ratio_[1]*100:.1f}%")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.colorbar(sc, label="Cluster")
    plot_path = PLOTS_DIR / "pca_scatter.png"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(plot_path, dpi=150); plt.close()

    metrics_txt = RESULTS_DIR / "metrics.txt"
    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write(f"Silhouette: {sil:.4f}\nDavies-Bouldin: {db:.4f}\nCalinski-Harabasz: {ch:.2f}\n")

    uniq = np.unique(labels)
    for c in uniq:
        (CLUSTERS_DIR / f"cluster_{c}").mkdir(parents=True, exist_ok=True)
    for img_path, lab in zip(image_paths, labels):
        dst = CLUSTERS_DIR / f"cluster_{lab}" / Path(img_path).name
        copy_to(Path(img_path), dst)

    with open(RESULTS_DIR / "features_used.json", "w", encoding="utf-8") as f:
        json.dump({"features": feature_cols}, f, ensure_ascii=False, indent=2)

    return {
        "csv": str(out_csv),
        "pca_plot": str(plot_path),
        "metrics": str(metrics_txt),
        "n_features": len(feature_cols),
        "scores": {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}
    }
