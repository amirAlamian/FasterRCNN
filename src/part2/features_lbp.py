import numpy as np
from skimage.feature import local_binary_pattern


def extract_lbp_features(gray, radius=1, n_points=8):
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-6)
    return {f"lbp_{i}": float(v) for i, v in enumerate(hist)}
