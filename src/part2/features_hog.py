from skimage.feature import hog


def extract_hog_features(gray):
    feats = hog(gray, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=False, feature_vector=True)
    return {f"hog_{i}": float(v) for i, v in enumerate(feats)}
