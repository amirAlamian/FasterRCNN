import math, numpy as np, cv2

def _binarize_otsu_auto(gray):
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thi = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    area_th  = np.count_nonzero(th == 0)
    area_thi = np.count_nonzero(thi == 255)
    return thi if area_thi >= area_th else (th == 0).astype(np.uint8) * 255

def _canny_auto(gray, sigma=0.33):
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    v = np.median(blur)
    low, high = int(max(0,(1-sigma)*v)), int(min(255,(1+sigma)*v))
    return cv2.Canny(blur, low, high)

def extract_shape_features(gray):
    bin_fg = _binarize_otsu_auto(gray)
    cnts, _ = cv2.findContours(bin_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return {k: 0.0 for k in [
            "area","perimeter","bbox_w","bbox_h","aspect_ratio",
            "edge_count","edge_density","circularity",
            "hu1","hu2","hu3","v_mass_ratio","h_mass_ratio"
        ]}
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    peri = float(cv2.arcLength(cnt, True))
    x,y,w,h = cv2.boundingRect(cnt)
    aspect_ratio = (w / h) if h > 0 else 0.0
    circularity = (4*math.pi*area/(peri**2)) if peri>0 else 0.0

    edges = _canny_auto(gray)
    edge_count = int(np.count_nonzero(edges))
    edge_density = float(edge_count) / (area + 1e-6)

    m = cv2.moments(bin_fg)
    hu = cv2.HuMoments(m).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    H, W = bin_fg.shape
    top    = np.count_nonzero(bin_fg[:H//2, :] == 255)
    bottom = np.count_nonzero(bin_fg[H//2:, :] == 255)
    left   = np.count_nonzero(bin_fg[:, :W//2] == 255)
    right  = np.count_nonzero(bin_fg[:, W//2:] == 255)
    v_mass_ratio = top / (bottom + 1e-6)
    h_mass_ratio = left / (right + 1e-6)


    return {
        "area": area, "perimeter": peri, "bbox_w": float(w), "bbox_h": float(h),
        "aspect_ratio": aspect_ratio, "edge_count": edge_count,
        "edge_density": edge_density, "circularity": circularity,
        "hu1": float(hu[0]), "hu2": float(hu[1]), "hu3": float(hu[2]),
        "v_mass_ratio": v_mass_ratio, "h_mass_ratio": h_mass_ratio
    }
