from pathlib import Path
import random
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "dataset" / "train" / "normalized_images"
RESULTS_DIR = ROOT / "results" / "part2"
CLUSTERS_DIR = RESULTS_DIR / "clusters"
PLOTS_DIR = RESULTS_DIR / "plots"

N_CLUSTERS = 0
SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)

