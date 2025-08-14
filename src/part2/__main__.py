import argparse
import sys
import subprocess
from pathlib import Path

import part2.config as cfg
from .config import ROOT
from .pipeline_part2 import run as run_part2
from .infer_on_raw import run_infer

def run_preprocess(python_bin="python"):
    script = Path(__file__).resolve().parent / "preprocess.py"
    if not script.exists():
        raise FileNotFoundError(f"preprocess.py not found at: {script}")
    print(" running preprocessing...")
    r = subprocess.run([python_bin, str(script)], stdout=sys.stdout, stderr=sys.stderr)
    if r.returncode != 0:
        raise RuntimeError("preprocess.py failed.")
    print(" preprocessing completed.")

def _list_ids_in_split(split_dir: Path):
    labels_dir = split_dir / "labels"
    if not labels_dir.exists():
        return []
    ids = []
    for p in sorted(labels_dir.iterdir()):
        if p.suffix.lower() == ".json":
            try:
                ids.append(int(p.stem))
            except ValueError:
                pass
    return ids


def main():
    parser = argparse.ArgumentParser(description="Part-2 pipeline launcher (train + optional raw inference)")

    parser.add_argument("--python-bin", type=str, default="python", help="python executable to run preprocess.py")
    parser.add_argument("--skip-preprocess", action="store_true", help="skip running preprocess.py")

    parser.add_argument("--features", type=str, default="shape,lbp,hog", help="comma-separated: shape,lbp,hog")
    parser.add_argument("--hog-pca", type=int, default=48, help="HOG PCA components")
    parser.add_argument("--k-min", type=int, default=12, help="min K for KMeans selection")
    parser.add_argument("--k-max", type=int, default=24, help="max K for KMeans selection")

    parser.add_argument("--infer", action="store_true", help="run inference on RAW images after training")
    parser.add_argument("--infer-split", type=str, default="test", choices=["train", "val", "test"],
                        help="which split to infer on (RAW images)")
    parser.add_argument("--infer-ids", type=str, default=None,
                        help="comma-separated image IDs for inference (e.g., 0,12,23). If omitted and --infer-all not set, does nothing.")
    parser.add_argument("--infer-all", action="store_true", help="infer on ALL ids in the chosen split")
    parser.add_argument("--prep", type=str, default="mnist", choices=["mnist", "resize", "none"],
                        help="character crop preparation mode (must match TRAIN setting for reliable results)")
    parser.add_argument("--art-suffix", type=str, default=None,
                        help="artifact subfolder suffix (if you used results/part2/artifacts_<suffix>)")

    args = parser.parse_args()

    if not args.skip_preprocess:
        run_preprocess(python_bin=args.python_bin)
    else:
        print("  skipping preprocessing.")

    normalized_dir = (ROOT / "dataset" / "train" / "normalized_images").resolve()
    if not normalized_dir.exists():
        raise FileNotFoundError(f"normalized_images not found at: {normalized_dir}")

    cfg.DATA_DIR = normalized_dir

    feature_sets = tuple(s.strip() for s in args.features.split(",") if s.strip())
    print(f" clustering from: {cfg.DATA_DIR}")
    print(f" features: {feature_sets}")
    print(f" hog_pca={args.hog_pca} | K-range=[{args.k_min}, {args.k_max}]")
    report = run_part2(
        feature_sets=feature_sets,
        hog_pca_components=args.hog_pca,
        k_min=args.k_min, k_max=args.k_max
    )

    print("\n== Train Summary ==")
    for k, v in report.items():
        print(f"{k}: {v}")
    print("\n results in:", cfg.RESULTS_DIR)
    print(" PCA plot:", Path(report["pca_plot"]).name)
    print(" metrics:", Path(report["metrics"]).name)

    if args.infer:
        split = args.infer_split
        split_dir = (ROOT / f"/dataset/{split}").resolve()
        if not split_dir.exists():
            raise FileNotFoundError(f"split dir not found: {split_dir}")

        ids = []
        if args.infer_all:
            ids = _list_ids_in_split(split_dir)
        elif args.infer_ids:
            try:
                ids = [int(x.strip()) for x in args.infer_ids.split(",") if x.strip() != ""]
            except ValueError:
                raise SystemExit("--infer-ids must be comma-separated integers, e.g., 0,12,23")

        if not ids:
            print("\n[infer] No IDs provided. Use --infer-all or --infer-ids  (e.g., --infer-ids 0,12,23).")
        else:
            print(f"\n[infer] split={split} | ids={len(ids)} | prep={args.prep} | art_suffix={args.art_suffix}")
            for i, img_id in enumerate(ids, 1):
                try:
                    run_infer(split=split, image_id=img_id, image_path=None,
                              prep=args.prep, art_suffix=args.art_suffix, save_vis=True)
                except Exception as e:
                    print(f"[infer] ID {img_id} failed: {e}")

    print("\n Done.")

if __name__ == "__main__":
    main()
