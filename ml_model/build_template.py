import numpy as np
from pathlib import Path

IN_DIR = Path("data/poses_64_npz")
OUT_PATH = Path("ml_model/models/optimal_swing_template.npz")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_all_X():
    paths = sorted(IN_DIR.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz found in {IN_DIR}")
    Xs = []
    names = []
    for p in paths:
        d = np.load(p)
        X = d["X"].astype(np.float32)  # (64,99)
        if X.shape != (64, 99):
            print("Skipping (wrong shape):", p.name, X.shape)
            continue
        Xs.append(X)
        names.append(p.name)
    return np.stack(Xs, axis=0), names  # (N,64,99)

def compute_outlier_mask(Xs, z_thresh=2.5):
    # distance of each sample to the mean trajectory
    mean = Xs.mean(axis=0, keepdims=True)            # (1,64,99)
    dists = np.linalg.norm((Xs - mean).reshape(Xs.shape[0], -1), axis=1)  # (N,)
    z = (dists - dists.mean()) / (dists.std() + 1e-6)
    keep = z < z_thresh
    return keep, dists, z

def main():
    Xs, names = load_all_X()  # (N,64,99)
    print(f"Loaded {Xs.shape[0]} swings")

    keep, dists, z = compute_outlier_mask(Xs, z_thresh=2.5)
    kept = Xs[keep]
    kept_names = [n for n, k in zip(names, keep) if k]

    print(f"Keeping {kept.shape[0]} swings, removing {Xs.shape[0]-kept.shape[0]} outliers")
    if kept.shape[0] < 5:
        print("Warning: too few swings kept; consider raising z_thresh or checking data quality.")

    template_mean = kept.mean(axis=0)  # (64,99)
    template_std  = kept.std(axis=0)   # (64,99)

    # Optional: pick a medoid (most representative swing)
    flat = kept.reshape(kept.shape[0], -1)
    dist_matrix = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=2)
    medoid_idx = np.argmin(dist_matrix.sum(axis=1))
    template_medoid = kept[medoid_idx]
    medoid_name = kept_names[medoid_idx] if kept_names else "unknown"

    np.savez_compressed(
        OUT_PATH,
        mean=template_mean,
        std=template_std,
        medoid=template_medoid,
        medoid_name=medoid_name,
        kept_names=np.array(kept_names),
    )
    print(f"Saved template to {OUT_PATH}")
    print("Medoid swing:", medoid_name)

if __name__ == "__main__":
    main()