from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "outputs" / "all_pose_scores.csv"


def main():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)

    plt.figure(figsize=(10, 5))
    plt.hist(df["overall_score"], bins=10)
    plt.title("Distribution of Swing Scores")
    plt.xlabel("Overall Score")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)

    out_path = PROJECT_ROOT / "data" / "outputs" / "score_distribution.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()