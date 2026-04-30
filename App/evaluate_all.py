from pathlib import Path
import pandas as pd

from analyzer import load_template, load_pose_file, analyze_swing

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSES_DIR = PROJECT_ROOT / "data" / "poses_64_npz"
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"


def main():
    mean, std = load_template()

    files = sorted(POSES_DIR.glob("*.npz"))
    if not files:
        print(f"No files found in {POSES_DIR}")
        return

    rows = []

    for f in files:
        try:
            X = load_pose_file(f)
            result = analyze_swing(X, mean, std)

            rows.append({
                "file": f.name,
                "overall_score": result["overall_score"],
                "grade": result["grade"],
                "worst_frame": result["worst_frame"],
                "backswing_score": result["phases"][0]["score"],
                "downswing_score": result["phases"][1]["score"],
                "impact_score": result["phases"][2]["score"],
                "followthrough_score": result["phases"][3]["score"],
            })

            print(f"{f.name}: {result['overall_score']}/100 ({result['grade']})")

        except Exception as e:
            print(f"Error processing {f.name}: {e}")

    df = pd.DataFrame(rows)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUTS_DIR / "all_pose_scores.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== Summary ===")
    print(df["overall_score"].describe())

    print(f"\nSaved results to: {out_csv}")


if __name__ == "__main__":
    main()