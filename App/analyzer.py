from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = Path(
    os.environ.get(
        "GOLF_TEMPLATE_PATH",
        str(PROJECT_ROOT / "ml_model" / "models" / "optimal_swing_template.npz"),
    )
)
POSES_DIR = PROJECT_ROOT / "data" / "poses_64_npz"
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"

LEFT_WRIST, RIGHT_WRIST = 15, 16

HEAD_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
SPINE_JOINTS = [11, 12, 23, 24]
HIP_JOINTS = [23, 24, 25, 26]
BALANCE_JOINTS = [27, 28, 29, 30, 31, 32]

JOINT_NAMES = {
    0: "nose", 1: "left eye (inner)", 2: "left eye", 3: "left eye (outer)",
    4: "right eye (inner)", 5: "right eye", 6: "right eye (outer)",
    7: "left ear", 8: "right ear", 9: "mouth (left)", 10: "mouth (right)",
    11: "left shoulder", 12: "right shoulder", 13: "left elbow",
    14: "right elbow", 15: "left wrist", 16: "right wrist",
    17: "left pinky", 18: "right pinky", 19: "left index", 20: "right index",
    21: "left thumb", 22: "right thumb", 23: "left hip", 24: "right hip",
    25: "left knee", 26: "right knee", 27: "left ankle", 28: "right ankle",
    29: "left heel", 30: "right heel", 31: "left foot index",
    32: "right foot index",
}


def ensure_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_template(template_path: Path = TEMPLATE_PATH) -> tuple[np.ndarray, np.ndarray]:
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    tpl = np.load(template_path)
    if "mean" not in tpl or "std" not in tpl:
        raise KeyError(f"Template file must contain 'mean' and 'std': {template_path}")
    return tpl["mean"].astype("float32"), tpl["std"].astype("float32")


def load_pose_file(npz_path: Path) -> np.ndarray:
    if not npz_path.exists():
        raise FileNotFoundError(f"Pose file not found: {npz_path}")
    data = np.load(npz_path)
    if "X" not in data:
        raise KeyError(f"Pose file must contain 'X': {npz_path}")
    X = data["X"].astype("float32")
    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D, got shape {X.shape} in {npz_path}")
    return X


def list_pose_files(poses_dir: Path = POSES_DIR) -> list[Path]:
    if not poses_dir.exists():
        raise FileNotFoundError(f"Pose directory not found: {poses_dir}")
    files = sorted(poses_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz pose files found in: {poses_dir}")
    return files


def score_sequence(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-6,
) -> tuple[float, np.ndarray, np.ndarray]:
    if mean.shape[-1] != X.shape[-1] or std.shape[-1] != X.shape[-1]:
        raise ValueError(
            f"Feature mismatch: X has {X.shape[-1]} features, "
            f"mean has {mean.shape[-1]}, std has {std.shape[-1]}"
        )

    z = (X - mean) / (std + eps)
    z_clipped = np.clip(np.abs(z), 0.0, 3.0)
    per_frame_error = np.mean(z_clipped, axis=1)
    overall_score = float(np.clip(100.0 - 16.7 * per_frame_error.mean(), 0.0, 100.0))
    return overall_score, per_frame_error, z


def error_to_score(err: float) -> float:
    return float(np.clip(100.0 - 16.7 * err, 0.0, 100.0))


def numeric_score_to_grade(score: float) -> str:
    thresholds = [
        (97, "A+"), (93, "A"), (90, "A-"),
        (87, "B+"), (83, "B"), (80, "B-"),
        (77, "C+"), (73, "C"), (70, "C-"),
        (67, "D+"), (63, "D"), (60, "D-"),
    ]
    for threshold, grade in thresholds:
        if score >= threshold:
            return grade
    return "Needs Work"


def feedback_for_phase(phase_name: str, score: float) -> str:
    feedback = {
        "Backswing": [
            (85, "Good shoulder turn and setup."),
            (70, "Backswing is decent but posture can improve."),
            (0,  "Backswing position is inconsistent."),
        ],
        "Downswing": [
            (85, "Good transition into the downswing."),
            (70, "Downswing path needs refinement."),
            (0,  "Downswing appears too steep or unstable."),
        ],
        "Impact": [
            (85, "Strong impact position."),
            (70, "Impact position is acceptable but could be cleaner."),
            (0,  "Impact alignment needs improvement."),
        ],
        "Follow-through": [
            (85, "Balanced finish."),
            (70, "Finish is mostly stable."),
            (0,  "Follow-through balance is weak."),
        ],
    }
    for threshold, text in feedback.get(phase_name, []):
        if score >= threshold:
            return text
    return "No feedback available."


def reshape_pose_sequence(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2 or X.shape[1] != 99:
        raise ValueError(f"Expected X shape (T, 99), got {X.shape}")
    return X.reshape(X.shape[0], 33, 3)


def detect_swing_events(
    X: np.ndarray,
    external_events: dict[str, int] | None = None,
) -> dict[str, int]:
    if external_events is not None:
        return {
            "address": int(external_events.get("address", 0)),
            "top_backswing": int(external_events["top_backswing"]),
            "impact": int(external_events["impact"]),
            "finish": int(external_events["finish"]),
        }

    Xp = reshape_pose_sequence(X)
    T = Xp.shape[0]

    wrist_mid = (Xp[:, LEFT_WRIST] + Xp[:, RIGHT_WRIST]) / 2.0
    wrist_xy = wrist_mid[:, :2]
    address_xy = wrist_xy[0]

    top_search_end = max(2, int(0.6 * T))
    dist_from_address = np.linalg.norm(wrist_xy[:top_search_end] - address_xy, axis=1)
    top_idx = int(np.argmax(dist_from_address))

    impact_search_start = min(top_idx + 1, T - 1)
    return_dist = np.linalg.norm(wrist_xy[impact_search_start:] - address_xy, axis=1)
    impact_idx = impact_search_start + int(np.argmin(return_dist))
    impact_idx = max(top_idx + 1, impact_idx)

    finish_search_start = min(max(impact_idx + 1, int(0.7 * T)), T - 1)
    if finish_search_start >= T - 1:
        finish_idx = T - 1
    else:
        finish_dist = np.linalg.norm(
            wrist_xy[finish_search_start:] - wrist_xy[impact_idx], axis=1
        )
        finish_idx = finish_search_start + int(np.argmax(finish_dist))

    return {
        "address": 0,
        "top_backswing": top_idx,
        "impact": impact_idx,
        "finish": finish_idx,
    }


def build_visual_phase_frames(
    X: np.ndarray,
    external_events: dict[str, int] | None = None,
) -> dict[str, int]:
    events = detect_swing_events(X, external_events=external_events)

    backswing_idx = events["top_backswing"]
    impact_idx = events["impact"]
    finish_idx = events["finish"]
    downswing_idx = max(backswing_idx, (backswing_idx + impact_idx) // 2)

    return {
        "Backswing": int(backswing_idx),
        "Downswing": int(downswing_idx),
        "Impact": int(impact_idx),
        "Follow-through": int(finish_idx),
    }


def build_phase_ranges_from_events(
    n_frames: int,
    visual_frames: dict[str, int],
) -> dict[str, tuple[int, int]]:
    backswing_end = max(1, visual_frames["Backswing"] + 1)
    downswing_end = max(backswing_end + 1, visual_frames["Impact"] + 1)
    impact_end = min(n_frames, max(downswing_end + 1, visual_frames["Impact"] + 3))

    return {
        "Backswing": (0, min(backswing_end, n_frames)),
        "Downswing": (min(backswing_end, n_frames), min(downswing_end, n_frames)),
        "Impact": (min(downswing_end, n_frames), min(impact_end, n_frames)),
        "Follow-through": (min(impact_end, n_frames), n_frames),
    }


def summarize_phases(
    X: np.ndarray,
    per_frame_error: np.ndarray,
    external_events: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    n_frames = len(per_frame_error)
    visual_frames = build_visual_phase_frames(X, external_events=external_events)
    phase_ranges = build_phase_ranges_from_events(n_frames, visual_frames)

    phases: list[dict[str, Any]] = []
    for phase_name in ["Backswing", "Downswing", "Impact", "Follow-through"]:
        start, end = phase_ranges[phase_name]
        if end <= start:
            end = min(start + 1, n_frames)

        phase_errors = per_frame_error[start:end]
        phase_score = error_to_score(float(phase_errors.mean()))

        phases.append({
            "name": phase_name,
            "start": int(start),
            "end": int(end),
            "frame_index": int(visual_frames[phase_name]),
            "score": round(phase_score, 1),
            "feedback": feedback_for_phase(phase_name, phase_score),
        })

    return phases


def _joint_indices_to_feature_indices(joint_indices: list[int]) -> list[int]:
    feats = []
    for j in joint_indices:
        feats.extend([j * 3, j * 3 + 1, j * 3 + 2])
    return feats


def compute_simple_metrics(z: np.ndarray) -> dict[str, float]:
    mean_abs_z = np.mean(np.abs(z), axis=0)

    def score_joints(joint_indices: list[int]) -> float:
        feat_idx = _joint_indices_to_feature_indices(joint_indices)
        value = float(np.mean(mean_abs_z[feat_idx]))
        return round(float(np.clip(100.0 - 16.7 * value, 0.0, 100.0)), 1)

    return {
        "head_stability": score_joints(HEAD_JOINTS),
        "spine_posture": score_joints(SPINE_JOINTS),
        "hip_rotation": score_joints(HIP_JOINTS),
        "balance": score_joints(BALANCE_JOINTS),
    }


def compute_worst_joints(z: np.ndarray, top_n: int = 3) -> list[dict[str, Any]]:
    mean_abs_z = np.mean(np.abs(z), axis=0)
    per_joint = mean_abs_z.reshape(33, 3)
    joint_magnitude = np.linalg.norm(per_joint, axis=1)

    worst_indices = np.argsort(joint_magnitude)[::-1][:top_n]
    return [
        {
            "joint_index": int(idx),
            "joint_name": JOINT_NAMES.get(int(idx), f"joint_{idx}"),
            "deviation": round(float(joint_magnitude[idx]), 3),
        }
        for idx in worst_indices
    ]


def build_summary(
    phases: list[dict[str, Any]],
    worst_frame: int,
    worst_joints: list[dict[str, Any]],
) -> list[str]:
    strongest = max(phases, key=lambda p: p["score"])
    weakest = min(phases, key=lambda p: p["score"])
    top_joint = worst_joints[0]["joint_name"] if worst_joints else "unknown"

    return [
        f"Strongest phase: {strongest['name']} ({strongest['score']:.1f}/100)",
        f"Most improvement needed: {weakest['name']} ({weakest['score']:.1f}/100)",
        f"Highest deviation detected at frame {worst_frame}",
        f"Most deviant joint: {top_joint}",
    ]


def analyze_swing(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    external_events: dict[str, int] | None = None,
) -> dict[str, Any]:
    overall_score, per_frame_error, z = score_sequence(X, mean, std)
    worst_frame = int(np.argmax(per_frame_error))
    phases = summarize_phases(X, per_frame_error, external_events=external_events)
    metrics = compute_simple_metrics(z)
    worst_joints = compute_worst_joints(z, top_n=3)
    summary = build_summary(phases, worst_frame, worst_joints)
    events = detect_swing_events(X, external_events=external_events)
    print("external_events received by analyze_swing:", external_events)

    return {
        "overall_score": round(overall_score, 1),
        "grade": numeric_score_to_grade(overall_score),
        "worst_frame": worst_frame,
        "num_frames": int(X.shape[0]),
        "phases": phases,
        "metrics": metrics,
        "worst_joints": worst_joints,
        "summary": summary,
        "per_frame_error": per_frame_error.tolist(),
        "events": {k: int(v) for k, v in events.items()},
    }


def analyze_pose_file(npz_path: Path) -> dict[str, Any]:
    ensure_dirs()
    mean, std = load_template()
    X = load_pose_file(npz_path)
    result = analyze_swing(X, mean, std)
    result["source_file"] = npz_path.name
    return result


def save_result_json(result: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)