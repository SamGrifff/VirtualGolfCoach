from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

mp_pose = mp.solutions.pose


def extract_pose_array(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    X_list, V_list = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            X = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            V = np.array([p.visibility for p in lm], dtype=np.float32)
        else:
            X = np.full((33, 3), np.nan, dtype=np.float32)
            V = np.zeros((33,), dtype=np.float32)

        X_list.append(X)
        V_list.append(V)

    cap.release()
    pose.close()

    if not X_list:
        raise ValueError("No frames could be read from the video.")

    X = np.stack(X_list, axis=0)
    V = np.stack(V_list, axis=0)
    return X, V, float(fps)


def fill_missing_frames(X):
    X_filled = X.copy()
    T = X.shape[0]

    valid = np.isfinite(X).all(axis=(1, 2))
    if not valid.any():
        raise ValueError("No valid pose frames found in video.")

    valid_idx = np.where(valid)[0]

    for t in range(T):
        if valid[t]:
            continue
        nearest = valid_idx[np.argmin(np.abs(valid_idx - t))]
        X_filled[t] = X_filled[nearest]

    return X_filled


def normalize_pose(X):
    Xn = X.copy()

    hip_center = (Xn[:, LEFT_HIP] + Xn[:, RIGHT_HIP]) / 2.0
    Xn = Xn - hip_center[:, None, :]

    shoulder_dist = np.linalg.norm(
        Xn[:, LEFT_SHOULDER] - Xn[:, RIGHT_SHOULDER],
        axis=1
    ).reshape(-1, 1, 1)

    Xn = Xn / (shoulder_dist + 1e-6)
    return Xn


def resample_time(X, target_len=64):
    T = X.shape[0]
    idx = np.linspace(0, T - 1, target_len).astype(int)
    return X[idx]


def process_video_to_model_input(video_path: Path, target_len=64):
    X, V, fps = extract_pose_array(video_path)

    valid_ratio = np.mean(np.isfinite(X).all(axis=(1, 2)))
    if valid_ratio < 0.5:
        raise ValueError(
            f"Too many missing pose frames in uploaded video "
            f"({valid_ratio * 100:.1f}% valid)."
        )

    X = fill_missing_frames(X)
    Xn = normalize_pose(X)
    Xflat = Xn.reshape(Xn.shape[0], -1)
    X64 = resample_time(Xflat, target_len=target_len)

    return X64, {
        "fps": fps,
        "num_raw_frames": int(X.shape[0]),
        "valid_ratio": float(valid_ratio),
    }