from __future__ import annotations

import sys
from pathlib import Path

GOLFDB_ROOT = Path(__file__).resolve().parent.parent / "swing_event_model"
sys.path.insert(0, str(GOLFDB_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import EventDetector
from eval import ToTensor, Normalize


# -------------------------------------------------------------------
# Event mapping (GolfDB → readable names)
# -------------------------------------------------------------------
GOLFDB_EVENT_NAMES = {
    0: "address",
    1: "toe_up",
    2: "mid_backswing",
    3: "top_backswing",
    4: "mid_downswing",
    5: "impact",
    6: "mid_follow_through",
    7: "finish",
}


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "swing_event_model" / "models" / "swingnet_1800.pth.tar"


# -------------------------------------------------------------------
# Dataset for video loading
# -------------------------------------------------------------------
class SampleVideo(Dataset):
    def __init__(self, path: str | Path, input_size: int = 160, transform=None):
        self.path = str(Path(path).resolve())
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.path)

        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            cap.release()
            raise ValueError(f"Could not read first frame from video: {self.path}")

        frame_h, frame_w = first_frame.shape[:2]

        ratio = self.input_size / max(frame_h, frame_w)
        new_h = int(frame_h * ratio)
        new_w = int(frame_w * ratio)

        delta_w = self.input_size - new_w
        delta_h = self.input_size - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        images = []

        def process_frame(img):
            resized = cv2.resize(img, (new_w, new_h))
            bordered = cv2.copyMakeBorder(
                resized,
                top, bottom, left, right,
                cv2.BORDER_CONSTANT,
                value=[0.406 * 255, 0.456 * 255, 0.485 * 255],
            )
            return cv2.cvtColor(bordered, cv2.COLOR_BGR2RGB)

        images.append(process_frame(first_frame))

        while True:
            ret, img = cap.read()
            if not ret or img is None:
                break
            images.append(process_frame(img))

        cap.release()

        if not images:
            raise ValueError(f"No readable frames found in video: {self.path}")

        labels = np.zeros(len(images), dtype=np.float32)
        sample = {"images": np.asarray(images), "labels": labels}

        if self.transform:
            sample = self.transform(sample)

        return sample


# -------------------------------------------------------------------
# Main detector
# -------------------------------------------------------------------
class GolfDBEventDetector:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        seq_length: int = 64,
        device: str | None = None,
    ):
        self.model_path = Path(model_path).resolve()
        self.seq_length = seq_length

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = EventDetector(
            pretrain=True,
            width_mult=1.0,
            lstm_layers=1,
            lstm_hidden=256,
            bidirectional=True,
            dropout=False,
        )

        save_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(save_dict["model_state_dict"])

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],
                      [0.229, 0.224, 0.225]),
        ])

    # ---------------------------------------------------------------
    # Raw predictions
    # ---------------------------------------------------------------
    @torch.no_grad()
    def predict_raw_events(self, video_path: str | Path) -> dict[str, int]:
        ds = SampleVideo(video_path, transform=self.transform)
        dl = DataLoader(ds, batch_size=1, shuffle=False)

        probs = None

        for sample in dl:
            images = sample["images"]

            batch = 0
            while batch * self.seq_length < images.shape[1]:
                start = batch * self.seq_length
                end = min((batch + 1) * self.seq_length, images.shape[1])

                image_batch = images[:, start:end, :, :, :]

                logits = self.model(image_batch.to(self.device))
                batch_probs = F.softmax(logits, dim=1).cpu().numpy()

                probs = batch_probs if probs is None else np.append(probs, batch_probs, axis=0)
                batch += 1

        if probs is None:
            raise RuntimeError("No predictions made.")

        event_frames = np.argmax(probs, axis=0)[:-1]

        return {
            GOLFDB_EVENT_NAMES[i]: int(frame_idx)
            for i, frame_idx in enumerate(event_frames)
        }

    # ---------------------------------------------------------------
    # FIXED event ordering (CRITICAL)
    # ---------------------------------------------------------------
    def predict_app_events(self, video_path: str | Path) -> dict[str, int]:
        raw = self.predict_raw_events(video_path)

        addr = raw["address"]
        top  = raw["top_backswing"]
        imp  = raw["impact"]
        fin  = raw["finish"]

        # enforce timeline: address < top < impact < finish
        if top < addr:
            top = addr

        if imp < top:
            imp = top + 1

        if fin < imp:
            fin = imp + 1

        return {
            "address": addr,
            "top_backswing": top,
            "impact": imp,
            "finish": fin,
        }


# -------------------------------------------------------------------
# Frame mapping helper
# -------------------------------------------------------------------
def raw_to_analysis_idx(raw_idx: int, num_raw_frames: int, num_analysis_frames: int = 64) -> int:
    if num_raw_frames <= 1:
        return 0
    return int((raw_idx / (num_raw_frames - 1)) * (num_analysis_frames - 1))


def convert_app_events_to_analysis_indices(
    app_events: dict[str, int],
    num_raw_frames: int,
    num_analysis_frames: int = 64,
) -> dict[str, int]:
    return {
        "address": raw_to_analysis_idx(app_events["address"], num_raw_frames, num_analysis_frames),
        "top_backswing": raw_to_analysis_idx(app_events["top_backswing"], num_raw_frames, num_analysis_frames),
        "impact": raw_to_analysis_idx(app_events["impact"], num_raw_frames, num_analysis_frames),
        "finish": raw_to_analysis_idx(app_events["finish"], num_raw_frames, num_analysis_frames),
    }