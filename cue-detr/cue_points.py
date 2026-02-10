from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import torch
from matplotlib import cm
from PIL import Image
from scipy.signal import find_peaks
from transformers import DetrForObjectDetection, DetrImageProcessor

# Constants
OVERLAP = 0.75
W_BBOX = 21
W_WIN = 355
PADDING = 266

def _normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if max_val - min_val <= 1e-12:
        return [1.0 for _ in values]
    return ((arr - min_val) / (max_val - min_val)).tolist()


def predict_cue_points_for_dir(
    tracks_dir: str | Path,
    checkpoint: str = "disco-eth/cue-detr",
    sensitivity: float = 0.9,
    radius: int = 16,
    *,
    print_points: bool = False,
    write_output: bool = True,
) -> dict[str, list[float]]:
    """
    Predict cue points for all mp3 files in a directory.
    Returns mapping: { "<filename>.mp3": [cue_seconds, ...] }.
    """
    tracks_path = Path(tracks_dir).expanduser().resolve()
    if not tracks_path.exists():
        raise FileNotFoundError(f"Track directory not found: {tracks_path}")
    if not tracks_path.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {tracks_path}")

    tracklist = sorted(
        file.name for file in tracks_path.iterdir() if file.is_file() and file.suffix.lower() == ".mp3"
    )
    cue_points: dict[str, list[float]] = {track: [] for track in tracklist}

    # Load model once for all tracks.
    image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DetrForObjectDetection.from_pretrained(checkpoint)
    model.to(device)

    for track in tracklist:
        y, sr = librosa.load(str(tracks_path / track))  # standard sr of 22050
        M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048)
        M_db = librosa.power_to_db(M, ref=np.max)

        # Convert spectrogram to RGB image
        arr = M_db[::-1]
        sm = cm.ScalarMappable(cmap="viridis")
        sm.set_clim(arr.min(), arr.max())
        rgba = sm.to_rgba(arr, bytes=True)
        rgb_shape = (rgba.shape[1], rgba.shape[0])
        rgba = np.require(rgba, requirements="C")
        im = Image.frombuffer("RGBA", rgb_shape, rgba, "raw", "RGBA", 0, 1)
        image = np.array(im)[:, :, :3]

        image_w = image.shape[1] + PADDING
        n_windows = int(np.floor(image_w / (W_WIN * (1 - OVERLAP))))

        images = []
        borders = []

        # Build image batch using sliding windows.
        for i in range(n_windows):
            left = int(np.floor(i * W_WIN * (1 - OVERLAP))) - PADDING
            right = left + W_WIN
            borders.append(left)

            if left < 0:
                segment = image[:, :right]
                pad = -left
                segment = np.pad(segment, ((0, 0), (pad, 0), (0, 0)), mode="linear_ramp")
            elif right > image.shape[1]:
                segment = image[:, left:]
                pad = right - left - segment.shape[1]
                segment = np.pad(segment, ((0, 0), (0, pad), (0, 0)), mode="linear_ramp")
            else:
                segment = image[:, left:right]

            images.append(segment)

        if not images:
            cue_points[track] = []
            continue

        encoding = image_processor.preprocess(images, do_resize=False, return_tensors="pt")
        pixel_values = encoding["pixel_values"].to(device)
        with torch.no_grad():
            outputs = model(pixel_values)

        to_pixel = [(128, 355)] * pixel_values.shape[0]
        predictions = image_processor.post_process_object_detection(outputs, 0, to_pixel)

        scores = []
        positions = []
        for prediction, left in zip(predictions, borders):
            scores.extend(prediction["scores"].tolist())
            pos = (prediction["boxes"][:, 0] + prediction["boxes"][:, 2]) // 2 + left
            positions.extend(pos.long().tolist())

        if not scores or not positions:
            cue_points[track] = []
            continue

        normalized_scores = _normalize_scores(scores)
        ordered = sorted(zip(positions, normalized_scores))
        ordered_positions = [position for position, _ in ordered]
        ordered_scores = np.asarray([score for _, score in ordered], dtype=float)

        peak_idx, _ = find_peaks(ordered_scores, height=sensitivity, distance=radius)
        cue_positions = [ordered_positions[idx] for idx in peak_idx]
        cue_points[track] = list(librosa.frames_to_time(cue_positions))
        if print_points:
            print(f"{track}: {cue_points[track]}")

    if write_output:
        output_path = tracks_path / "_cue_points.txt"
        with output_path.open("w", encoding="utf-8") as handle:
            for track in cue_points:
                handle.write(f"{track}: {cue_points[track]}\n")

    return cue_points


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Predict cue points for one or more tracks using CUE-DETR. "
            "The resulting _cue_points.txt will be saved in the directory with all tracks."
        )
    )
    parser.add_argument("-t", "--tracks", type=str, required=True, help="Path to track directory")
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="disco-eth/cue-detr",
        help="Optional local path to model checkpoint",
    )
    parser.add_argument(
        "-s",
        "--sensitivity",
        type=float,
        default=0.9,
        help="Threshold value for cue points (default = 0.9)",
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=int,
        default=16,
        help="Minimum distance in bars between cue points (default = 16)",
    )
    parser.add_argument("-p", "--print", dest="print_points", action="store_true", help="Print cue points")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    predict_cue_points_for_dir(
        args.tracks,
        checkpoint=args.checkpoint,
        sensitivity=args.sensitivity,
        radius=args.radius,
        print_points=args.print_points,
        write_output=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
