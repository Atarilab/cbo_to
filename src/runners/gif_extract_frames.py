#!/usr/bin/env python3
"""
Extract start, middle, and end frames from a GIF.

Requires: Pillow (pip install pillow)
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from PIL import Image, ImageSequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract start/middle/end frames from a GIF.")
    parser.add_argument("gif_path", help="Path to input GIF file.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="frames_out",
        help="Directory to save extracted frames (default: frames_out).",
    )
    parser.add_argument(
        "--middle-index",
        type=int,
        default=None,
        help="Override middle frame index (0-based). Defaults to midpoint.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Filename prefix for output images (default: input GIF name).",
    )
    return parser.parse_args()


def get_frame_count(img: Image.Image) -> int:
    try:
        return img.n_frames
    except AttributeError:
        return 1


def save_frame(frame: Image.Image, out_path: Path) -> None:
    # Convert to RGBA to preserve transparency in GIFs.
    frame_rgba = frame.convert("RGBA")
    frame_rgba.save(out_path, format="PNG")


def sanitize_token(value: str) -> str:
    # Keep filenames portable and readable.
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())


def read_commit_info_from_file(gif_path: Path) -> tuple[str, str]:
    commit_dir = gif_path.parent.parent.parent
    commit_file = commit_dir / "commit.txt"
    if not commit_file.exists():
        raise FileNotFoundError(f"commit.txt not found in {commit_dir}")

    commit = None
    branch = None
    with commit_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().startswith("branch:"):
                branch = line.split(":", 1)[1].strip()
                continue
            if commit is None:
                commit = line

    if not commit:
        raise ValueError(f"commit.txt missing commit hash in {commit_file}")
    if not branch:
        raise ValueError(f"commit.txt missing branch in {commit_file}")

    commit_short = commit[:7]
    return sanitize_token(branch), sanitize_token(commit_short)


def main() -> int:
    args = parse_args()
    gif_path = Path(args.gif_path)
    if not gif_path.exists():
        raise FileNotFoundError(f"GIF not found: {gif_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(gif_path) as img:
        prefix = args.prefix if args.prefix else gif_path.stem
        total_frames = get_frame_count(img)
        if total_frames <= 0:
            raise ValueError("GIF has no frames.")

        start_idx = 0
        end_idx = total_frames - 1
        middle_idx = args.middle_index
        if middle_idx is None:
            middle_idx = total_frames // 2
        if not (0 <= middle_idx < total_frames):
            raise ValueError(
                f"middle-index {middle_idx} out of range for {total_frames} frames"
            )

        indices = [
            ("start", start_idx),
            ("middle", middle_idx),
            ("end", end_idx),
        ]

        branch, commit = read_commit_info_from_file(gif_path)
        git_suffix = f"_{branch}_{commit}"

        for name, idx in indices:
            # ImageSequence.Iterator handles seeking to each frame.
            frame = ImageSequence.Iterator(img)[idx]
            out_path = output_dir / f"{prefix}_{idx:04d}_{name}{git_suffix}.png"
            save_frame(frame, out_path)
            print(f"Saved {name} frame {idx} to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
