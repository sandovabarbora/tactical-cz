"""End-to-end vision pipeline: video → per-frame tracking + minimap coords.

Composes:
    1. PlayerDetector (Roboflow sports YOLOv8)
    2. PitchDetector  (32 keypoints for homography)
    3. PlayerTracker  (ByteTrack via supervision)
    4. TeamClassifier (KMeans HSV — fit on first N frames)
    5. pitch.project_to_pitch (image foot point → 105×68 metres)

Outputs per call:
    DataFrame columns:
        frame_idx, tracker_id, class_id, team_id,
        x1, y1, x2, y2, cx, cy_foot,
        pitch_x_m, pitch_y_m            (NaN where homography failed)

Sampling strategy: by default we sample every K-th frame (K=2 → 12fps
from 25fps broadcast), enough for tactical analysis. Configurable.

Usage:
    from tactical_cz.vision.pipeline import VisionPipeline
    pipe = VisionPipeline()
    df = pipe.process_video("data/raw/match.mp4", frame_stride=2,
                             team_fit_frames=50, max_frames=None)
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tactical_cz.config import (
    PROCESSED_DIR,
    configure_logging,
    ensure_dirs,
)
from tactical_cz.vision.detector import (
    BallDetector,
    PitchDetector,
    PlayerDetector,
)
from tactical_cz.vision.pitch import compute_homography, project_to_pitch
from tactical_cz.vision.team import TeamClassifier
from tactical_cz.vision.tracker import PlayerTracker, TrackerConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Top-level pipeline knobs."""

    frame_stride: int = 2                  # process every Nth frame
    team_fit_frames: int = 50              # how many frames to sample for team KMeans
    use_ball_detector: bool = False         # second-pass ball detection (slow + optional)
    write_annotated_video: bool = False     # save annotated mp4 alongside parquet
    pitch_keypoint_min: int = 4             # min visible landmarks for homography
    tracker_backend: str = "botsort"        # "botsort" | "bytetrack"
    tracker_with_reid: bool = True          # OSNet appearance re-ID (BotSort only)


class VisionPipeline:
    """One instance per match. Heavy models loaded once."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        logger.info("Loading detectors (first run downloads ~150MB)…")
        self.player_detector = PlayerDetector()
        self.pitch_detector = PitchDetector()
        self.ball_detector = BallDetector() if self.config.use_ball_detector else None
        self.tracker = PlayerTracker(TrackerConfig(
            frame_rate=25,
            backend=self.config.tracker_backend,
            with_reid=self.config.tracker_with_reid,
        ))
        self.team_clf = TeamClassifier()
        self._team_fitted = False

    def process_video(
        self,
        video_path: Path | str,
        max_frames: int | None = None,
    ) -> pd.DataFrame:
        """Process video end-to-end. Returns long-format tracking DataFrame."""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "Video: %s  · %d frames @ %.1f fps  · stride=%d",
            video_path.name, total_frames, fps, self.config.frame_stride,
        )

        # First pass: sample frames for team-classifier fit
        sample_frames: list[np.ndarray] = []
        sample_dets: list = []

        # Main pass
        all_rows: list[dict] = []
        frame_idx = 0
        processed = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.config.frame_stride != 0:
                frame_idx += 1
                continue
            if max_frames is not None and processed >= max_frames:
                break

            # Detect players + GKs + refs + ball (multi-class)
            dets = self.player_detector(frame)
            player_dets = self.player_detector.players_only(dets)

            # Build team-fit sample (first team_fit_frames worth of player crops)
            if not self._team_fitted and len(sample_frames) < self.config.team_fit_frames:
                if len(player_dets) > 0:
                    sample_frames.append(frame.copy())
                    sample_dets.append(player_dets)
                # Fit when we have enough
                if len(sample_frames) >= self.config.team_fit_frames:
                    try:
                        self.team_clf.fit(sample_frames, sample_dets)
                        self._team_fitted = True
                        logger.info("TeamClassifier fitted at frame %d", frame_idx)
                    except ValueError as exc:
                        logger.warning("Team fit failed: %s — retrying later", exc)

            # Track (BoxMOT needs the raw frame for re-ID crops + CMC)
            tracked = self.tracker.update(frame_idx, player_dets, frame)

            # Team assignment (if fitted)
            team_ids = (
                self.team_clf.predict(frame, tracked)
                if self._team_fitted and len(tracked) > 0
                else np.full(len(tracked), -1, dtype=int)
            )

            # Pitch homography per frame (recompute — camera moves)
            kp = self.pitch_detector(frame)
            try:
                H = self._homography_from_pitch_dets(kp)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Frame %d homography failed: %s", frame_idx, exc)
                H = None

            # Project foot points if we have a homography
            pitch_xy = None
            if H is not None and len(tracked) > 0:
                foot_pts = np.column_stack([
                    (tracked.xyxy[:, 0] + tracked.xyxy[:, 2]) / 2,
                    tracked.xyxy[:, 3],
                ])
                pitch_xy = project_to_pitch(foot_pts, H)

            # Materialize rows for this frame
            for i in range(len(tracked)):
                x1, y1, x2, y2 = tracked.xyxy[i]
                row = {
                    "frame_idx": frame_idx,
                    "tracker_id": int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1,
                    "class_id": int(tracked.class_id[i]) if tracked.class_id is not None else -1,
                    "team_id": int(team_ids[i]) if i < len(team_ids) else -1,
                    "confidence": float(tracked.confidence[i]) if tracked.confidence is not None else float("nan"),
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "cx": float((x1 + x2) / 2),
                    "cy_foot": float(y2),
                }
                if pitch_xy is not None:
                    row["pitch_x_m"] = float(pitch_xy[i, 0])
                    row["pitch_y_m"] = float(pitch_xy[i, 1])
                else:
                    row["pitch_x_m"] = float("nan")
                    row["pitch_y_m"] = float("nan")
                all_rows.append(row)

            processed += 1
            if processed % 50 == 0:
                logger.info("  processed %d frames (current idx=%d)", processed, frame_idx)
            frame_idx += 1

        cap.release()
        df = pd.DataFrame(all_rows)
        logger.info("Done: %d rows from %d processed frames", len(df), processed)
        return df

    def _homography_from_pitch_dets(self, keypoints) -> np.ndarray | None:
        """Adapt PitchDetector output (KeyPoints) into compute_homography input.

        The pitch model is a YOLOv8-pose: it emits one "pitch" detection per
        frame with 32 keypoints (one per pitch landmark) inside. We pass
        the KeyPoints object straight through to compute_homography, which
        handles confidence filtering + landmark index mapping internally.
        """
        if keypoints is None or len(keypoints) == 0:
            return None
        return compute_homography(
            keypoints=keypoints,
            min_correspondences=self.config.pitch_keypoint_min,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to broadcast video (mp4)")
    parser.add_argument("--out", "-o", type=str,
                        default=str(PROCESSED_DIR / "vision_tracking.parquet"),
                        help="Output parquet path")
    parser.add_argument("--stride", type=int, default=2,
                        help="Process every Nth frame (default 2 = 12.5fps)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit total processed frames (for smoke tests)")
    parser.add_argument("--team-fit-frames", type=int, default=50)
    parser.add_argument("--use-ball-detector", action="store_true",
                        help="Run a second-pass ball-only detector (slow)")
    parser.add_argument("--tracker", choices=["botsort", "bytetrack"], default="botsort",
                        help="Tracker backend (default: botsort, supports re-ID)")
    parser.add_argument("--no-reid", action="store_true",
                        help="Disable OSNet appearance re-ID (BotSort only)")
    args = parser.parse_args(argv)

    configure_logging()
    ensure_dirs()

    cfg = PipelineConfig(
        frame_stride=args.stride,
        team_fit_frames=args.team_fit_frames,
        use_ball_detector=args.use_ball_detector,
        tracker_backend=args.tracker,
        tracker_with_reid=not args.no_reid,
    )
    pipe = VisionPipeline(cfg)
    df = pipe.process_video(args.input, max_frames=args.max_frames)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows → %s", len(df), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
