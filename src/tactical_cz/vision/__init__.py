"""Vision layer — detection, tracking, pitch homography, team classification.

Public API:
    PlayerDetector, BallDetector, PitchDetector   — pretrained YOLOv8 wrappers
    PlayerTracker                                  — ByteTrack adapter
    TeamClassifier                                 — KMeans on jersey HSV
    compute_homography, project_to_pitch          — pitch-keypoint math
    VisionPipeline                                 — end-to-end orchestrator
    main()                                         — CLI entry: tactical-vision
"""

from tactical_cz.vision.detector import (
    BallDetector,
    DetectionConfig,
    PitchDetector,
    PlayerDetector,
)
from tactical_cz.vision.pipeline import PipelineConfig, VisionPipeline, main
from tactical_cz.vision.pitch import (
    PITCH_LANDMARKS_M,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    compute_homography,
    project_to_pitch,
)
from tactical_cz.vision.team import TeamClassifier
from tactical_cz.vision.tracker import PlayerTracker, TrackerConfig

__all__ = [
    "BallDetector",
    "DetectionConfig",
    "PipelineConfig",
    "PITCH_LANDMARKS_M",
    "PITCH_LENGTH_M",
    "PITCH_WIDTH_M",
    "PitchDetector",
    "PlayerDetector",
    "PlayerTracker",
    "TeamClassifier",
    "TrackerConfig",
    "VisionPipeline",
    "compute_homography",
    "main",
    "project_to_pitch",
]
