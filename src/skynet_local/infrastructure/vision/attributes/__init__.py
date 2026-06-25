"""Attribute analyzer adapters for age, gender, emotion and chewing estimation."""

from skynet_local.infrastructure.vision.attributes.chewing_detector import ChewingDetector
from skynet_local.infrastructure.vision.attributes.emotion_analyzer import EmotionAnalyzer
from skynet_local.infrastructure.vision.attributes.ferplus_detector import FerplusEmotionDetector
from skynet_local.infrastructure.vision.attributes.ferplus_detector_calibrated import FerplusEmotionDetectorCalibrated
from skynet_local.infrastructure.vision.attributes.emotion_detector_base import EmotionDetectorBase
from skynet_local.infrastructure.vision.attributes.landmark_emotion_detector import LandmarkEmotionDetector
from skynet_local.infrastructure.vision.attributes.ensemble_emotion_detector import EnsembleEmotionDetector

__all__ = [
    "ChewingDetector",
    "EmotionAnalyzer",
    "FerplusEmotionDetector",
    "FerplusEmotionDetectorCalibrated",
    "EmotionDetectorBase",
    "LandmarkEmotionDetector",
    "EnsembleEmotionDetector"
]
