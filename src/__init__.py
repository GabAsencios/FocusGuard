"""
FocusGuard Package Initialization.
Exposes key classes for simplified importing and better code organization.
"""

from .webcam_module import FocusGuardWebcam
from .screen_classifier import ScreenClassifier

__all__ = ['FocusGuardWebcam']