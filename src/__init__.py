"""
Document AI System
End-to-End AI System for Intelligent Document Understanding
"""

__version__ = "1.0.0"
__author__ = "Flikt Technology AI Developer"

from .pipeline import DocumentAIPipeline
from .config import load_config

__all__ = [
    'DocumentAIPipeline',
    'load_config'
]
