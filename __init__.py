"""
refrakt_xai package initialization.

This module exposes the main XAI (eXplainable AI) method classes for convenient import.
It allows users to access all available XAI methods directly from the refrakt_xai package.

Typical usage:
    from refrakt_xai import SaliencyXAI, IntegratedGradientsXAI, LayerGradCAMXAI, ...

Each imported class provides a unified interface for model explanation methods.
"""

from .methods.concept_saliency import ConceptSaliencyXAI
from .methods.deeplift import DeepLiftXAI
from .methods.integrated_gradients import IntegratedGradientsXAI
from .methods.layer_gradcam import LayerGradCAMXAI
from .methods.occlusion import OcclusionXAI
from .methods.reconstruction_attribution import ReconstructionAttributionXAI
from .methods.saliency import SaliencyXAI

__all__ = [
    "ConceptSaliencyXAI",
    "DeepLiftXAI",
    "IntegratedGradientsXAI",
    "LayerGradCAMXAI",
    "OcclusionXAI",
    "ReconstructionAttributionXAI",
    "SaliencyXAI",
]
