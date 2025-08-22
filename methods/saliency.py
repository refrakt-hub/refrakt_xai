"""
Saliency map XAI method for refrakt_xai.

This module implements the Saliency method using Captum, providing gradient-based
attribution for model predictions. It registers the SaliencyXAI class for use in
the XAI registry.

Typical usage:
    xai = SaliencyXAI(model)
    attributions = xai.explain(input_tensor, target=target_class)
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
from captum.attr import Saliency  # type: ignore
from torch import Tensor

from refrakt_xai.base import BaseXAI
from refrakt_xai.registry import register_xai
from refrakt_xai.utils.model_utils import (
    cleanup_captum_tracing,
    setup_captum_tracing,
)


@register_xai("saliency")
@dataclass
class SaliencyXAI(BaseXAI):
    """
    Saliency map XAI method using Captum.

    Computes input gradients as attributions for model predictions.
    Supports absolute value option for visualization.

    Attributes:
        model: The model to be explained.
        abs_val: Whether to return absolute attributions (default: True).
        saliency: Captum Saliency object.
    """

    abs_val: bool = True

    def __post_init__(self) -> None:
        """Initialize the Captum Saliency object after dataclass initialization."""
        self.saliency = Saliency(self.model)

    def explain(
        self, input_tensor: Tensor, target: Optional[int] = None, **kwargs: Any
    ) -> Tensor:
        """
        Generate a saliency map for the given input and target.

        Args:
            input_tensor: Input tensor for which to compute attributions.
            target: Optional target class index for explanation.
            **kwargs: Additional parameters (e.g., 'abs' to override default).

        Returns:
            Tensor of attributions with the same shape as input_tensor.
        """
        abs_val: bool = kwargs.get("abs", self.abs_val)

        setup_captum_tracing(self.model)
        try:
            attributions: Tensor = self.saliency.attribute(input_tensor, target=target, abs=abs_val)
        finally:
            cleanup_captum_tracing(self.model)
        return attributions
