"""
DeepLift XAI method for refrakt_xai.

This module implements the DeepLift method using Captum, providing
gradient-based attribution for model predictions. It registers the
DeepLiftXAI class for use in the XAI registry.

Typical usage:
    xai = DeepLiftXAI(model)
    attributions = xai.explain(input_tensor, target=target_class)
"""

from dataclasses import dataclass
from typing import Any, Optional

from captum.attr import DeepLift  # type: ignore
from torch import Tensor

from refrakt_xai.base import BaseXAI
from refrakt_xai.registry import register_xai


@register_xai("deeplift")
@dataclass
class DeepLiftXAI(BaseXAI):
    """
    DeepLift XAI method using Captum.

    Computes attributions using the DeepLift algorithm, which provides
    gradient-based attribution with baseline comparison.

    Attributes:
        model: The model to be explained.
        deeplift: Captum DeepLift object.
    """

    def __post_init__(self) -> None:
        """Initialize the Captum DeepLift object after dataclass initialization."""
        self.deeplift = DeepLift(self.model)

    def explain(
        self, input_tensor: Tensor, target: Optional[int] = None, **kwargs: Any
    ) -> Tensor:
        """
        Generate DeepLift attributions for the given input and target.

        Args:
            input_tensor: Input tensor for which to compute attributions.
            target: Optional target class index for explanation.
            **kwargs: Additional parameters.

        Returns:
            Tensor of attributions with the same shape as input_tensor.
        """
        # Only pass target if it's provided (for multi-class models)
        if target is not None:
            attributions: Tensor = self.deeplift.attribute(input_tensor, target=target)
        else:
            attributions = self.deeplift.attribute(input_tensor)
        return attributions
