"""
Saliency map XAI method for refrakt_xai.

This module implements the Saliency method using Captum, providing gradient-based
attribution for model predictions. It registers the SaliencyXAI class for use in
the XAI registry.

Typical usage:
    xai = SaliencyXAI(model)
    attributions = xai.explain(input_tensor, target=target_class)
"""

from typing import Any, Optional

from captum.attr import Saliency  # type: ignore
from torch import Tensor

# pylint: disable=import-error
from refrakt_xai.base import BaseXAI
from refrakt_xai.registry import register_xai


# pylint: disable=too-few-public-methods
@register_xai("saliency")
class SaliencyXAI(BaseXAI):
    """
    Saliency map XAI method using Captum.

    Computes input gradients as attributions for model predictions.
    Supports absolute value option for visualization.

    Attributes:
        model: The model to be explained.
        abs: Whether to return absolute attributions (default: True).
        saliency: Captum Saliency object.
    """

    def __init__(self, model: Any, abs_val: bool = True, **kwargs: Any) -> None:
        """
        Initialize the SaliencyXAI method.

        Args:
            model: The model to be explained.
            abs_val: Whether to return absolute attributions (default: True).
            **kwargs: Additional parameters for the base class.
        """
        super().__init__(model, **kwargs)
        self.saliency = Saliency(self.model)
        self.abs_val = abs_val

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
        # Captum workaround: set _captum_tracing flag on model
        setattr(self.model, "_captum_tracing", True)
        try:
            attributions: Tensor = self.saliency.attribute(
                input_tensor, target=target, abs=abs_val
            )
        finally:
            if hasattr(self.model, "_captum_tracing"):
                delattr(self.model, "_captum_tracing")
        return attributions
