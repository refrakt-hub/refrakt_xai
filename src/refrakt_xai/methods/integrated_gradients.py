"""
Integrated Gradients XAI method for refrakt_xai.

This module implements the Integrated Gradients method using Captum, providing
path-integrated attribution for model predictions. It registers the
IntegratedGradientsXAI class for use in the XAI registry.

Typical usage:
    xai = IntegratedGradientsXAI(model, n_steps=50)
    attributions = xai.explain(input_tensor, target=target_class)
"""

from dataclasses import dataclass
from typing import Any, Optional

from captum.attr import IntegratedGradients  # type: ignore
from torch import Tensor

# pylint: disable=import-error
from refrakt_xai.base import BaseXAI
from refrakt_xai.registry import register_xai
from refrakt_xai.utils.model_utils import (
    cleanup_captum_tracing,
    setup_captum_tracing,
    validate_model_for_classification,
)


@register_xai("integrated_gradients")
@dataclass
class IntegratedGradientsXAI(BaseXAI):
    """
    Integrated Gradients XAI method using Captum.

    Computes attributions by integrating gradients along a path
    from a baseline to the input. Supports configurable number
    of integration steps.

    Attributes:
        model: The model to be explained.
        n_steps: Number of integration steps (default: 50).
        ig: Captum IntegratedGradients object.
    """

    n_steps: int = 50

    def __post_init__(self) -> None:
        """
        Initialize the Captum IntegratedGradients object
        after dataclass initialization.
        """
        self.ig = IntegratedGradients(self.model)

    def explain(
        self, input_tensor: Tensor, target: Optional[int] = None, **kwargs: Any
    ) -> Tensor:
        """
        Generate integrated gradients attributions for the given input and target.

        Args:
            input_tensor: Input tensor for which to compute attributions.
            target: Optional target class index for explanation.
            **kwargs: Additional parameters (e.g., 'n_steps' to override default).

        Returns:
            Tensor of attributions with the same shape as input_tensor.
        """
        n_steps: int = kwargs.get("n_steps", self.n_steps)

        validate_model_for_classification(
            self.model, input_tensor, "Integrated Gradients"
        )

        setup_captum_tracing(self.model)
        try:
            attributions: Tensor = self.ig.attribute(
                input_tensor, target=target, n_steps=n_steps
            )
        finally:
            cleanup_captum_tracing(self.model)
        return attributions
