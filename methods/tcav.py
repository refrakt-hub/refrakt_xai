"""
TCAV XAI method for refrakt_xai.

This module implements the TCAV (Testing with Concept Activation Vectors) method
using Captum, providing concept-based global explanations for model predictions.
It registers the TCAVXAI class for use in the XAI registry.

Typical usage:
    xai = TCAVXAI(model, bottleneck_layer, concepts)
    scores = xai.explain(input_tensor, target)

Note: TCAV requires concept datasets, a trained model, and a specified bottleneck layer.
"""

from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from torch import Tensor

try:
    from captum.concept import TCAV  # type: ignore
except ImportError:
    TCAV = None

from refrakt_xai.base import BaseXAI
from refrakt_xai.registry import register_xai


@register_xai("tcav")
@dataclass
class TCAVXAI(BaseXAI):
    """
    TCAV XAI method using Captum.

    Computes global concept importance scores using the TCAV algorithm.

    Args:
        model: The model to be explained.
        bottleneck: The model layer to use as the bottleneck
                    for TCAV (str or list of str).
        concepts: List of Captum Concept objects.
    """

    bottleneck: Optional[str] = None
    concepts: Optional[List[Any]] = None

    def __post_init__(self) -> None:
        """Initialize TCAV after dataclass initialization."""
        if self.bottleneck is None:
            self.bottleneck = self._auto_resolve_bottleneck(self.model)

        if self.concepts is None:
            raise ValueError(
                "You must provide a list of Concept objects for TCAV. "
                "Auto-resolution is not implemented."
            )
        if TCAV is not None:
            self.tcav = TCAV(
                self.model, [self.bottleneck], self.concepts  # type: ignore
            )
        else:
            self.tcav = None

    def _auto_resolve_bottleneck(self, model: Any) -> str:
        """
        Attempt to auto-resolve the bottleneck layer by
        picking the last Conv2d or Linear layer.
        """
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                return str(name)
        raise ValueError(
            "Could not auto-resolve bottleneck layer.", "Please specify it explicitly."
        )

    def explain(
        self, input_tensor: Tensor, target: Optional[int] = None, **kwargs: Any
    ) -> Any:
        """
        Run TCAV to compute concept importance scores.

        Args:
            input_tensor: Input tensor for which to compute TCAV scores.
            target: Target labels for the inputs.
            **kwargs: Additional parameters for TCAV.

        Returns:
            TCAV scores for each concept.
        """

        if self.tcav is None:
            raise ImportError(
                "TCAV could not be imported from captum.concept. "
                "Please install a compatible version of Captum."
            )

        raise NotImplementedError(
            "TCAVXAI requires concept datasets, a bottleneck layer, and targets. "
            "See Captum's TCAV documentation for details: "
            "https://captum.ai/api/concept.html#tcav"
        )
