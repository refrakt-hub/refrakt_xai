"""
DeepLift XAI method for refrakt_xai.

This module implements the DeepLift method using Captum, providing
reference-based attribution for model predictions. It includes a sum wrapper for
reconstruction models and registers the DeepLiftXAI class for use in the XAI
registry.

Typical usage:
    xai = DeepLiftXAI(model)
    attributions = xai.explain(input_tensor)
"""

from typing import Any, Optional

import torch
from captum.attr import DeepLift  # type: ignore
from torch import nn

# pylint: disable=import-error
from refrakt_xai.base import BaseXAI
from refrakt_xai.registry import register_xai


# pylint: disable=too-few-public-methods
class DeepLiftSumWrapper(nn.Module):
    """
    Wrapper module to sum model reconstructions for DeepLift attribution.

    This wrapper ensures compatibility with models that output reconstructions
    as attributes, dictionary entries, or tensors, and computes the sum for attribution.
    """

    def __init__(self, model: Any) -> None:
        """
        Initialize the DeepLiftSumWrapper.

        Args:
            model: The model to be wrapped for DeepLift attribution.
        """
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that extracts and sums the reconstruction output.

        Args:
            x: Input tensor.

        Returns:
            Summed reconstruction tensor for attribution.

        Raises:
            TypeError: If the model output is not compatible.
        """
        out = self.model(x)
        if hasattr(out, "recon"):
            recon = out.recon
        elif hasattr(out, "reconstruction"):
            recon = out.reconstruction
        elif isinstance(out, dict) and "recon" in out:
            recon = out["recon"]
        elif isinstance(out, torch.Tensor):
            recon = out
        else:
            raise TypeError(
                "Model output must be a Tensor, a dict with 'recon', have a 'recon' "
                "or 'reconstruction' attribute."
            )
        result = recon.view(x.size(0), -1).sum(dim=1)
        if not isinstance(result, torch.Tensor):
            raise TypeError("Result must be a Tensor.")
        return result


@register_xai("deeplift")
class DeepLiftXAI(BaseXAI):
    """
    DeepLift XAI method using Captum.

    Computes attributions by comparing model outputs to a reference baseline.
    Uses a sum wrapper for models with reconstruction outputs.

    Attributes:
        model: The model to be explained.
        sum_wrapper: Wrapper for summing model reconstructions.
    """

    def __init__(self, model: Any, **kwargs: Any) -> None:
        """
        Initialize the DeepLiftXAI method.

        Args:
            model: The model to be explained.
            **kwargs: Additional parameters for the base class.
        """
        super().__init__(model, **kwargs)
        self.sum_wrapper = DeepLiftSumWrapper(model)

    def explain(
        self,
        input_tensor: torch.Tensor,
        target: Optional[int] = None,
        **kwargs: Any,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        Generate DeepLift attributions for the given input.

        Args:
            input_tensor: Input tensor for which to compute attributions.
            **kwargs: Additional parameters (e.g., baseline,
                        additional_forward_args, abs).

        Returns:
            Tensor of attributions with the same shape as input_tensor.

        Raises:
            TypeError: If the attributions are not a tensor.
        """
        baseline = kwargs.get("baseline", torch.zeros_like(input_tensor))
        additional_forward_args = kwargs.get("additional_forward_args", None)
        abs_val = kwargs.get("abs", True)
        dl = DeepLift(self.sum_wrapper)
        attributions = dl.attribute(
            input_tensor,
            baselines=baseline,
            additional_forward_args=additional_forward_args,
        )
        if abs_val:
            attributions = attributions.abs()
        if not isinstance(attributions, torch.Tensor):
            raise TypeError("Attributions must be a Tensor.")
        return attributions
