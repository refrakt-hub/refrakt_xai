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

        # Create a wrapper that handles different model types
        def model_wrapper(x: Tensor) -> Tensor:
            output = self.model(x)
            # Extract primary tensor from ModelOutput
            if hasattr(output, "reconstruction") and output.reconstruction is not None:
                # For autoencoders, compute reconstruction loss (scalar per sample)
                reconstruction = output.reconstruction
                reconstruction_loss = torch.sum(
                    (reconstruction - x).pow(2),
                    dim=list(range(1, len(reconstruction.shape))),
                )
                return reconstruction_loss
            elif hasattr(output, "logits") and output.logits is not None:
                # For classification models, return logits
                return output.logits
            elif hasattr(output, "embeddings") and output.embeddings is not None:
                return output.embeddings
            elif hasattr(output, "image") and output.image is not None:
                return output.image
            elif hasattr(output, "_get_primary_tensor"):
                primary_tensor = output._get_primary_tensor()
                if primary_tensor is not None:
                    return primary_tensor
                else:
                    raise ValueError("No primary tensor available in ModelOutput")
            elif isinstance(output, Tensor):
                return output
            else:
                raise ValueError(
                    f"Unable to extract primary tensor from model output: "
                    f"{type(output)}"
                )

        self.saliency = Saliency(model_wrapper)

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

        # Ensure input tensor requires gradients and is on the correct device
        input_tensor = input_tensor.detach().requires_grad_(True)
        device = input_tensor.device

        # Ensure model is on the same device
        self.model = self.model.to(device)

        setup_captum_tracing(self.model)
        try:
            # For autoencoder models, we don't need a target since the wrapper
            # returns scalars. For classification models, target is required
            if target is None:
                # Check if this is an autoencoder by testing the output
                with torch.no_grad():
                    test_output = self.model(input_tensor[:1])
                    is_autoencoder = (
                        hasattr(test_output, "reconstruction")
                        and test_output.reconstruction is not None
                    )

                if is_autoencoder:
                    # For autoencoders, no target needed
                    attributions: Tensor = self.saliency.attribute(
                        input_tensor, abs=abs_val
                    )
                else:
                    # For classification models, we need a target
                    raise ValueError(
                        "Target not provided when necessary, cannot take gradient "
                        "with respect to multiple outputs."
                    )
            else:
                # Target provided, use it
                attributions: Tensor = self.saliency.attribute(
                    input_tensor, target=target, abs=abs_val
                )
        finally:
            cleanup_captum_tracing(self.model)
        return attributions
