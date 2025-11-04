"""
Reconstruction Attribution XAI method for refrakt_xai.

This module implements the Reconstruction Attribution method, providing
reconstruction-based attribution for model predictions. It registers the
ReconstructionAttributionXAI class for use in the XAI registry.

Typical usage:
    xai = ReconstructionAttributionXAI(model)
    attributions = xai.explain(input_tensor, target=target_class)
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor

from refrakt_xai.base import BaseXAI
from refrakt_xai.registry import register_xai


@register_xai("reconstruction_attribution")
@dataclass
class ReconstructionAttributionXAI(BaseXAI):
    """
    Reconstruction Attribution XAI method.

    Computes attributions by comparing model reconstructions to inputs.
    Suitable for autoencoder and reconstruction-based models.

    Attributes:
        model: The model to be explained.
    """

    def explain(
        self, input_tensor: Tensor, target: Optional[int] = None, **kwargs: Any
    ) -> Tensor:
        """
        Generate reconstruction attributions for the given input and target.

        Args:
            input_tensor: Input tensor for which to compute attributions.
            target: Optional target class index for explanation
            (ignored for reconstruction).
            **kwargs: Additional parameters.

        Returns:
            Tensor of attributions with the same shape as input_tensor.
        """
        # Ensure input tensor is on the correct device
        input_tensor = input_tensor.detach()
        device = input_tensor.device

        # Ensure model is on the same device
        self.model = self.model.to(device)

        with torch.no_grad():
            reconstruction = self.model(input_tensor)
            if hasattr(reconstruction, "recon"):
                reconstruction = reconstruction.recon
            elif hasattr(reconstruction, "reconstruction"):
                reconstruction = reconstruction.reconstruction
            elif isinstance(reconstruction, dict) and "recon" in reconstruction:
                reconstruction = reconstruction["recon"]

            if not isinstance(reconstruction, Tensor):
                raise ValueError(
                    "Model output must be a tensor for reconstruction attribution"
                )

            attributions = input_tensor - reconstruction

        return attributions
