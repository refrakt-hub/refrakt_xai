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

import torch
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
        # Ensure input tensor is on the correct device
        input_tensor = input_tensor.detach().requires_grad_(True)
        device = input_tensor.device
        
        # Ensure model is on the same device
        self.model = self.model.to(device)
        
        # Detect model type by checking output structure
        with torch.no_grad():
            sample_output = self.model(
                input_tensor[:1]
            )  # Use single sample to detect output

        # Check if this is a classification model (has logits) vs reconstruction model
        is_classification_model = (
            hasattr(sample_output, "logits")
            and sample_output.logits is not None
            and not (
                hasattr(sample_output, "reconstruction")
                and sample_output.reconstruction is not None
            )
        )

        if is_classification_model:
            # For classification models, use standard DeepLift with target
            if target is None:
                # If no target provided, use predicted class
                logits = sample_output.logits
                target = int(torch.argmax(logits, dim=1).item())

            # Use standard DeepLift for classification
            attributions = self.deeplift.attribute(input_tensor, target=target)
            return attributions
        else:
            # For autoencoder/reconstruction models, use reconstruction-based wrapper
            # Similar approach to OcclusionXAI - extract the reconstruction tensor directly
            import torch.nn as nn

            class AutoencoderReconstructionWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x: Tensor) -> Tensor:
                    output = self.model(x)
                    # Extract reconstruction from ModelOutput
                    reconstruction = None
                    if (
                        hasattr(output, "reconstruction")
                        and output.reconstruction is not None
                    ):
                        reconstruction = output.reconstruction
                    elif hasattr(output, "image") and output.image is not None:
                        reconstruction = output.image
                    elif hasattr(output, "_get_primary_tensor"):
                        recon = output._get_primary_tensor()
                        if recon is not None:
                            reconstruction = recon
                        else:
                            raise ValueError("No primary tensor available")
                    elif isinstance(output, Tensor):
                        reconstruction = output
                    else:
                        raise ValueError(
                            f"Unable to extract reconstruction tensor from model output: {type(output)}"
                        )

                    # For DeepLift to work with reconstruction, we need to return a scalar per sample
                    # We'll sum the reconstruction loss (MSE-like) for each sample
                    # This gives DeepLift a single target to compute gradients for
                    batch_size = x.shape[0]
                    reconstruction_loss = torch.sum(
                        (reconstruction - x).pow(2),
                        dim=list(range(1, len(reconstruction.shape))),
                    )
                    return reconstruction_loss

            # Create a wrapper module instance
            model_wrapper = AutoencoderReconstructionWrapper(self.model)

            # Create a new DeepLift instance with the reconstruction wrapper
            deeplift = DeepLift(model_wrapper)

            # For reconstruction-based attribution, DeepLift will compute attributions
            # with respect to the reconstruction loss (scalar per sample)
            # We don't need to specify a target since the output is already per-sample scalars
            attributions = deeplift.attribute(input_tensor)
            return attributions
