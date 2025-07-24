"""
Reconstruction Attribution XAI method for refrakt_xai.

This module implements the Reconstruction Attribution method using Captum's
Integrated Gradients, providing attributions for model reconstructions. It
supports both sum and index-based attributions and registers the
ReconstructionAttributionXAI class for use in the XAI registry.

Typical usage:
    xai = ReconstructionAttributionXAI(model)
    attributions = xai.explain(input_tensor)
"""

from typing import Any, Optional, Tuple

from captum.attr import IntegratedGradients  # type: ignore
from torch import Tensor

# pylint: disable=import-error
from refrakt_xai.base import BaseXAI
from refrakt_xai.registry import register_xai


# pylint: disable=too-few-public-methods
@register_xai("reconstruction_attribution")
class ReconstructionAttributionXAI(BaseXAI):
    """
    Reconstruction Attribution XAI method using Integrated Gradients.

    Computes attributions for model reconstructions,
    supporting both sum and index-based attributions.

    Attributes:
        model: The model to be explained.
        ig: Captum IntegratedGradients object for the reconstruction function.
    """

    def __init__(self, model: Any, **kwargs: Any) -> None:
        """
        Initialize the ReconstructionAttributionXAI method.

        Args:
            model: The model to be explained.
            **kwargs: Additional parameters for the base class.
        """
        super().__init__(model, **kwargs)
        self.ig = IntegratedGradients(self._reconstruction_forward)

    def _reconstruction_forward(self, x: Tensor) -> Tensor:
        """
        Forward function to extract the reconstruction from the model output.

        Args:
            x: Input tensor.

        Returns:
            The reconstruction tensor from the model output.

        Raises:
            TypeError: If the model output is not compatible.
        """
        out = self.model(x)
        # Handle ModelOutput or similar wrappers
        if hasattr(out, "recon"):
            recon = out.recon
            if not isinstance(recon, Tensor):
                raise TypeError("'recon' attribute must be a Tensor.")
            return recon
        if hasattr(out, "reconstruction"):
            recon = out.reconstruction
            if isinstance(recon, Tensor):
                return recon
        if isinstance(out, dict) and "recon" in out:
            recon = out["recon"]
            if not isinstance(recon, Tensor):
                raise TypeError("'recon' in model output dict must be a Tensor.")
            return recon
        if isinstance(out, Tensor):
            return out
        raise TypeError(
            "Model output must be a Tensor, a dict with 'recon', have a 'recon' "
            "or 'reconstruction' attribute."
        )

    def explain(
        self,
        input_tensor: Tensor,
        target: Optional[Tuple[int, ...]] = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Generate reconstruction attributions for the given input.

        Args:
            input_tensor: Input tensor for which to compute attributions.
            target: Optional index or indices for specific output
                    attribution.
            **kwargs: Additional parameters (e.g., baseline, additional_forward_args, abs).

        Returns:
            Tensor of attributions with the same shape as input_tensor.

        Raises:
            TypeError: If the attributions are not a tensor.
            ValueError: If the output shape is unsupported for index-based attribution.
        """
        baseline = kwargs.get("baseline", 0 * input_tensor)
        additional_forward_args = kwargs.get("additional_forward_args", None)
        abs_val = kwargs.get("abs", True)

        if target is None:
            # Attribute the sum of the reconstruction
            def agg_forward(x: Tensor) -> Tensor:
                return self._reconstruction_forward(x).view(x.size(0), -1).sum(dim=1)

            ig = IntegratedGradients(agg_forward)
            attributions = ig.attribute(
                input_tensor,
                baselines=baseline,
                additional_forward_args=additional_forward_args,
            )
        else:
            # Attribute a specific output index (e.g., pixel)
            def idx_forward(x: Tensor) -> Tensor:
                recon = self._reconstruction_forward(x)
                if recon.dim() == 2:
                    return recon[:, target[0]]
                if recon.dim() == 4:
                    # (B, C, H, W)
                    _, c, h, w = target
                    return recon[:, c, h, w]
                raise ValueError("Unsupported output shape for target attribution.")

            ig = IntegratedGradients(idx_forward)
            attributions = ig.attribute(
                input_tensor,
                baselines=baseline,
                additional_forward_args=additional_forward_args,
            )

        if abs_val:
            attributions = attributions.abs()
        if not isinstance(attributions, Tensor):
            raise TypeError("Attributions must be a Tensor.")
        return attributions
