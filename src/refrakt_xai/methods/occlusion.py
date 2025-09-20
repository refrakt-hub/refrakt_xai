"""
Occlusion XAI method for refrakt_xai.

This module implements the Occlusion method using Captum, providing
occlusion-based attribution for model predictions. It registers the
OcclusionXAI class for use in the XAI registry.

Typical usage:
    xai = OcclusionXAI(model)
    attributions = xai.explain(input_tensor, target=target_class)
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union

from captum.attr import Occlusion  # type: ignore
from torch import Tensor

from refrakt_xai.base import BaseXAI
from refrakt_xai.registry import register_xai
from refrakt_xai.utils.model_utils import (
    cleanup_captum_tracing,
    setup_captum_tracing,
)


@register_xai("occlusion")
@dataclass
class OcclusionXAI(BaseXAI):
    """
    Occlusion XAI method using Captum.

    Computes attributions by systematically occluding parts
    of the input and measuring changes in output. Supports configurable
    sliding window shapes, strides, and baselines.

    Attributes:
        model: The model to be explained.
        sliding_window_shapes: Shape of the occlusion window.
        strides: Stride of the occlusion window.
        baselines: Baseline value for occlusion.
        occlusion: Captum Occlusion object.
    """

    sliding_window_shapes: Union[Tuple[int, int, int], Tuple[int, ...]] = field(
        default=(3, 15, 15)
    )
    strides: Union[Tuple[int, int, int], Tuple[int, ...]] = field(default=(3, 8, 8))
    baselines: Union[int, float] = 0

    def __post_init__(self) -> None:
        """Initialize the Captum Occlusion object after dataclass initialization."""

        # Create a wrapper that extracts the primary tensor from ModelOutput
        def model_wrapper(x: Tensor) -> Tensor:
            output = self.model(x)
            # Extract primary tensor from ModelOutput
            if hasattr(output, "reconstruction") and output.reconstruction is not None:
                return output.reconstruction
            elif hasattr(output, "logits") and output.logits is not None:
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
                    f"Unable to extract primary tensor from model output: {type(output)}"
                )

        self.occlusion = Occlusion(model_wrapper)

    def explain(
        self, input_tensor: Tensor, target: Optional[int] = None, **kwargs: Any
    ) -> Tensor:
        """
        Generate occlusion attributions for the given input and target.

        Args:
            input_tensor: Input tensor for which to compute attributions.
            target: Optional target class index for explanation.
            **kwargs: Additional parameters (e.g., sliding_window_shapes,
                                            strides, baselines).

        Returns:
            Tensor of attributions with the same shape as input_tensor.
        """
        sliding_window_shapes = kwargs.get(
            "sliding_window_shapes", self.sliding_window_shapes
        )
        strides = kwargs.get("strides", self.strides)
        baselines = kwargs.get("baselines", self.baselines)

        if sliding_window_shapes is None or sliding_window_shapes == (3, 15, 15):
            # Handle different tensor shapes
            if len(input_tensor.shape) == 4:  # [batch, channels, height, width]
                c, h, w = (
                    input_tensor.shape[1],
                    input_tensor.shape[2],
                    input_tensor.shape[3],
                )
                window_h = min(7, h)
                window_w = min(7, w)
                sliding_window_shapes = (c, window_h, window_w)
            elif len(input_tensor.shape) == 2:  # [batch, features] - flattened data
                # For flattened data, use 1D occlusion
                features = input_tensor.shape[1]
                window_size = min(10, features // 10)  # Use 10% of features as window
                sliding_window_shapes = (window_size,)
            else:
                # Fallback for other shapes
                sliding_window_shapes = (1,)

        if strides is None or strides == (3, 8, 8):
            if len(sliding_window_shapes) == 3:  # 3D window
                stride_h = max(1, sliding_window_shapes[1] // 2)
                stride_w = max(1, sliding_window_shapes[2] // 2)
                strides = (sliding_window_shapes[0], stride_h, stride_w)
            elif len(sliding_window_shapes) == 1:  # 1D window
                strides = (max(1, sliding_window_shapes[0] // 2),)
            else:
                strides = (1,)

        # Ensure input tensor is on the correct device
        input_tensor = input_tensor.detach()
        device = input_tensor.device
        
        # Ensure model is on the same device
        self.model = self.model.to(device)

        setup_captum_tracing(self.model)
        try:
            attributions: Tensor = self.occlusion.attribute(
                input_tensor,
                target=target,
                sliding_window_shapes=sliding_window_shapes,
                strides=strides,
                baselines=baselines,
            )
        finally:
            cleanup_captum_tracing(self.model)
        return attributions
