"""
Layer GradCAM XAI method for refrakt_xai.

This module implements the Layer GradCAM method using Captum, providing
gradient-based attribution for specific layers. It registers the
LayerGradCAMXAI class for use in the XAI registry.

Typical usage:
    xai = LayerGradCAMXAI(model, layer='layer3.1.conv2')
    attributions = xai.explain(input_tensor, target=target_class)
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
from captum.attr import LayerGradCam  # type: ignore
from torch import Tensor
from torch.nn import Module

from refrakt_xai.base import BaseXAI
from refrakt_xai.registry import register_xai
from refrakt_xai.utils.layer_detection import (
    collect_conv_layers,
    find_layer_with_weights,
    resolve_layer_path,
    select_best_layer,
)
from refrakt_xai.utils.model_utils import (
    extract_primary_tensor,
    get_model_device,
    process_batch_attributions,
)


@register_xai("layer_gradcam")
@dataclass
class LayerGradCAMXAI(BaseXAI):
    """
    Layer GradCAM XAI method using Captum.

    Computes attributions using Layer GradCAM, which provides
    gradient-based attribution for specific layers.

    Attributes:
        model: The model to be explained.
        layer: The layer to compute GradCAM for.
        layer_gradcam: Captum LayerGradCam object.
    """

    layer: Optional[Union[Module, str]] = None

    def __post_init__(self) -> None:
        """
        Initialize the layer and LayerGradCam object
        after dataclass initialization.
        """
        if isinstance(self.layer, str):
            if self.layer == "auto":
                self.layer = self._auto_detect_layer()
            else:
                self.layer = self._resolve_layer_path(self.layer)

        if self.layer is None:
            self.layer = self._auto_detect_layer()

        def model_forward_wrapper(x: Tensor) -> Tensor:
            """Wrapper that extracts the primary tensor from model output."""
            output = self.model(x)
            return extract_primary_tensor(output)

        self.layer_gradcam = LayerGradCam(model_forward_wrapper, self.layer)

    def _auto_detect_layer(self) -> Module:
        """
        Auto-detect an appropriate layer for GradCAM based on spatial dimensions.

        Returns:
            The selected layer module

        Raises:
            ValueError: If no suitable layer is found
        """
        model_device = get_model_device(self.model)

        conv_layers = collect_conv_layers(self.model, model_device)

        if not conv_layers:
            fallback_layer = find_layer_with_weights(self.model)
            if fallback_layer is None:
                raise ValueError("No suitable layer found for GradCAM")
            return fallback_layer

        return select_best_layer(conv_layers)

    def _resolve_layer_path(self, layer_path: str) -> Module:
        """
        Resolve a string layer path to an actual layer module.

        Args:
            layer_path: String path like 'layer3.1.conv2' or 'backbone.layer1.1.conv2.0'

        Returns:
            The resolved layer module

        Raises:
            ValueError: If the layer path cannot be resolved
        """
        return resolve_layer_path(self.model, layer_path)

    def explain(
        self,
        input_tensor: Tensor,
        target: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Generate Layer GradCAM attributions for the given input and target.

        Args:
            input_tensor: Input tensor for which to compute attributions.
            target: Optional target class index for explanation.
            **kwargs: Additional parameters.

        Returns:
            Tensor of attributions with the same shape as input_tensor.
        """
        batch_size = input_tensor.shape[0]

        if (
            target is not None
            and isinstance(target, (list, tuple))
            and len(target) == batch_size
        ):
            attributions = self._process_batch_with_individual_targets(
                input_tensor, target, batch_size
            )
        else:
            attributions = self._process_batch_with_single_target(input_tensor, target)

        return process_batch_attributions(attributions, batch_size)

    def _process_batch_with_individual_targets(
        self, input_tensor: Tensor, target: Any, batch_size: int
    ) -> Tensor:
        """Process batch with individual targets for each sample."""
        all_attributions = []

        original_training = self.model.training

        try:
            self.model.eval()

            for i in range(batch_size):
                single_input = input_tensor[i : i + 1]
                single_target = (
                    target[i] if isinstance(target, (Tensor, list)) else target
                )

                sample_attributions = self.layer_gradcam.attribute(
                    single_input, target=single_target
                )

                all_attributions.append(sample_attributions)

            return torch.cat(all_attributions, dim=0)

        finally:
            if original_training:
                self.model.train()

    def _process_batch_with_single_target(
        self,
        input_tensor: Tensor,
        target: Optional[Union[int, List[int], Tuple[int, ...]]],
    ) -> Any:
        """Process entire batch with single target."""
        return self.layer_gradcam.attribute(input_tensor, target=target)
