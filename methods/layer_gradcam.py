"""
Layer Grad-CAM XAI method for refrakt_xai.

This module implements the Layer Grad-CAM method using Captum, providing
layer-wise attribution for model predictions. It includes utilities for
resolving target layers in various model architectures and registers the
LayerGradCAMXAI class for use in the XAI registry.

Typical usage:
    xai = LayerGradCAMXAI(model, target_layer="auto")
    attributions = xai.explain(input_tensor, target=target_class)
"""

from typing import Any, Optional

from captum.attr import LayerAttribution, LayerGradCam  # type: ignore
from torch import Tensor, nn

# pylint: disable=import-error
from refrakt_xai.base import BaseXAI
from refrakt_xai.registry import register_xai
from refrakt_xai.utils.layer_resolvers import (
    _resolve_auto_target_layer_fallback,
    _resolve_convnext_from_dict,
    _resolve_convnext_layer,
    _resolve_resnet_from_dict,
    _resolve_resnet_layer,
    _resolve_swin_from_dict,
    _resolve_swin_layer,
    _resolve_unknown_arch_fallback,
    _resolve_vit_from_dict,
    _resolve_vit_layer,
)


def resolve_layer(model: nn.Module, layer_path: str) -> nn.Module:
    """
    Resolve a dot-separated string path to a model layer.

    Args:
        model: The model containing the target layer.
        layer_path: Dot-separated path to the layer (e.g., 'backbone.layer3.5').

    Returns:
        The resolved nn.Module corresponding to the path.

    Raises:
        ValueError: If the resolved object is not an nn.Module.
    """
    parts = layer_path.split(".")
    layer: Any = model
    for part in parts:
        if part.isdigit():
            layer = layer[int(part)]
        else:
            layer = getattr(layer, part)
    if not isinstance(layer, nn.Module):
        raise ValueError(f"Resolved object at '{layer_path}' is not a nn.Module.")
    return layer


def resolve_auto_target_layer(model: nn.Module) -> str:
    """
    Return the recommended default layer path for
    Layer Grad-CAM for custom Refrakt models.
    """
    if hasattr(model, "backbone") and isinstance(getattr(model, "backbone"), nn.Module):
        return resolve_auto_target_layer(getattr(model, "backbone"))
    for resolver in (
        _resolve_resnet_layer,
        _resolve_convnext_layer,
        _resolve_vit_layer,
        _resolve_swin_layer,
    ):
        result = resolver(model)
        if result is not None:
            return result
    _resolve_auto_target_layer_fallback(model)
    raise ValueError(
        "Auto target_layer not implemented for this model architecture. "
        "See above for available layers."
    )


def resolve_target_layer_from_dict(model: nn.Module, layer_dict: dict[str, Any]) -> str:
    """
    Resolve a dict-based target_layer specification to a string path for various architectures.
    """
    prepend_backbone = layer_dict.get("prepend_backbone", True)
    arch = layer_dict.get("arch", "resnet").lower()
    if arch == "resnet":
        return _resolve_resnet_from_dict(model, layer_dict, prepend_backbone)
    if arch == "vit":
        return _resolve_vit_from_dict(model, layer_dict, prepend_backbone)
    if arch == "swin":
        return _resolve_swin_from_dict(model, layer_dict, prepend_backbone)
    if arch == "convnext":
        return _resolve_convnext_from_dict(model, layer_dict, prepend_backbone)
    _resolve_unknown_arch_fallback(model, arch)
    raise RuntimeError(
        "Unreachable: _resolve_unknown_arch_fallback should always raise"
    )


def resolve_target_layer(model: nn.Module, target_layer: Any) -> str:
    """
    Resolve the target_layer argument for Layer Grad-CAM.

    Supports 'auto', string paths, or dict-based specifications.

    Args:
        model: The model containing the target layer.
        target_layer: Target layer specification ('auto', str, or dict).

    Returns:
        The dot-separated path to the resolved target layer.

    Raises:
        ValueError: If the target_layer type is unsupported.
    """
    if isinstance(target_layer, str):
        if target_layer == "auto":
            resolved = resolve_auto_target_layer(model)
            if hasattr(model, "backbone") and not resolved.startswith("backbone."):
                return f"backbone.{resolved}"
            return resolved
        if hasattr(model, "backbone") and not target_layer.startswith("backbone."):
            return f"backbone.{target_layer}"
        return target_layer
    if isinstance(target_layer, dict):
        return resolve_target_layer_from_dict(model, target_layer)
    raise ValueError(
        f"[LayerGradCAMXAI] target_layer must be a string, dict, or 'auto'. "
        f"Got: {type(target_layer)}"
    )


@register_xai("layer_gradcam")
class LayerGradCAMXAI(BaseXAI):
    """
    Layer Grad-CAM XAI method using Captum.

    Computes attributions for a specific model layer using Grad-CAM.
    Supports auto-resolution and flexible specification of target layers.

    Attributes:
        model: The model to be explained.
        target_layer: Path to the target layer for Grad-CAM.
        layer: The resolved nn.Module for Grad-CAM.
        gradcam: Captum LayerGradCam object.
    """

    def __init__(self, model: Any, target_layer: Any, **kwargs: Any) -> None:
        """
        Initialize the LayerGradCAMXAI method.

        Args:
            model: The model to be explained.
            target_layer: Target layer specification ('auto', str, or dict).
            **kwargs: Additional parameters for the base class.
        """
        resolved_layer = resolve_target_layer(model, target_layer)
        print(f"[XAI-DEBUG] Resolved target layer: {resolved_layer}")
        super().__init__(model, target_layer=resolved_layer, **kwargs)
        self.target_layer = resolved_layer
        self.layer = resolve_layer(model, resolved_layer)
        print(f"[XAI-DEBUG] Target layer type: {type(self.layer)}")
        self.gradcam = LayerGradCam(self.model, self.layer)

    def explain(
        self, input_tensor: Tensor, target: Optional[int] = None, **kwargs: Any
    ) -> Tensor:
        """
        Generate Grad-CAM attributions for the given input and target.

        Args:
            input_tensor: Input tensor for which to compute attributions.
            target: Optional target class index for explanation.
            **kwargs: Additional parameters.

        Returns:
            Tensor of attributions with the same shape as input_tensor.
        """
        setattr(self.model, "_captum_tracing", True)
        try:

            def _hook_fn(module: nn.Module, inp: Any, out: Any) -> None:
                print(
                    f"[XAI-DEBUG] Target layer output shape: {getattr(out, 'shape', 'N/A')}"
                )
                handle.remove()

            handle = self.layer.register_forward_hook(_hook_fn)
            _ = self.model(input_tensor)
            attributions = self.gradcam.attribute(input_tensor, target=target)
        finally:
            if hasattr(self.model, "_captum_tracing"):
                delattr(self.model, "_captum_tracing")
        if hasattr(LayerAttribution, "interpolate"):
            if isinstance(attributions, tuple):
                attributions = attributions[0]
            interpolated = LayerAttribution.interpolate(
                attributions, input_tensor.shape[2:]
            )
            if isinstance(interpolated, tuple):
                attributions = interpolated[0]
            else:
                attributions = interpolated
        if isinstance(attributions, tuple):
            attributions = attributions[0]
        if not isinstance(attributions, Tensor):
            raise TypeError("Attributions must be a Tensor.")
        return attributions
