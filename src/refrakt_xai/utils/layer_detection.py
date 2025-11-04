"""
Layer detection utilities for Grad-CAM and related XAI methods.

This module provides helper functions for detecting and selecting appropriate
layers in neural networks for use in Grad-CAM and similar explainability techniques.
"""

from typing import Any, List, Optional, Tuple

import torch
from torch import nn


def collect_conv_layers(
    model: nn.Module, device: torch.device
) -> List[Tuple[str, nn.Module, Optional[Tuple[int, int]]]]:
    """
    Collect all convolutional layers with their spatial dimensions.

    Args:
        model: The model to analyze.
        device: Device to create test inputs on.

    Returns:
        List of tuples containing (layer_name, layer_module, spatial_dims).
    """
    conv_layers: List[Tuple[str, nn.Module, Optional[Tuple[int, int]]]] = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            spatial_dims = _test_layer_spatial_dims(module, name, device)
            conv_layers.append((name, module, spatial_dims))

    return conv_layers


def _test_layer_spatial_dims(
    module: nn.Module, name: str, device: torch.device
) -> Optional[Tuple[int, int]]:
    """
    Test a layer to get its spatial dimensions.

    Args:
        module: The layer module to test.
        name: Name of the layer.
        device: Device to create test input on.

    Returns:
        Spatial dimensions (H, W) if testable, None otherwise.
    """
    try:
        with torch.no_grad():
            # For layers that aren't the first conv layer, skip testing
            if name != "backbone.conv1.0":
                return None

            # For the first conv layer, test it directly
            dummy_input = torch.randn(1, 1, 28, 28, device=device)
            output = module(dummy_input)
            spatial_dims = output.shape[2:]  # H, W
            if len(spatial_dims) == 2:
                return (int(spatial_dims[0]), int(spatial_dims[1]))
            return None
    except Exception:
        return None


def select_best_layer(
    conv_layers: List[Tuple[str, nn.Module, Optional[Tuple[int, int]]]],
) -> nn.Module:
    """
    Select the best layer based on spatial dimensions.

    Args:
        conv_layers: List of (name, module, spatial_dims) tuples.

    Returns:
        The selected layer module.

    Raises:
        ValueError: If no suitable layer is found.
    """
    if not conv_layers:
        raise ValueError("No convolutional layers found")

    suitable_layers = _find_suitable_layers(conv_layers)

    if suitable_layers:
        suitable_layers.sort(key=lambda x: x[2][0] * x[2][1], reverse=True)
        return suitable_layers[0][1]

    fallback_layer = _find_fallback_layer(conv_layers)
    if fallback_layer:
        return fallback_layer

    return conv_layers[0][1]


def _find_suitable_layers(
    conv_layers: List[Tuple[str, nn.Module, Optional[Tuple[int, int]]]],
) -> List[Tuple[str, nn.Module, Tuple[int, int]]]:
    """
    Find layers with suitable spatial dimensions.

    Args:
        conv_layers: List of (name, module, spatial_dims) tuples.

    Returns:
        List of suitable layers with spatial dimensions.
    """
    suitable_layers = []
    for name, module, spatial_dims in conv_layers:
        if spatial_dims is not None and _is_suitable_spatial_dims(spatial_dims):
            suitable_layers.append((name, module, spatial_dims))
    return suitable_layers


def _is_suitable_spatial_dims(spatial_dims: Tuple[int, int]) -> bool:
    """
    Check if spatial dimensions are suitable for visualization.

    Args:
        spatial_dims: Tuple of (height, width).

    Returns:
        True if dimensions are suitable, False otherwise.
    """
    h, w = spatial_dims
    return 4 <= h <= 28 and 4 <= w <= 28


def _find_fallback_layer(
    conv_layers: List[Tuple[str, nn.Module, Optional[Tuple[int, int]]]],
) -> Optional[nn.Module]:
    """
    Find first layer with spatial dimensions as fallback.

    Args:
        conv_layers: List of (name, module, spatial_dims) tuples.

    Returns:
        First layer with spatial dimensions, or None.
    """
    for _name, module, spatial_dims in conv_layers:
        if spatial_dims is not None:
            return module
    return None


def find_layer_with_weights(model: nn.Module) -> Optional[nn.Module]:
    """
    Find any layer with weights as a fallback.

    Args:
        model: The model to search.

    Returns:
        A layer with weights, or None if not found.
    """
    for _name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            if isinstance(module, nn.Module):
                return module
    return None


def resolve_layer_path(model: nn.Module, layer_path: str) -> nn.Module:
    """
    Resolve a string layer path to an actual layer module.

    Args:
        model: The model to traverse.
        layer_path: String path like 'layer3.1.conv2'.

    Returns:
        The resolved layer module.

    Raises:
        ValueError: If the layer path cannot be resolved.
    """
    current_module = model

    for part in layer_path.split("."):
        current_module = _resolve_path_part(current_module, part, layer_path)

    if not isinstance(current_module, nn.Module):
        raise ValueError(f"Resolved path '{layer_path}' does not point to a Module")

    return current_module


def _resolve_path_part(current_module: Any, part: str, layer_path: str) -> Any:
    """
    Resolve a single path part.

    Args:
        current_module: Current module being traversed.
        part: Current path part to resolve.
        layer_path: Full layer path for error messages.

    Returns:
        Resolved module for this part.

    Raises:
        ValueError: If the path part cannot be resolved.
    """
    if hasattr(current_module, part):
        return getattr(current_module, part)

    # Try to convert to int for indexed access
    try:
        idx = int(part)
        if hasattr(current_module, "__getitem__"):
            return current_module[idx]  # type: ignore
        else:
            raise ValueError(f"Cannot access index {idx} on {type(current_module)}")
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"Cannot resolve layer path '{layer_path}' at part '{part}'"
        ) from exc
