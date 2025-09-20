"""
Layer resolving utilities for Grad-CAM and related XAI methods.

This module provides helper functions to resolve target layers in various model
architectures (e.g., ResNet, ConvNeXt, ViT, Swin) for use in Grad-CAM and
similar explainability techniques.

Typical usage:
    layer_path = _resolve_resnet_layer(model)
    layer_path = _resolve_resnet_from_dict(model, layer_dict, prepend_backbone)
"""

from typing import Any, Optional

from torch import nn


def _resolve_resnet_layer(model: nn.Module) -> Optional[str]:
    """
    Resolve the last block of layer3 in a ResNet-like model.

    Args:
        model: The model to inspect.

    Returns:
        The string path to the last block in layer3, or None if not found.
    """
    if hasattr(model, "layer3"):
        layer3 = getattr(model, "layer3")
        if hasattr(layer3, "__len__") and hasattr(layer3, "__getitem__"):
            last_idx = len(layer3) - 1
            return f"layer3.{last_idx}"
        return "layer3"
    return None


def _resolve_convnext_layer(model: nn.Module) -> Optional[str]:
    """
    Resolve the block3 layer in a ConvNeXt-like model.

    Args:
        model: The model to inspect.

    Returns:
        The string path to block3, or None if not found.
    """
    if hasattr(model, "block3"):
        return "block3"
    return None


def _resolve_vit_layer(model: nn.Module) -> Optional[str]:
    """
    Resolve the last block in a ViT-like model.

    Args:
        model: The model to inspect.

    Returns:
        The string path to the last block in blocks, or None if not found.
    """
    if hasattr(model, "blocks"):
        blocks = getattr(model, "blocks")
        if hasattr(blocks, "__len__") and hasattr(blocks, "__getitem__"):
            last_idx = len(blocks) - 1
            return f"blocks.{last_idx}"
        return "blocks"
    return None


def _resolve_swin_layer(model: nn.Module) -> Optional[str]:
    """
    Resolve the stage4 layer in a Swin-like model.

    Args:
        model: The model to inspect.

    Returns:
        The string path to stage4, or None if not found.
    """
    if hasattr(model, "stage4"):
        return "stage4"
    return None


def _resolve_auto_target_layer_fallback(model: nn.Module) -> None:
    """
    Print available named modules for debugging when auto-resolving fails.

    Args:
        model: The model to inspect.
    """
    print(
        "[LayerGradCAMXAI][DEBUG] Could not auto-resolve target layer. "
        "Available named modules:"
    )
    for name, module in model.named_modules():
        print(f"  {name}: {module.__class__.__name__}")


def _resolve_resnet_from_dict(
    model: nn.Module,
    layer_dict: dict[str, Any],
    prepend_backbone: bool,
) -> str:
    """
    Resolve a ResNet target layer from a configuration dictionary.

    Args:
        model: The model to inspect.
        layer_dict: Dictionary specifying block, index, and conv.
        prepend_backbone: Whether to prepend 'backbone.' to the path.

    Returns:
        The string path to the resolved layer.

    Raises:
        ValueError: If the block or conv cannot be found.
    """
    block = layer_dict.get("block", 3)
    index = layer_dict.get("index", "last")
    conv = layer_dict.get("conv", "last")
    block_name = f"layer{block}"
    base = model.backbone if prepend_backbone and hasattr(model, "backbone") else model
    block_seq = getattr(base, block_name, None)
    if block_seq is None:
        raise ValueError(
            f"[LayerGradCAMXAI] Could not find block '{block_name}' in model."
        )
    block_idx = len(block_seq) - 1 if index == "last" else int(index)
    block_module = block_seq[block_idx]
    conv_seq = getattr(block_module, "conv2", None)
    if conv_seq is None:
        raise ValueError(
            f"[LayerGradCAMXAI] Could not find 'conv2' in block '{block_name}.{block_idx}'."
        )
    conv_idx = len(conv_seq) - 1 if conv == "last" else int(conv)
    path = f"{'backbone.' if prepend_backbone else ''}{block_name}.{block_idx}.conv2.{conv_idx}"
    return path


def _resolve_vit_from_dict(
    model: nn.Module,
    layer_dict: dict[str, Any],
    prepend_backbone: bool,
) -> str:
    """
    Resolve a ViT target layer from a configuration dictionary.

    Args:
        model: The model to inspect.
        layer_dict: Dictionary specifying block.
        prepend_backbone: Whether to prepend 'backbone.' to the path.

    Returns:
        The string path to the resolved layer.

    Raises:
        ValueError: If the blocks attribute cannot be found.
    """
    block = layer_dict.get("block", "last")
    base = model.backbone if prepend_backbone and hasattr(model, "backbone") else model
    blocks = getattr(base, "blocks", None)
    if blocks is None:
        raise ValueError("[LayerGradCAMXAI] Could not find 'blocks' in ViT model.")
    block_idx = len(blocks) - 1 if block == "last" else int(block)
    path = f"{'backbone.' if prepend_backbone else ''}blocks.{block_idx}"
    return path


def _resolve_swin_from_dict(
    model: nn.Module,
    layer_dict: dict[str, Any],
    prepend_backbone: bool,
) -> str:
    """
    Resolve a Swin target layer from a configuration dictionary.

    Args:
        model: The model to inspect.
        layer_dict: Dictionary specifying stage.
        prepend_backbone: Whether to prepend 'backbone.' to the path.

    Returns:
        The string path to the resolved layer.

    Raises:
        ValueError: If the stage attribute cannot be found.
    """
    stage = layer_dict.get("stage", "stage4")
    base = model.backbone if prepend_backbone and hasattr(model, "backbone") else model
    if not hasattr(base, stage):
        raise ValueError(
            f"[LayerGradCAMXAI] Could not find stage '{stage}' in Swin model."
        )
    path = f"{'backbone.' if prepend_backbone else ''}{stage}"
    return path


def _resolve_convnext_from_dict(
    model: nn.Module,
    layer_dict: dict[str, Any],
    prepend_backbone: bool,
) -> str:
    """
    Resolve a ConvNeXt target layer from a configuration dictionary.

    Args:
        model: The model to inspect.
        layer_dict: Dictionary specifying block.
        prepend_backbone: Whether to prepend 'backbone.' to the path.

    Returns:
        The string path to the resolved layer.

    Raises:
        ValueError: If the block attribute cannot be found.
    """
    block = layer_dict.get("block", "block3")
    base = model.backbone if prepend_backbone and hasattr(model, "backbone") else model
    if not hasattr(base, block):
        raise ValueError(
            f"[LayerGradCAMXAI] Could not find block '{block}' in ConvNeXt model."
        )
    path = f"{'backbone.' if prepend_backbone else ''}{block}"
    return path


def _resolve_unknown_arch_fallback(model: nn.Module, arch: str) -> None:
    """
    Print available named modules and raise ValueError for unknown architectures.

    Args:
        model: The model to inspect.
        arch: The unknown architecture name.

    Raises:
        ValueError: Always raised to indicate unknown architecture.
    """
    print(
        f"[LayerGradCAMXAI][DEBUG] Unknown architecture '{arch}'. "
        "Available named modules:"
    )
    for name, module in model.named_modules():
        print(f"  {name}: {module.__class__.__name__}")
    raise ValueError(
        f"[LayerGradCAMXAI] Unknown architecture '{arch}'. "
        "See above for available layers."
    )
