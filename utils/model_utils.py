"""
Model utility functions for XAI methods.

This module provides helper functions for common model operations used across
various XAI methods, including model validation, device handling, and output processing.
"""

from typing import Any, Optional

import torch
from torch import Tensor


def validate_model_for_classification(
    model: Any, input_tensor: Tensor, method_name: str
) -> None:
    """
    Validate that a model is suitable for classification-based XAI methods.

    Args:
        model: The model to validate.
        input_tensor: Input tensor for testing.
        method_name: Name of the XAI method for error messages.

    Raises:
        ValueError: If the model is not suitable for classification.
    """
    try:
        with torch.no_grad():
            test_output = model(input_tensor[:1])  # Test with single sample
            output = extract_primary_tensor(test_output)

            # Check if output looks like embeddings rather than logits
            if _is_embedding_output(output):
                raise ValueError(
                    f"Model appears to output embeddings rather than logits. "
                    f"{method_name} requires classification models with logit outputs. "
                    f"Consider using 'saliency' or 'occlusion' methods instead."
                )
    except Exception as e:
        raise ValueError(
            f"Failed to validate model for {method_name}: {e}. "
            f"This method requires classification models with logit outputs."
        ) from e


def _is_embedding_output(output: Tensor) -> bool:
    """
    Check if output looks like embeddings rather than logits.

    Args:
        output: The model output tensor.

    Returns:
        True if output appears to be embeddings, False otherwise.
    """
    return (
        isinstance(output, torch.Tensor)
        and hasattr(output, "shape")
        and len(output.shape) > 0
        and hasattr(output.shape, "__getitem__")
        and output.shape[-1] > 1000  # Likely embeddings
    )


def setup_captum_tracing(model: Any) -> None:
    """
    Set up Captum tracing on a model.

    Args:
        model: The model to set up tracing for.
    """
    setattr(model, "_captum_tracing", True)


def cleanup_captum_tracing(model: Any) -> None:
    """
    Clean up Captum tracing from a model.

    Args:
        model: The model to clean up tracing from.
    """
    if hasattr(model, "_captum_tracing"):
        delattr(model, "_captum_tracing")


def extract_primary_tensor(model_output: Any) -> Tensor:
    """
    Extract the primary tensor from model output.

    Args:
        model_output: Output from a model (can be tensor, dict, or object).

    Returns:
        The primary tensor from the output.

    Raises:
        ValueError: If no tensor can be extracted.
    """
    if isinstance(model_output, Tensor):
        return model_output

    # Handle dict outputs
    if isinstance(model_output, dict):
        tensor = _extract_from_dict(model_output)
        if tensor is not None:
            return tensor

    # Handle object outputs with common attributes
    tensor = _extract_from_object(model_output)
    if tensor is not None:
        return tensor

    # Fallback: try to get the first tensor attribute
    tensor = _extract_fallback_tensor(model_output)
    if tensor is not None:
        return tensor

    raise ValueError("Cannot extract tensor from model output")


def _extract_from_dict(model_output: dict[str, Any]) -> Optional[Tensor]:
    """Extract tensor from dictionary output."""
    for key in ["logits", "recon", "embeddings", "image"]:
        if key in model_output:
            value = model_output[key]
            if isinstance(value, Tensor):
                return value
    return None


def _extract_from_object(model_output: Any) -> Optional[Tensor]:
    """Extract tensor from object output with common attributes."""
    common_attrs = ["logits", "embeddings", "image", "reconstruction", "recon"]
    for attr in common_attrs:
        if hasattr(model_output, attr):
            attr_value = getattr(model_output, attr)
            if isinstance(attr_value, Tensor):
                return attr_value
    return None


def _extract_fallback_tensor(model_output: Any) -> Optional[Tensor]:
    """Extract first tensor attribute as fallback."""
    for attr_name in dir(model_output):
        if not attr_name.startswith("_"):  # Skip private attributes
            try:
                attr_value = getattr(model_output, attr_name)
                if isinstance(attr_value, Tensor):
                    return attr_value
            except (AttributeError, TypeError):
                continue
    return None


def get_model_device(model: Any) -> torch.device:
    """
    Get the device of a model.

    Args:
        model: The model to get the device for.

    Returns:
        The device the model is on.
    """
    try:
        param_device = next(model.parameters()).device
        if isinstance(param_device, torch.device):
            return param_device
        else:
            return torch.device("cpu")
    except StopIteration:
        # If no parameters, return CPU device
        return torch.device("cpu")


def create_dummy_input(
    channels: int, height: int, width: int, device: torch.device
) -> Tensor:
    """
    Create a dummy input tensor for testing layers.

    Args:
        channels: Number of input channels.
        height: Height of the input.
        width: Width of the input.
        device: Device to create the tensor on.

    Returns:
        Dummy input tensor.
    """
    return torch.randn(1, channels, height, width, device=device)


def process_batch_attributions(attributions: Any, batch_size: int) -> Tensor:
    """
    Process attributions that might be returned as tuples.

    Args:
        attributions: Attributions from Captum method.
        batch_size: Expected batch size.

    Returns:
        Processed attribution tensor.

    Raises:
        ValueError: If attributions cannot be processed.
    """
    if isinstance(attributions, tuple):
        if len(attributions) > 0:
            attributions = attributions[0]
        else:
            raise ValueError("Captum method returned an empty tuple")

    if not isinstance(attributions, Tensor):
        raise ValueError("Attributions must be a tensor")

    return attributions
