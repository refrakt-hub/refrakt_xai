from typing import Any, Optional, Tuple
from torch import Tensor, nn
from captum.attr import LayerGradCam, LayerAttribution
from refrakt_xai.registry import register_xai
from refrakt_xai.base import BaseXAI


def resolve_layer(model: nn.Module, layer_path: str) -> nn.Module:
    """Resolve a dot-separated string path to a model layer."""
    parts = layer_path.split('.')
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
    Return the recommended default layer path for Layer Grad-CAM for custom Refrakt models.
    Now supports common wrappers (e.g., ResNetWrapper, ConvNeXtWrapper, ViTWrapper) by recursing into .backbone if present.
    Prints all available named modules if auto-resolution fails.
    """
    # If the model is a wrapper with a .backbone attribute, recurse into it
    if hasattr(model, "backbone") and isinstance(getattr(model, "backbone"), nn.Module):
        return resolve_auto_target_layer(getattr(model, "backbone"))
    # ResNet (custom)
    if hasattr(model, "layer3"):
        layer3 = getattr(model, "layer3")
        if hasattr(layer3, "__len__") and hasattr(layer3, "__getitem__"):
            last_idx = len(layer3) - 1
            return f"layer3.{last_idx}"
        return "layer3"
    # ConvNeXt (custom)
    if hasattr(model, "block3"):
        return "block3"
    # ViT (custom)
    if hasattr(model, "blocks"):
        blocks = getattr(model, "blocks")
        if hasattr(blocks, "__len__") and hasattr(blocks, "__getitem__"):
            last_idx = len(blocks) - 1
            return f"blocks.{last_idx}"
        return "blocks"
    # Swin (custom)
    if hasattr(model, "stage4"):
        return "stage4"
    # If we get here, print all available named modules for debugging
    print("[LayerGradCAMXAI][DEBUG] Could not auto-resolve target layer. Available named modules:")
    for name, module in model.named_modules():
        print(f"  {name}: {module.__class__.__name__}")
    raise ValueError("Auto target_layer not implemented for this model architecture. See above for available layers.")


def resolve_target_layer_from_dict(model: nn.Module, layer_dict: dict) -> str:
    """
    Resolve a dict-based target_layer specification to a string path for various architectures.
    Supports keys like block, index, conv (ResNet), stage (Swin), head/block (ViT), etc.
    Always prepends 'backbone.' unless prepend_backbone is False.
    """
    prepend_backbone = layer_dict.get('prepend_backbone', True)
    arch = layer_dict.get('arch', 'resnet').lower()  # Optionally allow explicit arch
    # --- ResNet ---
    if arch == 'resnet':
        block = layer_dict.get('block', 3)
        index = layer_dict.get('index', 'last')
        conv = layer_dict.get('conv', 'last')
        block_name = f"layer{block}"
        # Get block sequence
        base = model.backbone if prepend_backbone and hasattr(model, 'backbone') else model
        block_seq = getattr(base, block_name, None)
        if block_seq is None:
            raise ValueError(f"[LayerGradCAMXAI] Could not find block '{block_name}' in model.")
        block_idx = len(block_seq) - 1 if index == 'last' else int(index)
        block_module = block_seq[block_idx]
        # Get conv sequence (ResNet standard: conv2)
        conv_seq = getattr(block_module, 'conv2', None)
        if conv_seq is None:
            raise ValueError(f"[LayerGradCAMXAI] Could not find 'conv2' in block '{block_name}.{block_idx}'.")
        conv_idx = len(conv_seq) - 1 if conv == 'last' else int(conv)
        path = f"{'backbone.' if prepend_backbone else ''}{block_name}.{block_idx}.conv2.{conv_idx}"
        return path
    # --- ViT ---
    elif arch == 'vit':
        block = layer_dict.get('block', 'last')
        base = model.backbone if prepend_backbone and hasattr(model, 'backbone') else model
        blocks = getattr(base, 'blocks', None)
        if blocks is None:
            raise ValueError("[LayerGradCAMXAI] Could not find 'blocks' in ViT model.")
        block_idx = len(blocks) - 1 if block == 'last' else int(block)
        path = f"{'backbone.' if prepend_backbone else ''}blocks.{block_idx}"
        return path
    # --- Swin ---
    elif arch == 'swin':
        stage = layer_dict.get('stage', 'stage4')
        base = model.backbone if prepend_backbone and hasattr(model, 'backbone') else model
        if not hasattr(base, stage):
            raise ValueError(f"[LayerGradCAMXAI] Could not find stage '{stage}' in Swin model.")
        path = f"{'backbone.' if prepend_backbone else ''}{stage}"
        return path
    # --- ConvNeXt ---
    elif arch == 'convnext':
        block = layer_dict.get('block', 'block3')
        base = model.backbone if prepend_backbone and hasattr(model, 'backbone') else model
        if not hasattr(base, block):
            raise ValueError(f"[LayerGradCAMXAI] Could not find block '{block}' in ConvNeXt model.")
        path = f"{'backbone.' if prepend_backbone else ''}{block}"
        return path
    else:
        print(f"[LayerGradCAMXAI][DEBUG] Unknown architecture '{arch}'. Available named modules:")
        for name, module in model.named_modules():
            print(f"  {name}: {module.__class__.__name__}")
        raise ValueError(f"[LayerGradCAMXAI] Unknown architecture '{arch}'. See above for available layers.")


def resolve_target_layer(model: nn.Module, target_layer: Any) -> str:
    """
    Resolve the target_layer argument, which can be:
    - 'auto': use auto-resolver
    - str: use as path (prepend 'backbone.' if needed)
    - dict: use resolve_target_layer_from_dict
    """
    if isinstance(target_layer, str):
        if target_layer == 'auto':
            resolved = resolve_auto_target_layer(model)
            # If model has backbone and resolved does not start with 'backbone.', prepend it
            if hasattr(model, 'backbone') and not resolved.startswith('backbone.'):
                return f'backbone.{resolved}'
            return resolved
        # Patch: If model has backbone and path doesn't start with 'backbone.', prepend it
        if hasattr(model, 'backbone') and not target_layer.startswith('backbone.'):
            return f'backbone.{target_layer}'
        return target_layer
    elif isinstance(target_layer, dict):
        return resolve_target_layer_from_dict(model, target_layer)
    else:
        raise ValueError(f"[LayerGradCAMXAI] target_layer must be a string, dict, or 'auto'. Got: {type(target_layer)}")


@register_xai("layer_gradcam")
class LayerGradCAMXAI(BaseXAI):
    def __init__(self, model: Any, target_layer: Any, **kwargs: Any) -> None:
        # Support auto, string, or dict for target_layer
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
        # Captum workaround: set _captum_tracing flag on model
        setattr(self.model, '_captum_tracing', True)
        try:
            # Forward pass to get the output of the target layer
            def _hook_fn(module, inp, out):
                print(f"[XAI-DEBUG] Target layer output shape: {getattr(out, 'shape', 'N/A')}")
                # Remove hook after first call
                handle.remove()
            handle = self.layer.register_forward_hook(_hook_fn)
            # Run a forward pass to trigger the hook
            _ = self.model(input_tensor)
            # Now run GradCAM
            attributions = self.gradcam.attribute(input_tensor, target=target)
        finally:
            if hasattr(self.model, '_captum_tracing'):
                delattr(self.model, '_captum_tracing')
        # Optionally upsample to input size
        if hasattr(LayerAttribution, "interpolate"):
            # Ensure attributions is a Tensor before interpolation
            if isinstance(attributions, tuple):
                attributions = attributions[0]
            interpolated = LayerAttribution.interpolate(attributions, input_tensor.shape[2:])
            # If interpolate returns a tuple, take the first element
            if isinstance(interpolated, tuple):
                attributions = interpolated[0]
            else:
                attributions = interpolated
        # Ensure return type is Tensor
        if isinstance(attributions, tuple):
            attributions = attributions[0]
        return attributions 