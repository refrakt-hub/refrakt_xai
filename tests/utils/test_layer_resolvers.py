import pytest
import torch.nn as nn

from refrakt_xai.utils.layer_resolvers import (
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


def test_resnet_layer_resolver():
    class DummyResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer3 = nn.ModuleList([nn.Conv2d(3, 3, 3) for _ in range(2)])

    model = DummyResNet()
    assert _resolve_resnet_layer(model) == "layer3.1"


def test_convnext_layer_resolver():
    class DummyConvNeXt(nn.Module):
        def __init__(self):
            super().__init__()
            self.block3 = nn.Conv2d(3, 3, 3)

    model = DummyConvNeXt()
    assert _resolve_convnext_layer(model) == "block3"


def test_vit_layer_resolver():
    class DummyViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(3, 3) for _ in range(2)])

    model = DummyViT()
    assert _resolve_vit_layer(model) == "blocks.1"


def test_swin_layer_resolver():
    class DummySwin(nn.Module):
        def __init__(self):
            super().__init__()
            self.stage4 = nn.Conv2d(3, 3, 3)

    model = DummySwin()
    assert _resolve_swin_layer(model) == "stage4"


def test_resnet_from_dict():
    class DummyResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer3 = nn.ModuleList([nn.Module() for _ in range(2)])
            self.layer3[1].conv2 = nn.ModuleList([nn.Conv2d(3, 3, 3)])

    model = DummyResNet()
    layer_dict = {"block": 3, "index": 1, "conv": 0}
    assert _resolve_resnet_from_dict(model, layer_dict, False) == "layer3.1.conv2.0"


def test_unknown_arch_fallback_raises():
    class Dummy(nn.Module):
        pass

    model = Dummy()
    with pytest.raises(ValueError):
        _resolve_unknown_arch_fallback(model, "unknown")
