import pytest
import torch
from torch import nn

from refrakt_xai.utils.layer_detection import (
    _find_fallback_layer,
    _find_suitable_layers,
    _is_suitable_spatial_dims,
    _test_layer_spatial_dims,
    collect_conv_layers,
    find_layer_with_weights,
    resolve_layer_path,
    select_best_layer,
)


class DummyConvModel(nn.Module):
    """Dummy model with convolutional layers for testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DummyResNetLike(nn.Module):
    """Dummy ResNet-like model for testing."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.ModuleDict(
            {
                "conv1": nn.Conv2d(3, 64, 7, stride=2, padding=3),
                "layer1": nn.Conv2d(64, 64, 3, padding=1),
                "layer2": nn.Conv2d(64, 128, 3, stride=2, padding=1),
                "layer3": nn.Conv2d(128, 256, 3, stride=2, padding=1),
            }
        )
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.backbone["conv1"](x)
        x = self.backbone["layer1"](x)
        x = self.backbone["layer2"](x)
        x = self.backbone["layer3"](x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DummyLinearModel(nn.Module):
    """Dummy model with only linear layers for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.fc2(x)


def test_collect_conv_layers():
    """Test collecting convolutional layers from a model."""
    model = DummyConvModel()
    device = torch.device("cpu")

    conv_layers = collect_conv_layers(model, device)

    assert len(conv_layers) == 3  # conv1, conv2, conv3
    assert all(isinstance(layer[1], nn.Conv2d) for layer in conv_layers)
    assert conv_layers[0][0] == "conv1"
    assert conv_layers[1][0] == "conv2"
    assert conv_layers[2][0] == "conv3"


def test_collect_conv_layers_no_conv():
    """Test collecting conv layers from a model with no conv layers."""
    model = DummyLinearModel()
    device = torch.device("cpu")

    conv_layers = collect_conv_layers(model, device)

    assert len(conv_layers) == 0


def test_test_layer_spatial_dims():
    """Test testing layer spatial dimensions."""
    model = DummyConvModel()
    device = torch.device("cpu")

    # Test with a conv layer
    conv_layer = model.conv1
    spatial_dims = _test_layer_spatial_dims(conv_layer, "conv1", device)

    # Should return None for non-backbone.conv1.0 layers
    assert spatial_dims is None


def test_select_best_layer():
    """Test selecting the best layer from conv layers."""
    model = DummyConvModel()
    device = torch.device("cpu")

    conv_layers = collect_conv_layers(model, device)
    best_layer = select_best_layer(conv_layers)

    assert isinstance(best_layer, nn.Conv2d)
    assert best_layer in [layer[1] for layer in conv_layers]


def test_select_best_layer_empty():
    """Test selecting best layer with empty conv layers list."""
    with pytest.raises(ValueError, match="No convolutional layers found"):
        select_best_layer([])


def test_find_layer_with_weights():
    """Test finding a layer with weights."""
    model = DummyConvModel()

    layer = find_layer_with_weights(model)

    assert layer is not None
    assert hasattr(layer, "weight")


def test_find_layer_with_weights_no_weights():
    """Test finding layer with weights in a model without weights."""

    class NoWeightsModel(nn.Module):
        def forward(self, x):
            return x

    model = NoWeightsModel()

    layer = find_layer_with_weights(model)

    assert layer is None


def test_resolve_layer_path():
    """Test resolving layer paths."""
    model = DummyResNetLike()

    # Test simple layer path
    layer = resolve_layer_path(model, "fc")
    assert layer == model.fc

    # Test nested layer path
    layer = resolve_layer_path(model, "backbone.conv1")
    assert layer == model.backbone["conv1"]


def test_resolve_layer_path_invalid():
    """Test resolving invalid layer paths."""
    model = DummyResNetLike()

    with pytest.raises(ValueError, match="Cannot resolve layer path"):
        resolve_layer_path(model, "nonexistent_layer")


def test_is_suitable_spatial_dims():
    """Test checking if spatial dimensions are suitable."""
    # Test suitable dimensions (4 <= h,w <= 28)
    assert _is_suitable_spatial_dims((4, 4)) is True
    assert _is_suitable_spatial_dims((8, 8)) is True
    assert _is_suitable_spatial_dims((16, 16)) is True
    assert _is_suitable_spatial_dims((28, 28)) is True

    # Test unsuitable dimensions
    assert _is_suitable_spatial_dims((1, 1)) is False
    assert _is_suitable_spatial_dims((2, 2)) is False
    assert _is_suitable_spatial_dims((3, 3)) is False
    assert _is_suitable_spatial_dims((29, 29)) is False


def test_find_suitable_layers():
    """Test finding suitable layers."""
    model = DummyConvModel()
    device = torch.device("cpu")

    conv_layers = collect_conv_layers(model, device)
    suitable_layers = _find_suitable_layers(conv_layers)

    # No layers should be suitable since they don't have spatial dims
    # (only backbone.conv1.0 gets spatial dims in _test_layer_spatial_dims)
    assert len(suitable_layers) == 0


def test_find_fallback_layer():
    """Test finding fallback layer."""
    model = DummyConvModel()
    device = torch.device("cpu")

    conv_layers = collect_conv_layers(model, device)
    fallback_layer = _find_fallback_layer(conv_layers)

    # Should be None since no layers have spatial dims
    assert fallback_layer is None


def test_find_fallback_layer_empty():
    """Test finding fallback layer with empty list."""
    fallback_layer = _find_fallback_layer([])

    assert fallback_layer is None


def test_layer_detection_smoke():
    """Smoke test for the entire layer detection workflow."""
    model = DummyResNetLike()
    device = torch.device("cpu")

    # Collect conv layers
    conv_layers = collect_conv_layers(model, device)
    assert len(conv_layers) > 0

    # Select best layer
    best_layer = select_best_layer(conv_layers)
    assert isinstance(best_layer, nn.Conv2d)

    # Find layer with weights
    weight_layer = find_layer_with_weights(model)
    assert weight_layer is not None

    # Resolve layer path
    resolved_layer = resolve_layer_path(model, "backbone.conv1")
    assert resolved_layer == model.backbone["conv1"]
