import pytest
import torch
from torch import nn

from refrakt_xai.methods.layer_gradcam import LayerGradCAMXAI


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3)

    def forward(self, x):
        # Return a scalar per sample
        x = self.conv(x)
        return x.view(x.size(0), -1).sum(dim=1)


class DummyResNetLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer3 = nn.ModuleList([nn.Conv2d(3, 3, 3) for _ in range(2)])

    def forward(self, x):
        # Call all layers in layer3 so the forward hook is triggered
        for layer in self.layer3:
            x = layer(x)
        return x.view(x.size(0), -1).sum(dim=1)


def test_layergradcamxai_smoke():
    model = DummyModel()
    xai = LayerGradCAMXAI(model, layer="conv")
    input_tensor = torch.randn(2, 3, 8, 8, requires_grad=True)
    attributions = xai.explain(input_tensor)
    assert isinstance(attributions, torch.Tensor)
    assert attributions.shape[0] == input_tensor.shape[0]


def test_layergradcamxai_auto():
    model = DummyResNetLike()
    xai = LayerGradCAMXAI(model, layer="auto")
    input_tensor = torch.randn(1, 3, 8, 8, requires_grad=True)
    attributions = xai.explain(input_tensor)
    assert attributions.shape[0] == input_tensor.shape[0]


def test_layergradcamxai_invalid_model():
    class BadModel:
        def forward(self, x):
            return x

    # ValueError is expected for missing layers
    with pytest.raises(ValueError, match="Cannot resolve layer path"):
        LayerGradCAMXAI(BadModel(), layer="conv")
