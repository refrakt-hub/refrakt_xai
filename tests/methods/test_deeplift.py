import pytest
import torch

from refrakt_xai.methods.deeplift import DeepLiftXAI


class DummyModel(torch.nn.Module):
    def forward(self, x):
        # Return a scalar per sample (like a classification model)
        return x.view(x.size(0), -1).sum(dim=1)


def test_deepliftxai_smoke():
    model = DummyModel()
    xai = DeepLiftXAI(model)
    input_tensor = torch.randn(2, 3, 8, 8, requires_grad=True)
    attributions = xai.explain(input_tensor)  # No target needed for 1D output
    assert isinstance(attributions, torch.Tensor)
    assert attributions.shape == input_tensor.shape


def test_deepliftxai_baseline():
    model = DummyModel()
    xai = DeepLiftXAI(model)
    input_tensor = torch.randn(1, 3, 4, 4, requires_grad=True)
    baseline = torch.zeros_like(input_tensor)
    attributions = xai.explain(input_tensor, baseline=baseline)  # No target needed
    assert attributions.shape == input_tensor.shape


def test_deepliftxai_invalid_model():
    class BadModel:
        def forward(self, x):
            return x

    xai = DeepLiftXAI(BadModel())
    input_tensor = torch.randn(1, 3, 4, 4, requires_grad=True)
    with pytest.raises(Exception):
        xai.explain(input_tensor)
