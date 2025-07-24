import pytest
import torch

from refrakt_xai.methods.integrated_gradients import IntegratedGradientsXAI


class DummyModel(torch.nn.Module):
    def forward(self, x):
        # Return a scalar per sample (batch)
        return x.view(x.size(0), -1).sum(dim=1)


def test_igxai_smoke():
    model = DummyModel()
    xai = IntegratedGradientsXAI(model)
    input_tensor = torch.randn(2, 3, 8, 8, requires_grad=True)
    attributions = xai.explain(input_tensor)
    assert isinstance(attributions, torch.Tensor)
    assert attributions.shape == input_tensor.shape


def test_igxai_n_steps():
    model = DummyModel()
    xai = IntegratedGradientsXAI(model, n_steps=10)
    input_tensor = torch.randn(1, 3, 4, 4, requires_grad=True)
    attributions = xai.explain(input_tensor, n_steps=5)
    assert attributions.shape == input_tensor.shape


def test_igxai_invalid_model():
    class BadModel:
        def forward(self, x):
            return x

    xai = IntegratedGradientsXAI(BadModel())
    input_tensor = torch.randn(1, 3, 4, 4, requires_grad=True)
    with pytest.raises(Exception):
        xai.explain(input_tensor)
