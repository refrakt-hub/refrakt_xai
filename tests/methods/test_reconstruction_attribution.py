import pytest
import torch

from refrakt_xai.methods.reconstruction_attribution import ReconstructionAttributionXAI


class DummyModel(torch.nn.Module):
    def forward(self, x):
        return x


def test_reconstructionattrxai_smoke():
    model = DummyModel()
    xai = ReconstructionAttributionXAI(model)
    input_tensor = torch.randn(2, 3, 8, 8, requires_grad=True)
    attributions = xai.explain(input_tensor)
    assert isinstance(attributions, torch.Tensor)
    assert attributions.shape == input_tensor.shape


def test_reconstructionattrxai_target():
    model = DummyModel()
    xai = ReconstructionAttributionXAI(model)
    input_tensor = torch.randn(1, 3, 8, 8, requires_grad=True)
    attributions = xai.explain(input_tensor, target=(0, 0, 0, 0))
    assert attributions.shape == input_tensor.shape


def test_reconstructionattrxai_invalid_model():
    class BadModel:
        def forward(self, x):
            return x

    xai = ReconstructionAttributionXAI(BadModel())
    input_tensor = torch.randn(1, 3, 8, 8, requires_grad=True)
    with pytest.raises(Exception):
        xai.explain(input_tensor)
