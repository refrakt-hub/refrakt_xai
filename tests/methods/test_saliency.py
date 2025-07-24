import pytest
import torch

from refrakt_xai.methods.saliency import SaliencyXAI


class DummyModel(torch.nn.Module):
    def forward(self, x):
        # Return a scalar per sample (batch)
        return x.view(x.size(0), -1).sum(dim=1)


def test_saliencyxai_smoke():
    model = DummyModel()
    xai = SaliencyXAI(model)
    input_tensor = torch.randn(2, 3, 8, 8, requires_grad=True)
    attributions = xai.explain(input_tensor)
    assert isinstance(attributions, torch.Tensor)
    assert attributions.shape == input_tensor.shape


def test_saliencyxai_abs_flag():
    model = DummyModel()
    xai = SaliencyXAI(model, abs=False)
    input_tensor = torch.randn(1, 3, 4, 4, requires_grad=True)
    attributions = xai.explain(input_tensor, abs=True)
    assert attributions.shape == input_tensor.shape


def test_saliencyxai_invalid_model():
    class BadModel:
        def forward(self, x):
            return x

    xai = SaliencyXAI(BadModel())
    input_tensor = torch.randn(1, 3, 4, 4, requires_grad=True)
    with pytest.raises(Exception):
        xai.explain(input_tensor)
