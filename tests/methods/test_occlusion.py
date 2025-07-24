import pytest
import torch

from refrakt_xai.methods.occlusion import OcclusionXAI


class DummyModel(torch.nn.Module):
    def forward(self, x):
        # Return a scalar per sample (batch)
        return x.view(x.size(0), -1).sum(dim=1)


def test_occlusionxai_smoke():
    model = DummyModel()
    xai = OcclusionXAI(model)
    input_tensor = torch.randn(2, 3, 8, 8, requires_grad=True)
    attributions = xai.explain(input_tensor)
    # Only check non-batch dimensions
    assert attributions.shape[1:] == input_tensor.shape[1:]


def test_occlusionxai_window_override():
    model = DummyModel()
    xai = OcclusionXAI(model)
    input_tensor = torch.randn(1, 3, 8, 8, requires_grad=True)
    attributions = xai.explain(
        input_tensor, sliding_window_shapes=(3, 3, 3), strides=(3, 3, 3)
    )
    assert attributions.shape[1:] == input_tensor.shape[1:]


def test_occlusionxai_invalid_model():
    class BadModel:
        def forward(self, x):
            return x

    xai = OcclusionXAI(BadModel())
    input_tensor = torch.randn(1, 3, 8, 8, requires_grad=True)
    with pytest.raises(Exception):
        xai.explain(input_tensor)
