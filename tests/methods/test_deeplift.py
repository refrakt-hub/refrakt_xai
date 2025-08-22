import pytest
import torch

from refrakt_xai.methods.deeplift import DeepLiftXAI


class DummyClassificationModel(torch.nn.Module):
    """Mock classification model that returns tensor output directly"""
    def forward(self, x):
        # Return logits for classification (batch_size, num_classes)
        batch_size = x.size(0)
        return torch.randn(batch_size, 10)  # 10 classes


class DummyModelOutput:
    """Mock ModelOutput for classification"""
    def __init__(self, logits):
        self.logits = logits
        self.reconstruction = None


class DummyWrappedModel(torch.nn.Module):
    """Mock wrapped model that returns ModelOutput with logits"""
    def forward(self, x):
        batch_size = x.size(0)
        logits = torch.randn(batch_size, 10)
        return DummyModelOutput(logits)


def test_deepliftxai_smoke():
    model = DummyClassificationModel()
    xai = DeepLiftXAI(model)
    input_tensor = torch.randn(2, 3, 8, 8, requires_grad=True)
    attributions = xai.explain(input_tensor, target=0)  # Provide target for classification
    assert isinstance(attributions, torch.Tensor)
    assert attributions.shape == input_tensor.shape


def test_deepliftxai_wrapped_model():
    model = DummyWrappedModel()
    xai = DeepLiftXAI(model)
    input_tensor = torch.randn(1, 3, 4, 4, requires_grad=True)
    attributions = xai.explain(input_tensor)  # Should auto-detect target
    assert attributions.shape == input_tensor.shape


def test_deepliftxai_invalid_model():
    class BadModel:
        def forward(self, x):
            return x

    xai = DeepLiftXAI(BadModel())
    input_tensor = torch.randn(1, 3, 4, 4, requires_grad=True)
    with pytest.raises(Exception):
        xai.explain(input_tensor)
