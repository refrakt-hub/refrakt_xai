import pytest
import torch

from refrakt_xai.methods.tcav import TCAVXAI


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(192, 10)  # 3*8*8 = 192 flattened

    def get_latent(self, x):
        # Return a 1D feature vector per sample
        return x.view(-1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)


def dummy_dataloader():
    for i in range(2):
        x = torch.randn(1, 3, 8, 8)
        # Use label=1 for first, label=0 for second to ensure both pos/neg latents are filled
        label = torch.tensor([1]) if i == 0 else torch.tensor([0])
        yield (x, label)


def test_tcavxai_smoke():
    model = DummyModel()
    # TCAV requires concept objects, so expect ValueError during initialization
    with pytest.raises(ValueError, match="You must provide a list of Concept objects"):
        TCAVXAI(model)


def test_tcavxai_no_concepts():
    model = DummyModel()
    # TCAV requires concept objects, so expect ValueError during initialization
    with pytest.raises(ValueError, match="You must provide a list of Concept objects"):
        TCAVXAI(model)
