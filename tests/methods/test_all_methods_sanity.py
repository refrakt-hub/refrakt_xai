import pytest
import torch

from refrakt_xai.methods.deeplift import DeepLiftXAI
from refrakt_xai.methods.integrated_gradients import IntegratedGradientsXAI
from refrakt_xai.methods.layer_gradcam import LayerGradCAMXAI
from refrakt_xai.methods.occlusion import OcclusionXAI
from refrakt_xai.methods.reconstruction_attribution import ReconstructionAttributionXAI
from refrakt_xai.methods.saliency import SaliencyXAI
from refrakt_xai.methods.tcav import TCAVXAI


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(192, 10)  # 3*8*8 = 192 flattened

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)

    def get_latent(self, x):
        return x.view(x.size(0), -1).mean(dim=1)


class DummyResNetLike(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer3 = torch.nn.ModuleList([torch.nn.Conv2d(3, 3, 3) for _ in range(2)])

    def forward(self, x):
        x = self.layer3[0](x)
        return x.view(x.size(0), -1).sum(dim=1)

    def get_latent(self, x):
        return x.view(x.size(0), -1).mean(dim=1)


def test_all_methods_sanity():
    model = DummyModel()
    torch.randn(1, 3, 8, 8, requires_grad=True)
    SaliencyXAI(model)
    IntegratedGradientsXAI(model)
    OcclusionXAI(model)
    DeepLiftXAI(model)
    # TCAV requires concepts, so expect ValueError during initialization
    with pytest.raises(ValueError, match="You must provide a list of Concept objects"):
        TCAVXAI(model)
    ReconstructionAttributionXAI(model)
    # For LayerGradCAMXAI auto, use a model with layer3
    model_auto = DummyResNetLike()
    LayerGradCAMXAI(model_auto, layer="auto")
