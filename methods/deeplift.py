import torch
import torch.nn as nn
from captum.attr import DeepLift
from refrakt_xai.registry import register_xai
from refrakt_xai.base import BaseXAI

class DeepLiftSumWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        out = self.model(x)
        if hasattr(out, "recon"):
            recon = out.recon
        elif hasattr(out, "reconstruction"):
            recon = out.reconstruction
        elif isinstance(out, dict) and "recon" in out:
            recon = out["recon"]
        elif isinstance(out, torch.Tensor):
            recon = out
        else:
            raise TypeError("Model output must be a Tensor, a dict with 'recon', have a 'recon' or 'reconstruction' attribute.")
        return recon.view(x.size(0), -1).sum(dim=1)

@register_xai("deeplift")
class DeepLiftXAI(BaseXAI):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.sum_wrapper = DeepLiftSumWrapper(model)

    def explain(self, input_tensor, target=None, **kwargs):
        baseline = kwargs.get("baseline", torch.zeros_like(input_tensor))
        additional_forward_args = kwargs.get("additional_forward_args", None)
        abs_val = kwargs.get("abs", True)
        dl = DeepLift(self.sum_wrapper)
        attributions = dl.attribute(input_tensor, baselines=baseline, additional_forward_args=additional_forward_args)
        if abs_val:
            attributions = attributions.abs()
        return attributions 