from typing import Any, Optional, Tuple
from torch import Tensor
from captum.attr import IntegratedGradients
from refrakt_xai.registry import register_xai
from refrakt_xai.base import BaseXAI

@register_xai("reconstruction_attribution")
class ReconstructionAttributionXAI(BaseXAI):
    def __init__(self, model: Any, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.ig = IntegratedGradients(self._reconstruction_forward)

    def _reconstruction_forward(self, x: Tensor) -> Tensor:
        out = self.model(x)
        # Handle ModelOutput or similar wrappers
        if hasattr(out, "recon"):
            recon = out.recon
            if not isinstance(recon, Tensor):
                raise TypeError("'recon' attribute must be a Tensor.")
            return recon
        if hasattr(out, "reconstruction"):
            recon = out.reconstruction
            if isinstance(recon, Tensor):
                return recon
        if isinstance(out, dict) and "recon" in out:
            recon = out["recon"]
            if not isinstance(recon, Tensor):
                raise TypeError("'recon' in model output dict must be a Tensor.")
            return recon
        elif isinstance(out, Tensor):
            return out
        else:
            raise TypeError("Model output must be a Tensor, a dict with 'recon', have a 'recon' or 'reconstruction' attribute.")

    def explain(
        self, input_tensor: Tensor, target: Optional[Tuple[int, ...]] = None, **kwargs: Any
    ) -> Tensor:
        baseline = kwargs.get("baseline", 0 * input_tensor)
        additional_forward_args = kwargs.get("additional_forward_args", None)
        abs_val = kwargs.get("abs", True)

        if target is None:
            # Attribute the sum of the reconstruction
            def agg_forward(x):
                return self._reconstruction_forward(x).view(x.size(0), -1).sum(dim=1)
            ig = IntegratedGradients(agg_forward)
            attributions = ig.attribute(input_tensor, baselines=baseline, additional_forward_args=additional_forward_args)
        else:
            # Attribute a specific output index (e.g., pixel)
            def idx_forward(x):
                recon = self._reconstruction_forward(x)
                if recon.dim() == 2:
                    return recon[:, target[0]]
                elif recon.dim() == 4:
                    # (B, C, H, W)
                    b, c, h, w = target
                    return recon[:, c, h, w]
                else:
                    raise ValueError("Unsupported output shape for target attribution.")
            ig = IntegratedGradients(idx_forward)
            attributions = ig.attribute(input_tensor, baselines=baseline, additional_forward_args=additional_forward_args)

        if abs_val:
            attributions = attributions.abs()
        return attributions 