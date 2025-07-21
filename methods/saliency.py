from typing import Any, Optional
from torch import Tensor
from captum.attr import Saliency
from refrakt_xai.registry import register_xai
from refrakt_xai.base import BaseXAI

@register_xai("saliency")
class SaliencyXAI(BaseXAI):
    def __init__(self, model: Any, abs: bool = True, **kwargs: Any) -> None:
        super().__init__(model, abs=abs, **kwargs)
        self.saliency = Saliency(self.model)
        self.abs = abs

    def explain(
        self, input_tensor: Tensor, target: Optional[int] = None, **kwargs: Any
    ) -> Tensor:
        abs_val: bool = kwargs.get("abs", self.abs)
        # Captum workaround: set _captum_tracing flag on model
        setattr(self.model, '_captum_tracing', True)
        try:
            attributions: Tensor = self.saliency.attribute(input_tensor, target=target, abs=abs_val)
        finally:
            if hasattr(self.model, '_captum_tracing'):
                delattr(self.model, '_captum_tracing')
        return attributions 