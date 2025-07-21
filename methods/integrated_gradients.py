from captum.attr import IntegratedGradients
from refrakt_xai.registry import register_xai
from refrakt_xai.base import BaseXAI

@register_xai("integrated_gradients")
class IntegratedGradientsXAI(BaseXAI):
    def __init__(self, model, n_steps=50, **kwargs):
        super().__init__(model, n_steps=n_steps, **kwargs)
        self.ig = IntegratedGradients(self.model)
        self.n_steps = n_steps

    def explain(self, input_tensor, target=None, **kwargs):
        n_steps = kwargs.get("n_steps", self.n_steps)
        # Captum workaround: set _captum_tracing flag on model
        setattr(self.model, '_captum_tracing', True)
        try:
            attributions = self.ig.attribute(input_tensor, target=target, n_steps=n_steps)
        finally:
            if hasattr(self.model, '_captum_tracing'):
                delattr(self.model, '_captum_tracing')
        return attributions 