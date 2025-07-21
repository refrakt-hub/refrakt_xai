from typing import Any, Optional, Tuple, Union
from torch import Tensor
from captum.attr import Occlusion
from refrakt_xai.registry import register_xai
from refrakt_xai.base import BaseXAI

@register_xai("occlusion")
class OcclusionXAI(BaseXAI):
    def __init__(
        self,
        model: Any,
        sliding_window_shapes: Union[Tuple[int, int, int], Tuple[int, ...]] = (3, 15, 15),
        strides: Union[Tuple[int, int, int], Tuple[int, ...]] = (3, 8, 8),
        baselines: Union[int, float] = 0,
        **kwargs: Any
    ) -> None:
        super().__init__(
            model,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            baselines=baselines,
            **kwargs
        )
        self.occlusion = Occlusion(self.model)
        self.sliding_window_shapes = sliding_window_shapes
        self.strides = strides
        self.baselines = baselines

    def explain(
        self,
        input_tensor: Tensor,
        target: Optional[int] = None,
        **kwargs: Any
    ) -> Tensor:
        # Infer sliding_window_shapes and strides if not provided
        sliding_window_shapes = kwargs.get("sliding_window_shapes", self.sliding_window_shapes)
        strides = kwargs.get("strides", self.strides)
        baselines = kwargs.get("baselines", self.baselines)

        # If user did not override, infer from input
        if sliding_window_shapes is None or sliding_window_shapes == (3, 15, 15):
            c, h, w = input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]
            window_h = min(7, h)
            window_w = min(7, w)
            sliding_window_shapes = (c, window_h, window_w)
        if strides is None or strides == (3, 8, 8):
            stride_h = max(1, sliding_window_shapes[1] // 2)
            stride_w = max(1, sliding_window_shapes[2] // 2)
            strides = (sliding_window_shapes[0], stride_h, stride_w)

        # Captum workaround: set _captum_tracing flag on model
        setattr(self.model, '_captum_tracing', True)
        try:
            attributions: Tensor = self.occlusion.attribute(
                input_tensor,
                target=target,
                sliding_window_shapes=sliding_window_shapes,
                strides=strides,
                baselines=baselines,
            )
        finally:
            if hasattr(self.model, '_captum_tracing'):
                delattr(self.model, '_captum_tracing')
        return attributions 