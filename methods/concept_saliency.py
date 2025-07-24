"""
Concept Saliency XAI method for refrakt_xai.

This module implements the Concept Saliency method, providing concept-based
attribution for model predictions. It supports both label-based and index-based
concept selection and registers the ConceptSaliencyXAI class for use in the XAI
registry.

Typical usage:
    xai = ConceptSaliencyXAI(model, dataloader, concept_pos_label=1,
                            concept_neg_label=0)
    attributions = xai.explain(input_tensor)
"""

from typing import Any, Optional, Sequence

import torch

# pylint: disable=import-error
from refrakt_xai.base import BaseXAI
from refrakt_xai.registry import register_xai
from refrakt_xai.utils.concept_utils import (
    append_latents_by_index,
    append_latents_by_label,
    extract_label_from_batch,
)


# pylint: disable=too-few-public-methods,too-many-instance-attributes,too-many-arguments,too-many-positional-arguments
@register_xai("concept_saliency")
class ConceptSaliencyXAI(BaseXAI):
    """
    Concept Saliency XAI method.

    Computes attributions by measuring the gradient of the concept score with
    respect to the input. Supports automatic or user-specified concept selection
    and vector computation.

    Attributes:
        model: The model to be explained.
        dataloader: Dataloader for computing concept vectors.
        concept_vector: The computed concept vector.
        concept_pos_label: Positive concept label.
        concept_neg_label: Negative concept label.
        concept_pos_indices: Indices for positive concept samples.
        concept_neg_indices: Indices for negative concept samples.
        device: Device for computation.
        dataset_name: Name of the dataset (optional, for auto concept selection).
    """

    def __init__(
        self,
        model: Any,
        dataloader: Optional[Any] = None,
        concept_pos_indices: Optional[Sequence[int]] = None,
        concept_neg_indices: Optional[Sequence[int]] = None,
        concept_pos_label: Optional[int] = None,
        concept_neg_label: Optional[int] = None,
        device: str = "cpu",
        dataset_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ConceptSaliencyXAI method.

        Args:
            model: The model to be explained.
            dataloader: Dataloader for computing concept vectors.
            concept_pos_indices: Indices for positive concept samples.
            concept_neg_indices: Indices for negative concept samples.
            concept_pos_label: Positive concept label.
            concept_neg_label: Negative concept label.
            device: Device for computation (default: 'cpu').
            dataset_name: Name of the dataset (optional, for auto concept selection).
            **kwargs: Additional parameters for the base class.
        """
        super().__init__(model, **kwargs)
        self.device = device
        self.model = model
        self.dataloader = dataloader
        self.concept_vector: Optional[torch.Tensor] = None
        self.dataset_name = dataset_name
        self.concept_pos_label: Optional[int] = None
        self.concept_neg_label: Optional[int] = None
        if (
            concept_pos_indices is None
            and concept_neg_indices is None
            and concept_pos_label is None
            and concept_neg_label is None
        ):
            self.concept_pos_label, self.concept_neg_label = (
                self._default_concept_labels()
            )
        else:
            self.concept_pos_label = concept_pos_label
            self.concept_neg_label = concept_neg_label
        self.concept_pos_indices = concept_pos_indices
        self.concept_neg_indices = concept_neg_indices
        if dataloader is not None:
            self.concept_vector = self._compute_concept_vector()
        else:
            self.concept_vector = None

    def _default_concept_labels(self) -> tuple[int, int]:
        """
        Auto-select default concept labels for common datasets.

        Returns:
            Tuple of (positive_label, negative_label).
        """
        if self.dataset_name is not None:
            if "mnist" in self.dataset_name.lower():
                return 1, 0
            if "cifar" in self.dataset_name.lower():
                return 1, 0
        return 1, 0

    def _compute_concept_vector(self) -> torch.Tensor:
        """
        Compute the concept vector from the dataloader.

        Returns:
            The computed concept vector as a torch.Tensor.

        Raises:
            ValueError: If dataloader is not provided.
        """
        if self.dataloader is None:
            raise ValueError("Dataloader must be provided to compute concept vector.")
        pos_latents: list[torch.Tensor] = []
        neg_latents: list[torch.Tensor] = []
        for i, batch in enumerate(self.dataloader):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(self.device)
            label = extract_label_from_batch(batch)
            append_latents_by_label(
                self.model,
                x,
                label,
                self.concept_pos_label,
                self.concept_neg_label,
                pos_latents,
                neg_latents,
            )
            append_latents_by_index(
                self.model,
                x,
                i,
                self.concept_pos_indices,
                self.concept_neg_indices,
                pos_latents,
                neg_latents,
            )
        pos_mean = torch.cat(pos_latents).mean(dim=0)
        neg_mean = torch.cat(neg_latents).mean(dim=0)
        return (pos_mean - neg_mean).to(self.device)

    def explain(
        self,
        input_tensor: torch.Tensor,
        target: Optional[int] = None,
        **kwargs: Any,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        Generate concept saliency attributions for the given input.

        Args:
            input_tensor: Input tensor for which to compute attributions.

        Returns:
            Tensor of attributions with the same shape as input_tensor.

        Raises:
            ValueError: If the concept vector is not computed.
            RuntimeError: If input_tensor.grad is None after backward.
        """
        if self.concept_vector is None:
            raise ValueError(
                "Concept vector not computed. Please provide a dataloader at "
                "initialization."
            )
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        latent = self.model.get_latent(input_tensor)
        if latent.ndim > 1:
            concept_score = torch.matmul(latent, self.concept_vector)
            score = concept_score.sum()
        else:
            score = torch.dot(latent, self.concept_vector)
        score.backward()  # type: ignore
        if input_tensor.grad is not None:
            saliency = input_tensor.grad.detach()
        else:
            raise RuntimeError("input_tensor.grad is None")
        return saliency
