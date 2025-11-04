"""
Concept saliency vector computation utilities.

This module provides helper functions for extracting labels and appending latent
representations for concept-based XAI methods. These utilities are used to
simplify and modularize the logic in ConceptSaliencyXAI and related classes.

Typical usage:
    label = extract_label_from_batch(batch)
    append_latents_by_label(
        model, x, label, pos_label, neg_label, pos_latents, neg_latents
    )
    append_latents_by_index(
        model, x, i, pos_indices, neg_indices, pos_latents, neg_latents
    )
"""

from typing import Any, Optional, Sequence

import torch


# pylint: disable=too-many-arguments,too-many-positional-arguments
def extract_label_from_batch(batch: Any) -> Optional[Any]:
    """
    Extract the label from a batch, supporting tuple/list and dict formats.

    Args:
        batch: The batch from a dataloader, which may be a tuple, list, or dict.

    Returns:
        The label if present, otherwise None.
    """
    if isinstance(batch, (tuple, list)) and len(batch) > 1:
        return batch[1]
    if isinstance(batch, dict):
        return batch.get("label")
    return None


def append_latents_by_label(
    model: Any,
    x: torch.Tensor,
    label: Any,
    pos_label: Optional[Any],
    neg_label: Optional[Any],
    pos_latents: list[torch.Tensor],
    neg_latents: list[torch.Tensor],
) -> None:
    """
    Append latent representations to pos_latents or neg_latents based on label match.

    Args:
        model: The model with a get_latent method.
        x: The input tensor.
        label: The label tensor or value.
        pos_label: The positive concept label.
        neg_label: The negative concept label.
        pos_latents: List to append positive latents to.
        neg_latents: List to append negative latents to.
    """
    if pos_label is not None and label is not None:
        mask = label == pos_label
        if mask.any():
            pos_latents.append(model.get_latent(x[mask]).detach().cpu())
        mask = label == neg_label
        if mask.any():
            neg_latents.append(model.get_latent(x[mask]).detach().cpu())


def append_latents_by_index(
    model: Any,
    x: torch.Tensor,
    i: int,
    pos_indices: Optional[Sequence[int]],
    neg_indices: Optional[Sequence[int]],
    pos_latents: list[torch.Tensor],
    neg_latents: list[torch.Tensor],
) -> None:
    """
    Append latent representations to pos_latents or neg_latents based on index match.

    Args:
        model: The model with a get_latent method.
        x: The input tensor.
        i: The batch index.
        pos_indices: Sequence of positive indices.
        neg_indices: Sequence of negative indices.
        pos_latents: List to append positive latents to.
        neg_latents: List to append negative latents to.
    """
    if pos_indices is not None and i in pos_indices:
        pos_latents.append(model.get_latent(x).detach().cpu())
    elif neg_indices is not None and i in neg_indices:
        neg_latents.append(model.get_latent(x).detach().cpu())
