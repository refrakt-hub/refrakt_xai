import torch

from refrakt_xai.utils.concept_utils import (
    append_latents_by_index,
    append_latents_by_label,
    extract_label_from_batch,
)


class DummyModel:
    def get_latent(self, x):
        return x * 2


def test_extract_label_from_tuple():
    batch = (torch.randn(2, 3), torch.tensor([1, 0]))
    label = extract_label_from_batch(batch)
    assert (label == torch.tensor([1, 0])).all()


def test_extract_label_from_dict():
    batch = {"data": torch.randn(2, 3), "label": torch.tensor([1, 0])}
    label = extract_label_from_batch(batch)
    assert (label == torch.tensor([1, 0])).all()


def test_append_latents_by_label():
    model = DummyModel()
    x = torch.randn(2, 3)
    label = torch.tensor([1, 0])
    pos_latents = []
    neg_latents = []
    append_latents_by_label(model, x, label, 1, 0, pos_latents, neg_latents)
    assert len(pos_latents) == 1
    assert len(neg_latents) == 1


def test_append_latents_by_index():
    model = DummyModel()
    x = torch.randn(2, 3)
    pos_latents = []
    neg_latents = []
    append_latents_by_index(model, x, 0, [0], [1], pos_latents, neg_latents)
    assert len(pos_latents) == 1
    append_latents_by_index(model, x, 1, [0], [1], pos_latents, neg_latents)
    assert len(neg_latents) == 1
