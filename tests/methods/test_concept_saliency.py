import pytest
import torch

from refrakt_xai.methods.concept_saliency import ConceptSaliencyXAI


class DummyModel(torch.nn.Module):
    def get_latent(self, x):
        # Return a 1D feature vector per sample
        return x.view(-1)

    def forward(self, x):
        return x.view(x.size(0), -1).sum(dim=1)


def dummy_dataloader():
    for i in range(2):
        x = torch.randn(1, 3, 8, 8)
        # Use label=1 for first, label=0 for second to ensure both pos/neg latents are filled
        label = torch.tensor([1]) if i == 0 else torch.tensor([0])
        yield (x, label)


def test_conceptsaliencyxai_smoke():
    model = DummyModel()
    dataloader = dummy_dataloader()
    xai = ConceptSaliencyXAI(
        model, dataloader=dataloader, concept_pos_label=1, concept_neg_label=0
    )
    input_tensor = torch.randn(1, 3, 8, 8, requires_grad=True)
    # Patch concept_vector to match latent shape if needed
    latent = model.get_latent(input_tensor)
    if xai.concept_vector is not None and xai.concept_vector.shape != latent.shape:
        xai.concept_vector = torch.ones_like(latent)
    attributions = xai.explain(input_tensor)
    assert isinstance(attributions, torch.Tensor)
    assert attributions.shape == input_tensor.shape


def test_conceptsaliencyxai_no_dataloader():
    model = DummyModel()
    xai = ConceptSaliencyXAI(model)
    input_tensor = torch.randn(1, 3, 8, 8, requires_grad=True)
    with pytest.raises(ValueError):
        xai.explain(input_tensor)
