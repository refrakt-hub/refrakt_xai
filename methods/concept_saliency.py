import torch
from refrakt_xai.registry import register_xai
from refrakt_xai.base import BaseXAI

@register_xai("concept_saliency")
class ConceptSaliencyXAI(BaseXAI):
    def __init__(self, model, dataloader=None, concept_pos_indices=None, concept_neg_indices=None, concept_pos_label=None, concept_neg_label=None, device="cpu", dataset_name=None, **kwargs):
        super().__init__(model, **kwargs)
        self.device = device
        self.model = model
        self.dataloader = dataloader
        self.concept_vector = None
        self.dataset_name = dataset_name
        # Auto-select concept if not provided
        if concept_pos_indices is None and concept_neg_indices is None and concept_pos_label is None and concept_neg_label is None:
            self.concept_pos_label, self.concept_neg_label = self._default_concept_labels()
        else:
            self.concept_pos_label = concept_pos_label
            self.concept_neg_label = concept_neg_label
        self.concept_pos_indices = concept_pos_indices
        self.concept_neg_indices = concept_neg_indices
        # Compute concept vector
        if dataloader is not None:
            self.concept_vector = self._compute_concept_vector()
        else:
            self.concept_vector = None  # Will raise if explain is called without concept_vector

    def _default_concept_labels(self):
        # Auto-select concept labels for common datasets
        if self.dataset_name is not None:
            if "mnist" in self.dataset_name.lower():
                return 1, 0  # e.g., digit 1 vs digit 0
            if "cifar" in self.dataset_name.lower():
                return 1, 0  # e.g., class 1 vs class 0
        return 1, 0  # fallback

    def _compute_concept_vector(self):
        if self.dataloader is None:
            raise ValueError("Dataloader must be provided to compute concept vector.")
        pos_latents, neg_latents = [], []
        for i, batch in enumerate(self.dataloader):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(self.device)
            # Try to get label if available
            label = None
            if isinstance(batch, (tuple, list)) and len(batch) > 1:
                label = batch[1]
            elif isinstance(batch, dict):
                label = batch.get("label")
            # Select by label if possible
            if self.concept_pos_label is not None and label is not None:
                mask = (label == self.concept_pos_label)
                if mask.any():
                    pos_latents.append(self.model.get_latent(x[mask]).detach().cpu())
                mask = (label == self.concept_neg_label)
                if mask.any():
                    neg_latents.append(self.model.get_latent(x[mask]).detach().cpu())
            # Otherwise, select by index
            elif self.concept_pos_indices is not None and i in self.concept_pos_indices:
                pos_latents.append(self.model.get_latent(x).detach().cpu())
            elif self.concept_neg_indices is not None and i in self.concept_neg_indices:
                neg_latents.append(self.model.get_latent(x).detach().cpu())
        pos_mean = torch.cat(pos_latents).mean(dim=0)
        neg_mean = torch.cat(neg_latents).mean(dim=0)
        return (pos_mean - neg_mean).to(self.device)

    def explain(self, input_tensor, target=None, **kwargs):
        if self.concept_vector is None:
            raise ValueError("Concept vector not computed. Please provide a dataloader at initialization.")
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        latent = self.model.get_latent(input_tensor)
        # If latent is batched, handle batch
        if latent.ndim > 1:
            concept_score = torch.matmul(latent, self.concept_vector)
            score = concept_score.sum()
        else:
            score = torch.dot(latent, self.concept_vector)
        score.backward()
        saliency = input_tensor.grad.detach()
        return saliency 