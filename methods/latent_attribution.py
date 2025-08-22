"""
Latent Attribution XAI method for refrakt_xai.

This module implements the Latent Attribution method, providing
latent space-based attribution for autoencoder models. It registers the
LatentAttributionXAI class for use in the XAI registry.

Typical usage:
    xai = LatentAttributionXAI(model, latent_dim=0)
    attributions = xai.explain(input_tensor)
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor

from refrakt_xai.base import BaseXAI
from refrakt_xai.registry import register_xai


@register_xai("latent_attribution")
@dataclass
class LatentAttributionXAI(BaseXAI):
    """
    Latent Attribution XAI method for autoencoders.

    Computes attributions by analyzing which input features contribute
    most to specific latent dimensions. Useful for understanding what
    the autoencoder learns in its bottleneck representation.

    Attributes:
        model: The model to be explained.
        latent_dim: The specific latent dimension to analyze (default: 0).
        use_gradients: Whether to use gradient-based attribution (default: True).
    """

    latent_dim: int = 0
    use_gradients: bool = True

    def explain(
        self, input_tensor: Tensor, target: Optional[int] = None, **kwargs: Any
    ) -> Tensor:
        """
        Generate latent attributions for the given input.

        Args:
            input_tensor: Input tensor for which to compute attributions.
            target: Optional target latent dimension (overrides latent_dim).
            **kwargs: Additional parameters.

        Returns:
            Tensor of attributions with the same shape as input_tensor.
        """
        target_dim = kwargs.get("latent_dim", self.latent_dim)
        
        if self.use_gradients:
            return self._gradient_based_attribution(input_tensor, target_dim)
        else:
            return self._perturbation_based_attribution(input_tensor, target_dim)

    def _gradient_based_attribution(self, input_tensor: Tensor, latent_dim: int) -> Tensor:
        """Compute attributions using gradients w.r.t. latent dimension."""
        input_tensor.requires_grad_(True)
        
        # Forward pass to get latent representation
        with torch.enable_grad():
            output = self.model(input_tensor)
            
            # Extract latent representation based on model output format
            if hasattr(output, '__class__') and output.__class__.__name__ == 'ModelOutput':
                # ModelOutput from refrakt_core.schema.model_output
                if output.embeddings is not None:
                    latent = output.embeddings
                elif output.reconstruction is not None:
                    # If no embeddings, use reconstruction as fallback
                    latent = output.reconstruction
                else:
                    raise ValueError("ModelOutput has no embeddings or reconstruction")
            elif isinstance(output, dict):
                # VAE output format: {"recon": recon, "mu": mu, "logvar": logvar}
                if "mu" in output:
                    latent = output["mu"]
                elif "latent" in output:
                    latent = output["latent"]
                elif "z" in output:
                    latent = output["z"]
                else:
                    raise ValueError("Could not extract latent representation from model output dict")
            elif hasattr(output, 'mu') and output.mu is not None:
                # ModelOutput object with mu attribute
                latent = output.mu
            elif hasattr(output, 'latent') and output.latent is not None:
                # ModelOutput object with latent attribute
                latent = output.latent
            elif hasattr(output, 'z') and output.z is not None:
                # ModelOutput object with z attribute
                latent = output.z
            elif hasattr(output, 'embeddings') and output.embeddings is not None:
                # ModelOutput object with embeddings attribute
                latent = output.embeddings
            elif isinstance(output, Tensor):
                # Simple autoencoder returns just the reconstruction
                # We need to get the latent from the encoder
                if hasattr(self.model, 'get_latent'):
                    latent = self.model.get_latent(input_tensor)
                elif hasattr(self.model, 'encode'):
                    latent = self.model.encode(input_tensor)
                else:
                    raise ValueError("Model output is a tensor but no encoder method found")
            else:
                raise ValueError(f"Unable to extract latent representation from model output: {type(output)}")
            
            # Handle tuple output (mu, logvar) from VAE encode method
            if isinstance(latent, tuple):
                latent = latent[0]  # Take mu from (mu, logvar)
            
            # Select the specific latent dimension
            if len(latent.shape) > 1:
                target_latent = latent[:, latent_dim]
            else:
                target_latent = latent
            
            # Compute gradients
            target_latent.sum().backward()
            
            # Get gradients w.r.t. input
            if input_tensor.grad is not None:
                attributions = input_tensor.grad.clone()
                input_tensor.grad.zero_()
            else:
                # Fallback if gradients are None
                attributions = torch.zeros_like(input_tensor)
            
            # Scale attributions to reasonable range
            # The issue is that gradients w.r.t. single latent dimension are very small
            # We need to scale them up to be comparable to other attribution methods
            if attributions.numel() > 0:
                # Compute the scale factor to bring values to reasonable range
                current_std = attributions.std()
                if current_std > 0:
                    # Scale to have std around 0.1-0.2 (similar to occlusion method)
                    target_std = 0.15
                    scale_factor = target_std / current_std
                    attributions = attributions * scale_factor
                
                # Also ensure values are in reasonable range [-1, 1]
                max_val = attributions.abs().max()
                if max_val > 1.0:
                    attributions = attributions / max_val
            
        return attributions

    def _perturbation_based_attribution(self, input_tensor: Tensor, latent_dim: int) -> Tensor:
        """Compute attributions using input perturbation."""
        with torch.no_grad():
            # Get baseline latent value
            baseline_output = self.model(input_tensor)
            
            # Extract latent representation based on model output format
            if hasattr(baseline_output, '__class__') and baseline_output.__class__.__name__ == 'ModelOutput':
                # ModelOutput from refrakt_core.schema.model_output
                if baseline_output.embeddings is not None:
                    baseline_latent = baseline_output.embeddings
                elif baseline_output.reconstruction is not None:
                    # If no embeddings, use reconstruction as fallback
                    baseline_latent = baseline_output.reconstruction
                else:
                    raise ValueError("ModelOutput has no embeddings or reconstruction")
            elif isinstance(baseline_output, dict):
                # VAE output format: {"recon": recon, "mu": mu, "logvar": logvar}
                if "mu" in baseline_output:
                    baseline_latent = baseline_output["mu"]
                elif "latent" in baseline_output:
                    baseline_latent = baseline_output["latent"]
                elif "z" in baseline_output:
                    baseline_latent = baseline_output["z"]
                else:
                    raise ValueError("Could not extract latent representation from model output dict")
            elif hasattr(baseline_output, 'mu') and baseline_output.mu is not None:
                # ModelOutput object with mu attribute
                baseline_latent = baseline_output.mu
            elif hasattr(baseline_output, 'latent') and baseline_output.latent is not None:
                # ModelOutput object with latent attribute
                baseline_latent = baseline_output.latent
            elif hasattr(baseline_output, 'z') and baseline_output.z is not None:
                # ModelOutput object with z attribute
                baseline_latent = baseline_output.z
            elif hasattr(baseline_output, 'embeddings') and baseline_output.embeddings is not None:
                # ModelOutput object with embeddings attribute
                baseline_latent = baseline_output.embeddings
            elif isinstance(baseline_output, Tensor):
                # Simple autoencoder returns just the reconstruction
                # We need to get the latent from the encoder
                if hasattr(self.model, 'get_latent'):
                    baseline_latent = self.model.get_latent(input_tensor)
                elif hasattr(self.model, 'encode'):
                    baseline_latent = self.model.encode(input_tensor)
                else:
                    raise ValueError("Model output is a tensor but no encoder method found")
            else:
                raise ValueError(f"Unable to extract latent representation from model output: {type(baseline_output)}")
            
            # Handle tuple output (mu, logvar) from VAE encode method
            if isinstance(baseline_latent, tuple):
                baseline_latent = baseline_latent[0]  # Take mu from (mu, logvar)
            
            if len(baseline_latent.shape) > 1:
                baseline_value = baseline_latent[:, latent_dim]
            else:
                baseline_value = baseline_latent
            
            # Compute attributions by perturbing each input element
            attributions = torch.zeros_like(input_tensor)
            perturbation_size = 0.01  # Small perturbation
            
            for i in range(input_tensor.shape[1]):
                for j in range(input_tensor.shape[2]):
                    for k in range(input_tensor.shape[3]):
                        # Create perturbed input
                        perturbed_input = input_tensor.clone()
                        perturbed_input[:, i, j, k] += perturbation_size
                        
                        # Get perturbed latent value
                        perturbed_output = self.model(perturbed_input)
                        
                        # Extract latent representation based on model output format
                        if hasattr(perturbed_output, '__class__') and perturbed_output.__class__.__name__ == 'ModelOutput':
                            # ModelOutput from refrakt_core.schema.model_output
                            if perturbed_output.embeddings is not None:
                                perturbed_latent = perturbed_output.embeddings
                            elif perturbed_output.reconstruction is not None:
                                # If no embeddings, use reconstruction as fallback
                                perturbed_latent = perturbed_output.reconstruction
                            else:
                                continue
                        elif isinstance(perturbed_output, dict):
                            # VAE output format: {"recon": recon, "mu": mu, "logvar": logvar}
                            if "mu" in perturbed_output:
                                perturbed_latent = perturbed_output["mu"]
                            elif "latent" in perturbed_output:
                                perturbed_latent = perturbed_output["latent"]
                            elif "z" in perturbed_output:
                                perturbed_latent = perturbed_output["z"]
                            else:
                                continue
                        elif hasattr(perturbed_output, 'mu') and perturbed_output.mu is not None:
                            # ModelOutput object with mu attribute
                            perturbed_latent = perturbed_output.mu
                        elif hasattr(perturbed_output, 'latent') and perturbed_output.latent is not None:
                            # ModelOutput object with latent attribute
                            perturbed_latent = perturbed_output.latent
                        elif hasattr(perturbed_output, 'z') and perturbed_output.z is not None:
                            # ModelOutput object with z attribute
                            perturbed_latent = perturbed_output.z
                        elif hasattr(perturbed_output, 'embeddings') and perturbed_output.embeddings is not None:
                            # ModelOutput object with embeddings attribute
                            perturbed_latent = perturbed_output.embeddings
                        elif isinstance(perturbed_output, Tensor):
                            # Simple autoencoder returns just the reconstruction
                            # We need to get the latent from the encoder
                            if hasattr(self.model, 'get_latent'):
                                perturbed_latent = self.model.get_latent(perturbed_input)
                            elif hasattr(self.model, 'encode'):
                                perturbed_latent = self.model.encode(perturbed_input)
                            else:
                                continue
                        else:
                            continue
                        
                        # Handle tuple output (mu, logvar) from VAE encode method
                        if isinstance(perturbed_latent, tuple):
                            perturbed_latent = perturbed_latent[0]  # Take mu from (mu, logvar)
                        
                        if len(perturbed_latent.shape) > 1:
                            perturbed_value = perturbed_latent[:, latent_dim]
                        else:
                            perturbed_value = perturbed_latent
                        
                        # Compute attribution as change in latent value
                        attributions[:, i, j, k] = (perturbed_value - baseline_value) / perturbation_size
            
            # Scale attributions to reasonable range
            if attributions.numel() > 0:
                # Compute the scale factor to bring values to reasonable range
                current_std = attributions.std()
                if current_std > 0:
                    # Scale to have std around 0.1-0.2 (similar to occlusion method)
                    target_std = 0.15
                    scale_factor = target_std / current_std
                    attributions = attributions * scale_factor
                
                # Also ensure values are in reasonable range [-1, 1]
                max_val = attributions.abs().max()
                if max_val > 1.0:
                    attributions = attributions / max_val
            
            return attributions 