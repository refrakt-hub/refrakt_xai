"""
Reconstruction Quality Attribution XAI method for refrakt_xai.

This module implements the Reconstruction Quality Attribution method, providing
quality-based attribution for autoencoder models. It registers the
QualityAttributionXAI class for use in the XAI registry.

Typical usage:
    xai = QualityAttributionXAI(model, quality_metric='ssim')
    attributions = xai.explain(input_tensor)
"""

from dataclasses import dataclass
from typing import Any, Optional, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from refrakt_xai.base import BaseXAI
from refrakt_xai.registry import register_xai


@register_xai("quality_attribution")
@dataclass
class QualityAttributionXAI(BaseXAI):
    """
    Reconstruction Quality Attribution XAI method for autoencoders.

    Computes attributions by identifying which input regions contribute
    most to reconstruction quality. Uses metrics like SSIM or perceptual
    similarity to measure quality changes.

    Attributes:
        model: The model to be explained.
        quality_metric: Quality metric to use ('ssim', 'psnr', 'mse').
        window_size: Window size for SSIM computation (default: 11).
        use_gradients: Whether to use gradient-based attribution (default: True).
    """

    quality_metric: Literal["ssim", "psnr", "mse"] = "ssim"
    window_size: int = 11
    use_gradients: bool = True

    def explain(
        self, input_tensor: Tensor, target: Optional[int] = None, **kwargs: Any
    ) -> Tensor:
        """
        Generate quality attributions for the given input.

        Args:
            input_tensor: Input tensor for which to compute attributions.
            target: Not used for quality attribution.
            **kwargs: Additional parameters.

        Returns:
            Tensor of attributions with the same shape as input_tensor.
        """
        if self.use_gradients:
            return self._gradient_based_attribution(input_tensor)
        else:
            return self._perturbation_based_attribution(input_tensor)

    def _gradient_based_attribution(self, input_tensor: Tensor) -> Tensor:
        """Compute attributions using gradients w.r.t. quality metric."""
        input_tensor.requires_grad_(True)

        with torch.enable_grad():
            # Get reconstruction
            reconstruction = self.model(input_tensor)

            # Extract reconstruction from model output
            if (
                hasattr(reconstruction, "__class__")
                and reconstruction.__class__.__name__ == "ModelOutput"
            ):
                # ModelOutput from refrakt_core.schema.model_output
                if reconstruction.reconstruction is not None:
                    recon = reconstruction.reconstruction
                else:
                    raise ValueError("ModelOutput has no reconstruction")
            elif isinstance(reconstruction, dict):
                # VAE output format: {"recon": recon, "mu": mu, "logvar": logvar}
                if "recon" in reconstruction:
                    recon = reconstruction["recon"]
                elif "reconstruction" in reconstruction:
                    recon = reconstruction["reconstruction"]
                else:
                    raise ValueError(
                        "Could not extract reconstruction from model output dict"
                    )
            elif hasattr(reconstruction, "recon") and reconstruction.recon is not None:
                recon = reconstruction.recon
            elif (
                hasattr(reconstruction, "reconstruction")
                and reconstruction.reconstruction is not None
            ):
                recon = reconstruction.reconstruction
            elif isinstance(reconstruction, Tensor):
                # Simple autoencoder returns just the reconstruction
                recon = reconstruction
            else:
                raise ValueError("Could not extract reconstruction from model output")

            # Compute quality metric
            if self.quality_metric == "ssim":
                quality_score = self._compute_ssim(recon, input_tensor)
            elif self.quality_metric == "psnr":
                quality_score = self._compute_psnr(recon, input_tensor)
            elif self.quality_metric == "mse":
                quality_score = -self._compute_mse(
                    recon, input_tensor
                )  # Negative because lower MSE is better
            else:
                raise ValueError(f"Unsupported quality metric: {self.quality_metric}")

            # Compute gradients
            quality_score.sum().backward()

            # Get gradients w.r.t. input
            if input_tensor.grad is not None:
                attributions = input_tensor.grad.clone()
                input_tensor.grad.zero_()
            else:
                attributions = torch.zeros_like(input_tensor)

            # Scale attributions to reasonable range
            if attributions.numel() > 0:
                current_std = attributions.std()
                if current_std > 0:
                    target_std = 0.15
                    scale_factor = target_std / current_std
                    attributions = attributions * scale_factor
                max_val = attributions.abs().max()
                if max_val > 1.0:
                    attributions = attributions / max_val

        return attributions

    def _perturbation_based_attribution(self, input_tensor: Tensor) -> Tensor:
        """Compute attributions using input perturbation."""
        with torch.no_grad():
            # Get baseline reconstruction and quality
            baseline_reconstruction = self.model(input_tensor)

            # Extract reconstruction from model output
            if (
                hasattr(baseline_reconstruction, "__class__")
                and baseline_reconstruction.__class__.__name__ == "ModelOutput"
            ):
                # ModelOutput from refrakt_core.schema.model_output
                if baseline_reconstruction.reconstruction is not None:
                    baseline_recon = baseline_reconstruction.reconstruction
                else:
                    raise ValueError("ModelOutput has no reconstruction")
            elif isinstance(baseline_reconstruction, dict):
                # VAE output format: {"recon": recon, "mu": mu, "logvar": logvar}
                if "recon" in baseline_reconstruction:
                    baseline_recon = baseline_reconstruction["recon"]
                elif "reconstruction" in baseline_reconstruction:
                    baseline_recon = baseline_reconstruction["reconstruction"]
                else:
                    raise ValueError(
                        "Could not extract reconstruction from model output dict"
                    )
            elif (
                hasattr(baseline_reconstruction, "recon")
                and baseline_reconstruction.recon is not None
            ):
                baseline_recon = baseline_reconstruction.recon
            elif (
                hasattr(baseline_reconstruction, "reconstruction")
                and baseline_reconstruction.reconstruction is not None
            ):
                baseline_recon = baseline_reconstruction.reconstruction
            elif isinstance(baseline_reconstruction, Tensor):
                # Simple autoencoder returns just the reconstruction
                baseline_recon = baseline_reconstruction
            else:
                raise ValueError("Could not extract reconstruction from model output")

            if self.quality_metric == "ssim":
                baseline_quality = self._compute_ssim(baseline_recon, input_tensor)
            elif self.quality_metric == "psnr":
                baseline_quality = self._compute_psnr(baseline_recon, input_tensor)
            elif self.quality_metric == "mse":
                baseline_quality = -self._compute_mse(baseline_recon, input_tensor)
            else:
                raise ValueError(f"Unsupported quality metric: {self.quality_metric}")

            # Compute attributions by perturbing each input element
            attributions = torch.zeros_like(input_tensor)
            perturbation_size = 0.01  # Small perturbation

            for i in range(input_tensor.shape[1]):
                for j in range(input_tensor.shape[2]):
                    for k in range(input_tensor.shape[3]):
                        # Create perturbed input
                        perturbed_input = input_tensor.clone()
                        perturbed_input[:, i, j, k] += perturbation_size

                        # Get perturbed reconstruction and quality
                        perturbed_reconstruction = self.model(perturbed_input)

                        # Extract reconstruction from model output
                        if (
                            hasattr(perturbed_reconstruction, "__class__")
                            and perturbed_reconstruction.__class__.__name__
                            == "ModelOutput"
                        ):
                            # ModelOutput from refrakt_core.schema.model_output
                            if perturbed_reconstruction.reconstruction is not None:
                                perturbed_recon = (
                                    perturbed_reconstruction.reconstruction
                                )
                            else:
                                continue
                        elif isinstance(perturbed_reconstruction, dict):
                            # VAE output format: {"recon": recon, "mu": mu, "logvar": logvar}
                            if "recon" in perturbed_reconstruction:
                                perturbed_recon = perturbed_reconstruction["recon"]
                            elif "reconstruction" in perturbed_reconstruction:
                                perturbed_recon = perturbed_reconstruction[
                                    "reconstruction"
                                ]
                            else:
                                continue
                        elif (
                            hasattr(perturbed_reconstruction, "recon")
                            and perturbed_reconstruction.recon is not None
                        ):
                            perturbed_recon = perturbed_reconstruction.recon
                        elif (
                            hasattr(perturbed_reconstruction, "reconstruction")
                            and perturbed_reconstruction.reconstruction is not None
                        ):
                            perturbed_recon = perturbed_reconstruction.reconstruction
                        elif isinstance(perturbed_reconstruction, Tensor):
                            # Simple autoencoder returns just the reconstruction
                            perturbed_recon = perturbed_reconstruction
                        else:
                            continue

                        if self.quality_metric == "ssim":
                            perturbed_quality = self._compute_ssim(
                                perturbed_recon, perturbed_input
                            )
                        elif self.quality_metric == "psnr":
                            perturbed_quality = self._compute_psnr(
                                perturbed_recon, perturbed_input
                            )
                        elif self.quality_metric == "mse":
                            perturbed_quality = -self._compute_mse(
                                perturbed_recon, perturbed_input
                            )
                        else:
                            continue

                        # Compute attribution as change in quality
                        attributions[:, i, j, k] = (
                            perturbed_quality - baseline_quality
                        ) / perturbation_size

            # Scale attributions to reasonable range
            if attributions.numel() > 0:
                current_std = attributions.std()
                if current_std > 0:
                    target_std = 0.15
                    scale_factor = target_std / current_std
                    attributions = attributions * scale_factor
                max_val = attributions.abs().max()
                if max_val > 1.0:
                    attributions = attributions / max_val

            return attributions

    def _compute_ssim(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute SSIM between two tensors."""
        # Ensure tensors have the same shape
        if x.shape != y.shape:
            raise ValueError(f"Tensor shapes must match: {x.shape} vs {y.shape}")

        # Handle different tensor dimensions
        if x.dim() == 1:
            # 1D tensors - use simple correlation
            x_norm = (x - x.mean()) / (x.std() + 1e-8)
            y_norm = (y - y.mean()) / (y.std() + 1e-8)
            correlation = (x_norm * y_norm).mean()
            return correlation
        elif x.dim() == 2:
            # 2D tensors - reshape to 2D spatial if possible
            if x.shape[1] == 784:  # MNIST-like flattened
                # Reshape to (batch, 1, 28, 28)
                x_reshaped = x.view(x.shape[0], 1, 28, 28)
                y_reshaped = y.view(y.shape[0], 1, 28, 28)
                return self._compute_ssim_2d(x_reshaped, y_reshaped)
            else:
                # Use simple correlation for other 2D cases
                x_norm = (x - x.mean()) / (x.std() + 1e-8)
                y_norm = (y - y.mean()) / (y.std() + 1e-8)
                correlation = (x_norm * y_norm).mean()
                return correlation
        elif x.dim() == 4:
            # 4D tensors - use 2D SSIM
            return self._compute_ssim_2d(x, y)
        else:
            # For other dimensions, use simple correlation
            x_norm = (x - x.mean()) / (x.std() + 1e-8)
            y_norm = (y - y.mean()) / (y.std() + 1e-8)
            correlation = (x_norm * y_norm).mean()
            return correlation

    def _compute_ssim_2d(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute SSIM for 2D spatial tensors."""
        # Ensure we have 4D tensors (batch, channel, height, width)
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D")

        # Simple SSIM implementation
        mu_x = F.avg_pool2d(
            x, self.window_size, stride=1, padding=self.window_size // 2
        )
        mu_y = F.avg_pool2d(
            y, self.window_size, stride=1, padding=self.window_size // 2
        )

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x_sq = (
            F.avg_pool2d(
                x * x, self.window_size, stride=1, padding=self.window_size // 2
            )
            - mu_x_sq
        )
        sigma_y_sq = (
            F.avg_pool2d(
                y * y, self.window_size, stride=1, padding=self.window_size // 2
            )
            - mu_y_sq
        )
        sigma_xy = (
            F.avg_pool2d(
                x * y, self.window_size, stride=1, padding=self.window_size // 2
            )
            - mu_xy
        )

        c1 = 0.01**2
        c2 = 0.03**2

        ssim = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        )
        return ssim.mean()

    def _compute_psnr(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute PSNR between two tensors."""
        mse = F.mse_loss(x, y)
        if mse == 0:
            return torch.tensor(float("inf"))
        max_val = torch.max(x).item()
        return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)

    def _compute_mse(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute MSE between two tensors."""
        return F.mse_loss(x, y)
