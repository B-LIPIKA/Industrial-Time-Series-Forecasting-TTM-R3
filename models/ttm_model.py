"""
Tiny Time Mixer (TTM-R3 Architecture) & Fine-Tuning Pipeline
PyTorch implementation of the Tiny Time Mixer architecture for industrial sensor time-series forecasting.
Supports fine-tuning, checkpoint loading, and inference.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict


class PatchEmbedding(nn.Module):
    """Slices 1D input into patches and projects into embedding dimension."""
    def __init__(self, patch_len: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, context_len)
        B, L = x.shape
        num_patches = L // self.patch_len
        # Reshape to (B, num_patches, patch_len)
        x_patched = x[:, :num_patches * self.patch_len].view(B, num_patches, self.patch_len)
        return self.proj(x_patched)  # (B, num_patches, d_model)


class TTMBlock(nn.Module):
    """Tiny Time Mixer Block: combines Intra-Patch and Inter-Patch Mixing."""
    def __init__(self, num_patches: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.intra_mix = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.inter_mix = nn.Sequential(
            nn.Linear(num_patches, num_patches * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_patches * 2, num_patches),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Intra-patch (feature-level) mixing
        x = x + self.intra_mix(self.norm1(x))
        # Inter-patch (temporal-level) mixing across patch tokens
        x_t = x.transpose(1, 2)  # (B, d_model, num_patches)
        x_mixed = self.inter_mix(x_t).transpose(1, 2)  # (B, num_patches, d_model)
        x = x + self.norm2(x_mixed)
        return x


class GraniteTTMR3Forecaster(nn.Module):
    """
    IBM Granite TTM-R3 Core Forecaster.
    Compact, ultra-fast pre-trained foundation mixer architecture for time-series forecasting.
    """
    def __init__(
        self,
        context_len: int = 192,
        horizon_len: int = 96,
        patch_len: int = 16,
        d_model: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.patch_len = patch_len
        self.num_patches = context_len // patch_len

        self.patch_embed = PatchEmbedding(patch_len=patch_len, d_model=d_model)
        self.blocks = nn.ModuleList([
            TTMBlock(num_patches=self.num_patches, d_model=d_model, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_patches * d_model, 128),
            nn.GELU(),
            nn.Linear(128, horizon_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Instance Normalization (RevIN style)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-5
        x_norm = (x - mean) / std

        # Patch & Mixing
        feat = self.patch_embed(x_norm)
        for block in self.blocks:
            feat = block(feat)

        out = self.head(feat)

        # De-normalization
        out = (out * std) + mean
        return out


class IndustrialTTMFineTuningPipeline:
    """End-to-End Fine-Tuning & Serving Wrapper for TTM-R3 Forecaster."""
    def __init__(
        self,
        context_len: int = 192,
        horizon_len: int = 96,
        device: Optional[str] = None
    ):
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GraniteTTMR3Forecaster(
            context_len=context_len,
            horizon_len=horizon_len
        ).to(self.device)

    def fine_tune(
        self,
        series: np.ndarray,
        epochs: int = 8,
        batch_size: int = 32,
        lr: float = 1e-3
    ) -> Dict[str, list]:
        """Fine-tunes the mixer backbone on chronological sensor windows."""
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.SmoothL1Loss()

        # Build sliding windows
        X, Y = [], []
        window_size = self.context_len + self.horizon_len
        for i in range(len(series) - window_size + 1):
            X.append(series[i:i + self.context_len])
            Y.append(series[i + self.context_len:i + window_size])

        X_t = torch.tensor(np.array(X), dtype=torch.float32)
        Y_t = torch.tensor(np.array(Y), dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X_t, Y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history = {"loss": []}
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_x)
            avg_loss = total_loss / len(dataset)
            history["loss"].append(avg_loss)
            print(f"[Epoch {epoch+1}/{epochs}] Fine-Tuning Loss: {avg_loss:.4f}")

        return history

    def forecast(self, recent_observations: np.ndarray) -> np.ndarray:
        """Inference: forecasts next horizon_len steps given context_len observations."""
        self.model.eval()
        if len(recent_observations) < self.context_len:
            raise ValueError(f"Requires at least {self.context_len} context observations, got {len(recent_observations)}")

        context = recent_observations[-self.context_len:]
        x = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.model(x)
        return preds.squeeze(0).cpu().numpy()

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
