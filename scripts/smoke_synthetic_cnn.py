"""Tiny 1D-CNN sanity run on synthetic Lead-I-like signals (no dataset needed)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TinyCNN1D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    n, length = 256, 1000
    x = torch.randn(n, 1, length)
    y = (x.mean(dim=-1).squeeze(1) > 0).float()

    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
    model = TinyCNN1D().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    last_loss = None
    for step, (xb, yb) in enumerate(loader, start=1):
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
        if step >= 5:
            break

    print(f"device={device} steps={step} last_loss={last_loss:.4f}")
    print("synthetic CNN smoke OK")


if __name__ == "__main__":
    main()
