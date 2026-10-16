#!/usr/bin/env python3
# =============================================================================
# linear_regression_train.py — linear & logistic regression (PyTorch)
# =============================================================================
# Two classic first models from the "basics" category, both trained and then
# used for inference on held-out data:
#
#   Part 1 — Linear regression: recover y = w·x + b from noisy samples.
#            Self check: the learned weight vector must match the true one
#            (the model literally recovers the data-generating parameters).
#
#   Part 2 — Logistic regression: classify 2D points from two overlapping
#            Gaussian blobs. Self check: held-out accuracy >= 85%.
#
# Usage:
#   python examples/tensor/linear_regression_train.py [--epochs 200]
# =============================================================================

import argparse
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Part 1: linear regression (y = w·x + b + noise)
# -----------------------------------------------------------------------------

def train_linear_regression(epochs, lr):
    torch.manual_seed(1)
    dim = 4
    true_w = torch.tensor([1.5, -2.0, 0.5, 3.0])
    true_b = torch.tensor(0.7)

    n = 512
    x = torch.randn(n, dim)
    y = x @ true_w + true_b + 0.05 * torch.randn(n)

    model = nn.Linear(dim, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print("[pytorch] linear regression: y = w·x + b + noise, "
          f"true w = {true_w.tolist()}, b = {true_b.item():.2f}")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x).squeeze(-1)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch + 1:4d}  loss = {loss.item():.6f}")

    w_err = (model.weight.detach().squeeze(0) - true_w).abs().max().item()
    b_err = abs(model.bias.item() - true_b.item())
    print(f"  learned w = {model.weight.detach().squeeze(0).tolist()}")
    print(f"  learned b = {model.bias.item():.4f}  (max|w err| = {w_err:.4f})")

    # ---- inference on held-out data ----------------------------------------
    model.eval()
    with torch.no_grad():
        x_te = torch.randn(128, dim)
        y_te = x_te @ true_w + true_b
        pred_te = model(x_te).squeeze(-1)
        rmse = ((pred_te - y_te) ** 2).mean().sqrt().item()
    print(f"  inference rmse on held-out data = {rmse:.5f}")
    assert w_err < 0.1 and b_err < 0.1, "linear regression failed to recover w/b"
    assert rmse < 0.2, "linear regression inference rmse too large"
    print("  OK: weights recovered, inference matches")
    return rmse


# -----------------------------------------------------------------------------
# Part 2: logistic regression (binary classification)
# -----------------------------------------------------------------------------

def train_logistic_regression(epochs, lr):
    torch.manual_seed(2)
    n = 600
    x_a = torch.randn(n // 2, 2) + torch.tensor([2.0, 2.0])
    x_b = torch.randn(n // 2, 2) - torch.tensor([2.0, 2.0])
    x = torch.cat([x_a, x_b])
    y = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)])

    model = nn.Linear(2, 1)              # raw logits
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()     # sigmoid is fused into the loss

    print("[pytorch] logistic regression: 2D Gaussian blobs, "
          f"{n} samples")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(x).squeeze(-1)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            acc = ((torch.sigmoid(logits) > 0.5).float() == y).float().mean()
            print(f"  epoch {epoch + 1:4d}  loss = {loss.item():.6f}"
                  f"  train acc = {100.0 * acc.item():.1f}%")

    # ---- inference on held-out data ----------------------------------------
    model.eval()
    with torch.no_grad():
        x_a_te = torch.randn(150, 2) + torch.tensor([2.0, 2.0])
        x_b_te = torch.randn(150, 2) - torch.tensor([2.0, 2.0])
        x_te = torch.cat([x_a_te, x_b_te])
        y_te = torch.cat([torch.ones(150), torch.zeros(150)])
        prob = torch.sigmoid(model(x_te).squeeze(-1))
        pred = (prob > 0.5).float()
        acc = (pred == y_te).float().mean().item()
    print(f"  inference accuracy on held-out data = {100.0 * acc:.1f}%")
    print(f"  sample probabilities: "
          f"{[round(p, 3) for p in prob[:5].tolist()]} (class 1) ...")
    assert acc >= 0.85, "logistic regression failed on held-out data"
    print("  OK: held-out accuracy sufficient")
    return acc


def main():
    ap = argparse.ArgumentParser(
        description="Linear + logistic regression training and inference.")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    rmse = train_linear_regression(args.epochs, args.lr)
    acc = train_logistic_regression(args.epochs, args.lr)
    print(f"[pytorch] OK: linear rmse = {rmse:.5f}, "
          f"logistic acc = {100.0 * acc:.1f}% — training + inference completed")


if __name__ == "__main__":
    main()
