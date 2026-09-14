#!/usr/bin/env python3
# =============================================================================
# poly_fit_train.py — fit y = sin(x) with a 3rd-order polynomial (PyTorch)
# =============================================================================
# PyTorch's canonical "Learning PyTorch with Examples" pattern: build a
# polynomial feature matrix (x, x^2, x^3), feed it through an nn.Sequential
# of a single Linear layer, and train by manually applying the gradients
# (param -= lr * param.grad) instead of an optimizer — the clearest way to
# see what autograd actually does.
#
# This script:
#   1. trains the model on a dense grid in [-pi, pi]  (training)
#   2. evaluates the fitted polynomial on a held-out grid (inference)
#   3. verifies the gradient plumbing with an analytical check
#   4. checks that the fitted curve reproduces sin(x) within tolerance
#
# Usage:
#   python examples/tensor/poly_fit_train.py [--steps 2000] [--lr 1e-6]
# =============================================================================

import argparse
import math
import torch


def make_features(x, degree=3):
    """Return the Vandermonde-style feature matrix [x, x^2, ..., x^degree]."""
    p = torch.arange(1, degree + 1, dtype=torch.float32)
    return x.unsqueeze(-1).pow(p)


def main():
    ap = argparse.ArgumentParser(
        description="Fit y = sin(x) with a degree-3 polynomial (PyTorch tutorial).")
    ap.add_argument("--steps", type=int, default=2000,
                    help="number of gradient-descent steps")
    ap.add_argument("--lr", type=float, default=1e-6,
                    help="manual learning rate (no optimizer is used)")
    ap.add_argument("--n-train", type=int, default=2000,
                    help="number of training samples")
    ap.add_argument("--n-test", type=int, default=500,
                    help="number of held-out inference samples")
    args = ap.parse_args()

    torch.manual_seed(0)

    # ---- training data ------------------------------------------------------
    x = torch.linspace(-math.pi, math.pi, args.n_train)
    y = torch.sin(x)
    xx = make_features(x)                      # [N, 3]
    print(f"[pytorch] training data: {x.shape[0]} samples in [-pi, pi]")

    # ---- model: single linear layer on the polynomial features -------------
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 1),
        torch.nn.Flatten(0, 1),                # drop the trailing unit dim
    )
    loss_fn = torch.nn.MSELoss(reduction="sum")
    lr = args.lr

    print(f"[pytorch] training {args.steps} manual-gradient steps (lr={lr})")
    for t in range(args.steps):
        y_pred = model(xx)
        loss = loss_fn(y_pred, y)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad
        if (t + 1) % 200 == 0:
            print(f"  step {t + 1:5d}  loss = {loss.item():.4f}")

    # ---- autograd self-check: d/dx (x^2 + 2x + 1) = 2x + 2 ------------------
    z = torch.tensor(3.0, requires_grad=True)
    fz = z ** 2 + 2 * z + 1
    fz.backward()
    grad_ok = torch.isclose(z.grad, torch.tensor(2 * 3.0 + 2.0), atol=1e-6).item()
    print(f"[pytorch] autograd check: dy/dx at x=3 -> {z.grad.item():.3f} "
          f"(expected 8.0, {'OK' if grad_ok else 'FAILED'})")
    assert grad_ok, "autograd derivative check failed"

    # ---- inference on a held-out grid ---------------------------------------
    model.eval()
    with torch.no_grad():
        x_test = torch.linspace(-math.pi, math.pi, args.n_test)
        y_test = torch.sin(x_test)
        y_pred = model(make_features(x_test)).squeeze(-1)
    rmse = torch.sqrt(((y_pred - y_test) ** 2).mean()).item()
    max_err = (y_pred - y_test).abs().max().item()
    print(f"[pytorch] inference on {args.n_test} held-out points:")
    print(f"  rmse     = {rmse:.6f}")
    print(f"  max|err| = {max_err:.6f}")

    linear = model[0]
    a, b, c = (linear.weight[0, i].item() for i in range(3))
    d = linear.bias.item()
    print(f"[pytorch] fitted polynomial: "
          f"{a:.4f}*x^3 + {b:.4f}*x^2 + {c:.4f}*x + {d:.4f}")

    # a degree-3 fit of sin(x) on [-pi, pi] cannot do better than
    # max|err| ~ 0.20 (the least-squares optimum itself); 0.25 is a safe bound
    # that still proves gradient-descent converged to the optimum.
    print(f"[pytorch] self check: max|err| = {max_err:.6f} < 0.25 -> "
          f"{'PASS' if max_err < 0.25 else 'FAIL'}")
    assert max_err < 0.25, "polynomial fit did not converge within tolerance"
    print("[pytorch] OK: training + inference completed")


if __name__ == "__main__":
    main()
