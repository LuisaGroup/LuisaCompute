#!/usr/bin/env python3
# =============================================================================
# tensor_basics.py — interactive beginner exercises (PyTorch)
# =============================================================================
# A self-contained tour of the fundamentals, each step verified by an assert:
#   1. Tensors        — creation and shapes
#   2. Operations     — elementwise arithmetic
#   3. Autograd       — automatic differentiation (dy/dx = 2x + 2)
#   4. Simple network — a 1 -> 1 ReLU MLP trained with a manual loop,
#                       then used for inference on new points
# Run it as a test (all asserts must pass) or step through it to learn.
#
# Usage:
#   python examples/tensor/tensor_basics.py [--steps 2000]
# =============================================================================

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Exercise 1: tensors
# -----------------------------------------------------------------------------

def exercise_tensors():
    print("[exercise 1] tensors")
    t1 = torch.tensor([1.0, 2.0, 3.0])
    t2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
    print(f"  t1 = {t1.tolist()}  shape={tuple(t1.shape)} dtype={t1.dtype}")
    print(f"  t2 = {t2.tolist()}  shape={tuple(t2.shape)} dtype={t2.dtype}")
    assert t1.shape == (3,) and t1.dtype == torch.float32
    assert t2.shape == (3, 2) and t2.dtype == torch.int64
    print("  OK: tensor creation and shapes")


# -----------------------------------------------------------------------------
# Exercise 2: operations
# -----------------------------------------------------------------------------

def exercise_operations():
    print("[exercise 2] operations")
    a = torch.tensor([2.0, 4.0, 6.0])
    b = torch.tensor([1.0, 3.0, 5.0])
    print(f"  a + b = {a.add(b).tolist()}")
    print(f"  a - b = {a.sub(b).tolist()}")
    print(f"  a * b = {a.mul(b).tolist()}")
    print(f"  a / b = {a.div(b).tolist()}")
    assert torch.equal(a + b, torch.tensor([3.0, 7.0, 11.0]))
    assert torch.equal(a * b, torch.tensor([2.0, 12.0, 30.0]))
    print("  OK: elementwise arithmetic")


# -----------------------------------------------------------------------------
# Exercise 3: autograd
# -----------------------------------------------------------------------------

def exercise_autograd():
    print("[exercise 3] autograd")
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2 + 2 * x + 1
    y.backward()
    print(f"  y = x^2 + 2x + 1 at x=3 -> y = {y.item()}, dy/dx = {x.grad.item()}")
    assert x.grad is not None and abs(x.grad.item() - 8.0) < 1e-6
    print("  OK: dy/dx = 2x + 2 = 8.0")


# -----------------------------------------------------------------------------
# Exercise 4: simple neural network — train + inference
# -----------------------------------------------------------------------------

class SimpleNN(nn.Module):
    """The tiniest useful network: one Linear + ReLU, 1 input -> 1 output."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, x):
        return F.relu(self.fc1(x))


def exercise_simple_nn(steps):
    print(f"[exercise 4] simple neural network (train {steps} steps)")
    torch.manual_seed(0)
    net = SimpleNN()
    data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    targets = torch.tensor([[2.0], [4.0], [6.0], [8.0]])  # y = 2x (x >= 0)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()
    for step in range(steps):
        optimizer.zero_grad()
        out = net(data)
        loss = loss_fn(out, targets)
        loss.backward()
        optimizer.step()
        if (step + 1) % 500 == 0:
            print(f"  step {step + 1:5d}  loss = {loss.item():.6f}")

    # ---- inference: predict on points the network never saw -----------------
    net.eval()
    with torch.no_grad():
        x_new = torch.tensor([[0.5], [1.5], [2.5], [3.5]])
        pred = net(x_new)
    expected = 2.0 * x_new
    max_err = (pred - expected).abs().max().item()
    print(f"  inference on new points: pred = {pred.flatten().tolist()}")
    print(f"  expected (y = 2x)      : {expected.flatten().tolist()}")
    print(f"  max|err| = {max_err:.4f}")
    assert max_err < 0.1, "tiny NN failed to learn y = 2x"
    print("  OK: training + inference completed")


def main():
    ap = argparse.ArgumentParser(description="PyTorch beginner exercises.")
    ap.add_argument("--steps", type=int, default=2000,
                    help="training steps for the tiny network")
    args = ap.parse_args()

    exercise_tensors()
    exercise_operations()
    exercise_autograd()
    exercise_simple_nn(args.steps)
    print("[pytorch] OK: all exercises passed")


if __name__ == "__main__":
    main()
