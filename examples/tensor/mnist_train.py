#!/usr/bin/env python3
# =============================================================================
# mnist_train.py — full MNIST training loop with DataLoader (PyTorch)
# =============================================================================
# The "Hello World" of deep learning, following the classic beginner pattern:
# torchvision datasets + transforms -> DataLoader -> SimpleNN (784 -> 128 ->
# 10) -> CrossEntropyLoss + SGD -> train loop -> evaluation on the test set.
#
# The script runs in two modes:
#   --dataset mnist      : the real MNIST dataset (downloaded once by
#                          torchvision into --data-dir)
#   --dataset synthetic  : a fully reproducible stand-in (per-class random
#                          templates + noise) so the script also trains and
#                          infers offline, with no downloads (default)
#
# After training, the checkpoint is saved to --out, then reloaded from disk
# and used for inference on the test split.
#
# Usage:
#   python examples/tensor/mnist_train.py [--epochs 2] [--dataset synthetic]
#   python examples/tensor/mnist_train.py --dataset mnist --epochs 3
# =============================================================================

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from torchvision import datasets, transforms
    _HAS_TORCHVISION = True
except ImportError:  # torchvision is optional; synthetic mode still works
    _HAS_TORCHVISION = False


class SimpleNN(nn.Module):
    """Two-layer MLP: 784 -> 128 -> 10 (the classic MNIST beginner model)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def make_synthetic_mnist(n_train, n_test, seed=7):
    """Deterministic MNIST stand-in: 10 fixed random 28x28 templates, one per
    class, plus Gaussian noise. A 128-unit MLP can learn this to >95%."""
    g = torch.Generator().manual_seed(seed)
    templates = torch.randn(10, 28 * 28, generator=g)
    n = n_train + n_test
    y = torch.randint(0, 10, (n,), generator=g)
    x = templates[y] + 0.35 * torch.randn(n, 28 * 28, generator=g)
    return x.view(n, 1, 28, 28), y


def load_mnist(data_dir, n_train, n_test):
    """Real MNIST via torchvision; may download the dataset on first use."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True,
                              transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True,
                             transform=transform)
    # optional --limit-style subsampling for quick runs
    if n_train is not None and n_train < len(train_ds):
        train_ds = torch.utils.data.Subset(train_ds, range(n_train))
    if n_test is not None and n_test < len(test_ds):
        test_ds = torch.utils.data.Subset(test_ds, range(n_test))
    return train_ds, test_ds


def main():
    ap = argparse.ArgumentParser(
        description="Train a simple MLP on MNIST (or a synthetic stand-in) "
                    "and evaluate it.")
    ap.add_argument("--dataset", choices=["mnist", "synthetic"],
                    default="synthetic",
                    help="'synthetic' (default, offline) or 'mnist' "
                         "(requires torchvision, may download)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--data-dir", default="mnist_data",
                    help="directory for the real MNIST dataset")
    ap.add_argument("--out", default="mnist_model.pt",
                    help="checkpoint file (saved, then reloaded for inference)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of train/test samples (quick runs)")
    args = ap.parse_args()

    if args.dataset == "mnist" and not _HAS_TORCHVISION:
        print("[pytorch] torchvision not installed; falling back to the "
              "synthetic dataset")
        args.dataset = "synthetic"

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data loading -------------------------------------------------------
    if args.dataset == "mnist":
        train_ds, test_ds = load_mnist(args.data_dir, args.limit, args.limit)
        print(f"[pytorch] dataset: MNIST "
              f"({len(train_ds)} train / {len(test_ds)} test samples)")
    else:
        n_tr = args.limit if args.limit is not None else 6000
        n_te = args.limit if args.limit is not None else 1000
        x_all, y_all = make_synthetic_mnist(n_tr, n_te)
        x_tr, y_tr = x_all[:n_tr], y_all[:n_tr]
        x_te, y_te = x_all[n_tr:], y_all[n_tr:]
        train_ds = TensorDataset(x_tr, y_tr)
        test_ds = TensorDataset(x_te, y_te)
        print(f"[pytorch] dataset: synthetic MNIST stand-in "
              f"({len(train_ds)} train / {len(test_ds)} test samples)")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    # ---- training -----------------------------------------------------------
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    print(f"[pytorch] training SimpleNN(784->128->10) for {args.epochs} "
          f"epoch(s) on {device}")
    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, n_seen = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            n_seen += labels.numel()
        print(f"  epoch {epoch + 1}/{args.epochs}  "
              f"loss = {total_loss / len(train_loader):.4f}  "
              f"train acc = {100.0 * correct / n_seen:.1f}%")

    # ---- save checkpoint ----------------------------------------------------
    torch.save(model.state_dict(), args.out)
    print(f"[pytorch] saved checkpoint -> '{args.out}' "
          f"({os.path.getsize(args.out)} bytes)")

    # ---- inference: reload from checkpoint and evaluate ---------------------
    model = SimpleNN().to(device)
    model.load_state_dict(torch.load(args.out, map_location=device))
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.numel()
    test_acc = correct / total
    print(f"[pytorch] inference (reloaded from '{args.out}'): "
          f"test accuracy = {100.0 * test_acc:.1f}% ({correct}/{total})")

    # synthetic mode reaches >99%, full MNIST with 2 SGD epochs ~92%;
    # small --limit subsets train less, so use a slightly lower bound there
    bound = 0.75 if args.dataset == "mnist" else 0.80
    print(f"[pytorch] self check: test acc = {100.0 * test_acc:.1f}% "
          f">= {100.0 * bound:.0f}% -> {'PASS' if test_acc >= bound else 'FAIL'}")
    assert test_acc >= bound, "MNIST model failed to learn"
    print("[pytorch] OK: training + inference completed")


if __name__ == "__main__":
    main()
