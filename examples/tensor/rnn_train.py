#!/usr/bin/env python3
# =============================================================================
# rnn_train.py — RNN sequence classification (PyTorch)
# =============================================================================
# Classic recurrent-network toy task from the "NLP applications" category:
# given a binary sequence (e.g. "01101001"), decide whether it contains
# "enough" ones — a pure counting task that a feed-forward network on single
# symbols cannot solve, but an nn.RNN learns by accumulating state across
# timesteps.
#
# The dataset enumerates ALL 2^T binary sequences of length T, splits them
# into train / held-out test bitstrings, and after training evaluates on
# unseen sequences (real generalization, no data downloads).
#
# Usage:
#   python examples/tensor/rnn_train.py [--epochs 150] [--hidden 16]
# =============================================================================

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SequenceRNN(nn.Module):
    """nn.RNN over one-hot-free symbols -> last hidden state -> 2 classes."""

    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)          # out: [B, T, hidden]
        last = out[:, -1, :]          # use the final hidden state
        return self.fc(last)


def make_sequences(length, threshold):
    """All 2^length binary sequences; label = 1 iff count of ones >= threshold."""
    n = 1 << length
    bits = torch.tensor(
        [[(i >> (length - 1 - t)) & 1 for t in range(length)] for i in range(n)],
        dtype=torch.float32,
    )
    x = bits.unsqueeze(-1)                       # [n, T, 1]
    y = (bits.sum(dim=1) >= threshold).long()    # [n]
    return x, y


def main():
    ap = argparse.ArgumentParser(
        description="Train an RNN to count ones in binary sequences.")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--length", type=int, default=8,
                    help="sequence length T (2^T total bitstrings)")
    ap.add_argument("--threshold", type=int, default=3,
                    help="label = 1 iff #ones >= threshold")
    ap.add_argument("--lr", type=float, default=5e-2)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = make_sequences(args.length, args.threshold)
    n = x.shape[0]
    # deterministic 70/30 split over the enumerated bitstrings
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    n_tr = int(0.7 * n)
    x_tr, y_tr = x[perm[:n_tr]], y[perm[:n_tr]]
    x_te, y_te = x[perm[n_tr:]], y[perm[n_tr:]]
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=32, shuffle=True)

    model = SequenceRNN(1, args.hidden, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"[pytorch] RNN(hidden={args.hidden}) on all {n} binary sequences of "
          f"length {args.length} (label: #ones >= {args.threshold}); "
          f"train {n_tr} / test {n - n_tr} bitstrings, device={device}")
    model.train()
    for epoch in range(args.epochs):
        total, correct = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
            correct += (out.argmax(1) == yb).sum().item()
        if (epoch + 1) % 30 == 0:
            print(f"  epoch {epoch + 1:4d}  loss = {total / len(loader):.4f}"
                  f"  train acc = {100.0 * correct / n_tr:.1f}%")

    # ---- inference on held-out bitstrings ----------------------------------
    model.eval()
    with torch.no_grad():
        out = model(x_te.to(device))
        pred = out.argmax(1).cpu()
        acc = (pred == y_te).float().mean().item()
        # also check a handful of concrete examples
        for i in range(6):
            seq = "".join(str(int(b)) for b in x_te[i].flatten())
            print(f"  '{seq}' -> class {pred[i].item()} "
                  f"(true {y_te[i].item()}, #ones={int(x_te[i].sum())})")

    print(f"[pytorch] inference on {n - n_tr} held-out bitstrings: "
          f"accuracy = {100.0 * acc:.1f}%")
    assert acc >= 0.90, "RNN failed to learn the counting rule"
    print(f"[pytorch] self check: acc = {100.0 * acc:.1f}% >= 90% -> PASS")
    print("[pytorch] OK: training + inference completed")


if __name__ == "__main__":
    main()
