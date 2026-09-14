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
# After training the model is exported with torch.export (graph structure via
# torch.export.export() + run_decompositions(), weights/inputs/reference as
# base64) to the portable JSON artifact consumed by the C++ importer
# (example_tensor_stub <backend> --rnn-pt2 rnn_exported.pt2.json) and, for
# documentation, to the canonical torch.export.save() .pt2 archive.
#
# Usage:
# python examples/tensor/rnn_train.py [--epochs 150] [--hidden 16]
#   [--export] [--no-export] [--out rnn_exported.pt2.json]
#   [--out-pt2 rnn_exported.pt2] [--no-pt2]
# =============================================================================
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import torch2_export


class SequenceRNN(nn.Module):
    """nn.RNN over one-hot-free symbols -> last hidden state -> 2 classes."""
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)  # out: [B, T, hidden]
        last = out[:, -1, :]  # use the final hidden state
        return self.fc(last)


class SequenceRNNExport(nn.Module):
    """Explicit-loop twin of SequenceRNN for torch.export (no aten.rnn_tanh).

    nn.RNN traces to the opaque aten.rnn_tanh op which does not decompose to
    basic ATen ops on all torch versions, so this module unrolls the same math
    with torch.mm / torch.tanh / torch.add over a static range(T); torch.export
    statically unrolls it into a flat graph of basic ops.

    Parameters are stored in the exact shapes the graph needs (no permute):
      w_ih [input_size, hidden], w_hh [hidden, hidden],
      b_ih/b_hh [hidden], fc_w [hidden, num_classes], fc_b [num_classes].
    """

    def __init__(self, input_size, hidden_size, num_classes, T):
        super().__init__()
        self.T = T
        self.w_ih = nn.Parameter(torch.empty(input_size, hidden_size))
        self.w_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.empty(hidden_size))
        self.b_hh = nn.Parameter(torch.empty(hidden_size))
        self.fc_w = nn.Parameter(torch.empty(hidden_size, num_classes))
        self.fc_b = nn.Parameter(torch.empty(num_classes))

    def forward(self, x):  # x: [B, T, input_size]
        h = torch.zeros(x.shape[0], self.w_hh.shape[0],
                        dtype=x.dtype, device=x.device)
        for t in range(self.T):  # statically unrolled by torch.export
            x_t = x[:, t, :]  # aten.select
            h = torch.tanh(x_t @ self.w_ih + self.b_ih +
                           h @ self.w_hh + self.b_hh)
        return h @ self.fc_w + self.fc_b  # aten.mm + aten.add


def make_export_module(model, T):
    """Copy trained weights (with the transposes the export graph expects)."""
    export = SequenceRNNExport(1, model.rnn.hidden_size, model.fc.out_features, T)
    with torch.no_grad():
        export.w_ih.copy_(model.rnn.weight_ih_l0.t())
        export.w_hh.copy_(model.rnn.weight_hh_l0.t())
        export.b_ih.copy_(model.rnn.bias_ih_l0)
        export.b_hh.copy_(model.rnn.bias_hh_l0)
        export.fc_w.copy_(model.fc.weight.t())
        export.fc_b.copy_(model.fc.bias)
    return export


def make_sequences(length, threshold):
    """All 2^length binary sequences; label = 1 iff count of ones >= threshold."""
    n = 1 << length
    bits = torch.tensor(
        [[(i >> (length - 1 - t)) & 1 for t in range(length)] for i in range(n)],
        dtype=torch.float32,
    )
    x = bits.unsqueeze(-1)  # [n, T, 1]
    y = (bits.sum(dim=1) >= threshold).long()  # [n]
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
    ap.add_argument("--export", action="store_true", default=True,
                    help="export the trained graph after training (default)")
    ap.add_argument("--no-export", action="store_false", dest="export",
                    help="skip the torch.export JSON artifact")
    ap.add_argument("--out", default="rnn_exported.pt2.json",
                    help="portable JSON artifact path")
    ap.add_argument("--out-pt2", default="rnn_exported.pt2",
                    help="canonical torch.export.save() archive path")
    ap.add_argument("--no-pt2", action="store_true",
                    help="skip the canonical .pt2 archive (JSON only)")
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

    # ---- torch.export: structure + weights + reference I/O ------------------
    if args.export:
        x_exp = x_te[:32].to(device)   # [32, T, 1]
        y_exp = y_te[:32]              # [32] int64 labels
        export = make_export_module(model, args.length).to(device)
        ep = torch2_export.export_module_to_json(
            export, (x_exp,), args.out, model_name="SequenceRNN",
            labels=y_exp, ref_args=(x_exp,))
        # eager SequenceRNNExport vs exported program vs trained SequenceRNN
        with torch.no_grad():
            eager_out = export(x_exp).cpu()
            ep_out = ep.module()(x_exp).cpu()
            trained_out = model(x_exp).cpu()
        diff_eager = (eager_out - ep_out).abs().max().item()
        diff_trained = (trained_out - ep_out).abs().max().item()
        print(f"[pytorch] export self check: eager-vs-exported max diff = "
              f"{diff_eager:.3e} (< 1e-5), trained-vs-exported max diff = "
              f"{diff_trained:.3e} (< 1e-3)")
        assert diff_eager < 1e-5, "eager vs exported output mismatch"
        # cuDNN nn.RNN vs the explicit-loop twin accumulate in different order,
        # so the trained-vs-exported bound is relaxed to 1e-3 (still far below
        # the 13.9 error produced by a real weight-transpose bug).
        assert diff_trained < 1e-3, "trained vs exported output mismatch"
        if not args.no_pt2:
            torch2_export.save_canonical_pt2(ep, args.out_pt2)

    print("[pytorch] OK: training + inference completed")
    if args.export:
        print(f"[pytorch] OK: torch.export artifact written -> {args.out}")


if __name__ == "__main__":
    main()
