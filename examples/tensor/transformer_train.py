#!/usr/bin/env python3
# =============================================================================
# transformer_train.py — tiny Transformer sequence classification (PyTorch)
# =============================================================================
# Like rnn_train.py, this is a self-contained toy example:
#   - all 2^S binary sequences of length S,
#   - label = 1 iff the count of ones >= threshold,
#   - a tiny one-layer self-attention transformer is trained with Adam + CE.
#
# After training the model is exported with torch.export (Core ATen IR) into a
# portable JSON artifact consumed by the C++ importer:
#   example_tensor_stub <backend> --transformer-pt2 transformer_exported.pt2.json
#
# The exported graph contains only basic ATen ops (view, mm, _softmax, add,
# tanh) so the C++ Luisa tile-language executor can replay it exactly.
# =============================================================================
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import torch2_export


class TinyTransformer(nn.Module):
    """Batched training transformer (one self-attention block + linear head)."""

    def __init__(self, seq_len=8, d_model=8, num_classes=2):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.scale = 1.0 / math.sqrt(d_model)
        # self-attention q/k/v and output projection
        self.Wq = nn.Parameter(torch.empty(d_model, d_model))
        self.Wk = nn.Parameter(torch.empty(d_model, d_model))
        self.Wv = nn.Parameter(torch.empty(d_model, d_model))
        self.Wo = nn.Parameter(torch.empty(d_model, d_model))
        self.bo = nn.Parameter(torch.empty(d_model))
        # final classifier head: flatten the whole sequence
        self.W2 = nn.Parameter(torch.empty(seq_len * d_model, num_classes))
        self.b2 = nn.Parameter(torch.empty(num_classes))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in [self.Wq, self.Wk, self.Wv, self.Wo, self.W2]:
            nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.bo)
        nn.init.zeros_(self.b2)

    def forward(self, x):
        # x: [B, S, D]
        B, S, D = x.shape
        xs = x.reshape(B * S, D)               # [B*S, D]
        q = (xs @ self.Wq).view(B, S, D)       # [B, S, D]
        k = (xs @ self.Wk).view(B, S, D)       # [B, S, D]
        v = (xs @ self.Wv).view(B, S, D)       # [B, S, D]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, S, S]
        attn = F.softmax(scores, dim=-1)       # [B, S, S]
        o = torch.matmul(attn, v).view(B, S, D)                       # [B, S, D]
        o = torch.tanh(o @ self.Wo + self.bo)  # [B, S, D]
        o = torch.tanh(o + x)                  # residual
        h = o.view(B, S * D)                   # [B, S*D]
        return h @ self.W2 + self.b2           # [B, C]


class TinyTransformerExport(nn.Module):
    """Single-sample export twin (B == 1) that avoids aten.bmm.

    The attention scores are computed as  (x_s @ Wq) @ (Wk_t @ x_s.t())  where
    x_s = x.view(S, D).  The 1/sqrt(D) scale is baked into Wq so the graph
    contains no aten.div.  The resulting graph uses only aten.view, aten.mm,
    aten.permute, aten._softmax, aten.add and aten.tanh.
    """

    def __init__(self, seq_len=8, d_model=8, num_classes=2):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.Wq = nn.Parameter(torch.empty(d_model, d_model))
        self.Wk_t = nn.Parameter(torch.empty(d_model, d_model))
        self.Wv = nn.Parameter(torch.empty(d_model, d_model))
        self.Wo = nn.Parameter(torch.empty(d_model, d_model))
        self.bo = nn.Parameter(torch.empty(d_model))
        self.W2 = nn.Parameter(torch.empty(seq_len * d_model, num_classes))
        self.b2 = nn.Parameter(torch.empty(num_classes))

    def forward(self, x):
        # x: [1, S, D]
        xs = x.view(self.seq_len, self.d_model)          # [S, D]
        q = xs @ self.Wq                                  # [S, D]
        xd = xs.transpose(0, 1)                           # [D, S]
        kt = self.Wk_t @ xd                               # [D, S]
        scores = q @ kt                                   # [S, S]
        attn = torch.softmax(scores, dim=-1)              # [S, S]
        v = xs @ self.Wv                                  # [S, D]
        o = attn @ v                                      # [S, D]
        o = torch.tanh(o @ self.Wo + self.bo)             # [S, D]
        o = torch.tanh(o + xs)                            # [S, D]
        h = o.view(1, self.seq_len * self.d_model)        # [1, S*D]
        return h @ self.W2 + self.b2                      # [1, C]


def make_export_module(model, num_classes=2):
    """Copy trained weights into the B==1 export twin."""
    exp = TinyTransformerExport(model.seq_len, model.d_model, num_classes)
    scale = 1.0 / math.sqrt(model.d_model)
    with torch.no_grad():
        exp.Wq.copy_(model.Wq * scale)
        exp.Wk_t.copy_(model.Wk.t())
        exp.Wv.copy_(model.Wv)
        exp.Wo.copy_(model.Wo)
        exp.bo.copy_(model.bo)
        exp.W2.copy_(model.W2)
        exp.b2.copy_(model.b2)
    return exp


def make_sequences(seq_len, d_model, threshold):
    """All 2^seq_len binary sequences; x[:,:,0] is the bit, label by count."""
    n = 1 << seq_len
    bits = torch.tensor(
        [[(i >> (seq_len - 1 - t)) & 1 for t in range(seq_len)] for i in range(n)],
        dtype=torch.float32,
    )
    x = torch.zeros(n, seq_len, d_model)
    x[:, :, 0] = bits
    y = (bits.sum(dim=1) >= threshold).long()
    return x, y


def main():
    ap = argparse.ArgumentParser(
        description="Train a tiny Transformer to count ones in binary sequences.")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threshold", type=int, default=4)
    ap.add_argument("--export", action="store_true", default=True,
                    help="export the trained graph after training (default)")
    ap.add_argument("--no-export", action="store_false", dest="export",
                    help="skip the torch.export JSON artifact")
    ap.add_argument("--out", default="transformer_exported.pt2.json",
                    help="portable JSON artifact path")
    ap.add_argument("--out-pt2", default="transformer_exported.pt2",
                    help="canonical torch.export.save() archive path")
    ap.add_argument("--no-pt2", action="store_true",
                    help="skip the canonical .pt2 archive (JSON only)")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = make_sequences(args.seq_len, args.d_model, args.threshold)
    n = x.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    n_tr = int(0.7 * n)
    x_tr, y_tr = x[perm[:n_tr]], y[perm[:n_tr]]
    x_te, y_te = x[perm[n_tr:]], y[perm[n_tr:]]
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=32, shuffle=True)
    model = TinyTransformer(args.seq_len, args.d_model, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    print(f"[pytorch] Transformer(S={args.seq_len}, D={args.d_model}) on all {n} "
          f"binary sequences (label: #ones >= {args.threshold}); "
          f"train {n_tr} / test {n - n_tr}, device={device}")

    model.train()
    for epoch in range(args.epochs):
        total, correct = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            correct += (out.argmax(1) == yb).sum().item()
        if (epoch + 1) % 40 == 0:
            print(f"  epoch {epoch + 1:4d}  loss = {total / len(loader):.4f}"
                  f"  train acc = {100.0 * correct / n_tr:.1f}%")

    model.eval()
    with torch.no_grad():
        out = model(x_te.to(device))
        pred = out.argmax(1).cpu()
        acc = (pred == y_te).float().mean().item()
        for i in range(6):
            bits = x_te[i, :, 0].long().tolist()
            seq = "".join(str(b) for b in bits)
            print(f"  '{seq}' -> class {pred[i].item()} "
                  f"(true {y_te[i].item()}, #ones={int(sum(bits))})")
    print(f"[pytorch] inference on {n - n_tr} held-out sequences: "
          f"accuracy = {100.0 * acc:.1f}%")
    assert acc >= 0.90, "Transformer failed to learn the counting rule"
    print(f"[pytorch] self check: acc = {100.0 * acc:.1f}% >= 90% -> PASS")

    if args.export:
        x_exp = x_te[:1].to(device)            # [1, S, D]
        y_exp = y_te[:1]                        # [1]
        export = make_export_module(model, 2).to(device)
        ep = torch2_export.export_module_to_json(
            export, (x_exp,), args.out, model_name="TinyTransformer",
            labels=y_exp, ref_args=(x_exp,))

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
        assert diff_trained < 1e-3, "trained vs exported output mismatch"
        if not args.no_pt2:
            torch2_export.save_canonical_pt2(ep, args.out_pt2)

    print("[pytorch] OK: training + inference completed")
    if args.export:
        print(f"[pytorch] OK: torch.export artifact written -> {args.out}")


if __name__ == "__main__":
    main()
