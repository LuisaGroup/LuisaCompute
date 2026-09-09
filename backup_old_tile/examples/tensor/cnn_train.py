#!/usr/bin/env python3
# =============================================================================
# cnn_train.py — small convolutional neural network (PyTorch)
# =============================================================================
# Trains a tiny CNN on a synthetic, fully reproducible dataset (no downloads),
# runs inference, and exports the trained weights + one test input + the
# reference softmax probabilities to a plain binary file:
#
#     examples/tensor/cnn_input.bin
#
# The binary is consumed by the Luisa tile-language example
# (examples/tensor/cnn_inference.cpp, target example_tensor_cnn), which must
# reproduce the *same* inference result on the device (vk / dx backends).
#
# Network structure (mirrored 1:1 by the tile kernels in cnn_kernels.cpp):
#   input            : [B, 1, 8, 8]
#   conv1 (1 -> 4)   : 3x3, padding=0           -> [B, 4, 6, 6]
#   relu
#   conv2 (4 -> 8)   : 3x3, padding=0           -> [B, 8, 4, 4]
#   relu
#   flatten                                    -> [B, 128]
#   fc1   (128 -> 32) + relu                    -> [B, 32]
#   fc2   (32 -> 4)                             -> [B, 4]   (logits)
#   softmax(dim=1)                              -> [B, 4]   (probabilities)
# =============================================================================

import struct
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Network definition
# -----------------------------------------------------------------------------

BATCH = 4          # export batch size (tile kernels fold batch into the GEMM N)
NUM_CLASSES = 4
IMG = 8            # input height == width

C1 = 4             # conv1 output channels
C2 = 8             # conv2 output channels
F1 = 32            # fc1 output features


class TinyCNN(nn.Module):
    """Small CNN whose forward pass is exactly what the Luisa tile kernels
    reproduce: conv -> relu -> conv -> relu -> flatten -> fc -> relu -> fc
    -> softmax, all with f32 accumulation."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, C1, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(C1, C2, kernel_size=3, padding=0)
        self.fc1 = nn.Linear(C2 * 4 * 4, F1)
        self.fc2 = nn.Linear(F1, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.conv1(x))          # [B, C1, 6, 6]
        x = F.relu(self.conv2(x))          # [B, C2, 4, 4]
        x = torch.flatten(x, 1)            # [B, 128]
        x = F.relu(self.fc1(x))            # [B, F1]
        logits = self.fc2(x)               # [B, NUM_CLASSES]
        return logits

    def predict_probs(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


# -----------------------------------------------------------------------------
# Reproducible synthetic dataset
# -----------------------------------------------------------------------------

def make_dataset(n_train, n_test, seed=20260819):
    """Deterministic 8x8 grayscale images with random (fixed-seed) labels:
    a small CNN cannot fit random labels to zero loss, so it trains gradually
    and the softmax logits stay moderate (a much more interesting cross-check
    for the tile kernels than a perfectly-separable rule)."""
    g = torch.Generator().manual_seed(seed)
    n = n_train + n_test
    x = torch.rand(n, 1, IMG, IMG, generator=g)
    y = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    return (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:])


# -----------------------------------------------------------------------------
# Binary export (little-endian, exact float32 bytes)
# -----------------------------------------------------------------------------

def export_bin(path, model, x_test, probs_ref):
    sd = model.state_dict()
    with open(path, "wb") as f:
        def w(s):
            f.write(s)
        def wf32(a):
            w(struct.pack(f"<{a.numel()}f", *a.detach().cpu().float().flatten().tolist()))
        def wi32(v):
            w(struct.pack("<i", int(v)))
        w(b"LUISACNN")                       # magic
        wi32(BATCH)
        wi32(NUM_CLASSES)
        wf32(x_test)                         # [B, 1, 8, 8]
        wf32(sd["conv1.weight"])             # [C1, 1, 3, 3]
        wf32(sd["conv1.bias"])               # [C1]
        wf32(sd["conv2.weight"])             # [C2, C1, 3, 3]
        wf32(sd["conv2.bias"])               # [C2]
        wf32(sd["fc1.weight"])               # [F1, 128]
        wf32(sd["fc1.bias"])                 # [F1]
        wf32(sd["fc2.weight"])               # [NUM_CLASSES, F1]
        wf32(sd["fc2.bias"])                 # [NUM_CLASSES]
        wf32(probs_ref)                      # [B, NUM_CLASSES]


# -----------------------------------------------------------------------------
# main: train + infer + export
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--out", default="cnn_input.bin")
    args = ap.parse_args()

    torch.manual_seed(0)
    model = TinyCNN()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    (x_tr, y_tr), (x_te, y_te) = make_dataset(120, BATCH)
    n_tr = x_tr.shape[0]

    print(f"[pytorch] training TinyCNN on {n_tr} synthetic 8x8 samples "
          f"(epochs={args.epochs}, lr={args.lr})")
    model.train()
    for epoch in range(args.epochs):
        perm = torch.randperm(n_tr)
        x_b, y_b = x_tr[perm], y_tr[perm]
        opt.zero_grad()
        out = model(x_b)
        loss = loss_fn(out, y_b)
        loss.backward()
        opt.step()
        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch + 1:4d}  loss = {loss.item():.4f}")

    # ---- inference on the exported test batch -------------------------------
    model.eval()
    with torch.no_grad():
        probs = model.predict_probs(x_te)
        logits = model.forward(x_te)
        pred = probs.argmax(dim=1)

    print("[pytorch] inference:")
    print("  test input labels :", y_te.tolist())
    print("  predicted classes :", pred.tolist())
    print("  logits           :")
    for i in range(BATCH):
        print(f"    sample {i}: {logits[i].tolist()}")
    print("  softmax probs    :")
    for i in range(BATCH):
        print(f"    sample {i}: {[round(p, 6) for p in probs[i].tolist()]}")

    export_bin(args.out, model, x_te, probs)
    print(f"[pytorch] exported weights/input/reference to '{args.out}' "
          f"({len(open(args.out, 'rb').read())} bytes)")

    # ---- self check: conv folded into im2col+gmm must equal PyTorch conv ----
    # (mirrors the exact math the tile kernels run, so the device result can be
    #  compared to this reference with a tight tolerance)
    with torch.no_grad():
        conv1_ref = F.relu(F.conv2d(x_te, model.conv1.weight, model.conv1.bias))
        conv2_ref = F.relu(F.conv2d(conv1_ref, model.conv2.weight, model.conv2.bias))
        fc_ref = F.relu(F.linear(conv2_ref.flatten(1), model.fc1.weight, model.fc1.bias))
        logit_ref = F.linear(fc_ref, model.fc2.weight, model.fc2.bias)
        prob_ref = F.softmax(logit_ref, dim=1)
    max_diff = (prob_ref - probs).abs().max().item()
    print(f"[pytorch] self check: |direct - explicit| max diff = {max_diff:.3e}")
    assert max_diff < 1e-5, "internal PyTorch consistency check failed"


if __name__ == "__main__":
    main()
