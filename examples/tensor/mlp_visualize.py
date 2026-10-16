#!/usr/bin/env python3
# =============================================================================
# mlp_visualize.py — minimalist MLP training with visualization (PyTorch)
# =============================================================================
# The "minimalist example" pattern (Ilya Schurov, NN 101): the whole
# training loop is a few lines, and every N steps the current fit is plotted
# so you can watch the network learn. The final figure is saved to a PNG
# (matplotlib 'Agg' backend, so it works headless / in CI).
#
# Task: fit y = sin(x) + 0.1*noise on [-3, 3] with a tiny MLP
#       (1 -> 16 -> 16 -> 1, ReLU). After training, the same network is used
#       for inference on a dense grid and checked against the clean function.
#
# Usage:
#   python examples/tensor/mlp_visualize.py [--steps 2000] [--out-plot mlp_fit.png]
# =============================================================================

import argparse
import os

import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")            # headless rendering
import matplotlib.pyplot as plt  # noqa: E402


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser(
        description="Minimal MLP fit with live-ish visualization.")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--animate-each", type=int, default=200,
                    help="re-plot the fit every N steps")
    ap.add_argument("--out-plot", default="mlp_fit.png",
                    help="PNG file for the final figure")
    ap.add_argument("--n-train", type=int, default=200)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data ---------------------------------------------------------------
    x_train = torch.linspace(-3.0, 3.0, args.n_train).unsqueeze(-1)
    y_train = torch.sin(x_train) + 0.1 * torch.randn_like(x_train)

    network = TinyMLP().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

    # ---- training + periodic visualization ----------------------------------
    print(f"[pytorch] training TinyMLP on {args.n_train} noisy samples of "
          f"y = sin(x), {args.steps} steps, device={device}")
    x_train_d = x_train.to(device)
    y_train_d = y_train.to(device)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(args.steps):
        outputs = network(x_train_d)
        loss = criterion(outputs, y_train_d)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.animate_each == 0:
            with torch.no_grad():
                grid = torch.linspace(-3.0, 3.0, 200).unsqueeze(-1)
                fit = network(grid.to(device)).cpu()
            ax.clear()
            ax.scatter(x_train, y_train, s=8, alpha=0.6, label="training data")
            ax.plot(grid, fit, "r-", lw=2, label="network fit")
            ax.plot(grid, torch.sin(grid), "k--", lw=1, label="true sin(x)")
            ax.set_title(f"step {i}/{args.steps}   loss = {loss.item():.4f}")
            ax.legend()
            fig.canvas.draw()   # redraw the in-memory canvas (Agg, headless)

    # ---- inference on the clean function ------------------------------------
    network.eval()
    with torch.no_grad():
        x_test = torch.linspace(-3.0, 3.0, 500).unsqueeze(-1)
        pred = network(x_test.to(device)).cpu()
        y_clean = torch.sin(x_test)
        rmse = ((pred - y_clean) ** 2).mean().sqrt().item()
        max_err = (pred - y_clean).abs().max().item()

    print(f"[pytorch] inference on 500-point grid: "
          f"rmse = {rmse:.4f}, max|err| = {max_err:.4f} vs clean sin(x)")
    for xi, (p, t) in zip([-2.5, -0.5, 1.5], [(pred[83], y_clean[83]),
                                              (pred[291], y_clean[291]),
                                              (pred[458], y_clean[458])]):
        print(f"  x = {xi:+.2f}: predicted {p.item():+.4f}, "
              f"true {t.item():+.4f}")

    # ---- final figure --------------------------------------------------------
    ax.clear()
    ax.scatter(x_train, y_train, s=8, alpha=0.6, label="training data")
    ax.plot(x_test, pred, "r-", lw=2, label="network fit")
    ax.plot(x_test, y_clean, "k--", lw=1, label="true sin(x)")
    ax.set_title(f"MLP fit after {args.steps} steps (rmse = {rmse:.4f})")
    ax.legend()
    fig.savefig(args.out_plot, dpi=120)
    plt.close(fig)
    print(f"[pytorch] saved figure -> '{args.out_plot}' "
          f"({os.path.getsize(args.out_plot)} bytes)")

    assert rmse < 0.15, "MLP failed to approximate sin(x)"
    print(f"[pytorch] self check: rmse = {rmse:.4f} < 0.15 -> PASS")
    print("[pytorch] OK: training + inference + visualization completed")


if __name__ == "__main__":
    main()
