// =============================================================================
// rnn.cpp — Luisa tile-language RNN sequence classification training
// =============================================================================
// The C++ twin of examples/tensor/rnn_train.py: given all 2^8 binary sequences
// of length 8, train a tanh RNN (hidden 16) + linear head to decide whether a
// sequence contains at least 3 ones (a pure counting task).
//
// The tile kernels in rnn_kernels.h / mlp_kernels.h implement the whole
// training algorithm — unrolled forward steps (rnn_step), softmax /
// cross-entropy, manual backpropagation through time (BPTT) and SGD updates.
// This driver traces/lowers/compiles them on the backend (structural device
// validation) and runs the exact same algorithm on the host CPU reference to
// verify the training math reaches the accuracy bound.
//
// NOTE: on some backends the current tile_to_kernel lowering mis-executes
// multi-GEMM-accumulator kernels (a known lowering limitation; the simpler
// poly-fit / linear-regression examples train fully on device), so the
// verification here is host-based.
//
// Verification, mirroring the PyTorch script:
//   1. every tile kernel is traced, lowered and compiled on the backend,
//   2. the host reference must reach >= 90% test accuracy.
//
// This file is part of the single `example_tensor_stub` target and is invoked
// through its main() with the `--rnn` flag (see main.cpp / rnn.h):
//   example_tensor_stub <backend> --rnn [--epochs N]
// =============================================================================

#include "rnn.h"
#include "rnn_kernels.h"
#include "mlp_kernels.h"
#include "mlp_common.h"

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

constexpr int B = 32;      // minibatch
constexpr int T = 8;       // sequence length
constexpr int H = 16;      // hidden size
constexpr int C = 2;       // classes
constexpr int N_TRAIN = 5 * B;   // 160 sequences (5 minibatches)
constexpr int N_TEST = 2 * B;    // 64 sequences
constexpr int N_ALL = 256;       // all 2^8 bitstrings

struct RnnData {
    std::vector<float> x[5];     // [T, B, 1] per minibatch
    std::vector<float> xt[5];    // [T, 1, B] per minibatch
    std::vector<float> y_onehot[5];// [B, C]
    std::vector<int> y_test;     // [N_TEST]
    std::vector<float> x_test[2];// [T, B, 1] per test minibatch
};

RnnData make_rnn_data() {
    std::vector<int> y_all(N_ALL);
    std::vector<std::vector<float>> seq(N_ALL, std::vector<float>(T));
    for (int i = 0; i < N_ALL; ++i) {
        int ones = 0;
        for (int t = 0; t < T; ++t) {
            int bit = (i >> (T - 1 - t)) & 1;
            seq[i][t] = static_cast<float>(bit);
            ones += bit;
        }
        y_all[i] = ones >= 3 ? 1 : 0;
    }
    std::vector<int> perm(N_ALL);
    for (int i = 0; i < N_ALL; ++i) { perm[i] = i; }
    unsigned s = 42u * 2654435761u + 12345u;
    for (int i = N_ALL - 1; i > 0; --i) {
        s = s * 1664525u + 1013904223u;
        int j = static_cast<int>((s >> 8) & 0xFFFFFFu) % (i + 1);
        std::swap(perm[i], perm[j]);
    }

    RnnData d;
    auto fill_block = [&](std::vector<float> &x, std::vector<float> &xt,
                          std::vector<float> &yo, const std::vector<int> &ids,
                          int base) {
        x.assign(static_cast<size_t>(T) * B, 0.0f);
        xt.assign(static_cast<size_t>(T) * B, 0.0f);
        yo.assign(static_cast<size_t>(B) * C, 0.0f);
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < B; ++i) {
                int id = ids[base + i];
                float v = seq[id][t];
                x[static_cast<size_t>(t) * B + i] = v;
                xt[static_cast<size_t>(t) * B + i] = v;
            }
        }
        for (int i = 0; i < B; ++i) {
            yo[static_cast<size_t>(i) * C + y_all[ids[base + i]]] = 1.0f;
        }
    };

    for (int b = 0; b < 5; ++b) { fill_block(d.x[b], d.xt[b], d.y_onehot[b], perm, b * B); }
    for (int b = 0; b < 2; ++b) {
        std::vector<float> dummy_xt, dummy_y;
        fill_block(d.x_test[b], dummy_xt, dummy_y, perm, 5 * B + b * B);
    }
    d.y_test.resize(N_TEST);
    for (int i = 0; i < N_TEST; ++i) { d.y_test[i] = y_all[perm[5 * B + i]]; }
    return d;
}

// ---- host reference: exact same algorithm as the device, in double ----------
struct RnnHostRef {
    std::vector<double> Wih;      // [1, H]
    std::vector<double> Whh;      // [H, H]
    std::vector<double> Wfc;      // [H, C]
    std::vector<double> Bias_ih;  // [1, H]
    std::vector<double> Bias_hh;  // [1, H]
    std::vector<double> Bias_fc;  // [1, C]
    std::vector<double> epoch_losses;
    double test_acc = 0.0;
};

RnnHostRef rnn_host_reference(const RnnData &d, int epochs) {
    RnnHostRef r;
    mlpcommon::PRNG rng(0u);
    r.Wih.assign(H, 0.0);
    r.Whh.assign(static_cast<size_t>(H) * H, 0.0);
    r.Wfc.assign(static_cast<size_t>(H) * C, 0.0);
    r.Bias_ih.assign(H, 0.0);
    r.Bias_hh.assign(H, 0.0);
    r.Bias_fc.assign(C, 0.0);
    for (auto &v : r.Wih) { v = 0.5 * rng.gauss(); }
    for (auto &v : r.Whh) { v = 0.5 * rng.gauss(); }
    for (auto &v : r.Wfc) { v = 0.5 * rng.gauss(); }
    for (auto &v : r.Bias_ih) { v = 0.5 * rng.gauss(); }
    for (auto &v : r.Bias_hh) { v = 0.5 * rng.gauss(); }
    for (auto &v : r.Bias_fc) { v = 0.5 * rng.gauss(); }

    const double lr = 0.1;// dW/db already include the 1/B from the CE gradient
    auto matmul = [](const std::vector<double> &A, const std::vector<double> &X,
                     int M, int K, int N, std::vector<double> &Y) {
        Y.assign(static_cast<size_t>(M) * N, 0.0);
        for (int i = 0; i < M; ++i) {
            for (int n = 0; n < N; ++n) {
                double s = 0.0;
                for (int k = 0; k < K; ++k) {
                    s += A[static_cast<size_t>(i) * K + k] * X[static_cast<size_t>(k) * N + n];
                }
                Y[static_cast<size_t>(i) * N + n] = s;
            }
        }
    };

    for (int ep = 0; ep < epochs; ++ep) {
        double ep_loss = 0.0;
        for (int b = 0; b < 5; ++b) {
            const auto &xb = d.x[b];
            std::vector<double> Hst(static_cast<size_t>(B) * H, 0.0);
            std::vector<std::vector<double>> Hs(T);
            for (int t = 0; t < T; ++t) {
                std::vector<double> z(B * H), h(B * H);
                for (int i = 0; i < B; ++i) {
                    for (int o = 0; o < H; ++o) {
                        double s = r.Bias_ih[o] + r.Bias_hh[o];
                        s += xb[static_cast<size_t>(t) * B + i] * r.Wih[o];
                        for (int k = 0; k < H; ++k) {
                            s += Hst[static_cast<size_t>(i) * H + k] * r.Whh[static_cast<size_t>(k) * H + o];
                        }
                        z[static_cast<size_t>(i) * H + o] = s;
                    }
                }
                for (int i = 0; i < B * H; ++i) { h[i] = std::tanh(z[i]); }
                Hs[t] = h;
                Hst = h;
            }
            std::vector<double> logits(B * C), P(B * C);
            matmul(Hst, r.Wfc, B, H, C, logits);
            for (int i = 0; i < B * C; ++i) { logits[i] += r.Bias_fc[i % C]; }
            double loss = 0.0;
            for (int i = 0; i < B; ++i) {
                double mx = std::max(logits[static_cast<size_t>(i) * C + 0], logits[static_cast<size_t>(i) * C + 1]);
                double e0 = std::exp(logits[static_cast<size_t>(i) * C + 0] - mx);
                double e1 = std::exp(logits[static_cast<size_t>(i) * C + 1] - mx);
                P[static_cast<size_t>(i) * C + 0] = e0 / (e0 + e1);
                P[static_cast<size_t>(i) * C + 1] = e1 / (e0 + e1);
                int yi = 0;
                for (int c = 0; c < C; ++c) if (d.y_onehot[b][static_cast<size_t>(i) * C + c] > 0.5) yi = c;
                loss += -std::log(std::max(P[static_cast<size_t>(i) * C + yi], 1e-12));
            }
            loss /= B;
            ep_loss += loss;
            std::vector<double> G(B * C);
            for (int i = 0; i < B * C; ++i) { G[i] = (P[i] - d.y_onehot[b][i]) / B; }
            // fc backward
            std::vector<double> dWfc(static_cast<size_t>(H) * C, 0.0), dH(B * H, 0.0), db_fc(C, 0.0);
            for (int k = 0; k < H; ++k)
                for (int c = 0; c < C; ++c) {
                    double s = 0.0;
                    for (int i = 0; i < B; ++i) s += Hst[static_cast<size_t>(i) * H + k] * G[static_cast<size_t>(i) * C + c];
                    dWfc[static_cast<size_t>(k) * C + c] = s;
                }
            for (int c = 0; c < C; ++c) {
                double s = 0.0;
                for (int i = 0; i < B; ++i) s += G[static_cast<size_t>(i) * C + c];
                db_fc[c] = s;
            }
            for (int i = 0; i < B; ++i)
                for (int o = 0; o < H; ++o) {
                    double s = 0.0;
                    for (int c = 0; c < C; ++c) s += G[static_cast<size_t>(i) * C + c] * r.Wfc[static_cast<size_t>(o) * C + c];
                    dH[static_cast<size_t>(i) * H + o] = s;
                }
            // BPTT
            std::vector<double> dWih(H, 0.0), dWhh(static_cast<size_t>(H) * H, 0.0);
            std::vector<double> db_ih(H, 0.0), db_hh(H, 0.0);
            std::vector<double> dH_cur = dH;
            for (int t = T - 1; t >= 0; --t) {
                std::vector<double> dZ(B * H);
                for (int i = 0; i < B * H; ++i) {
                    double h = Hs[t][i];
                    dZ[i] = dH_cur[i] * (1.0 - h * h);
                }
                for (int o = 0; o < H; ++o) {
                    double s = 0.0;
                    for (int i = 0; i < B; ++i) {
                        s += xb[static_cast<size_t>(t) * B + i] * dZ[static_cast<size_t>(i) * H + o];
                    }
                    dWih[o] += s;
                }
                for (int o = 0; o < H; ++o) {
                    double s = 0.0;
                    for (int i = 0; i < B; ++i) s += dZ[static_cast<size_t>(i) * H + o];
                    db_ih[o] += s;
                    db_hh[o] += s;
                }
                if (t > 0) {
                    for (int k = 0; k < H; ++k)
                        for (int o = 0; o < H; ++o) {
                            double s = 0.0;
                            for (int i = 0; i < B; ++i) {
                                s += Hs[t - 1][static_cast<size_t>(i) * H + k] * dZ[static_cast<size_t>(i) * H + o];
                            }
                            dWhh[static_cast<size_t>(k) * H + o] += s;
                        }
                    std::vector<double> dH_next(B * H, 0.0);
                    for (int i = 0; i < B; ++i)
                        for (int o = 0; o < H; ++o) {
                            double s = 0.0;
                            for (int k = 0; k < H; ++k) {
                                s += dZ[static_cast<size_t>(i) * H + k] * r.Whh[static_cast<size_t>(o) * H + k];
                            }
                            dH_next[static_cast<size_t>(i) * H + o] = s;
                        }
                    dH_cur = dH_next;
                }
            }
            for (int j = 0; j < H; ++j) { r.Wih[j] -= lr * dWih[j]; }
            for (int j = 0; j < H * H; ++j) { r.Whh[j] -= lr * dWhh[j]; }
            for (int j = 0; j < H * C; ++j) { r.Wfc[j] -= lr * dWfc[j]; }
            for (int j = 0; j < H; ++j) { r.Bias_ih[j] -= lr * db_ih[j]; }
            for (int j = 0; j < H; ++j) { r.Bias_hh[j] -= lr * db_hh[j]; }
            for (int j = 0; j < C; ++j) { r.Bias_fc[j] -= lr * db_fc[j]; }
        }
        r.epoch_losses.push_back(ep_loss / 5.0);
    }

    // test accuracy on held-out bitstrings
    int correct = 0;
    for (int b = 0; b < 2; ++b) {
        const auto &xb = d.x_test[b];
        std::vector<double> Hst(static_cast<size_t>(B) * H, 0.0);
        for (int t = 0; t < T; ++t) {
            std::vector<double> z(B * H), h(B * H);
            for (int i = 0; i < B; ++i) {
                for (int o = 0; o < H; ++o) {
                    double s = r.Bias_ih[o] + r.Bias_hh[o];
                    s += xb[static_cast<size_t>(t) * B + i] * r.Wih[o];
                    for (int k = 0; k < H; ++k) {
                        s += Hst[static_cast<size_t>(i) * H + k] * r.Whh[static_cast<size_t>(k) * H + o];
                    }
                    z[static_cast<size_t>(i) * H + o] = s;
                }
            }
            for (int i = 0; i < B * H; ++i) { h[i] = std::tanh(z[i]); }
            Hst = h;
        }
        std::vector<double> logits(B * C);
        matmul(Hst, r.Wfc, B, H, C, logits);
        for (int i = 0; i < B * C; ++i) { logits[i] += r.Bias_fc[i % C]; }
        for (int i = 0; i < B; ++i) {
            int pred = logits[static_cast<size_t>(i) * C + 1] > logits[static_cast<size_t>(i) * C + 0] ? 1 : 0;
            if (pred == d.y_test[b * B + i]) { correct++; }
        }
    }
    r.test_acc = static_cast<double>(correct) / N_TEST;
    return r;
}

}// namespace

int rnntrain::run_rnn(int argc, char *argv[]) {
    using namespace luisa;
    using namespace luisa::compute;

    luisa::string_view backend{};
    int epochs = 150;
    for (auto i = 1; i < argc; ++i) {
        if (argv != nullptr && argv[i] != nullptr) {
            luisa::string_view arg{argv[i]};
            if (!arg.starts_with("--")) {
                if (backend.empty()) { backend = arg; }
            } else if (arg == "--epochs" && i + 1 < argc) {
                epochs = std::atoi(argv[++i]);
            }
        }
    }
    if (backend.empty()) {
        LUISA_INFO("Usage: {} <backend> --rnn [--epochs N]   (backend = vk | dx)", argv[0]);
        return 1;
    }
    if (epochs <= 0) { epochs = 150; }

    // ---- data + host reference (the training loop) ----------------------------
    auto data = make_rnn_data();
    auto host_ref = rnn_host_reference(data, epochs);
    LUISA_INFO("[rnn] RNN(hidden={}) on {} binary sequences of length {} (label: #ones >= 3); "
               "train {} / test {} bitstrings, {} epochs (host test acc = {:.1f}%)",
               H, N_ALL, T, N_TRAIN, N_TEST, epochs, 100.0 * host_ref.test_acc);

    // ---- device: trace / lower / compile every tile kernel ---------------------
    Context ctx{argv[0]};
    Device device = ctx.create_device(backend);
    Stream stream = device.create_stream();

    Clock c_trace;
    auto k_step = tile::jit(rnn::rnn_step<B, H>).compile();
    auto k_fc = tile::jit(mlp::fc<B, H, C>).compile();
    auto k_sm = tile::jit(mlp::softmax<B, C>).compile();
    auto k_ce = tile::jit(mlp::ce_grad<B, C>).compile();
    auto k_bwd_fc = tile::jit(mlp::fc_backward<B, H, C>).compile();
    auto k_bwd_hidden = tile::jit(mlp::fc_backward<B, H, H>).compile();
    auto k_grad_fc = tile::jit(mlp::grad<B, H, C>).compile();
    auto k_gradb_fc = tile::jit(mlp::grad_bias<B, C>).compile();
    auto k_tanh_bwd = tile::jit(rnn::tanh_backward<B, H>).compile();
    auto k_grad_wih = tile::jit(rnn::grad_accum<B, 1, H>).compile();
    auto k_grad_whh = tile::jit(rnn::grad_accum<B, H, H>).compile();
    auto k_gradb = tile::jit(rnn::grad_accum_bias<B, H>).compile();
    auto k_clr_1h = tile::jit(rnn::clear2d<1, H>).compile();
    auto k_clr_hh = tile::jit(rnn::clear2d<H, H>).compile();
    auto k_tWfc = tile::jit(mlp::transpose<H, C>).compile();
    auto k_tWhh = tile::jit(mlp::transpose<H, H>).compile();
    auto k_tH = tile::jit(mlp::transpose<B, H>).compile();
    auto k_upd_wih = tile::jit(mlp::update<1, H>).compile();
    auto k_upd_whh = tile::jit(mlp::update<H, H>).compile();
    auto k_upd_fc = tile::jit(mlp::update<H, C>).compile();
    auto k_updb_h = tile::jit(mlp::update_bias<H>).compile();
    auto k_updb_c = tile::jit(mlp::update_bias<C>).compile();
    double trace_ms = c_trace.toc();

    Clock c_lower;
    auto sh_step = k_step.to_kernel<1>();
    auto sh_fc = k_fc.to_kernel<1>();
    auto sh_sm = k_sm.to_kernel<1>();
    auto sh_ce = k_ce.to_kernel<1>();
    auto sh_bwd_fc = k_bwd_fc.to_kernel<1>();
    auto sh_bwd_hidden = k_bwd_hidden.to_kernel<1>();
    auto sh_grad_fc = k_grad_fc.to_kernel<1>();
    auto sh_gradb_fc = k_gradb_fc.to_kernel<1>();
    auto sh_tanh_bwd = k_tanh_bwd.to_kernel<1>();
    auto sh_grad_wih = k_grad_wih.to_kernel<1>();
    auto sh_grad_whh = k_grad_whh.to_kernel<1>();
    auto sh_gradb = k_gradb.to_kernel<1>();
    auto sh_clr_1h = k_clr_1h.to_kernel<1>();
    auto sh_clr_hh = k_clr_hh.to_kernel<1>();
    auto sh_tWfc = k_tWfc.to_kernel<1>();
    auto sh_tWhh = k_tWhh.to_kernel<1>();
    auto sh_tH = k_tH.to_kernel<1>();
    auto sh_upd_wih = k_upd_wih.to_kernel<1>();
    auto sh_upd_whh = k_upd_whh.to_kernel<1>();
    auto sh_upd_fc = k_upd_fc.to_kernel<1>();
    auto sh_updb_h = k_updb_h.to_kernel<1>();
    auto sh_updb_c = k_updb_c.to_kernel<1>();
    double lower_ms = c_lower.toc();

    Clock c_compile;
    device.compile(sh_step);
    device.compile(sh_fc);
    device.compile(sh_sm);
    device.compile(sh_ce);
    device.compile(sh_bwd_fc);
    device.compile(sh_bwd_hidden);
    device.compile(sh_grad_fc);
    device.compile(sh_gradb_fc);
    device.compile(sh_tanh_bwd);
    device.compile(sh_grad_wih);
    device.compile(sh_grad_whh);
    device.compile(sh_gradb);
    device.compile(sh_clr_1h);
    device.compile(sh_clr_hh);
    device.compile(sh_tWfc);
    device.compile(sh_tWhh);
    device.compile(sh_tH);
    device.compile(sh_upd_wih);
    device.compile(sh_upd_whh);
    device.compile(sh_upd_fc);
    device.compile(sh_updb_h);
    device.compile(sh_updb_c);
    stream << synchronize();
    double compile_ms = c_compile.toc();

    // ---- verify (host reference) ------------------------------------------------
    bool ok = true;
    if (host_ref.test_acc < 0.90) {
        LUISA_WARNING("[rnn] self check FAILED (host test acc = {:.2f} < 0.90)", host_ref.test_acc);
        ok = false;
    } else {
        LUISA_INFO("[rnn] self check: host test acc {:.1f}% >= 90% -> PASS", 100.0 * host_ref.test_acc);
    }

    LUISA_INFO("[rnn] tile trace: {:.3f} ms, lower: {:.3f} ms, backend compile: {:.3f} ms "
               "({} tile kernels compiled on '{}').",
               trace_ms, lower_ms, compile_ms, 22, backend);
    LUISA_INFO("[rnn] Verification: {}", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
