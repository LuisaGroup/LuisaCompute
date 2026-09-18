// =============================================================================
// rnn.cpp — New Tile DSL RNN sequence classification training
// =============================================================================
// Port of backup_old_tile/examples/tensor/rnn.cpp (the C++ twin of
// rnn_train.py): given all 2^8 binary sequences of length 8, train a tanh RNN
// (hidden 16) + linear head to decide whether a sequence contains at least 3
// ones (a pure counting task).
//
// The verification structure mirrors the old driver exactly:
//   1. every Tile kernel (the rnn_kernels.h / old mlp_kernels.h inventory,
//      22 template instances) is captured and compiled on the backend,
//   2. the host reference (the exact training algorithm in double precision)
//      must reach >= 90% test accuracy — the old driver's threshold.
// In addition — in line with this example's "every kernel is dispatched on
// real device buffers" contract — the trained weights are used for a device
// forward pass over the held-out test minibatches (one rnn_step dispatch per
// timestep, like the old per-step kernel launches), and the device logits are
// checked against the host reference (argmax equality plus a 1e-2 tolerance).
//
// Invoked through example_tensor's main() with the `--rnn` flag:
//   example_tensor <backend> --rnn [--epochs N]
// =============================================================================

#include "rnn.h"
#include "rnn_kernels.h"
#include "tensor_kernels.h"

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <cmath>
#include <cstdlib>

namespace tensor_example::rnntrain {

namespace {

constexpr int B = 32;          // minibatch
constexpr int T = 8;           // sequence length
constexpr int H = 16;          // hidden size
constexpr int C = 2;           // classes
constexpr int N_TRAIN = 5 * B; // 160 sequences (5 minibatches)
constexpr int N_TEST = 2 * B;  // 64 sequences
constexpr int N_ALL = 256;     // all 2^8 bitstrings

// Deterministic PRNG (LCG + Box-Muller), ported from the old mlp_common.h so
// the C++ twin is reproducible.
struct PRNG {
    unsigned s;
    explicit PRNG(unsigned seed) noexcept : s(seed * 2654435761u + 12345u) {}
    float unit() noexcept {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>((s >> 8) & 0xFFFFFFu) / static_cast<float>(0x1000000u);
    }
    float gauss() noexcept {
        auto u1 = std::max(unit(), 1e-6f);
        auto u2 = std::max(unit(), 1e-6f);
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265358979f * u2);
    }
};

struct RnnData {
    luisa::vector<float> x[5];       // [T, B, 1] per minibatch
    luisa::vector<float> xt[5];      // [T, 1, B] per minibatch
    luisa::vector<float> y_onehot[5];// [B, C]
    luisa::vector<int> y_test;       // [N_TEST]
    luisa::vector<float> x_test[2];  // [T, B, 1] per test minibatch
};

RnnData make_rnn_data() {
    luisa::vector<int> y_all(N_ALL);
    luisa::vector<luisa::vector<float>> seq(N_ALL, luisa::vector<float>(T));
    for (int i = 0; i < N_ALL; ++i) {
        int ones = 0;
        for (int t = 0; t < T; ++t) {
            int bit = (i >> (T - 1 - t)) & 1;
            seq[i][t] = static_cast<float>(bit);
            ones += bit;
        }
        y_all[i] = ones >= 3 ? 1 : 0;
    }
    luisa::vector<int> perm(N_ALL);
    for (int i = 0; i < N_ALL; ++i) { perm[i] = i; }
    unsigned s = 42u * 2654435761u + 12345u;
    for (int i = N_ALL - 1; i > 0; --i) {
        s = s * 1664525u + 1013904223u;
        int j = static_cast<int>((s >> 8) & 0xFFFFFFu) % (i + 1);
        luisa::swap(perm[i], perm[j]);
    }

    RnnData d;
    auto fill_block = [&](luisa::vector<float> &x, luisa::vector<float> &xt,
                          luisa::vector<float> &yo, const luisa::vector<int> &ids,
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
        luisa::vector<float> dummy_xt, dummy_y;
        fill_block(d.x_test[b], dummy_xt, dummy_y, perm, 5 * B + b * B);
    }
    d.y_test.resize(N_TEST);
    for (int i = 0; i < N_TEST; ++i) { d.y_test[i] = y_all[perm[5 * B + i]]; }
    return d;
}

// ---- host reference: exact same algorithm as the device, in double ----------
struct RnnHostRef {
    luisa::vector<double> Wih;     // [1, H]
    luisa::vector<double> Whh;     // [H, H]
    luisa::vector<double> Wfc;     // [H, C]
    luisa::vector<double> Bias_ih; // [1, H]
    luisa::vector<double> Bias_hh; // [1, H]
    luisa::vector<double> Bias_fc; // [1, C]
    luisa::vector<double> epoch_losses;
    double test_acc = 0.0;
};

RnnHostRef rnn_host_reference(const RnnData &d, int epochs) {
    RnnHostRef r;
    PRNG rng(0u);
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
    auto matmul = [](const luisa::vector<double> &A, const luisa::vector<double> &X,
                     int M, int K, int N, luisa::vector<double> &Y) {
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
            luisa::vector<double> Hst(static_cast<size_t>(B) * H, 0.0);
            luisa::vector<luisa::vector<double>> Hs(T);
            for (int t = 0; t < T; ++t) {
                luisa::vector<double> z(B * H), h(B * H);
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
            luisa::vector<double> logits(B * C), P(B * C);
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
                for (int c = 0; c < C; ++c) {
                    if (d.y_onehot[b][static_cast<size_t>(i) * C + c] > 0.5) { yi = c; }
                }
                loss += -std::log(std::max(P[static_cast<size_t>(i) * C + yi], 1e-12));
            }
            loss /= B;
            ep_loss += loss;
            luisa::vector<double> G(B * C);
            for (int i = 0; i < B * C; ++i) { G[i] = (P[i] - d.y_onehot[b][i]) / B; }
            // fc backward
            luisa::vector<double> dWfc(static_cast<size_t>(H) * C, 0.0), dH(B * H, 0.0), db_fc(C, 0.0);
            for (int k = 0; k < H; ++k) {
                for (int c = 0; c < C; ++c) {
                    double s = 0.0;
                    for (int i = 0; i < B; ++i) {
                        s += Hst[static_cast<size_t>(i) * H + k] * G[static_cast<size_t>(i) * C + c];
                    }
                    dWfc[static_cast<size_t>(k) * C + c] = s;
                }
            }
            for (int c = 0; c < C; ++c) {
                double s = 0.0;
                for (int i = 0; i < B; ++i) { s += G[static_cast<size_t>(i) * C + c]; }
                db_fc[c] = s;
            }
            for (int i = 0; i < B; ++i) {
                for (int o = 0; o < H; ++o) {
                    double s = 0.0;
                    for (int c = 0; c < C; ++c) {
                        s += G[static_cast<size_t>(i) * C + c] * r.Wfc[static_cast<size_t>(o) * C + c];
                    }
                    dH[static_cast<size_t>(i) * H + o] = s;
                }
            }
            // BPTT
            luisa::vector<double> dWih(H, 0.0), dWhh(static_cast<size_t>(H) * H, 0.0);
            luisa::vector<double> db_ih(H, 0.0), db_hh(H, 0.0);
            luisa::vector<double> dH_cur = dH;
            for (int t = T - 1; t >= 0; --t) {
                luisa::vector<double> dZ(B * H);
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
                    for (int i = 0; i < B; ++i) { s += dZ[static_cast<size_t>(i) * H + o]; }
                    db_ih[o] += s;
                    db_hh[o] += s;
                }
                if (t > 0) {
                    for (int k = 0; k < H; ++k) {
                        for (int o = 0; o < H; ++o) {
                            double s = 0.0;
                            for (int i = 0; i < B; ++i) {
                                s += Hs[t - 1][static_cast<size_t>(i) * H + k] * dZ[static_cast<size_t>(i) * H + o];
                            }
                            dWhh[static_cast<size_t>(k) * H + o] += s;
                        }
                    }
                    luisa::vector<double> dH_next(B * H, 0.0);
                    for (int i = 0; i < B; ++i) {
                        for (int o = 0; o < H; ++o) {
                            double s = 0.0;
                            for (int k = 0; k < H; ++k) {
                                s += dZ[static_cast<size_t>(i) * H + k] * r.Whh[static_cast<size_t>(o) * H + k];
                            }
                            dH_next[static_cast<size_t>(i) * H + o] = s;
                        }
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
        luisa::vector<double> Hst(static_cast<size_t>(B) * H, 0.0);
        for (int t = 0; t < T; ++t) {
            luisa::vector<double> z(B * H), h(B * H);
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
        luisa::vector<double> logits(B * C);
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

// Float forward pass over one test minibatch with the trained weights;
// returns logits[B, C]. Used as the reference for the device forward pass.
luisa::vector<float> forward_logits_f32(const RnnHostRef &r, const luisa::vector<float> &xb) {
    luisa::vector<float> Hst(static_cast<size_t>(B) * H, 0.0f);
    for (int t = 0; t < T; ++t) {
        luisa::vector<float> z(B * H), h(B * H);
        for (int i = 0; i < B; ++i) {
            for (int o = 0; o < H; ++o) {
                float s = static_cast<float>(r.Bias_ih[o] + r.Bias_hh[o]);
                s += xb[static_cast<size_t>(t) * B + i] * static_cast<float>(r.Wih[o]);
                for (int k = 0; k < H; ++k) {
                    s += Hst[static_cast<size_t>(i) * H + k] * static_cast<float>(r.Whh[static_cast<size_t>(k) * H + o]);
                }
                z[static_cast<size_t>(i) * H + o] = s;
            }
        }
        for (int i = 0; i < B * H; ++i) { h[i] = std::tanh(z[i]); }
        Hst = h;
    }
    luisa::vector<float> logits(static_cast<size_t>(B) * C, 0.0f);
    for (int i = 0; i < B; ++i) {
        for (int c = 0; c < C; ++c) {
            float s = static_cast<float>(r.Bias_fc[c]);
            for (int k = 0; k < H; ++k) {
                s += Hst[static_cast<size_t>(i) * H + k] * static_cast<float>(r.Wfc[static_cast<size_t>(k) * C + c]);
            }
            logits[static_cast<size_t>(i) * C + c] = s;
        }
    }
    return logits;
}

}// namespace

int run_rnn(int argc, char *argv[]) {
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
        LUISA_INFO("Usage: {} <backend> --rnn [--epochs N]", argv[0]);
        return 1;
    }
    if (epochs <= 0) { epochs = 150; }

    // ---- data + host reference (the training loop) ----------------------------
    auto data = make_rnn_data();
    auto host_ref = rnn_host_reference(data, epochs);
    LUISA_INFO("[rnn] RNN(hidden={}) on {} binary sequences of length {} (label: #ones >= 3); "
               "train {} / test {} bitstrings, {} epochs (host test acc = {:.1f}%)",
               H, N_ALL, T, N_TRAIN, N_TEST, epochs, 100.0 * host_ref.test_acc);

    // ---- device: compile the full old kernel inventory -----------------------
    Context ctx{argv[0]};
    Device device = ctx.create_device(backend);
    Stream stream = device.create_stream();

    luisa::Clock c_compile;
    auto sh_step = compile_tile(device, make_rnn_step_kernel<B, H>(), "rnn_step");
    record("rnn_compile_step", static_cast<bool>(sh_step),
           sh_step ? luisa::string_view{sh_step.metadata().realization} : luisa::string_view{sh_step.metadata().error});
    auto sh_fc = compile_tile(device, make_rnn_fc_kernel<B, H, C>(), "rnn_fc");
    record("rnn_compile_fc", static_cast<bool>(sh_fc),
           sh_fc ? luisa::string_view{sh_fc.metadata().realization} : luisa::string_view{sh_fc.metadata().error});
    int compile_failures = (sh_step ? 0 : 1) + (sh_fc ? 0 : 1);
    auto compile_and_record = [&](luisa::string_view name, tile::Kernel &&kernel) {
        auto shader = compile_tile(device, kernel, name);
        record(luisa::format("rnn_compile_{}", name), static_cast<bool>(shader),
               shader ? luisa::string_view{shader.metadata().realization} : luisa::string_view{shader.metadata().error});
        if (!shader) { compile_failures++; }
    };
    using namespace tensor_example::rnntrain;
    compile_and_record("softmax", make_rnn_softmax_kernel<B, C>());
    compile_and_record("ce_grad", make_rnn_ce_grad_kernel<B, C>());
    compile_and_record("fc_backward_hc", make_rnn_fc_backward_kernel<B, H, C>());
    compile_and_record("fc_backward_hh", make_rnn_fc_backward_kernel<B, H, H>());
    compile_and_record("grad_fc", make_rnn_grad_kernel<B, H, C>());
    compile_and_record("gradb_fc", make_rnn_grad_bias_kernel<B, C>());
    compile_and_record("tanh_backward", make_rnn_tanh_backward_kernel<B, H>());
    compile_and_record("grad_wih", make_rnn_grad_accum_kernel<B, 1, H>());
    compile_and_record("grad_whh", make_rnn_grad_accum_kernel<B, H, H>());
    compile_and_record("gradb", make_rnn_grad_accum_bias_kernel<B, H>());
    compile_and_record("clear_1h", make_rnn_clear2d_kernel<1, H>());
    compile_and_record("clear_hh", make_rnn_clear2d_kernel<H, H>());
    compile_and_record("transpose_wfc", make_rnn_transpose_kernel<H, C>());
    compile_and_record("transpose_whh", make_rnn_transpose_kernel<H, H>());
    compile_and_record("transpose_h", make_rnn_transpose_kernel<B, H>());
    compile_and_record("update_wih", make_rnn_update_kernel<1, H>());
    compile_and_record("update_whh", make_rnn_update_kernel<H, H>());
    compile_and_record("update_fc", make_rnn_update_kernel<H, C>());
    compile_and_record("updateb_h", make_rnn_update_bias_kernel<H>());
    compile_and_record("updateb_c", make_rnn_update_bias_kernel<C>());
    double compile_ms = c_compile.toc();
    if (compile_failures != 0) {
        record("rnn_compile_all", false, luisa::format("{} of 22 kernel instances failed to compile", compile_failures));
        return 1;
    }
    record("rnn_compile_all", true, luisa::format("22 kernel instances in {:.3f} ms", compile_ms));

    // ---- verify (host reference; the old driver's gate) ------------------------
    if (host_ref.test_acc < 0.90) {
        record("rnn_host_accuracy", false,
               luisa::format("host test acc = {:.2f} < 0.90", host_ref.test_acc));
    } else {
        record("rnn_host_accuracy", true,
               luisa::format("host test acc {:.1f}% >= 90%", 100.0 * host_ref.test_acc));
    }

    // ---- device forward pass over the held-out test minibatches --------------
    // One rnn_step dispatch per timestep (the old design's per-step launches),
    // ping-ponging the hidden state between two disjoint buffers, then one fc
    // dispatch for the logits. Compares argmax and values with the host.
    auto to_f32 = [](const luisa::vector<double> &v) {
        luisa::vector<float> f(v.size());
        for (size_t i = 0; i < v.size(); ++i) { f[i] = static_cast<float>(v[i]); }
        return f;
    };
    auto bufWih = device.create_buffer<float>(H);
    auto bufWhh = device.create_buffer<float>(H * H);
    auto bufWfc = device.create_buffer<float>(H * C);
    auto bufBiasIh = device.create_buffer<float>(H);
    auto bufBiasHh = device.create_buffer<float>(H);
    auto bufBiasFc = device.create_buffer<float>(C);
    luisa::vector<float> ones(B, 1.0f), zeros_h(static_cast<size_t>(B) * H, 0.0f);
    auto bufOnes = device.create_buffer<float>(B);
    auto bufH0 = device.create_buffer<float>(B * H);
    auto bufH1 = device.create_buffer<float>(B * H);
    auto bufX = device.create_buffer<float>(T * B);
    auto bufLogits = device.create_buffer<float>(B * C);
    // Host staging vectors must outlive the stream execution of the uploads.
    auto fWih = to_f32(host_ref.Wih);
    auto fWhh = to_f32(host_ref.Whh);
    auto fWfc = to_f32(host_ref.Wfc);
    auto fBiasIh = to_f32(host_ref.Bias_ih);
    auto fBiasHh = to_f32(host_ref.Bias_hh);
    auto fBiasFc = to_f32(host_ref.Bias_fc);
    stream << bufWih.copy_from(luisa::span{fWih}) << bufWhh.copy_from(luisa::span{fWhh})
           << bufWfc.copy_from(luisa::span{fWfc}) << bufBiasIh.copy_from(luisa::span{fBiasIh})
           << bufBiasHh.copy_from(luisa::span{fBiasHh}) << bufBiasFc.copy_from(luisa::span{fBiasFc})
           << bufOnes.copy_from(luisa::span{ones}) << synchronize();

    double err_logits = 0.0;
    int argmax_matches = 0;
    for (int mb = 0; mb < 2; ++mb) {
        stream << bufX.copy_from(luisa::span{data.x_test[mb]})
               << bufH0.copy_from(luisa::span{zeros_h}) << synchronize();
        auto *cur = &bufH0;
        auto *next = &bufH1;
        for (int t = 0; t < T; ++t) {
            stream << sh_step(bufX.view(static_cast<size_t>(t) * B, B), *cur,
                              bufWih, bufWhh, bufBiasIh, bufBiasHh, bufOnes, *next)
                          .dispatch()
                   << synchronize();
            luisa::swap(cur, next);
        }
        stream << sh_fc(*cur, bufWfc, bufBiasFc, bufOnes, bufLogits).dispatch() << synchronize();
        luisa::vector<float> logits(static_cast<size_t>(B) * C);
        stream << bufLogits.copy_to(luisa::span{logits}) << synchronize();

        auto ref = forward_logits_f32(host_ref, data.x_test[mb]);
        for (int i = 0; i < B; ++i) {
            for (int c = 0; c < C; ++c) {
                err_logits = std::max(err_logits,
                                      std::fabs(static_cast<double>(logits[static_cast<size_t>(i) * C + c]) -
                                                static_cast<double>(ref[static_cast<size_t>(i) * C + c])));
            }
            int pred_dev = logits[static_cast<size_t>(i) * C + 1] > logits[static_cast<size_t>(i) * C + 0] ? 1 : 0;
            int pred_ref = ref[static_cast<size_t>(i) * C + 1] > ref[static_cast<size_t>(i) * C + 0] ? 1 : 0;
            if (pred_dev == pred_ref) { argmax_matches++; }
        }
    }
    check("rnn_device_logits", err_logits, 1e-2);
    record("rnn_device_argmax", argmax_matches == N_TEST,
           luisa::format("{}/{} test predictions match the host reference", argmax_matches, N_TEST));

    LUISA_INFO("[rnn] backend compile: {:.3f} ms (22 tile kernel instances on '{}').", compile_ms, backend);
    return failure_count() == 0 ? 0 : 1;
}

}// namespace tensor_example::rnntrain
