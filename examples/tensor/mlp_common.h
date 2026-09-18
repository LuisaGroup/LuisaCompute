// =============================================================================
// mlp_common.h — shared data builders, host reference and device trainer
// =============================================================================
// Port of `backup_old_tile/examples/tensor/mlp_common.h`, used by both the
// `--mlp` (3-layer) and `--mnist` (2-layer) drivers.  All training data is
// generated on the host with a small deterministic PRNG, so the C++ twin runs
// offline with no downloads (mirroring the synthetic modes of
// mlp_train.py / mnist_train.py).  The host reference below runs the exact
// same minibatch-SGD / cross-entropy algorithm as the device tile kernels (in
// double precision) so the drivers can verify the device against it.
//
// Weights and biases are kept separate (W[K,O], Bias[1,O]); the device applies
// the bias with a second GEMM against an all-ones buffer (see mlp_kernels.h).
//
// Additions over the old header (host-side only, no semantic changes):
//   * init_parameters() — factors the PRNG(0)*0.05 weight init out of the host
//     reference so the device buffers start from the exact same values,
//   * evaluate() — a forward-only host pass (double precision) used to score
//     the device-trained parameters (final train loss / test accuracy),
//   * train_on_device() — a layer-generic device training loop that captures
//     the mlp:: kernels, compiles them through tensor_example::compile_tile
//     and dispatches the whole minibatch-SGD algorithm on real buffers.
// =============================================================================

#pragma once

#include "mlp_kernels.h"
#include "tensor_kernels.h"

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <luisa/core/stl/vector.h>

namespace mlpcommon {

namespace lc = luisa::compute;

// ---------------------------------------------------------------------------
// deterministic PRNG (LCG + Box-Muller), so the C++ twin is reproducible
// ---------------------------------------------------------------------------
struct PRNG {
    unsigned s;
    explicit PRNG(unsigned seed) : s(seed * 2654435761u + 12345u) {}
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

// ---------------------------------------------------------------------------
// problem description shared by the drivers
// ---------------------------------------------------------------------------
struct MlpHyper {
    int n_train = 1024;
    int n_test = 384;
    int batch = 32;
    int epochs = 60;
    float lr = 0.1f;
    int num_inputs = 50;
    int num_outputs = 4;
    luisa::vector<int> widths;    // hidden layer widths (excluding input/output)
    int seed = 123;
};

// ---------------------------------------------------------------------------
// dataset (all matrices row-major, raw features only — no bias column)
// ---------------------------------------------------------------------------
struct MlpData {
    luisa::vector<float> x_train;    // [n_train, num_inputs]
    luisa::vector<float> xT_train;   // [num_inputs, n_train]
    luisa::vector<int> y_train;      // [n_train]
    luisa::vector<float> y_onehot;   // [n_train, num_outputs]
    luisa::vector<float> x_test;     // [n_test, num_inputs]
    luisa::vector<int> y_test;       // [n_test]
};

// XOR-style classification task (mlp_train.py make_dataset): the class is the
// quadrant of the two informative features, the remaining features are noise.
inline MlpData make_xor_data(const MlpHyper &hp) {
    const int n = hp.n_train + hp.n_test;
    MlpData d;
    d.x_train.resize(static_cast<size_t>(hp.n_train) * hp.num_inputs);
    d.y_train.resize(hp.n_train);
    d.y_test.resize(hp.n_test);
    d.x_test.resize(static_cast<size_t>(hp.n_test) * hp.num_inputs);
    PRNG rng(static_cast<unsigned>(hp.seed));
    luisa::vector<float> x(n * hp.num_inputs);
    luisa::vector<int> y(n);
    for (int i = 0; i < n; ++i) {
        float z0 = 0.8f * rng.gauss();
        float z1 = 0.8f * rng.gauss();
        x[static_cast<size_t>(i) * hp.num_inputs + 0] = z0;
        x[static_cast<size_t>(i) * hp.num_inputs + 1] = z1;
        for (int k = 2; k < hp.num_inputs; ++k) {
            x[static_cast<size_t>(i) * hp.num_inputs + k] = rng.gauss();
        }
        y[i] = (z0 > 0.f ? 1 : 0) + 2 * (z1 > 0.f ? 1 : 0);
    }
    for (int i = 0; i < hp.n_train; ++i) {
        d.y_train[i] = y[i];
        for (int k = 0; k < hp.num_inputs; ++k) {
            d.x_train[static_cast<size_t>(i) * hp.num_inputs + k] =
                x[static_cast<size_t>(i) * hp.num_inputs + k];
        }
    }
    for (int i = 0; i < hp.n_test; ++i) {
        d.y_test[i] = y[hp.n_train + i];
        for (int k = 0; k < hp.num_inputs; ++k) {
            d.x_test[static_cast<size_t>(i) * hp.num_inputs + k] =
                x[static_cast<size_t>(hp.n_train + i) * hp.num_inputs + k];
        }
    }
    return d;
}

// Synthetic MNIST stand-in (mnist_train.py make_synthetic_mnist): fixed random
// per-class templates + Gaussian noise.  The driver uses a small 8x8 = 64
// input stand-in (matching the repo's TinyCNN scale) so the whole tile stays in
// on-chip shared memory.
inline MlpData make_synth_mnist_data(const MlpHyper &hp) {
    const int n = hp.n_train + hp.n_test;
    MlpData d;
    d.x_train.resize(static_cast<size_t>(hp.n_train) * hp.num_inputs);
    d.y_train.resize(hp.n_train);
    d.y_test.resize(hp.n_test);
    d.x_test.resize(static_cast<size_t>(hp.n_test) * hp.num_inputs);
    PRNG rng(static_cast<unsigned>(hp.seed));
    luisa::vector<float> templates(static_cast<size_t>(hp.num_outputs) * hp.num_inputs);
    for (auto &v : templates) { v = rng.gauss(); }
    luisa::vector<int> y(n);
    luisa::vector<float> x(static_cast<size_t>(n) * hp.num_inputs);
    for (int i = 0; i < n; ++i) {
        y[i] = static_cast<int>(rng.unit() * hp.num_outputs);
        y[i] = std::min(y[i], hp.num_outputs - 1);
        for (int k = 0; k < hp.num_inputs; ++k) {
            x[static_cast<size_t>(i) * hp.num_inputs + k] =
                templates[static_cast<size_t>(y[i]) * hp.num_inputs + k] +
                0.35f * rng.gauss();
        }
    }
    for (int i = 0; i < hp.n_train; ++i) {
        d.y_train[i] = y[i];
        for (int k = 0; k < hp.num_inputs; ++k) {
            d.x_train[static_cast<size_t>(i) * hp.num_inputs + k] =
                x[static_cast<size_t>(i) * hp.num_inputs + k];
        }
    }
    for (int i = 0; i < hp.n_test; ++i) {
        d.y_test[i] = y[hp.n_train + i];
        for (int k = 0; k < hp.num_inputs; ++k) {
            d.x_test[static_cast<size_t>(i) * hp.num_inputs + k] =
                x[static_cast<size_t>(hp.n_train + i) * hp.num_inputs + k];
        }
    }
    return d;
}

// Fill the host-side transpose of the training inputs and the one-hot labels.
inline void finalize_data(MlpHyper &hp, MlpData &d) {
    const int k0 = hp.num_inputs;
    d.xT_train.assign(static_cast<size_t>(k0) * hp.n_train, 0.0f);
    for (int i = 0; i < hp.n_train; ++i) {
        for (int k = 0; k < k0; ++k) {
            d.xT_train[static_cast<size_t>(k) * hp.n_train + i] =
                d.x_train[static_cast<size_t>(i) * k0 + k];
        }
    }
    d.y_onehot.assign(static_cast<size_t>(hp.n_train) * hp.num_outputs, 0.0f);
    for (int i = 0; i < hp.n_train; ++i) {
        d.y_onehot[static_cast<size_t>(i) * hp.num_outputs + d.y_train[i]] = 1.0f;
    }
}

// ---------------------------------------------------------------------------
// parameter init shared by the host reference and the device buffers:
// PRNG(0) * 0.05, layer by layer, weights then biases (same stream order as
// the old in-reference init, so all three parameter sets are identical).
// ---------------------------------------------------------------------------
struct HostMlpParam {
    luisa::vector<luisa::vector<double>> W;     // [layer][K*O]
    luisa::vector<luisa::vector<double>> Bias;  // [layer][O]
};

inline HostMlpParam init_parameters(const MlpHyper &hp) {
    const int L = static_cast<int>(hp.widths.size()) + 1;
    HostMlpParam p;
    p.W.resize(L);
    p.Bias.resize(L);
    luisa::vector<int> K(L), O(L);
    int prev = hp.num_inputs;
    for (int l = 0; l < L; ++l) {
        K[l] = prev;
        O[l] = (l == L - 1) ? hp.num_outputs : hp.widths[l];
        prev = O[l];
    }
    PRNG w_rng(0u);
    for (int l = 0; l < L; ++l) {
        p.W[l].assign(static_cast<size_t>(K[l]) * O[l], 0.0);
        p.Bias[l].assign(static_cast<size_t>(O[l]), 0.0);
        for (auto &v : p.W[l]) { v = 0.05 * w_rng.gauss(); }
        for (auto &v : p.Bias[l]) { v = 0.05 * w_rng.gauss(); }
    }
    return p;
}

// ---------------------------------------------------------------------------
// generic host reference: exact same minibatch-SGD / cross-entropy algorithm
// as the device, in double precision, over L layers (widths = hidden sizes).
// Returns final weights [L][K,O], biases [L][O], epoch-average CE losses and
// the test accuracy.
// ---------------------------------------------------------------------------
struct HostMlpResult {
    luisa::vector<luisa::vector<double>> W;      // [layer][K*O]
    luisa::vector<luisa::vector<double>> Bias;   // [layer][O]
    luisa::vector<double> epoch_losses;
    double test_acc = 0.0;
};

inline double relu_host(double x) noexcept { return x > 0.0 ? x : 0.0; }

inline HostMlpResult mlp_host_reference(const MlpHyper &hp, const MlpData &d) {
    const int L = static_cast<int>(hp.widths.size()) + 1;// hidden + output layers
    HostMlpResult r;
    auto initial = init_parameters(hp);
    r.W = std::move(initial.W);
    r.Bias = std::move(initial.Bias);
    // K_l = input dim; O_l = output dim
    luisa::vector<int> K(L), O(L);
    {
        int prev = hp.num_inputs;
        for (int l = 0; l < L; ++l) {
            K[l] = prev;
            O[l] = (l == L - 1) ? hp.num_outputs : hp.widths[l];
            prev = O[l];
        }
    }
    const int B = hp.batch;
    const int n_b = hp.n_train / B;
    for (int ep = 0; ep < hp.epochs; ++ep) {
        double ep_loss = 0.0;
        for (int b = 0; b < n_b; ++b) {
            const int base = b * B;
            // forward
            luisa::vector<luisa::vector<double>> A(L);
            luisa::vector<luisa::vector<double>> Z(L);
            A[0].resize(static_cast<size_t>(B) * K[0]);
            for (int i = 0; i < B; ++i) {
                for (int k = 0; k < K[0]; ++k) {
                    A[0][static_cast<size_t>(i) * K[0] + k] =
                        d.x_train[static_cast<size_t>(base + i) * K[0] + k];
                }
            }
            for (int l = 0; l < L; ++l) {
                Z[l].assign(static_cast<size_t>(B) * O[l], 0.0);
                for (int i = 0; i < B; ++i) {
                    for (int o = 0; o < O[l]; ++o) {
                        double s = r.Bias[l][o];
                        for (int k = 0; k < K[l]; ++k) {
                            s += A[l][static_cast<size_t>(i) * K[l] + k] * r.W[l][static_cast<size_t>(k) * O[l] + o];
                        }
                        Z[l][static_cast<size_t>(i) * O[l] + o] = s;
                    }
                }
                if (l < L - 1) {
                    A[l + 1].assign(static_cast<size_t>(B) * O[l], 0.0);
                    for (int i = 0; i < B; ++i) {
                        for (int o = 0; o < O[l]; ++o) {
                            A[l + 1][static_cast<size_t>(i) * O[l] + o] =
                                relu_host(Z[l][static_cast<size_t>(i) * O[l] + o]);
                        }
                    }
                }
            }
            // softmax + cross entropy
            const int C = hp.num_outputs;
            luisa::vector<double> P(static_cast<size_t>(B) * C);
            double loss = 0.0;
            for (int i = 0; i < B; ++i) {
                double mx = -1e30;
                for (int c = 0; c < C; ++c) mx = std::max(mx, Z[L - 1][static_cast<size_t>(i) * C + c]);
                double sum = 0.0;
                for (int c = 0; c < C; ++c) {
                    double e = std::exp(Z[L - 1][static_cast<size_t>(i) * C + c] - mx);
                    P[static_cast<size_t>(i) * C + c] = e;
                    sum += e;
                }
                for (int c = 0; c < C; ++c) P[static_cast<size_t>(i) * C + c] /= sum;
                int yi = d.y_train[base + i];
                loss += -std::log(std::max(P[static_cast<size_t>(i) * C + yi], 1e-12));
            }
            loss /= B;
            ep_loss += loss;
            // G = (P - Y) / B
            luisa::vector<double> G(static_cast<size_t>(B) * C);
            for (int i = 0; i < B; ++i) {
                for (int c = 0; c < C; ++c) {
                    G[static_cast<size_t>(i) * C + c] =
                        (P[static_cast<size_t>(i) * C + c] -
                         d.y_onehot[static_cast<size_t>(base + i) * C + c]) / B;
                }
            }
            // backward
            luisa::vector<luisa::vector<double>> dW(L);
            luisa::vector<luisa::vector<double>> db(L);
            luisa::vector<double> dZ;
            for (int l = L - 1; l >= 0; --l) {
                dW[l].assign(static_cast<size_t>(K[l]) * O[l], 0.0);
                db[l].assign(static_cast<size_t>(O[l]), 0.0);
                // copy by value: dZ is reassigned below for the next layer, and
                // a reference would be invalidated by that reassignment
                const luisa::vector<double> cur_dZ = (l == L - 1) ? G : dZ;
                // dW[l] = A[l]^T @ dZ
                for (int k = 0; k < K[l]; ++k) {
                    for (int o = 0; o < O[l]; ++o) {
                        double s = 0.0;
                        for (int i = 0; i < B; ++i) {
                            s += A[l][static_cast<size_t>(i) * K[l] + k] *
                                 cur_dZ[static_cast<size_t>(i) * O[l] + o];
                        }
                        dW[l][static_cast<size_t>(k) * O[l] + o] = s;
                    }
                }
                // db[l] = sum_i dZ[i,o]
                for (int o = 0; o < O[l]; ++o) {
                    double s = 0.0;
                    for (int i = 0; i < B; ++i) {
                        s += cur_dZ[static_cast<size_t>(i) * O[l] + o];
                    }
                    db[l][o] = s;
                }
                if (l > 0) {
                    // dA_prev = dZ @ W[l]^T  (no bias row to drop); the
                    // result has K[l] = O[l-1] columns (the input dim of
                    // layer l, i.e. the output dim of layer l-1)
                    const int prev_K = K[l];
                    dZ.assign(static_cast<size_t>(B) * prev_K, 0.0);
                    for (int i = 0; i < B; ++i) {
                        for (int k = 0; k < prev_K; ++k) {
                            double s = 0.0;
                            for (int o = 0; o < O[l]; ++o) {
                                s += cur_dZ[static_cast<size_t>(i) * O[l] + o] *
                                     r.W[l][static_cast<size_t>(k) * O[l] + o];
                            }
                            double z = Z[l - 1][static_cast<size_t>(i) * prev_K + k];
                            double rl = relu_host(z);
                            double step = std::min(rl / 1e-8, 1.0);
                            dZ[static_cast<size_t>(i) * prev_K + k] = s * step;
                        }
                    }
                }
            }
            // update W -= lr * dW; Bias -= lr * db (dW/db include 1/B from G)
            const double lr_eff = hp.lr;
            for (int l = 0; l < L; ++l) {
                for (size_t j = 0; j < r.W[l].size(); ++j) { r.W[l][j] -= lr_eff * dW[l][j]; }
                for (int o = 0; o < O[l]; ++o) { r.Bias[l][o] -= lr_eff * db[l][o]; }
            }
        }
        r.epoch_losses.push_back(ep_loss / n_b);
    }
    // test accuracy
    const int C = hp.num_outputs;
    int correct = 0;
    int n_b_te = hp.n_test / B;
    for (int b = 0; b < n_b_te; ++b) {
        const int base = b * B;
        luisa::vector<double> act(static_cast<size_t>(B) * K[0]);
        for (int i = 0; i < B; ++i)
            for (int k = 0; k < K[0]; ++k)
                act[static_cast<size_t>(i) * K[0] + k] = d.x_test[static_cast<size_t>(base + i) * K[0] + k];
        int prev = K[0];
        for (int l = 0; l < L; ++l) {
            luisa::vector<double> z(static_cast<size_t>(B) * O[l]);
            for (int i = 0; i < B; ++i)
                for (int o = 0; o < O[l]; ++o) {
                    double s = r.Bias[l][o];
                    for (int k = 0; k < prev; ++k) s += act[static_cast<size_t>(i) * prev + k] * r.W[l][static_cast<size_t>(k) * O[l] + o];
                    z[static_cast<size_t>(i) * O[l] + o] = s;
                }
            if (l < L - 1) {
                luisa::vector<double> na(static_cast<size_t>(B) * O[l]);
                for (int i = 0; i < B; ++i)
                    for (int o = 0; o < O[l]; ++o) na[static_cast<size_t>(i) * O[l] + o] = relu_host(z[static_cast<size_t>(i) * O[l] + o]);
                act = std::move(na);
                prev = O[l];
            } else {
                for (int i = 0; i < B; ++i) {
                    int best = 0;
                    for (int c = 1; c < C; ++c) if (z[static_cast<size_t>(i) * C + c] > z[static_cast<size_t>(i) * C + best]) best = c;
                    if (best == d.y_test[base + i]) correct++;
                }
            }
        }
    }
    r.test_acc = static_cast<double>(correct) / (n_b_te * B);
    return r;
}

// ---------------------------------------------------------------------------
// forward-only host evaluation (double precision, stabilised softmax): mean CE
// loss and accuracy of the given parameters over a split.  Used to score the
// device-trained weights after they are copied back.
// ---------------------------------------------------------------------------
struct EvalResult {
    double loss = 0.0;
    double acc = 0.0;
};

inline EvalResult evaluate(const MlpHyper &hp, const MlpData &d,
                           const luisa::vector<luisa::vector<double>> &W,
                           const luisa::vector<luisa::vector<double>> &Bias,
                           bool use_test) {
    const int L = static_cast<int>(hp.widths.size()) + 1;
    const int B = hp.batch;
    const int C = hp.num_outputs;
    luisa::vector<int> K(L), O(L);
    {
        int prev = hp.num_inputs;
        for (int l = 0; l < L; ++l) {
            K[l] = prev;
            O[l] = (l == L - 1) ? C : hp.widths[l];
            prev = O[l];
        }
    }
    const int n = use_test ? hp.n_test : hp.n_train;
    const auto &x = use_test ? d.x_test : d.x_train;
    const auto &y = use_test ? d.y_test : d.y_train;
    const int n_b = n / B;
    double loss_sum = 0.0;
    int correct = 0;
    for (int b = 0; b < n_b; ++b) {
        const int base = b * B;
        luisa::vector<double> act(static_cast<size_t>(B) * K[0]);
        for (int i = 0; i < B; ++i)
            for (int k = 0; k < K[0]; ++k)
                act[static_cast<size_t>(i) * K[0] + k] = x[static_cast<size_t>(base + i) * K[0] + k];
        double batch_loss = 0.0;
        int prev = K[0];
        for (int l = 0; l < L; ++l) {
            luisa::vector<double> z(static_cast<size_t>(B) * O[l]);
            for (int i = 0; i < B; ++i)
                for (int o = 0; o < O[l]; ++o) {
                    double s = Bias[l][o];
                    for (int k = 0; k < prev; ++k) s += act[static_cast<size_t>(i) * prev + k] * W[l][static_cast<size_t>(k) * O[l] + o];
                    z[static_cast<size_t>(i) * O[l] + o] = s;
                }
            if (l < L - 1) {
                luisa::vector<double> na(static_cast<size_t>(B) * O[l]);
                for (int i = 0; i < B; ++i)
                    for (int o = 0; o < O[l]; ++o) na[static_cast<size_t>(i) * O[l] + o] = relu_host(z[static_cast<size_t>(i) * O[l] + o]);
                act = std::move(na);
                prev = O[l];
            } else {
                for (int i = 0; i < B; ++i) {
                    double mx = -1e30;
                    for (int c = 0; c < C; ++c) mx = std::max(mx, z[static_cast<size_t>(i) * C + c]);
                    double sum = 0.0;
                    for (int c = 0; c < C; ++c) sum += std::exp(z[static_cast<size_t>(i) * C + c] - mx);
                    int best = 0;
                    for (int c = 0; c < C; ++c) {
                        double p = std::exp(z[static_cast<size_t>(i) * C + c] - mx) / sum;
                        if (c == y[base + i]) { batch_loss += -std::log(std::max(p, 1e-12)); }
                        if (z[static_cast<size_t>(i) * C + c] > z[static_cast<size_t>(i) * C + best]) best = c;
                    }
                    if (best == y[base + i]) correct++;
                }
            }
        }
        loss_sum += batch_loss / B;
    }
    EvalResult r;
    r.loss = loss_sum / n_b;
    r.acc = static_cast<double>(correct) / (n_b * B);
    return r;
}

// ---------------------------------------------------------------------------
// device training: captures one mlp:: kernel instance per (layer, role),
// compiles everything through tensor_example::compile_tile (never aborts; a
// failed kernel is recorded and reported through compiled_ok), then runs the
// exact minibatch-SGD algorithm of the host reference as kernel dispatches on
// real device buffers and copies the trained parameters back.
//
// Backward dispatch order per minibatch (matches the host reference):
//   for l = L-1 .. 0:  dZ_l (relu' for hidden layers), dW_l = A_{l-1}^T @ dZ_l,
//                      db_l = OnesT @ dZ_l, dA_{l-1} = dZ_l @ W_l^T
//   then W_l -= lr*dW_l, Bias_l -= lr*db_l for all layers (updates come last
//   because dA_{l-1} needs the pre-update W_l).
// ---------------------------------------------------------------------------
struct DeviceTrainResult {
    bool compiled_ok{false};
    luisa::vector<luisa::vector<double>> W;     // trained weights (host, double)
    luisa::vector<luisa::vector<double>> Bias;  // trained biases (host, double)
    double capture_ms = 0.0;
    double compile_ms = 0.0;
    int kernel_count = 0;
};

inline DeviceTrainResult train_on_device(lc::Device &device, lc::Stream &stream,
                                         const MlpHyper &hp, const MlpData &d,
                                         luisa::string_view tag) {
    namespace tile = luisa::compute::tile;
    const auto B = static_cast<int64_t>(hp.batch);
    const auto C = static_cast<int64_t>(hp.num_outputs);
    const int L = static_cast<int>(hp.widths.size()) + 1;
    luisa::vector<int64_t> K(L), O(L);
    {
        int64_t prev = hp.num_inputs;
        for (int l = 0; l < L; ++l) {
            K[l] = prev;
            O[l] = (l == L - 1) ? C : static_cast<int64_t>(hp.widths[l]);
            prev = O[l];
        }
    }

    DeviceTrainResult result;

    // ---- capture every kernel instance ---------------------------------------
    luisa::Clock c_capture;
    tile::Kernel k_sm = mlp::make_softmax(B, C);
    tile::Kernel k_ce = mlp::make_ce_grad(B, C);
    luisa::vector<tile::Kernel> k_fc, k_grad, k_gradb, k_upd, k_updb;
    luisa::vector<tile::Kernel> k_bwd, k_tw, k_ta, k_relu;
    k_fc.reserve(L); k_grad.reserve(L); k_gradb.reserve(L);
    k_upd.reserve(L); k_updb.reserve(L);
    k_bwd.reserve(L); k_tw.reserve(L); k_ta.reserve(L); k_relu.reserve(L);
    for (int l = 0; l < L; ++l) {
        k_fc.push_back(l < L - 1 ? mlp::make_fc_relu(B, K[l], O[l]) : mlp::make_fc(B, K[l], O[l]));
        k_grad.push_back(mlp::make_grad(B, K[l], O[l]));
        k_gradb.push_back(mlp::make_grad_bias(B, O[l]));
        k_upd.push_back(mlp::make_update(K[l], O[l]));
        k_updb.push_back(mlp::make_update_bias(O[l]));
        if (l > 0) {
            k_bwd.push_back(mlp::make_fc_backward(B, K[l], O[l]));// index l-1
            k_tw.push_back(mlp::make_transpose(K[l], O[l]));      // W_l^T, index l-1
            k_ta.push_back(mlp::make_transpose(B, O[l - 1]));     // A_{l-1}^T, index l-1
        }
        if (l < L - 1) { k_relu.push_back(mlp::make_relu_backward(B, O[l])); }// index l
    }
    result.capture_ms = c_capture.toc();

    // ---- compile everything (compile_tile never aborts) ----------------------
    luisa::Clock c_compile;
    bool failed = false;
    auto compile_one = [&](tile::Kernel &kernel, const luisa::string &name) noexcept {
        auto shader = tensor_example::compile_tile(device, kernel, name);
        result.kernel_count++;
        if (!shader) {
            tensor_example::record(name, false, shader.metadata().error);
            failed = true;
        }
        return shader;
    };
    luisa::vector<tile::Shader> s_fc, s_grad, s_gradb, s_upd, s_updb;
    luisa::vector<tile::Shader> s_bwd, s_tw, s_ta, s_relu;
    s_fc.reserve(L); s_grad.reserve(L); s_gradb.reserve(L);
    s_upd.reserve(L); s_updb.reserve(L);
    s_bwd.reserve(L); s_tw.reserve(L); s_ta.reserve(L); s_relu.reserve(L);
    luisa::vector<tile::Shader> s_sm, s_ce;
    s_sm.push_back(compile_one(k_sm, luisa::format("{}_softmax", tag)));
    s_ce.push_back(compile_one(k_ce, luisa::format("{}_ce_grad", tag)));
    for (int l = 0; l < L; ++l) {
        s_fc.push_back(compile_one(k_fc[l], luisa::format("{}_fc{}", tag, l + 1)));
        s_grad.push_back(compile_one(k_grad[l], luisa::format("{}_grad{}", tag, l + 1)));
        s_gradb.push_back(compile_one(k_gradb[l], luisa::format("{}_grad_bias{}", tag, l + 1)));
        s_upd.push_back(compile_one(k_upd[l], luisa::format("{}_update{}", tag, l + 1)));
        s_updb.push_back(compile_one(k_updb[l], luisa::format("{}_update_bias{}", tag, l + 1)));
        if (l > 0) {
            s_bwd.push_back(compile_one(k_bwd[l - 1], luisa::format("{}_fc_backward{}", tag, l + 1)));
            s_tw.push_back(compile_one(k_tw[l - 1], luisa::format("{}_transpose_w{}", tag, l + 1)));
            s_ta.push_back(compile_one(k_ta[l - 1], luisa::format("{}_transpose_a{}", tag, l + 1)));
        }
        if (l < L - 1) {
            s_relu.push_back(compile_one(k_relu[l], luisa::format("{}_relu_backward{}", tag, l + 1)));
        }
    }
    result.compile_ms = c_compile.toc();
    if (failed) { return result; }// driver records the abort and returns non-zero
    result.compiled_ok = true;

    // ---- device buffers -------------------------------------------------------
    luisa::vector<lc::Buffer<float>> buf;
    auto add_buf = [&](uint64_t n) noexcept {
        buf.push_back(device.create_buffer<float>(n));
        return static_cast<int>(buf.size()) - 1;
    };
    const auto i_ones = add_buf(static_cast<uint64_t>(B));   // [B,1] all ones
    const auto i_ones_t = add_buf(static_cast<uint64_t>(B)); // [1,B] all ones
    const auto i_x = add_buf(static_cast<uint64_t>(B) * static_cast<uint64_t>(K[0]));
    const auto i_x_t = add_buf(static_cast<uint64_t>(K[0]) * static_cast<uint64_t>(B));
    const auto i_y = add_buf(static_cast<uint64_t>(B) * static_cast<uint64_t>(C));
    const auto i_p = add_buf(static_cast<uint64_t>(B) * static_cast<uint64_t>(C));
    const auto i_g = add_buf(static_cast<uint64_t>(B) * static_cast<uint64_t>(C));
    int64_t max_hidden_or_input = K[0];
    int64_t max_width = C;
    for (int l = 0; l < L; ++l) {
        if (l < L - 1) { max_hidden_or_input = std::max(max_hidden_or_input, O[l]); }
        max_width = std::max(max_width, O[l]);
    }
    const auto i_at = add_buf(static_cast<uint64_t>(max_hidden_or_input) * static_cast<uint64_t>(B));
    const auto i_wt = add_buf(static_cast<uint64_t>(max_width) * static_cast<uint64_t>(max_hidden_or_input));
    const auto i_dz = add_buf(static_cast<uint64_t>(B) * static_cast<uint64_t>(max_width));
    const auto i_da = add_buf(static_cast<uint64_t>(B) * static_cast<uint64_t>(max_hidden_or_input));
    luisa::vector<int> i_w(L), i_bias(L), i_z(L), i_a(L), i_dw(L), i_db(L);
    for (int l = 0; l < L; ++l) {
        i_w[l] = add_buf(static_cast<uint64_t>(K[l]) * static_cast<uint64_t>(O[l]));
        i_bias[l] = add_buf(static_cast<uint64_t>(O[l]));
        i_z[l] = add_buf(static_cast<uint64_t>(B) * static_cast<uint64_t>(O[l]));
        i_dw[l] = add_buf(static_cast<uint64_t>(K[l]) * static_cast<uint64_t>(O[l]));
        i_db[l] = add_buf(static_cast<uint64_t>(O[l]));
        if (l < L - 1) { i_a[l] = add_buf(static_cast<uint64_t>(B) * static_cast<uint64_t>(O[l])); }
    }

    // ---- initialise constants and parameters on the device --------------------
    luisa::vector<float> h_ones(static_cast<size_t>(B), 1.0f);
    auto initial = init_parameters(hp);
    stream << buf[i_ones].copy_from(luisa::span{h_ones})
           << buf[i_ones_t].copy_from(luisa::span{h_ones});
    for (int l = 0; l < L; ++l) {
        luisa::vector<float> hw(initial.W[l].size());
        luisa::vector<float> hb(initial.Bias[l].size());
        std::copy(initial.W[l].begin(), initial.W[l].end(), hw.begin());
        std::copy(initial.Bias[l].begin(), initial.Bias[l].end(), hb.begin());
        stream << buf[i_w[l]].copy_from(luisa::span{hw})
               << buf[i_bias[l]].copy_from(luisa::span{hb});
    }
    stream << lc::synchronize();

    // ---- training loop (kernel dispatches, mirroring the host reference) ------
    const int n_b = hp.n_train / hp.batch;
    const int k0 = hp.num_inputs;
    const int c_out = hp.num_outputs;
    luisa::vector<float> hX(static_cast<size_t>(B) * k0);
    luisa::vector<float> hXT(static_cast<size_t>(k0) * B);
    luisa::vector<float> hY(static_cast<size_t>(B) * c_out);
    for (int ep = 0; ep < hp.epochs; ++ep) {
        for (int b = 0; b < n_b; ++b) {
            const int base = b * hp.batch;
            for (int i = 0; i < hp.batch; ++i) {
                for (int k = 0; k < k0; ++k) {
                    hX[static_cast<size_t>(i) * k0 + k] =
                        d.x_train[static_cast<size_t>(base + i) * k0 + k];
                    hXT[static_cast<size_t>(k) * hp.batch + i] =
                        d.x_train[static_cast<size_t>(base + i) * k0 + k];
                }
            }
            for (int i = 0; i < hp.batch; ++i) {
                for (int c = 0; c < c_out; ++c) {
                    hY[static_cast<size_t>(i) * c_out + c] =
                        d.y_onehot[static_cast<size_t>(base + i) * c_out + c];
                }
            }
            // host-staging upload; synchronize so the staging vectors can be
            // refilled for the next minibatch
            stream << buf[i_x].copy_from(luisa::span{hX})
                   << buf[i_x_t].copy_from(luisa::span{hXT})
                   << buf[i_y].copy_from(luisa::span{hY})
                   << lc::synchronize();
            // forward: Z_l = A_{l-1} @ W_l + Ones @ Bias_l (+ relu into A_l)
            for (int l = 0; l < L; ++l) {
                const auto &input = l == 0 ? buf[i_x] : buf[i_a[l - 1]];
                if (l < L - 1) {
                    stream << s_fc[l](input, buf[i_w[l]], buf[i_bias[l]], buf[i_ones],
                                        buf[i_z[l]], buf[i_a[l]])
                                  .dispatch();
                } else {
                    stream << s_fc[l](input, buf[i_w[l]], buf[i_bias[l]], buf[i_ones],
                                        buf[i_z[l]])
                                  .dispatch();
                }
            }
            stream << s_sm[0](buf[i_z[L - 1]], buf[i_p]).dispatch();
            stream << s_ce[0](buf[i_p], buf[i_y], buf[i_g]).dispatch();
            // backward: gradients for every layer first (dA_{l-1} needs the
            // pre-update W_l), SGD updates afterwards
            int cur = i_g;
            for (int l = L - 1; l >= 0; --l) {
                if (l < L - 1) {
                    // dZ_l = dA .* relu'(Z_l)
                    stream << s_relu[l](buf[i_z[l]], buf[cur], buf[i_dz]).dispatch();
                    cur = i_dz;
                }
                // dW_l = A_{l-1}^T @ dZ_l ; db_l = OnesT @ dZ_l
                if (l == 0) {
                    stream << s_grad[l](buf[i_x_t], buf[cur], buf[i_dw[l]]).dispatch();
                } else {
                    stream << s_ta[l - 1](buf[i_a[l - 1]], buf[i_at]).dispatch();
                    stream << s_grad[l](buf[i_at], buf[cur], buf[i_dw[l]]).dispatch();
                }
                stream << s_gradb[l](buf[i_ones_t], buf[cur], buf[i_db[l]]).dispatch();
                if (l > 0) {
                    // dA_{l-1} = dZ_l @ W_l^T (W_l not yet updated)
                    stream << s_tw[l - 1](buf[i_w[l]], buf[i_wt]).dispatch();
                    stream << s_bwd[l - 1](buf[cur], buf[i_wt], buf[i_da]).dispatch();
                    cur = i_da;
                }
            }
            for (int l = 0; l < L; ++l) {
                stream << s_upd[l](buf[i_dw[l]], buf[i_w[l]]).dispatch();
                stream << s_updb[l](buf[i_db[l]], buf[i_bias[l]]).dispatch();
            }
        }
    }

    // ---- copy the trained parameters back -------------------------------------
    stream << lc::synchronize();
    result.W.resize(L);
    result.Bias.resize(L);
    for (int l = 0; l < L; ++l) {
        luisa::vector<float> hw(static_cast<size_t>(K[l]) * static_cast<size_t>(O[l]));
        luisa::vector<float> hb(static_cast<size_t>(O[l]));
        stream << buf[i_w[l]].copy_to(luisa::span{hw})
               << buf[i_bias[l]].copy_to(luisa::span{hb})
               << lc::synchronize();
        result.W[l].resize(hw.size());
        result.Bias[l].resize(hb.size());
        std::copy(hw.begin(), hw.end(), result.W[l].begin());
        std::copy(hb.begin(), hb.end(), result.Bias[l].begin());
    }
    return result;
}

}// namespace mlpcommon
