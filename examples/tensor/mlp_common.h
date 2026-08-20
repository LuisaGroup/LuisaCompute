// =============================================================================
// mlp_common.h — shared data builders and host reference for MLP / MNIST
// =============================================================================
// Used by both the `--mlp` (3-layer) and `--mnist` (2-layer) drivers.  All
// training data is generated on the host with a small deterministic PRNG, so
// the C++ twin runs offline with no downloads (mirroring the synthetic modes
// of mlp_train.py / mnist_train.py).  The host reference below runs the exact
// same minibatch-SGD / cross-entropy algorithm as the device tile kernels (in
// double precision) so the drivers can verify the device against it.
//
// Weights and biases are kept separate (W[K,O], Bias[1,O]); the device applies
// the bias with a second GEMM against an all-ones buffer (see mlp_kernels.h).
// =============================================================================

#pragma once

#include <luisa/core/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace mlpcommon {

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
    std::vector<int> widths;    // hidden layer widths (excluding input/output)
    int seed = 123;
};

// ---------------------------------------------------------------------------
// dataset (all matrices row-major, raw features only — no bias column)
// ---------------------------------------------------------------------------
struct MlpData {
    std::vector<float> x_train;    // [n_train, num_inputs]
    std::vector<float> xT_train;   // [num_inputs, n_train]
    std::vector<int> y_train;      // [n_train]
    std::vector<float> y_onehot;   // [n_train, num_outputs]
    std::vector<float> x_test;     // [n_test, num_inputs]
    std::vector<int> y_test;       // [n_test]
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
    std::vector<float> x(n * hp.num_inputs);
    std::vector<int> y(n);
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
    std::vector<float> templates(static_cast<size_t>(hp.num_outputs) * hp.num_inputs);
    for (auto &v : templates) { v = rng.gauss(); }
    std::vector<int> y(n);
    std::vector<float> x(static_cast<size_t>(n) * hp.num_inputs);
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
// generic host reference: exact same minibatch-SGD / cross-entropy algorithm
// as the device, in double precision, over L layers (widths = hidden sizes).
// Returns final weights [L][K,O], biases [L][O], epoch-average CE losses and
// the test accuracy.
// ---------------------------------------------------------------------------
struct HostMlpResult {
    std::vector<std::vector<double>> W;      // [layer][K*O]
    std::vector<std::vector<double>> Bias;   // [layer][O]
    std::vector<double> epoch_losses;
    double test_acc = 0.0;
};

inline double relu_host(double x) noexcept { return x > 0.0 ? x : 0.0; }

inline HostMlpResult mlp_host_reference(const MlpHyper &hp, const MlpData &d) {
    const int L = static_cast<int>(hp.widths.size()) + 1;// hidden + output layers
    HostMlpResult r;
    r.W.resize(L);
    r.Bias.resize(L);
    // K_l = input dim; O_l = output dim
    std::vector<int> K(L), O(L);
    {
        int prev = hp.num_inputs;
        for (int l = 0; l < L; ++l) {
            K[l] = prev;
            O[l] = (l == L - 1) ? hp.num_outputs : hp.widths[l];
            prev = O[l];
        }
    }
    // same init as the device: PRNG(0) * 0.05
    PRNG w_rng(0u);
    for (int l = 0; l < L; ++l) {
        r.W[l].assign(static_cast<size_t>(K[l]) * O[l], 0.0);
        r.Bias[l].assign(static_cast<size_t>(O[l]), 0.0);
        for (auto &v : r.W[l]) { v = 0.05 * w_rng.gauss(); }
        for (auto &v : r.Bias[l]) { v = 0.05 * w_rng.gauss(); }
    }
    const int B = hp.batch;
    const int n_b = hp.n_train / B;
    for (int ep = 0; ep < hp.epochs; ++ep) {
        double ep_loss = 0.0;
        for (int b = 0; b < n_b; ++b) {
            const int base = b * B;
            // forward
            std::vector<std::vector<double>> A(L);
            std::vector<std::vector<double>> Z(L);
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
            std::vector<double> P(static_cast<size_t>(B) * C);
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
            std::vector<double> G(static_cast<size_t>(B) * C);
            for (int i = 0; i < B; ++i) {
                for (int c = 0; c < C; ++c) {
                    G[static_cast<size_t>(i) * C + c] =
                        (P[static_cast<size_t>(i) * C + c] -
                         d.y_onehot[static_cast<size_t>(base + i) * C + c]) / B;
                }
            }
            // backward
            std::vector<std::vector<double>> dW(L);
            std::vector<std::vector<double>> db(L);
            std::vector<double> dZ;
            for (int l = L - 1; l >= 0; --l) {
                dW[l].assign(static_cast<size_t>(K[l]) * O[l], 0.0);
                db[l].assign(static_cast<size_t>(O[l]), 0.0);
                // copy by value: dZ is reassigned below for the next layer, and
                // a reference would be invalidated by that reassignment
                const std::vector<double> cur_dZ = (l == L - 1) ? G : dZ;
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
        std::vector<double> act(static_cast<size_t>(B) * K[0]);
        for (int i = 0; i < B; ++i)
            for (int k = 0; k < K[0]; ++k)
                act[static_cast<size_t>(i) * K[0] + k] = d.x_test[static_cast<size_t>(base + i) * K[0] + k];
        int prev = K[0];
        for (int l = 0; l < L; ++l) {
            std::vector<double> z(static_cast<size_t>(B) * O[l]);
            for (int i = 0; i < B; ++i)
                for (int o = 0; o < O[l]; ++o) {
                    double s = r.Bias[l][o];
                    for (int k = 0; k < prev; ++k) s += act[static_cast<size_t>(i) * prev + k] * r.W[l][static_cast<size_t>(k) * O[l] + o];
                    z[static_cast<size_t>(i) * O[l] + o] = s;
                }
            if (l < L - 1) {
                std::vector<double> na(static_cast<size_t>(B) * O[l]);
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

}// namespace mlpcommon
