// =============================================================================
// linear_regression.cpp — Luisa tile-language linear & logistic regression
// =============================================================================
// The C++ twin of examples/tensor/linear_regression_train.py: trains a linear
// regression model (recover y = w·x + b) and a logistic regression classifier
// (2D Gaussian blobs) entirely with Luisa tile kernels (linear_regression_
// kernels.cpp), then verifies against an independent host CPU reference.
//
// Verification, mirroring the PyTorch script:
//   1. linear  : the learned weights must recover the true w/b (|err| < 0.1)
//                and the held-out inference RMSE must be small (< 0.2);
//   2. logistic: held-out accuracy must be >= 85%;
//   3. device losses / final weights must match the host reference (the same
//      gradient-descent algorithm run on the CPU).
//
// This file is part of the single `example_tensor_stub` target and is invoked
// through its main() with the `--linear-regression` flag (see main.cpp):
//   example_tensor_stub <backend> --linear-regression [--steps N]
// =============================================================================

#include "linear_regression.h"
#include "linear_regression_kernels.h"

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <cmath>
#include <cstdlib>
#include <luisa/core/stl/vector.h>

namespace lreg {

namespace {

// ---- host data builders -----------------------------------------------------

// Linear regression data: X[N,D] ~ N(0,1), y = X @ true_w + true_b + noise.
struct LinData {
    luisa::vector<float> Xb;      // [N, D+1] with bias column
    luisa::vector<float> XT;      // [D+1, N]
    luisa::vector<float> y;       // [N]
    luisa::vector<float> Xb_te;   // [NT, D+1] held-out
    luisa::vector<float> y_te;    // [NT]
};

LinData make_linear_data(int seed = 1) {
    constexpr int N = LIN_N, D = LIN_D, NT = LIN_NT;
    constexpr float true_w[D] = {1.5f, -2.0f, 0.5f, 3.0f};
    constexpr float true_b = 0.7f;
    // simple LCG so the C++ twin is reproducible without <random> state noise
    auto rnd = [&](unsigned &s) {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>((s >> 8) & 0xFFFFFFu) / static_cast<float>(0x1000000u) - 0.5f;
    };
    auto gauss = [&](unsigned &s) {// Box-Muller (u in [0,1))
        auto u1 = std::max(rnd(s) + 0.5f, 1e-6f);
        auto u2 = std::max(rnd(s) + 0.5f, 1e-6f);
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265358979f * u2);
    };
    LinData d;
    d.Xb.resize(static_cast<size_t>(N) * (D + 1));
    d.y.resize(N);
    d.Xb_te.resize(static_cast<size_t>(NT) * (D + 1));
    d.y_te.resize(NT);
    unsigned s = static_cast<unsigned>(seed * 7919u + 13u);
    for (int i = 0; i < N; ++i) {
        float sum = true_b;
        for (int k = 0; k < D; ++k) {
            float x = gauss(s);
            d.Xb[i * (D + 1) + k] = x;
            sum += true_w[k] * x;
        }
        d.Xb[i * (D + 1) + D] = 1.0f;
        d.y[i] = sum + 0.05f * gauss(s);
    }
    for (int i = 0; i < NT; ++i) {
        float sum = true_b;
        for (int k = 0; k < D; ++k) {
            float x = gauss(s);
            d.Xb_te[i * (D + 1) + k] = x;
            sum += true_w[k] * x;
        }
        d.Xb_te[i * (D + 1) + D] = 1.0f;
        d.y_te[i] = sum;
    }
    d.XT.resize(static_cast<size_t>(D + 1) * N);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < D + 1; ++k)
            d.XT[k * N + i] = d.Xb[i * (D + 1) + k];
    return d;
}

// Logistic regression data: two 2D Gaussian blobs at (±2, ±2).
struct LogData {
    luisa::vector<float> Xb;      // [N, D+1]
    luisa::vector<float> XT;      // [D+1, N]
    luisa::vector<float> y;       // [N]
    luisa::vector<float> Xb_te;   // [NT, D+1]
    luisa::vector<float> y_te;    // [NT]
};

LogData make_logistic_data(int seed = 2) {
    constexpr int N = LOG_N, D = LOG_D, NT = LOG_NT;
    auto rnd = [&](unsigned &s) {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>((s >> 8) & 0xFFFFFFu) / static_cast<float>(0x1000000u) - 0.5f;
    };
    auto gauss = [&](unsigned &s) {
        auto u1 = std::max(rnd(s) + 0.5f, 1e-6f);
        auto u2 = std::max(rnd(s) + 0.5f, 1e-6f);
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265358979f * u2);
    };
    LogData d;
    d.Xb.resize(static_cast<size_t>(N) * (D + 1));
    d.y.resize(N);
    d.Xb_te.resize(static_cast<size_t>(NT) * (D + 1));
    d.y_te.resize(NT);
    unsigned s = static_cast<unsigned>(seed * 104729u + 7u);
    for (int i = 0; i < N; ++i) {
        bool cls = i < N / 2;
        d.Xb[i * (D + 1) + 0] = gauss(s) + (cls ? 2.0f : -2.0f);
        d.Xb[i * (D + 1) + 1] = gauss(s) + (cls ? 2.0f : -2.0f);
        d.Xb[i * (D + 1) + 2] = 1.0f;
        d.y[i] = cls ? 1.0f : 0.0f;
    }
    for (int i = 0; i < NT; ++i) {
        bool cls = i < NT / 2;
        d.Xb_te[i * (D + 1) + 0] = gauss(s) + (cls ? 2.0f : -2.0f);
        d.Xb_te[i * (D + 1) + 1] = gauss(s) + (cls ? 2.0f : -2.0f);
        d.Xb_te[i * (D + 1) + 2] = 1.0f;
        d.y_te[i] = cls ? 1.0f : 0.0f;
    }
    d.XT.resize(static_cast<size_t>(D + 1) * N);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < D + 1; ++k)
            d.XT[k * N + i] = d.Xb[i * (D + 1) + k];
    return d;
}

// ---- host references --------------------------------------------------------
// The exact same gradient-descent loops the device kernels execute, run in
// double precision on the CPU.

struct LinRef {
    luisa::vector<double> W;      // D+1
    luisa::vector<double> losses; // logged MSE at each log step
    luisa::vector<int> log_steps;
};

LinRef linear_host_reference(const LinData &d, int steps, int log_every) {
    constexpr int N = LIN_N, K = LIN_D + 1;
    LinRef ref;
    ref.W.assign(K, 0.0);
    luisa::vector<double> pred(N), err(N), G(K);
    for (int t = 1; t <= steps; ++t) {
        for (int i = 0; i < N; ++i) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) s += d.Xb[i * K + k] * ref.W[k];
            pred[i] = s;
            err[i] = pred[i] - d.y[i];
        }
        for (int k = 0; k < K; ++k) {
            double s = 0.0;
            for (int i = 0; i < N; ++i) s += d.XT[k * N + i] * err[i];
            G[k] = s;
        }
        for (int k = 0; k < K; ++k) ref.W[k] -= (LIN_LR / N) * G[k];
        if (t % log_every == 0 || t == steps) {
            double loss = 0.0;
            for (int i = 0; i < N; ++i) loss += err[i] * err[i];
            ref.losses.push_back(loss / N);
            ref.log_steps.push_back(t);
        }
    }
    return ref;
}

struct LogRef {
    luisa::vector<double> W;
    luisa::vector<double> losses;
    luisa::vector<int> log_steps;
};

LogRef logistic_host_reference(const LogData &d, int steps, int log_every) {
    constexpr int N = LOG_N, K = LOG_D + 1;
    LogRef ref;
    ref.W.assign(K, 0.0);
    luisa::vector<double> z(N), p(N), res(N), G(K);
    for (int t = 1; t <= steps; ++t) {
        for (int i = 0; i < N; ++i) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) s += d.Xb[i * K + k] * ref.W[k];
            z[i] = s;
            p[i] = 1.0 / (1.0 + std::exp(-s));
            res[i] = p[i] - d.y[i];
        }
        for (int k = 0; k < K; ++k) {
            double s = 0.0;
            for (int i = 0; i < N; ++i) s += d.XT[k * N + i] * res[i];
            G[k] = s;
        }
        for (int k = 0; k < K; ++k) ref.W[k] -= (LOG_LR / N) * G[k];
        if (t % log_every == 0 || t == steps) {
            double loss = 0.0;
            for (int i = 0; i < N; ++i) {
                double pv = std::clamp(p[i], 1e-12, 1.0 - 1e-12);
                loss += -(d.y[i] * std::log(pv) + (1.0 - d.y[i]) * std::log(1.0 - pv));
            }
            ref.losses.push_back(loss / N);
            ref.log_steps.push_back(t);
        }
    }
    return ref;
}

}// namespace

}// namespace lreg

// =============================================================================
// run_linear_regression — driver (invoked by example_tensor_stub's main with
// --linear-regression)
// =============================================================================
int lreg::run_linear_regression(int argc, char *argv[]) {
    using namespace luisa;
    using namespace luisa::compute;

    luisa::string_view backend{};
    int steps = LIN_STEPS;
    for (auto i = 1; i < argc; ++i) {
        if (argv != nullptr && argv[i] != nullptr) {
            luisa::string_view arg{argv[i]};
            if (!arg.starts_with("--")) {
                if (backend.empty()) { backend = arg; }
            } else if (arg == "--steps" && i + 1 < argc) {
                steps = std::atoi(argv[++i]);
            }
        }
    }
    if (backend.empty()) {
        LUISA_INFO("Usage: {} <backend> --linear-regression [--steps N]   (backend = vk | dx)", argv[0]);
        return 1;
    }
    if (steps <= 0) { steps = LIN_STEPS; }
    constexpr int log_every = 50;

    LUISA_INFO("[linear-regression] linear: {} samples x {} features, {} gradient steps, "
               "logistic: {} samples x {} features",
               static_cast<int>(LIN_N), static_cast<int>(LIN_D), steps,
               static_cast<int>(LOG_N), static_cast<int>(LOG_D));

    auto lin_data = lreg::make_linear_data();
    auto log_data = lreg::make_logistic_data();
    auto lin_ref = lreg::linear_host_reference(lin_data, steps, log_every);
    auto log_ref = lreg::logistic_host_reference(log_data, steps, log_every);

    // ---- device ---------------------------------------------------------------
    Context ctx{argv[0]};
    Device device = ctx.create_device(backend);
    Stream stream = device.create_stream();

    auto make_buf = [&](luisa::span<const float> host) {
        auto buf = device.create_buffer<float>(static_cast<uint32_t>(host.size()));
        stream << buf.copy_from(host);
        return buf;
    };

    // ---- compile the tile kernels ---------------------------------------------
    Clock c_trace;
    auto k_fwd_lin = tile::jit(lreg::forward<LIN_N, LIN_K>).compile();
    auto k_fwd_lin_te = tile::jit(lreg::forward<LIN_NT, LIN_K>).compile();
    auto k_err = tile::jit(lreg::linear_error<LIN_N>).compile();
    auto k_grad_lin = tile::jit(lreg::grad<LIN_N, LIN_K>).compile();
    auto k_upd_lin = tile::jit(lreg::update_lin).compile();
    auto k_fwd_log = tile::jit(lreg::forward<LOG_N, LOG_K>).compile();
    auto k_fwd_log_te = tile::jit(lreg::forward<LOG_NT, LOG_K>).compile();
    auto k_res = tile::jit(lreg::logistic_residual<LOG_N>).compile();
    auto k_grad_log = tile::jit(lreg::grad<LOG_N, LOG_K>).compile();
    auto k_upd_log = tile::jit(lreg::update_log).compile();
    double trace_ms = c_trace.toc();

    Clock c_lower;
    auto sh_fwd_lin = k_fwd_lin.to_kernel<1>();
    auto sh_fwd_lin_te = k_fwd_lin_te.to_kernel<1>();
    auto sh_err = k_err.to_kernel<1>();
    auto sh_grad_lin = k_grad_lin.to_kernel<1>();
    auto sh_upd_lin = k_upd_lin.to_kernel<1>();
    auto sh_fwd_log = k_fwd_log.to_kernel<1>();
    auto sh_fwd_log_te = k_fwd_log_te.to_kernel<1>();
    auto sh_res = k_res.to_kernel<1>();
    auto sh_grad_log = k_grad_log.to_kernel<1>();
    auto sh_upd_log = k_upd_log.to_kernel<1>();
    double lower_ms = c_lower.toc();

    Clock c_compile;
    auto shc_fwd_lin = device.compile(sh_fwd_lin);
    auto shc_fwd_lin_te = device.compile(sh_fwd_lin_te);
    auto shc_err = device.compile(sh_err);
    auto shc_grad_lin = device.compile(sh_grad_lin);
    auto shc_upd_lin = device.compile(sh_upd_lin);
    auto shc_fwd_log = device.compile(sh_fwd_log);
    auto shc_fwd_log_te = device.compile(sh_fwd_log_te);
    auto shc_res = device.compile(sh_res);
    auto shc_grad_log = device.compile(sh_grad_log);
    auto shc_upd_log = device.compile(sh_upd_log);
    double compile_ms = c_compile.toc();

    // ---- linear regression: device training loop ------------------------------
    auto bufXb = make_buf(luisa::span{lin_data.Xb});
    auto bufXT = make_buf(luisa::span{lin_data.XT});
    auto bufYref = make_buf(luisa::span{lin_data.y});
    auto bufXb_te = make_buf(luisa::span{lin_data.Xb_te});
    luisa::vector<float> W0(LIN_D + 1, 0.0f);
    auto bufW = make_buf(luisa::span{W0});
    auto bufW2 = device.create_buffer<float>(LIN_D + 1);
    auto bufY = device.create_buffer<float>(LIN_N);
    auto bufErr = device.create_buffer<float>(LIN_N);
    auto bufG = device.create_buffer<float>(LIN_D + 1);
    auto bufY_te = device.create_buffer<float>(LIN_NT);

    Clock clock;
    luisa::vector<float> dev_err(LIN_N), dev_losses;
    luisa::vector<int> dev_log_steps;
    auto train_step = [&](auto &buf_in, auto &buf_out) {
        stream << shc_fwd_lin(bufXb, buf_in, bufY).dispatch(64u)
               << shc_err(bufY, bufYref, bufErr).dispatch(64u)
               << shc_grad_lin(bufXT, bufErr, bufG).dispatch(64u)
               << shc_upd_lin(buf_in, bufG, buf_out).dispatch(32u);
    };
    bool w_in_a = true;
    for (int t = 1; t <= steps; ++t) {
        if (w_in_a) { train_step(bufW, bufW2); } else { train_step(bufW2, bufW); }
        w_in_a = !w_in_a;
        if (t % log_every == 0 || t == steps) {
            stream << bufErr.copy_to(luisa::span{dev_err}) << synchronize();
            float loss = 0.0f;
            for (auto e : dev_err) { loss += e * e; }
            dev_losses.push_back(loss / LIN_N);
            dev_log_steps.push_back(t);
            LUISA_INFO("[linear-regression]   linear step {:4d}  MSE = {:.6f}", t, loss / LIN_N);
        }
    }
    luisa::vector<float> dev_W(LIN_D + 1);
    if (w_in_a) {
        stream << bufW.copy_to(luisa::span{dev_W}) << synchronize();
    } else {
        stream << bufW2.copy_to(luisa::span{dev_W}) << synchronize();
    }

    // inference on held-out data
    luisa::vector<float> dev_pred(LIN_NT);
    if (w_in_a) {
        stream << shc_fwd_lin_te(bufXb_te, bufW, bufY_te).dispatch(64u);
    } else {
        stream << shc_fwd_lin_te(bufXb_te, bufW2, bufY_te).dispatch(64u);
    }
    stream << bufY_te.copy_to(luisa::span{dev_pred}) << synchronize();

    float lin_rmse = 0.0f, lin_max_w_err = 0.0f;
    for (int i = 0; i < LIN_NT; ++i) {
        float e = dev_pred[i] - lin_data.y_te[i];
        lin_rmse += e * e;
    }
    lin_rmse = std::sqrt(lin_rmse / LIN_NT);
    for (int k = 0; k < LIN_D; ++k) {
        float e = std::fabs(dev_W[k] - luisa::vector<float>{1.5f, -2.0f, 0.5f, 3.0f}[k]);
        lin_max_w_err = std::max(lin_max_w_err, e);
    }
    double gpu_ms = clock.toc();

    // ---- logistic regression: device training loop ----------------------------
    auto bufXb2 = make_buf(luisa::span{log_data.Xb});
    auto bufXT2 = make_buf(luisa::span{log_data.XT});
    auto bufYref2 = make_buf(luisa::span{log_data.y});
    auto bufXb2_te = make_buf(luisa::span{log_data.Xb_te});
    luisa::vector<float> W2_0(LOG_D + 1, 0.0f);
    auto bufW2_0 = make_buf(luisa::span{W2_0});
    auto bufW2_1 = device.create_buffer<float>(LOG_D + 1);
    auto bufZ = device.create_buffer<float>(LOG_N);
    auto bufRes = device.create_buffer<float>(LOG_N);
    auto bufG2 = device.create_buffer<float>(LOG_D + 1);
    auto bufZ_te = device.create_buffer<float>(LOG_NT);

    Clock clock2;
    luisa::vector<float> dev_losses2;
    luisa::vector<int> dev_log_steps2;
    auto train_step2 = [&](auto &buf_in, auto &buf_out) {
        stream << shc_fwd_log(bufXb2, buf_in, bufZ).dispatch(64u)
               << shc_res(bufZ, bufYref2, bufRes).dispatch(64u)
               << shc_grad_log(bufXT2, bufRes, bufG2).dispatch(64u)
               << shc_upd_log(buf_in, bufG2, buf_out).dispatch(32u);
    };
    bool w2_in_a = true;
    for (int t = 1; t <= steps; ++t) {
        if (w2_in_a) { train_step2(bufW2_0, bufW2_1); } else { train_step2(bufW2_1, bufW2_0); }
        w2_in_a = !w2_in_a;
        if (t % log_every == 0 || t == steps) {
            stream << bufRes.copy_to(luisa::span{dev_err}) << synchronize();
            // read back the logits to compute the BCE loss on the host
            stream << bufZ.copy_to(luisa::span{dev_pred}) << synchronize();
            float loss = 0.0f;
            for (int i = 0; i < LOG_N; ++i) {
                float p = 1.0f / (1.0f + std::exp(-dev_pred[i]));
                p = std::clamp(p, 1e-6f, 1.0f - 1e-6f);
                loss += -(log_data.y[i] * std::log(p) + (1.0f - log_data.y[i]) * std::log(1.0f - p));
            }
            dev_losses2.push_back(loss / LOG_N);
            dev_log_steps2.push_back(t);
            LUISA_INFO("[linear-regression]   logistic step {:4d}  BCE = {:.6f}", t, loss / LOG_N);
        }
    }
    luisa::vector<float> dev_W2(LOG_D + 1);
    if (w2_in_a) {
        stream << bufW2_0.copy_to(luisa::span{dev_W2}) << synchronize();
    } else {
        stream << bufW2_1.copy_to(luisa::span{dev_W2}) << synchronize();
    }

    // inference on held-out data
    luisa::vector<float> dev_z_te(LOG_NT);
    if (w2_in_a) {
        stream << shc_fwd_log_te(bufXb2_te, bufW2_0, bufZ_te).dispatch(64u);
    } else {
        stream << shc_fwd_log_te(bufXb2_te, bufW2_1, bufZ_te).dispatch(64u);
    }
    stream << bufZ_te.copy_to(luisa::span{dev_z_te}) << synchronize();

    int correct = 0;
    for (int i = 0; i < LOG_NT; ++i) {
        float p = 1.0f / (1.0f + std::exp(-dev_z_te[i]));
        bool pred = p > 0.5f;
        bool truth = log_data.y_te[i] > 0.5f;
        if (pred == truth) { correct++; }
    }
    float log_acc = static_cast<float>(correct) / LOG_NT;
    double gpu2_ms = clock2.toc();

    // ---- verify against the host reference -------------------------------------
    bool ok = true;
    auto check = [&](luisa::string_view what, float dev, double host, double tol) {
        double diff = std::fabs(static_cast<double>(dev) - host);
        LUISA_INFO("[linear-regression]   {}: device = {:.6f}, host = {:.6f}, |diff| = {:.3e}",
                   what, dev, host, diff);
        if (diff > tol) {
            LUISA_WARNING("[linear-regression]   {} mismatch: |diff| {:.3e} > {:.3e}", what, diff, tol);
            ok = false;
        }
    };

    LUISA_INFO("[linear-regression] linear loss trajectory vs host reference:");
    LUISA_ASSERT(dev_losses.size() == lin_ref.losses.size(),
                 "[linear-regression] host/device linear log-step mismatch ({} vs {}).",
                 dev_losses.size(), lin_ref.losses.size());
    for (size_t i = 0; i < dev_losses.size(); ++i) {
        double tol = 2e-2 * std::max(lin_ref.losses[i], 1e-3);
        check(luisa::format("MSE @ step {}", dev_log_steps[i]), dev_losses[i], lin_ref.losses[i], tol);
    }
    LUISA_INFO("[linear-regression] linear final weights vs host reference:");
    for (int k = 0; k < LIN_D + 1; ++k) {
        check(luisa::format("W[{}]", k), dev_W[k], lin_ref.W[k], 2e-2);
    }

    LUISA_INFO("[linear-regression] logistic loss trajectory vs host reference:");
    LUISA_ASSERT(dev_losses2.size() == log_ref.losses.size(),
                 "[linear-regression] host/device logistic log-step mismatch ({} vs {}).",
                 dev_losses2.size(), log_ref.losses.size());
    for (size_t i = 0; i < dev_losses2.size(); ++i) {
        double tol = 2e-2 * std::max(log_ref.losses[i], 1e-2);
        check(luisa::format("BCE @ step {}", dev_log_steps2[i]), dev_losses2[i], log_ref.losses[i], tol);
    }

    // ---- self checks (mirror linear_regression_train.py) -----------------------
    LUISA_INFO("[linear-regression] linear: learned w = [{:.4f}, {:.4f}, {:.4f}, {:.4f}], b = {:.4f}",
               dev_W[0], dev_W[1], dev_W[2], dev_W[3], dev_W[4]);
    LUISA_INFO("[linear-regression] linear: max|w err| = {:.4f}, inference RMSE = {:.5f}",
               lin_max_w_err, lin_rmse);
    if (lin_max_w_err >= 0.1f || lin_rmse >= 0.2f) {
        LUISA_WARNING("[linear-regression] linear self check FAILED (w_err={:.4f}, rmse={:.5f})",
                      lin_max_w_err, lin_rmse);
        ok = false;
    } else {
        LUISA_INFO("[linear-regression] linear self check: weights recovered, inference matches -> PASS");
    }

    LUISA_INFO("[linear-regression] logistic: inference accuracy = {:.1f}%", 100.0f * log_acc);
    if (log_acc < 0.85f) {
        LUISA_WARNING("[linear-regression] logistic self check FAILED (acc={:.2f})", log_acc);
        ok = false;
    } else {
        LUISA_INFO("[linear-regression] logistic self check: held-out accuracy sufficient -> PASS");
    }

    LUISA_INFO("[linear-regression] tile trace: {:.3f} ms, lower: {:.3f} ms, backend compile: {:.3f} ms, "
               "device training: linear {:.2f} ms + logistic {:.2f} ms ({} steps x 4 kernels each).",
               trace_ms, lower_ms, compile_ms, gpu_ms, gpu2_ms, steps);
    LUISA_INFO("[linear-regression] Verification: {}", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
