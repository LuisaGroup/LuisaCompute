// =============================================================================
// linear_regression.cpp — linear & logistic regression on the new Tile DSL
// =============================================================================
// The C++ twin of examples/tensor/linear_regression_train.py: trains a linear
// regression model (recover y = w·x + b) and a logistic regression classifier
// (2D Gaussian blobs) entirely with Tile DSL kernels
// (linear_regression_kernels.{h,cpp}), then verifies against an independent
// host CPU reference.
//
// Port of backup_old_tile/examples/tensor/linear_regression.cpp.
// Verification, mirroring the PyTorch script:
//   1. linear  : the learned weights must recover the true w/b (|err| < 0.1)
//                and the held-out inference RMSE must be small (< 0.2);
//   2. logistic: held-out accuracy must be >= 85%;
//   3. device losses / final weights must match the host reference (the same
//      gradient-descent algorithm run on the CPU).
//
// Invoked through example_tensor's main():
//   example_tensor <backend> --linear-regression [--steps N]
// =============================================================================

#include "linear_regression.h"
#include "linear_regression_kernels.h"
#include "tensor_kernels.h"

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace tensor_example::lreg {

namespace {

// ---- host data builders -----------------------------------------------------

// Linear regression data: X[N,D] ~ N(0,1), y = X @ true_w + true_b + noise.
struct LinData {
    luisa::vector<float> Xb;    // [N, D+1] with bias column
    luisa::vector<float> XT;    // [D+1, N]
    luisa::vector<float> y;     // [N]
    luisa::vector<float> Xb_te; // [NT, D+1] held-out
    luisa::vector<float> y_te;  // [NT]
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
    luisa::vector<float> Xb;    // [N, D+1]
    luisa::vector<float> XT;    // [D+1, N]
    luisa::vector<float> y;     // [N]
    luisa::vector<float> Xb_te; // [NT, D+1]
    luisa::vector<float> y_te;  // [NT]
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
}// namespace tensor_example::lreg

// =============================================================================
// run_linear_regression — driver (invoked by example_tensor's main with
// --linear-regression)
// =============================================================================
int tensor_example::lreg::run_linear_regression(int argc, char *argv[]) {
    using luisa::string_view;

    string_view backend{};
    int steps = LIN_STEPS;
    for (auto i = 1; i < argc; ++i) {
        if (argv != nullptr && argv[i] != nullptr) {
            string_view arg{argv[i]};
            if (!arg.starts_with("--")) {
                if (backend.empty()) { backend = arg; }
            } else if (arg == "--steps" && i + 1 < argc) {
                steps = std::atoi(argv[++i]);
            }
        }
    }
    if (backend.empty()) {
        LUISA_INFO("Usage: {} <backend> --linear-regression [--steps N]   (backend = vk | dx | cuda | metal)",
                   argv[0]);
        return 1;
    }
    if (steps <= 0) { steps = LIN_STEPS; }
    constexpr int log_every = 50;

    LUISA_INFO("[linear-regression] linear: {} samples x {} features, {} gradient steps, "
               "logistic: {} samples x {} features",
               static_cast<int>(LIN_N), static_cast<int>(LIN_D), steps,
               static_cast<int>(LOG_N), static_cast<int>(LOG_D));

    auto lin_data = make_linear_data();
    auto log_data = make_logistic_data();
    auto lin_ref = linear_host_reference(lin_data, steps, log_every);
    auto log_ref = logistic_host_reference(log_data, steps, log_every);

    // ---- device ---------------------------------------------------------------
    lc::Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream(lc::StreamTag::COMPUTE);

    auto make_buf = [&](luisa::span<const float> host) {
        auto buf = device.create_buffer<float>(static_cast<uint32_t>(host.size()));
        stream << buf.copy_from(host);
        return buf;
    };

    // ---- compile the tile kernels ---------------------------------------------
    auto shc_fwd_lin = compile_tile(device, make_forward_kernel<LIN_N, LIN_K>(), "lreg_forward_lin");
    auto shc_fwd_lin_te = compile_tile(device, make_forward_kernel<LIN_NT, LIN_K>(), "lreg_forward_lin_te");
    auto shc_err = compile_tile(device, make_linear_error_kernel(), "lreg_linear_error");
    auto shc_grad_lin = compile_tile(device, make_grad_kernel<LIN_N, LIN_K>(), "lreg_grad_lin");
    auto shc_upd_lin = compile_tile(device, make_update_lin_kernel(), "lreg_update_lin");
    auto shc_fwd_log = compile_tile(device, make_forward_kernel<LOG_N, LOG_K>(), "lreg_forward_log");
    auto shc_fwd_log_te = compile_tile(device, make_forward_kernel<LOG_NT, LOG_K>(), "lreg_forward_log_te");
    auto shc_res = compile_tile(device, make_logistic_residual_kernel(), "lreg_logistic_residual");
    auto shc_grad_log = compile_tile(device, make_grad_kernel<LOG_N, LOG_K>(), "lreg_grad_log");
    auto shc_upd_log = compile_tile(device, make_update_log_kernel(), "lreg_update_log");
    if (!shc_fwd_lin || !shc_fwd_lin_te || !shc_err || !shc_grad_lin || !shc_upd_lin ||
        !shc_fwd_log || !shc_fwd_log_te || !shc_res || !shc_grad_log || !shc_upd_log) {
        auto error_of = [](const auto &s) { return s.metadata().error; };
        record("lreg_compile", false,
               !shc_fwd_lin ? error_of(shc_fwd_lin) :
               !shc_fwd_lin_te ? error_of(shc_fwd_lin_te) :
               !shc_err ? error_of(shc_err) :
               !shc_grad_lin ? error_of(shc_grad_lin) :
               !shc_upd_lin ? error_of(shc_upd_lin) :
               !shc_fwd_log ? error_of(shc_fwd_log) :
               !shc_fwd_log_te ? error_of(shc_fwd_log_te) :
               !shc_res ? error_of(shc_res) :
               !shc_grad_log ? error_of(shc_grad_log) : error_of(shc_upd_log));
        return 1;
    }

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

    luisa::vector<float> dev_err(LIN_N), dev_losses;
    luisa::vector<int> dev_log_steps;
    auto train_step = [&](auto &buf_in, auto &buf_out) {
        stream << shc_fwd_lin(bufXb, buf_in, bufY).dispatch()
               << shc_err(bufY, bufYref, bufErr).dispatch()
               << shc_grad_lin(bufXT, bufErr, bufG).dispatch()
               << shc_upd_lin(buf_in, bufG, buf_out).dispatch();
    };
    bool w_in_a = true;
    for (int t = 1; t <= steps; ++t) {
        if (w_in_a) { train_step(bufW, bufW2); } else { train_step(bufW2, bufW); }
        w_in_a = !w_in_a;
        if (t % log_every == 0 || t == steps) {
            stream << bufErr.copy_to(luisa::span{dev_err}) << lc::synchronize();
            float loss = 0.0f;
            for (auto e : dev_err) { loss += e * e; }
            dev_losses.push_back(loss / LIN_N);
            dev_log_steps.push_back(t);
            LUISA_INFO("[linear-regression]   linear step {:4d}  MSE = {:.6f}", t, loss / LIN_N);
        }
    }
    luisa::vector<float> dev_W(LIN_D + 1);
    if (w_in_a) {
        stream << bufW.copy_to(luisa::span{dev_W}) << lc::synchronize();
    } else {
        stream << bufW2.copy_to(luisa::span{dev_W}) << lc::synchronize();
    }

    // inference on held-out data
    luisa::vector<float> dev_pred(LIN_NT);
    if (w_in_a) {
        stream << shc_fwd_lin_te(bufXb_te, bufW, bufY_te).dispatch();
    } else {
        stream << shc_fwd_lin_te(bufXb_te, bufW2, bufY_te).dispatch();
    }
    stream << bufY_te.copy_to(luisa::span{dev_pred}) << lc::synchronize();

    auto lin_rmse = 0.0, lin_max_w_err = 0.0;
    for (int i = 0; i < LIN_NT; ++i) {
        auto e = static_cast<double>(dev_pred[i]) - lin_data.y_te[i];
        lin_rmse += e * e;
    }
    lin_rmse = std::sqrt(lin_rmse / LIN_NT);
    const float true_w[LIN_D] = {1.5f, -2.0f, 0.5f, 3.0f};
    for (int k = 0; k < LIN_D; ++k) {
        lin_max_w_err = luisa::max(lin_max_w_err, luisa::abs(static_cast<double>(dev_W[k]) - true_w[k]));
    }

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

    luisa::vector<float> dev_losses2, dev_z(LOG_N);
    luisa::vector<int> dev_log_steps2;
    auto train_step2 = [&](auto &buf_in, auto &buf_out) {
        stream << shc_fwd_log(bufXb2, buf_in, bufZ).dispatch()
               << shc_res(bufZ, bufYref2, bufRes).dispatch()
               << shc_grad_log(bufXT2, bufRes, bufG2).dispatch()
               << shc_upd_log(buf_in, bufG2, buf_out).dispatch();
    };
    bool w2_in_a = true;
    for (int t = 1; t <= steps; ++t) {
        if (w2_in_a) { train_step2(bufW2_0, bufW2_1); } else { train_step2(bufW2_1, bufW2_0); }
        w2_in_a = !w2_in_a;
        if (t % log_every == 0 || t == steps) {
            // read back the logits to compute the BCE loss on the host
            stream << bufZ.copy_to(luisa::span{dev_z}) << lc::synchronize();
            float loss = 0.0f;
            for (int i = 0; i < LOG_N; ++i) {
                float p = 1.0f / (1.0f + std::exp(-dev_z[i]));
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
        stream << bufW2_0.copy_to(luisa::span{dev_W2}) << lc::synchronize();
    } else {
        stream << bufW2_1.copy_to(luisa::span{dev_W2}) << lc::synchronize();
    }

    // inference on held-out data
    luisa::vector<float> dev_z_te(LOG_NT);
    if (w2_in_a) {
        stream << shc_fwd_log_te(bufXb2_te, bufW2_0, bufZ_te).dispatch();
    } else {
        stream << shc_fwd_log_te(bufXb2_te, bufW2_1, bufZ_te).dispatch();
    }
    stream << bufZ_te.copy_to(luisa::span{dev_z_te}) << lc::synchronize();

    int correct = 0;
    for (int i = 0; i < LOG_NT; ++i) {
        float p = 1.0f / (1.0f + std::exp(-dev_z_te[i]));
        bool pred = p > 0.5f;
        bool truth = log_data.y_te[i] > 0.5f;
        if (pred == truth) { correct++; }
    }
    auto log_acc = static_cast<double>(correct) / LOG_NT;

    // ---- verify against the host reference -------------------------------------
    bool ok = true;
    auto check_value = [&](luisa::string_view what, double dev, double host, double tol) {
        double diff = std::fabs(dev - host);
        LUISA_INFO("[linear-regression]   {}: device = {:.6f}, host = {:.6f}, |diff| = {:.3e}",
                   what, dev, host, diff);
        check(what, diff, tol);
        if (diff > tol) {
            LUISA_WARNING("[linear-regression]   {} mismatch: |diff| {:.3e} > {:.3e}", what, diff, tol);
            ok = false;
        }
    };

    LUISA_INFO("[linear-regression] linear loss trajectory vs host reference:");
    if (dev_losses.size() != lin_ref.losses.size()) {
        record("lreg_linear_log_steps", false,
               luisa::format("host/device linear log-step mismatch ({} vs {})",
                             lin_ref.losses.size(), dev_losses.size()));
        ok = false;
    }
    for (size_t i = 0; i < dev_losses.size() && i < lin_ref.losses.size(); ++i) {
        double tol = 2e-2 * std::max(lin_ref.losses[i], 1e-3);
        check_value(luisa::format("MSE @ step {}", dev_log_steps[i]), dev_losses[i], lin_ref.losses[i], tol);
    }
    LUISA_INFO("[linear-regression] linear final weights vs host reference:");
    for (int k = 0; k < LIN_D + 1; ++k) {
        check_value(luisa::format("W[{}]", k), dev_W[k], lin_ref.W[k], 2e-2);
    }

    LUISA_INFO("[linear-regression] logistic loss trajectory vs host reference:");
    if (dev_losses2.size() != log_ref.losses.size()) {
        record("lreg_logistic_log_steps", false,
               luisa::format("host/device logistic log-step mismatch ({} vs {})",
                             log_ref.losses.size(), dev_losses2.size()));
        ok = false;
    }
    for (size_t i = 0; i < dev_losses2.size() && i < log_ref.losses.size(); ++i) {
        double tol = 2e-2 * std::max(log_ref.losses[i], 1e-2);
        check_value(luisa::format("BCE @ step {}", dev_log_steps2[i]), dev_losses2[i], log_ref.losses[i], tol);
    }

    // ---- self checks (mirror linear_regression_train.py) -----------------------
    LUISA_INFO("[linear-regression] linear: learned w = [{:.4f}, {:.4f}, {:.4f}, {:.4f}], b = {:.4f}",
               dev_W[0], dev_W[1], dev_W[2], dev_W[3], dev_W[4]);
    LUISA_INFO("[linear-regression] linear: max|w err| = {:.4f}, inference RMSE = {:.5f}",
               lin_max_w_err, lin_rmse);
    check("lreg_linear_weights", lin_max_w_err, 0.1);
    check("lreg_linear_rmse", lin_rmse, 0.2);
    if (lin_max_w_err >= 0.1 || lin_rmse >= 0.2) {
        LUISA_WARNING("[linear-regression] linear self check FAILED (w_err={:.4f}, rmse={:.5f})",
                      lin_max_w_err, lin_rmse);
        ok = false;
    } else {
        LUISA_INFO("[linear-regression] linear self check: weights recovered, inference matches -> PASS");
    }

    LUISA_INFO("[linear-regression] logistic: inference accuracy = {:.1f}%", 100.0 * log_acc);
    check("lreg_logistic_accuracy", 1.0 - log_acc, 0.15);
    if (log_acc < 0.85) {
        LUISA_WARNING("[linear-regression] logistic self check FAILED (acc={:.2f})", log_acc);
        ok = false;
    } else {
        LUISA_INFO("[linear-regression] logistic self check: held-out accuracy sufficient -> PASS");
    }

    LUISA_INFO("[linear-regression] Verification: {}", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
