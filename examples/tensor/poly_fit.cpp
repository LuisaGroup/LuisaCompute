// =============================================================================
// poly_fit.cpp — Luisa tile-language training for the polynomial fit
// =============================================================================
// The C++ twin of examples/tensor/poly_fit_train.py: fit y = sin(x) on
// [-pi, pi] with a degree-3 polynomial (a single linear layer on the
// Vandermonde features [x, x^2, x^3], bias folded in as an all-ones column),
// trained by manually applying the gradients — but every step of the loop
// (forward GEMM, MSE residual, gradient GEMM, SGD update) runs as a Luisa
// tile kernel (poly_fit_kernels.cpp) on the device.
//
// Verification, mirroring the PyTorch script:
//   1. an independent host CPU reference runs the exact same gradient descent
//      (same zero init, same lr/steps); the device loss trajectory and the
//      final weights must match it,
//   2. the fitted curve is evaluated on a held-out grid through the device
//      forward kernel; max|err| must be < 0.25 (the degree-3 least-squares
//      optimum itself is ~0.20, see poly_fit_train.py).
//
// This file is part of the single `example_tensor_stub` target and is invoked
// through its main() with the `--poly-fit` flag (see main.cpp / poly_fit.h):
//   example_tensor_stub <backend> --poly-fit [--steps N]
// =============================================================================

#include "poly_fit.h"
#include "poly_fit_kernels.h"

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <cmath>
#include <vector>

namespace polyfit {

namespace {

constexpr float PI = 3.14159265358979323846f;

// Vandermonde features with the bias folded in: [x, x^2, x^3, 1] (row-major).
std::vector<float> make_features(int n) {
    std::vector<float> X(static_cast<size_t>(n) * F);
    for (int i = 0; i < n; ++i) {
        float x = -PI + 2.0f * PI * static_cast<float>(i) / static_cast<float>(n - 1);
        X[i * F + 0] = x;
        X[i * F + 1] = x * x;
        X[i * F + 2] = x * x * x;
        X[i * F + 3] = 1.0f;
    }
    return X;
}

std::vector<float> make_targets(int n) {
    std::vector<float> y(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        float x = -PI + 2.0f * PI * static_cast<float>(i) / static_cast<float>(n - 1);
        y[i] = std::sin(x);
    }
    return y;
}

// ---------------------------------------------------------------------------
// Host reference: the exact training loop (zero init, sum-reduction MSE,
// manual gradient `W -= 2*lr * XT @ err`), used to verify the device
// trajectory independently of PyTorch.
// ---------------------------------------------------------------------------
struct HostRef {
    std::vector<float> W;      // F, column vector
    std::vector<float> losses; // loss at each logged step
    std::vector<int> log_steps;
};

HostRef host_reference(const std::vector<float> &X, const std::vector<float> &XT,
                       const std::vector<float> &y, int steps, int log_every) {
    HostRef ref;
    ref.W.assign(F, 0.0f);// zero init (device starts from the same)
    std::vector<float> Y(N_TRAIN), err(N_TRAIN), G(F);
    for (int t = 1; t <= steps; ++t) {
        for (int i = 0; i < N_TRAIN; ++i) {// forward: Y = X @ W
            float s = 0.0f;
            for (int k = 0; k < F; ++k) { s += X[i * F + k] * ref.W[k]; }
            Y[i] = s;
        }
        for (int i = 0; i < N_TRAIN; ++i) { err[i] = Y[i] - y[i]; }
        for (int k = 0; k < F; ++k) {// gradient: G = XT @ err
            float s = 0.0f;
            for (int i = 0; i < N_TRAIN; ++i) { s += XT[k * N_TRAIN + i] * err[i]; }
            G[k] = s;
        }
        for (int k = 0; k < F; ++k) { ref.W[k] -= 2.0f * LR * G[k]; }
        if (t % log_every == 0 || t == steps) {
            float loss = 0.0f;
            for (int i = 0; i < N_TRAIN; ++i) { loss += err[i] * err[i]; }
            ref.losses.push_back(loss);
            ref.log_steps.push_back(t);
        }
    }
    return ref;
}

}// namespace

}// namespace polyfit

// =============================================================================
// run_poly_fit — driver (invoked by example_tensor_stub's main with --poly-fit)
// =============================================================================
int polyfit::run_poly_fit(int argc, char *argv[]) {
    using namespace luisa;
    using namespace luisa::compute;

    // Collect the backend name from the positional arguments (skipping flags
    // such as --poly-fit), so the same executable serves both the stub and
    // the polynomial-fit training modes.
    luisa::string_view backend{};
    int steps = STEPS;
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
        LUISA_INFO("Usage: {} <backend> --poly-fit [--steps N]   (backend = vk | dx)", argv[0]);
        return 1;
    }
    if (steps <= 0) { steps = STEPS; }
    constexpr int log_every = 200;

    LUISA_INFO("[poly-fit] training data: {} samples in [-pi, pi] ({} gradient steps, lr={})",
               static_cast<int>(N_TRAIN), steps, LR);

    // ---- host data -----------------------------------------------------------
    auto X = polyfit::make_features(N_TRAIN);
    auto Xt = polyfit::make_features(N_TEST);
    auto y = polyfit::make_targets(N_TRAIN);
    auto yt = polyfit::make_targets(N_TEST);
    std::vector<float> XT(static_cast<size_t>(F) * N_TRAIN);
    for (int i = 0; i < N_TRAIN; ++i) {
        for (int k = 0; k < F; ++k) { XT[k * N_TRAIN + i] = X[i * F + k]; }
    }
    auto ref = polyfit::host_reference(X, XT, y, steps, log_every);

    // ---- device --------------------------------------------------------------
    Context ctx{argv[0]};
    Device device = ctx.create_device(backend);
    Stream stream = device.create_stream();

    auto make_buf = [&](luisa::span<const float> host) {
        auto buf = device.create_buffer<float>(static_cast<uint32_t>(host.size()));
        stream << buf.copy_from(host);
        return buf;
    };

    auto bufX = make_buf(luisa::span{X});
    auto bufXT = make_buf(luisa::span{XT});
    auto bufYref = make_buf(luisa::span{y});
    auto bufXt = make_buf(luisa::span{Xt});
    std::vector<float> W0(F, 0.0f);
    auto bufWa = make_buf(luisa::span{W0});
    auto bufWb = device.create_buffer<float>(F);
    auto bufY = device.create_buffer<float>(N_TRAIN);
    auto bufErr = device.create_buffer<float>(N_TRAIN);
    auto bufG = device.create_buffer<float>(F);
    auto bufYt = device.create_buffer<float>(N_TEST);

    // ---- compile the four tile kernels ----------------------------------------
    // Phase 1 (tile language): trace each tile program with tile::jit.compile().
    Clock c_trace;
    auto fwd = luisa::compute::tile::jit(polyfit::poly_forward<polyfit::N_TRAIN>).compile();
    auto fwd_test = luisa::compute::tile::jit(polyfit::poly_forward<polyfit::N_TEST>).compile();
    auto err_k = luisa::compute::tile::jit(polyfit::poly_error).compile();
    auto grad = luisa::compute::tile::jit(polyfit::poly_grad).compile();
    auto upd = luisa::compute::tile::jit(polyfit::poly_update).compile();
    double trace_ms = c_trace.toc();

    fwd.validate(bufX, bufWa, bufY);
    fwd_test.validate(bufXt, bufWa, bufYt);
    err_k.validate(bufY, bufYref, bufErr);
    grad.validate(bufXT, bufErr, bufG);
    upd.validate(bufWa, bufG, bufWb);

    // Phase 2 (AST): lower each traced tile function to a regular Luisa kernel
    // via tile_to_kernel (inside to_kernel<1>).
    Clock c_lower;
    auto k_fwd = fwd.to_kernel<1>();
    auto k_fwd_test = fwd_test.to_kernel<1>();
    auto k_err = err_k.to_kernel<1>();
    auto k_grad = grad.to_kernel<1>();
    auto k_upd = upd.to_kernel<1>();
    double lower_ms = c_lower.toc();

    // Phase 3 (backend): compile the lowered kernels for the target device.
    Clock c_compile;
    auto sh_fwd = device.compile(k_fwd);
    auto sh_fwd_test = device.compile(k_fwd_test);
    auto sh_err = device.compile(k_err);
    auto sh_grad = device.compile(k_grad);
    auto sh_upd = device.compile(k_upd);
    double compile_ms = c_compile.toc();

    // ---- device training loop -------------------------------------------------
    // Each step: forward -> error -> gradient -> SGD update, with the weights
    // ping-ponging between bufWa and bufWb (zero-copy in-place update).
    Clock clock;
    std::vector<float> dev_err(N_TRAIN);
    std::vector<float> dev_losses;
    std::vector<int> dev_log_steps;
    auto train_step = [&](auto &buf_in, auto &buf_out) {
        stream << sh_fwd(bufX, buf_in, bufY).dispatch(64u)
               << sh_err(bufY, bufYref, bufErr).dispatch(64u)
               << sh_grad(bufXT, bufErr, bufG).dispatch(64u)
               << sh_upd(buf_in, bufG, buf_out).dispatch(32u);
    };
    bool w_in_a = true;// current weights live in bufWa
    for (int t = 1; t <= steps; ++t) {
        if (w_in_a) { train_step(bufWa, bufWb); } else { train_step(bufWb, bufWa); }
        w_in_a = !w_in_a;
        if (t % log_every == 0 || t == steps) {
            stream << bufErr.copy_to(luisa::span{dev_err}) << synchronize();
            float loss = 0.0f;
            for (auto e : dev_err) { loss += e * e; }
            dev_losses.push_back(loss);
            dev_log_steps.push_back(t);
            LUISA_INFO("[poly-fit]   step {:5d}  loss = {:.4f}", t, loss);
        }
    }
    double gpu_ms = clock.toc();

    // Read back the trained weights (bufWa after an odd number of steps).
    std::vector<float> dev_W(F);
    if (w_in_a) {
        stream << bufWa.copy_to(luisa::span{dev_W}) << synchronize();
    } else {
        stream << bufWb.copy_to(luisa::span{dev_W}) << synchronize();
    }

    // ---- inference on a held-out grid (device forward kernel) -----------------
    std::vector<float> dev_pred(N_TEST);
    if (w_in_a) {
        stream << sh_fwd_test(bufXt, bufWa, bufYt).dispatch(64u);
    } else {
        stream << sh_fwd_test(bufXt, bufWb, bufYt).dispatch(64u);
    }
    stream << bufYt.copy_to(luisa::span{dev_pred}) << synchronize();

    float max_err = 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < N_TEST; ++i) {
        float e = dev_pred[i] - yt[i];
        max_err = std::max(max_err, std::fabs(e));
        sum_sq += e * e;
    }
    float rmse = std::sqrt(sum_sq / static_cast<float>(N_TEST));

    // ---- verify against the host CPU reference --------------------------------
    bool ok = true;
    auto check = [&](luisa::string_view what, float dev, float host, float tol) {
        float diff = std::fabs(dev - host);
        LUISA_INFO("[poly-fit]   {}: device = {:.6f}, host = {:.6f}, |diff| = {:.3e}",
                   what, dev, host, diff);
        if (diff > tol) {
            LUISA_WARNING("[poly-fit]   {} mismatch: |diff| {:.3e} > {:.3e}", what, diff, tol);
            ok = false;
        }
    };
    LUISA_INFO("[poly-fit] loss trajectory vs host reference:");
    LUISA_ASSERT(dev_losses.size() == ref.losses.size(),
                 "[poly-fit] host/device log-step mismatch ({} vs {}).",
                 dev_losses.size(), ref.losses.size());
    for (size_t i = 0; i < dev_losses.size(); ++i) {
        // fp32 GEMM/reduce order differs between host and device; allow a
        // relative tolerance on the (large) sum-reduction loss.
        float tol = 1e-2f * std::max(ref.losses[i], 1.0f);
        check(luisa::format("loss @ step {}", dev_log_steps[i]),
              dev_losses[i], ref.losses[i], tol);
    }
    LUISA_INFO("[poly-fit] final weights vs host reference:");
    for (int k = 0; k < F; ++k) {
        check(luisa::format("W[{}]", k), dev_W[k], ref.W[k], 1e-3f);
    }

    LUISA_INFO("[poly-fit] inference on {} held-out points (device forward kernel):",
               static_cast<int>(N_TEST));
    LUISA_INFO("[poly-fit]   rmse     = {:.6f}", rmse);
    LUISA_INFO("[poly-fit]   max|err| = {:.6f}", max_err);
    LUISA_INFO("[poly-fit] fitted polynomial: {:.4f}*x^3 + {:.4f}*x^2 + {:.4f}*x + {:.4f}",
               dev_W[2], dev_W[1], dev_W[0], dev_W[3]);

    // a degree-3 fit of sin(x) on [-pi, pi] cannot do better than
    // max|err| ~ 0.20 (the least-squares optimum itself); 0.25 is a safe
    // bound that still proves gradient descent converged to the optimum.
    if (max_err >= 0.25f) {
        LUISA_WARNING("[poly-fit] self check FAILED: max|err| = {:.6f} >= 0.25", max_err);
        ok = false;
    } else {
        LUISA_INFO("[poly-fit] self check: max|err| = {:.6f} < 0.25 -> PASS", max_err);
    }

    LUISA_INFO("[poly-fit] tile trace: {:.3f} ms, lower: {:.3f} ms, backend compile: {:.3f} ms, "
               "device training: {:.2f} ms ({} steps x 4 kernels).",
               trace_ms, lower_ms, compile_ms, gpu_ms, steps);
    LUISA_INFO("[poly-fit] Verification: {}", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
