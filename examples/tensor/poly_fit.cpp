// =============================================================================
// poly_fit.cpp — polynomial-fit training on the new Tile DSL
// =============================================================================
// The C++ twin of examples/tensor/poly_fit_train.py: fit y = sin(x) on
// [-pi, pi] with a degree-3 polynomial (a single linear layer on the
// Vandermonde features [x, x^2, x^3], bias folded in as an all-ones column),
// trained by manually applying the gradients — but every step of the loop
// (forward GEMM, MSE residual, gradient GEMM, SGD update) runs as a Tile DSL
// kernel (poly_fit_kernels.{h,cpp}) on the device.
//
// Port of backup_old_tile/examples/tensor/poly_fit.cpp. Verification, mirroring
// the PyTorch script:
//   1. an independent host CPU reference runs the exact same gradient descent
//      (same zero init, same lr/steps); the device loss trajectory and the
//      final weights must match it,
//   2. the fitted curve is evaluated on a held-out grid through the device
//      forward kernel; max|err| must be < 0.25 (the degree-3 least-squares
//      optimum itself is ~0.20, see poly_fit_train.py).
//
// Invoked through example_tensor's main():
//   example_tensor <backend> --poly-fit [--steps N]
// =============================================================================

#include "poly_fit.h"
#include "poly_fit_kernels.h"
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

namespace tensor_example::polyfit {

namespace {

constexpr float PI = 3.14159265358979323846f;

// Vandermonde features with the bias folded in: [x, x^2, x^3, 1] (row-major).
luisa::vector<float> make_features(int n) {
    luisa::vector<float> X(static_cast<size_t>(n) * F);
    for (int i = 0; i < n; ++i) {
        float x = -PI + 2.0f * PI * static_cast<float>(i) / static_cast<float>(n - 1);
        X[i * F + 0] = x;
        X[i * F + 1] = x * x;
        X[i * F + 2] = x * x * x;
        X[i * F + 3] = 1.0f;
    }
    return X;
}

luisa::vector<float> make_targets(int n) {
    luisa::vector<float> y(static_cast<size_t>(n));
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
    luisa::vector<float> W;      // F, column vector
    luisa::vector<float> losses; // loss at each logged step
    luisa::vector<int> log_steps;
};

HostRef host_reference(const luisa::vector<float> &X, const luisa::vector<float> &XT,
                       const luisa::vector<float> &y, int steps, int log_every) {
    HostRef ref;
    ref.W.assign(F, 0.0f);// zero init (device starts from the same)
    luisa::vector<float> Y(N_TRAIN), err(N_TRAIN), G(F);
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
}// namespace tensor_example::polyfit

// =============================================================================
// run_poly_fit — driver (invoked by example_tensor's main with --poly-fit)
// =============================================================================
int tensor_example::polyfit::run_poly_fit(int argc, char *argv[]) {
    using luisa::string_view;

    // Collect the backend name from the positional arguments (skipping flags
    // such as --poly-fit), so the same executable serves both the kernel suite
    // and the polynomial-fit training modes.
    string_view backend{};
    int steps = STEPS;
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
        LUISA_INFO("Usage: {} <backend> --poly-fit [--steps N]   (backend = vk | dx | cuda | metal)", argv[0]);
        return 1;
    }
    if (steps <= 0) { steps = STEPS; }
    constexpr int log_every = 200;

    LUISA_INFO("[poly-fit] training data: {} samples in [-pi, pi] ({} gradient steps, lr={})",
               static_cast<int>(N_TRAIN), steps, LR);

    // ---- host data -----------------------------------------------------------
    auto X = make_features(N_TRAIN);
    auto Xt = make_features(N_TEST);
    auto y = make_targets(N_TRAIN);
    auto yt = make_targets(N_TEST);
    luisa::vector<float> XT(static_cast<size_t>(F) * N_TRAIN);
    for (int i = 0; i < N_TRAIN; ++i) {
        for (int k = 0; k < F; ++k) { XT[k * N_TRAIN + i] = X[i * F + k]; }
    }
    auto ref = host_reference(X, XT, y, steps, log_every);

    // ---- device --------------------------------------------------------------
    lc::Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream(lc::StreamTag::COMPUTE);

    auto make_buf = [&](luisa::span<const float> host) {
        auto buf = device.create_buffer<float>(static_cast<uint32_t>(host.size()));
        stream << buf.copy_from(host);
        return buf;
    };

    auto bufX = make_buf(luisa::span{X});
    auto bufXT = make_buf(luisa::span{XT});
    auto bufYref = make_buf(luisa::span{y});
    auto bufXt = make_buf(luisa::span{Xt});
    luisa::vector<float> W0(F, 0.0f);
    auto bufWa = make_buf(luisa::span{W0});
    auto bufWb = device.create_buffer<float>(F);
    auto bufY = device.create_buffer<float>(N_TRAIN);
    auto bufErr = device.create_buffer<float>(N_TRAIN);
    auto bufG = device.create_buffer<float>(F);
    auto bufYt = device.create_buffer<float>(N_TEST);

    // ---- compile the tile kernels --------------------------------------------
    auto sh_fwd = compile_tile(device, make_poly_forward_kernel<N_TRAIN>(), "poly_forward");
    auto sh_fwd_test = compile_tile(device, make_poly_forward_kernel<N_TEST>(), "poly_forward_test");
    auto sh_err = compile_tile(device, make_poly_error_kernel(), "poly_error");
    auto sh_grad = compile_tile(device, make_poly_grad_kernel(), "poly_grad");
    auto sh_upd = compile_tile(device, make_poly_update_kernel(), "poly_update");
    if (!sh_fwd || !sh_fwd_test || !sh_err || !sh_grad || !sh_upd) {
        record("poly_fit_compile", false,
               !sh_fwd ? sh_fwd.metadata().error :
               !sh_fwd_test ? sh_fwd_test.metadata().error :
               !sh_err ? sh_err.metadata().error :
               !sh_grad ? sh_grad.metadata().error : sh_upd.metadata().error);
        return 1;
    }

    // ---- device training loop -------------------------------------------------
    // Each step: forward -> error -> gradient -> SGD update, with the weights
    // ping-ponging between bufWa and bufWb.
    luisa::vector<float> dev_err(N_TRAIN);
    luisa::vector<float> dev_losses;
    luisa::vector<int> dev_log_steps;
    auto train_step = [&](auto &buf_in, auto &buf_out) {
        stream << sh_fwd(bufX, buf_in, bufY).dispatch()
               << sh_err(bufY, bufYref, bufErr).dispatch()
               << sh_grad(bufXT, bufErr, bufG).dispatch()
               << sh_upd(buf_in, bufG, buf_out).dispatch();
    };
    bool w_in_a = true;// current weights live in bufWa
    for (int t = 1; t <= steps; ++t) {
        if (w_in_a) { train_step(bufWa, bufWb); } else { train_step(bufWb, bufWa); }
        w_in_a = !w_in_a;
        if (t % log_every == 0 || t == steps) {
            stream << bufErr.copy_to(luisa::span{dev_err}) << lc::synchronize();
            float loss = 0.0f;
            for (auto e : dev_err) { loss += e * e; }
            dev_losses.push_back(loss);
            dev_log_steps.push_back(t);
            LUISA_INFO("[poly-fit]   step {:5d}  loss = {:.4f}", t, loss);
        }
    }

    // Read back the trained weights (bufWa after an odd number of steps).
    luisa::vector<float> dev_W(F);
    if (w_in_a) {
        stream << bufWa.copy_to(luisa::span{dev_W}) << lc::synchronize();
    } else {
        stream << bufWb.copy_to(luisa::span{dev_W}) << lc::synchronize();
    }

    // ---- inference on a held-out grid (device forward kernel) -----------------
    luisa::vector<float> dev_pred(N_TEST);
    if (w_in_a) {
        stream << sh_fwd_test(bufXt, bufWa, bufYt).dispatch();
    } else {
        stream << sh_fwd_test(bufXt, bufWb, bufYt).dispatch();
    }
    stream << bufYt.copy_to(luisa::span{dev_pred}) << lc::synchronize();

    auto max_err = 0.0;
    auto sum_sq = 0.0;
    for (int i = 0; i < N_TEST; ++i) {
        auto e = static_cast<double>(dev_pred[i]) - yt[i];
        max_err = luisa::max(max_err, luisa::abs(e));
        sum_sq += e * e;
    }
    auto rmse = std::sqrt(sum_sq / static_cast<double>(N_TEST));

    // ---- verify against the host CPU reference --------------------------------
    bool ok = true;
    auto check_value = [&](luisa::string_view what, double dev, double host, double tol) {
        double diff = std::fabs(dev - host);
        LUISA_INFO("[poly-fit]   {}: device = {:.6f}, host = {:.6f}, |diff| = {:.3e}",
                   what, dev, host, diff);
        check(what, diff, tol);
        if (diff > tol) {
            LUISA_WARNING("[poly-fit]   {} mismatch: |diff| {:.3e} > {:.3e}", what, diff, tol);
            ok = false;
        }
    };
    LUISA_INFO("[poly-fit] loss trajectory vs host reference:");
    if (dev_losses.size() != ref.losses.size()) {
        record("poly_fit_log_steps", false,
               luisa::format("host/device log-step mismatch ({} vs {})",
                             ref.losses.size(), dev_losses.size()));
        ok = false;
    }
    for (size_t i = 0; i < dev_losses.size() && i < ref.losses.size(); ++i) {
        // fp32 GEMM/reduce order differs between host and device; allow a
        // relative tolerance on the (large) sum-reduction loss.
        double tol = 1e-2 * std::max(static_cast<double>(ref.losses[i]), 1.0);
        check_value(luisa::format("loss @ step {}", dev_log_steps[i]),
                    dev_losses[i], ref.losses[i], tol);
    }
    LUISA_INFO("[poly-fit] final weights vs host reference:");
    for (int k = 0; k < F; ++k) {
        check_value(luisa::format("W[{}]", k), dev_W[k], ref.W[k], 1e-3);
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
    check("poly_fit_held_out", max_err, 0.25);
    if (max_err >= 0.25) {
        LUISA_WARNING("[poly-fit] self check FAILED: max|err| = {:.6f} >= 0.25", max_err);
        ok = false;
    } else {
        LUISA_INFO("[poly-fit] self check: max|err| = {:.6f} < 0.25 -> PASS", max_err);
    }

    LUISA_INFO("[poly-fit] Verification: {}", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
