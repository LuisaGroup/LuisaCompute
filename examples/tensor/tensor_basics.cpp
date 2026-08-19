// =============================================================================
// tensor_basics.cpp — Luisa tile-language tensor basics exercises
// =============================================================================
// The C++ twin of examples/tensor/tensor_basics.py: a self-contained tour of
// the fundamentals, each step verified by an assert:
//   1. Tensors    — creation and shapes (host-side vector bookkeeping)
//   2. Operations — elementwise arithmetic through a tile add/mul kernel
//   3. Autograd   — y = x^2 + 2x + 1 at x = 3 -> y = 16, dy/dx = 8 through a
//                   tile kernel computing both the value and the derivative
//   4. Tiny NN    — a 1 -> 1 net (Linear + ReLU; ReLU is identity on the
//                   positive training inputs) trained with manual SGD tile
//                   kernels, then used for inference on new points
//
// This file is part of the single `example_tensor_stub` target and is invoked
// through its main() with the `--basics` flag (see main.cpp / tensor_basics.h):
//   example_tensor_stub <backend> --basics
// =============================================================================

#include "tensor_basics.h"
#include "tensor_basics_kernels.h"

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

constexpr int N = 4;        // tiny-NN dataset size
constexpr int STEPS = 2000; // tiny-NN training steps (tensor_basics.py default)

}// namespace

int basics::run_basics(int argc, char *argv[]) {
    using namespace luisa;
    using namespace luisa::compute;

    luisa::string_view backend{};
    for (auto i = 1; i < argc; ++i) {
        if (argv != nullptr && argv[i] != nullptr && !luisa::string_view{argv[i]}.starts_with("--")) {
            if (backend.empty()) { backend = argv[i]; }
        }
    }
    if (backend.empty()) {
        LUISA_INFO("Usage: {} <backend> --basics   (backend = vk | dx)", argv[0]);
        return 1;
    }

    // ---- device -------------------------------------------------------------
    Context ctx{argv[0]};
    Device device = ctx.create_device(backend);
    Stream stream = device.create_stream();
    auto make_buf = [&](luisa::span<const float> host) {
        auto buf = device.create_buffer<float>(static_cast<uint32_t>(host.size()));
        stream << buf.copy_from(host);
        return buf;
    };

    bool ok = true;

    // =========================================================================
    // Exercise 1: tensors — creation and shapes (host-side)
    // =========================================================================
    LUISA_INFO("[basics] [exercise 1] tensors");
    const std::vector<float> t1{1.0f, 2.0f, 3.0f};
    LUISA_INFO("[basics]   t1.size() = {} (shape (3,), float32)", t1.size());
    if (t1.size() != 3u) { ok = false; }

    // =========================================================================
    // Exercise 2: operations — elementwise add / mul through tile kernels
    // =========================================================================
    LUISA_INFO("[basics] [exercise 2] operations");
    auto k_addmul = tile::jit(basics::basic_addmul<3>).compile();
    auto sh_addmul = k_addmul.to_kernel<1>();
    auto sc_addmul = device.compile(sh_addmul);
    auto bufA = make_buf(std::vector<float>{2.0f, 4.0f, 6.0f});
    auto bufB = make_buf(std::vector<float>{1.0f, 3.0f, 5.0f});
    auto bufC = device.create_buffer<float>(3);
    auto bufD = device.create_buffer<float>(3);
    std::vector<float> hC(3), hD(3);
    stream << sc_addmul(bufA, bufB, bufD, bufC).dispatch(32u)
           << bufC.copy_to(luisa::span{hC}) << bufD.copy_to(luisa::span{hD}) << synchronize();
    LUISA_INFO("[basics]   a + b = [{:.1f}, {:.1f}, {:.1f}]", hC[0], hC[1], hC[2]);
    LUISA_INFO("[basics]   a * b = [{:.1f}, {:.1f}, {:.1f}]", hD[0], hD[1], hD[2]);
    const float add_ref[3] = {3.0f, 7.0f, 11.0f};
    const float mul_ref[3] = {2.0f, 12.0f, 30.0f};
    for (int i = 0; i < 3; ++i) {
        if (std::fabs(hC[i] - add_ref[i]) > 1e-5f || std::fabs(hD[i] - mul_ref[i]) > 1e-5f) { ok = false; }
    }

    // =========================================================================
    // Exercise 3: autograd — y = x^2 + 2x + 1 at x = 3 -> y = 16, dy/dx = 8
    // =========================================================================
    LUISA_INFO("[basics] [exercise 3] autograd");
    auto k_sq = tile::jit(basics::basic_square_grad<1>).compile();
    auto sh_sq = k_sq.to_kernel<1>();
    auto sc_sq = device.compile(sh_sq);
    auto bufX = make_buf(std::vector<float>{3.0f});
    auto bufY = device.create_buffer<float>(1);
    auto bufDY = device.create_buffer<float>(1);
    std::vector<float> hY(1), hDY(1);
    stream << sc_sq(bufX, bufDY, bufY).dispatch(32u)
           << bufY.copy_to(luisa::span{hY}) << bufDY.copy_to(luisa::span{hDY}) << synchronize();
    LUISA_INFO("[basics]   y = x^2 + 2x + 1 at x = 3 -> y = {:.3f}, dy/dx = {:.3f}", hY[0], hDY[0]);
    if (std::fabs(hY[0] - 16.0f) > 1e-4f || std::fabs(hDY[0] - 8.0f) > 1e-4f) {
        LUISA_WARNING("[basics] autograd check FAILED (y={}, dy={})", hY[0], hDY[0]);
        ok = false;
    }

    // =========================================================================
    // Exercise 4: simple neural network — train + inference (1 -> 1 net)
    // =========================================================================
    LUISA_INFO("[basics] [exercise 4] simple neural network (train {} steps)", STEPS);
    auto k_fwd = tile::jit(basics::nn_forward<N>).compile();
    auto k_err = tile::jit(basics::nn_error<N>).compile();
    auto k_grad = tile::jit(basics::nn_grad<N>).compile();
    auto k_upd = tile::jit(basics::nn_update<2>).compile();
    auto sh_fwd = k_fwd.to_kernel<1>();
    auto sh_err = k_err.to_kernel<1>();
    auto sh_grad = k_grad.to_kernel<1>();
    auto sh_upd = k_upd.to_kernel<1>();
    auto sc_fwd = device.compile(sh_fwd);
    auto sc_err = device.compile(sh_err);
    auto sc_grad = device.compile(sh_grad);
    auto sc_upd = device.compile(sh_upd);

    // data: x = [1,2,3,4], targets = 2x (augmented with a bias column)
    std::vector<float> Xb{1.0f, 1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 4.0f, 1.0f};
    std::vector<float> targets{2.0f, 4.0f, 6.0f, 8.0f};
    std::vector<float> XT{1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    auto bufXb = make_buf(luisa::span{Xb});
    auto bufXT = make_buf(luisa::span{XT});
    auto bufTgt = make_buf(luisa::span{targets});
    std::vector<float> W0{0.0f, 0.0f};
    auto bufW = make_buf(luisa::span{W0});
    auto bufW2 = device.create_buffer<float>(2);
    auto bufYtrain = device.create_buffer<float>(N);
    auto bufErr = device.create_buffer<float>(N);
    auto bufG = device.create_buffer<float>(2);
    std::vector<float> hErr(N);

    auto train_step = [&](auto &buf_in, auto &buf_out) {
        stream << sc_fwd(bufXb, buf_in, bufYtrain).dispatch(32u)
               << sc_err(bufYtrain, bufTgt, bufErr).dispatch(32u)
               << sc_grad(bufXT, bufErr, bufG).dispatch(32u)
               << sc_upd(buf_in, bufG, buf_out).dispatch(32u);
    };
    bool w_in_a = true;
    for (int step = 1; step <= STEPS; ++step) {
        if (w_in_a) { train_step(bufW, bufW2); } else { train_step(bufW2, bufW); }
        w_in_a = !w_in_a;
        if (step % 500 == 0 || step == STEPS) {
            stream << bufErr.copy_to(luisa::span{hErr}) << synchronize();
            float loss = 0.0f;
            for (auto e : hErr) { loss += e * e; }
            LUISA_INFO("[basics]   step {:5d}  loss = {:.6f}", step, loss / N);
        }
    }
    std::vector<float> hW(2);
    if (w_in_a) {
        stream << bufW.copy_to(luisa::span{hW}) << synchronize();
    } else {
        stream << bufW2.copy_to(luisa::span{hW}) << synchronize();
    }

    // inference on new points the network never saw: x = [0.5, 1.5, 2.5, 3.5]
    std::vector<float> Xb_new{0.5f, 1.0f, 1.5f, 1.0f, 2.5f, 1.0f, 3.5f, 1.0f};
    auto bufXb_new = make_buf(luisa::span{Xb_new});
    auto bufY_new = device.create_buffer<float>(N);
    if (w_in_a) {
        stream << sc_fwd(bufXb_new, bufW, bufY_new).dispatch(32u);
    } else {
        stream << sc_fwd(bufXb_new, bufW2, bufY_new).dispatch(32u);
    }
    std::vector<float> hPred(N);
    stream << bufY_new.copy_to(luisa::span{hPred}) << synchronize();

    float max_err = 0.0f;
    const float expected[4] = {1.0f, 3.0f, 5.0f, 7.0f};
    for (int i = 0; i < N; ++i) { max_err = std::max(max_err, std::fabs(hPred[i] - expected[i])); }
    LUISA_INFO("[basics]   inference on new points: pred = [{:.3f}, {:.3f}, {:.3f}, {:.3f}]",
               hPred[0], hPred[1], hPred[2], hPred[3]);
    LUISA_INFO("[basics]   expected (y = 2x)      : [1.000, 3.000, 5.000, 7.000]");
    LUISA_INFO("[basics]   max|err| = {:.4f}", max_err);
    if (max_err >= 0.1f) {
        LUISA_WARNING("[basics] tiny NN failed to learn y = 2x (max|err| = {:.4f})", max_err);
        ok = false;
    } else {
        LUISA_INFO("[basics]   OK: training + inference completed");
    }

    LUISA_INFO("[basics] Verification: {}", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
