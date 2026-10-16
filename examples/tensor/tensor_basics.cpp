// =============================================================================
// tensor_basics.cpp — tensor basics exercises on the new Tile DSL
// =============================================================================
// The C++ twin of examples/tensor/tensor_basics.py: a self-contained tour of
// the fundamentals, each step verified against a host reference:
//   1. Tensors    — creation and shapes (host-side vector bookkeeping)
//   2. Operations — elementwise arithmetic through a tile add/mul kernel
//   3. Autograd   — y = x^2 + 2x + 1 at x = 3 -> y = 16, dy/dx = 8 through a
//                   tile kernel computing both the value and the derivative
//   4. Tiny NN    — a 1 -> 1 net (Linear + ReLU; ReLU is identity on the
//                   positive training inputs) trained with manual SGD tile
//                   kernels, then used for inference on new points
//
// Port of backup_old_tile/examples/tensor/tensor_basics.cpp. Every kernel is
// captured with the new execution-structure-first Tile DSL and compiled via
// tensor_example::compile_tile (never aborts; an invalid shader carries the
// backend diagnostic in shader.metadata().error).
//
// Invoked through example_tensor's main():
//   example_tensor <backend> --basics
// =============================================================================

#include "tensor_basics.h"
#include "tensor_basics_kernels.h"
#include "tensor_kernels.h"

#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

namespace {

constexpr int N = 4;        // tiny-NN dataset size
constexpr int STEPS = 2000; // tiny-NN training steps (tensor_basics.py default)

}// namespace

int tensor_example::basics::run_basics(int argc, char *argv[]) {
    using luisa::string_view;

    string_view backend{};
    for (auto i = 1; i < argc; ++i) {
        if (argv != nullptr && argv[i] != nullptr && !string_view{argv[i]}.starts_with("--")) {
            if (backend.empty()) { backend = argv[i]; }
        }
    }
    if (backend.empty()) {
        LUISA_INFO("Usage: {} <backend> --basics   (backend = vk | dx | cuda | metal)", argv[0]);
        return 1;
    }

    // ---- device -------------------------------------------------------------
    lc::Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream(lc::StreamTag::COMPUTE);
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
    const luisa::vector<float> t1{1.0f, 2.0f, 3.0f};
    LUISA_INFO("[basics]   t1.size() = {} (shape (3,), float32)", t1.size());
    record("basics_tensors", t1.size() == 3u);
    if (t1.size() != 3u) { ok = false; }

    // =========================================================================
    // Exercise 2: operations — elementwise add / mul through tile kernels
    // =========================================================================
    LUISA_INFO("[basics] [exercise 2] operations");
    auto shader_addmul = compile_tile(device, make_basic_addmul_kernel<3>(), "basic_addmul");
    if (!shader_addmul) {
        record("basic_addmul", false, shader_addmul.metadata().error);
        return 1;
    }
    auto bufA = make_buf(luisa::vector<float>{2.0f, 4.0f, 6.0f});
    auto bufB = make_buf(luisa::vector<float>{1.0f, 3.0f, 5.0f});
    auto bufC = device.create_buffer<float>(3);
    auto bufD = device.create_buffer<float>(3);
    luisa::vector<float> hC(3), hD(3);
    stream << shader_addmul(bufA, bufB, bufC, bufD).dispatch()
           << bufC.copy_to(luisa::span{hC}) << bufD.copy_to(luisa::span{hD}) << lc::synchronize();
    LUISA_INFO("[basics]   a + b = [{:.1f}, {:.1f}, {:.1f}]", hC[0], hC[1], hC[2]);
    LUISA_INFO("[basics]   a * b = [{:.1f}, {:.1f}, {:.1f}]", hD[0], hD[1], hD[2]);
    const float add_ref[3] = {3.0f, 7.0f, 11.0f};
    const float mul_ref[3] = {2.0f, 12.0f, 30.0f};
    auto addmul_err = 0.0;
    for (int i = 0; i < 3; ++i) {
        addmul_err = luisa::max(addmul_err, luisa::abs(static_cast<double>(hC[i]) - add_ref[i]));
        addmul_err = luisa::max(addmul_err, luisa::abs(static_cast<double>(hD[i]) - mul_ref[i]));
    }
    check("basic_addmul", addmul_err, 1e-5);
    if (addmul_err > 1e-5) { ok = false; }

    // =========================================================================
    // Exercise 3: autograd — y = x^2 + 2x + 1 at x = 3 -> y = 16, dy/dx = 8
    // =========================================================================
    LUISA_INFO("[basics] [exercise 3] autograd");
    auto shader_sq = compile_tile(device, make_basic_square_grad_kernel<1>(), "basic_square_grad");
    if (!shader_sq) {
        record("basic_square_grad", false, shader_sq.metadata().error);
        return 1;
    }
    auto bufX = make_buf(luisa::vector<float>{3.0f});
    auto bufY = device.create_buffer<float>(1);
    auto bufDY = device.create_buffer<float>(1);
    luisa::vector<float> hY(1), hDY(1);
    stream << shader_sq(bufX, bufY, bufDY).dispatch()
           << bufY.copy_to(luisa::span{hY}) << bufDY.copy_to(luisa::span{hDY}) << lc::synchronize();
    LUISA_INFO("[basics]   y = x^2 + 2x + 1 at x = 3 -> y = {:.3f}, dy/dx = {:.3f}", hY[0], hDY[0]);
    auto sq_err = luisa::max(luisa::abs(static_cast<double>(hY[0]) - 16.0),
                             luisa::abs(static_cast<double>(hDY[0]) - 8.0));
    check("basic_square_grad", sq_err, 1e-4);
    if (sq_err > 1e-4) {
        LUISA_WARNING("[basics] autograd check FAILED (y={}, dy={})", hY[0], hDY[0]);
        ok = false;
    }

    // =========================================================================
    // Exercise 4: simple neural network — train + inference (1 -> 1 net)
    // =========================================================================
    LUISA_INFO("[basics] [exercise 4] simple neural network (train {} steps)", STEPS);
    auto shader_fwd = compile_tile(device, make_nn_forward_kernel<N>(), "nn_forward");
    auto shader_err = compile_tile(device, make_nn_error_kernel<N>(), "nn_error");
    auto shader_grad = compile_tile(device, make_nn_grad_kernel<N>(), "nn_grad");
    auto shader_upd = compile_tile(device, make_nn_update_kernel<2>(), "nn_update");
    if (!shader_fwd || !shader_err || !shader_grad || !shader_upd) {
        record("basics_tiny_nn", false,
               !shader_fwd ? shader_fwd.metadata().error :
               !shader_err ? shader_err.metadata().error :
               !shader_grad ? shader_grad.metadata().error : shader_upd.metadata().error);
        return 1;
    }

    // data: x = [1,2,3,4], targets = 2x (augmented with a bias column)
    luisa::vector<float> Xb{1.0f, 1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 4.0f, 1.0f};
    luisa::vector<float> targets{2.0f, 4.0f, 6.0f, 8.0f};
    luisa::vector<float> XT{1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    auto bufXb = make_buf(luisa::span{Xb});
    auto bufXT = make_buf(luisa::span{XT});
    auto bufTgt = make_buf(luisa::span{targets});
    luisa::vector<float> W0{0.0f, 0.0f};
    auto bufW = make_buf(luisa::span{W0});
    auto bufW2 = device.create_buffer<float>(2);
    auto bufYtrain = device.create_buffer<float>(N);
    auto bufErr = device.create_buffer<float>(N);
    auto bufG = device.create_buffer<float>(2);
    luisa::vector<float> hErr(N);

    auto train_step = [&](auto &buf_in, auto &buf_out) {
        stream << shader_fwd(bufXb, buf_in, bufYtrain).dispatch()
               << shader_err(bufYtrain, bufTgt, bufErr).dispatch()
               << shader_grad(bufXT, bufErr, bufG).dispatch()
               << shader_upd(buf_in, bufG, buf_out).dispatch();
    };
    bool w_in_a = true;
    for (int step = 1; step <= STEPS; ++step) {
        if (w_in_a) { train_step(bufW, bufW2); } else { train_step(bufW2, bufW); }
        w_in_a = !w_in_a;
        if (step % 500 == 0 || step == STEPS) {
            stream << bufErr.copy_to(luisa::span{hErr}) << lc::synchronize();
            float loss = 0.0f;
            for (auto e : hErr) { loss += e * e; }
            LUISA_INFO("[basics]   step {:5d}  loss = {:.6f}", step, loss / N);
        }
    }
    luisa::vector<float> hW(2);
    if (w_in_a) {
        stream << bufW.copy_to(luisa::span{hW}) << lc::synchronize();
    } else {
        stream << bufW2.copy_to(luisa::span{hW}) << lc::synchronize();
    }

    // inference on new points the network never saw: x = [0.5, 1.5, 2.5, 3.5]
    luisa::vector<float> Xb_new{0.5f, 1.0f, 1.5f, 1.0f, 2.5f, 1.0f, 3.5f, 1.0f};
    auto bufXb_new = make_buf(luisa::span{Xb_new});
    auto bufY_new = device.create_buffer<float>(N);
    if (w_in_a) {
        stream << shader_fwd(bufXb_new, bufW, bufY_new).dispatch();
    } else {
        stream << shader_fwd(bufXb_new, bufW2, bufY_new).dispatch();
    }
    luisa::vector<float> hPred(N);
    stream << bufY_new.copy_to(luisa::span{hPred}) << lc::synchronize();

    auto max_err = 0.0;
    const float expected[4] = {1.0f, 3.0f, 5.0f, 7.0f};
    for (int i = 0; i < N; ++i) {
        max_err = luisa::max(max_err, luisa::abs(static_cast<double>(hPred[i]) - expected[i]));
    }
    LUISA_INFO("[basics]   inference on new points: pred = [{:.3f}, {:.3f}, {:.3f}, {:.3f}]",
               hPred[0], hPred[1], hPred[2], hPred[3]);
    LUISA_INFO("[basics]   expected (y = 2x)      : [1.000, 3.000, 5.000, 7.000]");
    LUISA_INFO("[basics]   max|err| = {:.4f}", max_err);
    check("basics_tiny_nn", max_err, 0.1);
    if (max_err >= 0.1) {
        LUISA_WARNING("[basics] tiny NN failed to learn y = 2x (max|err| = {:.4f})", max_err);
        ok = false;
    } else {
        LUISA_INFO("[basics]   OK: training + inference completed");
    }

    LUISA_INFO("[basics] Verification: {}", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
