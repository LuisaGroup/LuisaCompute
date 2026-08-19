// =============================================================================
// mnist.cpp — Luisa tile-language MLP training on synthetic MNIST
// =============================================================================
// The C++ twin of examples/tensor/mnist_train.py --dataset synthetic: a 2-layer
// MLP (64 -> 32 -> 10, an 8x8 TinyMNIST stand-in so the whole tile stays in
// on-chip shared memory, matching the repo's TinyCNN scale) trained on
// per-class random templates + Gaussian noise with minibatch SGD +
// cross-entropy.
//
// The tile kernels in mlp_kernels.h implement the whole training algorithm;
// this driver traces/lowers/compiles them on the backend (structural device
// validation) and runs the exact same algorithm on the host CPU reference
// (mlp_common.h) to verify the training math reaches the accuracy bound.
//
// NOTE: on some backends the current tile_to_kernel lowering mis-executes
// multi-GEMM-accumulator kernels (a known lowering limitation; the simpler
// poly-fit / linear-regression examples train fully on device), so the
// verification here is host-based.
//
// Verification, mirroring the PyTorch script:
//   1. every tile kernel is traced, lowered and compiled on the backend,
//   2. the host reference must reach >= 80% test accuracy.
//
// This file is part of the single `example_tensor_stub` target and is invoked
// through its main() with the `--mnist` flag (see main.cpp / mnist.h):
//   example_tensor_stub <backend> --mnist [--epochs N]
// =============================================================================

#include "mnist.h"
#include "mlp_kernels.h"
#include "mlp_common.h"

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <cstdlib>
#include <vector>

namespace {

constexpr int B = 32;          // minibatch
constexpr int K1 = 64;         // 8x8 = 64 inputs
constexpr int O1 = 32;         // hidden
constexpr int K2 = O1;         // 32
constexpr int C = 10;          // classes

}// namespace

int mnisttrain::run_mnist(int argc, char *argv[]) {
    using namespace luisa;
    using namespace luisa::compute;

    luisa::string_view backend{};
    int epochs = 30;
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
        LUISA_INFO("Usage: {} <backend> --mnist [--epochs N]   (backend = vk | dx)", argv[0]);
        return 1;
    }
    if (epochs <= 0) { epochs = 30; }

    // ---- data + host reference (the training loop) ----------------------------
    mlpcommon::MlpHyper hp;
    hp.n_train = 20 * 32;      // 640 samples -> exactly 20 minibatches of 32
    hp.n_test = 10 * 32;       // 320 held-out samples
    hp.batch = B;
    hp.epochs = epochs;
    hp.lr = 0.1f;
    hp.num_inputs = K1;
    hp.num_outputs = C;
    hp.widths = {O1};
    hp.seed = 7;

    auto data = mlpcommon::make_synth_mnist_data(hp);
    mlpcommon::finalize_data(hp, data);
    auto host_ref = mlpcommon::mlp_host_reference(hp, data);

    LUISA_INFO("[mnist] SimpleNN({}->{}->{}) on {} synthetic TinyMNIST samples, {} epochs, "
               "minibatch {}, lr {} (host test acc = {:.1f}%)",
               K1, O1, C, hp.n_train, epochs, hp.batch, hp.lr, 100.0 * host_ref.test_acc);

    // ---- device: trace / lower / compile every tile kernel ---------------------
    Context ctx{argv[0]};
    Device device = ctx.create_device(backend);
    Stream stream = device.create_stream();

    Clock c_trace;
    auto k_fc1 = tile::jit(mlp::fc_relu<B, K1, O1>).compile();
    auto k_fc2 = tile::jit(mlp::fc<B, K2, C>).compile();
    auto k_sm = tile::jit(mlp::softmax<B, C>).compile();
    auto k_ce = tile::jit(mlp::ce_grad<B, C>).compile();
    auto k_grad2 = tile::jit(mlp::grad<B, K2, C>).compile();
    auto k_gradb2 = tile::jit(mlp::grad_bias<B, C>).compile();
    auto k_bwd2 = tile::jit(mlp::fc_backward<B, K1, C>).compile();
    auto k_relu1 = tile::jit(mlp::relu_backward<B, O1>).compile();
    auto k_grad1 = tile::jit(mlp::grad<B, K1, O1>).compile();
    auto k_gradb1 = tile::jit(mlp::grad_bias<B, O1>).compile();
    auto k_tA1 = tile::jit(mlp::transpose<B, O1>).compile();
    auto k_tW1 = tile::jit(mlp::transpose<K1, O1>).compile();
    auto k_tW2 = tile::jit(mlp::transpose<K2, C>).compile();
    auto k_upd1 = tile::jit(mlp::update<K1, O1>).compile();
    auto k_upd2 = tile::jit(mlp::update<K2, C>).compile();
    auto k_updb1 = tile::jit(mlp::update_bias<O1>).compile();
    auto k_updb2 = tile::jit(mlp::update_bias<C>).compile();
    double trace_ms = c_trace.toc();

    Clock c_lower;
    auto sh_fc1 = k_fc1.to_kernel<1>();
    auto sh_fc2 = k_fc2.to_kernel<1>();
    auto sh_sm = k_sm.to_kernel<1>();
    auto sh_ce = k_ce.to_kernel<1>();
    auto sh_grad2 = k_grad2.to_kernel<1>();
    auto sh_gradb2 = k_gradb2.to_kernel<1>();
    auto sh_bwd2 = k_bwd2.to_kernel<1>();
    auto sh_relu1 = k_relu1.to_kernel<1>();
    auto sh_grad1 = k_grad1.to_kernel<1>();
    auto sh_gradb1 = k_gradb1.to_kernel<1>();
    auto sh_tA1 = k_tA1.to_kernel<1>();
    auto sh_tW1 = k_tW1.to_kernel<1>();
    auto sh_tW2 = k_tW2.to_kernel<1>();
    auto sh_upd1 = k_upd1.to_kernel<1>();
    auto sh_upd2 = k_upd2.to_kernel<1>();
    auto sh_updb1 = k_updb1.to_kernel<1>();
    auto sh_updb2 = k_updb2.to_kernel<1>();
    double lower_ms = c_lower.toc();

    Clock c_compile;
    device.compile(sh_fc1);
    device.compile(sh_fc2);
    device.compile(sh_sm);
    device.compile(sh_ce);
    device.compile(sh_grad2);
    device.compile(sh_gradb2);
    device.compile(sh_bwd2);
    device.compile(sh_relu1);
    device.compile(sh_grad1);
    device.compile(sh_gradb1);
    device.compile(sh_tA1);
    device.compile(sh_tW1);
    device.compile(sh_tW2);
    device.compile(sh_upd1);
    device.compile(sh_upd2);
    device.compile(sh_updb1);
    device.compile(sh_updb2);
    stream << synchronize();
    double compile_ms = c_compile.toc();

    // ---- verify (host reference) ------------------------------------------------
    bool ok = true;
    if (host_ref.test_acc < 0.80) {
        LUISA_WARNING("[mnist] self check FAILED (host test acc = {:.2f} < 0.80)", host_ref.test_acc);
        ok = false;
    } else {
        LUISA_INFO("[mnist] self check: host test acc {:.1f}% >= 80% -> PASS", 100.0 * host_ref.test_acc);
    }

    LUISA_INFO("[mnist] tile trace: {:.3f} ms, lower: {:.3f} ms, backend compile: {:.3f} ms "
               "({} tile kernels compiled on '{}').",
               trace_ms, lower_ms, compile_ms, 17, backend);
    LUISA_INFO("[mnist] Verification: {}", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
