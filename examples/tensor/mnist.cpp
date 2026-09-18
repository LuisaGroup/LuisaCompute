// =============================================================================
// mnist.cpp — new Tile DSL MLP training on synthetic MNIST
// =============================================================================
// The C++ twin of examples/tensor/mnist_train.py --dataset synthetic: a 2-layer
// MLP (64 -> 32 -> 10, an 8x8 TinyMNIST stand-in so the whole tile stays in
// on-chip shared memory, matching the repo's TinyCNN scale) trained on
// per-class random templates + Gaussian noise with minibatch SGD +
// cross-entropy.
//
// The tile kernels in mlp_kernels.h implement the whole training algorithm;
// this driver captures/compiles them with tensor_example::compile_tile (a
// failure is recorded and aborts the demo with a non-zero return), dispatches
// the full training loop on device buffers via mlpcommon::train_on_device, and
// runs the exact same algorithm on the host CPU reference (mlp_common.h,
// double precision) to verify the training math reaches the accuracy bound.
//
// Verification, mirroring the PyTorch script and the old driver:
//   1. every tile kernel is captured and compiled on the backend, and the
//      full training loop is dispatched on real device buffers,
//   2. the host reference must reach >= 80% test accuracy,
//   3. the device-trained net must reach >= 80% test accuracy and its final
//      train loss must agree with the host reference within tolerance.
//
// This file is part of the single `example_tensor` target and is invoked
// through its main() with the `--mnist` flag (see main.cpp / mnist.h):
//   example_tensor <backend> --mnist [--epochs N]
// =============================================================================

#include "mnist.h"
#include "mlp_common.h"// brings in the mlp:: kernels and the tensor_example helpers

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <cmath>
#include <cstdlib>
#include <luisa/core/stl/vector.h>

namespace {

constexpr int B = 32;          // minibatch
constexpr int K1 = 64;         // 8x8 = 64 inputs
constexpr int O1 = 32;         // hidden
constexpr int K2 = O1;         // 32
constexpr int C = 10;          // classes

// Same accuracy bound as the old driver.
constexpr double test_acc_threshold = 0.80;
// f32 device vs f64 host drift tolerance on the final epoch-average CE loss.
constexpr double loss_diff_tolerance = 0.25;

}// namespace

int tensor_example::mnisttrain::run_mnist(int argc, char *argv[]) {
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

    // ---- device: capture / compile / dispatch the full training loop ----------
    Context ctx{argv[0]};
    Device device = ctx.create_device(backend);
    Stream stream = device.create_stream();

    auto dev = mlpcommon::train_on_device(device, stream, hp, data, "mnist");
    if (!dev.compiled_ok) {
        LUISA_WARNING("[mnist] aborting: a tile kernel failed to compile on '{}' (see FAIL records above).",
                      backend);
        return 1;
    }

    // ---- score the device-trained parameters with a host forward pass ---------
    auto dev_train = mlpcommon::evaluate(hp, data, dev.W, dev.Bias, false);
    auto dev_test = mlpcommon::evaluate(hp, data, dev.W, dev.Bias, true);
    const auto host_final_loss = host_ref.epoch_losses.back();
    const auto loss_diff = std::abs(dev_train.loss - host_final_loss);
    LUISA_INFO("[mnist] device training done: final train loss {:.4f} (host ref {:.4f}), "
               "test acc {:.1f}% (host ref {:.1f}%)",
               dev_train.loss, host_final_loss, 100.0 * dev_test.acc, 100.0 * host_ref.test_acc);

    // ---- verify ----------------------------------------------------------------
    bool ok = true;
    if (host_ref.test_acc < test_acc_threshold) {
        LUISA_WARNING("[mnist] self check FAILED (host test acc = {:.2f} < {:.2f})", host_ref.test_acc, test_acc_threshold);
        ok = false;
    } else {
        LUISA_INFO("[mnist] self check: host test acc {:.1f}% >= {:.0f}% -> PASS",
                   100.0 * host_ref.test_acc, 100.0 * test_acc_threshold);
    }
    tensor_example::record("mnist device test acc", dev_test.acc >= test_acc_threshold,
                           luisa::format("{:.4f} (threshold {:.2f})", dev_test.acc, test_acc_threshold));
    if (dev_test.acc < test_acc_threshold || !std::isfinite(dev_test.acc)) { ok = false; }
    tensor_example::check("mnist device final loss vs host", loss_diff, loss_diff_tolerance);
    if (loss_diff > loss_diff_tolerance || !std::isfinite(loss_diff)) { ok = false; }

    LUISA_INFO("[mnist] tile capture: {:.3f} ms, backend compile: {:.3f} ms "
               "({} tile kernels compiled on '{}').",
               dev.capture_ms, dev.compile_ms, dev.kernel_count, backend);
    LUISA_INFO("[mnist] Verification: {}", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
