// =============================================================================
// mlp.cpp — new Tile DSL 3-layer MLP training
// =============================================================================
// The C++ twin of examples/tensor/mlp_train.py: a 50 -> 30 -> 15 -> 4 ReLU MLP
// trained on the synthetic XOR-style quadrant task (two informative features +
// 48 noise features) with minibatch SGD + cross-entropy.
//
// The tile kernels in mlp_kernels.h implement the whole training algorithm —
// forward GEMMs + ReLU + bias, softmax / cross-entropy gradient, manual
// backprop GEMMs and SGD updates.  This driver captures and compiles every
// kernel with tensor_example::compile_tile (a failure is recorded and aborts
// the demo with a non-zero return), then runs the same algorithm as real
// kernel dispatches on device buffers through mlpcommon::train_on_device.
// The exact same algorithm also runs on the host CPU reference (mlp_common.h,
// double precision) to verify the training math reaches the PyTorch accuracy
// bound, and the device-trained parameters are scored with a host forward pass
// and compared against the host reference.
//
// Verification, mirroring the PyTorch script and the old driver:
//   1. every tile kernel is captured and compiled on the backend, and the
//      full training loop is dispatched on real device buffers,
//   2. the host reference must reach >= 82% test accuracy (the PyTorch bound),
//   3. the device-trained net must reach >= 82% test accuracy and its final
//      train loss must agree with the host reference within tolerance.
//
// This file is part of the single `example_tensor` target and is invoked
// through its main() with the `--mlp` flag (see main.cpp / mlp.h):
//   example_tensor <backend> --mlp [--epochs N]
// =============================================================================

#include "mlp.h"
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
constexpr int K1 = 50;         // input features
constexpr int O1 = 30;         // layer-1 hidden
constexpr int K2 = O1;         // 30
constexpr int O2 = 15;         // layer-2 hidden
constexpr int K3 = O2;         // 15
constexpr int C = 4;           // classes

// Same PyTorch accuracy bound as the old driver.
constexpr double test_acc_threshold = 0.82;
// The device trains in f32 while the host reference is double precision; after
// 60 epochs x 32 minibatches the runs may drift slightly, so the final
// epoch-average CE loss only has to agree loosely.
constexpr double loss_diff_tolerance = 0.25;

}// namespace

int tensor_example::mlptrain::run_mlp(int argc, char *argv[]) {
    using namespace luisa;
    using namespace luisa::compute;

    luisa::string_view backend{};
    int epochs = 60;
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
        LUISA_INFO("Usage: {} <backend> --mlp [--epochs N]   (backend = vk | dx)", argv[0]);
        return 1;
    }
    if (epochs <= 0) { epochs = 60; }

    // ---- data + host reference (the training loop) ----------------------------
    mlpcommon::MlpHyper hp;
    hp.n_train = 32 * 32;      // 1024 samples -> exactly 32 minibatches of 32
    hp.n_test = 12 * 32;       // 384 held-out samples
    hp.batch = B;
    hp.epochs = epochs;
    hp.lr = 0.1f;
    hp.num_inputs = K1;
    hp.num_outputs = C;
    hp.widths = {O1, O2};
    hp.seed = 123;

    auto data = mlpcommon::make_xor_data(hp);
    mlpcommon::finalize_data(hp, data);
    auto host_ref = mlpcommon::mlp_host_reference(hp, data);

    LUISA_INFO("[mlp] MLP(50->30->15->4) on {} synthetic XOR samples, {} epochs, "
               "minibatch {}, lr {} (host test acc = {:.1f}%)",
               hp.n_train, epochs, hp.batch, hp.lr, 100.0 * host_ref.test_acc);

    // ---- device: capture / compile / dispatch the full training loop ----------
    Context ctx{argv[0]};
    Device device = ctx.create_device(backend);
    Stream stream = device.create_stream();

    auto dev = mlpcommon::train_on_device(device, stream, hp, data, "mlp");
    if (!dev.compiled_ok) {
        LUISA_WARNING("[mlp] aborting: a tile kernel failed to compile on '{}' (see FAIL records above).",
                      backend);
        return 1;
    }

    // ---- score the device-trained parameters with a host forward pass ---------
    auto dev_train = mlpcommon::evaluate(hp, data, dev.W, dev.Bias, false);
    auto dev_test = mlpcommon::evaluate(hp, data, dev.W, dev.Bias, true);
    const auto host_final_loss = host_ref.epoch_losses.back();
    const auto loss_diff = std::abs(dev_train.loss - host_final_loss);
    LUISA_INFO("[mlp] device training done: final train loss {:.4f} (host ref {:.4f}), "
               "test acc {:.1f}% (host ref {:.1f}%)",
               dev_train.loss, host_final_loss, 100.0 * dev_test.acc, 100.0 * host_ref.test_acc);

    // ---- verify ----------------------------------------------------------------
    bool ok = true;
    if (host_ref.test_acc < test_acc_threshold) {
        LUISA_WARNING("[mlp] self check FAILED (host test acc = {:.2f} < {:.2f})", host_ref.test_acc, test_acc_threshold);
        ok = false;
    } else {
        LUISA_INFO("[mlp] self check: host test acc {:.1f}% >= {:.0f}% -> PASS",
                   100.0 * host_ref.test_acc, 100.0 * test_acc_threshold);
    }
    tensor_example::record("mlp device test acc", dev_test.acc >= test_acc_threshold,
                           luisa::format("{:.4f} (threshold {:.2f})", dev_test.acc, test_acc_threshold));
    if (dev_test.acc < test_acc_threshold || !std::isfinite(dev_test.acc)) { ok = false; }
    tensor_example::check("mlp device final loss vs host", loss_diff, loss_diff_tolerance);
    if (loss_diff > loss_diff_tolerance || !std::isfinite(loss_diff)) { ok = false; }

    LUISA_INFO("[mlp] tile capture: {:.3f} ms, backend compile: {:.3f} ms "
               "({} tile kernels compiled on '{}').",
               dev.capture_ms, dev.compile_ms, dev.kernel_count, backend);
    LUISA_INFO("[mlp] Verification: {}", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
