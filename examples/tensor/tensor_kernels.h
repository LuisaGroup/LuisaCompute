// =============================================================================
// tensor_kernels.h — Common header for the new Tile DSL tensor example
// =============================================================================
// Rewrite of backup_old_tile/examples/tensor/kernels.h against the
// execution-structure-first C++ Tile DSL (<luisa/tile/dsl.h>). Every kernel is
// captured with tile_kernel(...).capture(...), compiled with
// tile::compile(device, kernel), dispatched on real device buffers through the
// tile::Shader invocation ABI, and checked against a host-side reference.
//
// Each kernel_<name>.cpp in this directory implements, inside namespace
// tensor_example:
//   * make_<name>_kernel() -> tile::Kernel   — capture only (structural)
//   * run_<name>(Device &, Stream &)         — full device pass + verification
// main.cpp owns the case registry and the CLI.
// =============================================================================

#pragma once

#include <luisa/core/logging.h>
#include <luisa/core/mathematics.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/tile/algorithms.h>
#include <luisa/tile/dsl.h>
#include <luisa/tile/runtime.h>
#include <luisa/tile/verifier.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <cmath>
#include <limits>

namespace tensor_example {

namespace tile = luisa::compute::tile;
namespace lc = luisa::compute;

using luisa::string_view;

// ---- result accounting -----------------------------------------------------
// One record() per sub-check; the runner prints a PASS/FAIL line per record
// and exits non-zero when anything failed. Compile failures are recoverable
// and recorded the same way (they usually mean a backend has no Tile support
// for a given feature — see the unimplemented-feature report in main.cpp).
void record(string_view name, bool ok, string_view detail = {}) noexcept;

[[nodiscard]] int failure_count() noexcept;
[[nodiscard]] int pass_count() noexcept;

// Records max-abs-error |err| <= tol for the named case.
void check(string_view name, double err, double tol) noexcept;

// Records a case that is intentionally not executed on the active backend
// (a documented backend limitation, not a failure).
void skip(string_view name, string_view detail) noexcept;

// The backend name passed on the command line, set by main() before any case
// runs. Individual cases use it for documented backend-specific workarounds
// (e.g. the DX12 <4-byte buffer-stride limit for the i8 dtype kernel).
void set_active_backend(string_view name) noexcept;
[[nodiscard]] string_view active_backend() noexcept;

// ---- compilation helper ----------------------------------------------------
// Compiles a captured Tile kernel for the device. Tries the requested lowering
// first; when the device rejects NATIVE lowering because it needs the TVM TIRx
// bridge (CUDA), transparently retries with tile::Lowering::TIRX and logs the
// choice. An invalid return carries the backend diagnostic in
// shader.metadata().error — never aborts.
[[nodiscard]] tile::Shader compile_tile(lc::Device &device, const tile::Kernel &kernel,
                                        string_view name,
                                        tile::Lowering lowering = tile::Lowering::NATIVE) noexcept;

// ---- kernel suite interface ------------------------------------------------
// Implemented by the kernel_<name>.cpp files; registered in main.cpp.

// 2-D tiled elementwise add: C = A + B (elementwise_add).
[[nodiscard]] tile::Kernel make_elementwise_add_kernel();
void run_elementwise_add(lc::Device &device, lc::Stream &stream);

// Tiled f16 GEMM with a software pipeline + ReLU: C = max(A @ B, 0).
[[nodiscard]] tile::Kernel make_pipelined_matmul_kernel();
void run_pipelined_matmul(lc::Device &device, lc::Stream &stream);

// RMSNorm: B[r][c] = A[r][c] * rsqrt(mean_c A[r][c]^2 + 1e-12).
[[nodiscard]] tile::Kernel make_rms_norm_kernel();
void run_rms_norm(lc::Device &device, lc::Stream &stream);

// Fill: C[i] = 3.5.
[[nodiscard]] tile::Kernel make_tile_fill_kernel();
void run_tile_fill(lc::Device &device, lc::Stream &stream);

// Transpose: B[i][j] = A[j][i].
[[nodiscard]] tile::Kernel make_tile_transpose_kernel();
void run_tile_transpose(lc::Device &device, lc::Stream &stream);

// Clamp into [0.1, 0.9].
[[nodiscard]] tile::Kernel make_tile_clamp_kernel();
void run_tile_clamp(lc::Device &device, lc::Stream &stream);

// Row-wise max / min / abssum / absmax reductions.
[[nodiscard]] tile::Kernel make_tile_reduce_kernel();
void run_tile_reduce(lc::Device &device, lc::Stream &stream);

// Inclusive row-wise cumsum / cummax.
[[nodiscard]] tile::Kernel make_tile_scan_kernel();
void run_tile_scan(lc::Device &device, lc::Stream &stream);

// min(|a|, |b|) elementwise.
[[nodiscard]] tile::Kernel make_tile_min_abs_kernel();
void run_tile_min_abs(lc::Device &device, lc::Stream &stream);

// Copy with an explicit producer/consumer dependency (legacy T.sync_threads
// equivalent — synchronization is implicit in the Tile dataflow).
[[nodiscard]] tile::Kernel make_tile_sync_kernel();
void run_tile_sync(lc::Device &device, lc::Stream &stream);

// Whole-tile reduction (legacy T.warp_reduce_sum/max semantics): fill 7.0,
// reduce sum (=448 over 64 lanes) and max (=7).
[[nodiscard]] tile::Kernel make_tile_warp_reduce_kernel();
void run_tile_warp_reduce(lc::Device &device, lc::Stream &stream);

// Row-wise softmax.
[[nodiscard]] tile::Kernel make_softmax_kernel();
void run_softmax(lc::Device &device, lc::Stream &stream);

// Rank-1 elementwise unary math kernels.
[[nodiscard]] tile::Kernel make_exp_kernel();
void run_exp(lc::Device &device, lc::Stream &stream);
[[nodiscard]] tile::Kernel make_log_kernel();
void run_log(lc::Device &device, lc::Stream &stream);
[[nodiscard]] tile::Kernel make_sqrt_kernel();
void run_sqrt(lc::Device &device, lc::Stream &stream);
[[nodiscard]] tile::Kernel make_tanh_kernel();
void run_tanh(lc::Device &device, lc::Stream &stream);
[[nodiscard]] tile::Kernel make_sigmoid_kernel();
void run_sigmoid(lc::Device &device, lc::Stream &stream);
[[nodiscard]] tile::Kernel make_relu_kernel();
void run_relu(lc::Device &device, lc::Stream &stream);
[[nodiscard]] tile::Kernel make_leaky_relu_kernel();
void run_leaky_relu(lc::Device &device, lc::Stream &stream);
[[nodiscard]] tile::Kernel make_gelu_kernel();
void run_gelu(lc::Device &device, lc::Stream &stream);
[[nodiscard]] tile::Kernel make_identity_kernel();
void run_identity(lc::Device &device, lc::Stream &stream);
[[nodiscard]] tile::Kernel make_reciprocal_kernel();
void run_reciprocal(lc::Device &device, lc::Stream &stream);
[[nodiscard]] tile::Kernel make_neg_kernel();
void run_neg(lc::Device &device, lc::Stream &stream);
[[nodiscard]] tile::Kernel make_cast_kernel();
void run_cast(lc::Device &device, lc::Stream &stream);

// pow(a, b) composed as exp(b * log(a)).
[[nodiscard]] tile::Kernel make_pow_kernel();
void run_pow(lc::Device &device, lc::Stream &stream);

// Dtype coverage: 1-D copy for f16/f32/i32/i8, elementwise add for f32/i32,
// neg for f32. (FP8 / I4 / FP4 have no ScalarType in the new TileIR — see the
// unimplemented-feature report.)
void run_dtypes(lc::Device &device, lc::Stream &stream);

// ---- training demos (poly_fit.h, mlp.h, ...) -------------------------------
// Each driver parses its own extra flags and creates its own device/stream,
// mirroring the old example's split-usage style.
namespace polyfit { int run_poly_fit(int argc, char *argv[]); }
namespace lreg { int run_linear_regression(int argc, char *argv[]); }
namespace mlptrain { int run_mlp(int argc, char *argv[]); }
namespace mnisttrain { int run_mnist(int argc, char *argv[]); }
namespace rnntrain { int run_rnn(int argc, char *argv[]); }
namespace cnn { int run_cnn_inference(int argc, char *argv[]); }
namespace basics { int run_basics(int argc, char *argv[]); }

}// namespace tensor_example
