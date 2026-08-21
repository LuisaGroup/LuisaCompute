// =============================================================================
// kernels.h — Common header for all tile/tensor DSL kernel functions
// =============================================================================
// Declares every kernel function that was originally in tensor_stub.cpp.
// Each kernel is traced with tile::jit(...).compile(), lowered to a regular
// Luisa kernel with tile_to_kernel, compiled on a backend, dispatched on real
// buffers, and checked against a host-side reference.
// =============================================================================

#pragma once

#include <luisa/dsl/tensor.h>
#include <luisa/dsl/tile_to_kernel.h>
#include <luisa/dsl/func.h>
#include <luisa/core/mathematics.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <string>
#include <limits>

// TileLang's `import tilelang.language as T` is exposed as the `LuisaTensor`
// constexpr handle (a C++ namespace can only be addressed with `::`, so the
// `LuisaTensor.*` dot syntax comes from the `dsl` handle object).
constexpr auto LuisaTensor = luisa::compute::tile::language::dsl;
using luisa::compute::tile::language::Tensor;

using tile_f16 = luisa::compute::tile::half;
using tile_f32 = luisa::compute::tile::float32;
using tile_i32 = luisa::compute::tile::int32;

// =============================================================================
// Kernel function declarations
// =============================================================================

// 1. Elementwise add — ALLOC / CEILDIV / KERNEL_2D / COPY / BINARY / STORE
Tensor<tile_f32, 2> elementwise_add(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> B);

// 2. Tiled GEMM with a software pipeline — CLEAR / PIPELINED / GEMM / MAX / PRINT / STORE
Tensor<tile_f16, 2> pipelined_matmul(Tensor<tile_f16, 2> A, Tensor<tile_f16, 2> B);

// 3. RMSNorm — KERNEL_1D / COPY / BINARY / STORE / REDUCE_SUM / RSQRT
Tensor<tile_f32, 2> rms_norm(Tensor<tile_f32, 2> A);

// 4. T.fill — FILL
Tensor<tile_f32, 1> tile_fill_kernel();

// 5. T.transpose — TRANSPOSE
Tensor<tile_f32, 2> tile_transpose_kernel(Tensor<tile_f32, 2> A);

// 6. T.clamp — CLAMP
Tensor<tile_f32, 2> tile_clamp_kernel(Tensor<tile_f32, 2> A);

// 7. T.atomic_* — ATOMIC
Tensor<tile_i32, 1> tile_atomic_kernel();

// 8. T.sync_threads — SYNC
Tensor<tile_f32, 2> tile_sync_kernel(Tensor<tile_f32, 2> A);

// 9. T.warp_reduce_sum / max — WARP_REDUCE
Tensor<tile_f32, 1> tile_warp_reduce_kernel();

// 10. T.loop_break — LOOP_BREAK (traced only)
Tensor<tile_f32, 2> loop_break_kernel(Tensor<tile_f32, 2> A);

// 12. T.reduce_max / reduce_min / reduce_abssum / reduce_absmax — REDUCE
Tensor<tile_f32, 1> tile_reduce_kernel(Tensor<tile_f32, 2> A,
                                       Tensor<tile_f32, 1> Bmax,
                                       Tensor<tile_f32, 1> Bmin,
                                       Tensor<tile_f32, 1> Babssum);

// 13. T.cumsum / T.cummax — CUMSUM / CUMMAX
Tensor<tile_f32, 1> tile_scan_kernel(Tensor<tile_f32, 1> A, Tensor<tile_f32, 1> S);

// 14. T.min / T.abs — MIN / ABS
Tensor<tile_f32, 2> tile_min_abs_kernel(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> B);

// 15. T.any_of / T.all_of / T.shfl_* — ANY_OF / ALL_OF / SHUFFLE
Tensor<tile_f32, 1> tile_vote_shuffle_kernel();

// 17. exp_kernel — EXP
Tensor<tile_f32, 1> exp_kernel(Tensor<tile_f32, 1> A);

// 18. log_kernel — LOG
Tensor<tile_f32, 1> log_kernel(Tensor<tile_f32, 1> A);

// 19. sqrt_kernel — SQRT
Tensor<tile_f32, 1> sqrt_kernel(Tensor<tile_f32, 1> A);

// 20. sin_kernel — SIN
Tensor<tile_f32, 1> sin_kernel(Tensor<tile_f32, 1> A);

// 21. cos_kernel — COS
Tensor<tile_f32, 1> cos_kernel(Tensor<tile_f32, 1> A);

// 22. tan_kernel — TAN
Tensor<tile_f32, 1> tan_kernel(Tensor<tile_f32, 1> A);

// 23. tanh_kernel — TANH
Tensor<tile_f32, 1> tanh_kernel(Tensor<tile_f32, 1> A);

// 24. erf_kernel — ERF
Tensor<tile_f32, 1> erf_kernel(Tensor<tile_f32, 1> A);

// 25. ceil_kernel — CEIL
Tensor<tile_f32, 1> ceil_kernel(Tensor<tile_f32, 1> A);

// 26. floor_kernel — FLOOR
Tensor<tile_f32, 1> floor_kernel(Tensor<tile_f32, 1> A);

// 27. round_kernel — ROUND
Tensor<tile_f32, 1> round_kernel(Tensor<tile_f32, 1> A);

// 28. isinf_kernel — ISINF (returns int32)
Tensor<tile_i32, 1> isinf_kernel(Tensor<tile_f32, 1> A);

// 29. isnan_kernel — ISNAN (returns int32)
Tensor<tile_i32, 1> isnan_kernel(Tensor<tile_f32, 1> A);

// 30. cast_kernel — CAST (int32 -> float32)
Tensor<tile_f32, 1> cast_kernel(Tensor<tile_i32, 1> A);

// 31. neg_kernel — NEG
Tensor<tile_f32, 1> neg_kernel(Tensor<tile_f32, 1> A);

// 32. relu_kernel — RELU
Tensor<tile_f32, 1> relu_kernel(Tensor<tile_f32, 1> A);

// 33. sigmoid_kernel — SIGMOID
Tensor<tile_f32, 1> sigmoid_kernel(Tensor<tile_f32, 1> A);

// 34. leaky_relu_kernel — LEAKY_RELU
Tensor<tile_f32, 1> leaky_relu_kernel(Tensor<tile_f32, 1> A);

// 35. softmax_kernel — SOFTMAX
Tensor<tile_f32, 2> softmax_kernel(Tensor<tile_f32, 2> A);

// 36. pow_kernel — POW
Tensor<tile_f32, 1> pow_kernel(Tensor<tile_f32, 1> A, Tensor<tile_f32, 1> B);

// 37. gelu_kernel — GELU
Tensor<tile_f32, 1> gelu_kernel(Tensor<tile_f32, 1> A);

// 38. identity_kernel — IDENTITY
Tensor<tile_f32, 1> identity_kernel(Tensor<tile_f32, 1> A);

// 39. reciprocal_kernel — RECIPROCAL
Tensor<tile_f32, 1> reciprocal_kernel(Tensor<tile_f32, 1> A);

// 40. Multiple-T.Kernel guard (INVALID — triggers abort)
Tensor<tile_f32, 2> two_kernels(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> B);