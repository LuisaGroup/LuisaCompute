// =============================================================================
// kernel_dtypes.cpp — COPY / BINARY / NEG for every TensorElementType
// =============================================================================
// Exercises the tile-to-kernel lowering for all seven TensorElementType values
// (F16, F32, I32, I8, FP8, I4, FP4).  Each dtype_copy_* kernel copies a 1-D
// tensor through a fragment tile (ALLOC / KERNEL_1D / COPY / STORE), which
// drives the dtype-erased raw access path in tile_to_kernel for the quantized
// dtypes (FP8 / I4 / FP4) and the typed DSL-sugar path for the rest.

#include "kernels.h"

// ---- F16 ----------------------------------------------------------------
Tensor<tile_f16, 1> dtype_copy_f16(Tensor<tile_f16, 1> A) {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f16, 1> B = LuisaTensor.empty(LuisaTensor.shape(N), tile_f16{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f16{});
        LuisaTensor.copy(A(0), A_local(N));
        LuisaTensor.copy(A_local(N), B(0));
    }
    return B;
}

// ---- F32 ----------------------------------------------------------------
Tensor<tile_f32, 1> dtype_copy_f32(Tensor<tile_f32, 1> A) {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 1> B = LuisaTensor.empty(LuisaTensor.shape(N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});
        LuisaTensor.copy(A(0), A_local(N));
        LuisaTensor.copy(A_local(N), B(0));
    }
    return B;
}

// ---- I32 ----------------------------------------------------------------
Tensor<tile_i32, 1> dtype_copy_i32(Tensor<tile_i32, 1> A) {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;

    Tensor<tile_i32, 1> B = LuisaTensor.empty(LuisaTensor.shape(N), tile_i32{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_i32{});
        LuisaTensor.copy(A(0), A_local(N));
        LuisaTensor.copy(A_local(N), B(0));
    }
    return B;
}

// ---- I8 -----------------------------------------------------------------
Tensor<tile_i8, 1> dtype_copy_i8(Tensor<tile_i8, 1> A) {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;

    Tensor<tile_i8, 1> B = LuisaTensor.empty(LuisaTensor.shape(N), tile_i8{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_i8{});
        LuisaTensor.copy(A(0), A_local(N));
        LuisaTensor.copy(A_local(N), B(0));
    }
    return B;
}

// ---- FP8 (e4m3) ---------------------------------------------------------
Tensor<tile_fp8, 1> dtype_copy_fp8(Tensor<tile_fp8, 1> A) {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;

    Tensor<tile_fp8, 1> B = LuisaTensor.empty(LuisaTensor.shape(N), tile_fp8{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_fp8{});
        LuisaTensor.copy(A(0), A_local(N));
        LuisaTensor.copy(A_local(N), B(0));
    }
    return B;
}

// ---- I4 (signed 4-bit) --------------------------------------------------
Tensor<tile_i4, 1> dtype_copy_i4(Tensor<tile_i4, 1> A) {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;

    Tensor<tile_i4, 1> B = LuisaTensor.empty(LuisaTensor.shape(N), tile_i4{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_i4{});
        LuisaTensor.copy(A(0), A_local(N));
        LuisaTensor.copy(A_local(N), B(0));
    }
    return B;
}

// ---- FP4 (e2m1) ---------------------------------------------------------
Tensor<tile_fp4, 1> dtype_copy_fp4(Tensor<tile_fp4, 1> A) {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;

    Tensor<tile_fp4, 1> B = LuisaTensor.empty(LuisaTensor.shape(N), tile_fp4{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_fp4{});
        LuisaTensor.copy(A(0), A_local(N));
        LuisaTensor.copy(A_local(N), B(0));
    }
    return B;
}

// ---- BINARY ADD (typed dtypes) ------------------------------------------
Tensor<tile_f32, 1> dtype_add_f32(Tensor<tile_f32, 1> A, Tensor<tile_f32, 1> B) {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 1> C = LuisaTensor.empty(LuisaTensor.shape(N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});
        auto B_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});
        LuisaTensor.copy(A(0), A_local(N));
        LuisaTensor.copy(B(0), B_local(N));
        C(0) = A_local(N) + B_local(N);
    }
    return C;
}

Tensor<tile_i32, 1> dtype_add_i32(Tensor<tile_i32, 1> A, Tensor<tile_i32, 1> B) {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;

    Tensor<tile_i32, 1> C = LuisaTensor.empty(LuisaTensor.shape(N), tile_i32{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_i32{});
        auto B_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_i32{});
        LuisaTensor.copy(A(0), A_local(N));
        LuisaTensor.copy(B(0), B_local(N));
        C(0) = A_local(N) + B_local(N);
    }
    return C;
}

// ---- NEG (divide by -1) for F32 ----------------------------------------
Tensor<tile_f32, 1> dtype_neg_f32(Tensor<tile_f32, 1> A) {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 1> B = LuisaTensor.empty(LuisaTensor.shape(N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});
        LuisaTensor.copy(A(0), A_local(N));
        // B = A / -1  (tile binary DIV with a scalar literal rhs)
        B(0) = A_local(N) / -1.0f;
    }
    return B;
}
