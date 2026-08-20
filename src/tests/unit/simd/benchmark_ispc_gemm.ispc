// ISPC control for benchmark_simd_gemm.cpp.
//
// Keep the algorithm and storage layout identical to the Luisa DSL kernel:
// lanes cover consecutive output columns, B remains row-major (and therefore
// strided in the inner loop), and one invocation covers one or more rows.

#ifndef LUISA_ISPC_GEMM_VARIANT
#error LUISA_ISPC_GEMM_VARIANT must select an exported symbol
#endif

#if LUISA_ISPC_GEMM_VARIANT == 1
#define LUISA_ISPC_GEMM_FUNCTION luisa_ispc_gemm_avx2_w4
#elif LUISA_ISPC_GEMM_VARIANT == 2
#define LUISA_ISPC_GEMM_FUNCTION luisa_ispc_gemm_avx2_w8
#elif LUISA_ISPC_GEMM_VARIANT == 3
#define LUISA_ISPC_GEMM_FUNCTION luisa_ispc_gemm_avx512_w4
#elif LUISA_ISPC_GEMM_VARIANT == 4
#define LUISA_ISPC_GEMM_FUNCTION luisa_ispc_gemm_avx512_w8
#elif LUISA_ISPC_GEMM_VARIANT == 5
#define LUISA_ISPC_GEMM_FUNCTION luisa_ispc_gemm_avx512_w16
#else
#error Invalid LUISA_ISPC_GEMM_VARIANT
#endif

static uniform uint32 matrix_size = 256u;

export void LUISA_ISPC_GEMM_FUNCTION(
    uniform const float lhs[], uniform const float rhs[],
    uniform float output[], uniform uint32 row_begin,
    uniform uint32 row_end) {
    foreach (row = row_begin ... row_end, column = 0 ... matrix_size) {
        float sum = 0.0f;
        for (uniform uint32 inner = 0u; inner < matrix_size; inner++) {
            sum += lhs[row * matrix_size + inner] *
                   rhs[inner * matrix_size + column];
        }
        output[row * matrix_size + column] = sum;
    }
}
