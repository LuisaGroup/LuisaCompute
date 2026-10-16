// Test for CUDA tensor dispatch example (raw CUDA + runtime NVRTC, lc-core only).
// This test covers:
// - CUDA runtime + driver API (cudaSetDevice/cudaMalloc, cuModuleLoadData/cuLaunchKernel)
// - Runtime NVRTC JIT compilation of a raw kernel string (no .cu compilation at build time)
// - A small tensor-dispatch layer: CUDATensor (F16/F32) + CUDATensorRuntime with a
//   lazily-JIT-compiled, per-kernel-name CUfunction cache
// - FP16/FP32 elementwise add and GEMM, including an FP16 tensor-core GEMM (sm_70+)
//
// Hardware note: the FP16 tensor-core path requires sm_70+ (RTX 4060 is sm_89).
// TMA (cp_async_bulk_tensor) and tcgen05 are intentionally NOT used: they need
// sm_90+/sm_100+ and are unavailable on this device.

#include "ut/ut.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <nvrtc.h>

#include <luisa/core/logging.h>

#include <cmath>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

using namespace luisa;
using namespace boost::ut;

namespace {

// Kernel source compiled at runtime by NVRTC. All kernels are plain CUDA C++:
// the build system never invokes nvcc for this test.
static constexpr const char *kKernelSource = R"CUDA(
#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// Tensor-core helpers.
//
// Why inline PTX instead of the nvcuda::wmma API? The wmma API itself works
// fine on sm_89: wmma.load/wmma.mma/wmma.store to GLOBAL memory lower to real
// HMMA.16816 tensor-core instructions (verified by SASS inspection). The trap
// comes from wmma::store_matrix_sync into a LOCAL stack array (the natural way
// to apply alpha/beta before writing C): that emits a wmma.store with
// local-memory space, which ptxas 12.9 cannot lower on sm_89 and silently
// replaces with a BPT.TRAP stub, crashing the kernel at launch (verified;
// even alignas(16) does not help). The m16n16k16 WMMA op is exactly two
// classic mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 instructions,
// which ptxas lowers correctly on sm_70+, so we implement the tensor-core GEMM
// with the classic instruction + manual fragment loads and store straight to
// GLOBAL (same hardware path, no local-memory wmma.store).
// ---------------------------------------------------------------------------

// Load two adjacent halfs (low, high) as one 32-bit register. The address must
// be 4-byte aligned (guaranteed for the even row strides used in the test).
__device__ __forceinline__ unsigned load_half2(const __half *p) {
    return *reinterpret_cast<const unsigned *>(p);
}

// Pack two halfs (lo, hi) into one 32-bit register.
__device__ __forceinline__ unsigned pack_half2(__half lo, __half hi) {
    return static_cast<unsigned>(__half_as_ushort(lo)) |
           (static_cast<unsigned>(__half_as_ushort(hi)) << 16);
}

// Classic tensor-core MMA: D(16x8) = A(16x16) * B(16x8) + C(16x8).
// Fragment layouts (PTX ISA, mma.m16n8k16 f16):
//   A: 4 x b32; a0=A[g  ][t*2+0..1], a1=A[g+8][t*2+0..1],
//               a2=A[g  ][t*2+8..9], a3=A[g+8][t*2+8..9]
//   B (col-major conv.): b0=B[t*2  ][g], b1=B[t*2+1][g],
//                         b2=B[t*2+8][g], b3=B[t*2+9][g]
//   C/D: c0=C[g][t*2+0], c1=C[g][t*2+1], c2=C[g+8][t*2+0], c3=C[g+8][t*2+1]
// where g = lane>>2 and t = lane%4.
__device__ __forceinline__ void mma_m16n8k16(float *d, const unsigned *a,
                                             const unsigned *b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

extern "C" __global__ void elementwise_add_f16(const __half *__restrict__ a,
                                               const __half *__restrict__ b,
                                               __half *__restrict__ out,
                                               int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

extern "C" __global__ void elementwise_add_f32(const float *__restrict__ a,
                                               const float *__restrict__ b,
                                               float *__restrict__ out,
                                               int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

extern "C" __global__ void gemm_f32(const float *__restrict__ A,
                                    const float *__restrict__ B,
                                    float *__restrict__ C,
                                    int M, int N, int K,
                                    float alpha, float beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) {
        return;
    }
    int row = idx / N;
    int col = idx - row * N;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[idx] = alpha * sum + beta * C[idx];
}

extern "C" __global__ void wmma_gemm_f16(const __half *__restrict__ A,
                                         const __half *__restrict__ B,
                                         float *__restrict__ C,
                                         int M, int N, int K,
                                         float alpha, float beta) {
    // One warp computes one 16x16 output tile: two m16n8k16 tensor-core MMAs
    // per k-step (the exact decomposition of a 16x16x16 WMMA operation).
    // A is MxK row-major (ld = K), B is KxN row-major (ld = N), C is MxN
    // row-major with an FP32 accumulator.
    constexpr int TILE = 16;
    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;
    const int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }

    const int g = lane >> 2;     // groupID 0..7
    const int t = lane & 3;      // threadID in group 0..3
    const int m_off = tile_m * TILE;
    const int n_off = tile_n * TILE;

    // Accumulator: two 16x8 halves of the 16x16 tile.
    float c[2][4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            c[h][i] = 0.0f;
        }
    }

    if (m_off < M && n_off < N) {
        for (int k = 0; k < K; k += TILE) {
            if (k + TILE > K) {
                break;
            }
            // Load the 16x16 A fragment (4 x b32).
            const __half *a_base =
                A + (m_off + g) * K + k + t * 2;
            unsigned a[4];
            a[0] = load_half2(a_base);              // A[g  ][t*2..t*2+1]
            a[1] = load_half2(a_base + 8 * K);      // A[g+8][t*2..t*2+1]
            a[2] = load_half2(a_base + 8);          // A[g  ][t*2+8..t*2+9]
            a[3] = load_half2(a_base + 8 * K + 8);  // A[g+8][t*2+8..t*2+9]
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                // Load the 16x8 B fragment (col-major convention, 2 x b32).
                const int col = n_off + h * 8 + g;
                const __half *b_base = B + (k + t * 2) * N + col;
                unsigned b[2];
                b[0] = pack_half2(b_base[0], b_base[N]);       // B[t*2  ][col], B[t*2+1][col]
                b[1] = pack_half2(b_base[8 * N], b_base[9 * N]);  // B[t*2+8][col], B[t*2+9][col]
                mma_m16n8k16(c[h], a, b);
            }
        }
    }

    // Store the two 16x8 accumulator halves with alpha/beta scaling.
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int col0 = n_off + h * 8 + t * 2;
        if (col0 + 1 >= N) {
            break;
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int row = m_off + g + i * 8;
            if (row >= M) {
                break;
            }
            const int idx0 = row * N + col0;
            C[idx0] = alpha * c[h][i * 2 + 0] + beta * C[idx0];
            C[idx0 + 1] = alpha * c[h][i * 2 + 1] + beta * C[idx0 + 1];
        }
    }
}
)CUDA";

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t e = (call);                                                   \
        if (e != cudaSuccess) {                                                   \
            LUISA_ERROR("CUDA error at {}:{}: {} ({})", __FILE__, __LINE__,       \
                        cudaGetErrorString(e), static_cast<int>(e));              \
        }                                                                         \
    } while (0)

#define CUDRV_CHECK(call)                                                         \
    do {                                                                          \
        CUresult e = (call);                                                      \
        if (e != CUDA_SUCCESS) {                                                  \
            const char *s = nullptr;                                              \
            cuGetErrorString(e, &s);                                              \
            LUISA_ERROR("CUDA driver error at {}:{}: {} ({})", __FILE__, __LINE__, \
                        s != nullptr ? s : "unknown", static_cast<int>(e));       \
        }                                                                         \
    } while (0)

// Resolve the CUDA include directory for NVRTC. Prefer the define injected by
// xmake (forward-slash path), then CUDA_PATH, then the well-known 12.9 location.
static luisa::string cuda_include_dir() noexcept {
#ifdef LUISA_TEST_CUDA_PATH
    return luisa::string(LUISA_TEST_CUDA_PATH) + "/include";
#else
    if (const char *p = std::getenv("CUDA_PATH"); p != nullptr && p[0] != '\0') {
        return luisa::string(p) + "/include";
    }
    return "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include";
#endif
}

enum class TensorDType { F16, F32 };

struct CUDATensor {
    TensorDType dtype = TensorDType::F32;
    size_t rows = 0u;
    size_t cols = 0u;
    void *data = nullptr;

    size_t elements() const noexcept { return rows * cols; }
    size_t bytes() const noexcept {
        return elements() * (dtype == TensorDType::F16 ? sizeof(__half) : sizeof(float));
    }
};

class CUDATensorRuntime {
public:
    int major = 0;
    int minor = 0;
    int nvrtc_major = 0;
    int nvrtc_minor = 0;
    std::string name;

    bool init(int device_id = 0) noexcept {
        auto err = cudaSetDevice(device_id);
        if (err != cudaSuccess) {
            LUISA_WARNING("cudaSetDevice({}) failed: {}", device_id, cudaGetErrorString(err));
            return false;
        }
        cudaDeviceProp prop{};
        err = cudaGetDeviceProperties(&prop, device_id);
        if (err != cudaSuccess) {
            LUISA_WARNING("cudaGetDeviceProperties({}) failed: {}", device_id, cudaGetErrorString(err));
            return false;
        }
        major = prop.major;
        minor = prop.minor;
        name = prop.name;
        LUISA_INFO("CUDA tensor dispatch device {}: {} (sm_{}{})",
                   device_id, name, major, minor);
        auto nvrtc_err = nvrtcVersion(&nvrtc_major, &nvrtc_minor);
        if (nvrtc_err != NVRTC_SUCCESS) {
            LUISA_WARNING("nvrtcVersion failed: {}", nvrtcGetErrorString(nvrtc_err));
        } else {
            LUISA_INFO("NVRTC version: {}.{}", nvrtc_major, nvrtc_minor);
        }
        return true;
    }

    void destroy() noexcept {
        _functions.clear();
        if (_module != nullptr) {
            cuModuleUnload(_module);
            _module = nullptr;
        }
        cudaDeviceReset();
    }

    bool wmma_supported() const noexcept { return major >= 7; }

    CUDATensor alloc(TensorDType dtype, size_t rows, size_t cols) {
        CUDATensor t{dtype, rows, cols, nullptr};
        CUDA_CHECK(cudaMalloc(&t.data, t.bytes()));
        return t;
    }

    void upload_f16(CUDATensor &t, const float *host) {
        auto n = t.elements();
        std::vector<__half> h(n);
        for (auto i = 0u; i < n; ++i) {
            h[i] = __float2half(host[i]);
        }
        CUDA_CHECK(cudaMemcpy(t.data, h.data(), t.bytes(), cudaMemcpyHostToDevice));
    }

    void upload_f32(CUDATensor &t, const float *host) {
        CUDA_CHECK(cudaMemcpy(t.data, host, t.bytes(), cudaMemcpyHostToDevice));
    }

    void download(CUDATensor &t, float *host) {
        auto n = t.elements();
        if (t.dtype == TensorDType::F16) {
            std::vector<__half> h(n);
            CUDA_CHECK(cudaMemcpy(h.data(), t.data, t.bytes(), cudaMemcpyDeviceToHost));
            for (auto i = 0u; i < n; ++i) {
                host[i] = __half2float(h[i]);
            }
        } else {
            CUDA_CHECK(cudaMemcpy(host, t.data, t.bytes(), cudaMemcpyDeviceToHost));
        }
    }

    void elementwise_add(const CUDATensor &a, const CUDATensor &b, CUDATensor &out) {
        if (a.dtype != b.dtype || a.dtype != out.dtype || a.elements() != b.elements() || a.elements() != out.elements()) {
            LUISA_ERROR("elementwise_add: tensor shape/dtype mismatch");
        }
        auto n = static_cast<int>(a.elements());
        const char *kernel = (a.dtype == TensorDType::F16) ? "elementwise_add_f16" : "elementwise_add_f32";
        void *a_ptr = a.data;
        void *b_ptr = b.data;
        void *out_ptr = out.data;
        void *args[] = {&a_ptr, &b_ptr, &out_ptr, &n};
        auto grid = static_cast<unsigned>((static_cast<size_t>(n) + 255u) / 256u);
        launch(kernel, dim3(grid), dim3(256u), args);
    }

    void gemm(const CUDATensor &a, const CUDATensor &b, CUDATensor &c, float alpha, float beta) {
        if (a.cols != b.rows || a.rows != c.rows || b.cols != c.cols) {
            LUISA_ERROR("gemm: tensor shape mismatch (A {}x{}, B {}x{}, C {}x{})",
                        a.rows, a.cols, b.rows, b.cols, c.rows, c.cols);
        }
        if (a.dtype == TensorDType::F16) {
            if (!wmma_supported()) {
                LUISA_ERROR("WMMA tensor-core GEMM requires sm_70+ (device is sm_{}{})", major, minor);
            }
            // FP16 GEMM writes an FP32 accumulator (C must be F32).
            auto M = static_cast<int>(a.rows);
            auto N = static_cast<int>(b.cols);
            auto K = static_cast<int>(a.cols);
            void *a_ptr = a.data;
            void *b_ptr = b.data;
            void *c_ptr = c.data;
            void *args[] = {&a_ptr, &b_ptr, &c_ptr, &M, &N, &K, &alpha, &beta};
            dim3 grid{static_cast<unsigned>((static_cast<size_t>(N) + 15u) / 16u),
                      static_cast<unsigned>((static_cast<size_t>(M) + 15u) / 16u)};
            launch("wmma_gemm_f16", grid, dim3(32u), args);
        } else {
            auto M = static_cast<int>(a.rows);
            auto N = static_cast<int>(b.cols);
            auto K = static_cast<int>(a.cols);
            auto total = static_cast<unsigned>(static_cast<size_t>(M) * static_cast<size_t>(N));
            void *a_ptr = a.data;
            void *b_ptr = b.data;
            void *c_ptr = c.data;
            void *args[] = {&a_ptr, &b_ptr, &c_ptr, &M, &N, &K, &alpha, &beta};
            auto grid = (total + 255u) / 256u;
            launch("gemm_f32", dim3(grid), dim3(256u), args);
        }
    }

private:
    CUmodule _module = nullptr;
    std::map<std::string, CUfunction> _functions;

    static CUresult launch_impl(CUfunction f, dim3 grid, dim3 block, void **args) noexcept {
        return cuLaunchKernel(f, grid.x, grid.y, grid.z,
                              block.x, block.y, block.z,
                              0u, nullptr, args, nullptr);
    }

    void launch(const char *name, dim3 grid, dim3 block, void **args) {
        auto f = function(name);
        CUDRV_CHECK(launch_impl(f, grid, block, args));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }

    CUfunction function(const char *name) {
        ensure_compiled();
        auto it = _functions.find(name);
        if (it != _functions.end()) {
            return it->second;
        }
        CUfunction f = nullptr;
        CUDRV_CHECK(cuModuleGetFunction(&f, _module, name));
        _functions.emplace(name, f);
        return f;
    }

    void ensure_compiled() {
        if (_module != nullptr) {
            return;
        }
        nvrtcProgram prog = nullptr;
        auto result = nvrtcCreateProgram(&prog, kKernelSource, "cuda_tensor_dispatch.cu",
                                         0, nullptr, nullptr);
        if (result != NVRTC_SUCCESS) {
            LUISA_ERROR("nvrtcCreateProgram failed: {}", nvrtcGetErrorString(result));
        }
        auto arch = major > 0 ? luisa::format("-arch=compute_{}{}", major, minor)
                              : luisa::string("-arch=compute_89");
        auto include_path = luisa::format("--include-path={}", cuda_include_dir());
        std::vector<luisa::string> option_storage{
            arch,
            "--std=c++17",
            "-default-device",
            "-restrict",
            "-w",
            include_path,
        };
        std::vector<const char *> options;
        options.reserve(option_storage.size());
        for (auto &o : option_storage) {
            options.emplace_back(o.c_str());
        }
        LUISA_INFO("JIT-compiling tensor dispatch kernels (arch={})", arch);
        result = nvrtcCompileProgram(prog, static_cast<int>(options.size()), options.data());
        if (result != NVRTC_SUCCESS) {
            size_t log_size = 0u;
            nvrtcGetProgramLogSize(prog, &log_size);
            luisa::string log;
            if (log_size > 1u) {
                log.resize(log_size);
                nvrtcGetProgramLog(prog, log.data());
            }
            LUISA_ERROR("NVRTC compile failed ({}): {}", nvrtcGetErrorString(result), log);
        }
        size_t ptx_size = 0u;
        nvrtcGetPTXSize(prog, &ptx_size);
        luisa::string ptx(ptx_size, '\0');
        nvrtcGetPTX(prog, ptx.data());
        nvrtcDestroyProgram(&prog);
        CUDRV_CHECK(cuModuleLoadData(&_module, ptx.data()));
    }
};

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

bool test_elementwise_add(CUDATensorRuntime &rt) {
    constexpr auto N = 1024u;
    LUISA_INFO("Running elementwise_add test (N={})", N);
    std::vector<float> a(N), b(N), got(N), expected(N);
    for (auto i = 0u; i < N; ++i) {
        a[i] = static_cast<float>(i);        // 0..1023, exactly representable in fp16
        b[i] = static_cast<float>(i % 17u);  // small non-negative values
    }

    // F32 path.
    auto da = rt.alloc(TensorDType::F32, 1u, N);
    auto db = rt.alloc(TensorDType::F32, 1u, N);
    auto dout = rt.alloc(TensorDType::F32, 1u, N);
    rt.upload_f32(da, a.data());
    rt.upload_f32(db, b.data());
    rt.elementwise_add(da, db, dout);
    rt.download(dout, got.data());
    for (auto i = 0u; i < N; ++i) {
        expected[i] = a[i] + b[i];
    }
    bool ok = true;
    for (auto i = 0u; i < N; ++i) {
        if (std::abs(got[i] - expected[i]) > 1e-4f) {
            LUISA_WARNING("F32 elementwise_add mismatch at [{}]: got {}, expected {}",
                          i, got[i], expected[i]);
            ok = false;
            break;
        }
    }
    expect(ok) << "F32 elementwise add matches CPU reference";
    LUISA_INFO("F32 elementwise_add: {}", ok ? "OK" : "FAILED");

    // F16 path: fp16 addition of integers <= 2048 is exact, so a loose tolerance
    // (0.1) still catches layout/conversion bugs without fp16 rounding noise.
    auto ha = rt.alloc(TensorDType::F16, 1u, N);
    auto hb = rt.alloc(TensorDType::F16, 1u, N);
    auto hout = rt.alloc(TensorDType::F16, 1u, N);
    rt.upload_f16(ha, a.data());
    rt.upload_f16(hb, b.data());
    rt.elementwise_add(ha, hb, hout);
    rt.download(hout, got.data());
    for (auto i = 0u; i < N; ++i) {
        expected[i] = __half2float(__float2half(a[i] + b[i]));
    }
    ok = true;
    for (auto i = 0u; i < N; ++i) {
        if (std::abs(got[i] - expected[i]) > 0.1f) {
            LUISA_WARNING("F16 elementwise_add mismatch at [{}]: got {}, expected {}",
                          i, got[i], expected[i]);
            ok = false;
            break;
        }
    }
    expect(ok) << "F16 elementwise add matches CPU reference";
    LUISA_INFO("F16 elementwise_add: {}", ok ? "OK" : "FAILED");
    LUISA_INFO("elementwise_add test done");
    return ok;
}

bool test_gemm_f32(CUDATensorRuntime &rt) {
    constexpr auto M = 128u, N = 128u, K = 128u;
    constexpr float alpha = 1.25f;
    constexpr float beta = 0.5f;
    LUISA_INFO("Running gemm_f32 test ({}x{}x{})", M, N, K);
    std::vector<float> A(M * K), B(K * N), C0(M * N), ref(M * N), got(M * N);
    // Quarter-integer values: exact in fp32 and the products/sums stay exact in
    // fp32 for these sizes, so the reference comparison is robust.
    for (auto i = 0u; i < A.size(); ++i) {
        A[i] = static_cast<float>((i * 7u) % 9u) * 0.25f;
    }
    for (auto i = 0u; i < B.size(); ++i) {
        B[i] = static_cast<float>((i * 5u) % 9u) * 0.25f;
    }
    for (auto i = 0u; i < C0.size(); ++i) {
        C0[i] = static_cast<float>((i % 5u)) * 0.5f;
    }
    // CPU reference in double precision.
    for (auto m = 0u; m < M; ++m) {
        for (auto n = 0u; n < N; ++n) {
            double sum = 0.0;
            for (auto k = 0u; k < K; ++k) {
                sum += static_cast<double>(A[m * K + k]) * static_cast<double>(B[k * N + n]);
            }
            ref[m * N + n] = static_cast<float>(alpha * sum + beta * static_cast<double>(C0[m * N + n]));
        }
    }
    auto da = rt.alloc(TensorDType::F32, M, K);
    auto db = rt.alloc(TensorDType::F32, K, N);
    auto dc = rt.alloc(TensorDType::F32, M, N);
    rt.upload_f32(da, A.data());
    rt.upload_f32(db, B.data());
    rt.upload_f32(dc, C0.data());
    rt.gemm(da, db, dc, alpha, beta);
    rt.download(dc, got.data());
    bool ok = true;
    for (auto i = 0u; i < ref.size(); ++i) {
        if (std::abs(got[i] - ref[i]) > 1e-3f) {
            LUISA_WARNING("gemm_f32 mismatch at [{}]: got {}, expected {}",
                          i, got[i], ref[i]);
            ok = false;
            break;
        }
    }
    expect(ok) << "F32 GEMM matches CPU reference";
    LUISA_INFO("gemm_f32 test done: {}", ok ? "OK" : "FAILED");
    return ok;
}

bool test_gemm_f16_wmma(CUDATensorRuntime &rt) {
    constexpr auto M = 128u, N = 128u, K = 128u;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    if (!rt.wmma_supported()) {
        LUISA_WARNING("WMMA requires sm_70+ (device is sm_{}{}); "
                      "skipping tensor-core GEMM (hardware limitation)",
                      rt.major, rt.minor);
        return true;
    }
    LUISA_INFO("Running gemm_f16_wmma test ({}x{}x{}, tensor cores)", M, N, K);
    std::vector<float> A(M * K), B(K * N), C0(M * N, 0.0f), ref(M * N), got(M * N);
    // Small integers 0..7: exactly representable in fp16, so fp16 rounding is
    // negligible and the fp32 tensor-core accumulation is exact for these sizes.
    for (auto i = 0u; i < A.size(); ++i) {
        A[i] = static_cast<float>(i % 8u);
    }
    for (auto i = 0u; i < B.size(); ++i) {
        B[i] = static_cast<float>((i * 3u) % 8u);
    }
    // CPU reference in double precision, then cast to float.
    for (auto m = 0u; m < M; ++m) {
        for (auto n = 0u; n < N; ++n) {
            double sum = 0.0;
            for (auto k = 0u; k < K; ++k) {
                sum += static_cast<double>(A[m * K + k]) * static_cast<double>(B[k * N + n]);
            }
            ref[m * N + n] = static_cast<float>(alpha * sum + beta * static_cast<double>(C0[m * N + n]));
        }
    }
    auto da = rt.alloc(TensorDType::F16, M, K);
    auto db = rt.alloc(TensorDType::F16, K, N);
    auto dc = rt.alloc(TensorDType::F32, M, N);  // FP32 accumulator for the F16 GEMM
    rt.upload_f16(da, A.data());
    rt.upload_f16(db, B.data());
    rt.upload_f32(dc, C0.data());
    rt.gemm(da, db, dc, alpha, beta);
    rt.download(dc, got.data());
    bool ok = true;
    for (auto i = 0u; i < ref.size(); ++i) {
        if (std::abs(got[i] - ref[i]) > 0.05f) {
            LUISA_WARNING("gemm_f16_wmma mismatch at [{}]: got {}, expected {}",
                          i, got[i], ref[i]);
            ok = false;
            break;
        }
    }
    expect(ok) << "F16 WMMA GEMM matches CPU reference";
    LUISA_INFO("gemm_f16_wmma test done: {}", ok ? "OK" : "FAILED");
    return ok;
}

}  // namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    CUDATensorRuntime rt;
    if (!rt.init(0)) {
        LUISA_ERROR("Failed to initialize CUDA tensor dispatch runtime (CUDA unavailable?)");
        return 1;  // unreachable: LUISA_ERROR aborts
    }

    bool all_ok = true;
    all_ok &= test_elementwise_add(rt);
    all_ok &= test_gemm_f32(rt);
    all_ok &= test_gemm_f16_wmma(rt);

    rt.destroy();
    LUISA_INFO("CUDA tensor dispatch test {}", all_ok ? "PASSED" : "FAILED");
    return all_ok ? 0 : 1;
}
