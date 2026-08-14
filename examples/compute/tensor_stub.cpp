// =============================================================================
// tensor_stub.cpp — TileLang-style tile / tensor DSL demo (stub)
// =============================================================================
// A *compilable* adaptation of the pseudo-code in
// `D:/tilelang/dsl_report/tilelang_cpp_tile_style.cpp`, built on the
// header-only stub `include/luisa/dsl/tensor.h`.
//
// The three kernels are written in pure tile style — no threads, no
// `set_block_size`, no `dispatch`, no shared-memory loops, no barriers:
//   * elementwise_add  -> mirrors examples/elementwise/example_elementwise_add.py
//   * matmul           -> mirrors examples/gemm/example_gemm.py (T.Pipelined)
//   * rms_norm         -> mirrors examples/norm/rms_norm.py
//
// Pseudo-code -> valid C++ adaptations (the stub's job):
//   T.empty({M, N}, f32)                  -> LuisaTensor.empty(LuisaTensor.shape(M, N), tile_f32{})
//   T.alloc_shared({BM, BK}, f16)         -> LuisaTensor.alloc_shared(LuisaTensor.shape(BM, BK), tile_f16{})
//   T.alloc_fragment({blk_m}, f32)        -> LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m), tile_f32{})
//   A[by * BM, bx * BN]                   -> A(by * BM, bx * BN)
//                                        (multi-arg operator[] is C++23-only,
//                                         so tile indexing uses operator())
//   A[bx*blk_m : (bx+1)*blk_m, :]         -> A(LuisaTensor.range(bx*blk_m, (bx+1)*blk_m), LuisaTensor.all())
//   f32 / f16 as a value argument         -> tile_f32{} / tile_f16{} (dtype handle)
//   f32(N)                                -> stays (dtype handles are scalars)
//   luisa::compute::tile::jit(matmul).compile()        -> stays; logs "kernel.compile"
//   (no shape/tile args: M, N, K, block_*, threads, num_stages are baked into
//   the matmul function itself, exactly like tilelang keeps config in the
//   @tilelang.jit function rather than in compile())
//
// The stub logs every tile op through the LuisaCompute core logger
// (lc_core / LUISA_INFO).  Running this target must print `kernel.compile`.
// =============================================================================

#include <luisa/dsl/tensor.h>
#include <luisa/ast/tile_to_kernel.h>
#include <luisa/dsl/func.h>
#include <luisa/core/mathematics.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <string>

// TileLang's `import tilelang.language as T` is exposed as the `LuisaTensor`
// constexpr handle (a C++ namespace can only be addressed with `::`, so the
// `LuisaTensor.*` dot syntax comes from the `dsl` handle object).
constexpr auto LuisaTensor = luisa::compute::tile::language::dsl;
using luisa::compute::tile::language::Tensor;

using tile_f16 = luisa::compute::tile::half;
using tile_f32 = luisa::compute::tile::float32;
using tile_i32 = luisa::compute::tile::int32;

// =============================================================================
// 1. Elementwise add — tiles only
// =============================================================================
TILELANG_PRIM_FUNC
Tensor<tile_f32, 2> elementwise_add(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> B) {
    constexpr tile_i32 M = 512, N = 512;
    constexpr tile_i32 block_M = 64, block_N = 64;
    constexpr tile_i32 threads = 256;

    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(M, N), tile_f32{});

    // Grid is in *blocks*: LuisaTensor.Kernel(gx, gy) means gx*gy blocks; the
    // range-for binds (bx, by) to each block id (the C++ spelling of
    // `with T.Kernel(...) as (bx, by)`).
    for (auto [bx, by] : LuisaTensor.Kernel(LuisaTensor.ceildiv(N, block_N), LuisaTensor.ceildiv(M, block_M), threads)) {
        // Per-block on-chip staging: global -> shared -> fragment -> global.
        auto A_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_M, block_N), tile_f32{});
        auto B_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_M, block_N), tile_f32{});
        auto C_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(block_M, block_N), tile_f32{});

        // Tile copies (global -> shared); a real lowering emits a coalesced
        // SIMT/TMA copy.
        LuisaTensor.copy(A(by * block_M, bx * block_N), A_shared);
        LuisaTensor.copy(B(by * block_M, bx * block_N), B_shared);

        // Whole-tile elementwise op (block style): the (block_M x block_N)
        // tile of C_local becomes the elementwise sum of the two source tiles.
        C_local(block_M, block_N) = A_shared(block_M, block_N) + B_shared(block_M, block_N);

        // Fragment -> global.
        LuisaTensor.copy(C_local, C(by * block_M, bx * block_N));
    }
    return C;
}

// =============================================================================
// 2. Tiled GEMM with a software pipeline — tiles only
// =============================================================================
TILELANG_PRIM_FUNC
Tensor<tile_f16, 2> matmul(Tensor<tile_f16, 2> A, Tensor<tile_f16, 2> B) {
    constexpr tile_i32 M = 1024, N = 1024, K = 1024;
    constexpr tile_i32 block_M = 128, block_N = 128, block_K = 32;
    constexpr tile_i32 threads = 128;
    constexpr tile_i32 num_stages = 3;

    Tensor<tile_f16, 2> C = LuisaTensor.empty(LuisaTensor.shape(M, N), tile_f16{});

    for (auto [bx, by] : LuisaTensor.Kernel(LuisaTensor.ceildiv(N, block_N), LuisaTensor.ceildiv(M, block_M), threads)) {
        auto A_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_M, block_K), tile_f16{});
        auto B_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_K, block_N), tile_f16{});
        auto C_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(block_M, block_N), tile_f32{});// tile_f32 accumulator

        LuisaTensor.clear(C_local);// T.clear

        // Software pipeline: copies and GEMMs are overlapped by the compiler.
        for (auto ko : LuisaTensor.Pipelined(LuisaTensor.ceildiv(K, block_K), num_stages)) {
            LuisaTensor.copy(A(by * block_M, ko * block_K), A_shared);// global -> shared
            LuisaTensor.copy(B(ko * block_K, bx * block_N), B_shared);
            LuisaTensor.gemm(A_shared, B_shared, C_local);// tile GEMM
        }

        // Fused ReLU on the fragment tile (whole-tile op, quickstart.py).
        C_local(block_M, block_N) = LuisaTensor.max(C_local(block_M, block_N), tile_f32(0.0f));

        // Optional device-side debug (TileLang prints from a single thread).
        LuisaTensor.print(C_local, /*msg=*/"C tile:");

        LuisaTensor.copy(C_local, C(by * block_M, bx * block_N));// fragment -> global
    }
    return C;
}

// =============================================================================
// 3. RMSNorm — tiles only
// =============================================================================
Tensor<tile_f32, 2> rms_norm(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 512, N = 512;
    constexpr tile_i32 blk_m = 8;
    constexpr tile_i32 threads = 128;

    Tensor<tile_f32, 2> B = LuisaTensor.empty(LuisaTensor.shape(M, N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(LuisaTensor.ceildiv(M, blk_m), threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m, N), tile_f32{});
        auto A_pow_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m, N), tile_f32{});
        auto A_powsum = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m), tile_f32{});// per-row scalars

        // Row-slice copy: pseudo `A[bx*blk_m : (bx+1)*blk_m, :]`.
        LuisaTensor.copy(A(LuisaTensor.range(bx * blk_m, (bx + 1) * blk_m), LuisaTensor.all()), A_local);

        // Whole-tile square: the (blk_m x N) tile, elementwise.
        A_pow_local(blk_m, N) = A_local(blk_m, N) * A_local(blk_m, N);

        LuisaTensor.reduce_sum(A_pow_local, A_powsum, /*dim=*/1);// row sums

        // Whole-tile per-row scale factor (1-D tile, scalar broadcast).
        A_powsum(blk_m) = LuisaTensor.rsqrt(A_powsum(blk_m) / tile_f32(N) + 1e-12f);

        // Broadcast: every column of row i is scaled by the scalar A_powsum[i].
        A_local(blk_m, N) *= A_powsum(blk_m);

        LuisaTensor.copy(A_local, B(LuisaTensor.range(bx * blk_m, (bx + 1) * blk_m), LuisaTensor.all()));// store row slice
    }
    return B;
}

// =============================================================================
// 4. Multiple-T.Kernel guard — an INVALID tile function (for the trigger at
//    the end of main).  A tile function maps to exactly ONE kernel launch
//    (TileLang emits one `__global__` per `T.Kernel`), so tracing a second
//    T.Kernel must be rejected: jit(...).compile() derives the SIMT launch
//    metadata (TileFunctionBuilder::compile_meta_data) and logs an error +
//    aborts when the body contains more than one T.Kernel.
// =============================================================================
TILELANG_PRIM_FUNC
Tensor<tile_f32, 2> two_kernels(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> B) {
    // First launch: 4x4 blocks of 32 threads.
    for (auto [bx, by] : LuisaTensor.Kernel(4, 4, 32)) {
        LuisaTensor.copy(A(0, 0), B(0, 0));
    }
    // Second launch inside the same function: forbidden (two different
    // block/grid shapes cannot be expressed by a single Shader).
    for (auto [bx, by] : LuisaTensor.Kernel(8, 8, 64)) {
        LuisaTensor.copy(A(0, 0), B(0, 0));
    }
    return B;
}

// CPU reference used only by the host harness below (stub: no data).
Tensor<tile_f16, 2> reference_matmul(const Tensor<tile_f16, 2> &A, const Tensor<tile_f16, 2> &B) {
    LUISA_INFO("[tensor-stub] reference_matmul: stub CPU reference (no data)");
    return Tensor<tile_f16, 2>{};
}

int main(int argc, char *argv[]) {
    auto executable = argc > 0 && argv != nullptr && argv[0] != nullptr ? argv[0] : "";
    // Same baked sizes as matmul() above; the @jit wrapper infers the target
    // from the tensors on the first call (luisa::compute::tile::jit == @tilelang.jit).
    constexpr tile_i32 M = 1024, N = 1024, K = 1024;

    Tensor<tile_f16, 2> A{M, K};
    Tensor<tile_f16, 2> B{K, N};

    // ---- Trace the tile-style prim functions (host side, stub) -------------
    {
        Tensor<tile_f32, 2> A_f{512, 512};
        Tensor<tile_f32, 2> B_f{512, 512};

        LUISA_INFO("=== tensor-dsl: trace elementwise_add ===");
        auto C_add = elementwise_add(A_f, B_f);
        LUISA_INFO("[tensor-stub] elementwise_add -> {}", C_add.describe());

        LUISA_INFO("=== tensor-dsl: trace rms_norm ===");
        auto C_norm = rms_norm(A_f);
        LUISA_INFO("[tensor-stub] rms_norm -> {}", C_norm.describe());
    }

    // ---- Pseudo kernel: trace a prim function into a TileFunctionBuilder ----
    // luisa::compute::tile::Kernel is the tile-DSL analogue of the Kernel in
    // <luisa/dsl/func.h>: constructing it executes the lambda / prim function
    // once on the host and records every tile op (T.empty, T.alloc_shared,
    // T.copy, tile-store, ...) as a TensorStmt in a
    // luisa::compute::detail::TileFunctionBuilder.
    LUISA_INFO("=== tensor-dsl: tile::Kernel{elementwise_add} ===");
    luisa::compute::tile::Kernel elementwise_kernel{elementwise_add};
    auto elementwise_ir = elementwise_kernel.function();// shared_ptr<const TileFunctionBuilder>
    LUISA_INFO("[tensor-stub] elementwise_add traced {} statements: [{}]",
               elementwise_ir->body()->size(), elementwise_kernel.describe());

    // ---- Compile the matmul TIR function into an executable module ---------
    // This logs "kernel.compile" (required output of this example).  Like
    // tilelang, compile() carries NO shape/tile parameters (M, N, K, block_M,
    // block_N, block_K, threads, num_stages) — those are baked into the matmul
    // kernel function above, so `jit(matmul).compile()` just traces it into a
    // TileFunctionBuilder, which the compiled kernel keeps for introspection.
    auto matmul_kernel = luisa::compute::tile::jit(matmul).compile();
    LUISA_INFO("[tensor-stub] matmul traced {} statements: [{}]",
               matmul_kernel.function()->body()->size(), matmul_kernel.describe());

    Tensor<tile_f16, 2> C = matmul_kernel(A, B);// runs on the GPU (stub)

    // Reference & correctness check (Torch here; assert_close in C++).
    auto C_ref = reference_matmul(A, B);
    luisa::compute::tile::testing::assert_close(C, C_ref, /*rtol=*/1e-2f, /*atol=*/1e-2f);

    // Optional: dump the generated CUDA source, no threads visible to us.
    auto cuda_source = matmul_kernel.get_kernel_source();
    luisa::compute::tile::print(cuda_source);

    using namespace luisa::compute;// Kernel / Device / Context / detail for the translation test
    // ---- tile_to_kernel: translate the traced tile kernels into regular ----
    // Luisa FunctionBuilder kernels (include/luisa/ast/tile_to_kernel.h).
    // The translation is verified structurally (dispatch grid, block size and
    // the per-Global-tensor buffer argument list) and, when a backend name is
    // passed on the command line, by actually compiling the generated kernel
    // on that backend.
    auto same_u3 = [](luisa::uint3 a, luisa::uint3 b) noexcept {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    };
    auto translate_and_verify = [&](luisa::string_view name,
                                    luisa::shared_ptr<const luisa::compute::detail::TileFunctionBuilder> const &tile_fn,
                                    luisa::uint3 expected_dispatch, luisa::uint3 expected_block,
                                    size_t expected_buffers) -> TileCompileResult {
        LUISA_INFO("=== tensor-dsl: tile_to_kernel({}) ===", name);
        // The traced builder is const; the lowering only reads it.
        auto result = tile_to_kernel(tile_fn);
        LUISA_ASSERT(result.function != nullptr,
                     "[tensor-stub] tile_to_kernel({}) produced a null FunctionBuilder.", name);
        LUISA_ASSERT(same_u3(result.dispatch_size, expected_dispatch),
                     "[tensor-stub] tile_to_kernel({}) dispatch mismatch: got ({},{},{}), want ({},{},{}).",
                     name, result.dispatch_size.x, result.dispatch_size.y, result.dispatch_size.z,
                     expected_dispatch.x, expected_dispatch.y, expected_dispatch.z);
        auto block = result.function->block_size();
        LUISA_ASSERT(same_u3(block, expected_block),
                     "[tensor-stub] tile_to_kernel({}) block-size mismatch: got ({},{},{}), want ({},{},{}).",
                     name, block.x, block.y, block.z, expected_block.x, expected_block.y, expected_block.z);
        auto arg_count = result.function->arguments().size();
        LUISA_ASSERT(arg_count == expected_buffers,
                     "[tensor-stub] tile_to_kernel({}) buffer-argument count mismatch: got {}, want {}.",
                     name, arg_count, expected_buffers);
        LUISA_INFO("[tensor-stub] tile_to_kernel({}) -> FunctionBuilder dispatch=({},{},{}), "
                   "block=({},{},{}), {} buffer argument(s), body has {} statement(s).",
                   name, result.dispatch_size.x, result.dispatch_size.y, result.dispatch_size.z,
                   block.x, block.y, block.z, arg_count,
                   result.function->body()->statements().size());
        return result;
    };

    // elementwise_add: T.Kernel(ceildiv(512,64)=8, ceildiv(512,64)=8, 256);
    // globals A, B, C -> 3 buffer arguments.
    // T.Kernel(8, 8, 256) -> 8x8 blocks x 256 threads -> dispatch (8*256, 8, 1).
    auto elementwise_result = translate_and_verify(
        "elementwise_add", elementwise_kernel.function(),
        luisa::uint3{2048u, 8u, 1u}, luisa::uint3{256u, 1u, 1u}, 3u);

    // matmul (pipelined): T.Kernel(ceildiv(1024,128)=8, ceildiv(1024,128)=8, 128).
    auto matmul_result = translate_and_verify(
        "matmul", matmul_kernel.function(),
        luisa::uint3{1024u, 8u, 1u}, luisa::uint3{128u, 1u, 1u}, 3u);

    // rms_norm: T.Kernel(ceildiv(512,8)=64, 128); globals A, B -> 2 buffers.
    luisa::compute::tile::Kernel rms_kernel{rms_norm};
    auto rms_result = translate_and_verify(
        "rms_norm", rms_kernel.function(),
        luisa::uint3{8192u, 1u, 1u}, luisa::uint3{128u, 1u, 1u}, 2u);

    // Optional: compile the translated kernels on a real backend to prove the
    // generated FunctionBuilder is a valid, compilable Luisa kernel.
  if (argc > 1 && argv != nullptr && argv[1] != nullptr && argv[1][0] != '\0') {
      LUISA_INFO("=== tensor-dsl: compile translated kernels on backend '{}' ===", argv[1]);
      Context ctx(executable);
      Device device = ctx.create_device(argv[1]);
      // Kernel<N> takes a shared_ptr<const FunctionBuilder>; the translation
      // returns a mutable builder, so wrap it in a const shared_ptr first.
      auto compile1 = [&](luisa::shared_ptr<luisa::compute::detail::FunctionBuilder> fb, const char *name) {
          luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder> const_fb{std::move(fb)};
          auto shader = device.compile(luisa::compute::Kernel<1>{const_fb});
          LUISA_INFO("[tensor-stub] {} kernel compiled (1D, block {}x{}x{}).",
                     name, const_fb->block_size().x, const_fb->block_size().y,
                     const_fb->block_size().z);
      };
      auto compile2 = [&](luisa::shared_ptr<luisa::compute::detail::FunctionBuilder> fb, const char *name) {
          luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder> const_fb{std::move(fb)};
          auto shader = device.compile(luisa::compute::Kernel<2>{const_fb});
          LUISA_INFO("[tensor-stub] {} kernel compiled (2D, block {}x{}x{}).",
                     name, const_fb->block_size().x, const_fb->block_size().y,
                     const_fb->block_size().z);
      };
      compile2(elementwise_result.function, "elementwise_add");
      compile2(matmul_result.function, "matmul");
      compile1(rms_result.function, "rms_norm");
      LUISA_INFO("[tensor-stub] all three translated kernels compiled successfully.");

      // ---- runtime numerical checks of the translated kernels ----------------
      // Dispatch the generated kernels on real buffers and compare against a
      // host reference.  This proves the SIMD->SIMT translation is not only
      // compilable but semantically correct.
      auto stream = device.create_stream();
      {
          constexpr uint32_t M = 512u, N = 512u;// same as elementwise_add/rms_norm
          auto bufA = device.create_buffer<float>(M * N);
          auto bufB = device.create_buffer<float>(M * N);
          auto bufC = device.create_buffer<float>(M * N);
          luisa::vector<float> hA(M * N), hB(M * N), hC(M * N);
          for (auto i = 0u; i < M * N; ++i) {
              hA[i] = static_cast<float>(i) * 0.5f;
              hB[i] = static_cast<float>(i) * 1.5f + 1.0f;
          }
          stream << bufA.copy_from(luisa::span{hA}) << bufB.copy_from(luisa::span{hB}) << synchronize();

          // elementwise_add: C = A + B (2D grid 8x8, tiles 64x64)
          luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder> elem_fb{elementwise_result.function};
          auto sh_elem = device.compile(luisa::compute::Kernel<2, Buffer<float>, Buffer<float>, Buffer<float>>{elem_fb});
          stream << sh_elem(bufA, bufB, bufC).dispatch(elementwise_result.dispatch_size.x, elementwise_result.dispatch_size.y) << bufC.copy_to(luisa::span{hC}) << synchronize();
          auto max_err = 0.0f;
          for (auto i = 0u; i < M * N; ++i) {
              max_err = luisa::max(max_err, luisa::abs(hC[i] - (hA[i] + hB[i])));
          }
          LUISA_INFO("[tensor-stub] elementwise_add runtime check: max |C-(A+B)| = {}", max_err);
          LUISA_ASSERT(max_err < 1e-3f, "elementwise_add translation produced wrong results!");

          // rms_norm: B[r][c] = A[r][c] * rsqrt(sum_c A[r][c]^2 / N + 1e-12)
          luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder> rms_fb{rms_result.function};
          auto sh_rms = device.compile(luisa::compute::Kernel<1, Buffer<float>, Buffer<float>>{rms_fb});
          stream << sh_rms(bufA, bufB).dispatch(rms_result.dispatch_size.x) << bufB.copy_to(luisa::span{hB}) << synchronize();
          max_err = 0.0f;
          for (auto r = 0u; r < M; ++r) {
              auto s = 0.0f;
              for (auto c = 0u; c < N; ++c) { s += hA[r * N + c] * hA[r * N + c]; }
              auto scale = 1.0f / luisa::sqrt(s / static_cast<float>(N) + 1e-12f);
              for (auto c = 0u; c < N; ++c) {
                  max_err = luisa::max(max_err, luisa::abs(hB[r * N + c] - hA[r * N + c] * scale));
              }
          }
          LUISA_INFO("[tensor-stub] rms_norm runtime check: max error = {}", max_err);
          LUISA_ASSERT(max_err < 1e-3f, "rms_norm translation produced wrong results!");
          LUISA_INFO("[tensor-stub] translated kernels produce correct results on the device.");
      }
  } else {
        LUISA_INFO("[tensor-stub] no backend given: translation verified structurally only "
                   "(pass a backend name, e.g. 'cuda'/'dx'/'vk', to also compile).");
    }

    // ---- Trigger the multiple-T.Kernel guard (invalid tile function) --------
    // A tile function maps to exactly ONE kernel launch; tracing a second
    // T.Kernel must fail.  compile() validates the traced body and logs an
    // error + aborts (LUISA_ERROR_WITH_LOCATION -> std::abort), so this line
    // intentionally terminates the example after the normal flow above.
    LUISA_INFO("=== tensor-dsl: trigger the multiple-T.Kernel guard ===");
    auto invalid_kernel = luisa::compute::tile::jit(two_kernels).compile();// aborts here
    (void)invalid_kernel;

    LUISA_INFO("[tensor-stub] finished: tensor DSL stub traced and kernel.compile done.");
    return 0;
}
