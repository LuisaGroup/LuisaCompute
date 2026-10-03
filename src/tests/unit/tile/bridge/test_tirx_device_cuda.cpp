// Host-side CUDA device-artifact tests for the shared TIRx bridge. These do
// not need a CUDA GPU or TVMx CUDA runtime: compile_device() stops at the
// CUDA C / NVPTX source artifact, which the CUDA backend later feeds to the
// standalone-NVRTC pipeline. Requires the pinned TVMx build to provide the
// "cuda" (CUDA C) and optionally "nvptx" code generators.

#include "ut/ut.hpp"
#include "../../../../tile/bridge/tirx/execution.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/arith/analyzer.h>

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <luisa/core/stl/string.h>
#include <luisa/core/platform.h>
#include <luisa/core/logging.h>
#include <luisa/core/mathematics.h>
#include <luisa/tile/bridge/tirx/compiler.h>
#include <luisa/tile/bridge/tirx/lower.h>
#include <luisa/tile/dsl.h>
#include <luisa/tile/algorithms.h>

#include <array>
#include <cstdint>
#include <string_view>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

using namespace luisa;
using namespace luisa::compute::tile;
using namespace luisa::compute::tile::bridge::tirx;
using namespace boost::ut;

namespace {

[[nodiscard]] constexpr luisa::string_view cuda_target() noexcept {
    return R"({"kind":"cuda","thread_warp_size":32,"max_num_threads":1024,"max_shared_memory_per_block":98304})";
}

void test_cuda_elementwise_artifact() {
    constexpr int64_t n = 128;
    auto definition = tile_kernel("cuda_tirx_elementwise", [](TensorView<const float, 1> input,
                                                              TensorView<float, 1> output) {
        auto element = axis("element", n);
        for (auto &nest : parallel(shape(element))) {
            auto index = nest.index();
            output(index).store(input(index).load() * 1.5f + 0.25f);
        }
    });
    auto kernel = definition.capture(tensor_shape(n), tensor_shape(n));
    expect(kernel.valid());
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    CompileOptions options;
    options.target = cuda_target();
    options.noalias = true;
    auto result = compile_device(native.value, kernel.function().name(), options);
    expect(static_cast<bool>(result)) << result.error;
    if (!result) { return; }
    auto &artifact = result.artifact;
    expect(artifact.format == DeviceArtifact::Format::CUDA_SOURCE);
    expect(!artifact.entry.empty());
    expect(!artifact.source.empty());
    expect(artifact.requires_metal4 == false);
    expect(artifact.grid[0] > 0u && artifact.grid[1] == 1u && artifact.grid[2] == 1u);
    auto threads = static_cast<uint64_t>(artifact.block[0]) * artifact.block[1] * artifact.block[2];
    expect(threads > 0u && threads <= 1024u && threads % 32u == 0u);
    expect(!artifact.buffer_arguments.empty());
    for (auto index : artifact.buffer_arguments) { expect(index < 2u); }
    // The reference realization must never contain Metal-only cooperative scopes.
    auto source = luisa::string_view{artifact.source.data(), artifact.source.size()};
    expect(source.find("metal.cooperative_tensor") == luisa::string_view::npos);
    expect(source.find("cooperative_tensor") == luisa::string_view::npos);
    // Native C++ codegen must supply complete headers without importing TVM's
    // Python package. Unsupported tags remain a recoverable compilation error.
    auto header_generator = tvm::ffi::Function::GetGlobal("tirx.intrinsics.cuda.header_generator");
    expect(header_generator.has_value());
    if (header_generator) {
        auto header = (*header_generator)(tvm::ffi::Array<tvm::ffi::String>{"cuda", "math_constants"}).cast<tvm::ffi::String>();
        auto text = std::string_view{header.data(), header.size()};
        expect(text.find("cuda/std/cstdint") != std::string_view::npos);
        expect(text.find("math_constants.h") != std::string_view::npos);
        auto rejected = header_generator->CallExpected<tvm::ffi::String>(
            tvm::ffi::Array<tvm::ffi::String>{"luisa_unknown_header_tag"});
        expect(!rejected.is_ok());
        if (!rejected.is_ok()) { expect(rejected.error().kind() == "ValueError"); }
    }
}

void test_cuda_reduction_artifact() {
    constexpr int64_t rows = 37;
    constexpr int64_t columns = 19;
    auto definition = tile_kernel("cuda_tirx_row_sum", [](TensorView<const float, 2> input,
                                                          TensorView<float, 1> result) {
        auto row = axis("row", rows);
        auto column = axis("column", columns);
        for (auto &row_nest : parallel(shape(row))) {
            auto sum = Scalar<float>{0.0f};
            for (auto &item : row_nest.reduce(shape(column))) {
                sum += input(row_nest.index(row), item.index(column)).load();
            }
            result(row_nest.index(row)).store(sum);
        }
    });
    auto kernel = definition.capture(tensor_shape(rows, columns), tensor_shape(rows));
    expect(kernel.valid());
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    CompileOptions options;
    options.target = cuda_target();
    options.noalias = true;
    auto result = compile_device(native.value, kernel.function().name(), options);
    expect(static_cast<bool>(result)) << result.error;
    if (!result) { return; }
    expect(result.artifact.format == DeviceArtifact::Format::CUDA_SOURCE);
    expect(!result.artifact.source.empty());
    expect(result.artifact.requires_metal4 == false);
    // REDUCE is realized by the reference left fold, never a Metal-only
    // subgroup/cooperative reduction planner.
    auto source = luisa::string_view{result.artifact.source.data(), result.artifact.source.size()};
    expect(source.find("#pragma unroll 1") == luisa::string_view::npos);
    expect(source.find("simdgroup") == luisa::string_view::npos);
    expect(source.find("metal.") == luisa::string_view::npos);
}

void test_cuda_matmul_artifact() {
    constexpr int64_t m = 32, n = 32, k = 16, tile = 16;
    auto definition = tile_kernel("cuda_tirx_gemm", [](TensorView<const float, 2> A,
                                                       TensorView<const float, 2> B,
                                                       TensorView<float, 2> C) {
        auto gm = axis("groups_m", m / tile);
        auto gn = axis("groups_n", n / tile);
        auto row = axis("row", tile);
        auto col = axis("column", tile);
        auto k_axis = axis("k", k);
        for (auto &nest : parallel(shape(gm, gn))) {
            auto m0 = nest.index(gm) * tile;
            auto n0 = nest.index(gn) * tile;
            auto a = A.tile(coord(m0, 0), shape(row, k_axis)).load();
            auto b = B.tile(coord(0, n0), shape(k_axis, col)).load();
            auto acc = zeros<float>(shape(row, col));
            acc = mma(a, b, acc);
            C(coord(m0, n0), shape(row, col)).store(acc);
        }
    });
    auto kernel = definition.capture(
        tensor_shape(m, k), tensor_shape(k, n), tensor_shape(m, n));
    expect(kernel.valid());
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    CompileOptions options;
    options.target = cuda_target();
    options.noalias = true;
    auto result = compile_device(native.value, kernel.function().name(), options);
    expect(static_cast<bool>(result)) << result.error;
    if (!result) { return; }
    auto &artifact = result.artifact;
    expect(artifact.format == DeviceArtifact::Format::CUDA_SOURCE);
    expect(!artifact.source.empty());
    expect(artifact.requires_metal4 == false);
    auto source = luisa::string_view{artifact.source.data(), artifact.source.size()};
    expect(source.find("#pragma unroll 1") == luisa::string_view::npos);
    // Semantic MMA stays reference-expanded for CUDA: no cooperative tensor,
    // WMMA/mma.sync, MPP fragment or Metal scope may leak into the artifact.
    expect(source.find("cooperative_tensor") == luisa::string_view::npos);
    expect(source.find("metal.") == luisa::string_view::npos);
    expect(source.find("fragment") == luisa::string_view::npos);
}

[[nodiscard]] constexpr luisa::string_view nvptx_target() noexcept {
    // The nvptx target kind declares mcpu/mtriple/max_num_threads/thread_warp_size
    // only: passing max_shared_memory_per_block makes Target construction throw
    // "Unknown config option" before any codegen runs.
    return R"({"kind":"nvptx","thread_warp_size":32,"max_num_threads":1024})";
}

void test_cuda_nvptx_elementwise_artifact() {
    constexpr int64_t n = 128;
    auto definition = tile_kernel("cuda_tirx_nvptx_elementwise", [](TensorView<const float, 1> input,
                                                                    TensorView<float, 1> output) {
        auto element = axis("element", n);
        for (auto &nest : parallel(shape(element))) {
            output(nest.index()).store(input(nest.index()).load() * 1.5f + 0.25f);
        }
    });
    auto kernel = definition.capture(tensor_shape(n), tensor_shape(n));
    expect(kernel.valid());
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    CompileOptions options;
    options.target = nvptx_target();
    options.noalias = true;
    auto result = compile_device(native.value, kernel.function().name(), options);
    if (!tvm::ffi::Function::GetGlobal("target.build.nvptx")) {
        // NVPTX is optional in the linked TVM build. Its absence must still
        // produce an explicit error, never a substituted CUDA-source artifact.
        expect(!static_cast<bool>(result));
        expect(result.error.find("target.build.nvptx") != luisa::string::npos) << result.error;
        return;
    }
    expect(static_cast<bool>(result)) << result.error;
    if (!result) { return; }
    auto &artifact = result.artifact;
    expect(artifact.format == DeviceArtifact::Format::PTX);
    expect(!artifact.entry.empty());
    expect(!artifact.source.empty());
    expect(artifact.grid[0] > 0u && artifact.grid[1] == 1u && artifact.grid[2] == 1u);
    auto threads = static_cast<uint64_t>(artifact.block[0]) * artifact.block[1] * artifact.block[2];
    expect(threads > 0u && threads <= 1024u && threads % 32u == 0u);
    expect(!artifact.buffer_arguments.empty());
    for (auto index : artifact.buffer_arguments) { expect(index < 2u); }
    auto source = luisa::string_view{artifact.source.data(), artifact.source.size()};
    expect(source.find(".target") != luisa::string_view::npos);
    expect(source.find("metal.") == luisa::string_view::npos);
}

void test_cuda_nvptx_reduction_artifact_warp_aligned() {
    constexpr int64_t rows = 37;
    constexpr int64_t columns = 19;
    auto definition = tile_kernel("cuda_tirx_nvptx_row_sum", [](TensorView<const float, 2> input,
                                                                TensorView<float, 1> result) {
        auto row = axis("row", rows);
        auto column = axis("column", columns);
        for (auto &row_nest : parallel(shape(row))) {
            auto sum = Scalar<float>{0.0f};
            for (auto &item : row_nest.reduce(shape(column))) {
                sum += input(row_nest.index(row), item.index(column)).load();
            }
            result(row_nest.index(row)).store(sum);
        }
    });
    auto kernel = definition.capture(tensor_shape(rows, columns), tensor_shape(rows));
    expect(kernel.valid());
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    CompileOptions options;
    options.target = nvptx_target();
    options.noalias = true;
    auto result = compile_device(native.value, kernel.function().name(), options);
    if (!tvm::ffi::Function::GetGlobal("target.build.nvptx")) {
        expect(!static_cast<bool>(result));
        expect(result.error.find("target.build.nvptx") != luisa::string::npos) << result.error;
        return;
    }
    expect(static_cast<bool>(result)) << result.error;
    if (!result) { return; }
    auto &artifact = result.artifact;
    expect(artifact.format == DeviceArtifact::Format::PTX);
    expect(!artifact.source.empty());
    // A 37-row reference root must not produce a 37-thread CUDA block: the
    // CUDA/NVPTX mapper pads partial domains up to the 32-thread warp.
    auto threads = static_cast<uint64_t>(artifact.block[0]) * artifact.block[1] * artifact.block[2];
    expect(threads > 0u && threads <= 1024u && threads % 32u == 0u) << artifact.block[0];
    expect(artifact.grid[0] > 0u);
    auto source = luisa::string_view{artifact.source.data(), artifact.source.size()};
    expect(source.find("simdgroup") == luisa::string_view::npos);
    expect(source.find("metal.") == luisa::string_view::npos);
}

void test_cuda_ragged_reordered_gemm_artifact() {
    constexpr int64_t m = 48, n = 33, k = 24, tile = 16;
    // Deliberately order the parameters so the *output* buffer is the second
    // host argument. extract_device_artifact records device-slot -> host-index
    // buffer_arguments; the CUDA launcher relies on that permutation rather
    // than on host order matching device binding order.
    auto definition = tile_kernel("cuda_tirx_ragged_reordered_gemm", [](TensorView<const float, 2> A,
                                                                        TensorView<float, 2> C,
                                                                        TensorView<const float, 2> B) {
        auto gm = axis("groups_m", ceil_div(m, tile));
        auto gn = axis("groups_n", ceil_div(n, tile));
        auto row = axis("row", tile);
        auto col = axis("column", tile);
        auto k_axis = axis("k", k);
        for (auto &nest : parallel(shape(gm, gn))) {
            auto m0 = nest.index(gm) * tile;
            auto n0 = nest.index(gn) * tile;
            auto a = A.tile(coord(m0, 0), shape(row, k_axis)).load();
            auto b = B.tile(coord(0, n0), shape(k_axis, col)).load();
            auto acc = zeros<float>(shape(row, col));
            acc = mma(a, b, acc);
            C(coord(m0, n0), shape(row, col)).store(acc);
        }
    });
    auto kernel = definition.capture(tensor_shape(m, k), tensor_shape(m, n), tensor_shape(k, n));
    expect(kernel.valid());
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    CompileOptions options;
    options.target = cuda_target();
    options.noalias = true;
    auto result = compile_device(native.value, kernel.function().name(), options);
    expect(static_cast<bool>(result)) << result.error;
    if (!result) { return; }
    auto &artifact = result.artifact;
    expect(artifact.format == DeviceArtifact::Format::CUDA_SOURCE);
    expect(!artifact.source.empty());
    // buffer_arguments is a bijection onto the three host arguments; each host
    // index appears exactly once regardless of TVM's device binding order.
    expect(artifact.buffer_arguments.size() == 3u);
    std::array<bool, 3u> seen{};
    for (auto index : artifact.buffer_arguments) {
        expect(index < 3u && !seen[index]);
        if (index < 3u) { seen[index] = true; }
    }
    expect(seen[0u] && seen[1u] && seen[2u]);
    auto source = luisa::string_view{artifact.source.data(), artifact.source.size()};
    expect(source.find("cooperative_tensor") == luisa::string_view::npos);
    expect(source.find("simdgroup") == luisa::string_view::npos);
    expect(source.find("metal.") == luisa::string_view::npos);
    expect(source.find("fragment") == luisa::string_view::npos);
}

void test_cuda_permutation_unroll_budget() {
    // The same portable sort covers both sides of the structural budget. This
    // test stops at CUDA source generation; runtime primitives retain the full
    // numerical/guard tests through NVRTC and the actual CUDA backend.
    for (auto width : {int64_t{8}, int64_t{32}, int64_t{128}}) {
        auto definition = tile_kernel("cuda_tirx_permutation_budget", [=](TensorView<const float, 2> input,
                                                                          TensorView<float, 2> values,
                                                                          TensorView<int64_t, 2> indices) {
            auto row = axis("row", 4), column = axis("column", width);
            for (auto &nest : parallel(shape(1))) {
                static_cast<void>(nest);
                auto value = input.tile(coord(0, 0), shape(row, column)).load();
                auto ranked = luisa::compute::tile::sort(value, column);
                values(coord(0, 0), ranked.values.space()).store(ranked.values);
                indices(coord(0, 0), ranked.indices.space()).store(ranked.indices);
            }
        });
        auto kernel = definition.capture(tensor_shape(4, width), tensor_shape(4, width), tensor_shape(4, width));
        expect(kernel.valid());
        if (!kernel.valid()) { return; }
        auto native = lower(kernel.function());
        expect(native.ok()) << native.error;
        if (!native) { return; }
        CompileOptions options;
        options.target = cuda_target();
        options.noalias = true;
        auto result = compile_device(native.value, kernel.function().name(), options);
        expect(static_cast<bool>(result)) << result.error;
        if (!result) { return; }
        expect(result.artifact.format == DeviceArtifact::Format::CUDA_SOURCE);
        auto guarded = result.artifact.source.find("#pragma unroll 1") != luisa::string::npos;
        expect(guarded == (width >= 32)) << "permutation width=" << width;
    }
}

void test_cuda_fail_closed_options() {
    constexpr int64_t n = 64;
    auto definition = tile_kernel("cuda_tirx_fail_closed", [](TensorView<const float, 1> input,
                                                              TensorView<float, 1> output) {
        auto element = axis("element", n);
        for (auto &nest : parallel(shape(element))) {
            output(nest.index()).store(input(nest.index()).load() + 1.0f);
        }
    });
    auto kernel = definition.capture(tensor_shape(n), tensor_shape(n));
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }

    CompileOptions cooperative;
    cooperative.target = cuda_target();
    cooperative.cooperative_matrix = true;
    auto result = compile_device(native.value, kernel.function().name(), cooperative);
    expect(!static_cast<bool>(result));
    expect(luisa::string_view{result.error.data(), result.error.size()}.find("cooperative") != luisa::string_view::npos)
        << result.error;

    CompileOptions mpp;
    mpp.target = cuda_target();
    mpp.metal_mpp = true;
    result = compile_device(native.value, kernel.function().name(), mpp);
    expect(!static_cast<bool>(result));
    expect(luisa::string_view{result.error.data(), result.error.size()}.find("MPP") != luisa::string_view::npos)
        << result.error;

    CompileOptions subgroup;
    subgroup.target = cuda_target();
    subgroup.planner.metal_subgroup_reductions = true;
    result = compile_device(native.value, kernel.function().name(), subgroup);
    expect(!static_cast<bool>(result));
    expect(luisa::string_view{result.error.data(), result.error.size()}.find("subgroup") != luisa::string_view::npos ||
           luisa::string_view{result.error.data(), result.error.size()}.find("SIMD") != luisa::string_view::npos)
        << result.error;

    CompileOptions unknown_target;
    unknown_target.target = "vulkan";
    result = compile_device(native.value, kernel.function().name(), unknown_target);
    expect(!static_cast<bool>(result));
}

// CUDA source-only coverage for the opt-in shared reduction mapper. Numerical
// execution, CUDA compiler acceptance, and physical resources are separate tests.
[[nodiscard]] CompileOptions cuda_subgroup_options() {
    CompileOptions options;
    options.target = cuda_target();
    options.noalias = true;
    options.planner.cuda_subgroup_reductions = true;
    return options;
}

[[nodiscard]] Kernel cuda_subgroup_sum_kernel() {
    auto definition = tile_kernel("cuda_subgroup_contract", [](TensorView<const float, 2> input,
                                                               TensorView<float, 1> output) {
        auto one = axis("one", 1), column = axis("column", 37);
        for (auto &nest : parallel(shape(3))) {
            auto x = input.tile(coord(nest.index(), 0), shape(one, column)).load();
            output(coord(nest.index()), shape(one)).store(reduce(x, column, add));
        }
    });
    return definition.capture(tensor_shape(3, 37), tensor_shape(3));
}

void expect_cuda_subgroup_source(const DeviceArtifact &artifact) {
    expect(artifact.format == DeviceArtifact::Format::CUDA_SOURCE);
    expect(!artifact.requires_metal4);
    auto source = luisa::string_view{artifact.source.data(), artifact.source.size()};
    for (auto forbidden : {"metal.", "simdgroup", "simd_sum(", "simd_max(", "simd_min(", "threadgroup_barrier("}) {
        expect(source.find(forbidden) == luisa::string_view::npos) << forbidden;
    }
}

void cuda_subgroup_unroll_boundaries() {
    // Small domains exercise min(chunks, U); 4096/64 exercises a complete
    // 64-chunk stripe. These are CUDA-source artifacts, not NVRTC/GPU tests.
    for (auto [columns, factor] : {std::pair{37u, 16u}, {37u, 17u}, {37u, 32u}, {37u, 64u}, {4096u, 64u}}) {
        auto definition = tile_kernel("cuda_subgroup_unroll_boundary", [=](TensorView<const float, 2> input,
                                                                           TensorView<float, 1> output) {
            auto one = axis("one", 1), column = axis("column", columns);
            for (auto &nest : parallel(shape(3))) {
                auto x = input.tile(coord(nest.index(), 0), shape(one, column)).load();
                output(coord(nest.index()), shape(one)).store(reduce(x, column, add));
            }
        });
        auto kernel = definition.capture(tensor_shape(3, columns), tensor_shape(3));
        expect(kernel.valid());
        auto native = lower(kernel.function());
        expect(native.ok()) << native.error;
        if (!native) { continue; }
        auto options = cuda_subgroup_options();
        options.planner.threads_per_group = 64u;
        options.planner.reduction_programs_per_group = 1u;
        options.planner.reduction_unroll_factor = factor;
        auto result = compile_device(native.value, kernel.function().name(), options);
        expect(static_cast<bool>(result)) << "N=" << columns << " U=" << factor << " " << result.error;
        if (!result) { continue; }
        expect_cuda_subgroup_source(result.artifact);
        expect(result.artifact.grid == (std::array<uint32_t, 3u>{3u, 1u, 1u}));
        expect(result.artifact.block == (std::array<uint32_t, 3u>{64u, 1u, 1u}));
        expect(result.artifact.buffer_arguments.size() == 2u);
        expect(result.plans.size() == 1u);
        if (result.plans.size() == 1u) {
            auto &plan = result.plans.front();
            expect(plan.reduction_unroll_factor == factor);
            expect(plan.reduction_lane_elements == 1u);
            expect(plan.reduction_operations == 1u);
            expect(plan.striped_storage_scalars_per_worker <= options.planner.max_reduction_striped_scalars_per_worker);
        }
    }

    // A reused narrow snapshot forces an actual private stripe. Inspect the
    // resulting typed artifact rather than guessing scalarization from the
    // requested GroupPlan field or generated C++ string count.
    {
        auto definition = tile_kernel("cuda_subgroup_unroll_private_indices", [](TensorView<const bfloat16, 2> input,
                                                                                  TensorView<bfloat16, 2> output) {
            auto one = axis("one", 1), feature = axis("feature", 4096);
            for (auto &nest : parallel(shape(3))) {
                auto origin = coord(nest.index(), 0);
                auto x = cast<float>(input.tile(origin, shape(one, feature)).load());
                auto total = reduce(x * x, feature, add);
                output(origin, shape(one, feature)).store(cast<bfloat16>(x / sqrt(total + 1e-5f)));
            }
        });
        auto kernel = definition.capture(tensor_shape(3, 4096), tensor_shape(3, 4096));
        expect(kernel.valid());
        auto native = lower(kernel.function());
        expect(native.ok()) << native.error;
        if (native) {
            auto options = cuda_subgroup_options();
            options.planner.threads_per_group = 64u;
            options.planner.reduction_programs_per_group = 1u;
            options.planner.reduction_unroll_factor = 64u;
            auto result = compile_device(native.value, kernel.function().name(), options);
            expect(static_cast<bool>(result)) << result.error;
            if (result) {
                expect(result.plans.size() == 1u);
                if (result.plans.size() == 1u) {
                    expect(result.plans.front().reduction_unroll_factor == 64u);
                    expect(result.plans.front().striped_storage_scalars_per_worker == 64u);
                }
                auto local_accesses = 0u, nonconstant_indices = 0u;
                auto inspect = [&](const auto &buffer, const auto &indices) {
                    if (buffer.scope() != "local") { return; }
                    local_accesses++;
                    for (auto &index : indices) {
                        nonconstant_indices += index.template as<tvm::IntImmNode>() == nullptr;
                    }
                };
                tvm::tirx::PostOrderVisit(result.artifact.function->body, [&](const tvm::ffi::ObjectRef &node) {
                    if (auto load = node.as<tvm::tirx::BufferLoadNode>()) { inspect(load->buffer, load->indices); }
                    if (auto store = node.as<tvm::tirx::BufferStoreNode>()) { inspect(store->buffer, store->indices); }
                });
                expect(nonconstant_indices == 0u) << "local accesses=" << local_accesses;
                // Zero remaining local accesses is also valid if preceding
                // scalarization has removed every private Buffer object.
                expect_cuda_subgroup_source(result.artifact);
            }
        }
    }

    auto kernel = cuda_subgroup_sum_kernel();
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    for (auto factor : {65u, UINT32_MAX}) {
        auto options = cuda_subgroup_options();
        options.planner.reduction_unroll_factor = factor;
        auto rejected = compile_device(native.value, kernel.function().name(), options);
        expect(!static_cast<bool>(rejected));
        expect(rejected.error.find("[1,64]") != luisa::string::npos) << rejected.error;
    }
    // Metal rejects the new range during mapping, before requesting a Metal
    // code generator. This negative needs neither a Metal device nor runtime.
    for (auto factor : {0u, 17u, 64u}) {
        auto options = cuda_subgroup_options();
        options.target = R"({"kind":"metal","thread_warp_size":32,"max_num_threads":1024,"max_shared_memory_per_block":32768})";
        options.planner.cuda_subgroup_reductions = false;
        options.planner.metal_subgroup_reductions = true;
        options.planner.reduction_unroll_factor = factor;
        auto rejected = compile_device(native.value, kernel.function().name(), options);
        expect(!static_cast<bool>(rejected));
        expect(rejected.error.find("[1,16]") != luisa::string::npos) << rejected.error;
    }
    for (auto mode = 0u; mode != 5u; mode++) {
        auto options = cuda_subgroup_options();
        options.planner.reduction_unroll_factor = 64u;
        if (mode == 0u) { options.planner.cuda_subgroup_reductions = false; }
        if (mode == 1u) { options.target = nvptx_target(); }
        if (mode == 2u) { options.noalias = false; }
        if (mode == 3u) { options.planner.enabled = false; }
        if (mode == 4u) {
            options.target = R"({"kind":"cuda","thread_warp_size":16,"max_num_threads":1024,"max_shared_memory_per_block":98304})";
        }
        auto rejected = compile_device(native.value, kernel.function().name(), options);
        expect(!static_cast<bool>(rejected)) << "mode=" << mode;
        expect(!rejected.error.empty());
    }
    auto options = cuda_subgroup_options();
    options.planner.reduction_unroll_factor = 64u;
    auto non_device = luisa::compute::tile::bridge::tirx::compile(native.value, kernel.function().name(), options);
    expect(!non_device.ok());
    expect(!non_device.error().empty());
}

void cuda_subgroup_automatic_unroll_gates() {
    auto kernel = cuda_subgroup_sum_kernel();
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    auto options = cuda_subgroup_options();
    options.planner.threads_per_group = 64u;
    options.planner.reduction_programs_per_group = 1u;
    auto exact = compile_device(native.value, kernel.function().name(), options);
    options.planner.reduction_unroll_factor = 0u;
    auto automatic = compile_device(native.value, kernel.function().name(), options);
    expect(static_cast<bool>(exact) && static_cast<bool>(automatic));
    if (exact && automatic) {
        expect(automatic.artifact.source == exact.artifact.source);
        expect(automatic.plans.size() == 1u);
        if (automatic.plans.size() == 1u) {
            expect(automatic.plans.front().striped_storage_scalars_per_worker == 0u);
            expect(automatic.plans.front().reduction_unroll_factor == 1u);
        }
    }
    for (auto mode = 0u; mode != 5u; mode++) {
        auto rejected_options = options;
        if (mode == 0u) { rejected_options.planner.cuda_subgroup_reductions = false; }
        if (mode == 1u) { rejected_options.target = nvptx_target(); }
        if (mode == 2u) { rejected_options.noalias = false; }
        if (mode == 3u) { rejected_options.planner.enabled = false; }
        if (mode == 4u) { rejected_options.planner.max_reduction_striped_scalars_per_worker = 0u; }
        auto rejected = compile_device(native.value, kernel.function().name(), rejected_options);
        expect(!static_cast<bool>(rejected)) << "auto gate=" << mode;
        expect(!rejected.error.empty());
    }
    class Ordered final : public tvm::tirx::StmtMutator {
        tvm::tirx::Stmt VisitStmt_(const tvm::tirx::ForNode *loop) final {
            auto result = StmtMutator::VisitStmt_(loop);
            if (loop->annotations.count("luisa.tile.reduction_policy")) {
                auto rewritten = result.as_or_throw<tvm::tirx::For>();
                rewritten.CopyOnWrite()->annotations.Set("luisa.tile.reduction_policy",
                    tvm::IntImm::Int64(static_cast<int64_t>(reduction::ordered_tree)));
                result = rewritten;
            }
            return result;
        }
    } ordered;
    auto function = native.value;
    function.CopyOnWrite()->body = ordered(function->body);
    auto rejected = compile_device(function, kernel.function().name(), options);
    expect(!static_cast<bool>(rejected));
    expect(!rejected.error.empty());
}

void cuda_subgroup_private_index_facts() {
    class Observe final : public AnalyticExecutionCostPolicy {
    public:
        mutable luisa::optional<ReductionCandidate> last;
        ReductionCost reduction_cost(const ReductionCandidate &candidate, const ExecutionCostModel &model) const noexcept override {
            last = candidate;
            return AnalyticExecutionCostPolicy::reduction_cost(candidate, model);
        }
    } policy;
    // The same named SSA snapshot is used by its materialization, two
    // reductions and an epilogue. Domain multiplicity must not add the needed
    // unroll factors. J=0,1,15,16,64 pins both sides of the real pack boundary.
    for (auto [columns, required] : {std::pair{37u, 1u}, {64u, 1u}, {960u, 8u}, {1024u, 9u}, {4096u, 33u}}) {
        auto definition = tile_kernel("cuda_private_index_fact", [=](TensorView<const bfloat16, 2> input,
                                                                     TensorView<bfloat16, 2> output) {
            auto one = axis("one", 1), feature = axis("feature", columns);
            for (auto &nest : parallel(shape(3))) {
                auto origin = coord(nest.index(), 0);
                auto x = cast<float>(input.tile(origin, shape(one, feature)).load());
                auto sum = reduce(x * x, feature, add);
                auto largest = reduce(x, feature, maximum);
                output(origin, shape(one, feature)).store(cast<bfloat16>(x / sqrt(sum + 1e-5f) + largest));
            }
        });
        auto kernel = definition.capture(tensor_shape(3, columns), tensor_shape(3, columns));
        expect(kernel.valid());
        auto native = lower(kernel.function());
        expect(native.ok()) << native.error;
        if (!native) { continue; }
        auto options = cuda_subgroup_options();
        options.planner.threads_per_group = 64u;
        options.planner.reduction_programs_per_group = 1u;
        options.planner.reduction_unroll_factor = required;
        auto control = compile_device(native.value, kernel.function().name(), options);
        expect(static_cast<bool>(control)) << control.error;
        options.planner.cost_policy = &policy;
        policy.last.reset();
        auto result = compile_device(native.value, kernel.function().name(), options);
        expect(static_cast<bool>(result)) << result.error;
        expect(policy.last.has_value());
        if (!result || !policy.last || !control) { continue; }
        auto fact = policy.last->source_constant_striped_index_min_unroll;
        expect(fact.has_value());
        if (fact) { expect(*fact == required) << "N=" << columns; }
        expect(result.artifact.source == control.artifact.source);
        options.planner.reduction_unroll_factor = 0u;
        auto automatic = compile_device(native.value, kernel.function().name(), options);
        expect(static_cast<bool>(automatic)) << automatic.error;
        if (automatic) {
            expect(automatic.artifact.source == control.artifact.source);
            expect(automatic.artifact.grid == control.artifact.grid);
            expect(automatic.artifact.block == control.artifact.block);
            expect(automatic.plans.size() == 1u);
            if (automatic.plans.size() == 1u) {
                expect(automatic.plans.front().reduction_unroll_factor == required);
                expect(automatic.plans.front().cost.kernel_score == control.plans.front().cost.kernel_score);
            }
            expect(policy.last->unroll_factor == required);
            expect(policy.last->source_constant_striped_index_min_unroll == fact);
        }
        expect(policy.last->reductions == 2u);
        expect(policy.last->striped_scalars_per_worker == (columns + 63u) / 64u);
        auto nonconstant = 0u;
        auto inspect = [&](const auto &buffer, const auto &indices) {
            if (buffer.scope() == "local") {
                for (auto &index : indices) {
                    nonconstant += index.template as<tvm::IntImmNode>() == nullptr;
                }
            }
        };
        tvm::tirx::PostOrderVisit(result.artifact.function->body, [&](const tvm::ffi::ObjectRef &node) {
            if (auto load = node.as<tvm::tirx::BufferLoadNode>()) { inspect(load->buffer, load->indices); }
            if (auto store = node.as<tvm::tirx::BufferStoreNode>()) { inspect(store->buffer, store->indices); }
        });
        expect(nonconstant == 0u) << "N=" << columns << " U=" << required;
        if (required > 1u) {
            options.planner.reduction_unroll_factor = required - 1u;
            auto insufficient = compile_device(native.value, kernel.function().name(), options);
            expect(static_cast<bool>(insufficient)) << insufficient.error;
            if (insufficient) {
                nonconstant = 0u;
                tvm::tirx::PostOrderVisit(insufficient.artifact.function->body, [&](const tvm::ffi::ObjectRef &node) {
                    if (auto load = node.as<tvm::tirx::BufferLoadNode>()) { inspect(load->buffer, load->indices); }
                    if (auto store = node.as<tvm::tirx::BufferStoreNode>()) { inspect(store->buffer, store->indices); }
                });
                expect(nonconstant > 0u);
                expect(policy.last->source_constant_striped_index_min_unroll == fact);
            }
        }
    }

    // Raw rank-two private materialization: the exact row-major flattening
    // is accepted; transposed and data-dependent ownership remain rejected.
    // A raised explicit storage budget lets the known required U exceed 64;
    // that is distinct from unknown and must not silently become U64.
    auto i64 = [](int64_t value) { return tvm::IntImm::Int64(value); };
    auto f32 = [](float value) { return tvm::FloatImm{tvm::PrimType::Float(32), value}; };
    for (auto mode = 0u; mode != 4u; mode++) {
        auto columns = int64_t{mode == 3u ? 8256 : 1024};
        auto input = tvm::tirx::decl_buffer({i64(3), i64(columns)}, tvm::PrimType::Float(32), "input");
        auto output = tvm::tirx::decl_buffer({i64(3)}, tvm::PrimType::Float(32), "output");
        auto private_tile = tvm::tirx::decl_buffer({i64(2), i64(columns / 2)}, tvm::PrimType::Float(32), "snapshot", "local");
        auto carry = tvm::tirx::decl_buffer({i64(1)}, tvm::PrimType::Float(32), "carry", "local");
        auto next = tvm::tirx::decl_buffer({i64(1)}, tvm::PrimType::Float(32), "next", "local");
        auto p = tvm::tirx::PrimVar{"program", tvm::PrimType::Int(64)};
        auto a = tvm::tirx::PrimVar{"axis0", tvm::PrimType::Int(64)};
        auto b = tvm::tirx::PrimVar{"axis1", tvm::PrimType::Int(64)};
        auto k = tvm::tirx::PrimVar{"reduce", tvm::PrimType::Int(64)};
        auto e = tvm::tirx::PrimVar{"out", tvm::PrimType::Int(64)};
        auto fill = tvm::tirx::For{a, i64(0), i64(2), tvm::tirx::ForKind::kSerial,
            tvm::tirx::For{b, i64(0), i64(columns / 2), tvm::tirx::ForKind::kSerial,
                tvm::tirx::BufferStore{private_tile, tvm::tirx::BufferLoad{input, {p, a * i64(columns / 2) + b}}, {a, b}}}, {},
            {{"luisa.tile.independent_elements", i64(2)}, {"luisa.tile.contract.materialized_pure_tile", i64(1)}}};
        tvm::ffi::Array<tvm::PrimExpr> indices{tvm::floordiv(k, i64(columns / 2)), tvm::floormod(k, i64(columns / 2))};
        if (mode == 1u) { indices = {tvm::floormod(k, i64(2)), tvm::floordiv(k, i64(2))}; }
        if (mode == 2u) { indices = {i64(0), tvm::cast(tvm::PrimType::Int(64), tvm::tirx::BufferLoad{input, {p, k}})}; }
        auto update = tvm::tirx::SeqStmt::Flatten(tvm::ffi::Array<tvm::tirx::Stmt>{
            tvm::tirx::AllocBuffer{next},
            tvm::tirx::BufferStore{next, tvm::tirx::BufferLoad{carry, {i64(0)}} + tvm::tirx::BufferLoad{private_tile, indices}, {i64(0)}},
            tvm::tirx::BufferStore{carry, tvm::tirx::BufferLoad{next, {i64(0)}}, {i64(0)}}});
        auto reduce_loop = tvm::tirx::For{k, i64(0), i64(columns), tvm::tirx::ForKind::kSerial, update, {},
            {{"luisa.tile.contract.reduction", i64(1)}, {"luisa.tile.reduction_policy", i64(static_cast<int64_t>(reduction::unordered_tree))}}};
        auto store = tvm::tirx::For{e, i64(0), i64(1), tvm::tirx::ForKind::kSerial,
            tvm::tirx::BufferStore{output, tvm::tirx::BufferLoad{carry, {i64(0)}}, {p + e}}, {}, {{"luisa.tile.independent_elements", i64(1)}}};
        auto body = tvm::tirx::SeqStmt::Flatten(tvm::ffi::Array<tvm::tirx::Stmt>{tvm::tirx::AllocBuffer{private_tile}, fill,
            tvm::tirx::AllocBuffer{carry}, tvm::tirx::BufferStore{carry, f32(0.f), {i64(0)}}, reduce_loop, store});
        auto function = tvm::tirx::PrimFunc{{input, output}, tvm::tirx::For{p, i64(0), i64(3), tvm::tirx::ForKind::kSerial,
            body, {}, {{"luisa.tile.logical_parallel", i64(1)}}}};
        auto options = cuda_subgroup_options();
        options.planner.threads_per_group = 64u;
        options.planner.reduction_programs_per_group = 1u;
        options.planner.max_reduction_striped_scalars_per_worker = 256u;
        options.planner.cost_policy = &policy;
        policy.last.reset();
        auto result = compile_device(function, "private_index_domain", options);
        expect(eq(static_cast<bool>(result), mode == 0u || mode == 3u)) << result.error;
        if (mode == 1u || mode == 2u) { expect(!policy.last); }
        else {
            expect(policy.last.has_value());
            if (policy.last) {
                auto fact = policy.last->source_constant_striped_index_min_unroll;
                expect(fact.has_value());
                if (fact) { expect(*fact == (mode == 3u ? 65u : 9u)); }
            }
        }
        options.planner.reduction_unroll_factor = 0u;
        auto automatic = compile_device(function, "private_index_domain", options);
        expect(eq(static_cast<bool>(automatic), mode == 0u)) << automatic.error;
        if (automatic) {
            expect(automatic.plans.front().reduction_unroll_factor == 9u);
            // Unknown ownership and known U65 are both rejected automatically;
            // the preceding manual U1 control preserves their distinct facts.
        }
    }
    cuda_subgroup_automatic_unroll_gates();
}

void test_cuda_subgroup_integer_extrema() {
    constexpr auto name = "LUISA_DIAGNOSTIC_TIRX_INTEGER_EXTREMA";
    // Match the process-environment test: DLL-local CRT tables are insufficient
    // on Windows, and an existing empty value must survive this test as empty.
    auto set_value = [](const char *key, const char *value) noexcept {
#ifdef _WIN32
        return SetEnvironmentVariableA(key, value) != 0 ||
               (value == nullptr && GetLastError() == ERROR_ENVVAR_NOT_FOUND);
#else
        return value == nullptr ? unsetenv(key) == 0 : setenv(key, value, 1) == 0;
#endif
    };
    struct RestoreEnvironment {
        const char *name;
        decltype(set_value) set;
        luisa::optional<luisa::string> previous;
        ~RestoreEnvironment() noexcept {
            LUISA_ASSERT(set(name, previous ? previous->c_str() : nullptr),
                         "Could not restore integer-extrema test environment.");
        }
    } restore{name, set_value, luisa::get_environment_variable(name)};

    // Reuse the existing unordered SUM fixture. The generated helper prefix
    // also contains MIN/MAX; their numerical use is covered by the GPU tests.
    auto kernel = cuda_subgroup_sum_kernel();
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    auto options = cuda_subgroup_options();
    options.planner.threads_per_group = 128u;
    options.planner.reduction_programs_per_group = 2u;
    auto compile = [&](const char *value) {
        auto changed = set_value(name, value);
        expect(changed);
        if (!changed) { return DeviceCompilationResult{}; }
        return compile_device(native.value, kernel.function().name(), options);
    };
    auto original = compile(nullptr);
    auto zero = compile("0");
    auto candidate = compile("1");
    expect(static_cast<bool>(original)) << original.error;
    expect(static_cast<bool>(zero)) << zero.error;
    expect(static_cast<bool>(candidate)) << candidate.error;
    if (!original || !zero || !candidate) { return; }
    expect(zero.artifact.source == original.artifact.source);
    expect(candidate.artifact.source != original.artifact.source);
    expect(candidate.artifact.entry == original.artifact.entry);
    expect(candidate.artifact.grid == original.artifact.grid);
    expect(candidate.artifact.block == original.artifact.block);
    expect(candidate.artifact.buffer_arguments == original.artifact.buffer_arguments);
    expect(original.plans.size() == 1u && candidate.plans.size() == 1u);
    if (original.plans.size() == 1u && candidate.plans.size() == 1u) {
        auto &a = original.plans.front();
        auto &b = candidate.plans.front();
        expect(a.reduction_operations == b.reduction_operations);
        expect(a.reduction_subgroups_per_program == b.reduction_subgroups_per_program);
        expect(a.reduction_programs_per_group == b.reduction_programs_per_group);
        expect(a.group_barrier_sites_after == b.group_barrier_sites_after);
        expect(a.shared_memory_bytes == b.shared_memory_bytes);
    }

    auto helper = [](luisa::string_view source, luisa::string_view signature) {
        auto start = source.find(signature);
        if (start == luisa::string_view::npos) { return luisa::string_view{}; }
        auto end = source.find("\n}\n", start);
        return end == luisa::string_view::npos ? luisa::string_view{} : source.substr(start, end + 3u - start);
    };
    auto normalized = candidate.artifact.source;
    for (auto kind : {"min", "max"}) {
        auto signature = luisa::string{"static __device__ __forceinline__ float __luisa_tile_cuda_warp_"} + kind + "(float value) {";
        auto before = helper(original.artifact.source, signature);
        auto after = helper(normalized, signature);
        expect(!before.empty() && !after.empty());
        if (before.empty() || after.empty()) { return; }
        auto intrinsic = luisa::string{"__reduce_"} + kind + "_sync(0xffffffffu, key)";
        expect(before.find("__reduce_") == luisa::string_view::npos);
        expect(after.find(intrinsic) != luisa::string_view::npos);
        expect(after.find("#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)") != luisa::string_view::npos);
        auto loop_position = before.find("#pragma unroll");
        expect(loop_position != luisa::string_view::npos);
        if (loop_position == luisa::string_view::npos) { return; }
        auto old_loop = before.substr(loop_position);
        old_loop.remove_suffix(2u);// final function brace; loop and return stay
        expect(after.find(old_loop) != luisa::string_view::npos);
        auto position = static_cast<size_t>(after.data() - normalized.data());
        normalized.replace(position, after.size(), before.data(), before.size());
    }
    // Exact reverse normalization proves default helpers, SUM/local arithmetic,
    // emitted kernel, bindings and all other source text stayed unchanged.
    expect(normalized == original.artifact.source);

    for (auto value : {"", "2", "true", "01", " 1"}) {
        auto rejected = compile(value);
        expect(!static_cast<bool>(rejected));
        expect(rejected.error == "private CUDA integer extrema must be exactly 0 or 1");
        expect(rejected.artifact.source.empty() && rejected.plans.empty());
    }
    auto changed = set_value(name, "1");
    expect(changed);
    if (!changed) { return; }
    auto reference = options;
    reference.planner.cuda_subgroup_reductions = false;
    auto rejected = compile_device(native.value, kernel.function().name(), reference);
    expect(!static_cast<bool>(rejected));
    expect(rejected.error == "private CUDA integer extrema require CUDA subgroup reductions");
    expect(rejected.artifact.source.empty() && rejected.plans.empty());
    auto restored = compile(nullptr);
    expect(static_cast<bool>(restored)) << restored.error;
    if (restored) { expect(restored.artifact.source == original.artifact.source); }
}


void test_prepared_reduction_candidate() {
    constexpr auto flag = "LUISA_DIAGNOSTIC_TIRX_VECTOR_PACKS";
    auto set = [](const char *key, const char *value) noexcept {
#ifdef _WIN32
        return SetEnvironmentVariableA(key, value) != 0 || (value == nullptr && GetLastError() == ERROR_ENVVAR_NOT_FOUND);
#else
        return value ? setenv(key, value, 1) == 0 : unsetenv(key) == 0;
#endif
    };
    struct Restore {
        decltype(set) setter;
        luisa::optional<luisa::string> prior;
        ~Restore() noexcept { LUISA_ASSERT(setter("LUISA_DIAGNOSTIC_TIRX_VECTOR_PACKS", prior ? prior->c_str() : nullptr)); }
    } restore{set, luisa::get_environment_variable(flag)};
    class Observe final : public AnalyticExecutionCostPolicy {
    public:
        mutable std::array<std::array<double, 11u>, 32u> calls{};
        mutable size_t count{0u};
        ReductionCost reduction_cost(const ReductionCandidate &c, const ExecutionCostModel &m) const noexcept override {
            auto cost = AnalyticExecutionCostPolicy::reduction_cost(c, m);
            LUISA_ASSERT(count < calls.size());
            calls[count++] = {static_cast<double>(c.threads), static_cast<double>(c.subgroups_per_program),
                              static_cast<double>(c.programs_per_group), static_cast<double>(c.striped_scalars_per_worker),
                              c.scalar_rounds, c.scalar_elements, c.lane_utilization, static_cast<double>(c.payload_accesses_known),
                              cost.program_score, cost.concurrent_waves, cost.kernel_score};
            // Retain a middle candidate even after later candidates are prepared.
            cost.program_score = c.subgroups_per_program == 2u ? 0.0 : 1.0 + cost.program_score;
            cost.kernel_score = cost.program_score * cost.concurrent_waves;
            return cost;
        }
    } policy;
    auto definition = tile_kernel("prepared_candidate_probe", [](TensorView<const float, 2> input, TensorView<float, 1> output) {
        auto one = axis("one", 1), column = axis("column", 512);
        for (auto &nest : parallel(shape(3))) {
            // Squeeze the singleton row through the existing reshape algorithm:
            // the contribution then indexes the input with literal row offset 0.
            auto x = reshape(input.tile(coord(nest.index(), 0), shape(one, column)).load(), shape(column));
            output(coord(nest.index()), shape(one)).store(reduce(x, column, add));
        }
    });
    auto kernel = definition.capture(tensor_shape(3, 512), tensor_shape(3));
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    for (auto metal : {false, true}) {
        if (metal && !tvm::ffi::Function::GetGlobal("target.build.metal")) {
            LUISA_INFO("PreparedCandidate Metal source coverage unavailable: target.build.metal is absent.");
            continue;
        }
        auto options = cuda_subgroup_options();
        options.target = metal ? R"({"kind":"metal","thread_warp_size":32,"max_num_threads":128,"max_shared_memory_per_block":32768})" :
                                 R"({"kind":"cuda","thread_warp_size":32,"max_num_threads":128,"max_shared_memory_per_block":32768})";
        options.planner.cuda_subgroup_reductions = !metal;
        options.planner.metal_subgroup_reductions = metal;
        options.planner.reduction_programs_per_group = 1u;
        options.planner.reduction_lane_elements = 4u;
        options.planner.reduction_unroll_factor = 16u;
        options.planner.cache_reduction_inputs = false;
        options.planner.cost_policy = &policy;
        std::array<std::array<double, 11u>, 32u> original_calls{};
        for (auto vector : {false, true}) {
            if (metal && vector) { continue; }
            expect(set(flag, vector ? "1" : "0"));
            options.planner.threads_per_group = 0u;
            policy.count = 0u;
            policy.calls = {};
            auto automatic = compile_device(native.value, kernel.function().name(), options);
            expect(static_cast<bool>(automatic)) << automatic.error;
            if (!automatic) { continue; }
            expect(policy.count == 4u);
            for (auto i = 0u; i < policy.count; i++) {
                expect(policy.calls[i][1u] == static_cast<double>(i + 1u));
                LUISA_INFO("PreparedCandidate callback metal={} vector={} index={} T={} S={} P={} Q={} rounds={} elements={} utilization={} known={} program={} waves={} kernel={}",
                           metal, vector, i, policy.calls[i][0], policy.calls[i][1], policy.calls[i][2], policy.calls[i][3],
                           policy.calls[i][4], policy.calls[i][5], policy.calls[i][6], policy.calls[i][7],
                           policy.calls[i][8], policy.calls[i][9], policy.calls[i][10]);
            }
            if (vector) { expect(static_cast<bool>(policy.calls == original_calls)); }
            else { original_calls = policy.calls; }
            expect(automatic.plans.size() == 1u);
            if (automatic.plans.size() != 1u) { continue; }
            auto &plan = automatic.plans.front();
            expect(plan.threads == 64u && plan.reduction_subgroups_per_program == 2u);
            expect(plan.striped_storage_scalars_per_worker == (vector ? 4u : 0u));
            if (vector) { expect(!plan.reduction_payload_accesses_known); }
            expect((automatic.artifact.source.find("contribution_pack") != luisa::string::npos) == vector);
            // Exact same winner must emit the retained body, ABI and resources.
            options.planner.threads_per_group = 64u;
            policy.count = 0u;
            auto exact = compile_device(native.value, kernel.function().name(), options);
            expect(static_cast<bool>(exact)) << exact.error;
            expect(policy.count == 1u);
            if (exact) {
                expect(exact.artifact.source == automatic.artifact.source);
                expect(exact.artifact.entry == automatic.artifact.entry && exact.artifact.grid == automatic.artifact.grid &&
                       exact.artifact.block == automatic.artifact.block && exact.artifact.buffer_arguments == automatic.artifact.buffer_arguments);
            }
            // Raw sections permit the same host binary's old/new-DLL logs to be compared bytewise.
            std::printf("PREPARED_SOURCE_BEGIN metal=%d vector=%d\n%s\nPREPARED_SOURCE_END\n", metal, vector, automatic.artifact.source.c_str());
        }
        if (!metal) {
            expect(set(flag, "1"));
            options.planner.threads_per_group = 0u;
            options.planner.max_reduction_striped_scalars_per_worker = 3u;
            policy.count = 0u;
            auto rejected = compile_device(native.value, kernel.function().name(), options);
            expect(!static_cast<bool>(rejected));
            expect(policy.count == 4u);// No eligible phase must not silently pick a runner-up.
            expect(rejected.error.find("no eligible independent") != luisa::string::npos) << rejected.error;
            expect(rejected.artifact.source.empty());
        }
    }
    // Exact S1 geometry, real pointer branches, and a residual scalar chunk.
    // These are source-artifact checks; runtime misalignment/tail oracles remain
    // the existing CUDA tests and retained standalone correctness receipts.
    expect(set(flag, "1"));
    for (auto mode = 0u; mode != 3u; mode++) {
        auto rows = mode == 1u ? 1 : 3;
        auto columns = mode == 0u ? 512 : 513;
        for (auto packed : {1u, 2u, 4u}) {
            if (mode != 0u && packed != 2u) { continue; }
            for (auto squeeze : {false, true}) {
                if (mode == 2u && squeeze) { continue; }
                auto definition = tile_kernel("prepared_vector_bounds", [=](TensorView<const float, 2> input, TensorView<float, 1> output) {
                    auto one = axis("one", 1), column = axis("column", columns);
                    for (auto &nest : parallel(shape(rows))) {
                        auto x = input.tile(coord(nest.index(), 0), shape(one, column)).load();
                        if (squeeze) {
                            output(coord(nest.index()), shape(one)).store(reduce(reshape(x, shape(column)), column, add));
                        } else {
                            output(coord(nest.index()), shape(one)).store(reduce(x, column, add));
                        }
                    }
                });
                auto kernel = definition.capture(tensor_shape(rows, columns), tensor_shape(rows));
                auto lowered = lower(kernel.function());
                expect(lowered.ok()) << lowered.error;
                if (!lowered) { continue; }
                auto options = cuda_subgroup_options();
                options.planner.threads_per_group = 32u * packed;
                options.planner.reduction_programs_per_group = packed;
                options.planner.reduction_lane_elements = 4u;
                options.planner.reduction_unroll_factor = 16u;
                options.planner.cache_reduction_inputs = false;
                auto result = compile_device(lowered.value, kernel.function().name(), options);
                if (mode == 2u) {
                    // Across three odd-pitch rows, a root-only aligned branch
                    // cannot prove every row's four-element pack origin.
                    expect(!static_cast<bool>(result));
                    expect(result.error.find("no eligible independent") != luisa::string::npos) << result.error;
                    expect(result.artifact.source.empty());
                    continue;
                }
                expect(static_cast<bool>(result)) << "P=" << packed << " N=" << columns << " squeeze=" << squeeze << " " << result.error;
                if (!result) { continue; }
                expect(result.plans.size() == 1u);
                if (result.plans.size() != 1u) { continue; }
                auto &plan = result.plans.front();
                expect(plan.reduction_subgroups_per_program == 1u && plan.reduction_programs_per_group == packed);
                expect(plan.striped_storage_scalars_per_worker == 4u && !plan.reduction_payload_accesses_known);
                expect(result.artifact.block[0u] == 32u * packed && result.artifact.grid[0u] == (rows + packed - 1u) / packed);
                expect(result.artifact.buffer_arguments.size() == 2u);
                auto global_loads = [](const tvm::tirx::Stmt &body, int lanes) {
                    auto count = 0u;
                    tvm::tirx::PostOrderVisit(body, [&](const tvm::ffi::ObjectRef &node) {
                        if (auto load = node.as<tvm::tirx::BufferLoadNode>(); load &&
                            (load->buffer.scope() == "global" || load->buffer.scope().empty()) &&
                            tvm::ffi::GetRef<tvm::tirx::BufferLoad>(load).ty().lanes() == lanes) { count++; }
                    });
                    return count;
                };
                auto aligned_branches = 0u;
                tvm::tirx::PostOrderVisit(result.artifact.function->body, [&](const tvm::ffi::ObjectRef &node) {
                    auto branch = node.as<tvm::tirx::IfThenElseNode>();
                    if (!branch) { return; }
                    auto pointer = false, mask15 = false;
                    tvm::tirx::PostOrderVisit(branch->condition, [&](const tvm::ffi::ObjectRef &part) {
                        if (auto call = part.as<tvm::CallNode>()) {
                            pointer |= call->op.same_as(tvm::tirx::builtin::reinterpret());
                            if (call->op.same_as(tvm::tirx::builtin::bitwise_and()) && call->args.size() == 2u) {
                                auto mask = call->args[1u].as<tvm::IntImmNode>();
                                mask15 |= mask && mask->value == 15;
                            }
                        }
                    });
                    if (!pointer || !mask15) { return; }
                    aligned_branches++;
                    expect(branch->else_case.has_value());
                    expect(global_loads(branch->then_case, 4) != 0u);
                    if (branch->else_case) { expect(global_loads(branch->else_case.value(), 1) != 0u); }
                    if (mode == 1u) { expect(global_loads(branch->then_case, 1) != 0u); }
                });
                expect(aligned_branches == 1u);
            }
        }
    }

}


void test_cuda_subgroup_target_contract() {
    auto kernel = cuda_subgroup_sum_kernel();
    expect(kernel.valid());
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }

    // Default reference behavior is not an implicit request for this family.
    CompileOptions reference;
    reference.target = cuda_target();
    reference.noalias = true;
    expect(!reference.planner.cuda_subgroup_reductions);
    auto original = compile_device(native.value, kernel.function().name(), reference);
    expect(static_cast<bool>(original)) << original.error;
    if (original) {
        expect_cuda_subgroup_source(original.artifact);
        for (auto &plan : original.plans) { expect(plan.reduction_subgroups_per_program == 0u); }
        expect(original.artifact.source.find("__luisa_tile_cuda_warp_") == luisa::string::npos);
    }

    auto options = cuda_subgroup_options();
    options.planner.threads_per_group = 64u;
    options.planner.reduction_programs_per_group = 1u;
    auto accepted = compile_device(native.value, kernel.function().name(), options);
    expect(static_cast<bool>(accepted)) << accepted.error;
    if (accepted) {
        expect_cuda_subgroup_source(accepted.artifact);
        expect(accepted.artifact.source.find("__luisa_tile_cuda_warp_sum") != luisa::string::npos);
        expect(accepted.artifact.block[0] == 64u);
        expect(accepted.artifact.grid[0] == 3u);
        expect(accepted.plans.size() == 1u);
        if (accepted.plans.size() == 1u) { expect(accepted.plans.front().reduction_operations == 1u); }
    }

    options.noalias = false;
    auto aliased = compile_device(native.value, kernel.function().name(), options);
    expect(!static_cast<bool>(aliased));
    expect(!aliased.error.empty());
    options.noalias = true;
    options.target = nvptx_target();
    auto nvptx = compile_device(native.value, kernel.function().name(), options);
    expect(!static_cast<bool>(nvptx));
    expect(!nvptx.error.empty());
    options.target = cuda_target();
    auto unsupported_module = luisa::compute::tile::bridge::tirx::compile(native.value, kernel.function().name(), options);
    expect(!unsupported_module.ok());
    expect(!unsupported_module.error().empty());
    options.planner.metal_subgroup_reductions = true;
    auto conflicting_targets = compile_device(native.value, kernel.function().name(), options);
    expect(!static_cast<bool>(conflicting_targets));
    expect(!conflicting_targets.error.empty());
    // The existing test_cuda_fail_closed_options remains unchanged, including
    // its separate rejection of metal_subgroup_reductions on a CUDA target.
    cuda_subgroup_unroll_boundaries();
    cuda_subgroup_private_index_facts();
}

template<typename T>
void cuda_subgroup_fused_artifacts_typed() {
    // rows, columns, workers/program, programs/group. Cover one inactive packed
    // program, a nondivisible tail, and the single-subgroup no-shared path.
    for (auto dimensions : {std::array<uint32_t, 4u>{1u, 33u, 64u, 2u},
                            {5u, 257u, 64u, 3u},
                            {7u, 129u, 32u, 3u}}) {
        auto [rows, columns, workers, packing] = dimensions;
        for (auto rms : {false, true}) {
            // Output is deliberately slot 1; gamma is slot 3 and slot 2 unused.
            // Device argument extraction must not assume a two-buffer or
            // output-last ABI when the repeated gamma root is forwarded.
            auto definition = tile_kernel("cuda_subgroup_fused", [=](TensorView<const T, 2> input,
                                                                    TensorView<T, 2> output,
                                                                    TensorView<const T, 2> unused,
                                                                    TensorView<const T, 2> gamma) {
                static_cast<void>(unused);
                auto one = axis("one", 1), feature = axis("feature", columns);
                for (auto &nest : parallel(shape(rows))) {
                    auto origin = coord(nest.index() + 1, 0);
                    auto x = cast<float>(input.tile(origin, shape(one, feature)).load());
                    if (rms) {
                        auto variance = reduce(x * x, feature, add) / static_cast<float>(columns);
                        auto scale = cast<float>(gamma.tile(coord(0, 0), shape(one, feature)).load());
                        output(origin, shape(one, feature)).store(cast<T>(x / sqrt(variance + 1e-5f) * scale));
                    } else {
                        auto shifted = exp(x - reduce(x, feature, maximum));
                        output(origin, shape(one, feature)).store(cast<T>(shifted / reduce(shifted, feature, add)));
                    }
                }
            });
            auto kernel = definition.capture(tensor_shape(rows + 2u, columns), tensor_shape(rows + 2u, columns),
                                             tensor_shape(1, columns), tensor_shape(1, columns));
            expect(kernel.valid());
            auto native = lower(kernel.function());
            expect(native.ok()) << native.error;
            if (!native) { continue; }
            auto options = cuda_subgroup_options();
            options.planner.threads_per_group = workers * packing;
            options.planner.reduction_programs_per_group = packing;
            options.planner.reduction_lane_elements = rows == 1u ? 8u : 4u;
            options.planner.cache_reduction_inputs = true;
            auto result = compile_device(native.value, kernel.function().name(), options);
            expect(static_cast<bool>(result)) << "rows=" << rows << " N=" << columns << " rms=" << rms << " " << result.error;
            if (!result) { continue; }
            expect_cuda_subgroup_source(result.artifact);
            expect(result.artifact.grid == (std::array<uint32_t, 3u>{ceil_div(rows, packing), 1u, 1u}));
            expect(result.artifact.block == (std::array<uint32_t, 3u>{workers * packing, 1u, 1u}));
            expect(result.plans.size() == 1u);
            if (result.plans.size() == 1u) {
                auto &plan = result.plans.front();
                auto reductions = rms ? 1u : 2u;
                auto partial_bytes = workers > 32u ? reductions * (workers / 32u) * packing * sizeof(float) : size_t{0u};
                expect(plan.reduction_operations == reductions);
                expect(plan.striped_storage_scalars_per_worker > 0u);
                expect(plan.reduction_subgroups_per_program == workers / 32u);
                expect(plan.reduction_programs_per_group == packing);
                expect(plan.reduction_threadgroups == ceil_div(rows, packing));
                expect(plan.shared_memory_bytes == partial_bytes);
                expect(plan.group_barrier_sites_after == (workers > 32u ? reductions : 0u));
            }
            std::array<bool, 4u> seen{};
            for (auto index : result.artifact.buffer_arguments) {
                expect(index < seen.size());
                if (index < seen.size()) {
                    expect(!seen[index]);
                    seen[index] = true;
                }
            }
            expect(seen[0u] && seen[1u]);
            if (rms) { expect(seen[3u]); }
        }
    }
}

void test_cuda_subgroup_fused_artifacts() {
    cuda_subgroup_fused_artifacts_typed<float>();
    cuda_subgroup_fused_artifacts_typed<luisa::half>();
    cuda_subgroup_fused_artifacts_typed<bfloat16>();
}

void test_cuda_subgroup_resource_rejections() {
    auto definition = tile_kernel("cuda_subgroup_budget", [](TensorView<const float, 2> input,
                                                             TensorView<float, 2> output) {
        auto one = axis("one", 1), feature = axis("feature", 33);
        for (auto &nest : parallel(shape(5))) {
            auto x = input.tile(coord(nest.index(), 0), shape(one, feature)).load();
            auto shifted = exp(x - reduce(x, feature, maximum));
            output(coord(nest.index(), 0), shape(one, feature)).store(shifted / reduce(shifted, feature, add));
        }
    });
    auto kernel = definition.capture(tensor_shape(5, 33), tensor_shape(5, 33));
    expect(kernel.valid());
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    for (auto mode = 0u; mode != 5u; mode++) {
        auto options = cuda_subgroup_options();
        options.planner.reduction_programs_per_group = 3u;
        options.planner.threads_per_group = 192u;
        if (mode == 0u) { options.planner.threads_per_group = 160u; }
        if (mode == 1u) { options.planner.threads_per_group = 1056u; }
        if (mode == 2u) { options.planner.reduction_lane_elements = 3u; }
        if (mode == 3u) {
            options.planner.reduction_programs_per_group = 1u;
            options.planner.threads_per_group = 32u;
            options.planner.reduction_lane_elements = 8u;
            options.planner.max_reduction_striped_scalars_per_worker = 1u;
            options.planner.reduction_unroll_factor = 64u;
        }
        if (mode == 4u) {
            options.target = R"({"kind":"cuda","thread_warp_size":32,"max_num_threads":1024,"max_shared_memory_per_block":1})";
        }
        auto result = compile_device(native.value, kernel.function().name(), options);
        expect(!static_cast<bool>(result)) << "mode=" << mode;
        expect(!result.error.empty());
    }
}

// Source/typed-IR only; this group does not execute a CUDA kernel.
void test_cuda_subgroup_terminal_row_suffix() {
    namespace ir = tvm::tirx;
    constexpr auto flag = "LUISA_DIAGNOSTIC_TIRX_ROW_ONLY_COLLECTIVE";
    auto set = [](const char *key, const char *value) noexcept {
#ifdef _WIN32
        return SetEnvironmentVariableA(key, value) != 0 || (value == nullptr && GetLastError() == ERROR_ENVVAR_NOT_FOUND);
#else
        return value ? setenv(key, value, 1) == 0 : unsetenv(key) == 0;
#endif
    };
    struct Restore {
        decltype(set) setter;
        std::array<luisa::optional<luisa::string>, 3u> old;
        ~Restore() noexcept {
            constexpr std::array keys{"LUISA_DIAGNOSTIC_TIRX_ROW_ONLY_COLLECTIVE", "LUISA_DIAGNOSTIC_TIRX_VECTOR_PACKS", "LUISA_DIAGNOSTIC_TIRX_FORWARD_COORDINATES"};
            for (size_t i = 0; i < keys.size(); i++) { LUISA_ASSERT(setter(keys[i], old[i] ? old[i]->c_str() : nullptr)); }
        }
    } restore{set, {luisa::get_environment_variable(flag), luisa::get_environment_variable("LUISA_DIAGNOSTIC_TIRX_VECTOR_PACKS"), luisa::get_environment_variable("LUISA_DIAGNOSTIC_TIRX_FORWARD_COORDINATES")}};
    expect(set("LUISA_DIAGNOSTIC_TIRX_VECTOR_PACKS", "0"));
    expect(set("LUISA_DIAGNOSTIC_TIRX_FORWARD_COORDINATES", "0"));
    auto i64 = [](int64_t n) { return tvm::IntImm::Int64(n); };
    auto f32 = [](double n) { return tvm::FloatImm{tvm::PrimType::Float(32), n}; };
    // SUM/MAX/MIN and three packed geometries use the same proof. The last
    // cases exercise an actual distributed consumer, real-memory reread, and
    // two terminal outputs; each must retain the original all-warp source.
    struct Case { int64_t kind, mode; uint32_t packed, subgroups; bool changed; };
    constexpr std::array cases{
        Case{1, 0, 2, 2, true}, Case{2, 1, 4, 2, true}, Case{3, 0, 1, 2, true},
        Case{1, 0, 2, 1, false}, Case{1, 2, 2, 2, false}, Case{1, 3, 2, 2, false}, Case{1, 4, 2, 2, false},
        Case{1, 5, 2, 2, false}, Case{1, 6, 2, 2, false}, Case{1, 7, 2, 2, false},
        Case{1, 8, 2, 2, false}, Case{1, 9, 2, 2, false}, Case{1, 10, 2, 2, false}, Case{1, 11, 2, 2, false},
        Case{1, 12, 2, 2, true}, Case{1, 13, 2, 2, false}};
    for (auto c : cases) {
        auto input = ir::decl_buffer({i64(3), i64(65)}, tvm::PrimType::Float(32), "input");
        auto narrow_type = c.mode >= 12 ? tvm::PrimType::BFloat(16) : tvm::PrimType::Float(16);
        auto output = ir::decl_buffer({i64(12)}, narrow_type, "output");
        auto carry = ir::decl_buffer({i64(1)}, tvm::PrimType::Float(32), "carry", "local");
        auto next = ir::decl_buffer({i64(1)}, tvm::PrimType::Float(32), "next", "local");
        auto copy = ir::decl_buffer({i64(1)}, tvm::PrimType::Float(32), "copy", "local");
        auto narrow = ir::decl_buffer({i64(1)}, narrow_type, "narrow", "local");
        ir::PrimVar p{"program", tvm::PrimType::Int(64)}, k{"column", tvm::PrimType::Int(64)}, e{"element", tvm::PrimType::Int(64)}, once{"once", tvm::PrimType::Int(64)};
        tvm::PrimExpr row = c.mode == 1 ? p + once - i64(7) : tvm::PrimExpr{p};
        tvm::PrimExpr left = ir::BufferLoad{carry, {i64(0)}}, right = ir::BufferLoad{input, {row, k}};
        if (c.mode == 7) {
            auto type = ir::CopyBufferType(output);
            type->elem_offset = tvm::if_then_else(left > f32(0.0), i64(0), i64(1));
            output = ir::RebuildBufferVar(output, std::move(type));
        }
        tvm::PrimExpr combine = c.kind == 1 ? left + right : c.kind == 2 ? tvm::max(left, right) : tvm::min(left, right);
        auto reduction_body = ir::SeqStmt::Flatten(tvm::ffi::Array<ir::Stmt>{
            ir::AllocBuffer{next}, ir::BufferStore{next, combine, {i64(0)}},
            ir::BufferStore{carry, ir::BufferLoad{next, {i64(0)}}, {i64(0)}}});
        auto reduction_loop = ir::For{k, i64(0), i64(65), ir::ForKind::kSerial, reduction_body, {},
            {{"luisa.tile.contract.reduction", i64(c.kind)}, {"luisa.tile.reduction_policy", i64(static_cast<int64_t>(reduction::unordered_tree))}}};
        auto elements = [&](ir::Stmt body, int64_t count = 1) {
            return ir::For{e, i64(0), i64(count), ir::ForKind::kSerial, std::move(body), {}, {{"luisa.tile.independent_elements", i64(1)}}};
        };
        auto identity = c.kind == 1 ? 0.0 : c.kind == 2 ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
        tvm::ffi::Array<ir::Stmt> statements{ir::AllocBuffer{carry}, ir::BufferStore{carry, f32(identity), {i64(0)}}, reduction_loop};
        // Unit wrapper has a nonzero min and is eliminated only along the
        // cut path. Its variable is substituted in both resulting pieces.
        if (c.mode == 1) {
            statements = {ir::For{once, i64(7), i64(1), ir::ForKind::kSerial, ir::SeqStmt::Flatten(statements)}};
        }
        statements.push_back(ir::AllocBuffer{copy});
        tvm::PrimExpr value = ir::BufferLoad{carry, {i64(0)}} * f32(0.5) + f32(1.0);
        if (c.mode == 3) { value = value + ir::BufferLoad{input, {p, i64(0)}}; }
        ir::Stmt copy_store = ir::BufferStore{copy, value, {i64(0)}};
        // Deliberately non-dominating definition: this is a proof counterexample,
        // never a numerical fixture. Later unconditional reads must not license it.
        if (c.mode == 8) { copy_store = ir::IfThenElse{p < i64(1), std::move(copy_store)}; }
        statements.push_back(std::move(copy_store));
        if (c.mode == 2) {
            statements.push_back(elements(ir::BufferStore{output, ir::Cast{tvm::PrimType::Float(16), ir::BufferLoad{copy, {i64(0)}}}, {p * i64(4) + e}}, 4));
        } else {
            statements.push_back(ir::AllocBuffer{narrow});
            tvm::PrimExpr rounded_value = ir::Cast{narrow_type, ir::BufferLoad{copy, {i64(0)}}};
            if (c.mode >= 12) {
                // Exact current lower.cpp::_round_bfloat16 expression, fed by
                // the already materialized FP32 scalar snapshot. Mode 12 is
                // the positive BF16 RNE witness, including packed row tails.
                auto u32 = tvm::PrimType::UInt(32);
                auto word = [u32](uint32_t value) { return tvm::IntImm{u32, static_cast<int64_t>(value)}; };
                auto bits = tvm::reinterpret(u32, ir::BufferLoad{copy, {i64(0)}});
                if (c.mode == 13) { bits = tvm::bitwise_xor(bits, word(1u)); }
                auto high = bits >> word(16u);
                auto odd = tvm::bitwise_and(high, word(1u));
                auto rounded = (bits + word(0x7fffu) + odd) >> word(16u);
                auto nan = tvm::greater(tvm::bitwise_and(bits, word(0x7fffffffu)), word(0x7f800000u));
                auto quiet = tvm::bitwise_or(high, word(0x40u));
                auto storage = tvm::cast(tvm::PrimType::UInt(16), tvm::if_then_else(nan, quiet, rounded));
                rounded_value = tvm::reinterpret(tvm::PrimType::BFloat(16), storage);
            }
            statements.push_back(elements(ir::BufferStore{narrow, rounded_value, {e}}));
            tvm::PrimExpr destination = p + e;
            if (c.mode == 6) { destination = p * i64(2) + tvm::if_then_else(left > f32(0.0), i64(0), i64(1)); }
            ir::Stmt output_store = ir::BufferStore{output, ir::BufferLoad{narrow, {e}}, {destination}};
            if (c.mode == 5) { output_store = ir::IfThenElse{left > f32(0.0), std::move(output_store)}; }
            statements.push_back(elements(std::move(output_store)));
            if (c.mode == 4) { statements.push_back(elements(ir::BufferStore{output, ir::BufferLoad{narrow, {e}}, {p + i64(3) + e}})); }
        }
        if (c.mode == 9) {
            statements.push_back(ir::Evaluate{tvm::Call{tvm::PrimType::Float(32), ir::builtin::call_pure_extern(),
                {ir::StringImm{"terminal_row_pointer_escape_witness"}, copy.data()}}});
        }
        if (c.mode == 11) {
            auto another = ir::decl_buffer({i64(1)}, tvm::PrimType::Float(32), "later_carry", "local");
            auto temporary = ir::decl_buffer({i64(1)}, tvm::PrimType::Float(32), "later_next", "local");
            ir::PrimVar j{"later_column", tvm::PrimType::Int(64)};
            statements.push_back(ir::AllocBuffer{another});
            statements.push_back(ir::BufferStore{another, f32(0.0), {i64(0)}});
            statements.push_back(ir::For{j, i64(0), i64(65), ir::ForKind::kSerial,
                ir::SeqStmt::Flatten(tvm::ffi::Array<ir::Stmt>{ir::AllocBuffer{temporary},
                    ir::BufferStore{temporary, ir::BufferLoad{another, {i64(0)}} + ir::BufferLoad{input, {p, j}}, {i64(0)}},
                    ir::BufferStore{another, ir::BufferLoad{temporary, {i64(0)}}, {i64(0)}}}), {},
                {{"luisa.tile.contract.reduction", i64(1)}, {"luisa.tile.reduction_policy", i64(static_cast<int64_t>(reduction::unordered_tree))}}});
        }
        ir::Stmt program = ir::SeqStmt::Flatten(statements);
        if (c.mode == 10) { program = ir::For{once, i64(0), i64(2), ir::ForKind::kSerial, std::move(program)}; }
        auto function = ir::PrimFunc{{input, output}, ir::For{p, i64(0), i64(3), ir::ForKind::kSerial,
            std::move(program), {}, {{"luisa.tile.logical_parallel", i64(1)}}}};
        auto options = cuda_subgroup_options();
        options.planner.threads_per_group = c.packed * c.subgroups * 32u;
        options.planner.reduction_programs_per_group = c.packed;
        options.planner.reduction_lane_elements = 1u;
        options.planner.reduction_unroll_factor = 1u;
        options.planner.cache_reduction_inputs = false;
        expect(set(flag, "0"));
        auto original = compile_device(function, "terminal_row_probe", options);
        expect(set(flag, "1"));
        auto selected = compile_device(function, "terminal_row_probe", options);
        // Malformed metadata/escape and repeated-fence witnesses may be rejected
        // by an earlier proof. Distinguish that from a successful row-only rejection.
        if (!original && (c.mode == 7 || c.mode == 9 || c.mode == 10)) {
            expect(!static_cast<bool>(selected));
            expect(selected.artifact.source.empty());
            LUISA_INFO("Terminal-row witness mode={} rejected by original mapper/pipeline: {}", c.mode, original.error);
            continue;
        }
        expect(static_cast<bool>(original)) << "mode=" << c.mode << " " << original.error;
        expect(static_cast<bool>(selected)) << "mode=" << c.mode << " " << selected.error;
        if (!original || !selected) { continue; }
        expect((original.artifact.source != selected.artifact.source) == c.changed) << "kind=" << c.kind << " mode=" << c.mode;
        expect(original.artifact.grid == selected.artifact.grid && original.artifact.block == selected.artifact.block &&
               original.artifact.buffer_arguments == selected.artifact.buffer_arguments);
        expect(original.plans.size() == 1u && selected.plans.size() == 1u);
        if (original.plans.size() != 1u || selected.plans.size() != 1u) { continue; }
        expect(original.plans.front().cost.kernel_score == selected.plans.front().cost.kernel_score);
        if (c.changed) { expect(!selected.plans.front().reduction_payload_accesses_known); }
        class Inspect final : public ir::StmtExprVisitor {
        private:
            tvm::PrimExpr _path{tvm::IntImm{tvm::PrimType::Bool(), 1}};
            tvm::ffi::Map<ir::Var, tvm::Expr> _bindings;
            uint32_t _depth{0u};
            void VisitStmt_(const ir::AttrStmtNode *attribute) final {
                if (attribute->attr_key == ir::attr::thread_extent) {
                    auto axis = attribute->node.as<ir::IterVar>();
                    auto extent = attribute->value.as<tvm::IntImmNode>();
                    if (!axis || !extent || extent->value <= 0) { bindings_valid = false; }
                    else if (axis.value()->thread_tag == "threadIdx.x") {
                        thread_bindings++;
                        thread_axis = axis.value()->var;
                        thread_extent = extent->value;
                    } else if (axis.value()->thread_tag == "blockIdx.x") {
                        block_bindings++;
                        block_axis = axis.value()->var;
                        block_extent = extent->value;
                    } else { bindings_valid = false; }
                }
                StmtExprVisitor::VisitStmt_(attribute);
            }
            void VisitStmt_(const ir::SeqStmtNode *sequence) final {
                auto saved = _bindings;
                StmtExprVisitor::VisitStmt_(sequence);
                _bindings = std::move(saved);
            }
            void VisitStmt_(const ir::BindNode *binding) final {
                StmtExprVisitor::VisitStmt_(binding);
                // CSE definitions belong to the finalized artifact. Expand them
                // in lexical order; never guess a variable's meaning by name.
                _bindings.Set(binding->var, ir::Substitute(binding->value, _bindings));
            }
            void VisitStmt_(const ir::IfThenElseNode *branch) final {
                VisitExpr(branch->condition);
                auto saved = _path;
                auto saved_bindings = _bindings;
                auto condition = ir::Substitute(branch->condition, _bindings);
                _depth++;
                _path = saved && condition;
                VisitStmt(branch->then_case);
                _bindings = saved_bindings;
                if (branch->else_case) { _path = saved && !condition; VisitStmt(branch->else_case.value()); }
                _bindings = std::move(saved_bindings);
                _path = saved;
                _depth--;
            }
            void VisitStmt_(const ir::BufferStoreNode *store) final {
                if (store->buffer.scope() == "global") { stores.push_back(_path); }
                StmtExprVisitor::VisitStmt_(store);
            }
            void VisitExpr_(const tvm::CallNode *call) final {
                if (call->op.same_as(ir::builtin::tvm_storage_sync())) { fences++; uniform &= _depth == 0u; }
                if (call->op.same_as(ir::builtin::call_pure_extern()) && !call->args.empty()) {
                    auto name = call->args[0].as<ir::StringImmNode>();
                    if (name && std::string_view{name->value.data(), name->value.size()}.starts_with("__luisa_tile_cuda_warp_")) { collectives.push_back(_path); }
                }
                StmtExprVisitor::VisitExpr_(call);
            }
        public:
            uint32_t fences{0u};
            bool uniform{true};
            bool bindings_valid{true};
            uint32_t thread_bindings{0u}, block_bindings{0u};
            int64_t thread_extent{0}, block_extent{0};
            tvm::ffi::Optional<ir::PrimVar> thread_axis, block_axis;
            std::vector<tvm::PrimExpr> collectives, stores;
        } observed;
        observed(selected.artifact.function->body);
        auto reductions = c.mode == 11 ? 2u : 1u;
        expect(observed.fences == (c.subgroups == 1u ? 0u : reductions) && observed.uniform);
        expect(observed.collectives.size() == (c.subgroups == 1u ? 1u : 2u) * reductions);
        // LowerIntrin has already turned integer AND/shift into builtin Calls.
        // The arithmetic Analyzer does not fold every such literal Call. Fold
        // only the nonnegative scalar integer cases used by physical indices;
        // unknown operations/types and invalid shifts remain unproved.
        class FoldPhysicalIndexCalls final : public ir::ExprMutator {
        private:
            [[nodiscard]] static bool supported(const tvm::IntImmNode *value) noexcept {
                if (!value || value->value < 0) { return false; }
                auto type = value->ty.as<tvm::PrimType>();
                if (!type || !type.value().IsScalar() ||
                    !type.value().MatchesCode(kDLInt, kDLUInt) ||
                    (type.value().bits() != 32 && type.value().bits() != 64)) { return false; }
                auto bits = type.value().bits();
                auto maximum = bits == 64 ? static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) :
                               type.value().code() == kDLInt ? uint64_t{0x7fffffff} : uint64_t{0xffffffff};
                return static_cast<uint64_t>(value->value) <= maximum;
            }
            tvm::Expr VisitExpr_(const tvm::CallNode *original) final {
                auto expression = ExprMutator::VisitExpr_(original);
                auto call = expression.as<tvm::CallNode>();
                if (!call || call->args.size() != 2u) { return expression; }
                auto left = call->args[0].as<tvm::IntImmNode>();
                auto right = call->args[1].as<tvm::IntImmNode>();
                auto type = call->ty.as<tvm::PrimType>();
                if (!supported(left) || !supported(right) || !type || !type.value().IsScalar()) { return expression; }
                auto left_type = left->ty.as<tvm::PrimType>();
                auto right_type = right->ty.as<tvm::PrimType>();
                if (type.value().code() != left_type.value().code() || type.value().bits() != left_type.value().bits()) { return expression; }
                auto a = static_cast<uint64_t>(left->value);
                auto b = static_cast<uint64_t>(right->value);
                if (call->op.same_as(ir::builtin::bitwise_and()) &&
                    type.value().code() == right_type.value().code() && type.value().bits() == right_type.value().bits()) {
                    return tvm::IntImm{type.value(), static_cast<int64_t>(a & b)};
                }
                if (call->op.same_as(ir::builtin::shift_right()) && b < static_cast<uint64_t>(type.value().bits())) {
                    // Nonnegative signed and unsigned inputs have the same
                    // right-shift result; no signed shift or overflow in C++.
                    return tvm::IntImm{type.value(), static_cast<int64_t>(a >> b)};
                }
                return expression;
            }
        } fold_physical_indices;
        auto truth_at = [&](const tvm::PrimExpr &predicate, uint32_t thread, uint32_t block) {
            tvm::ffi::Map<ir::Var, tvm::Expr> substitutions;
            auto known = true;
            ir::PostOrderVisit(predicate, [&](const tvm::ffi::ObjectRef &node) {
                if (auto var = node.as<ir::VarNode>()) {
                    auto type = var->ty.as<tvm::PrimType>();
                    if (!type) { known = false; return; }
                    if (observed.thread_axis && observed.thread_axis.value().get() == var) {
                        substitutions.Set(tvm::ffi::GetRef<ir::Var>(var), tvm::IntImm{type.value(), thread});
                    } else if (observed.block_axis && observed.block_axis.value().get() == var) {
                        substitutions.Set(tvm::ffi::GetRef<ir::Var>(var), tvm::IntImm{type.value(), block});
                    } else { known = false; }
                }
            });
            tvm::arith::Analyzer analyzer;
            auto folded = fold_physical_indices(ir::Substitute(predicate, substitutions)).as<tvm::PrimExpr>();
            expect(folded.has_value());
            if (!folded) { return false; }
            auto result = analyzer->Simplify(folded.value());
            auto literal = result.as<tvm::IntImmNode>();
            expect(known && literal);
            return known && literal && literal->value != 0;
        };
        if (c.changed && observed.collectives.size() == 2u) {
            expect(observed.bindings_valid && observed.thread_bindings == 1u &&
                   observed.thread_extent == selected.artifact.block[0]);
            // A unit block binding may be removed by the normal simplifier.
            expect((observed.block_bindings == 1u && observed.block_extent == selected.artifact.grid[0]) ||
                   (observed.block_bindings == 0u && selected.artifact.grid[0] == 1u));
            expect(observed.stores.size() == 1u);
            // Enumerate real physical threads AND blocks, including inactive
            // packed programs. First tree/barrier stay collective; second tree
            // selects a full warp per program; external output remains worker0
            // and logical-program-active, rather than every selected lane.
            for (auto b = 0u; b < selected.artifact.grid[0]; b++) {
                for (auto t = 0u; t < selected.artifact.block[0]; t++) {
                    auto worker = t % (c.subgroups * 32u);
                    auto program = b * c.packed + t / (c.subgroups * 32u);
                    expect(truth_at(observed.collectives[0], t, b));
                    expect(truth_at(observed.collectives[1], t, b) == (worker < 32u));
                    for (auto &&predicate : observed.stores) { expect(truth_at(predicate, t, b) == (worker == 0u && program < 3u)); }
                }
            }
        }
        if (c.mode == 0 && c.kind == 1 && c.subgroups == 2) {
            expect(set(flag, nullptr));
            auto unset = compile_device(function, "terminal_row_probe", options);
            expect(static_cast<bool>(unset));
            if (unset) { expect(unset.artifact.source == original.artifact.source); }
            expect(set(flag, "yes"));
            auto invalid = compile_device(function, "terminal_row_probe", options);
            expect(!static_cast<bool>(invalid));
            expect(invalid.error.find("exactly 0 or 1") != luisa::string::npos);
        }
    }
}


void test_cuda_subgroup_packing_proofs() {
    auto i64 = [](int64_t value) { return tvm::IntImm::Int64(value); };
    auto f32 = [](float value) { return tvm::FloatImm{tvm::PrimType::Float(32), value}; };
    for (auto mode = 0u; mode != 11u; mode++) {
        auto input = tvm::tirx::decl_buffer({i64(5), i64(257)}, tvm::PrimType::Float(32), "input");
        auto output = tvm::tirx::decl_buffer({i64(5)}, tvm::PrimType::Float(32), "output");
        auto carry = tvm::tirx::decl_buffer({i64(1)}, tvm::PrimType::Float(32), "carry", "local");
        auto temporary = tvm::tirx::decl_buffer({i64(1)}, tvm::PrimType::Float(32), "next", "local");
        auto p = tvm::tirx::PrimVar{"program", tvm::PrimType::Int(64)};
        auto k = tvm::tirx::PrimVar{"reduction", tvm::PrimType::Int(64)};
        auto e = tvm::tirx::PrimVar{"output_element", tvm::PrimType::Int(64)};
        auto step = tvm::tirx::PrimVar{"step", tvm::PrimType::Int(64)};
        auto update = tvm::tirx::SeqStmt::Flatten(tvm::ffi::Array<tvm::tirx::Stmt>{
            tvm::tirx::AllocBuffer{temporary},
            tvm::tirx::BufferStore{temporary, tvm::tirx::BufferLoad{carry, {i64(0)}} + tvm::tirx::BufferLoad{input, {p - i64(2), k}}, {i64(0)}},
            tvm::tirx::BufferStore{carry, tvm::tirx::BufferLoad{temporary, {i64(0)}}, {i64(0)}}});
        auto reduction = tvm::tirx::For{k, i64(0), i64(257), tvm::tirx::ForKind::kSerial, std::move(update), {}, {{"luisa.tile.contract.reduction", i64(1)}}};
        // Body/identity provenance alone is not numerical permission. Keep
        // the original packing counterexamples and independently test absent
        // permission, ordered trees, and both explicit fold directions.
        if (mode != 6u) {
            auto policy = mode == 7u ? reduction::ordered_tree :
                          mode == 8u ? reduction::fold_left :
                          mode == 9u ? reduction::fold_right :
                                       reduction::unordered_tree;
            reduction.CopyOnWrite()->annotations.Set("luisa.tile.reduction_policy", i64(static_cast<int64_t>(policy)));
        }
        auto destination = mode == 4u ? input : output;
        auto indices = mode == 4u ? tvm::ffi::Array<tvm::PrimExpr>{p - i64(2), e} : tvm::ffi::Array<tvm::PrimExpr>{p - i64(2) + e};
        auto store = tvm::tirx::For{e, i64(0), i64(1), tvm::tirx::ForKind::kSerial, tvm::tirx::BufferStore{destination, tvm::tirx::BufferLoad{carry, {i64(0)}}, indices}, {}, {{"luisa.tile.independent_elements", i64(1)}}};
        tvm::tirx::Stmt body = tvm::tirx::SeqStmt::Flatten(tvm::ffi::Array<tvm::tirx::Stmt>{
            tvm::tirx::AllocBuffer{carry}, tvm::tirx::BufferStore{carry, f32(0.0f), {i64(0)}}, reduction, store});
        if (mode >= 1u && mode <= 3u) {
            // A unit wrapper is safe. Repeated partial reuse and row-varying
            // fence counts are distinct proof failures, even without a tail.
            tvm::PrimExpr count = mode == 3u ? p - i64(1) : i64(mode);
            body = tvm::tirx::For{step, i64(0), count, tvm::tirx::ForKind::kSerial, std::move(body)};
        }
        // Multiple subgroups also need phase-reuse proof when P is one.
        if (mode == 10u) { body = tvm::tirx::For{step, i64(0), i64(2), tvm::tirx::ForKind::kSerial, std::move(body)}; }
        body = tvm::tirx::For{p, i64(2), i64(5), tvm::tirx::ForKind::kSerial, std::move(body), {}, {{"luisa.tile.logical_parallel", i64(1)}}};
        CompileOptions options;
        options.target = cuda_target();
        options.noalias = true;
        options.planner.cuda_subgroup_reductions = true;
        options.planner.reduction_programs_per_group = mode == 10u ? 1u : 3u;
        options.planner.threads_per_group = mode == 10u ? 64u : mode == 5u ? 1056u : 288u;
        auto compiled = compile_device(tvm::tirx::PrimFunc{{input, output}, body}, "packing_proof", options);
        expect(eq(static_cast<bool>(compiled), mode <= 1u)) << "mode=" << mode << " " << compiled.error;
        if (!compiled) { continue; }
        class FenceAudit final : public tvm::tirx::StmtExprVisitor {
        private:
            uint32_t _branch_depth{0u};
            void VisitStmt_(const tvm::tirx::IfThenElseNode *branch) final {
                _branch_depth++;
                StmtExprVisitor::VisitStmt_(branch);
                _branch_depth--;
            }
            void VisitExpr_(const tvm::CallNode *call) final {
                if (call->op.same_as(tvm::tirx::builtin::tvm_storage_sync())) {
                    fences++;
                    uniform &= _branch_depth == 0u;
                }
                StmtExprVisitor::VisitExpr_(call);
            }
        public:
            uint32_t fences{0u};
            bool uniform{true};
        } audit;
        audit(compiled.artifact.function->body);
        expect(eq(audit.fences, 1u));
        expect(audit.uniform);
        expect(eq(compiled.artifact.block[0u], 288u));
        expect(eq(compiled.artifact.grid[0u], 2u));
    }
}

// These raw-IR tests exercise the private proof directly, before downstream
// simplification can hide a rejected producer or change materialization.
namespace coordinate_tests {
namespace ir = tvm::tirx;
namespace proof = luisa::compute::tile::bridge::tirx::detail;

auto i64(int64_t value) { return tvm::IntImm::Int64(value); }
auto f32(double value) { return tvm::FloatImm{tvm::PrimType::Float(32), value}; }

ir::For elements(ir::PrimVar axis, int64_t extent, ir::Stmt body, int64_t rank = 1) {
    return ir::For{axis, i64(0), i64(extent), ir::ForKind::kSerial, std::move(body), {},
                   {{proof::independent_elements_annotation, i64(rank)}}};
}

struct Uses final : ir::StmtExprVisitor {
    ir::BufferVar buffer;
    uint32_t allocations{0u}, reads{0u}, writes{0u};
    tvm::PrimExpr last_value;
    explicit Uses(ir::BufferVar value) : buffer{std::move(value)} {}
    void VisitStmt_(const ir::AllocBufferNode *op) final {
        allocations += op->buffer.same_as(buffer);
        StmtExprVisitor::VisitStmt_(op);
    }
    void VisitExpr_(const ir::BufferLoadNode *op) final {
        reads += op->buffer.same_as(buffer);
        StmtExprVisitor::VisitExpr_(op);
    }
    void VisitStmt_(const ir::BufferStoreNode *op) final {
        if (op->buffer.same_as(buffer)) { writes++; last_value = op->value; }
        StmtExprVisitor::VisitStmt_(op);
    }
};

struct Chain {
    ir::PrimVar p{"program", tvm::PrimType::Int(64)}, a{"iota_axis", tvm::PrimType::Int(64)},
        b{"mask_axis", tvm::PrimType::Int(64)}, row{"element_row", tvm::PrimType::Int(64)},
        n{"element_column", tvm::PrimType::Int(64)};
    ir::BufferVar x{ir::decl_buffer({i64(8)}, tvm::PrimType::Float(32), "input")},
        y{ir::decl_buffer({i64(3), i64(2), i64(8)}, tvm::PrimType::Float(32), "output")},
        index{ir::decl_buffer({i64(8)}, tvm::PrimType::Int(64), "coordinate", "local")},
        mask{ir::decl_buffer({i64(8)}, tvm::PrimType::Bool(), "mask", "local")};

    ir::Stmt producers(tvm::PrimExpr predicate) const {
        auto iota = elements(a, 8, ir::BufferStore{index, a, {a}});
        auto compare = elements(b, 8, ir::BufferStore{mask, predicate, {b}});
        compare.CopyOnWrite()->annotations.Set(proof::materialized_pure_tile_annotation, i64(1));
        return ir::SeqStmt::Flatten(tvm::ffi::Array<ir::Stmt>{ir::AllocBuffer{index}, ir::AllocBuffer{mask}, iota, compare});
    }
    ir::PrimFunc function(tvm::PrimExpr predicate, tvm::PrimExpr value) const {
        auto consumer = elements(row, 2, ir::For{n, i64(0), i64(8), ir::ForKind::kSerial,
            ir::BufferStore{y, value, {p, row, n}}}, 2);
        auto body = ir::SeqStmt::Flatten(tvm::ffi::Array<ir::Stmt>{producers(predicate), consumer});
        body = ir::For{p, i64(0), i64(3), ir::ForKind::kSerial, body, {}, {{proof::logical_parallel_annotation, i64(1)}}};
        return ir::PrimFunc{{x, y}, body};
    }
};

void fixed_point_and_math() {
    for (auto mode : {0u, 1u, 2u, 3u}) {
        Chain c;
        // Keep this lazy floating-point expression exactly, including operand
        // order and the potential division. Forwarding is independent of the
        // later compiler fast-math choice and may not rewrite either arm.
        auto load = ir::BufferLoad{c.x, {c.n}};
        auto arithmetic = ir::Div{ir::Add{load, f32(-0.0)}, ir::Mul{load, f32(2.0)}};
        auto floating_lazy = tvm::if_then_else(load > f32(0.0), arithmetic, f32(-7.0));
        auto predicate = mode == 3u ? c.p < i64(3) : ir::BufferLoad{c.index, {c.b}} < i64(mode == 0u ? 8 : mode == 1u ? 0 : 7);
        // The root-dependent mode still reads iota so both definitions have
        // a real use; the root bound must not become an element-domain fact.
        if (mode == 3u) { predicate = predicate && (ir::BufferLoad{c.index, {c.b}} < i64(8)); }
        auto value = tvm::if_then_else(ir::BufferLoad{c.mask, {c.n}}, floating_lazy, f32(19.0));
        auto function = c.function(predicate, value);
        auto original = function->body;
        uint64_t removed = 0u;
        auto body = proof::forward_coordinate_tiles(function, removed);
        expect(eq(removed, uint64_t{2u})) << mode;
        Uses index{c.index}, mask{c.mask}, output{c.y};
        index(body); mask(body); output(body);
        expect(eq(index.allocations + index.reads + index.writes, 0u));
        expect(eq(mask.allocations + mask.reads + mask.writes, 0u));
        expect(eq(output.writes, 1u));
        if (mode == 0u) { expect(tvm::ffi::StructuralEqual{}(output.last_value, floating_lazy)); }
        if (mode == 1u) { expect(tvm::ffi::StructuralEqual{}(output.last_value, f32(19.0))); }
        if (mode >= 2u) {
            auto call = output.last_value.as<tvm::CallNode>();
            expect(call != nullptr);
            if (call) {
                expect(call->op.same_as(ir::builtin::if_then_else()));
                expect(call->args[0].as<tvm::IntImmNode>() == nullptr);
                expect(tvm::ffi::StructuralEqual{}(call->args[1], floating_lazy));
                expect(tvm::ffi::StructuralEqual{}(call->args[2], f32(19.0)));
            }
        }
        expect(function->body.same_as(original));
        auto again = function;
        again.CopyOnWrite()->body = body;
        auto unchanged = proof::forward_coordinate_tiles(again, removed);
        expect(eq(removed, uint64_t{0u}));
        expect(tvm::ffi::StructuralEqual{}(unchanged, body));
    }
    // A singleton axis projects through literal zero; no whole-domain rank
    // equality or invented axis is needed at its rank-two consumer.
    auto singleton = ir::decl_buffer({i64(1)}, tvm::PrimType::Int(64), "singleton", "local");
    auto out = ir::decl_buffer({i64(2), i64(8)}, tvm::PrimType::Int(64), "out");
    ir::PrimVar a{"a", tvm::PrimType::Int(64)}, r{"r", tvm::PrimType::Int(64)}, n{"n", tvm::PrimType::Int(64)};
    auto body = ir::SeqStmt::Flatten(tvm::ffi::Array<ir::Stmt>{ir::AllocBuffer{singleton},
        elements(a, 1, ir::BufferStore{singleton, a, {a}}),
        elements(r, 2, ir::For{n, i64(0), i64(8), ir::ForKind::kSerial,
            ir::BufferStore{out, ir::BufferLoad{singleton, {i64(0)}}, {r, n}}}, 2)});
    uint64_t removed = 0u;
    auto forwarded = proof::forward_coordinate_tiles(ir::PrimFunc{{out}, body}, removed);
    expect(eq(removed, uint64_t{1u}));
    Uses output{out}; output(forwarded);
    expect(tvm::ffi::StructuralEqual{}(output.last_value, i64(0)));
}

void rejection_boundaries() {
    enum class Bad { EFFECT, ESCAPE, BEFORE_PRODUCER, NEIGHBOR, MEMORY_VALUE, FLOAT_VALUE, MANUAL, TWO_WRITERS };
    for (auto mode : {Bad::EFFECT, Bad::ESCAPE, Bad::BEFORE_PRODUCER, Bad::NEIGHBOR,
                      Bad::MEMORY_VALUE, Bad::FLOAT_VALUE, Bad::MANUAL, Bad::TWO_WRITERS}) {
        ir::PrimVar a{"a", tvm::PrimType::Int(64)}, n{"n", tvm::PrimType::Int(64)};
        auto input = ir::decl_buffer({i64(8)}, tvm::PrimType::Int(64), "input");
        auto floats = ir::decl_buffer({i64(8)}, tvm::PrimType::Float(32), "floats");
        auto output = ir::decl_buffer({i64(8)}, tvm::PrimType::Int(64), "output");
        auto local = ir::decl_buffer({i64(8)}, tvm::PrimType::Int(64), "coordinate", "local");
        tvm::PrimExpr value = a;
        if (mode == Bad::MEMORY_VALUE) { value = ir::BufferLoad{input, {a}}; }
        if (mode == Bad::FLOAT_VALUE) { value = ir::Cast{tvm::PrimType::Int(64), ir::Cast{tvm::PrimType::Float(32), a}}; }
        auto allocation = ir::AllocBuffer{local};
        if (mode == Bad::MANUAL) { allocation.CopyOnWrite()->annotations.Set(proof::manual_memory_annotation, i64(1)); }
        auto producer = elements(a, 8, ir::BufferStore{local, value, {a}});
        tvm::PrimExpr index = mode == Bad::NEIGHBOR ? tvm::floormod(n + i64(1), i64(8)) : n;
        auto read = ir::BufferLoad{local, {index}};
        auto consumer = elements(n, 8, ir::BufferStore{output, read, {n}});
        tvm::ffi::Array<ir::Stmt> parts{allocation};
        // The exact same read Expr occurs on both sides of the producer.
        if (mode == Bad::BEFORE_PRODUCER) { parts.push_back(consumer); }
        parts.push_back(producer);
        if (mode == Bad::TWO_WRITERS) { parts.push_back(producer); }
        if (mode == Bad::EFFECT) {
            parts.push_back(ir::Evaluate{tvm::Call{tvm::PrimType::Int(32), ir::builtin::call_extern(), {ir::StringImm{"unknown_effect"}}}});
        }
        if (mode == Bad::ESCAPE) {
            auto pointer = tvm::Call{local.DataPointerType(), ir::builtin::address_of(), {ir::BufferLoad{local, {i64(0)}}}};
            parts.push_back(ir::Evaluate{tvm::reinterpret(tvm::Type{tvm::PrimType::UInt(64)}, pointer)});
        }
        parts.push_back(consumer);
        auto function = ir::PrimFunc{{input, floats, output}, ir::SeqStmt::Flatten(parts)};
        uint64_t removed = 0u;
        auto body = proof::forward_coordinate_tiles(function, removed);
        expect(eq(removed, uint64_t{0u})) << static_cast<uint32_t>(mode);
        expect(tvm::ffi::StructuralEqual{}(body, function->body));
    }
}

void snapshot_order() {
    Chain c;
    auto snapshot = ir::decl_buffer({i64(8)}, tvm::PrimType::Float(32), "float_snapshot", "local");
    auto y0 = ir::decl_buffer({i64(8)}, tvm::PrimType::Float(32), "y0");
    auto y1 = ir::decl_buffer({i64(8)}, tvm::PrimType::Float(32), "y1");
    ir::PrimVar copy{"copy", tvm::PrimType::Int(64)}, first{"first", tvm::PrimType::Int(64)}, second{"second", tvm::PrimType::Int(64)};
    auto load = tvm::if_then_else(ir::BufferLoad{c.mask, {copy}}, ir::BufferLoad{c.x, {copy}}, f32(0.0));
    auto body = ir::SeqStmt::Flatten(tvm::ffi::Array<ir::Stmt>{
        c.producers(ir::BufferLoad{c.index, {c.b}} < i64(8)), ir::AllocBuffer{snapshot},
        elements(copy, 8, ir::BufferStore{snapshot, load, {copy}}),
        elements(first, 8, ir::BufferStore{y0, ir::Add{ir::BufferLoad{snapshot, {first}}, f32(1.0)}, {first}}),
        elements(second, 8, ir::BufferStore{y1, ir::Mul{ir::BufferLoad{snapshot, {second}}, f32(2.0)}, {second}})});
    auto function = ir::PrimFunc{{c.x, y0, y1}, body};
    auto readonly = proof::forward_readonly_tile_loads(function, true, true, false, true);
    auto post = function;
    post.CopyOnWrite()->body = readonly.body;
    uint64_t removed = 0u;
    post.CopyOnWrite()->body = proof::forward_coordinate_tiles(post, removed);
    expect(eq(removed, uint64_t{2u}));
    Uses kept{snapshot}, once{c.x}; kept(post->body); once(post->body);
    expect(eq(kept.allocations, 1u));
    expect(eq(kept.reads, 2u));
    expect(eq(kept.writes, 1u));
    expect(eq(once.reads, 1u));
    expect(readonly.inputs.empty());
    // Counter-witness: the former order licenses the existing immutable-view
    // transform after the mask disappears. It is legal under noalias, but it
    // changes snapshot choice and is not coordinate-only work removal.
    auto pre = function;
    pre.CopyOnWrite()->body = proof::forward_coordinate_tiles(pre, removed);
    auto replay = proof::forward_readonly_tile_loads(pre, true, true, false, true);
    Uses gone{snapshot}, twice{c.x}; gone(replay.body); twice(replay.body);
    expect(eq(gone.allocations + gone.reads + gone.writes, 0u));
    expect(eq(twice.reads, 2u));
    expect(eq(replay.inputs.size(), size_t{1u}));
    expect(function->body.same_as(body));
}
}// namespace coordinate_tests

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_tirx_coordinate_fixed_point_and_math"_test = coordinate_tests::fixed_point_and_math;
    "tile_tirx_coordinate_rejection_boundaries"_test = coordinate_tests::rejection_boundaries;
    "tile_tirx_coordinate_snapshot_order"_test = coordinate_tests::snapshot_order;
    "tile_tirx_cuda_elementwise_artifact"_test = test_cuda_elementwise_artifact;
    "tile_tirx_cuda_reduction_artifact"_test = test_cuda_reduction_artifact;
    "tile_tirx_cuda_matmul_artifact"_test = test_cuda_matmul_artifact;
    "tile_tirx_cuda_nvptx_elementwise_artifact"_test = test_cuda_nvptx_elementwise_artifact;
    "tile_tirx_cuda_nvptx_reduction_artifact_warp_aligned"_test = test_cuda_nvptx_reduction_artifact_warp_aligned;
    "tile_tirx_cuda_ragged_reordered_gemm_artifact"_test = test_cuda_ragged_reordered_gemm_artifact;
    "tile_tirx_cuda_permutation_unroll_budget"_test = test_cuda_permutation_unroll_budget;
    "tile_tirx_cuda_fail_closed_options"_test = test_cuda_fail_closed_options;
    "tile_tirx_cuda_subgroup_target_contract"_test = test_cuda_subgroup_target_contract;
    "tile_tirx_cuda_subgroup_integer_extrema"_test = test_cuda_subgroup_integer_extrema;
    "tile_tirx_prepared_reduction_candidate"_test = test_prepared_reduction_candidate;
    "tile_tirx_cuda_subgroup_fused_artifacts"_test = test_cuda_subgroup_fused_artifacts;
    "tile_tirx_cuda_subgroup_resource_rejections"_test = test_cuda_subgroup_resource_rejections;
    "tile_tirx_cuda_subgroup_packing_proofs"_test = test_cuda_subgroup_packing_proofs;
    "tile_tirx_cuda_subgroup_terminal_row_suffix"_test = test_cuda_subgroup_terminal_row_suffix;
}
