// Host-side CUDA device-artifact tests for the shared TIRx bridge. These do
// not need a CUDA GPU or TVMx CUDA runtime: compile_device() stops at the
// CUDA C / NVPTX source artifact, which the CUDA backend later feeds to the
// standalone-NVRTC pipeline. Requires the pinned TVMx build to provide the
// "cuda" (CUDA C) and optionally "nvptx" code generators.

#include "ut/ut.hpp"

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

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
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
    "tile_tirx_cuda_subgroup_fused_artifacts"_test = test_cuda_subgroup_fused_artifacts;
    "tile_tirx_cuda_subgroup_resource_rejections"_test = test_cuda_subgroup_resource_rejections;
    "tile_tirx_cuda_subgroup_packing_proofs"_test = test_cuda_subgroup_packing_proofs;
}
