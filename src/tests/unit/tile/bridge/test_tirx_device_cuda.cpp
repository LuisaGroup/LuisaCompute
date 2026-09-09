// Host-side CUDA device-artifact tests for the shared TIRx bridge. These do
// not need a CUDA GPU or TVMx CUDA runtime: compile_device() stops at the
// CUDA C / NVPTX source artifact, which the CUDA backend later feeds to the
// standalone-NVRTC pipeline. Requires the pinned TVMx build to provide the
// "cuda" (CUDA C) and optionally "nvptx" code generators.

#include "ut/ut.hpp"

#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>

#include <luisa/core/stl/string.h>
#include <luisa/tile/bridge/tirx/compiler.h>
#include <luisa/tile/bridge/tirx/lower.h>
#include <luisa/tile/dsl.h>

#include <array>
#include <cstdint>
#include <string_view>

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
    auto source = std::string_view{artifact.source.data(), artifact.source.size()};
    expect(source.find("metal.cooperative_tensor") == std::string_view::npos);
    expect(source.find("cooperative_tensor") == std::string_view::npos);
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
    auto source = std::string_view{result.artifact.source.data(), result.artifact.source.size()};
    expect(source.find("simdgroup") == std::string_view::npos);
    expect(source.find("metal.") == std::string_view::npos);
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
    auto source = std::string_view{artifact.source.data(), artifact.source.size()};
    // Semantic MMA stays reference-expanded for CUDA: no cooperative tensor,
    // WMMA/mma.sync, MPP fragment or Metal scope may leak into the artifact.
    expect(source.find("cooperative_tensor") == std::string_view::npos);
    expect(source.find("metal.") == std::string_view::npos);
    expect(source.find("fragment") == std::string_view::npos);
}

[[nodiscard]] constexpr luisa::string_view nvptx_target() noexcept {
    return R"({"kind":"nvptx","thread_warp_size":32,"max_num_threads":1024,"max_shared_memory_per_block":98304})";
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
    auto source = std::string_view{artifact.source.data(), artifact.source.size()};
    expect(source.find(".target") != std::string_view::npos);
    expect(source.find("metal.") == std::string_view::npos);
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
    auto source = std::string_view{artifact.source.data(), artifact.source.size()};
    expect(source.find("simdgroup") == std::string_view::npos);
    expect(source.find("metal.") == std::string_view::npos);
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
    auto source = std::string_view{artifact.source.data(), artifact.source.size()};
    expect(source.find("cooperative_tensor") == std::string_view::npos);
    expect(source.find("simdgroup") == std::string_view::npos);
    expect(source.find("metal.") == std::string_view::npos);
    expect(source.find("fragment") == std::string_view::npos);
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
    expect(std::string_view{result.error.data(), result.error.size()}.find("cooperative") != std::string_view::npos)
        << result.error;

    CompileOptions mpp;
    mpp.target = cuda_target();
    mpp.metal_mpp = true;
    result = compile_device(native.value, kernel.function().name(), mpp);
    expect(!static_cast<bool>(result));
    expect(std::string_view{result.error.data(), result.error.size()}.find("MPP") != std::string_view::npos)
        << result.error;

    CompileOptions subgroup;
    subgroup.target = cuda_target();
    subgroup.planner.metal_subgroup_reductions = true;
    result = compile_device(native.value, kernel.function().name(), subgroup);
    expect(!static_cast<bool>(result));
    expect(std::string_view{result.error.data(), result.error.size()}.find("subgroup") != std::string_view::npos ||
           std::string_view{result.error.data(), result.error.size()}.find("SIMD") != std::string_view::npos)
        << result.error;

    CompileOptions unknown_target;
    unknown_target.target = "vulkan";
    result = compile_device(native.value, kernel.function().name(), unknown_target);
    expect(!static_cast<bool>(result));
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
    "tile_tirx_cuda_fail_closed_options"_test = test_cuda_fail_closed_options;
}
