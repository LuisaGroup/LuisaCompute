// Cross-target execution tests for one portable Tile DSL capture.

#include "ut/ut.hpp"

#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/tensor.h>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/tile/bridge/tirx/compiler.h>
#include <luisa/tile/bridge/tirx/lower.h>
#include <luisa/tile/dsl.h>

#include <cmath>
#include <cstdint>

using namespace luisa;
using namespace luisa::compute::tile;
using namespace luisa::compute::tile::bridge::tirx;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct Executable {
    tvm::ffi::Optional<tvm::ffi::Module> module;
    tvm::ffi::Optional<tvm::ffi::Function> entry;
    luisa::string error;

    [[nodiscard]] bool ok() const noexcept {
        return error.empty() && module.has_value() && entry.has_value();
    }
};

[[nodiscard]] Executable build(
    const tvm::tirx::PrimFunc &function,
    luisa::string_view name,
    luisa::string_view target) {
    Executable result;
    CompileOptions options;
    options.target.assign(target.data(), target.size());
    options.host = "llvm";
    auto compilation = compile(function, name, options);
    if (!compilation) {
        result.error = luisa::string{compilation.error()};
        return result;
    }
    result.module = compilation.module();
    result.entry = result.module.value()->GetFunction(
        tvm::ffi::String{name.data(), name.size()}, true);
    if (!result.entry) { result.error = "compiled module has no requested entry function"; }
    return result;
}

[[nodiscard]] tvm::runtime::Tensor allocate(
    std::initializer_list<int64_t> shape,
    tvm::Device device) {
    return tvm::runtime::Tensor::Empty(
        tvm::ffi::Shape{shape},
        DLDataType{kDLFloat, 32, 1},
        device);
}

[[nodiscard]] tvm::runtime::Tensor upload(
    std::initializer_list<int64_t> shape,
    const luisa::vector<float> &values,
    tvm::Device device) {
    auto host = allocate(shape, tvm::Device{kDLCPU, 0});
    host.CopyFromBytes(values.data(), values.size() * sizeof(float));
    return device.device_type == kDLCPU ? host : host.CopyTo(device);
}

[[nodiscard]] luisa::vector<float> download(
    const tvm::runtime::Tensor &tensor,
    size_t count) {
    luisa::vector<float> values(count);
    tensor.CopyToBytes(values.data(), values.size() * sizeof(float));
    return values;
}

[[nodiscard]] bool close(float lhs, float rhs) noexcept {
    return std::abs(lhs - rhs) <= 1e-5f * std::max(1.0f, std::abs(rhs));
}

void test_same_axpy_on_cpu_and_metal() {
    constexpr int64_t n = 1003;
    auto definition = tile_kernel(
        "tile_tirx_dual_axpy",
        [](TensorView<const float, 1> x,
           TensorView<const float, 1> y,
           TensorView<float, 1> result) {
            auto element = axis("element", result.extent<0>());
            for (auto &item : parallel(shape(element))) {
                auto index = item.index();
                result(index).store(1.25f * x(index).load() - 0.75f * y(index).load() + 0.5f);
            }
        });
    auto kernel = definition.capture(
        tensor_shape("x", n), tensor_shape("y", n), tensor_shape("result", n));
    expect(kernel.valid());
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }

    auto cpu = build(native.value, kernel.function().name(), "llvm");
    expect(cpu.ok()) << cpu.error;
    auto metal = build(native.value, kernel.function().name(), "metal");
    expect(metal.ok()) << metal.error;
    if (!cpu.ok() || !metal.ok()) { return; }

    tvm::Device cpu_device{kDLCPU, 0};
    tvm::Device metal_device{kDLMetal, 0};
    expect(tvm::runtime::DeviceAPI::Get(metal_device, true) != nullptr);
    luisa::vector<float> x_values(n);
    luisa::vector<float> y_values(n);
    for (auto i = 0u; i < x_values.size(); i++) {
        x_values[i] = static_cast<float>(i % 37u) * 0.125f - 2.0f;
        y_values[i] = static_cast<float>(i % 19u) * -0.25f + 1.0f;
    }

    auto x_cpu = upload({n}, x_values, cpu_device);
    auto y_cpu = upload({n}, y_values, cpu_device);
    auto result_cpu = allocate({n}, cpu_device);
    (*cpu.entry)(x_cpu, y_cpu, result_cpu);

    auto x_metal = upload({n}, x_values, metal_device);
    auto y_metal = upload({n}, y_values, metal_device);
    auto result_metal = allocate({n}, metal_device);
    try {
        (*metal.entry)(x_metal, y_metal, result_metal);
    } catch (const tvm::ffi::Error &error) {
        expect(false) << error.what();
        return;
    }

    auto cpu_values = download(result_cpu, n);
    auto metal_values = download(result_metal, n);
    for (auto i = 0u; i < x_values.size(); i++) {
        auto reference = 1.25f * x_values[i] - 0.75f * y_values[i] + 0.5f;
        expect(close(cpu_values[i], reference));
        expect(close(metal_values[i], reference));
        expect(close(metal_values[i], cpu_values[i]));
    }
}

void test_same_reduction_on_cpu_and_metal() {
    constexpr int64_t rows = 37;
    constexpr int64_t columns = 19;
    auto definition = tile_kernel(
        "tile_tirx_dual_row_sum",
        [](TensorView<const float, 2> x,
           TensorView<float, 1> result) {
            auto row = axis("row", x.extent<0>());
            auto column = axis("column", x.extent<1>());
            for (auto &row_nest : parallel(shape(row))) {
                auto sum = Scalar<float>{0.0f};
                for (auto &item : row_nest.reduce(shape(column))) {
                    sum += x(row_nest[row], item[column]).load();
                }
                result(row_nest[row]).store(sum);
            }
        });
    auto kernel = definition.capture(
        tensor_shape("x", rows, columns), tensor_shape("result", rows));
    expect(kernel.valid());
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }

    auto cpu = build(native.value, kernel.function().name(), "llvm");
    expect(cpu.ok()) << cpu.error;
    auto metal = build(native.value, kernel.function().name(), "metal");
    expect(metal.ok()) << metal.error;
    if (!cpu.ok() || !metal.ok()) { return; }

    tvm::Device cpu_device{kDLCPU, 0};
    tvm::Device metal_device{kDLMetal, 0};
    luisa::vector<float> input_values(rows * columns);
    for (auto i = 0u; i < input_values.size(); i++) {
        input_values[i] = static_cast<float>(static_cast<int>(i % 23u) - 11) * 0.0625f;
    }
    auto input_cpu = upload({rows, columns}, input_values, cpu_device);
    auto result_cpu = allocate({rows}, cpu_device);
    (*cpu.entry)(input_cpu, result_cpu);
    auto input_metal = upload({rows, columns}, input_values, metal_device);
    auto result_metal = allocate({rows}, metal_device);
    try {
        (*metal.entry)(input_metal, result_metal);
    } catch (const tvm::ffi::Error &error) {
        expect(false) << error.what();
        return;
    }

    auto cpu_values = download(result_cpu, rows);
    auto metal_values = download(result_metal, rows);
    for (auto row = 0; row < rows; row++) {
        auto reference = 0.0f;
        for (auto column = 0; column < columns; column++) {
            reference += input_values[static_cast<size_t>(row * columns + column)];
        }
        expect(close(cpu_values[row], reference));
        expect(close(metal_values[row], reference));
        expect(close(metal_values[row], cpu_values[row]));
    }
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_tirx_same_axpy_executes_on_cpu_and_metal"_test = test_same_axpy_on_cpu_and_metal;
    "tile_tirx_same_reduction_executes_on_cpu_and_metal"_test = test_same_reduction_on_cpu_and_metal;
}
