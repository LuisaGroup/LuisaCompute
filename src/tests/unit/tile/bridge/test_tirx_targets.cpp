// Cross-target execution tests for one portable Tile DSL capture, plus
// compile-only recoverable rejection of missing Metal precise-math contracts.

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
#include <string_view>

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

class GlobalFunctionRestorer {
private:
    std::string_view _name;
    tvm::ffi::Optional<tvm::ffi::Function> _original;

public:
    explicit GlobalFunctionRestorer(std::string_view name)
        : _name{name}, _original{tvm::ffi::Function::GetGlobal(name)} {}
    GlobalFunctionRestorer(const GlobalFunctionRestorer &) = delete;
    GlobalFunctionRestorer &operator=(const GlobalFunctionRestorer &) = delete;
    ~GlobalFunctionRestorer() {
        if (_original) {
            tvm::ffi::Function::SetGlobal(_name, _original.value(), true);
        } else {
            tvm::ffi::Function::RemoveGlobal(tvm::ffi::String{_name.data(), _name.size()});
        }
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
    (*metal.entry)(x_metal, y_metal, result_metal);

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
                    sum += x(row_nest.index(row), item.index(column)).load();
                }
                result(row_nest.index(row)).store(sum);
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
    (*metal.entry)(input_metal, result_metal);

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

void test_cpu_target_model_reaches_codegen() {
    auto cpu = tvm::ffi::Function::GetGlobalRequired("target.llvm_get_system_cpu")().cast<tvm::ffi::String>();
    auto triple = tvm::ffi::Function::GetGlobalRequired("target.llvm_get_system_triple")().cast<tvm::ffi::String>();
    expect(tvm::ffi::Function::GetGlobalRequired("target.llvm_is_valid_cpu")(cpu, triple).cast<bool>());
    auto model_target = luisa::string{"{\"kind\":\"llvm\",\"mcpu\":\""}.append(cpu.data(), cpu.size()).append("\"}");
    auto definition = tile_kernel("cpu_target_policy", [](TensorView<const float, 1> input, TensorView<float, 1> output) {
        for (auto &worker : parallel(shape(37))) {
            auto i = worker.index();
            output(i).store(input(i).load() * 1.5f + 0.25f);
        }
    });
    auto kernel = definition.capture(tensor_shape(37), tensor_shape(37));
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    for (auto mode = 0u; mode < 3u; mode++) {
        auto model_compute = mode == 1u;
        auto gpu_compute = mode == 2u;
        CompileOptions options;
        options.target = gpu_compute ? "metal" : model_compute ? model_target :
                                                                 R"({"kind":"llvm","mcpu":"generic"})";
        // The host controls GPU launch wrappers, not the ISA of a standalone
        // CPU compute entry. Test both directions so defaults cannot mask it.
        options.host = model_compute ? "llvm" : model_target;
        auto compiled = compile(native.value, kernel.function().name(), options);
        expect(compiled.ok()) << compiled.error();
        if (!compiled) { continue; }
        auto source = compiled.module().value()->InspectSource("ll");
        auto expected_cpu = model_compute || gpu_compute ? luisa::string{cpu.data(), cpu.size()} : luisa::string{"generic"};
        auto attribute = luisa::string{"\"target-cpu\"=\""} + expected_cpu + "\"";
        expect(std::string_view{source.data(), source.size()}.find(attribute) != std::string_view::npos)
            << "requested target=" << options.target << " host=" << options.host << " must emit " << attribute;
        auto entry = compiled.module().value()->GetFunction("cpu_target_policy", true);
        expect(entry.has_value());
        if (!entry) { continue; }
        luisa::vector<float> values(37);
        for (auto i = 0u; i < values.size(); i++) { values[i] = static_cast<float>(i) * 0.125f - 1.0f; }
        tvm::Device device{gpu_compute ? kDLMetal : kDLCPU, 0};
        auto input = upload({37}, values, device);
        auto output = allocate({37}, device);
        (*entry)(input, output);
        auto actual = download(output, values.size());
        for (auto i = 0u; i < values.size(); i++) { expect(close(actual[i], values[i] * 1.5f + 0.25f)); }
    }
}

void test_ordered_metal_reduction_missing_contract_is_recoverable() {
    constexpr auto rows = int64_t{3}, columns = int64_t{37};
    auto definition = tile_kernel("metal_fold_left_missing_precise_math", [=](TensorView<const float, 2> input,
                                                                              TensorView<float, 2> output) {
        auto one = axis("one", 1), column = axis("column", columns);
        for (auto &nest : parallel(shape(rows))) {
            auto x = input[coord(nest.index(), 0), shape(one, column)];
            auto fold = reduce(x, column, add, reduction::fold_left);
            output(coord(nest.index(), 0), shape(1, 1)).store(full<float>(shape(1, 1), fold.at(coord(0))));
        }
    });
    auto kernel = definition.capture(tensor_shape(rows, columns), tensor_shape(rows, 1));
    expect(kernel.valid());
    if (!kernel.valid()) { return; }
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    CompileOptions options;
    options.target = "metal";
    options.host = "llvm";
    constexpr auto target_contract = "target.metal.precise_math_contract_version";
    constexpr auto runtime_contract = "runtime.metal.precise_math_contract_version";
    auto original_target = tvm::ffi::Function::GetGlobal(target_contract);
    auto original_runtime = tvm::ffi::Function::GetGlobal(runtime_contract);
    // UT invokes these standalone test bodies serially. Registry mutation is
    // confined to this synchronous compile-only scope, never GPU execution.
    for (auto missing : {target_contract, runtime_contract}) {
        {
            GlobalFunctionRestorer restore_target{target_contract};
            GlobalFunctionRestorer restore_runtime{runtime_contract};
            // Register both versions first so the runtime-missing case reaches
            // the second check even when neither real extension is installed.
            auto version_one = tvm::ffi::Function::FromTyped([] { return int64_t{1}; });
            tvm::ffi::Function::SetGlobal(target_contract, version_one, true);
            tvm::ffi::Function::SetGlobal(runtime_contract, version_one, true);
            tvm::ffi::Function::RemoveGlobal(tvm::ffi::String{missing});
            expect(!tvm::ffi::Function::GetGlobal(missing).has_value());
            auto present_name = std::string_view{missing} == target_contract ? runtime_contract : target_contract;
            auto present = tvm::ffi::Function::GetGlobal(present_name);
            expect(present.has_value());
            if (present) { expect(eq((*present)().cast<int64_t>(), int64_t{1})); }
            auto compiled = compile(native.value, kernel.function().name(), options);
            expect(!compiled.ok());
            expect(!compiled.module().has_value());
            expect(compiled.error() == "ordered reductions on TVM's Metal runtime require metal-precise-math-v1.patch; Luisa Runtime compile_device does not require this extension")
                << "missing " << missing << ": " << compiled.error();
        }
        auto restored_target = tvm::ffi::Function::GetGlobal(target_contract);
        auto restored_runtime = tvm::ffi::Function::GetGlobal(runtime_contract);
        expect(eq(restored_target.has_value(), original_target.has_value()));
        expect(eq(restored_runtime.has_value(), original_runtime.has_value()));
        if (original_target && restored_target) { expect(restored_target.value().same_as(original_target.value())); }
        if (original_runtime && restored_runtime) { expect(restored_runtime.value().same_as(original_runtime.value())); }
    }
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_tirx_same_axpy_executes_on_cpu_and_metal"_test = test_same_axpy_on_cpu_and_metal;
    "tile_tirx_same_reduction_executes_on_cpu_and_metal"_test = test_same_reduction_on_cpu_and_metal;
    "tile_tirx_cpu_target_model_reaches_codegen"_test = test_cpu_target_model_reaches_codegen;
    "tile_tirx_ordered_metal_reduction_missing_contract_is_recoverable"_test = test_ordered_metal_reduction_missing_contract_is_recoverable;
}
