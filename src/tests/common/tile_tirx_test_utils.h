#pragma once

// Native TVMx execution shared by Tile correctness tests and benchmarks.
// Choosing Metal must allocate and execute on Metal; there is no CPU fallback.

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/tensor.h>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/tile/bridge/tirx/compiler.h>
#include <luisa/tile/bridge/tirx/lower.h>
#include <luisa/tile/dsl.h>
#include <luisa/tile/verifier.h>

namespace luisa::test::tile_tirx {

[[nodiscard]] inline size_t count_operations(const compute::tile::Region &region, compute::tile::OperationKind kind) {
    size_t count = 0u;
    for (auto block : region.blocks()) {
        for (auto operation : block->operations()) {
            count += operation->kind() == kind;
            for (auto &&child : operation->regions()) { count += count_operations(*child, kind); }
        }
    }
    return count;
}

[[nodiscard]] inline bool uses_only_subtile_memory(const compute::tile::Region &region) {
    using compute::tile::OperationKind;
    for (auto block : region.blocks()) {
        for (auto operation : block->operations()) {
            if ((operation->kind() == OperationKind::VIEW_LOAD || operation->kind() == OperationKind::VIEW_STORE) &&
                !operation->domain()) { return false; }
            for (auto &&child : operation->regions()) {
                if (!uses_only_subtile_memory(*child)) { return false; }
            }
        }
    }
    return true;
}

struct Executable {
    tvm::ffi::Optional<tvm::ffi::Module> module;
    tvm::ffi::Optional<tvm::ffi::Function> entry;
    luisa::string error;
    luisa::vector<compute::tile::bridge::tirx::GroupPlan> plans;

    [[nodiscard]] bool ok() const noexcept {
        return error.empty() && module.has_value() && entry.has_value();
    }
};

template<typename T>
[[nodiscard]] constexpr DLDataType dl_data_type() noexcept {
    if constexpr (std::is_same_v<T, float>) {
        return DLDataType{kDLFloat, 32, 1};
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return DLDataType{kDLInt, 32, 1};
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return DLDataType{kDLInt, 64, 1};
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported test tensor element type");
    }
}

class Runtime final {

private:
    tvm::Device _device{kDLCPU, 0};
    luisa::string _target{"llvm"};
    luisa::string _cpu_model{"generic"};
    uint32_t _metal_max_threads{0u};

public:
    explicit Runtime(luisa::string_view backend, bool native_cpu = false) {
        if (native_cpu && backend != "cpu") { throw std::invalid_argument{"native CPU model requires the CPU backend"}; }
        if (backend == "metal") {
            _device = tvm::Device{kDLMetal, 0};
            _target = "metal";
        } else if (backend != "cpu") {
            throw std::invalid_argument{"Tile test backend must be cpu or metal"};
        }
        if (native_cpu) {
            auto cpu = tvm::ffi::Function::GetGlobalRequired("target.llvm_get_system_cpu")().cast<tvm::ffi::String>();
            auto triple = tvm::ffi::Function::GetGlobalRequired("target.llvm_get_system_triple")().cast<tvm::ffi::String>();
            if (cpu.empty() || !tvm::ffi::Function::GetGlobalRequired("target.llvm_is_valid_cpu")(cpu, triple).cast<bool>()) {
                throw std::runtime_error{"LLVM cannot represent the detected host CPU; no generic fallback"};
            }
            _cpu_model.assign(cpu.data(), cpu.size());
        }
        auto api = tvm::runtime::DeviceAPI::Get(_device, true);
        if (api == nullptr) { throw std::runtime_error{"requested TVMx device runtime is unavailable"}; }
        tvm::ffi::Any exists;
        api->GetAttr(_device, tvm::runtime::DeviceAttrKind::kExist, &exists);
        if (exists.cast<int64_t>() == 0) {
            throw std::runtime_error{"requested TVMx physical device is unavailable"};
        }
        if (_device.device_type == kDLMetal) {
            tvm::ffi::Any maximum;
            api->GetAttr(_device, tvm::runtime::DeviceAttrKind::kMaxThreadsPerBlock, &maximum);
            auto threads = maximum.cast<int64_t>();
            if (threads <= 0 || threads > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error{"Metal runtime did not report a valid threadgroup limit"};
            }
            _metal_max_threads = static_cast<uint32_t>(threads);
        }
    }

    [[nodiscard]] tvm::Device device() const noexcept { return _device; }
    [[nodiscard]] luisa::string_view target() const noexcept { return _target; }
    [[nodiscard]] luisa::string_view cpu_model() const noexcept { return _cpu_model; }
    [[nodiscard]] uint32_t metal_max_threads() const noexcept { return _metal_max_threads; }

    [[nodiscard]] Executable build(const compute::tile::Kernel &kernel, bool noalias = false, bool cooperative_matrix = false, bool vectorize = true, bool auto_vectorize = false,
                                   const compute::tile::bridge::tirx::PlannerOptions &planner = {}, bool metal_mpp = false, bool forward_readonly_tile_loads = false,
                                   compute::tile::bridge::tirx::CpuMatrixBackend cpu_matrix_backend = compute::tile::bridge::tirx::CpuMatrixBackend::REFERENCE,
                                   compute::tile::bridge::tirx::CpuMathBackend cpu_math_backend = compute::tile::bridge::tirx::CpuMathBackend::REFERENCE,
                                   const compute::tile::bridge::tirx::LowerOptions &lower_options = {}) const {
        using namespace compute::tile::bridge::tirx;
        Executable result;
        if (!kernel.valid()) {
            result.error = "Tile DSL capture or verification failed";
            for (auto &&diagnostic : kernel.diagnostics()) {
                result.error.append(": ");
                result.error.append(diagnostic);
            }
            auto verified = compute::tile::verify(kernel.module());
            for (auto &&diagnostic : verified.diagnostics()) {
                result.error.append(": ");
                result.error.append(diagnostic.message);
            }
            return result;
        }
        if (!uses_only_subtile_memory(kernel.function().body())) {
            result.error = "operator POCs must access subtiles; scalar memory belongs in explicit low-level tests";
            return result;
        }
        auto native = lower(kernel.function(), lower_options);
        if (!native) {
            result.error = std::move(native.error);
            return result;
        }
        CompileOptions options;
        options.target = _target;
        if (_target == "llvm" && _cpu_model != "generic") {
            options.target = luisa::string{"{\"kind\":\"llvm\",\"mcpu\":\""} + _cpu_model + "\"}";
        }
        if ((cooperative_matrix || planner.metal_subgroup_reductions) && _target == "metal") {
            // Opt-in tests/benchmarks require an Apple-family-7+ device, not
            // merely the existence of an arbitrary Metal runtime.
            options.target = luisa::string{R"({"kind":"metal","thread_warp_size":32,"max_num_threads":)"} +
                             std::to_string(_metal_max_threads) + R"(,"max_threads_per_block":)" +
                             std::to_string(_metal_max_threads) + "}";
        }
        options.noalias = noalias;
        options.cooperative_matrix = cooperative_matrix;
        options.metal_mpp = metal_mpp;
        options.forward_readonly_tile_loads = forward_readonly_tile_loads;
        options.vectorize = vectorize;
        options.auto_vectorize = auto_vectorize;
        options.cpu_matrix_backend = cpu_matrix_backend;
        options.cpu_math_backend = cpu_math_backend;
        options.planner = planner;
        auto compilation = compile(std::move(native.value), kernel.function().name(), options);
        if (!compilation) {
            result.error = luisa::string{compilation.error()};
            return result;
        }
        result.module = compilation.module();
        if (_target == "llvm") {
            auto source = result.module.value()->InspectSource("ll");
            auto expected = luisa::string{"\"target-cpu\"=\""} + _cpu_model + "\"";
            if (luisa::string_view{source.data(), source.size()}.find(expected) == luisa::string_view::npos) {
                result.error = "LLVM codegen did not preserve the requested CPU model";
                return result;
            }
        }
        result.plans.assign(compilation.plans().begin(), compilation.plans().end());
        auto name = kernel.function().name();
        result.entry = result.module.value()->GetFunction(
            tvm::ffi::String{name.data(), name.size()}, true);
        if (!result.entry) { result.error = "compiled module has no requested entry function"; }
        return result;
    }

    template<typename T>
    [[nodiscard]] tvm::runtime::Tensor allocate(std::initializer_list<int64_t> shape) const {
        return tvm::runtime::Tensor::Empty(tvm::ffi::Shape{shape}, dl_data_type<T>(), _device);
    }

    template<typename T>
    [[nodiscard]] tvm::runtime::Tensor upload(
        std::initializer_list<int64_t> shape, const luisa::vector<T> &values) const {
        auto host = tvm::runtime::Tensor::Empty(
            tvm::ffi::Shape{shape}, dl_data_type<T>(), tvm::Device{kDLCPU, 0});
        host.CopyFromBytes(values.data(), values.size() * sizeof(T));
        return _device.device_type == kDLCPU ? host : host.CopyTo(_device);
    }

    template<typename T>
    [[nodiscard]] luisa::vector<T> download(const tvm::runtime::Tensor &tensor, size_t count) const {
        luisa::vector<T> values(count);
        // TVMx CopyToBytes completes the device-to-host copy before returning.
        tensor.CopyToBytes(values.data(), values.size() * sizeof(T));
        return values;
    }

    void synchronize() const {
        if (_device.device_type != kDLCPU) {
            tvm::runtime::DeviceAPI::Get(_device)->StreamSync(_device, nullptr);
        }
    }
};

}// namespace luisa::test::tile_tirx
