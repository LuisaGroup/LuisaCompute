#include <exception>
#include <string>
#include <utility>

#include <tvm/ffi/function.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/function.h>
#include <tvm/s_tir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/transform.h>

#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/tile/bridge/tirx/compiler.h>

namespace luisa::compute::tile::bridge::tirx {

namespace detail {

using FunctionMap = tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc>;

[[nodiscard]] tvm::IRModule make_module(
    FunctionMap functions,
    tvm::DictAttrs attributes = tvm::DictAttrs{},
    tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Array<tvm::GlobalInfo>> global_infos = {}) {
    static auto constructor = tvm::ffi::Function::GetGlobalRequired("ir.IRModule");
    return constructor(std::move(functions), std::move(attributes), std::move(global_infos))
        .cast<tvm::IRModule>();
}

[[nodiscard]] tvm::IRModule run_pass(tvm::transform::Pass pass, tvm::IRModule module) {
    // Some prebuilt TVM packages intentionally hide Pass::operator() while
    // exporting the public pass registry. Dispatching through the native C++
    // registry supports both those packages and ordinary development builds.
    static auto run = tvm::ffi::Function::GetGlobalRequired("transform.RunPass");
    return run(std::move(pass), std::move(module)).cast<tvm::IRModule>();
}

void run_common_pipeline(tvm::IRModule &module, const CompileOptions &options, const tvm::Target &target) {
    auto apply = [&module](tvm::transform::Pass pass) {
        module = run_pass(std::move(pass), std::move(module));
    };
    apply(tvm::tirx::transform::BindTarget(target));
    if (options.pipeline == PipelineKind::TILE) {
        apply(tvm::tirx::transform::LowerTIRx());
    } else {
        apply(tvm::s_tir::transform::LowerInitBlock());
    }
    apply(tvm::s_tir::transform::UnifyThreadBinding());
    apply(tvm::tirx::transform::StmtSimplify());
    if (options.pipeline == PipelineKind::TILE) {
        apply(tvm::tirx::transform::LowerTIRxOpaque());
    }
    apply(tvm::tirx::transform::FlattenBuffer());
    apply(tvm::tirx::transform::BF16ComputeLegalize());
    apply(tvm::tirx::transform::NarrowDataType(32));
    apply(tvm::tirx::transform::VectorizeLoop(options.vectorize));
    apply(tvm::tirx::transform::UnrollLoop());
    apply(tvm::tirx::transform::StmtSimplify());
    if (options.eliminate_common_subexpressions) {
        apply(tvm::tirx::transform::CommonSubexprElim());
    }
    apply(tvm::tirx::transform::FP8ComputeLegalize());
    apply(tvm::tirx::transform::VerifyMemory());
    apply(tvm::tirx::transform::AnnotateEntryFunc());
    apply(tvm::tirx::transform::SplitHostDevice());
    apply(tvm::tirx::transform::MakePackedAPI());
    apply(tvm::tirx::transform::FP8StorageLegalize());
    apply(tvm::tirx::transform::BF16StorageLegalize());
}

void finalize_host(tvm::IRModule &module) {
    module = run_pass(tvm::tirx::transform::LowerTVMBuiltin(), std::move(module));
    module = run_pass(tvm::tirx::transform::LowerIntrin(), std::move(module));
}

void finalize_device(tvm::IRModule &module) {
    module = run_pass(tvm::tirx::transform::LowerWarpMemory(), std::move(module));
    module = run_pass(tvm::tirx::transform::StmtSimplify(), std::move(module));
    module = run_pass(tvm::tirx::transform::LowerIntrin(), std::move(module));
}

[[nodiscard]] bool is_host_target(const tvm::Target &target) noexcept {
    auto name = target->kind->name;
    return name == "llvm" || name == "c";
}

[[nodiscard]] tvm::ffi::Module codegen(tvm::IRModule module, const tvm::Target &target) {
    auto builder_name = std::string{"target.build."} + target->kind->name.operator std::string();
    auto builder = tvm::ffi::Function::GetGlobalRequired(builder_name);
    return builder(std::move(module), target).cast<tvm::ffi::Module>();
}

}// namespace detail

CompilationResult compile(tvm::IRModule module, const CompileOptions &options) noexcept {
    if (!module.defined()) { return CompilationResult{luisa::string{"cannot compile an undefined TIRx module"}}; }
    if (options.target.empty()) { return CompilationResult{luisa::string{"TIRx target must not be empty"}}; }
    if (options.host.empty()) { return CompilationResult{luisa::string{"TIRx host target must not be empty"}}; }
    try {
        tvm::Target device_target{tvm::ffi::String{options.target}};
        tvm::Target host_target{tvm::ffi::String{options.host}};
        tvm::Target bound_target{device_target, host_target};
        detail::run_common_pipeline(module, options, bound_target);

        detail::FunctionMap host_functions;
        struct DevicePartition {
            tvm::Target target;
            detail::FunctionMap functions;
        };
        luisa::unordered_map<luisa::string, size_t> device_partition_indices;
        luisa::vector<DevicePartition> device_partitions;
        for (auto &&[global, base_function] : module->functions) {
            auto function = base_function.as<tvm::tirx::PrimFunc>();
            if (!function) {
                return CompilationResult{luisa::string{"TIRx bridge only accepts PrimFunc modules"}};
            }
            auto function_target = function.value()->GetAttr<tvm::Target>(tvm::attr::kTarget);
            if (!function_target) {
                return CompilationResult{luisa::string{"lowered TIRx PrimFunc is missing its target attribute"}};
            }
            if (detail::is_host_target(function_target.value())) {
                host_functions.Set(global, function.value());
                continue;
            }
            auto key = luisa::string{function_target.value()->str()};
            auto [iter, inserted] = device_partition_indices.try_emplace(key, device_partitions.size());
            if (inserted) {
                device_partitions.emplace_back(DevicePartition{function_target.value(), {}});
            }
            device_partitions[iter->second].functions.Set(global, function.value());
        }

        auto host_module = detail::make_module(
            std::move(host_functions), module->attrs, module->global_infos);
        detail::finalize_host(host_module);
        auto runtime_module = detail::codegen(std::move(host_module), host_target);
        for (auto &&partition : device_partitions) {
            auto device_module = detail::make_module(
                std::move(partition.functions), module->attrs, module->global_infos);
            detail::finalize_device(device_module);
            runtime_module->ImportModule(detail::codegen(std::move(device_module), partition.target));
        }
        return CompilationResult{std::move(runtime_module)};
    } catch (const tvm::ffi::Error &error) {
        return CompilationResult{luisa::string{error.what()}};
    } catch (const std::exception &error) {
        return CompilationResult{luisa::string{error.what()}};
    } catch (...) {
        return CompilationResult{luisa::string{"unknown failure in native TIRx compilation"}};
    }
}

CompilationResult compile(
    tvm::tirx::PrimFunc function,
    luisa::string_view name,
    const CompileOptions &options) noexcept {
    if (!function.defined()) { return CompilationResult{luisa::string{"cannot compile an undefined TIRx PrimFunc"}}; }
    if (name.empty()) { return CompilationResult{luisa::string{"TIRx PrimFunc name must not be empty"}}; }
    try {
        auto symbol = tvm::ffi::String{std::string{name}};
        function = tvm::WithAttr(std::move(function), tvm::attr::kGlobalSymbol, symbol);
        if (options.noalias) {
            function = tvm::WithAttr(std::move(function), "tirx.noalias", true);
        }
        detail::FunctionMap functions{{tvm::GlobalVar{symbol}, function}};
        return compile(detail::make_module(std::move(functions)), options);
    } catch (const tvm::ffi::Error &error) {
        return CompilationResult{luisa::string{error.what()}};
    } catch (const std::exception &error) {
        return CompilationResult{luisa::string{error.what()}};
    } catch (...) {
        return CompilationResult{luisa::string{"unknown failure while constructing a native TIRx module"}};
    }
}

}// namespace luisa::compute::tile::bridge::tirx
