#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

#include <tvm/ffi/function.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/function.h>
#include <tvm/s_tir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/tile/bridge/tirx/compiler.h>

#include "execution.h"

namespace luisa::compute::tile::bridge::tirx {

namespace detail {

using FunctionMap = tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc>;

enum class RootParallelBinding : uint8_t {
    SERIAL,
    CPU_THREADS,
    GPU_GRID
};

[[nodiscard]] tvm::IRModule make_module(
    FunctionMap functions,
    tvm::DictAttrs attributes,
    tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Array<tvm::GlobalInfo>> global_infos);

class ExecutionMapper final : public tvm::tirx::StmtMutator {

private:
    RootParallelBinding _binding;
    uint32_t _gpu_threads_per_block;
    uint64_t _shared_memory_limit;
    uint32_t _logical_parallel_depth{0u};
    uint32_t _vector_depth{0u};
    bool _vectorize;
    std::string _target_name;

private:
    [[noreturn]] void _scope_error(
        const tvm::tirx::ForNode *loop,
        const std::string &scope,
        const std::string &reason) const {
        throw std::runtime_error{
            "TileIR " + std::string{loop->loop_var->name} + ": execution scope '" + scope +
            "' on target '" + _target_name + "' " + reason};
    }

    // This reference planner realizes a worker prefix and, on LLVM, a vector
    // suffix. Hardware cooperation scopes require a distribution/resource
    // plan, not just replacing an induction variable with a thread index.
    [[nodiscard]] bool _resolve_vector(const tvm::tirx::ForNode *loop) const {
        auto constraint = loop->annotations.Get(execution_scope_annotation);
        if (!constraint) { return false; }
        auto scope = constraint.value().as<tvm::ffi::String>();
        if (!scope) { _scope_error(loop, "<invalid>", "must have a string scope constraint"); }
        auto name = std::string{scope.value()};
        if (name == "worker") {
            if (_binding == RootParallelBinding::SERIAL) {
                _scope_error(loop, name, "has no worker execution mapping");
            }
            if (_logical_parallel_depth != 0u) {
                _scope_error(loop, name, "requires an explicit coordinate factorization for nested worker bindings");
            }
            return false;
        }
        if (name == "vector") {
            if (_binding != RootParallelBinding::CPU_THREADS) {
                _scope_error(loop, name, "is not supported by this target's execution mapper");
            }
            if (!_vectorize) { _scope_error(loop, name, "conflicts with disabled vectorization"); }
            if (_vector_depth != 0u) {
                _scope_error(loop, name, "requires an explicit coordinate factorization for nested vector bindings");
            }
            return true;
        }
        _scope_error(loop, name, "is not supported by the current execution mapper");
    }

    [[nodiscard]] tvm::tirx::Stmt _loop(
        const tvm::tirx::ForNode *loop,
        tvm::tirx::ForKind kind,
        tvm::tirx::Stmt body,
        tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> annotations) const {
        return tvm::tirx::For{
            loop->loop_var,
            loop->min,
            loop->extent,
            kind,
            std::move(body),
            std::nullopt,
            std::move(annotations),
            loop->step,
            loop->span};
    }

    [[nodiscard]] tvm::tirx::Stmt _gpu_grid(
        const tvm::tirx::ForNode *loop,
        tvm::tirx::Stmt body) const {
        auto extent_constant = loop->extent.as<tvm::IntImmNode>();
        if (extent_constant == nullptr || extent_constant->value <= 0) {
            throw std::runtime_error{
                "GPU execution binding currently requires a positive static logical parallel extent"};
        }
        auto extent = static_cast<uint64_t>(extent_constant->value);
        auto thread_count = std::min<uint64_t>(extent, _gpu_threads_per_block);
        auto block_count = (extent + thread_count - 1u) / thread_count;
        auto type = loop->loop_var.ty();
        auto zero = tvm::IntImm{type, 0};
        auto threads = tvm::IntImm{type, static_cast<int64_t>(thread_count)};
        auto blocks = tvm::IntImm{type, static_cast<int64_t>(block_count)};
        auto block = tvm::tirx::PrimVar{
            tvm::ffi::String{std::string{loop->loop_var->name} + "_block"}, type};
        auto thread = tvm::tirx::PrimVar{
            tvm::ffi::String{std::string{loop->loop_var->name} + "_thread"}, type};
        tvm::PrimExpr linear = block * threads + thread;
        tvm::PrimExpr logical = loop->min + linear;
        body = tvm::tirx::Substitute(
            std::move(body),
            tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>{{loop->loop_var, logical}});
        if (block_count * thread_count != extent) {
            body = tvm::tirx::IfThenElse{linear < loop->extent, std::move(body)};
        }
        auto thread_axis = tvm::tirx::IterVar{
            tvm::Range::FromMinExtent(zero, threads),
            thread,
            tvm::tirx::IterVarType::kThreadIndex,
            "threadIdx.x"};
        body = tvm::tirx::For{
            thread,
            zero,
            threads,
            tvm::tirx::ForKind::kThreadBinding,
            std::move(body),
            std::move(thread_axis)};
        auto block_axis = tvm::tirx::IterVar{
            tvm::Range::FromMinExtent(zero, blocks),
            block,
            tvm::tirx::IterVarType::kThreadIndex,
            "blockIdx.x"};
        return tvm::tirx::For{
            block,
            zero,
            blocks,
            tvm::tirx::ForKind::kThreadBinding,
            std::move(body),
            std::move(block_axis)};
    }

protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::AllocBufferNode *allocation) final {
        auto result = StmtMutator::VisitStmt_(allocation).as_or_throw<tvm::tirx::AllocBuffer>();
        if (auto constraint = result->annotations.Get(memory_resource_annotation)) {
            auto resource = constraint.value().as<tvm::ffi::String>();
            if (!resource || resource.value() != "private") {
                auto name = resource ? std::string{resource.value()} : std::string{"<invalid>"};
                throw std::runtime_error{"Memory resource '" + name + "' on target '" + _target_name +
                                         "' has no allocation plan in this execution scope"};
            }
            result.CopyOnWrite()->annotations.erase(memory_resource_annotation);
        }
        return result;
    }

    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::ForNode *loop) final {
        if (!loop->annotations.count(logical_parallel_annotation)) {
            if (loop->annotations.count(execution_scope_annotation)) {
                _scope_error(loop, "<orphaned>", "requires its logical parallel domain");
            }
            auto result = StmtMutator::VisitStmt_(loop);
            if (loop->annotations.count(independent_elements_annotation)) {
                auto mapped = result.as_or_throw<tvm::tirx::For>();
                mapped.CopyOnWrite()->annotations.erase(independent_elements_annotation);
                return mapped;
            }
            return result;
        }
        if (auto scope = loop->annotations.Get(execution_scope_annotation);
            scope && scope.value().as<tvm::ffi::String>() && scope.value().cast<tvm::ffi::String>() == "group") {
            if (_target_name != "metal") { _scope_error(loop, "group", "is not supported by this target's execution mapper"); }
            if (_logical_parallel_depth != 0u) { _scope_error(loop, "group", "requires a coordinate factorization for nested group bindings"); }
            return map_metal_cooperative_group(tvm::ffi::GetRef<tvm::tirx::For>(loop), _gpu_threads_per_block, _shared_memory_limit);
        }
        // Resolve before mutating the body, including through unbound or
        // serial intermediate levels. Unsupported constraints are hard errors,
        // never optional hints that disappear during structural export.
        auto is_vector = _resolve_vector(loop);
        auto annotations = loop->annotations;
        annotations.erase(logical_parallel_annotation);
        annotations.erase(execution_scope_annotation);
        auto is_outermost = _logical_parallel_depth == 0u;
        _logical_parallel_depth++;
        _vector_depth += is_vector;
        auto body = VisitStmt(loop->body);
        _vector_depth -= is_vector;
        _logical_parallel_depth--;
        if (auto extent = loop->extent.as<tvm::IntImmNode>(); extent != nullptr && extent->value == 0) {
            return tvm::tirx::Evaluate{tvm::IntImm::Int32(0)};
        }
        if (is_vector) {
            auto vector_loop = _loop(loop, tvm::tirx::ForKind::kVectorized, std::move(body), std::move(annotations));
            return privatize_vector_storage(vector_loop.as_or_throw<tvm::tirx::For>());
        }
        if (!is_outermost || _binding == RootParallelBinding::SERIAL) {
            return _loop(loop, tvm::tirx::ForKind::kSerial, std::move(body), std::move(annotations));
        }
        if (_binding == RootParallelBinding::CPU_THREADS) {
            return _loop(loop, tvm::tirx::ForKind::kParallel, std::move(body), std::move(annotations));
        }
        if (_binding == RootParallelBinding::GPU_GRID) {
            return _gpu_grid(loop, std::move(body));
        }
        throw std::runtime_error{"unresolved TileIR logical parallel binding"};
    }

public:
    ExecutionMapper(RootParallelBinding binding, uint32_t gpu_threads_per_block, uint64_t shared_memory_limit,
                    bool vectorize, std::string target_name) noexcept
        : _binding{binding}, _gpu_threads_per_block{gpu_threads_per_block}, _shared_memory_limit{shared_memory_limit},
          _vectorize{vectorize}, _target_name{std::move(target_name)} {}

    using StmtMutator::operator();
};

[[nodiscard]] bool is_gpu_target(const tvm::Target &target) noexcept {
    switch (target->GetTargetDeviceType()) {
        case kDLCUDA:
        case kDLMetal:
        case kDLROCM:
        case kDLVulkan:
        case kDLOpenCL:
        case kDLWebGPU: return true;
        default: return false;
    }
}

[[nodiscard]] RootParallelBinding resolve_parallel_binding(
    const tvm::Target &target) noexcept {
    if (is_gpu_target(target)) { return RootParallelBinding::GPU_GRID; }
    return target->kind->name == "llvm" ?
               RootParallelBinding::CPU_THREADS :
               RootParallelBinding::SERIAL;
}

[[nodiscard]] tvm::IRModule map_execution(
    tvm::IRModule module,
    const tvm::Target &target,
    bool vectorize,
    bool noalias) {
    auto binding = resolve_parallel_binding(target);
    auto threads = uint32_t{1u};
    auto shared_memory_limit = uint64_t{0u};
    if (binding == RootParallelBinding::GPU_GRID) {
        threads = 256u;
        if (auto maximum = target->GetAttr<int64_t>("max_num_threads")) {
            threads = std::min<uint32_t>(
                threads,
                static_cast<uint32_t>(std::max<int64_t>(1, maximum.value())));
        }
        if (auto maximum = target->GetAttr<int64_t>("max_shared_memory_per_block")) {
            shared_memory_limit = static_cast<uint64_t>(std::max<int64_t>(0, maximum.value()));
        }
    }
    FunctionMap functions;
    for (auto &&[global, base_function] : module->functions) {
        auto function = base_function.as<tvm::tirx::PrimFunc>();
        if (!function) {
            throw std::runtime_error{"Tile TIRx execution mapping only accepts PrimFunc modules"};
        }
        auto mapped = function.value();
        mapped.CopyOnWrite()->body = schedule_pipelines(mapped->body, noalias, shared_memory_limit);
        mapped.CopyOnWrite()->body = ExecutionMapper{binding, threads, shared_memory_limit, vectorize, std::string{target->kind->name}}(mapped->body);
        functions.Set(global, std::move(mapped));
    }
    return make_module(std::move(functions), module->attrs, module->global_infos);
}

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

class BufferUseCollector final : public tvm::tirx::StmtExprVisitor {

public:
    luisa::unordered_set<const tvm::tirx::VarNode *> used;

protected:
    void VisitBufferUse(const tvm::tirx::BufferVar &buffer) final { used.emplace(buffer.get()); }
    void VisitExpr_(const tvm::tirx::VarNode *variable) final { used.emplace(variable); }
};

class EmptyAllocationPruner final : public tvm::tirx::StmtMutator {

private:
    const luisa::unordered_set<const tvm::tirx::VarNode *> &_used;

protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::AllocBufferNode *allocation) final {
        auto empty = false;
        for (auto &&dimension : allocation->buffer->shape) {
            auto extent = dimension.as<tvm::IntImmNode>();
            if (extent == nullptr || extent->value < 0) { return StmtMutator::VisitStmt_(allocation); }
            empty |= extent->value == 0;
        }
        if (!empty || !allocation->annotations.empty()) { return StmtMutator::VisitStmt_(allocation); }
        auto offset = allocation->buffer->elem_offset.as<tvm::IntImmNode>();
        if (!allocation->buffer->strides.empty() || allocation->buffer->layout.has_value() ||
            !allocation->buffer->allocated_addr.empty() || offset == nullptr || offset->value != 0) {
            // Only erase plain storage. Unknown layout/address expressions
            // can carry additional constraints or observable effects.
            return StmtMutator::VisitStmt_(allocation);
        }
        if (_used.contains(allocation->buffer.get())) {
            throw std::runtime_error{"zero-sized Tile storage still has live buffer uses after simplification"};
        }
        return tvm::tirx::Evaluate{tvm::IntImm::Int32(0)};
    }

public:
    explicit EmptyAllocationPruner(const luisa::unordered_set<const tvm::tirx::VarNode *> &used) noexcept
        : _used{used} {}
};

void remove_empty_allocations(tvm::IRModule &module) {
    FunctionMap functions;
    for (auto &&[global, base_function] : module->functions) {
        auto function = base_function.as_or_throw<tvm::tirx::PrimFunc>();
        BufferUseCollector uses;
        uses(function->body);
        function.CopyOnWrite()->body = EmptyAllocationPruner{uses.used}(function->body);
        functions.Set(global, std::move(function));
    }
    module = make_module(std::move(functions), module->attrs, module->global_infos);
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
    // Resource/execution constraints have already been validated. Zero-trip
    // effects are gone, so an unused empty buffer needs no physical storage.
    // TVMx's host and Metal code generators reject zero-sized allocations.
    remove_empty_allocations(module);
    // Logical GPU bindings are ordinary TIRx thread-binding loops at this
    // point. Lower them to thread_extent regions before host/device splitting,
    // regardless of whether the function also contains TilePrimitive calls.
    apply(tvm::tirx::transform::LowerTIRxOpaque());
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
        module = detail::map_execution(std::move(module), device_target, options.vectorize, options.noalias);
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
