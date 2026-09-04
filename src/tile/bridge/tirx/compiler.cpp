#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <tvm/ffi/function.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/function.h>
#include <tvm/s_tir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
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

class CpuCallAudit final : public tvm::tirx::StmtExprVisitor {
protected:
    void VisitExpr_(const tvm::CallNode *call) final {
        // Transcendentals can dominate a small loop body's cost (exp in
        // softmax is the current example). Test registered Op identity, not
        // diagnostic names on the encountered operation.
        static const std::array expensive{
            tvm::Op::Get("tirx.exp"), tvm::Op::Get("tirx.exp2"),
            tvm::Op::Get("tirx.exp10"), tvm::Op::Get("tirx.erf"),
            tvm::Op::Get("tirx.tanh"), tvm::Op::Get("tirx.sigmoid"),
            tvm::Op::Get("tirx.log"), tvm::Op::Get("tirx.log2"),
            tvm::Op::Get("tirx.log1p"), tvm::Op::Get("tirx.log10")};
        has_expensive_call |= std::any_of(
            expensive.begin(), expensive.end(),
            [&](const tvm::Op &op) noexcept { return call->op.same_as(op); });
        // A synchronous external/provider call is opaque target work. Keeping
        // a small automatic root parallel is conservative; the launch model
        // must not reinterpret it as a cheap scalar expression.
        auto external = call->op.same_as(tvm::tirx::builtin::call_extern());
        auto cheap_array_reduction = false;
        if (external && !call->args.empty()) {
            if (auto callee = call->args[0u].as<tvm::tirx::StringImmNode>()) {
                cheap_array_reduction =
                    callee->value == "luisa_tile_accelerate_reduce_add_f32" ||
                    callee->value == "luisa_tile_accelerate_reduce_max_f32" ||
                    callee->value == "luisa_tile_accelerate_reduce_min_f32";
            }
        }
        has_expensive_call |= (external && !cheap_array_reduction) ||
                              call->op.same_as(tvm::tirx::builtin::tvm_call_packed());
        StmtExprVisitor::VisitExpr_(call);
    }

public:
    bool has_expensive_call{false};
};

[[nodiscard]] bool cpu_body_has_expensive_call(const tvm::tirx::Stmt &body) {
    CpuCallAudit audit;
    audit(body);
    return audit.has_expensive_call;
}

[[nodiscard]] tvm::IRModule make_module(
    FunctionMap functions,
    tvm::DictAttrs attributes,
    tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Array<tvm::GlobalInfo>> global_infos);

class ExecutionMapper final : public tvm::tirx::StmtMutator {

private:
    RootParallelBinding _binding;
    uint32_t _gpu_threads_per_block;
    uint32_t _gpu_group_thread_limit;
    uint64_t _shared_memory_limit;
    uint32_t _logical_parallel_depth{0u};
    uint32_t _vector_depth{0u};
    bool _vectorize;
    bool _auto_vectorize;
    bool _cooperative_matrix;
    bool _metal_mpp;
    std::string _target_name;
    const PlannerOptions &_planner;
    luisa::vector<GroupPlan> &_plans;
    luisa::span<const tvm::tirx::BufferVar> _readonly_inputs;

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
        if (_binding != RootParallelBinding::CPU_THREADS) {
            result.CopyOnWrite()->annotations.erase(manual_memory_annotation);
        }
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
                mapped.CopyOnWrite()->annotations.erase(materialized_exp_annotation);
                if (_binding == RootParallelBinding::CPU_THREADS && _auto_vectorize && _vector_depth == 0u) {
                    auto packed = vectorize_independent_elements(mapped, _planner.enabled ? _planner.max_cpu_vector_lanes : 16u);
                    if (packed.defined()) { return packed; }
                }
                mapped.CopyOnWrite()->annotations.erase(independent_elements_annotation);
                mapped.CopyOnWrite()->annotations.erase(mma_annotation);
                return mapped;
            }
            auto mapped = result.as_or_throw<tvm::tirx::For>();
            mapped.CopyOnWrite()->annotations.erase(deferred_pipeline_annotation);
            mapped.CopyOnWrite()->annotations.erase(reduction_contract_annotation);
            mapped.CopyOnWrite()->annotations.erase(materialized_exp_annotation);
            return mapped;
        }
        if (auto scope = loop->annotations.Get(execution_scope_annotation);
            scope && scope.value().as<tvm::ffi::String>() && scope.value().cast<tvm::ffi::String>() == "group") {
            if (_target_name != "metal") { _scope_error(loop, "group", "is not supported by this target's execution mapper"); }
            if (_logical_parallel_depth != 0u) { _scope_error(loop, "group", "requires a coordinate factorization for nested group bindings"); }
            return map_metal_cooperative_group(tvm::ffi::GetRef<tvm::tirx::For>(loop), _gpu_group_thread_limit, _shared_memory_limit, _cooperative_matrix, _metal_mpp, _planner, _plans, _readonly_inputs);
        }
        if (_target_name == "metal" && _planner.enabled && _planner.metal_subgroup_reductions &&
            _logical_parallel_depth == 0u) {
            auto constraint = loop->annotations.Get(execution_scope_annotation);
            auto scope = constraint ? constraint.value().as<tvm::ffi::String>() : tvm::ffi::Optional<tvm::ffi::String>{};
            auto automatic_or_subgroup = !constraint || (scope && scope.value() == "subgroup");
            if (automatic_or_subgroup) {
                auto mapped = try_map_metal_subgroup_reduction(
                    tvm::ffi::GetRef<tvm::tirx::For>(loop), _gpu_group_thread_limit,
                    _shared_memory_limit, _planner, _plans);
                if (mapped.defined()) { return mapped; }
                if (constraint) {
                    _scope_error(loop, "subgroup", "does not contain a realizable uniform reduction program");
                }
            }
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
            auto constraint = loop->annotations.Get(execution_scope_annotation);
            auto explicit_worker = constraint && constraint.value().as<tvm::ffi::String>() &&
                                   constraint.value().cast<tvm::ffi::String>() == "worker";
            auto extent = loop->extent.as<tvm::IntImmNode>();
            if (_planner.enabled && !explicit_worker && extent != nullptr && extent->value >= 0 &&
                static_cast<uint64_t>(extent->value) < _planner.min_cpu_parallel_tasks &&
                !cpu_body_has_expensive_call(loop->body)) {
                return _loop(loop, tvm::tirx::ForKind::kSerial, std::move(body), std::move(annotations));
            }
            return _loop(loop, tvm::tirx::ForKind::kParallel, std::move(body), std::move(annotations));
        }
        if (_binding == RootParallelBinding::GPU_GRID) {
            return _gpu_grid(loop, std::move(body));
        }
        throw std::runtime_error{"unresolved TileIR logical parallel binding"};
    }

public:
    ExecutionMapper(RootParallelBinding binding, uint32_t gpu_threads_per_block, uint32_t gpu_group_thread_limit, uint64_t shared_memory_limit,
                    bool vectorize, bool auto_vectorize, bool cooperative_matrix, bool metal_mpp, std::string target_name,
                    const PlannerOptions &planner, luisa::vector<GroupPlan> &plans, luisa::span<const tvm::tirx::BufferVar> readonly_inputs) noexcept
        : _binding{binding}, _gpu_threads_per_block{gpu_threads_per_block}, _gpu_group_thread_limit{gpu_group_thread_limit}, _shared_memory_limit{shared_memory_limit},
          _vectorize{vectorize}, _auto_vectorize{auto_vectorize}, _cooperative_matrix{cooperative_matrix}, _metal_mpp{metal_mpp}, _target_name{std::move(target_name)},
          _planner{planner}, _plans{plans}, _readonly_inputs{readonly_inputs} {}

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
    const CompileOptions &options, luisa::vector<GroupPlan> &plans) {
    auto binding = resolve_parallel_binding(target);
    if (options.cpu_matrix_backend != CpuMatrixBackend::REFERENCE &&
        binding != RootParallelBinding::CPU_THREADS) {
        throw std::runtime_error{"CPU matrix realization requires an LLVM target"};
    }
    if (options.cpu_math_backend != CpuMathBackend::REFERENCE &&
        binding != RootParallelBinding::CPU_THREADS) {
        throw std::runtime_error{"CPU array-math realization requires an LLVM target"};
    }
    if (options.planner.max_cpu_stack_bytes > 65536u ||
        (options.planner.max_cpu_stack_bytes != 0u && binding != RootParallelBinding::CPU_THREADS)) {
        throw std::runtime_error{"CPU stack planning requires an LLVM target and a byte budget in [0,65536]"};
    }
    if (options.planner.min_cpu_parallel_tasks == 0u ||
        (options.planner.min_cpu_parallel_tasks != 64u &&
         binding != RootParallelBinding::CPU_THREADS)) {
        throw std::runtime_error{"CPU parallel launch threshold requires an LLVM target and a positive task count"};
    }
    auto lanes = options.planner.max_cpu_vector_lanes;
    if (lanes < 16u || lanes > 128u || (lanes & (lanes - 1u)) != 0u ||
        (lanes != 16u && (binding != RootParallelBinding::CPU_THREADS || !options.auto_vectorize || !options.vectorize))) {
        throw std::runtime_error{"CPU vector packing requires 16/32/64/128 logical lanes and LLVM auto-vectorization when non-default"};
    }
    auto threads = uint32_t{1u};
    auto group_thread_limit = uint32_t{1u};
    auto shared_memory_limit = uint64_t{0u};
    if (binding == RootParallelBinding::GPU_GRID) {
        threads = 256u;
        group_thread_limit = threads;
        if (auto maximum = target->GetAttr<int64_t>("max_num_threads")) {
            if (maximum.value() <= 0 || maximum.value() > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error{"target thread capacity must be a positive uint32 value"};
            }
            group_thread_limit = static_cast<uint32_t>(maximum.value());
            // The reference worker launch width is a scheduling choice, not a
            // hardware limit imposed on cooperative group plans.
            threads = std::min(threads, group_thread_limit);
        }
        if (auto maximum = target->GetAttr<int64_t>("max_shared_memory_per_block")) {
            shared_memory_limit = static_cast<uint64_t>(std::max<int64_t>(0, maximum.value()));
        }
    }
    auto cooperative_matrix = options.cooperative_matrix && target->kind->name == "metal" && target->GetAttr<int64_t>("thread_warp_size").value_or(0) == 32;
    auto subgroup_reductions = options.planner.metal_subgroup_reductions;
    if (subgroup_reductions &&
        (!options.planner.enabled || !options.noalias || target->kind->name != "metal" ||
         target->GetAttr<int64_t>("thread_warp_size").value_or(0) != 32)) {
        throw std::runtime_error{"Metal SIMD-group reductions require an enabled planner, noalias, and a Metal target with thread_warp_size=32"};
    }
    if (options.forward_readonly_tile_loads && !options.metal_mpp && !subgroup_reductions && binding != RootParallelBinding::CPU_THREADS) {
        throw std::runtime_error{"read-only Tile view forwarding requires LLVM or Metal MPP realization"};
    }
    if (options.metal_mpp) {
        if (!cooperative_matrix || !options.planner.enabled) {
            throw std::runtime_error{"Metal MPP requires the Metal cooperative matrix capability and an enabled planner"};
        }
        auto capability = tvm::ffi::Function::GetGlobal("target.metal.mpp_memory_contract_version");
        if (!capability || (*capability)().cast<int64_t>() != 2) {
            throw std::runtime_error{"This TVM build lacks Metal MPP memory contract v2; use SIMD-group lowering or the documented TVM patch"};
        }
    }
    FunctionMap functions;
    for (auto &&[global, base_function] : module->functions) {
        auto function = base_function.as<tvm::tirx::PrimFunc>();
        if (!function) {
            throw std::runtime_error{"Tile TIRx execution mapping only accepts PrimFunc modules"};
        }
        auto mapped = function.value();
        if (options.cpu_matrix_backend == CpuMatrixBackend::CBLAS) {
            mapped = realize_cpu_whole_gemm(std::move(mapped), options.noalias);
            functions.Set(global, std::move(mapped));
            continue;
        }
        if (options.cpu_math_backend == CpuMathBackend::ACCELERATE) {
            mapped.CopyOnWrite()->body = realize_cpu_vector_math(mapped->body);
            mapped = tvm::WithAttr(
                std::move(mapped), cpu_math_realization_annotation,
                tvm::ffi::String{"accelerate"});
        }
        auto forward_views = options.forward_readonly_tile_loads || subgroup_reductions;
        auto views = forward_views ? forward_readonly_tile_loads(mapped, options.noalias, binding == RootParallelBinding::CPU_THREADS || subgroup_reductions) : ReadonlyViews{mapped->body, {}};
        mapped.CopyOnWrite()->body = std::move(views.body);
        mapped.CopyOnWrite()->body = schedule_pipelines(mapped->body, options.noalias, shared_memory_limit,
                                                        !options.metal_mpp && cooperative_matrix && options.planner.enabled && options.planner.max_pipeline_prefetch_scalars_per_lane != 0u);
        mapped.CopyOnWrite()->body = ExecutionMapper{binding, threads, group_thread_limit, shared_memory_limit, options.vectorize, options.auto_vectorize, cooperative_matrix, options.metal_mpp, std::string{target->kind->name}, options.planner, plans, views.inputs}(mapped->body);
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
        // Placement constraints were already validated. Retaining a manual
        // marker for the later CPU storage audit must not keep dead empty
        // storage alive: there is no allocation to remap in the first place.
        if (!empty || allocation->annotations.size() != allocation->annotations.count(manual_memory_annotation)) {
            return StmtMutator::VisitStmt_(allocation);
        }
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

void run_common_pipeline(tvm::IRModule &module, const CompileOptions &options, const tvm::Target &target, bool packed_api = true) {
    auto apply = [&module](tvm::transform::Pass pass) {
        module = run_pass(std::move(pass), std::move(module));
    };
    apply(tvm::tirx::transform::BindTarget(target));
    // Full/tail factoring can copy bodies containing loop-local definitions.
    // Let the native pass renew those definitions before subsequent analyses.
    apply(tvm::tirx::transform::ConvertSSA());
    if (options.pipeline == PipelineKind::TILE) {
        apply(tvm::tirx::transform::LowerTIRx());
    } else {
        apply(tvm::s_tir::transform::LowerInitBlock());
    }
    apply(tvm::s_tir::transform::UnifyThreadBinding());
    apply(tvm::tirx::transform::StmtSimplify());
    apply(tvm::tirx::transform::RemoveNoOp());
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
    if (packed_api) {
        apply(tvm::tirx::transform::MakePackedAPI());
        apply(tvm::tirx::transform::FP8StorageLegalize());
        apply(tvm::tirx::transform::BF16StorageLegalize());
    }
}

void finalize_host(tvm::IRModule &module, uint32_t stack_budget) {
    FunctionMap functions;
    for (auto &&[global, base_function] : module->functions) {
        auto function = base_function.as_or_throw<tvm::tirx::PrimFunc>();
        function.CopyOnWrite()->body = plan_cpu_storage(function->body, stack_budget);
        functions.Set(global, std::move(function));
    }
    module = make_module(std::move(functions), module->attrs, module->global_infos);
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

// Deliberately closed host grammar: a device artifact is not an interpreter
// for the host program. In particular a loop/conditional around a launch must
// not be mistaken for a single unconditional dispatch.
using HostBufferArguments = luisa::unordered_map<const tvm::tirx::VarNode *, uint32_t>;

[[nodiscard]] uint32_t buffer_argument(const tvm::Expr &expr, const HostBufferArguments &buffers) {
    auto projection = expr.as<tvm::CallNode>();
    if (!projection || !projection->op.same_as(tvm::tirx::builtin::buffer_data()) || projection->args.size() != 1u) {
        throw std::runtime_error{"device artifact requires direct buffer parameters (no scalars, pointer offsets, or host temporaries)"};
    }
    auto variable = projection->args[0u].as<tvm::tirx::VarNode>();
    auto iter = buffers.find(variable);
    if (iter == buffers.end()) { throw std::runtime_error{"device buffer does not originate at a host parameter"}; }
    return iter->second;
}

void collect_static_launch(const tvm::tirx::Stmt &stmt, const tvm::CallNode *&launch, HostBufferArguments &buffers) {
    if (auto sequence = stmt.as<tvm::tirx::SeqStmtNode>()) {
        for (auto &child : sequence->seq) { collect_static_launch(child, launch, buffers); }
        return;
    }
    if (auto declaration = stmt.as<tvm::tirx::DeclBufferNode>()) {
        // FlattenBuffer introduces pure aliases of parameter storage. Track
        // pointer identity through those declarations, without dropping an
        // allocation/copy or guessing from buffer names.
        auto index = buffer_argument(declaration->data, buffers);
        if (!buffers.emplace(declaration->buffer.get(), index).second) {
            throw std::runtime_error{"device artifact has a redefined host buffer"};
        }
        return;
    }
    if (auto evaluate = stmt.as<tvm::tirx::EvaluateNode>()) {
        if (auto constant = evaluate->value.as<tvm::IntImmNode>(); constant && constant->value == 0) { return; }
        if (auto call = evaluate->value.as<tvm::CallNode>();
            call && call->op.same_as(tvm::tirx::builtin::tvm_call_packed()) && launch == nullptr) {
            launch = call;
            return;
        }
    }
    throw std::runtime_error{"device artifact requires exactly one unconditional launch and no host effects; rejected " + std::string{stmt->GetTypeKey()}};
}

[[nodiscard]] DeviceArtifact extract_device_artifact(const tvm::tirx::PrimFunc &host,
                                                     const tvm::tirx::PrimFunc &device) {
    DeviceArtifact artifact;
    const tvm::CallNode *launch = nullptr;
    HostBufferArguments buffers;
    for (auto i = size_t{0u}; i < host->params.size(); i++) {
        if (!host->params[i]->ty.as<tvm::tirx::BufferTypeNode>()) { throw std::runtime_error{"device artifact host ABI requires buffer parameters"}; }
        buffers.emplace(host->params[i].get(), static_cast<uint32_t>(i));
    }
    collect_static_launch(host->body, launch, buffers);
    if (launch == nullptr) { throw std::runtime_error{"device artifact has no launch"}; }
    auto symbol = device->GetAttr<tvm::ffi::String>(tvm::attr::kGlobalSymbol);
    auto tags = device->GetAttr<tvm::ffi::Array<tvm::ffi::String>>(tvm::tirx::attr::kKernelLaunchParams);
    auto callee = launch->args.empty() ? nullptr : launch->args[0].as<tvm::tirx::StringImmNode>();
    if (!symbol || !tags || !callee || callee->value != symbol.value() ||
        launch->args.size() != 1u + device->params.size() + tags.value().size()) {
        throw std::runtime_error{"device artifact launch signature mismatch"};
    }
    artifact.entry = std::string{symbol.value()};
    for (auto i = size_t{0u}; i < device->params.size(); i++) {
        auto ptr = device->params[i]->ty.as<tvm::PointerTypeNode>();
        if (!ptr) { throw std::runtime_error{"device artifact requires a pointer-only device ABI"}; }
        artifact.buffer_arguments.emplace_back(buffer_argument(launch->args[i + 1u], buffers));
    }
    std::array<bool, 6u> seen{};
    for (auto i = size_t{0u}; i < tags.value().size(); i++) {
        constexpr std::array names{"blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x", "threadIdx.y", "threadIdx.z"};
        auto tag = std::string{tags.value()[i]};
        auto iter = std::find(names.begin(), names.end(), tag);
        auto extent = launch->args[1u + device->params.size() + i].as<tvm::IntImmNode>();
        if (iter == names.end() || !extent || extent->value <= 0 || extent->value > UINT32_MAX) {
            throw std::runtime_error{"device artifact requires static uint32 grid/block extents and no dynamic launch resources"};
        }
        auto index = static_cast<size_t>(iter - names.begin());
        if (seen[index]) { throw std::runtime_error{"duplicate device launch dimension"}; }
        seen[index] = true;
        (index < 3u ? artifact.grid[index] : artifact.block[index - 3u]) = static_cast<uint32_t>(extent->value);
    }
    return artifact;
}

}// namespace detail

DeviceCompilationResult compile_device(tvm::tirx::PrimFunc function, luisa::string_view name,
                                       const CompileOptions &options) noexcept {
    DeviceCompilationResult result;
    try {
        if (!function.defined() || name.empty()) { throw std::runtime_error{"device artifact requires a defined, named PrimFunc"}; }
        if (options.auto_vectorize && !options.vectorize) { throw std::runtime_error{"automatic vectorization requires vectorization"}; }
        tvm::Target target{tvm::ffi::String{options.target}};
        if (target->kind->name != "metal") { throw std::runtime_error{"device artifact currently supports only Metal"}; }
        // Use a host target only for TVMx's typed host/device partition pass.
        // There is no LLVM code generation or packed-function JIT here.
        tvm::Target bound{target, tvm::Target{tvm::ffi::String{"llvm"}}};
        auto symbol = tvm::ffi::String{std::string{name}};
        function = tvm::WithAttr(std::move(function), tvm::attr::kGlobalSymbol, symbol);
        if (options.noalias) { function = tvm::WithAttr(std::move(function), "tirx.noalias", true); }
        auto global = tvm::GlobalVar{symbol};
        auto module = detail::make_module({{global, std::move(function)}});
        module = detail::map_execution(std::move(module), target, options, result.plans);
        detail::run_common_pipeline(module, options, bound, false);
        if (module->functions.size() != 2u) { throw std::runtime_error{"device artifact requires exactly one host entry and one device entry"}; }
        auto host = module->functions.at(global).as_or_throw<tvm::tirx::PrimFunc>();
        for (auto &[device_global, base] : module->functions) {
            if (device_global.same_as(global)) { continue; }
            auto device = base.as_or_throw<tvm::tirx::PrimFunc>();
            result.artifact = detail::extract_device_artifact(host, device);
            auto device_module = detail::make_module({{device_global, device}}, module->attrs, module->global_infos);
            // Storage ABI legalization cannot consume the still-typed host
            // Buffer parameters. Run it on the pointer-ABI device partition.
            device_module = detail::run_pass(tvm::tirx::transform::FP8StorageLegalize(), std::move(device_module));
            device_module = detail::run_pass(tvm::tirx::transform::BF16StorageLegalize(), std::move(device_module));
            detail::finalize_device(device_module);
            result.artifact.function = device_module->functions.at(device_global).as_or_throw<tvm::tirx::PrimFunc>();
            tvm::tirx::PostOrderVisit(result.artifact.function->body, [&](const tvm::ffi::ObjectRef &node) {
                if (auto allocation = node.as<tvm::tirx::AllocBufferNode>()) {
                    result.artifact.requires_metal4 |= allocation->buffer.scope() == "metal.cooperative_tensor";
                }
            });
            // InspectSource returns the code generator's own output unchanged.
            // ABI metadata was already extracted from the typed launch above.
            auto compiled = detail::codegen(std::move(device_module), target);
            auto source = compiled->InspectSource("metal");
            result.artifact.source.assign(source.data(), source.size());
            if (source.empty()) { throw std::runtime_error{"Metal code generator returned no source artifact"}; }
        }
    } catch (const tvm::ffi::Error &error) {
        result.error = error.what();
    } catch (const std::exception &error) {
        result.error = error.what();
    } catch (...) {
        result.error = "unknown failure while compiling a TIRx device artifact";
    }
    return result;
}

CompilationResult compile(tvm::IRModule module, const CompileOptions &options) noexcept {
    if (!module.defined()) { return CompilationResult{luisa::string{"cannot compile an undefined TIRx module"}}; }
    if (options.target.empty()) { return CompilationResult{luisa::string{"TIRx target must not be empty"}}; }
    if (options.host.empty()) { return CompilationResult{luisa::string{"TIRx host target must not be empty"}}; }
    if (options.auto_vectorize && !options.vectorize) { return CompilationResult{luisa::string{"automatic vectorization requires vectorization to be enabled"}}; }
    try {
        tvm::Target device_target{tvm::ffi::String{options.target}};
        // MakePackedAPI replaces a CPU entry's target with its host target,
        // and LLVM codegen uses the module target for every function. Keep
        // both stages on the requested CPU ISA; GPU wrappers still use host.
        auto host_target = detail::is_host_target(device_target) ? tvm::Target{device_target, tvm::Target{}} : tvm::Target{tvm::ffi::String{options.host}};
        tvm::Target bound_target{device_target, host_target};
        luisa::vector<GroupPlan> plans;
        module = detail::map_execution(std::move(module), device_target, options, plans);
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
        detail::finalize_host(host_module, options.planner.enabled ? options.planner.max_cpu_stack_bytes : 0u);
        auto runtime_module = detail::codegen(std::move(host_module), host_target);
        for (auto &&partition : device_partitions) {
            auto device_module = detail::make_module(
                std::move(partition.functions), module->attrs, module->global_infos);
            detail::finalize_device(device_module);
            runtime_module->ImportModule(detail::codegen(std::move(device_module), partition.target));
        }
        return CompilationResult{std::move(runtime_module), std::move(plans)};
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
