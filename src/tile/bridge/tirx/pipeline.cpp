#include <algorithm>
#include <limits>
#include <stdexcept>

#include <tvm/ffi/function.h>
#include <tvm/ir/module.h>
#include <tvm/ir/op.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>

#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

#include "execution.h"

namespace luisa::compute::tile::bridge::tirx::detail {

namespace {

using Statements = tvm::ffi::Array<tvm::tirx::Stmt>;
using BufferMap = luisa::unordered_map<const tvm::tirx::VarNode *, tvm::tirx::BufferVar>;

[[nodiscard]] uint64_t storage_bytes(const tvm::tirx::BufferVar &buffer) {
    auto bytes = static_cast<uint64_t>((buffer->dtype.bits() * buffer->dtype.lanes() + 7) / 8);
    for (auto &&dimension : buffer->shape) {
        auto extent = dimension.as<tvm::IntImmNode>();
        if (extent == nullptr || extent->value < 0 ||
            (extent->value != 0 && bytes > std::numeric_limits<uint64_t>::max() / extent->value)) {
            return std::numeric_limits<uint64_t>::max();
        }
        bytes *= static_cast<uint64_t>(extent->value);
    }
    return bytes;
}

// A conservative capacity bound: summing all lexical allocations also counts
// mutually exclusive scopes and worker-private storage. It may decline an
// optimization, but never lets automatic versioning break a fitting group.
class StorageFootprint final : public tvm::tirx::StmtVisitor {
public:
    uint64_t bytes{0u};
    bool has_group{false};

protected:
    void VisitStmt_(const tvm::tirx::AllocBufferNode *allocation) final {
        auto size = storage_bytes(allocation->buffer);
        bytes += std::min(size, std::numeric_limits<uint64_t>::max() - bytes);
    }
    void VisitStmt_(const tvm::tirx::ForNode *loop) final {
        if (auto scope = loop->annotations.Get(execution_scope_annotation)) {
            auto name = scope.value().as<tvm::ffi::String>();
            has_group |= name && name.value() == "group";
        }
        StmtVisitor::VisitStmt_(loop);
    }
};

class StageAccess final : public tvm::tirx::StmtExprVisitor {
public:
    BufferMap reads;
    BufferMap writes;
    luisa::vector<tvm::tirx::BufferVar> read_order;
    luisa::vector<tvm::tirx::BufferVar> write_order;
    luisa::unordered_set<const tvm::tirx::VarNode *> local;
    bool opaque{false};

    void analyze(const tvm::tirx::Stmt &body) {
        (*this)(body);
        for (auto buffer : local) {
            reads.erase(buffer);
            writes.erase(buffer);
        }
    }

protected:
    void VisitBufferDef(const tvm::tirx::BufferVar &buffer, bool allocate) final {
        local.emplace(buffer.get());
        opaque |= !allocate;
        StmtExprVisitor::VisitBufferDef(buffer, allocate);
    }
    void VisitExpr_(const tvm::tirx::BufferLoadNode *load) final {
        if (reads.emplace(load->buffer.get(), load->buffer).second) { read_order.push_back(load->buffer); }
        StmtExprVisitor::VisitExpr_(load);
    }
    void VisitStmt_(const tvm::tirx::BufferStoreNode *store) final {
        if (writes.emplace(store->buffer.get(), store->buffer).second) { write_order.push_back(store->buffer); }
        StmtExprVisitor::VisitStmt_(store);
    }
    void VisitExpr_(const tvm::CallNode *call) final {
        // Buffer accesses are accounted for explicitly. An opaque intrinsic,
        // async effect, clock read, or external call needs a richer contract.
        static auto effects = tvm::Op::GetAttrMap<tvm::tirx::TCallEffectKind>("TCallEffectKind");
        auto op = call->op.as<tvm::Op>();
        opaque |= !op || effects.count(op.value()) == 0u ||
                  effects[op.value()] > static_cast<int64_t>(tvm::tirx::CallEffectKind::kPure);
        StmtExprVisitor::VisitExpr_(call);
    }
    void VisitStmt_(const tvm::tirx::AttrStmtNode *attribute) final {
        opaque |= attribute->attr_key != pipeline_stage_annotation;
        StmtExprVisitor::VisitStmt_(attribute);
    }
    void VisitStmt_(const tvm::tirx::ForNode *loop) final {
        opaque |= loop->kind != tvm::tirx::ForKind::kSerial || loop->thread_binding.has_value();
        StmtExprVisitor::VisitStmt_(loop);
    }
    void VisitStmt_(const tvm::tirx::BindNode *) final { opaque = true; }
    void VisitStmt_(const tvm::tirx::WhileNode *) final { opaque = true; }
    void VisitStmt_(const tvm::tirx::ReturnNode *) final { opaque = true; }
    void VisitStmt_(const tvm::tirx::BreakNode *) final { opaque = true; }
    void VisitStmt_(const tvm::tirx::ContinueNode *) final { opaque = true; }
    void VisitStmt_(const tvm::tirx::AssertStmtNode *) final { opaque = true; }
    void VisitStmt_(const tvm::tirx::SBlockNode *) final { opaque = true; }
    void VisitStmt_(const tvm::tirx::ScopeIdDefStmtNode *) final { opaque = true; }
    void VisitStmt_(const tvm::tirx::TilePrimitiveCallNode *) final { opaque = true; }
};

[[nodiscard]] bool touches(const StageAccess &access, const tvm::tirx::VarNode *buffer) {
    return access.reads.contains(buffer) || access.writes.contains(buffer);
}

[[nodiscard]] bool cross_iteration_hazard(const StageAccess &producer, const StageAccess &consumer,
                                          const BufferMap &iteration_local, bool noalias) {
    for (auto &&[buffer, unused] : producer.writes) {
        if (!iteration_local.contains(buffer) && touches(consumer, buffer)) { return true; }
    }
    for (auto &&[buffer, unused] : consumer.writes) {
        if (!iteration_local.contains(buffer) && touches(producer, buffer)) { return true; }
    }
    if (!noalias) {
        auto global_access = [](const BufferMap &buffers) {
            return std::any_of(buffers.begin(), buffers.end(), [](auto &&entry) {
                return entry.second.scope() != "local";
            });
        };
        auto producer_reads = global_access(producer.reads);
        auto producer_writes = global_access(producer.writes);
        auto consumer_reads = global_access(consumer.reads);
        auto consumer_writes = global_access(consumer.writes);
        // Different external BufferVars may overlap, even when the source
        // views have different constness or shapes. Read-only aliasing is fine.
        if ((producer_writes && (consumer_reads || consumer_writes)) ||
            (consumer_writes && (producer_reads || producer_writes))) { return true; }
    }
    return false;
}

[[nodiscard]] tvm::tirx::BufferRegion whole_buffer(const tvm::tirx::BufferVar &buffer) {
    tvm::ffi::Array<tvm::Range> region;
    for (auto &&extent : buffer->shape) { region.push_back(tvm::Range::FromMinExtent(0, extent)); }
    return tvm::tirx::BufferRegion{buffer, std::move(region)};
}

[[nodiscard]] tvm::tirx::Stmt stage_block(tvm::tirx::Stmt body, const StageAccess &access, int64_t phase) {
    tvm::ffi::Array<tvm::tirx::BufferRegion> reads;
    tvm::ffi::Array<tvm::tirx::BufferRegion> writes;
    for (auto &&buffer : access.read_order) {
        if (access.reads.contains(buffer.get())) { reads.push_back(whole_buffer(buffer)); }
    }
    for (auto &&buffer : access.write_order) {
        auto key = buffer.get();
        if (!access.writes.contains(key)) { continue; }
        writes.push_back(whole_buffer(buffer));
        // TVMx sizes versions from definition to last read. Include writes in
        // the conservative live range too: a late dead store must not clobber
        // the next iteration's still-live version.
        if (!access.reads.contains(key)) { reads.push_back(whole_buffer(buffer)); }
    }
    auto name = tvm::ffi::String{phase == 0 ? "tile_pipeline_producer" : "tile_pipeline_consumer"};
    // An attribute retains one phase as one scheduling component. In
    // particular, TVMx must not flatten a multi-statement phase into pieces
    // whose count no longer matches the stage/order arrays.
    body = tvm::tirx::AttrStmt{name, pipeline_stage_annotation, tvm::IntImm::Int64(phase), std::move(body)};
    auto block = tvm::tirx::SBlock{{}, std::move(reads), std::move(writes), name, std::move(body)};
    return tvm::tirx::SBlockRealize{{}, tvm::IntImm::Bool(true), std::move(block)};
}

// SBlock is a temporary adapter to an existing TVMx optimization, not the
// public bridge boundary. Restore flat TIRx allocations before execution
// mapping so worker/private and group/shared placement stay target-specific.
class StageEraser final : public tvm::tirx::StmtMutator {
protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::AttrStmtNode *attribute) final {
        return attribute->attr_key == pipeline_stage_annotation ?
                   VisitStmt(attribute->body) :
                   StmtMutator::VisitStmt_(attribute);
    }
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::SBlockRealizeNode *realize) final {
        auto &&block = realize->block;
        if (!realize->iter_values.empty() || !block->iter_vars.empty() || block->init || !block->match_buffers.empty()) {
            throw std::runtime_error{"software pipeline adapter expected an opaque, init-free block"};
        }
        Statements statements;
        for (auto &&buffer : block->alloc_buffers) { statements.push_back(tvm::tirx::AllocBuffer{buffer}); }
        statements.push_back(VisitStmt(block->body));
        auto body = tvm::tirx::SeqStmt::Flatten(statements);
        if (auto predicate = realize->predicate.as<tvm::IntImmNode>(); predicate != nullptr && predicate->value != 0) { return body; }
        return tvm::tirx::IfThenElse{realize->predicate, std::move(body)};
    }
};

[[nodiscard]] tvm::tirx::Stmt inject_pipeline(const tvm::tirx::For &loop,
                                              const Statements &phases,
                                              tvm::ffi::Array<tvm::tirx::BufferVar> allocations) {
    auto annotated = loop;
    auto node = annotated.CopyOnWrite();
    node->annotations.erase(logical_pipeline_annotation);
    node->annotations.erase(pipeline_window_annotation);
    node->annotations.erase(pipeline_interval_annotation);
    node->annotations.Set("software_pipeline_stage", tvm::ffi::Array<int64_t>{0, 1});
    node->annotations.Set("software_pipeline_order", tvm::ffi::Array<int64_t>{0, 1});
    node->body = tvm::tirx::SBlockRealize{
        {}, tvm::IntImm::Bool(true), tvm::tirx::SBlock{tvm::ffi::String{"tile_pipeline"}, tvm::tirx::SeqStmt{phases}, std::move(allocations)}};
    // Borrow this loop as a standalone pass input. All surrounding buffers
    // and coordinates are explicit free parameters; no source is serialized.
    auto function = tvm::tirx::PrimFunc{tvm::tirx::UndefinedVars(annotated, {}), annotated};
    auto global = tvm::GlobalVar{"tile_pipeline"};
    tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> functions{{global, std::move(function)}};
    static auto make_module = tvm::ffi::Function::GetGlobalRequired("ir.IRModule");
    static auto run_pass = tvm::ffi::Function::GetGlobalRequired("transform.RunPass");
    auto module = make_module(std::move(functions), tvm::DictAttrs{},
                              tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Array<tvm::GlobalInfo>>{})
                      .cast<tvm::IRModule>();
    module = run_pass(tvm::s_tir::transform::InjectSoftwarePipeline(), std::move(module)).cast<tvm::IRModule>();
    auto transformed = module->functions.at(global).as<tvm::tirx::PrimFunc>().value();
    return StageEraser{}(transformed->body);
}

class PipelineScheduler final : public tvm::tirx::StmtMutator {
private:
    bool _noalias;
    uint64_t _version_budget;
    bool _defer_prefetch;

    [[nodiscard]] static int64_t _integer_annotation(const tvm::tirx::For &loop, const char *name, int64_t fallback) {
        if (auto value = loop->annotations.Get(name)) {
            auto integer = value.value().as<tvm::IntImm>();
            if (!integer || integer.value()->value < 0) {
                throw std::runtime_error{"invalid native Tile pipeline policy annotation"};
            }
            return integer.value()->value;
        }
        return fallback;
    }

    [[nodiscard]] static tvm::tirx::Stmt _ordered(tvm::tirx::For loop, bool defer = false) {
        auto node = loop.CopyOnWrite();
        node->annotations.erase(logical_pipeline_annotation);
        node->annotations.erase(pipeline_window_annotation);
        node->annotations.erase(pipeline_interval_annotation);
        if (defer) { node->annotations.Set(deferred_pipeline_annotation, tvm::IntImm::Int64(1)); }
        node->body = StageEraser{}(node->body);
        return loop;
    }

protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::ForNode *original) final {
        auto loop = StmtMutator::VisitStmt_(original).as_or_throw<tvm::tirx::For>();
        if (loop->annotations.count(logical_pipeline_annotation) == 0u) { return loop; }
        auto window = _integer_annotation(loop, pipeline_window_annotation, 0);
        auto interval = _integer_annotation(loop, pipeline_interval_annotation, 1);
        if (interval == 0) { throw std::runtime_error{"pipeline initiation interval must be positive"}; }
        auto extent = loop->extent.as<tvm::IntImmNode>();
        // The native pass has unit issue spacing. Other timing policies need
        // a target latency model; retain their ordered reference execution.
        if (interval != 1 || window == 1 || extent == nullptr || extent->value <= 1) { return _ordered(loop); }
        auto ordered = [&] { return _ordered(loop, _defer_prefetch); };
        auto sequence = loop->body.as<tvm::tirx::SeqStmtNode>();
        if (sequence == nullptr) { return ordered(); }
        BufferMap local;
        tvm::ffi::Array<tvm::tirx::AllocBuffer> allocations;
        Statements segments;
        for (auto &&statement : sequence->seq) {
            if (auto allocation = statement.as<tvm::tirx::AllocBuffer>()) {
                // Do not lose resource constraints, custom placement, or
                // symbolic extents in a pass that rebuilds BufferVars.
                auto &&buffer = allocation.value()->buffer;
                auto offset = buffer->elem_offset.as<tvm::IntImmNode>();
                if (!allocation.value()->annotations.empty() || buffer.scope() != "local" ||
                    buffer->layout || !buffer->allocated_addr.empty() || !buffer->strides.empty() ||
                    offset == nullptr || offset->value != 0 ||
                    storage_bytes(buffer) == std::numeric_limits<uint64_t>::max()) { return ordered(); }
                local.emplace(buffer.get(), buffer);
                allocations.push_back(allocation.value());
            } else if (auto stage = statement.as<tvm::tirx::AttrStmtNode>(); stage != nullptr && stage->attr_key == pipeline_stage_annotation) {
                segments.push_back(stage->body);
            } else {
                return ordered();
            }
        }
        if (segments.size() < 2u || local.empty()) { return ordered(); }
        // Source stages are not hardware cycles. Find a safe cut into an
        // early producer and a late consumer; all other source cuts retain
        // their order inside those phases. This uses at most two in-flight
        // iterations, independently of source stage count or buffer count.
        for (auto cut = 1u; cut < segments.size(); cut++) {
            Statements early;
            Statements late;
            for (auto i = 0u; i < segments.size(); i++) { (i < cut ? early : late).push_back(segments[i]); }
            auto producer_body = tvm::tirx::SeqStmt::Flatten(early);
            auto consumer_body = tvm::tirx::SeqStmt::Flatten(late);
            StageAccess producer;
            StageAccess consumer;
            producer.analyze(producer_body);
            consumer.analyze(consumer_body);
            if (producer.opaque || consumer.opaque || cross_iteration_hazard(producer, consumer, local, _noalias)) { continue; }
            auto has_transfer = false;
            auto safe = true;
            auto extra_bytes = uint64_t{0u};
            tvm::ffi::Array<tvm::tirx::BufferVar> pipeline_allocations;
            Statements unused_allocations;
            for (auto &&allocation : allocations) {
                auto &&buffer = allocation->buffer;
                auto key = buffer.get();
                if (!touches(producer, key) && !touches(consumer, key)) {
                    unused_allocations.push_back(allocation);
                    continue;
                }
                pipeline_allocations.push_back(buffer);
                if (producer.reads.contains(key) && !producer.writes.contains(key)) { safe = false; }
                if (producer.writes.contains(key) && touches(consumer, key)) {
                    has_transfer |= consumer.reads.contains(key);
                    auto bytes = storage_bytes(buffer);
                    if (bytes > _version_budget - extra_bytes) {
                        safe = false;
                        break;
                    }
                    extra_bytes += bytes;
                }
            }
            if (!safe || !has_transfer) { continue; }
            Statements phases{stage_block(std::move(producer_body), producer, 0),
                              stage_block(std::move(consumer_body), consumer, 1)};
            auto scheduled = inject_pipeline(loop, phases, std::move(pipeline_allocations));
            _version_budget -= extra_bytes;
            unused_allocations.push_back(std::move(scheduled));
            return tvm::tirx::SeqStmt::Flatten(unused_allocations);
        }
        return ordered();
    }

public:
    PipelineScheduler(bool noalias, uint64_t version_budget, bool defer_prefetch) noexcept
        : _noalias{noalias}, _version_budget{version_budget}, _defer_prefetch{defer_prefetch} {}
};

using Coordinates = tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>;

[[nodiscard]] bool pure_coordinate(const tvm::PrimExpr &expression, const tvm::tirx::PrimVar &exclude = {}) {
    auto pure = true;
    tvm::tirx::PostOrderVisit(expression, [&](const tvm::ffi::ObjectRef &node) {
        pure &= node.as<tvm::tirx::BufferLoadNode>() == nullptr && node.as<tvm::CallNode>() == nullptr;
        pure &= !exclude.defined() || !node.same_as(exclude);
    });
    return pure;
}

[[nodiscard]] bool readonly_global_value(const tvm::PrimExpr &expression) {
    auto valid = true;
    auto reads = 0u;
    tvm::tirx::PostOrderVisit(expression, [&](const tvm::ffi::ObjectRef &node) {
        if (auto load = node.as<tvm::tirx::BufferLoadNode>()) {
            valid &= load->buffer.scope() == "global" && !load->predicate;
            for (auto &&index : load->indices) { valid &= pure_coordinate(index); }
            reads++;
        }
        if (auto call = node.as<tvm::CallNode>()) {
            // Retain short-circuit bounds::zero loads; no opaque effect or
            // indirect shared/local value may be speculated across a cycle.
            valid &= call->op.same_as(tvm::tirx::builtin::if_then_else());
        }
    });
    return valid && reads != 0u;
}

void flatten_sequence(const tvm::tirx::Stmt &statement, Statements &statements) {
    if (auto sequence = statement.as<tvm::tirx::SeqStmtNode>()) {
        for (auto &&child : sequence->seq) { flatten_sequence(child, statements); }
    } else {
        statements.push_back(statement);
    }
}

[[nodiscard]] bool empty_statement(const tvm::tirx::Stmt &statement) {
    auto evaluate = statement.as<tvm::tirx::EvaluateNode>();
    auto constant = evaluate == nullptr ? nullptr : evaluate->value.as<tvm::IntImmNode>();
    return constant != nullptr && constant->value == 0;
}

struct PrefetchedCopy {
    tvm::tirx::BufferStore store;
    tvm::PrimExpr guard;
};

// Unroll only the bounded worker-copy prefix. This is after distribution, so
// the budget counts actual per-worker values, not the original whole Tile.
class WorkerCopyCollector {
private:
    const BufferMap &_shared;
    const tvm::tirx::PrimVar &_iteration;
    uint64_t _budget;

public:
    luisa::vector<PrefetchedCopy> copies;

    WorkerCopyCollector(const BufferMap &shared, const tvm::tirx::PrimVar &iteration, uint64_t budget)
        : _shared{shared}, _iteration{iteration}, _budget{budget} {}

    [[nodiscard]] bool append(const tvm::tirx::Stmt &statement, Coordinates coordinates = {},
                              tvm::PrimExpr guard = tvm::IntImm::Bool(true), uint32_t depth = 0u) {
        if (depth > 8u) { return false; }
        if (empty_statement(statement)) { return true; }
        if (auto sequence = statement.as<tvm::tirx::SeqStmtNode>()) {
            for (auto &&child : sequence->seq) {
                if (auto bind = child.as<tvm::tirx::BindNode>()) {
                    auto value = bind->value.as<tvm::PrimExpr>();
                    if (!value) { return false; }
                    auto substituted = tvm::tirx::Substitute(value.value(), coordinates);
                    if (!pure_coordinate(substituted) && !readonly_global_value(substituted)) { return false; }
                    coordinates.Set(bind->var, std::move(substituted));
                } else if (!append(child, coordinates, guard, depth + 1u)) {
                    return false;
                }
            }
            return true;
        }
        if (auto loop = statement.as<tvm::tirx::ForNode>()) {
            auto minimum = loop->min.as<tvm::IntImmNode>();
            auto extent = loop->extent.as<tvm::IntImmNode>();
            auto step = loop->step ? loop->step.value().as<tvm::IntImmNode>() : nullptr;
            if (loop->kind != tvm::tirx::ForKind::kSerial || loop->thread_binding || !loop->annotations.empty() ||
                minimum == nullptr || minimum->value < 0 || extent == nullptr || extent->value <= 0 ||
                static_cast<uint64_t>(extent->value) > _budget || minimum->value > std::numeric_limits<int64_t>::max() - extent->value ||
                (loop->step && (step == nullptr || step->value != 1))) { return false; }
            for (auto i = int64_t{0}; i < extent->value; i++) {
                auto replacement = coordinates;
                replacement.Set(loop->loop_var, tvm::IntImm{loop->loop_var.ty(), minimum->value + i});
                if (!append(loop->body, std::move(replacement), guard, depth + 1u)) { return false; }
            }
            return true;
        }
        if (auto branch = statement.as<tvm::tirx::IfThenElseNode>()) {
            auto condition = tvm::tirx::Substitute(branch->condition, coordinates);
            if (branch->else_case || !pure_coordinate(condition, _iteration)) { return false; }
            return append(branch->then_case, std::move(coordinates), guard && condition, depth + 1u);
        }
        if (auto store = statement.as<tvm::tirx::BufferStoreNode>()) {
            if (store->predicate || !_shared.contains(store->buffer.get()) || store->buffer->dtype != tvm::PrimType::Float(32) || copies.size() >= _budget) { return false; }
            auto indices = tvm::tirx::Substitute(store->indices, coordinates);
            for (auto &&index : indices) {
                if (!pure_coordinate(index, _iteration)) { return false; }
            }
            auto value = tvm::tirx::Substitute(store->value, coordinates);
            if (!readonly_global_value(value)) { return false; }
            copies.emplace_back(PrefetchedCopy{tvm::tirx::BufferStore{store->buffer, std::move(value), std::move(indices)}, std::move(guard)});
            return true;
        }
        return false;
    }
};

// No global write (even through a different, possibly aliased argument),
// explicit synchronization, or unknown effect may occur in the consumer.
// The only effectful calls accepted here are the already emitted matrix
// loads/accumulates; the caller has separately proved the closed recurrence.
[[nodiscard]] bool readonly_matrix_consumer(const tvm::tirx::Stmt &statement,
                                            const tvm::tirx::Stmt &barrier, const BufferMap &shared) {
    if (statement.same_as(barrier) || empty_statement(statement)) { return true; }
    if (auto sequence = statement.as<tvm::tirx::SeqStmtNode>()) {
        return std::all_of(sequence->seq.begin(), sequence->seq.end(), [&](auto &&child) { return readonly_matrix_consumer(child, barrier, shared); });
    }
    if (auto loop = statement.as<tvm::tirx::ForNode>()) {
        return loop->kind == tvm::tirx::ForKind::kSerial && !loop->thread_binding && loop->annotations.empty() &&
               pure_coordinate(loop->min) && pure_coordinate(loop->extent) && (!loop->step || pure_coordinate(loop->step.value())) &&
               readonly_matrix_consumer(loop->body, barrier, shared);
    }
    if (auto branch = statement.as<tvm::tirx::IfThenElseNode>()) {
        return pure_coordinate(branch->condition) && readonly_matrix_consumer(branch->then_case, barrier, shared) &&
               (!branch->else_case || readonly_matrix_consumer(branch->else_case.value(), barrier, shared));
    }
    if (auto allocation = statement.as<tvm::tirx::AllocBufferNode>()) {
        return allocation->annotations.empty() && allocation->buffer.scope() == "metal.simdgroup";
    }
    auto evaluate = statement.as<tvm::tirx::EvaluateNode>();
    auto call = evaluate == nullptr ? nullptr : evaluate->value.as<tvm::CallNode>();
    if (call == nullptr) { return false; }
    static const auto load_op = tvm::Op::Get("tirx.simdgroup_load");
    static const auto mma_op = tvm::Op::Get("tirx.simdgroup_multiply_accumulate");
    if (!call->op.same_as(load_op) && !call->op.same_as(mma_op)) { return false; }
    auto valid = true;
    tvm::tirx::PostOrderVisit(evaluate->value, [&](const tvm::ffi::ObjectRef &node) {
        if (auto load = node.as<tvm::tirx::BufferLoadNode>()) {
            valid &= shared.contains(load->buffer.get()) && !load->predicate;
            for (auto &&index : load->indices) { valid &= pure_coordinate(index); }
        }
        if (auto nested = node.as<tvm::CallNode>()) {
            valid &= nested == call || nested->op.same_as(tvm::tirx::builtin::address_of());
        }
    });
    return valid;
}

[[nodiscard]] tvm::tirx::Stmt guarded(tvm::PrimExpr condition, tvm::tirx::Stmt statement) {
    if (auto constant = condition.as<tvm::IntImmNode>(); constant != nullptr && constant->value != 0) { return statement; }
    return tvm::tirx::IfThenElse{std::move(condition), std::move(statement)};
}

}// namespace

tvm::tirx::Stmt try_prefetch_matrix_pipeline(const tvm::tirx::For &loop, const tvm::tirx::Stmt &compiler_barrier,
                                             uint32_t scalar_budget, GroupPlan &plan) {
    // This bounded realization is optional, not an exhaustive register plan.
    auto budget = std::min<uint64_t>(scalar_budget, 256u);
    budget -= std::min(budget, plan.prefetch_storage_scalars_per_lane);
    auto minimum = loop->min.as<tvm::IntImmNode>();
    auto extent = loop->extent.as<tvm::IntImmNode>();
    if (budget == 0u || loop->loop_var.ty() != tvm::PrimType::Int(64) || minimum == nullptr || minimum->value < 0 || extent == nullptr || extent->value <= 1 ||
        minimum->value > std::numeric_limits<int64_t>::max() - extent->value) { return {}; }
    Statements statements;
    flatten_sequence(loop->body, statements);
    BufferMap shared;
    Statements allocations;
    WorkerCopyCollector collector{shared, loop->loop_var, budget};
    auto cut = size_t{0u};
    for (; cut < statements.size(); cut++) {
        auto &&statement = statements[cut];
        if (empty_statement(statement) || statement.same_as(compiler_barrier)) { continue; }
        if (auto allocation = statement.as<tvm::tirx::AllocBufferNode>(); allocation != nullptr && allocation->buffer.scope() == "shared") {
            if (!allocation->annotations.empty() || storage_bytes(allocation->buffer) == std::numeric_limits<uint64_t>::max()) { return {}; }
            shared.emplace(allocation->buffer.get(), allocation->buffer);
            allocations.push_back(statement);
            continue;
        }
        auto old_count = collector.copies.size();
        if (!statement.as<tvm::tirx::ForNode>() || !collector.append(statement)) {
            collector.copies.resize(old_count);
            break;
        }
    }
    if (collector.copies.empty() || cut == statements.size()) { return {}; }
    Statements consumer;
    for (auto i = cut; i < statements.size(); i++) {
        if (!readonly_matrix_consumer(statements[i], compiler_barrier, shared)) { return {}; }
        if (!empty_statement(statements[i])) { consumer.push_back(statements[i]); }
    }
    // Reuse cannot start until every subgroup has finished reading this slot.
    if (consumer.empty() || !consumer.back().same_as(compiler_barrier)) { return {}; }
    auto storage = tvm::tirx::decl_buffer({tvm::IntImm::Int64(static_cast<int64_t>(collector.copies.size()))},
                                          tvm::PrimType::Float(32), loop->loop_var->name + "_prefetch", "local");
    allocations.push_back(tvm::tirx::AllocBuffer{storage});
    Statements initial, reads, writes;
    auto one = tvm::IntImm::Int64(1);
    Coordinates first{{loop->loop_var, loop->min}};
    Coordinates next{{loop->loop_var, loop->loop_var + one}};
    for (auto i = size_t{0u}; i < collector.copies.size(); i++) {
        auto &&copy = collector.copies[i];
        tvm::ffi::Array<tvm::PrimExpr> slot{tvm::IntImm::Int64(static_cast<int64_t>(i))};
        initial.push_back(guarded(copy.guard, tvm::tirx::BufferStore{storage, tvm::tirx::Substitute(copy.store->value, first), slot}));
        reads.push_back(guarded(copy.guard, tvm::tirx::BufferStore{storage, tvm::tirx::Substitute(copy.store->value, next), slot}));
        writes.push_back(guarded(copy.guard, tvm::tirx::BufferStore{copy.store->buffer, tvm::tirx::BufferLoad{storage, slot}, copy.store->indices}));
    }
    writes.push_back(compiler_barrier);
    // The final cycle does not issue speculative out-of-range global reads.
    writes.push_back(guarded(loop->loop_var < loop->min + loop->extent - one, tvm::tirx::SeqStmt::Flatten(reads)));
    writes.push_back(tvm::tirx::SeqStmt::Flatten(consumer));
    auto scheduled = loop;
    auto node = scheduled.CopyOnWrite();
    node->annotations.erase(deferred_pipeline_annotation);
    node->body = tvm::tirx::SeqStmt::Flatten(writes);
    allocations.push_back(tvm::tirx::SeqStmt::Flatten(initial));
    allocations.push_back(std::move(scheduled));
    plan.prefetched_pipeline_loops++;
    plan.prefetch_storage_scalars_per_lane += collector.copies.size();
    return tvm::tirx::SeqStmt::Flatten(allocations);
}

tvm::tirx::Stmt schedule_pipelines(tvm::tirx::Stmt body, bool noalias, uint64_t shared_memory_limit, bool defer_prefetch) {
    auto budget = std::numeric_limits<uint64_t>::max();
    if (shared_memory_limit != 0u) {
        StorageFootprint footprint;
        footprint(body);
        if (footprint.has_group) { budget = shared_memory_limit - std::min(shared_memory_limit, footprint.bytes); }
    }
    return PipelineScheduler{noalias, budget, defer_prefetch}(std::move(body));
}

}// namespace luisa::compute::tile::bridge::tirx::detail
