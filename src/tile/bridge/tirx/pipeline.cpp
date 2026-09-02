#include <algorithm>
#include <limits>
#include <stdexcept>

#include <tvm/ffi/function.h>
#include <tvm/ir/module.h>
#include <tvm/ir/op.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/analysis.h>
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

    [[nodiscard]] static tvm::tirx::Stmt _ordered(tvm::tirx::For loop) {
        auto node = loop.CopyOnWrite();
        node->annotations.erase(logical_pipeline_annotation);
        node->annotations.erase(pipeline_window_annotation);
        node->annotations.erase(pipeline_interval_annotation);
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
        auto sequence = loop->body.as<tvm::tirx::SeqStmtNode>();
        if (sequence == nullptr) { return _ordered(loop); }
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
                    storage_bytes(buffer) == std::numeric_limits<uint64_t>::max()) { return _ordered(loop); }
                local.emplace(buffer.get(), buffer);
                allocations.push_back(allocation.value());
            } else if (auto stage = statement.as<tvm::tirx::AttrStmtNode>(); stage != nullptr && stage->attr_key == pipeline_stage_annotation) {
                segments.push_back(stage->body);
            } else {
                return _ordered(loop);
            }
        }
        if (segments.size() < 2u || local.empty()) { return _ordered(loop); }
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
        return _ordered(loop);
    }

public:
    PipelineScheduler(bool noalias, uint64_t version_budget) noexcept
        : _noalias{noalias}, _version_budget{version_budget} {}
};

}// namespace

tvm::tirx::Stmt schedule_pipelines(tvm::tirx::Stmt body, bool noalias, uint64_t shared_memory_limit) {
    auto budget = std::numeric_limits<uint64_t>::max();
    if (shared_memory_limit != 0u) {
        StorageFootprint footprint;
        footprint(body);
        if (footprint.has_group) { budget = shared_memory_limit - std::min(shared_memory_limit, footprint.bytes); }
    }
    return PipelineScheduler{noalias, budget}(std::move(body));
}

}// namespace luisa::compute::tile::bridge::tirx::detail
