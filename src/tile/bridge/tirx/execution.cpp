#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/mathematics.h>

#include "execution.h"

namespace luisa::compute::tile::bridge::tirx::detail {

namespace {

class VectorStorageExpander final : public tvm::tirx::StmtExprMutator {

private:
    tvm::PrimExpr _lane;
    tvm::PrimExpr _extent;
    uint64_t _lane_count;
    luisa::unordered_map<const tvm::tirx::VarNode *, tvm::tirx::BufferVar> _buffers;
    tvm::ffi::Array<tvm::tirx::Stmt> _allocations;

private:
    [[nodiscard]] tvm::ffi::Optional<tvm::PrimExpr> _predicate(
        const tvm::ffi::Optional<tvm::PrimExpr> &predicate) {
        return predicate ? VisitPrimExpr(predicate.value()) : tvm::ffi::Optional<tvm::PrimExpr>{};
    }

protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::AllocBufferNode *allocation) final {
        auto buffer = allocation->buffer;
        // These are virtual, compact compiler temporaries. An explicitly
        // placed or dynamically shaped resource needs its own address-map
        // transformation; do not discard such a contract here.
        auto offset = buffer->elem_offset.as<tvm::IntImmNode>();
        if (buffer.scope() != "local" || !buffer->strides.empty() ||
            buffer->layout || !buffer->allocated_addr.empty() ||
            offset == nullptr || offset->value != 0) {
            throw std::runtime_error{"TileIR vector scope requires compact compiler-local allocations"};
        }
        auto volume = _lane_count;
        for (auto &&dimension : buffer->shape) {
            auto extent = dimension.as<tvm::IntImmNode>();
            if (extent == nullptr || extent->value < 0 ||
                (extent->value != 0 && volume > static_cast<uint64_t>(std::numeric_limits<int64_t>::max() / extent->value))) {
                throw std::runtime_error{"TileIR vector private allocation needs a static shape within int64 range"};
            }
            volume *= static_cast<uint64_t>(extent->value);
        }
        // Address((local indices), lane) = flatten(local indices) * lanes +
        // lane. The trailing axis becomes a contiguous SIMD vector after
        // FlattenBuffer; no execution coordinate changes the logical Tile.
        auto shape = buffer->shape;
        shape.push_back(_extent);
        auto type = tvm::tirx::BufferType{
            buffer->storage_scope, buffer->dtype, std::move(shape), {}, buffer->elem_offset, buffer->data_alignment, buffer->offset_factor};
        auto expanded = tvm::tirx::BufferVar{buffer.name() + "_lanes", std::move(type), buffer.span()};
        _buffers.emplace(buffer.get(), expanded);
        _allocations.push_back(tvm::tirx::AllocBuffer{std::move(expanded), allocation->annotations, allocation->span});
        return tvm::tirx::Evaluate{tvm::IntImm::Int32(0)};
    }

    [[nodiscard]] tvm::Expr VisitExpr_(const tvm::tirx::BufferLoadNode *load) final {
        auto indices = load->indices.Map([this](const tvm::PrimExpr &index) { return VisitPrimExpr(index); });
        auto buffer = load->buffer;
        if (auto iter = _buffers.find(buffer.get()); iter != _buffers.end()) {
            buffer = iter->second;
            indices.push_back(_lane);
        }
        return tvm::tirx::BufferLoad{std::move(buffer), std::move(indices), _predicate(load->predicate), load->span};
    }

    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::BufferStoreNode *store) final {
        auto indices = store->indices.Map([this](const tvm::PrimExpr &index) { return VisitPrimExpr(index); });
        auto value = VisitPrimExpr(store->value);
        auto buffer = store->buffer;
        if (auto iter = _buffers.find(buffer.get()); iter != _buffers.end()) {
            buffer = iter->second;
            indices.push_back(_lane);
        }
        return tvm::tirx::BufferStore{std::move(buffer), std::move(value), std::move(indices), _predicate(store->predicate), store->span};
    }

    [[nodiscard]] tvm::Expr VisitExpr_(const tvm::tirx::VarNode *variable) final {
        if (_buffers.contains(variable)) {
            throw std::runtime_error{"TileIR vector private allocation cannot escape through an opaque buffer use"};
        }
        return StmtExprMutator::VisitExpr_(variable);
    }

public:
    explicit VectorStorageExpander(const tvm::tirx::For &loop)
        : _lane{loop->loop_var - loop->min}, _extent{loop->extent} {
        auto extent = _extent.as<tvm::IntImmNode>();
        if (extent == nullptr || extent->value <= 0 || extent->value > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error{"TileIR vector scope requires a positive static width representable by TIRx"};
        }
        _lane_count = static_cast<uint64_t>(extent->value);
    }

    [[nodiscard]] tvm::tirx::Stmt run(tvm::tirx::For loop) {
        auto body = VisitStmt(loop->body);
        loop.CopyOnWrite()->body = std::move(body);
        // Lexical compiler storage inside a vector instance is allocated once
        // for the whole vector and indexed separately by every lane. Parent
        // storage stays outside this visitor and is neither replicated nor
        // silently moved to a different resource class.
        _allocations.push_back(std::move(loop));
        return tvm::tirx::SeqStmt::Flatten(_allocations);
    }
};

class ProvenVectorGuards final : public tvm::tirx::StmtExprMutator {
private:
    const luisa::vector<tvm::PrimExpr> &_guards;

protected:
    [[nodiscard]] tvm::Expr VisitExpr(const tvm::Expr &expression) final {
        for (auto &guard : _guards) {
            if (tvm::ffi::StructuralEqual{}(expression, guard)) { return tvm::IntImm::Bool(true); }
        }
        return StmtExprMutator::VisitExpr(expression);
    }

public:
    explicit ProvenVectorGuards(const luisa::vector<tvm::PrimExpr> &guards) : _guards{guards} {}
};

// A lane-dependent lazy load otherwise scalarizes the entire SIMD operation.
// Version the pack once: the fast arm assumes only predicates checked for
// EVERY lane, while the slow arm retains the original guards and padding.
// Scalar conditions (including the ordered reduction coordinate) stay put.
class VectorGuardSpecializer final : public tvm::tirx::StmtMutator {
protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::ForNode *node) final {
        if (node->kind != tvm::tirx::ForKind::kVectorized) { return StmtMutator::VisitStmt_(node); }
        auto loop = tvm::ffi::GetRef<tvm::tirx::For>(node);
        auto minimum = loop->min.as<tvm::IntImmNode>();
        auto extent = loop->extent.as<tvm::IntImmNode>();
        if (minimum == nullptr || extent == nullptr || extent->value < 4 || extent->value > 16 || loop->step ||
            minimum->value > std::numeric_limits<int64_t>::max() - extent->value) { return loop; }
        luisa::unordered_set<const tvm::tirx::VarNode *> available;
        for (auto &variable : tvm::tirx::UndefinedVars(loop, {})) { available.emplace(variable.get()); }
        available.emplace(loop->loop_var.get());
        auto safe = [&](const tvm::PrimExpr &condition) {
            auto valid = true;
            auto uses_lane = false;
            tvm::tirx::PostOrderVisit(condition, [&](const tvm::ffi::ObjectRef &object) {
                if (auto variable = object.as<tvm::tirx::VarNode>()) {
                    valid &= available.contains(variable) &&
                             (variable->ty == tvm::PrimType::Int(64) || variable->ty == tvm::PrimType::Int(32) || variable->ty == tvm::PrimType::Bool());
                    uses_lane |= variable == loop->loop_var.get();
                } else if (auto division = object.as<tvm::tirx::FloorDivNode>()) {
                    auto divisor = division->b.as<tvm::IntImmNode>();
                    valid &= divisor != nullptr && divisor->value > 0;
                } else if (auto modulo = object.as<tvm::tirx::FloorModNode>()) {
                    auto divisor = modulo->b.as<tvm::IntImmNode>();
                    valid &= divisor != nullptr && divisor->value > 0;
                } else {
                    // No memory reads, calls, floating point, dynamic division,
                    // or local definitions may be speculated outside the pack.
                    valid &= object.as<tvm::IntImmNode>() || object.as<tvm::tirx::AddNode>() || object.as<tvm::tirx::SubNode>() ||
                             object.as<tvm::tirx::MulNode>() || object.as<tvm::tirx::MinNode>() || object.as<tvm::tirx::MaxNode>() ||
                             object.as<tvm::tirx::LTNode>() || object.as<tvm::tirx::LENode>() || object.as<tvm::tirx::GTNode>() ||
                             object.as<tvm::tirx::GENode>() || object.as<tvm::tirx::EQNode>() || object.as<tvm::tirx::NENode>() ||
                             object.as<tvm::tirx::AndNode>() || object.as<tvm::tirx::OrNode>() || object.as<tvm::tirx::NotNode>();
                }
            });
            return valid && uses_lane;
        };
        luisa::vector<tvm::PrimExpr> guards;
        std::function<void(const tvm::PrimExpr &)> collect = [&](const tvm::PrimExpr &condition) {
            if (auto conjunction = condition.as<tvm::tirx::AndNode>()) {
                collect(conjunction->a);
                collect(conjunction->b);
            } else if (safe(condition) && std::none_of(guards.begin(), guards.end(), [&](auto &guard) { return tvm::ffi::StructuralEqual{}(condition, guard); })) {
                guards.emplace_back(condition);
            }
        };
        auto versionable = true;
        tvm::tirx::PreOrderVisit(loop->body, [&](const tvm::ffi::ObjectRef &object) {
            // Address subexpressions of a predicated read are not necessarily
            // evaluated. Only specialize value-level lazy loads, not indices.
            if (object.as<tvm::tirx::BufferLoadNode>()) { return false; }
            if (object.as<tvm::tirx::LetNode>() || object.as<tvm::tirx::ReduceNode>() || object.as<tvm::tirx::ProducerLoadNode>()) {
                versionable = false;
                return false;
            }
            if (auto call = object.as<tvm::CallNode>()) {
                if (call->op.same_as(tvm::tirx::builtin::if_then_else()) && call->args.size() == 3u) {
                    collect(call->args[0].as_or_throw<tvm::PrimExpr>());
                } else {
                    versionable = false;
                }
                // Do not speculate predicates from either lazy branch. Even
                // integer arithmetic there may overflow on an untaken arm.
                return false;
            }
            if (auto child = object.as<tvm::tirx::ForNode>()) {
                auto count = child->extent.as<tvm::IntImmNode>();
                auto step = child->step ? child->step.value().as<tvm::IntImmNode>() : nullptr;
                versionable &= child->kind == tvm::tirx::ForKind::kSerial && !child->thread_binding && count != nullptr && count->value > 0 &&
                               (!child->step || (step != nullptr && step->value == 1));
            } else if (auto conditional = object.as<tvm::tirx::IfThenElseNode>()) {
                auto lane_dependent = false;
                tvm::tirx::PostOrderVisit(conditional->condition, [&](const tvm::ffi::ObjectRef &node) {
                    lane_dependent |= node.same_as(loop->loop_var);
                });
                auto before = guards.size();
                collect(conditional->condition);
                // A statement guard may protect an otherwise invalid store.
                // Version it only when every lane-dependent part contributes
                // a proved pack predicate. Scalar conditions remain inside
                // both versions and are never speculated.
                versionable &= !conditional->else_case &&
                               (!lane_dependent || guards.size() != before);
            } else if (auto store = object.as<tvm::tirx::BufferStoreNode>()) {
                versionable &= !store->predicate;
            } else if (object.as<tvm::tirx::StmtNode>() && !object.as<tvm::tirx::SeqStmtNode>()) {
                // No local definitions, early exits, or opaque effects. Every
                // selected predicate must already be evaluated for every lane
                // on each defined original execution.
                versionable = false;
            }
            return versionable;
        });
        // Bound both the predicate construction and code duplication. This is
        // one binary version, not an exponential decision tree per predicate.
        if (!versionable || guards.empty() || guards.size() > 8u) { return loop; }
        tvm::PrimExpr full = tvm::IntImm::Bool(true);
        for (auto &guard : guards) {
            for (auto lane = int64_t{0}; lane < extent->value; lane++) {
                auto coordinate = tvm::IntImm{loop->loop_var.ty(), minimum->value + lane};
                full = full && tvm::tirx::Substitute(guard, tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>{{loop->loop_var, coordinate}});
            }
        }
        auto fast = ProvenVectorGuards{guards}(loop);
        return tvm::tirx::IfThenElse{std::move(full), std::move(fast), std::move(loop)};
    }
};

}// namespace

tvm::tirx::Stmt privatize_vector_storage(const tvm::tirx::For &loop) {
    return VectorStorageExpander{loop}.run(loop);
}

tvm::tirx::Stmt vectorize_independent_elements(const tvm::tirx::For &loop, uint32_t max_lanes) {
    auto annotation = loop->annotations.Get(independent_elements_annotation);
    auto rank = annotation ? annotation.value().as<tvm::IntImmNode>() : nullptr;
    if (rank == nullptr || rank->value <= 0 ||
        loop->annotations.size() != 1u + loop->annotations.count(mma_annotation)) { return {}; }
    luisa::vector<const tvm::tirx::ForNode *> axes;
    auto inner = loop.get();
    for (auto i = int64_t{0}; i < rank->value; i++) {
        auto step = inner && inner->step ? inner->step.value().as<tvm::IntImmNode>() : nullptr;
        if (inner == nullptr || inner->kind != tvm::tirx::ForKind::kSerial || inner->thread_binding ||
            inner->loop_var.ty() != tvm::PrimType::Int(64) ||
            inner->min.as<tvm::IntImmNode>() == nullptr || inner->extent.as<tvm::IntImmNode>() == nullptr ||
            (inner->step && (step == nullptr || step->value != 1)) || (i != 0 && !inner->annotations.empty())) { return {}; }
        axes.push_back(inner);
        if (i + 1 != rank->value) { inner = inner->body.as<tvm::tirx::ForNode>(); }
    }
    auto extent = inner->extent.as<tvm::IntImmNode>()->value;
    if (extent < 4) { return {}; }
    // Logical packs may span several hardware vectors. Keeping independent
    // accumulators together exposes instruction-level parallelism to LLVM
    // without unrolling/reassociating the temporal recurrence itself.
    auto width = int64_t{4};
    while (width < 16 && extent >= width * 2) { width *= 2; }
    auto compatible = true;
    tvm::tirx::PostOrderVisit(inner->body, [&](const tvm::ffi::ObjectRef &node) {
        // These need storage privatization or a richer vectorizer. Automatic
        // packing must not make an otherwise valid reference kernel fail.
        compatible &= node.as<tvm::tirx::AllocBufferNode>() == nullptr && node.as<tvm::tirx::WhileNode>() == nullptr;
        if (auto child = node.as<tvm::tirx::ForNode>()) {
            compatible &= child->kind == tvm::tirx::ForKind::kSerial && !child->thread_binding;
        }
    });
    if (!compatible) { return {}; }

    // Factor the two innermost independent coordinates into a Cartesian
    // register pack. Keep row vectors separate: flattening the coordinates
    // into one wide vector loses contiguous load/store structure in TIRx.
    // Jam rectangular serial loops across the rows instead. Independence is
    // supplied by the element-domain contract, not an MMA marker; every
    // element retains its original temporal order and arithmetic expression.
    auto rows = int64_t{1};
    const tvm::tirx::ForNode *row_axis = nullptr;
    if (max_lanes > 16u && axes.size() >= 2u) {
        row_axis = axes[axes.size() - 2u];
        auto row_count = row_axis->extent.as<tvm::IntImmNode>()->value;
        while (rows * width * 2 <= max_lanes && rows * 2 <= row_count) { rows *= 2; }
    }
    if (rows > 1) {
        auto row_count = row_axis->extent.as<tvm::IntImmNode>()->value;
        auto row_pack = tvm::tirx::PrimVar{row_axis->loop_var->name + "_pack", tvm::PrimType::Int(64)};
        auto column_pack = tvm::tirx::PrimVar{inner->loop_var->name + "_pack", tvm::PrimType::Int(64)};
        auto invariant = [&](const tvm::PrimExpr &expression) {
            auto result = true;
            tvm::tirx::PostOrderVisit(expression, [&](const tvm::ffi::ObjectRef &node) {
                result &= !node.same_as(row_axis->loop_var) && !node.same_as(inner->loop_var);
            });
            return result;
        };
        // Only distribute statement sequences and rectangular serial loops.
        // Definitions, conditions, opaque effects, or lane-dependent bounds
        // need a richer region transform: retain the single-row fallback.
        std::function<tvm::tirx::Stmt(const tvm::tirx::Stmt &)> pack;
        pack = [&](const tvm::tirx::Stmt &statement) -> tvm::tirx::Stmt {
            if (auto sequence = statement.as<tvm::tirx::SeqStmtNode>()) {
                tvm::ffi::Array<tvm::tirx::Stmt> packed;
                for (auto &child : sequence->seq) {
                    auto result = pack(child);
                    if (!result.defined()) { return {}; }
                    packed.push_back(std::move(result));
                }
                return tvm::tirx::SeqStmt::Flatten(std::move(packed));
            }
            if (auto temporal = statement.as<tvm::tirx::ForNode>()) {
                if (temporal->kind != tvm::tirx::ForKind::kSerial || temporal->thread_binding || !temporal->annotations.empty() ||
                    !invariant(temporal->min) || !invariant(temporal->extent) || (temporal->step && !invariant(temporal->step.value()))) { return {}; }
                auto result = pack(temporal->body);
                if (!result.defined()) { return {}; }
                return tvm::tirx::For{temporal->loop_var, temporal->min, temporal->extent, temporal->kind, std::move(result), std::nullopt, {}, temporal->step, temporal->span};
            }
            if (!statement.as<tvm::tirx::BufferStoreNode>()) { return {}; }
            tvm::ffi::Array<tvm::tirx::Stmt> packed;
            for (auto row = int64_t{0}; row < rows; row++) {
                auto lane = tvm::tirx::PrimVar{inner->loop_var->name + "_lane", tvm::PrimType::Int(64)};
                tvm::ffi::Map<tvm::tirx::Var, tvm::Expr> coordinates{
                    {row_axis->loop_var, row_axis->min + row_pack * tvm::IntImm::Int64(rows) + tvm::IntImm::Int64(row)},
                    {inner->loop_var, inner->min + column_pack * tvm::IntImm::Int64(width) + lane}};
                auto result = tvm::tirx::Substitute(statement, coordinates);
                packed.push_back(tvm::tirx::For{lane, tvm::IntImm::Int64(0), tvm::IntImm::Int64(width), tvm::tirx::ForKind::kVectorized, std::move(result)});
            }
            return tvm::tirx::SeqStmt::Flatten(std::move(packed));
        };
        auto body = pack(inner->body);
        if (!body.defined()) { return vectorize_independent_elements(loop, 16u); }
        body = tvm::tirx::For{column_pack, tvm::IntImm::Int64(0), tvm::IntImm::Int64(extent / width), tvm::tirx::ForKind::kSerial, std::move(body)};
        body = tvm::tirx::For{row_pack, tvm::IntImm::Int64(0), tvm::IntImm::Int64(row_count / rows), tvm::tirx::ForKind::kSerial, std::move(body)};
        tvm::ffi::Array<tvm::tirx::Stmt> pieces{std::move(body)};
        if (extent % width != 0) {
            auto tail = tvm::tirx::For{inner->loop_var, inner->min + tvm::IntImm::Int64(extent / width * width), tvm::IntImm::Int64(extent % width), tvm::tirx::ForKind::kSerial, inner->body};
            pieces.push_back(tvm::tirx::For{row_axis->loop_var, row_axis->min, tvm::IntImm::Int64(row_count / rows * rows), tvm::tirx::ForKind::kSerial, std::move(tail)});
        }
        if (row_count % rows != 0) {
            // The row tail covers every column, disjoint from both the full
            // Cartesian packs and the column tail of their complete rows.
            auto tail = tvm::tirx::For{inner->loop_var, inner->min, inner->extent, tvm::tirx::ForKind::kSerial, inner->body};
            pieces.push_back(tvm::tirx::For{row_axis->loop_var, row_axis->min + tvm::IntImm::Int64(row_count / rows * rows), tvm::IntImm::Int64(row_count % rows), tvm::tirx::ForKind::kSerial, std::move(tail)});
        }
        body = tvm::tirx::SeqStmt::Flatten(std::move(pieces));
        for (auto i = axes.size() - 2u; i != 0u; i--) {
            auto outer = axes[i - 1u];
            auto annotations = outer->annotations;
            annotations.erase(independent_elements_annotation);
            annotations.erase(mma_annotation);
            body = tvm::tirx::For{outer->loop_var, outer->min, outer->extent, tvm::tirx::ForKind::kSerial, std::move(body), std::nullopt, std::move(annotations), outer->step, outer->span};
        }
        return VectorGuardSpecializer{}(body);
    }

    // Independence is supplied by the element-domain contract. This is only
    // a coordinate factorization; it neither re-proves memory dependencies
    // nor changes serial K/reduction recurrences inside an element instance.
    auto chunk = tvm::tirx::PrimVar{inner->loop_var->name + "_pack", tvm::PrimType::Int(64)};
    auto lane = tvm::tirx::PrimVar{inner->loop_var->name + "_lane", tvm::PrimType::Int(64)};
    auto coordinate = inner->min + chunk * tvm::IntImm::Int64(width) + lane;
    auto body = tvm::tirx::Substitute(inner->body, tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>{{inner->loop_var, coordinate}});
    body = tvm::tirx::For{lane, tvm::IntImm::Int64(0), tvm::IntImm::Int64(width), tvm::tirx::ForKind::kVectorized, std::move(body)};
    body = tvm::tirx::For{chunk, tvm::IntImm::Int64(0), tvm::IntImm::Int64(extent / width), tvm::tirx::ForKind::kSerial, std::move(body)};
    if (extent % width != 0) {
        auto tail = tvm::tirx::For{inner->loop_var, inner->min + tvm::IntImm::Int64(extent / width * width), tvm::IntImm::Int64(extent % width), tvm::tirx::ForKind::kSerial, inner->body};
        body = tvm::tirx::SeqStmt::Flatten(tvm::ffi::Array<tvm::tirx::Stmt>{std::move(body), std::move(tail)});
    }
    for (auto i = axes.size() - 1u; i != 0u; i--) {
        auto outer = axes[i - 1u];
        auto annotations = outer->annotations;
        annotations.erase(independent_elements_annotation);
        annotations.erase(mma_annotation);
        body = tvm::tirx::For{outer->loop_var, outer->min, outer->extent, tvm::tirx::ForKind::kSerial, std::move(body), std::nullopt, std::move(annotations), outer->step, outer->span};
    }
    return VectorGuardSpecializer{}(body);
}

namespace {

// Strip only empty structural-export placeholders. An allocation, stage cut,
// second effect domain, or lexical definition keeps the reference mapping.
[[nodiscard]] tvm::tirx::Stmt sole_effect(const tvm::tirx::Stmt &body) {
    if (auto evaluate = body.as<tvm::tirx::EvaluateNode>();
        evaluate && (evaluate->value.as<tvm::IntImmNode>() || evaluate->value.as<tvm::FloatImmNode>())) { return {}; }
    if (auto sequence = body.as<tvm::tirx::SeqStmtNode>()) {
        tvm::tirx::Stmt result;
        for (auto &child : sequence->seq) {
            auto effect = sole_effect(child);
            if (!effect.defined()) { continue; }
            if (result.defined()) { return body; }
            result = std::move(effect);
        }
        return result;
    }
    return body;
}

class ElementGridAudit final : public tvm::tirx::StmtExprVisitor {
private:
    const luisa::vector<const tvm::tirx::ForNode *> &_axes;
    luisa::vector<const tvm::tirx::ForNode *> _domain;
    luisa::unordered_set<const tvm::tirx::VarNode *> _reads, _writes, _escaped;
    uint32_t _stores{0u};

protected:
    void VisitStmt(const tvm::tirx::Stmt &statement) final {
        if (!statement.as<tvm::tirx::BufferStoreNode>() && !statement.as<tvm::tirx::IfThenElseNode>() &&
            !statement.as<tvm::tirx::SeqStmtNode>() && !statement.as<tvm::tirx::EvaluateNode>()) {
            valid = false;
            return;
        }
        StmtExprVisitor::VisitStmt(statement);
    }
    void VisitExpr_(const tvm::tirx::VarNode *variable) final { _escaped.emplace(variable); }
    void VisitExpr_(const tvm::tirx::ProducerLoadNode *) final { valid = false; }
    void VisitExpr_(const tvm::CallNode *call) final {
        static auto effects = tvm::Op::GetAttrMap<tvm::tirx::TCallEffectKind>("TCallEffectKind");
        auto op = call->op.as<tvm::Op>();
        valid &= !call->op.same_as(tvm::tirx::builtin::address_of()) && op && effects.count(op.value()) &&
                 effects[op.value()] <= static_cast<int64_t>(tvm::tirx::CallEffectKind::kPure);
        StmtExprVisitor::VisitExpr_(call);
    }
    void VisitExpr_(const tvm::tirx::BufferLoadNode *load) final {
        valid &= load->buffer.scope() == "global";
        _reads.emplace(load->buffer.get());
        StmtExprVisitor::VisitExpr_(load);
        if (load->predicate) { VisitExpr(load->predicate.value()); }
    }
    void VisitStmt_(const tvm::tirx::BufferStoreNode *store) final {
        valid &= ++_stores == 1u;
        valid &= store->buffer.scope() == "global";
        // Coordinate injectivity implies address injectivity only for this
        // compact buffer family. Arbitrary strides/layouts need their own
        // address-map proof before they may participate in fusion.
        valid &= store->buffer->strides.empty() && !store->buffer->layout &&
                 store->buffer->allocated_addr.empty() &&
                 store->indices.size() == store->buffer->shape.size();
        auto offset = store->buffer->elem_offset.as<tvm::IntImmNode>();
        valid &= offset && offset->value == 0;
        auto output_volume = uint64_t{1u};
        for (auto &extent : store->buffer->shape) {
            auto size = extent.as<tvm::IntImmNode>();
            if (!size || size->value <= 0 || output_volume > INT64_MAX / static_cast<uint64_t>(size->value)) {
                valid = false;
                break;
            }
            output_volume *= static_cast<uint64_t>(size->value);
        }
        _writes.emplace(store->buffer.get());
        tvm::ffi::Map<tvm::tirx::Var, tvm::Expr> origin;
        for (auto axis : _axes) { origin.Set(axis->loop_var, axis->min); }
        // Prove injectivity inside one logical program independently of the
        // annotation: every nontrivial local coordinate has its own output
        // coordinate, with coefficient one and no other local dependence.
        for (auto axis : _axes) {
            if (axis->extent.as<tvm::IntImmNode>()->value == 1) { continue; }
            auto covered = false;
            for (auto &index : store->indices) {
                auto base = tvm::tirx::Substitute(index, origin);
                covered |= prove_in_loop_domain(index - base == axis->loop_var - axis->min, _domain);
            }
            valid &= covered;
        }
        StmtExprVisitor::VisitStmt_(store);
        if (store->predicate) { VisitExpr(store->predicate.value()); }
    }

public:
    bool valid{true};
    ElementGridAudit(const tvm::tirx::ForNode *root, const luisa::vector<const tvm::tirx::ForNode *> &axes)
        : _axes{axes}, _domain{root} { _domain.insert(_domain.end(), axes.begin(), axes.end()); }
    [[nodiscard]] bool run(const tvm::tirx::Stmt &body) {
        VisitStmt(body);
        for (auto buffer : _writes) { valid &= !_reads.contains(buffer) && !_escaped.contains(buffer); }
        for (auto buffer : _reads) { valid &= !_escaped.contains(buffer); }
        return valid && !_writes.empty();
    }
};

}// namespace

tvm::tirx::Stmt try_map_gpu_elementwise(const tvm::tirx::Stmt &body, uint32_t max_threads,
                                        const PlannerOptions &options, luisa::vector<GroupPlan> &plans) {
    auto root_statement = sole_effect(body);
    auto root = root_statement.as<tvm::tirx::ForNode>();
    auto unit_domain = [](const tvm::tirx::ForNode *loop) {
        if (!loop || loop->kind != tvm::tirx::ForKind::kSerial || loop->thread_binding || loop->step ||
            loop->loop_var.ty() != tvm::PrimType::Int(64)) { return false; }
        auto extent = loop->extent.as<tvm::IntImmNode>();
        auto minimum = loop->min.as<tvm::IntImmNode>();
        return extent && minimum && extent->value > 0 && minimum->value <= INT64_MAX - extent->value;
    };
    if (!unit_domain(root) || root->annotations.size() != 1u || !root->annotations.count(logical_parallel_annotation) ||
        max_threads == 0u || options.threads_per_group > max_threads) { return {}; }
    auto elements = sole_effect(root->body);
    auto outer = elements.as<tvm::tirx::ForNode>();
    if (!outer || outer->annotations.size() != 1u) { return {}; }
    auto rank_attribute = outer->annotations.Get(independent_elements_annotation);
    auto rank = rank_attribute ? rank_attribute.value().as<tvm::IntImmNode>() : nullptr;
    if (!rank || rank->value <= 0 || rank->value > 16) { return {}; }
    luisa::vector<const tvm::tirx::ForNode *> axes;
    auto volume = uint64_t{1u};
    for (auto i = int64_t{0}; i < rank->value; i++) {
        auto axis = elements.as<tvm::tirx::ForNode>();
        if (!unit_domain(axis) || (i != 0 && !axis->annotations.empty())) { return {}; }
        auto extent = static_cast<uint64_t>(axis->extent.as<tvm::IntImmNode>()->value);
        if (volume > INT64_MAX / extent) { return {}; }
        volume *= extent;
        axes.emplace_back(axis);
        elements = axis->body;
    }
    auto programs = static_cast<uint64_t>(root->extent.as<tvm::IntImmNode>()->value);
    if (programs > INT64_MAX / volume || !ElementGridAudit{root, axes}.run(elements)) { return {}; }
    auto count = programs * volume;
    auto threads = options.threads_per_group ? options.threads_per_group : std::min<uint64_t>(count, std::min(max_threads, 256u));
    auto blocks = luisa::ceil_div(count, threads);
    if (blocks > INT64_MAX / threads) { return {}; }
    auto zero = tvm::IntImm::Int64(0);
    auto block = tvm::tirx::PrimVar{root->loop_var->name + "_element_block", tvm::PrimType::Int(64)};
    auto worker = tvm::tirx::PrimVar{root->loop_var->name + "_element_worker", tvm::PrimType::Int(64)};
    auto width = tvm::IntImm::Int64(static_cast<int64_t>(threads));
    auto linear = block * width + worker;
    auto tile_volume = tvm::IntImm::Int64(static_cast<int64_t>(volume));
    tvm::ffi::Map<tvm::tirx::Var, tvm::Expr> coordinates{{root->loop_var, root->min + tvm::floordiv(linear, tile_volume)}};
    auto local = tvm::floormod(linear, tile_volume);
    for (auto i = axes.size(); i != 0u; i--) {
        auto axis = axes[i - 1u];
        coordinates.Set(axis->loop_var, axis->min + tvm::floormod(local, axis->extent));
        local = tvm::floordiv(local, axis->extent);
    }
    auto result = tvm::tirx::Substitute(elements, coordinates);
    if (count % threads) { result = tvm::tirx::IfThenElse{linear < tvm::IntImm::Int64(static_cast<int64_t>(count)), std::move(result)}; }
    auto thread_axis = tvm::tirx::IterVar{tvm::Range::FromMinExtent(zero, width), worker, tvm::tirx::IterVarType::kThreadIndex, "threadIdx.x"};
    result = tvm::tirx::For{worker, zero, width, tvm::tirx::ForKind::kThreadBinding, std::move(result), thread_axis};
    auto block_count = tvm::IntImm::Int64(static_cast<int64_t>(blocks));
    auto block_axis = tvm::tirx::IterVar{tvm::Range::FromMinExtent(zero, block_count), block, tvm::tirx::IterVarType::kThreadIndex, "blockIdx.x"};
    result = tvm::tirx::For{block, zero, block_count, tvm::tirx::ForKind::kThreadBinding, std::move(result), block_axis};
    GroupPlan plan;
    plan.name = std::string{root->loop_var->name};
    plan.programs = programs;
    plan.threads = static_cast<uint32_t>(threads);
    plan.elementwise_elements_per_program = volume;
    plan.candidates_considered = 1u;
    plan.optimized = true;
    plans.emplace_back(std::move(plan));
    return result;
}

}// namespace luisa::compute::tile::bridge::tirx::detail
