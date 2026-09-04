#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>

#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

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

}// namespace luisa::compute::tile::bridge::tirx::detail
