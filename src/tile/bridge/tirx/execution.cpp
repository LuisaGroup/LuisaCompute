#include <cstdint>
#include <limits>
#include <stdexcept>

#include <tvm/tirx/buffer.h>
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

}// namespace

tvm::tirx::Stmt privatize_vector_storage(const tvm::tirx::For &loop) {
    return VectorStorageExpander{loop}.run(loop);
}

tvm::tirx::Stmt vectorize_independent_elements(const tvm::tirx::For &loop) {
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
    return body;
}

}// namespace luisa::compute::tile::bridge::tirx::detail
