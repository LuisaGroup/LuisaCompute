#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

#include "execution.h"

namespace luisa::compute::tile::bridge::tirx::detail {

namespace {

[[nodiscard]] uint64_t static_extent(const tvm::PrimExpr &expression) {
    auto constant = expression.as<tvm::IntImmNode>();
    if (constant == nullptr || constant->value < 0) {
        throw std::runtime_error{"cooperative Tile execution requires nonnegative static extents"};
    }
    return static_cast<uint64_t>(constant->value);
}

void validate_domain(const tvm::tirx::ForNode *loop) {
    auto step = loop->step ? loop->step.value().as<tvm::IntImmNode>() : nullptr;
    if (loop->kind != tvm::tirx::ForKind::kSerial || loop->thread_binding ||
        (loop->step && (step == nullptr || step->value != 1))) {
        throw std::runtime_error{"cooperative Tile execution requires serial unit-step domains before binding"};
    }
}

struct ElementDomain {
    luisa::vector<const tvm::tirx::ForNode *> axes;
    tvm::tirx::Stmt body;
    uint64_t count{1u};
};

[[nodiscard]] ElementDomain element_domain(const tvm::tirx::ForNode *loop) {
    auto rank = int64_t{1};
    if (auto annotation = loop->annotations.Get(independent_elements_annotation)) {
        auto value = annotation.value().as<tvm::IntImmNode>();
        if (value == nullptr || value->value <= 0) {
            throw std::runtime_error{"cooperative Tile element domain requires a positive static rank"};
        }
        rank = value->value;
    }
    ElementDomain result;
    auto current = loop;
    for (auto i = int64_t{0}; i < rank; i++) {
        if (current == nullptr || (i != 0 && !current->annotations.empty()) ||
            current->min.as<tvm::IntImmNode>() == nullptr) {
            throw std::runtime_error{"cooperative Tile element domain requires a perfect static rectangular nest"};
        }
        validate_domain(current);
        auto extent = static_extent(current->extent);
        if (extent != 0u && result.count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / extent) {
            throw std::runtime_error{"cooperative Tile element domain exceeds int64 range"};
        }
        result.count *= extent;
        result.axes.emplace_back(current);
        result.body = current->body;
        current = current->body.as<tvm::tirx::ForNode>();
    }
    return result;
}

[[nodiscard]] tvm::tirx::Stmt metal_group_barrier() {
    // TIRx's built-in shared barrier only fences threadgroup memory on Metal.
    // A Tile phase may also write a global view consumed by the next phase.
    // Use native external-call nodes for the public MSL overload with both
    // fences. Keep the enum conversion opaque so CSE cannot assign the MSL
    // enum class to a primitive integer temporary.
    auto flags = tvm::Call{tvm::PrimType::Int(32), tvm::tirx::builtin::call_extern(), {tvm::tirx::StringImm{"metal::mem_flags"}, tvm::IntImm::Int32(3)}};
    return tvm::tirx::Evaluate{tvm::Call{
        tvm::PrimType::Void(), tvm::tirx::builtin::call_extern(), {tvm::tirx::StringImm{"metal::threadgroup_barrier"}, std::move(flags)}}};
}

class CooperativeGroupMapper final : public tvm::tirx::StmtExprMutator {

private:
    tvm::tirx::PrimVar _thread;
    uint64_t _threads;
    uint64_t _shared_memory_limit;
    uint64_t _shared_memory_used{0u};
    uint32_t _lane_depth{0u};
    luisa::unordered_map<const tvm::tirx::VarNode *, tvm::tirx::BufferVar> _buffers;

private:
    [[nodiscard]] tvm::tirx::Stmt _synchronize(tvm::tirx::Stmt statement) const {
        return tvm::tirx::SeqStmt::Flatten(tvm::ffi::Array<tvm::tirx::Stmt>{std::move(statement), metal_group_barrier()});
    }

    [[nodiscard]] tvm::tirx::Stmt _distribute(const tvm::tirx::ForNode *loop) {
        auto domain = element_domain(loop);
        auto count = domain.count;
        _lane_depth++;
        auto body = VisitStmt(domain.body);
        _lane_depth--;
        if (count == 0u) { return tvm::tirx::Evaluate{tvm::IntImm::Int32(0)}; }
        auto chunks = (count + _threads - 1u) / _threads;
        auto chunk = tvm::tirx::PrimVar{loop->loop_var->name + "_chunk", tvm::PrimType::Int(64)};
        auto linear = chunk * tvm::IntImm::Int64(static_cast<int64_t>(_threads)) + _thread;
        tvm::ffi::Map<tvm::tirx::Var, tvm::Expr> coordinates;
        auto trailing = count;
        for (auto axis : domain.axes) {
            auto extent = static_extent(axis->extent);
            trailing /= extent;
            tvm::PrimExpr coordinate = linear;
            if (domain.axes.size() != 1u) {
                coordinate = tvm::floormod(tvm::floordiv(std::move(coordinate), tvm::IntImm::Int64(static_cast<int64_t>(trailing))), axis->extent);
            }
            coordinates.Set(axis->loop_var, axis->min + coordinate);
        }
        body = tvm::tirx::Substitute(std::move(body), coordinates);
        if (chunks * _threads != count) {
            body = tvm::tirx::IfThenElse{linear < tvm::IntImm::Int64(static_cast<int64_t>(count)), std::move(body)};
        }
        auto distributed = tvm::tirx::For{
            chunk, tvm::IntImm::Int64(0), tvm::IntImm::Int64(static_cast<int64_t>(chunks)),
            tvm::tirx::ForKind::kSerial, std::move(body)};
        // A barrier is outside the tail predicate: inactive workers still
        // participate, and the next operation may read any produced element.
        return _synchronize(std::move(distributed));
    }

    [[nodiscard]] tvm::ffi::Optional<tvm::PrimExpr> _predicate(const tvm::ffi::Optional<tvm::PrimExpr> &value) {
        return value ? VisitPrimExpr(value.value()) : tvm::ffi::Optional<tvm::PrimExpr>{};
    }

    [[nodiscard]] tvm::tirx::BufferVar _buffer(tvm::tirx::BufferVar buffer) const {
        if (auto iter = _buffers.find(buffer.get()); iter != _buffers.end()) { return iter->second; }
        if (buffer.scope() == "local") {
            throw std::runtime_error{"cooperative group capture of host-local storage requires a device allocation plan"};
        }
        return buffer;
    }

protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::ForNode *loop) final {
        auto logical = loop->annotations.count(logical_parallel_annotation) != 0u;
        auto elements = loop->annotations.count(independent_elements_annotation) != 0u;
        if (logical || elements) { validate_domain(loop); }
        if (auto constraint = loop->annotations.Get(execution_scope_annotation)) {
            auto scope = constraint.value().as<tvm::ffi::String>();
            if (!logical || !scope || scope.value() != "worker" || _lane_depth != 0u) {
                auto name = scope ? std::string{scope.value()} : std::string{"<invalid>"};
                throw std::runtime_error{"nested execution scope '" + name + "' in a cooperative group requires an available, unfactored worker level"};
            }
        }
        if ((logical || elements) && _lane_depth == 0u) { return _distribute(loop); }
        auto result = StmtExprMutator::VisitStmt_(loop).as_or_throw<tvm::tirx::For>();
        auto node = result.CopyOnWrite();
        node->annotations.erase(logical_parallel_annotation);
        node->annotations.erase(execution_scope_annotation);
        node->annotations.erase(independent_elements_annotation);
        return result;
    }

    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::AllocBufferNode *allocation) final {
        auto buffer = allocation->buffer;
        auto annotations = allocation->annotations;
        if (auto constraint = annotations.Get(memory_resource_annotation)) {
            auto resource = constraint.value().as<tvm::ffi::String>();
            auto expected = _lane_depth == 0u ? "shared" : "private";
            if (!resource || resource.value() != expected) {
                auto name = resource ? std::string{resource.value()} : std::string{"<invalid>"};
                throw std::runtime_error{"Memory resource '" + name +
                                         "' cannot realize this logical owner in cooperative Metal execution"};
            }
            annotations.erase(memory_resource_annotation);
        }
        if (_lane_depth != 0u) {
            auto result = StmtExprMutator::VisitStmt_(allocation).as_or_throw<tvm::tirx::AllocBuffer>();
            result.CopyOnWrite()->annotations = std::move(annotations);
            _buffers.emplace(buffer.get(), buffer);
            return result;
        }
        auto offset = buffer->elem_offset.as<tvm::IntImmNode>();
        if (buffer.scope() != "local" || !buffer->strides.empty() || buffer->layout || !buffer->allocated_addr.empty() ||
            offset == nullptr || offset->value != 0) {
            throw std::runtime_error{"cooperative Tile storage requires an unplaced compact compiler temporary"};
        }
        auto bytes = static_cast<uint64_t>((buffer->dtype.bits() * buffer->dtype.lanes() + 7) / 8);
        for (auto &&dimension : buffer->shape) {
            auto extent = static_extent(dimension);
            if (extent != 0u && bytes > std::numeric_limits<uint64_t>::max() / extent) {
                throw std::runtime_error{"cooperative Tile storage size exceeds uint64 range"};
            }
            bytes *= extent;
        }
        if (bytes > _shared_memory_limit - _shared_memory_used) {
            throw std::runtime_error{"cooperative Tile storage exceeds target shared-memory capacity"};
        }
        _shared_memory_used += bytes;
        auto type = tvm::tirx::BufferType{"shared", buffer->dtype, buffer->shape, {}, buffer->elem_offset, buffer->data_alignment, buffer->offset_factor};
        auto shared = tvm::tirx::BufferVar{buffer.name() + "_shared", std::move(type), buffer.span()};
        _buffers.emplace(buffer.get(), shared);
        return tvm::tirx::AllocBuffer{std::move(shared), std::move(annotations), allocation->span};
    }

    [[nodiscard]] tvm::Expr VisitExpr_(const tvm::tirx::BufferLoadNode *load) final {
        auto buffer = _buffer(load->buffer);
        return tvm::tirx::BufferLoad{std::move(buffer), load->indices.Map([this](const tvm::PrimExpr &index) { return VisitPrimExpr(index); }),
                                     _predicate(load->predicate), load->span};
    }

    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::BufferStoreNode *store) final {
        auto buffer = _buffer(store->buffer);
        auto statement = tvm::tirx::BufferStore{std::move(buffer), VisitPrimExpr(store->value),
                                                store->indices.Map([this](const tvm::PrimExpr &index) { return VisitPrimExpr(index); }),
                                                _predicate(store->predicate), store->span};
        if (_lane_depth != 0u) { return statement; }
        // A scalar effect at group scope has one logical invocation, not one
        // copy per hardware thread. Publish it before any worker consumes it.
        return _synchronize(tvm::tirx::IfThenElse{tvm::equal(_thread, tvm::IntImm::Int64(0)), std::move(statement)});
    }

    [[nodiscard]] tvm::Expr VisitExpr_(const tvm::tirx::VarNode *variable) final {
        if (_buffers.contains(variable)) {
            throw std::runtime_error{"cooperative Tile storage cannot escape through an opaque buffer use"};
        }
        return StmtExprMutator::VisitExpr_(variable);
    }

public:
    CooperativeGroupMapper(tvm::tirx::PrimVar thread, uint64_t threads, uint64_t shared_memory_limit)
        : _thread{std::move(thread)}, _threads{threads}, _shared_memory_limit{shared_memory_limit} {}

    using StmtExprMutator::operator();
};

}// namespace

tvm::tirx::Stmt map_metal_cooperative_group(const tvm::tirx::For &loop, uint32_t max_threads, uint64_t shared_memory_limit) {
    validate_domain(loop.get());
    auto groups = static_extent(loop->extent);
    auto grain = uint64_t{1u};
    tvm::tirx::PostOrderVisit(loop->body, [&](const tvm::ffi::ObjectRef &node) {
        if (auto child = node.as<tvm::tirx::ForNode>(); child != nullptr &&
                                                        (child->annotations.count(independent_elements_annotation) || child->annotations.count(logical_parallel_annotation))) {
            grain = std::max(grain, element_domain(child).count);
        }
    });
    auto threads = std::min<uint64_t>(grain, std::max<uint32_t>(max_threads, 1u));
    auto thread = tvm::tirx::PrimVar{loop->loop_var->name + "_worker", tvm::PrimType::Int(64)};
    auto group = tvm::tirx::PrimVar{loop->loop_var->name + "_group", tvm::PrimType::Int(64)};
    auto body = CooperativeGroupMapper{thread, threads, shared_memory_limit}(loop->body);
    // Empty domains are no-ops, but must not hide unsupported descendants.
    if (groups == 0u) { return tvm::tirx::Evaluate{tvm::IntImm::Int32(0)}; }
    body = tvm::tirx::Substitute(std::move(body),
                                 tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>{{loop->loop_var, group + loop->min}});
    auto zero = tvm::IntImm::Int64(0);
    auto worker_count = tvm::IntImm::Int64(static_cast<int64_t>(threads));
    auto worker_axis = tvm::tirx::IterVar{tvm::Range::FromMinExtent(zero, worker_count), thread,
                                          tvm::tirx::IterVarType::kThreadIndex, "threadIdx.x"};
    body = tvm::tirx::For{thread, zero, worker_count, tvm::tirx::ForKind::kThreadBinding, std::move(body), std::move(worker_axis)};
    auto group_axis = tvm::tirx::IterVar{tvm::Range::FromMinExtent(zero, loop->extent), group,
                                         tvm::tirx::IterVarType::kThreadIndex, "blockIdx.x"};
    return tvm::tirx::For{group, zero, loop->extent, tvm::tirx::ForKind::kThreadBinding, std::move(body), std::move(group_axis)};
}

}// namespace luisa::compute::tile::bridge::tirx::detail
