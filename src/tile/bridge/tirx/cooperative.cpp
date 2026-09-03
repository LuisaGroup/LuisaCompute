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

using MatrixPlanIndices = luisa::unordered_map<const tvm::tirx::ForNode *, size_t>;

struct AccumulatorLoop {
    const tvm::tirx::ForNode *matrix;
    const tvm::tirx::ForNode *update;
    MatrixCarry carry;
    size_t matrix_index;
};

using AccumulatorLoops = luisa::unordered_map<const tvm::tirx::ForNode *, AccumulatorLoop>;

[[nodiscard]] bool is_carry_update(const tvm::tirx::ForNode *loop, const MatrixCarry &carry) {
    if (loop->annotations.size() != 1u || !loop->annotations.count(independent_elements_annotation)) { return false; }
    auto domain = element_domain(loop);
    if (domain.axes.size() != 2u || static_extent(domain.axes[0]->extent) != carry.rows || static_extent(domain.axes[1]->extent) != carry.columns) { return false; }
    auto store = domain.body.as<tvm::tirx::BufferStoreNode>();
    if (store == nullptr || store->predicate || !store->buffer.same_as(carry.initial) || store->indices.size() != 2u) { return false; }
    auto load = store->value.as<tvm::tirx::BufferLoadNode>();
    if (load == nullptr || load->predicate || !load->buffer.same_as(carry.result) || load->indices.size() != 2u) { return false; }
    for (auto i = 0u; i < 2u; i++) {
        auto minimum = domain.axes[i]->min.as<tvm::IntImmNode>();
        if (minimum == nullptr || minimum->value != 0 || !store->indices[i].same_as(domain.axes[i]->loop_var) ||
            !load->indices[i].same_as(domain.axes[i]->loop_var)) { return false; }
    }
    return true;
}

[[nodiscard]] uint64_t saturating_multiply(uint64_t a, uint64_t b) noexcept {
    return b != 0u && a > std::numeric_limits<uint64_t>::max() / b ? std::numeric_limits<uint64_t>::max() : a * b;
}

// Collect facts before binding workers. Temporary shared BufferVars are only
// proof objects for the common MMA matcher; actual resource placement remains
// in the emitter and is checked there again. No source names drive semantics.
class GroupWorkloadAnalysis final : public tvm::tirx::StmtVisitor {
private:
    bool _matrix;
    uint32_t _lane_depth{0u};
    uint64_t _executions{1u};
    luisa::unordered_map<const tvm::tirx::VarNode *, tvm::tirx::BufferVar> _buffers;

    void _find_accumulator_loop(const tvm::tirx::ForNode *loop) {
        auto extent = loop->extent.as<tvm::IntImmNode>();
        auto sequence = loop->body.as<tvm::tirx::SeqStmtNode>();
        auto step = loop->step ? loop->step.value().as<tvm::IntImmNode>() : nullptr;
        if (loop->kind != tvm::tirx::ForKind::kSerial || loop->thread_binding || !loop->annotations.empty() || extent == nullptr || extent->value <= 0 ||
            (loop->step && (step == nullptr || step->value != 1)) || sequence == nullptr) { return; }
        const tvm::tirx::ForNode *matrix = nullptr;
        for (auto &&statement : sequence->seq) {
            auto candidate = statement.as<tvm::tirx::ForNode>();
            if (candidate != nullptr && matrices.contains(candidate)) {
                if (matrix != nullptr) { return; }
                matrix = candidate;
            }
        }
        if (matrix == nullptr) { return; }
        auto carry = metal_matrix_carry(tvm::ffi::GetRef<tvm::tirx::For>(matrix), [this](tvm::tirx::BufferVar buffer) {
            auto iter = _buffers.find(buffer.get());
            return iter == _buffers.end() ? tvm::tirx::BufferVar{} : iter->second;
        });
        if (!carry) { return; }
        const tvm::tirx::ForNode *update = nullptr;
        auto result_allocations = 0u;
        auto seen_matrix = false;
        for (auto &&statement : sequence->seq) {
            if (statement.get() == matrix) {
                seen_matrix = true;
                continue;
            }
            if (auto allocation = statement.as<tvm::tirx::AllocBufferNode>(); allocation != nullptr && allocation->buffer.same_as(carry->result)) {
                if (!allocation->annotations.empty()) { return; }
                result_allocations++;
                continue;
            }
            if (auto copy = statement.as<tvm::tirx::ForNode>(); copy != nullptr && is_carry_update(copy, *carry)) {
                if (!seen_matrix || update != nullptr) { return; }
                update = copy;
                continue;
            }
            auto observes_carry = false;
            tvm::tirx::PostOrderVisit(statement, [&](const tvm::ffi::ObjectRef &node) {
                observes_carry |= node.same_as(carry->initial) || node.same_as(carry->result);
                if (auto load = node.as<tvm::tirx::BufferLoadNode>()) {
                    observes_carry |= load->buffer.same_as(carry->initial) || load->buffer.same_as(carry->result);
                }
                if (auto store = node.as<tvm::tirx::BufferStoreNode>()) {
                    observes_carry |= store->buffer.same_as(carry->initial) || store->buffer.same_as(carry->result);
                }
                if (auto allocation = node.as<tvm::tirx::AllocBufferNode>()) {
                    observes_carry |= allocation->buffer.same_as(carry->initial) || allocation->buffer.same_as(carry->result);
                }
            });
            // An intermediate observation (including another yielded state)
            // invalidates residency. Never drop it to recognize a GEMM shape.
            if (observes_carry) { return; }
        }
        if (result_allocations != 1u || update == nullptr) { return; }
        auto index = matrices.at(matrix);
        workload.matrices[index].accumulator_iterations = static_cast<uint64_t>(extent->value);
        accumulators.emplace(loop, AccumulatorLoop{matrix, update, *carry, index});
    }

protected:
    void VisitStmt_(const tvm::tirx::AllocBufferNode *allocation) final {
        if (_lane_depth != 0u) { return; }
        auto buffer = allocation->buffer;
        auto bytes = static_cast<uint64_t>((buffer->dtype.bits() * buffer->dtype.lanes() + 7) / 8);
        for (auto &&dimension : buffer->shape) { bytes = saturating_multiply(bytes, static_extent(dimension)); }
        workload.shared_memory_bytes += std::min(bytes, std::numeric_limits<uint64_t>::max() - workload.shared_memory_bytes);
        auto offset = buffer->elem_offset.as<tvm::IntImmNode>();
        if (!_matrix || buffer.scope() != "local" || !buffer->strides.empty() || buffer->layout || !buffer->allocated_addr.empty() ||
            offset == nullptr || offset->value != 0) { return; }
        auto type = tvm::tirx::BufferType{"shared", buffer->dtype, buffer->shape, {}, buffer->elem_offset, buffer->data_alignment, buffer->offset_factor};
        _buffers.emplace(buffer.get(), tvm::tirx::BufferVar{buffer.name() + "_planned", std::move(type), buffer.span()});
    }

    void VisitStmt_(const tvm::tirx::ForNode *loop) final {
        auto independent = loop->annotations.count(independent_elements_annotation) || loop->annotations.count(logical_parallel_annotation);
        if (independent) {
            auto domain = element_domain(loop);
            workload.max_independent_elements = std::max(workload.max_independent_elements, domain.count);
            if (_lane_depth == 0u) {
                auto matrix = _matrix ? metal_matrix_workload(tvm::ffi::GetRef<tvm::tirx::For>(loop), [this](tvm::tirx::BufferVar buffer) {
                    auto iter = _buffers.find(buffer.get());
                    return iter == _buffers.end() ? tvm::tirx::BufferVar{} : iter->second;
                }) :
                                        std::nullopt;
                if (matrix) {
                    matrix->executions = _executions;
                    matrices.emplace(loop, workload.matrices.size());
                    workload.matrices.emplace_back(*matrix);
                } else {
                    auto work = saturating_multiply(domain.count, _executions);
                    workload.independent_elements += std::min(work, std::numeric_limits<uint64_t>::max() - workload.independent_elements);
                }
            }
            _lane_depth++;
            VisitStmt(domain.body);
            _lane_depth--;
        } else {
            auto previous = _executions;
            if (auto extent = loop->extent.as<tvm::IntImmNode>(); extent != nullptr && extent->value >= 0) {
                _executions = saturating_multiply(_executions, static_cast<uint64_t>(extent->value));
            }
            StmtVisitor::VisitStmt_(loop);
            if (_lane_depth == 0u && _matrix) { _find_accumulator_loop(loop); }
            _executions = previous;
        }
    }

public:
    GroupWorkload workload;
    MatrixPlanIndices matrices;
    AccumulatorLoops accumulators;
    explicit GroupWorkloadAnalysis(bool matrix) noexcept : _matrix{matrix} {}
};

class CooperativeGroupMapper final : public tvm::tirx::StmtExprMutator {

private:
    tvm::tirx::PrimVar _thread;
    uint64_t _threads;
    uint64_t _shared_memory_limit;
    uint64_t _shared_memory_used{0u};
    uint32_t _lane_depth{0u};
    bool _cooperative_matrix;
    const MatrixPlanIndices &_matrix_indices;
    GroupPlan &_plan;
    const AccumulatorLoops &_accumulators;
    const AccumulatorLoop *_active_accumulator{nullptr};
    MatrixLoopEmission *_loop_emission{nullptr};
    luisa::unordered_map<const tvm::tirx::VarNode *, tvm::tirx::BufferVar> _buffers;

private:
    [[nodiscard]] tvm::tirx::Stmt _synchronize(tvm::tirx::Stmt statement) const {
        return tvm::tirx::SeqStmt::Flatten(tvm::ffi::Array<tvm::tirx::Stmt>{std::move(statement), metal_group_barrier()});
    }

    [[nodiscard]] bool _can_batch_copy(const tvm::tirx::Stmt &body) const {
        auto store = body.as<tvm::tirx::BufferStoreNode>();
        // Restrict this realization to compiler-owned shared destinations.
        // External writes, opaque effects, and conditional stores keep the
        // reference sequence. The surrounding element domain already carries
        // its semantic independence contract.
        if (store == nullptr || store->predicate || store->buffer.scope() != "shared") { return false; }
        auto owned = std::any_of(_buffers.begin(), _buffers.end(), [&](auto &&entry) {
            return entry.second.same_as(store->buffer);
        });
        if (!owned) { return false; }
        auto loads = 0u;
        auto compatible = true;
        tvm::tirx::PostOrderVisit(store->value, [&](const tvm::ffi::ObjectRef &node) {
            if (auto load = node.as<tvm::tirx::BufferLoadNode>()) {
                // Copies from another compiler buffer or external input are
                // allowed, but not a read/modify/write of this destination.
                compatible &= !load->buffer.same_as(store->buffer);
                loads++;
            }
            if (auto call = node.as<tvm::CallNode>()) {
                // Keep short-circuit bounded loads intact. Do not batch atomics,
                // clocks, opaque reads, or other effectful calls merely because
                // they occur in the value of a store.
                compatible &= call->op.same_as(tvm::tirx::builtin::if_then_else());
            }
        });
        for (auto &&index : store->indices) {
            tvm::tirx::PostOrderVisit(index, [&](const tvm::ffi::ObjectRef &node) {
                compatible &= node.as<tvm::tirx::BufferLoadNode>() == nullptr && node.as<tvm::CallNode>() == nullptr;
            });
        }
        return compatible && loads != 0u;
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
        auto element = [&](tvm::PrimExpr ordinal, bool guard) {
            auto linear = ordinal * tvm::IntImm::Int64(static_cast<int64_t>(_threads)) + _thread;
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
            auto result = tvm::tirx::Substitute(body, coordinates);
            if (guard) { result = tvm::tirx::IfThenElse{linear < tvm::IntImm::Int64(static_cast<int64_t>(count)), std::move(result)}; }
            return result;
        };
        tvm::ffi::Array<tvm::tirx::Stmt> distributed;
        auto consumed = uint64_t{0u};
        auto batch = std::min<uint64_t>(_plan.max_copy_batch, count / _threads);
        if (batch > 1u && _can_batch_copy(body)) {
            auto batches = count / _threads / batch;
            tvm::ffi::Array<tvm::tirx::Stmt> reads, writes;
            for (auto i = uint64_t{0u}; i < batch; i++) {
                auto copy = element(chunk * tvm::IntImm::Int64(static_cast<int64_t>(batch)) + tvm::IntImm::Int64(static_cast<int64_t>(i)), false)
                                .as_or_throw<tvm::tirx::BufferStore>();
                auto value = tvm::tirx::PrimVar{loop->loop_var->name + "_copy_value_" + std::to_string(i), copy->value.ty()};
                reads.push_back(tvm::tirx::Bind{value, copy->value});
                writes.push_back(tvm::tirx::BufferStore{copy->buffer, value, copy->indices, std::nullopt, copy->span});
            }
            for (auto &&write : writes) { reads.push_back(write); }
            distributed.push_back(tvm::tirx::For{chunk, tvm::IntImm::Int64(0), tvm::IntImm::Int64(static_cast<int64_t>(batches)),
                                                 tvm::tirx::ForKind::kSerial, tvm::tirx::SeqStmt::Flatten(reads)});
            consumed = batches * batch;
            _plan.batched_copy_operations++;
        }
        // Only complete worker chunks were batched. Remainders retain the
        // original load predicate and domain guard, so no inactive worker
        // speculates an out-of-bounds read just to fill a batch.
        if (consumed != chunks) {
            distributed.push_back(tvm::tirx::For{chunk, tvm::IntImm::Int64(static_cast<int64_t>(consumed)),
                                                 tvm::IntImm::Int64(static_cast<int64_t>(chunks - consumed)),
                                                 tvm::tirx::ForKind::kSerial, element(chunk, chunks * _threads != count)});
        }
        // A barrier is outside the tail predicate: inactive workers still
        // participate, and the next operation may read any produced element.
        return _synchronize(tvm::tirx::SeqStmt::Flatten(distributed));
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
        if (_active_accumulator != nullptr && loop == _active_accumulator->update) { return tvm::tirx::Evaluate{tvm::IntImm::Int32(0)}; }
        if (auto iter = _accumulators.find(loop); iter != _accumulators.end() &&
                                                  _plan.matrices[iter->second.matrix_index].persistent_accumulator) {
            if (_lane_depth != 0u || !_buffers.contains(iter->second.carry.initial.get())) {
                throw std::runtime_error{"planned accumulator must have group-owned storage outside its recurrence"};
            }
            auto previous = _active_accumulator;
            auto previous_emission = _loop_emission;
            MatrixLoopEmission emission;
            _active_accumulator = &iter->second;
            _loop_emission = &emission;
            auto body = StmtExprMutator::VisitStmt_(loop);
            _active_accumulator = previous;
            _loop_emission = previous_emission;
            if (!emission.before.defined() || !emission.after.defined()) {
                throw std::runtime_error{"planned accumulator recurrence was not emitted"};
            }
            return tvm::tirx::SeqStmt::Flatten(tvm::ffi::Array<tvm::tirx::Stmt>{emission.before, std::move(body), _synchronize(emission.after)});
        }
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
        if (elements && _lane_depth == 0u && _cooperative_matrix) {
            MatrixDistribution distribution;
            if (auto iter = _matrix_indices.find(loop); iter != _matrix_indices.end()) { distribution = _plan.matrices.at(iter->second); }
            auto emission = _active_accumulator != nullptr && _active_accumulator->matrix == loop ? _loop_emission : nullptr;
            auto matrix = try_metal_matrix(tvm::ffi::GetRef<tvm::tirx::For>(loop), _thread, _threads, [this](tvm::tirx::BufferVar buffer) {
                                               // Atom alias proofs apply only to allocations seen by
                                               // this mapper, never an external buffer's scope label.
                                               if (auto iter = _buffers.find(buffer.get()); iter != _buffers.end()) { return iter->second; }
                                               return tvm::tirx::BufferVar{}; }, distribution, emission);
            if (emission != nullptr && !matrix.defined()) { throw std::runtime_error{"planned matrix recurrence failed emission verification"}; }
            if (matrix.defined()) { return _synchronize(std::move(matrix)); }
        }
        if ((logical || elements) && _lane_depth == 0u) { return _distribute(loop); }
        auto result = StmtExprMutator::VisitStmt_(loop).as_or_throw<tvm::tirx::For>();
        auto node = result.CopyOnWrite();
        node->annotations.erase(logical_parallel_annotation);
        node->annotations.erase(execution_scope_annotation);
        node->annotations.erase(independent_elements_annotation);
        node->annotations.erase(mma_annotation);
        return result;
    }

    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::AllocBufferNode *allocation) final {
        auto buffer = allocation->buffer;
        if (_active_accumulator != nullptr && buffer.same_as(_active_accumulator->carry.result)) {
            // Retain a proof-only buffer identity for the matrix matcher. The
            // emitted recurrence has no D accesses and needs no D allocation.
            auto type = tvm::tirx::BufferType{"shared", buffer->dtype, buffer->shape, {}, buffer->elem_offset, buffer->data_alignment, buffer->offset_factor};
            _buffers.emplace(buffer.get(), tvm::tirx::BufferVar{buffer.name() + "_elided", std::move(type), buffer.span()});
            return tvm::tirx::Evaluate{tvm::IntImm::Int32(0)};
        }
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
        auto empty = std::any_of(buffer->shape.begin(), buffer->shape.end(), [](auto &&dimension) noexcept {
            auto extent = dimension.template as<tvm::IntImmNode>();
            return extent != nullptr && extent->value == 0;
        });
        auto bytes = empty ? uint64_t{0u} : static_cast<uint64_t>((buffer->dtype.bits() * buffer->dtype.lanes() + 7) / 8);
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
    CooperativeGroupMapper(tvm::tirx::PrimVar thread, uint64_t threads, uint64_t shared_memory_limit, bool cooperative_matrix,
                           const MatrixPlanIndices &matrix_indices, GroupPlan &plan, const AccumulatorLoops &accumulators)
        : _thread{std::move(thread)}, _threads{threads}, _shared_memory_limit{shared_memory_limit}, _cooperative_matrix{cooperative_matrix},
          _matrix_indices{matrix_indices}, _plan{plan}, _accumulators{accumulators} {}

    using StmtExprMutator::operator();
};

}// namespace

tvm::tirx::Stmt map_metal_cooperative_group(const tvm::tirx::For &loop, uint32_t max_threads, uint64_t shared_memory_limit,
                                            bool cooperative_matrix, const PlannerOptions &options, luisa::vector<GroupPlan> &plans) {
    validate_domain(loop.get());
    auto groups = static_extent(loop->extent);
    GroupWorkloadAnalysis analysis{cooperative_matrix};
    analysis.workload.programs = groups;
    analysis(loop->body);
    auto planned = plan_group(analysis.workload, ExecutionLimits{max_threads, 32u, shared_memory_limit}, options);
    if (!planned) { throw std::runtime_error{planned.error.c_str()}; }
    auto &plan = planned.plan;
    plan.name = std::string{loop->loop_var->name};
    auto threads = plan.threads;
    auto thread = tvm::tirx::PrimVar{loop->loop_var->name + "_worker", tvm::PrimType::Int(64)};
    auto group = tvm::tirx::PrimVar{loop->loop_var->name + "_group", tvm::PrimType::Int(64)};
    auto body = CooperativeGroupMapper{thread, threads, shared_memory_limit, cooperative_matrix, analysis.matrices, plan, analysis.accumulators}(loop->body);
    plans.emplace_back(std::move(plan));
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
