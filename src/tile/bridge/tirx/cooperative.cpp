#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <string>

#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

#include "diagnostic.h"
#include "execution.h"

#include <luisa/tile/bridge/tirx/layout.h>

namespace luisa::compute::tile::bridge::tirx::detail {

namespace {

[[nodiscard]] uint64_t static_extent(const tvm::PrimExpr &expression, Diagnostic &diagnostic) {
    if (diagnostic.failed()) { return 0u; }
    auto constant = expression.as<tvm::IntImmNode>();
    if (constant == nullptr || constant->value < 0) {
        diagnostic.set_error("cooperative Tile execution requires nonnegative static extents");
        return 0u;
    }
    return static_cast<uint64_t>(constant->value);
}

void validate_domain(const tvm::tirx::ForNode *loop, Diagnostic &diagnostic) {
    auto step = loop->step ? loop->step.value().as<tvm::IntImmNode>() : nullptr;
    if (loop->kind != tvm::tirx::ForKind::kSerial || loop->thread_binding ||
        (loop->step && (step == nullptr || step->value != 1))) {
        diagnostic.set_error("cooperative Tile execution requires serial unit-step domains before binding");
    }
}

struct ElementDomain {
    luisa::vector<const tvm::tirx::ForNode *> axes;
    tvm::tirx::Stmt body;
    uint64_t count{1u};
};

[[nodiscard]] ElementDomain element_domain(const tvm::tirx::ForNode *loop, Diagnostic &diagnostic) {
    if (diagnostic.failed()) { return {}; }
    auto rank = int64_t{1};
    if (auto annotation = loop->annotations.Get(independent_elements_annotation)) {
        auto value = annotation.value().as<tvm::IntImmNode>();
        if (value == nullptr || value->value <= 0) {
            diagnostic.set_error("cooperative Tile element domain requires a positive static rank");
            return {};
        }
        rank = value->value;
    }
    ElementDomain result;
    auto current = loop;
    for (auto i = int64_t{0}; i < rank; i++) {
        if (current == nullptr || (i != 0 && !current->annotations.empty()) ||
            current->min.as<tvm::IntImmNode>() == nullptr) {
            diagnostic.set_error("cooperative Tile element domain requires a perfect static rectangular nest");
            return {};
        }
        validate_domain(current, diagnostic);
        if (diagnostic.failed()) { return {}; }
        auto extent = static_extent(current->extent, diagnostic);
        if (diagnostic.failed()) { return {}; }
        if (extent != 0u && result.count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / extent) {
            diagnostic.set_error("cooperative Tile element domain exceeds int64 range");
            return {};
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
    uint64_t iterations;
    struct DirectOutput {
        const tvm::tirx::ForNode *initial;
        const tvm::tirx::ForNode *store;
        tvm::PrimExpr value;
        MatrixLoopEmission::Output destination;
        luisa::vector<const tvm::tirx::AllocBufferNode *> temporaries;
    };
    std::optional<DirectOutput> direct;
};

using AccumulatorLoops = luisa::unordered_map<const tvm::tirx::ForNode *, AccumulatorLoop>;

[[nodiscard]] bool is_positive_zero(const tvm::PrimExpr &expression) noexcept {
    auto value = expression.as<tvm::FloatImmNode>();
    return value != nullptr && expression.ty() == tvm::PrimType::Float(32) &&
           value->value == 0.0 && !std::signbit(value->value);
}

[[nodiscard]] bool is_carry_update(const tvm::tirx::ForNode *loop, const MatrixCarry &carry, Diagnostic &diagnostic) {
    if (loop->annotations.size() != 1u || !loop->annotations.count(independent_elements_annotation)) { return false; }
    auto domain = element_domain(loop, diagnostic);
    if (diagnostic.failed() || domain.axes.size() != 2u) { return false; }
    auto rows = static_extent(domain.axes[0]->extent, diagnostic);
    if (diagnostic.failed() || rows != carry.rows) { return false; }
    auto columns = static_extent(domain.axes[1]->extent, diagnostic);
    if (diagnostic.failed() || columns != carry.columns) { return false; }
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

[[nodiscard]] bool observes_buffer(const tvm::tirx::Stmt &statement, const tvm::tirx::BufferVar &buffer) {
    auto observed = false;
    tvm::tirx::PostOrderVisit(statement, [&](const tvm::ffi::ObjectRef &node) {
        observed |= node.same_as(buffer);
        if (auto load = node.as<tvm::tirx::BufferLoadNode>()) { observed |= load->buffer.same_as(buffer); }
        if (auto store = node.as<tvm::tirx::BufferStoreNode>()) { observed |= store->buffer.same_as(buffer); }
        if (auto allocation = node.as<tvm::tirx::AllocBufferNode>()) { observed |= allocation->buffer.same_as(buffer); }
    });
    return observed;
}

[[nodiscard]] tvm::PrimExpr literal_initial(const tvm::tirx::ForNode *loop, const MatrixCarry &carry, Diagnostic &diagnostic) {
    if (loop->annotations.size() != 1u || !loop->annotations.count(independent_elements_annotation)) { return {}; }
    auto domain = element_domain(loop, diagnostic);
    if (diagnostic.failed() || domain.axes.size() != 2u) { return {}; }
    auto rows = static_extent(domain.axes[0]->extent, diagnostic);
    if (diagnostic.failed() || rows != carry.rows) { return {}; }
    auto columns = static_extent(domain.axes[1]->extent, diagnostic);
    if (diagnostic.failed() || columns != carry.columns) { return {}; }
    auto store = domain.body.as<tvm::tirx::BufferStoreNode>();
    if (store == nullptr || store->predicate || !store->buffer.same_as(carry.initial) || store->indices.size() != 2u ||
        store->value.as<tvm::FloatImmNode>() == nullptr || store->value.ty() != tvm::PrimType::Float(32)) { return {}; }
    for (auto i = 0u; i < 2u; i++) {
        auto minimum = domain.axes[i]->min.as<tvm::IntImmNode>();
        if (minimum == nullptr || minimum->value != 0 || !store->indices[i].same_as(domain.axes[i]->loop_var)) { return {}; }
    }
    return store->value;
}

// Collect facts before binding workers. Temporary shared BufferVars are only
// proof objects for the common MMA matcher; actual resource placement remains
// in the emitter and is checked there again. No source names drive semantics.
class GroupWorkloadAnalysis final : public tvm::tirx::StmtVisitor {
private:
    Diagnostic &_diagnostic;
    bool _cooperative_reductions;
    bool _matrix;
    bool _metal_mpp;
    bool _matrix_epilogues;
    uint32_t _lane_depth{0u};
    uint64_t _executions{1u};
    const tvm::tirx::ForNode *_root;
    luisa::vector<const tvm::tirx::ForNode *> _ancestors;
    luisa::unordered_map<const tvm::tirx::VarNode *, tvm::tirx::BufferVar> _buffers;
    luisa::span<const tvm::tirx::BufferVar> _readonly_inputs;
    luisa::unordered_map<const tvm::tirx::ForNode *, uint64_t> _element_work;

    [[nodiscard]] tvm::tirx::BufferVar _matrix_buffer(tvm::tirx::BufferVar buffer) const {
        if (auto iter = _buffers.find(buffer.get()); iter != _buffers.end()) { return iter->second; }
        for (auto &&input : _readonly_inputs) {
            if (buffer.same_as(input)) { return buffer; }
        }
        return {};
    }

    [[nodiscard]] std::optional<AccumulatorLoop::DirectOutput> _find_direct_output(
        const tvm::tirx::SeqStmtNode *sequence, const tvm::tirx::ForNode *recurrence, const MatrixCarry &carry) const {
        auto seen_loop = false;
        auto allocated = false;
        const tvm::tirx::AllocBufferNode *initial_allocation = nullptr;
        const tvm::tirx::ForNode *initial = nullptr;
        std::optional<AccumulatorLoop::DirectOutput> result;
        tvm::PrimExpr value;
        MatrixEpilogue epilogue;
        luisa::unordered_map<const tvm::tirx::VarNode *, const tvm::tirx::AllocBufferNode *> temporary_allocations;
        luisa::vector<const tvm::tirx::AllocBufferNode *> temporaries;
        for (auto &&statement : sequence->seq) {
            if (statement.get() == recurrence) {
                if (!allocated || initial == nullptr) { return {}; }
                seen_loop = true;
                continue;
            }
            if (auto allocation = statement.as<tvm::tirx::AllocBufferNode>(); allocation != nullptr && allocation->buffer.same_as(carry.initial)) {
                if (allocated || seen_loop || !allocation->annotations.empty()) { return {}; }
                allocated = true;
                initial_allocation = allocation;
                continue;
            }
            if (auto allocation = statement.as<tvm::tirx::AllocBufferNode>(); seen_loop && allocation != nullptr) {
                auto buffer = allocation->buffer;
                auto offset = buffer->elem_offset.as<tvm::IntImmNode>();
                if (allocation->annotations.empty() && buffer.scope() == "local" && buffer->strides.empty() &&
                    !buffer->layout && buffer->allocated_addr.empty() && offset && offset->value == 0 &&
                    buffer->shape.size() == 2u) {
                    auto rows = static_extent(buffer->shape[0], _diagnostic);
                    if (_diagnostic.failed()) { return {}; }
                    if (rows == carry.rows) {
                        auto columns = static_extent(buffer->shape[1], _diagnostic);
                        if (_diagnostic.failed()) { return {}; }
                        if (columns == carry.columns && !temporary_allocations.emplace(buffer.get(), allocation).second) { return {}; }
                    }
                }
            }
            if (auto loop = statement.as<tvm::tirx::ForNode>()) {
                auto fill = literal_initial(loop, carry, _diagnostic);
                if (_diagnostic.failed()) { return {}; }
                if (fill.defined()) {
                    if (!allocated || initial != nullptr || seen_loop) { return {}; }
                    initial = loop;
                    value = fill;
                    continue;
                }
                if (seen_loop) {
                    if (_metal_mpp && _matrix_epilogues && loop->annotations.count(materialized_pure_tile_annotation) &&
                        metal_matrix_epilogue_binding(tvm::ffi::GetRef<tvm::tirx::For>(loop), carry, epilogue)) {
                        if (result || epilogue.bindings.size() > 64u) { return {}; }
                        auto allocation = temporary_allocations.find(epilogue.bindings.back().buffer.get());
                        if (allocation == temporary_allocations.end()) { return {}; }
                        temporaries.emplace_back(allocation->second);
                        temporary_allocations.erase(allocation);
                        continue;
                    }
                    if (auto output = metal_matrix_output(tvm::ffi::GetRef<tvm::tirx::For>(loop), carry, _ancestors, _metal_mpp, _matrix_epilogues ? &epilogue : nullptr)) {
                        if (result) { return {}; }
                        result = AccumulatorLoop::DirectOutput{initial, loop, value, *output, temporaries};
                        continue;
                    }
                }
            }
            // This also catches opaque pointer escape, a second consumer, and
            // nested/conditional uses. Manual memory annotations above prevent
            // storage removal even if the current consumers happen to match.
            if (observes_buffer(statement, carry.initial)) { return {}; }
        }
        if (!seen_loop || !result) { return {}; }
        // SeqStmt is a grouping node, not an authority to hide uses in another
        // sequence. Audit the whole group, pruning only the four proved pieces.
        auto closed = true;
        tvm::tirx::PreOrderVisit(_root->body, [&](const tvm::ffi::ObjectRef &node) {
            if (node.get() == initial_allocation || node.get() == initial || node.get() == recurrence || node.get() == result->store) { return false; }
            for (auto allocation : temporaries) {
                if (node.get() == allocation) { return false; }
            }
            for (auto &binding : epilogue.bindings) {
                if (node.get() == binding.producer) { return false; }
            }
            closed &= !node.same_as(carry.initial);
            if (auto load = node.as<tvm::tirx::BufferLoadNode>()) { closed &= !load->buffer.same_as(carry.initial); }
            if (auto store = node.as<tvm::tirx::BufferStoreNode>()) { closed &= !store->buffer.same_as(carry.initial); }
            if (auto allocation = node.as<tvm::tirx::AllocBufferNode>()) { closed &= !allocation->buffer.same_as(carry.initial); }
            for (auto &binding : epilogue.bindings) {
                closed &= !node.same_as(binding.buffer);
                if (auto load = node.as<tvm::tirx::BufferLoadNode>()) { closed &= !load->buffer.same_as(binding.buffer); }
                if (auto store = node.as<tvm::tirx::BufferStoreNode>()) { closed &= !store->buffer.same_as(binding.buffer); }
            }
            return closed;
        });
        return closed ? result : std::nullopt;
    }

    void _find_accumulator_loop(const tvm::tirx::ForNode *loop) {
        auto extent = loop->extent.as<tvm::IntImmNode>();
        auto sequence = loop->body.as<tvm::tirx::SeqStmtNode>();
        auto step = loop->step ? loop->step.value().as<tvm::IntImmNode>() : nullptr;
        auto ordinary_annotations = loop->annotations.size() == loop->annotations.count(deferred_pipeline_annotation) + loop->annotations.count(reduction_policy_annotation) &&
                                    (!loop->annotations.count(reduction_policy_annotation) || permits_unordered_reduction(loop));
        if (loop->kind != tvm::tirx::ForKind::kSerial || loop->thread_binding || !ordinary_annotations || extent == nullptr || extent->value <= 0 ||
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
        auto carry = metal_matrix_carry(tvm::ffi::GetRef<tvm::tirx::For>(matrix), [this](tvm::tirx::BufferVar buffer) { return _matrix_buffer(std::move(buffer)); }, _metal_mpp, _ancestors);
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
            if (auto copy = statement.as<tvm::tirx::ForNode>()) {
                auto matches = is_carry_update(copy, *carry, _diagnostic);
                if (_diagnostic.failed()) { return; }
                if (matches) {
                    if (!seen_matrix || update != nullptr) { return; }
                    update = copy;
                    continue;
                }
            }
            auto observes_carry = false;
            tvm::tirx::PostOrderVisit(statement, [&](const tvm::ffi::ObjectRef &node) {
                // An exit between MMA and yield can discard the new D. CF
                // residency must not turn that discarded update into live C.
                observes_carry |= node.as<tvm::tirx::BreakNode>() != nullptr || node.as<tvm::tirx::ContinueNode>() != nullptr || node.as<tvm::tirx::ReturnNode>() != nullptr;
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
        workload.matrices[index].recurrence_elements = _element_work.at(update);
        accumulators.emplace(loop, AccumulatorLoop{matrix, update, *carry, index, static_cast<uint64_t>(extent->value)});
    }

protected:
    void VisitStmt(const tvm::tirx::Stmt &statement) final {
        if (!_diagnostic.failed()) { StmtVisitor::VisitStmt(statement); }
    }

    void VisitStmt_(const tvm::tirx::SeqStmtNode *sequence) final {
        StmtVisitor::VisitStmt_(sequence);
        if (_diagnostic.failed() || _lane_depth != 0u || !_matrix) { return; }
        for (auto &&statement : sequence->seq) {
            auto loop = statement.as<tvm::tirx::ForNode>();
            if (auto iter = accumulators.find(loop); iter != accumulators.end()) {
                auto direct = _find_direct_output(sequence, loop, iter->second.carry);
                if (_diagnostic.failed()) { return; }
                if (direct) {
                    iter->second.direct = std::move(direct);
                    workload.matrices[iter->second.matrix_index].has_direct_output = true;
                    auto &matrix = workload.matrices[iter->second.matrix_index];
                    for (auto allocation : iter->second.direct->temporaries) {
                        auto type = allocation->buffer->dtype;
                        auto bytes = saturating_multiply(matrix.rows, matrix.columns);
                        bytes = saturating_multiply(bytes, (type.bits() * type.lanes() + 7u) / 8u);
                        matrix.epilogue_storage_bytes += std::min(bytes, std::numeric_limits<uint64_t>::max() - matrix.epilogue_storage_bytes);
                    }
                    auto initial_work = _element_work.at(iter->second.direct->initial);
                    auto store_work = _element_work.at(iter->second.direct->store);
                    // The epilogue's math still executes in fragment storage.
                    // Keep its producer/sink work in the generic scalar proxy;
                    // only the initializer is free. This is conservative, not
                    // an instruction-count or hardware-register prediction.
                    if (iter->second.direct->destination.epilogue) { store_work = 0u; }
                    matrix.direct_output_elements = initial_work + std::min(store_work, std::numeric_limits<uint64_t>::max() - initial_work);
                    workload.matrices[iter->second.matrix_index].overwrites_accumulator =
                        iter->second.iterations == 1u && is_positive_zero(iter->second.direct->value);
                }
            }
        }
    }

    void VisitStmt_(const tvm::tirx::AllocBufferNode *allocation) final {
        if (_lane_depth != 0u) { return; }
        auto buffer = allocation->buffer;
        auto bytes = static_cast<uint64_t>((buffer->dtype.bits() * buffer->dtype.lanes() + 7) / 8);
        for (auto &&dimension : buffer->shape) {
            auto extent = static_extent(dimension, _diagnostic);
            if (_diagnostic.failed()) { return; }
            bytes = saturating_multiply(bytes, extent);
        }
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
            auto domain = element_domain(loop, _diagnostic);
            if (_diagnostic.failed()) { return; }
            workload.max_independent_elements = std::max(workload.max_independent_elements, domain.count);
            if (_lane_depth == 0u) {
                if (_cooperative_reductions) {
                    if (auto count = metal_reduction_tile_output_count(tvm::ffi::GetRef<tvm::tirx::For>(loop))) {
                        workload.max_collective_outputs = std::max(workload.max_collective_outputs, *count);
                    }
                }
                auto matrix = _matrix ? metal_matrix_workload(tvm::ffi::GetRef<tvm::tirx::For>(loop), [this](tvm::tirx::BufferVar buffer) { return _matrix_buffer(std::move(buffer)); }, _metal_mpp, _ancestors) : std::nullopt;
                if (matrix) {
                    matrix->executions = _executions;
                    matrices.emplace(loop, workload.matrices.size());
                    workload.matrices.emplace_back(*matrix);
                } else {
                    auto work = saturating_multiply(domain.count, _executions);
                    _element_work.emplace(loop, work);
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
            _ancestors.emplace_back(loop);
            StmtVisitor::VisitStmt_(loop);
            if (!_diagnostic.failed() && _lane_depth == 0u && _matrix) { _find_accumulator_loop(loop); }
            _ancestors.pop_back();
            _executions = previous;
        }
    }

public:
    GroupWorkload workload;
    MatrixPlanIndices matrices;
    AccumulatorLoops accumulators;
    GroupWorkloadAnalysis(bool matrix, bool metal_mpp, bool matrix_epilogues, bool cooperative_reductions, const tvm::tirx::ForNode *root, luisa::span<const tvm::tirx::BufferVar> readonly_inputs, Diagnostic &diagnostic)
        : _diagnostic{diagnostic}, _cooperative_reductions{cooperative_reductions}, _matrix{matrix}, _metal_mpp{metal_mpp}, _matrix_epilogues{matrix_epilogues}, _root{root}, _ancestors{root}, _readonly_inputs{readonly_inputs} {}
};

class CooperativeGroupMapper final : public DiagnosticStmtExprMutator {

private:
    tvm::tirx::PrimVar _thread;
    luisa::vector<const tvm::tirx::ForNode *> _ancestors;
    uint64_t _threads;
    uint64_t _shared_memory_limit;
    uint64_t _shared_memory_used{0u};
    uint32_t _prefetch_budget;
    uint32_t _lane_depth{0u};
    bool _cooperative_matrix;
    bool _cooperative_reductions{false};
    const MatrixPlanIndices &_matrix_indices;
    GroupPlan &_plan;
    const AccumulatorLoops &_accumulators;
    luisa::span<const tvm::tirx::BufferVar> _readonly_inputs;
    const AccumulatorLoop *_active_accumulator{nullptr};
    MatrixLoopEmission *_loop_emission{nullptr};
    luisa::unordered_map<const tvm::tirx::VarNode *, tvm::tirx::BufferVar> _buffers;
    luisa::unordered_set<const tvm::tirx::VarNode *> _elided_buffers;
    luisa::unordered_set<const tvm::tirx::ForNode *> _elided_initializers;
    luisa::unordered_map<const tvm::tirx::ForNode *, tvm::tirx::Stmt> _direct_stores;
    tvm::tirx::Stmt _compiler_barrier{metal_group_barrier()};
    luisa::vector<tvm::tirx::BufferVar> _shared_allocations;
    luisa::vector<tvm::tirx::Stmt> _subgroup_private_operations;
    luisa::vector<tvm::tirx::Stmt> _subgroup_output_stores;

private:
    void _record_subgroup_private(const tvm::tirx::Stmt &statement) {
        // SeqStmt::Flatten may remove grouping nodes later, but it preserves
        // leaf identities. Facts are never recovered by matching source names.
        if (auto sequence = statement.as<tvm::tirx::SeqStmtNode>()) {
            for (auto &&child : sequence->seq) { _record_subgroup_private(child); }
        } else {
            _subgroup_private_operations.emplace_back(statement);
        }
    }

    [[nodiscard]] tvm::tirx::Stmt _synchronize(tvm::tirx::Stmt statement) const {
        return tvm::tirx::SeqStmt::Flatten(tvm::ffi::Array<tvm::tirx::Stmt>{std::move(statement), _compiler_barrier});
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
        auto domain = element_domain(loop, _diagnostic);
        if (_diagnostic.failed()) { return tvm::ffi::GetRef<tvm::tirx::For>(loop); }
        auto count = domain.count;
        _lane_depth++;
        auto body = VisitStmt(domain.body);
        _lane_depth--;
        if (_diagnostic.failed()) { return tvm::ffi::GetRef<tvm::tirx::For>(loop); }
        if (count == 0u) { return tvm::tirx::Evaluate{tvm::IntImm::Int32(0)}; }
        auto chunks = (count + _threads - 1u) / _threads;
        auto chunk = tvm::tirx::PrimVar{loop->loop_var->name + "_chunk", tvm::PrimType::Int(64)};
        auto element = [&](tvm::PrimExpr ordinal, bool guard) -> tvm::tirx::Stmt {
            auto linear = ordinal * tvm::IntImm::Int64(static_cast<int64_t>(_threads)) + _thread;
            tvm::ffi::Map<tvm::tirx::Var, tvm::Expr> coordinates;
            auto trailing = count;
            for (auto axis : domain.axes) {
                auto extent = static_extent(axis->extent, _diagnostic);
                if (_diagnostic.failed()) { return tvm::ffi::GetRef<tvm::tirx::For>(loop); }
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
                auto statement = element(chunk * tvm::IntImm::Int64(static_cast<int64_t>(batch)) + tvm::IntImm::Int64(static_cast<int64_t>(i)), false);
                if (_diagnostic.failed()) { return tvm::ffi::GetRef<tvm::tirx::For>(loop); }
                auto copy = statement.as_or_throw<tvm::tirx::BufferStore>();
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
            auto tail = element(chunk, chunks * _threads != count);
            if (_diagnostic.failed()) { return tvm::ffi::GetRef<tvm::tirx::For>(loop); }
            distributed.push_back(tvm::tirx::For{chunk, tvm::IntImm::Int64(static_cast<int64_t>(consumed)),
                                                 tvm::IntImm::Int64(static_cast<int64_t>(chunks - consumed)),
                                                 tvm::tirx::ForKind::kSerial, std::move(tail)});
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
            _diagnostic.set_error("cooperative group capture of host-local storage requires a device allocation plan");
        }
        return buffer;
    }

protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::ForNode *loop) final {
        _ancestors.emplace_back(loop);
        struct PopDomain {
            luisa::vector<const tvm::tirx::ForNode *> &domain;
            ~PopDomain() noexcept { domain.pop_back(); }
        } pop{_ancestors};
        if (_elided_initializers.contains(loop)) { return tvm::tirx::Evaluate{tvm::IntImm::Int32(0)}; }
        if (auto iter = _direct_stores.find(loop); iter != _direct_stores.end()) {
            auto store = std::move(iter->second);
            _direct_stores.erase(iter);
            return _synchronize(std::move(store));
        }
        if (_active_accumulator != nullptr && loop == _active_accumulator->update) { return tvm::tirx::Evaluate{tvm::IntImm::Int32(0)}; }
        if (auto iter = _accumulators.find(loop); iter != _accumulators.end() &&
                                                  _plan.matrices[iter->second.matrix_index].persistent_accumulator) {
            if (_lane_depth != 0u || !_buffers.contains(iter->second.carry.initial.get())) {
                _diagnostic.set_error("planned accumulator must have group-owned storage outside its recurrence");
                return tvm::ffi::GetRef<tvm::tirx::For>(loop);
            }
            auto previous = _active_accumulator;
            auto previous_emission = _loop_emission;
            MatrixLoopEmission emission;
            auto direct = _plan.matrices[iter->second.matrix_index].direct_accumulator_store;
            if (direct) {
                emission.initial = iter->second.direct->value;
                emission.output = iter->second.direct->destination;
                emission.overwrite_accumulator = iter->second.iterations == 1u && is_positive_zero(emission.initial);
            }
            _active_accumulator = &iter->second;
            _loop_emission = &emission;
            auto body = StmtExprMutator::VisitStmt_(loop);
            _active_accumulator = previous;
            _loop_emission = previous_emission;
            if (_diagnostic.failed()) { return tvm::ffi::GetRef<tvm::tirx::For>(loop); }
            auto mapped_loop = body.as_or_throw<tvm::tirx::For>();
            if (loop->annotations.count(deferred_pipeline_annotation)) {
                auto prefetched = try_prefetch_matrix_pipeline(mapped_loop, _compiler_barrier, _prefetch_budget, _plan);
                if (prefetched.defined()) { body = std::move(prefetched); }
            }
            if (body.same_as(mapped_loop)) {
                mapped_loop.CopyOnWrite()->annotations.erase(deferred_pipeline_annotation);
                body = mapped_loop;
            }
            if (!emission.before.defined() || !emission.after.defined()) {
                _diagnostic.set_error("planned accumulator recurrence was not emitted");
                return tvm::ffi::GetRef<tvm::tirx::For>(loop);
            }
            if (direct) {
                if (emission.subgroup_inputs && emission.subgroup_step.defined() &&
                    std::all_of(emission.subgroup_inputs->begin(), emission.subgroup_inputs->end(), [this](auto &&input) {
                        return std::any_of(_readonly_inputs.begin(), _readonly_inputs.end(), [&](auto &&proved) { return input.same_as(proved); });
                    })) {
                    // The matrix emitter proves the private CF and complete
                    // synchronous operation footprint. The forwarding pass
                    // proves input identities immutable under noalias. Neither
                    // a storage scope string nor the MPP option alone suffices.
                    _record_subgroup_private(emission.before);
                    _record_subgroup_private(emission.subgroup_step);
                    _subgroup_output_stores.emplace_back(emission.after);
                }
                // Keep the global write at the original sink, including any
                // intervening reads of that output. Only C storage disappears.
                _direct_stores.emplace(iter->second.direct->store, emission.after);
                return tvm::tirx::SeqStmt::Flatten(tvm::ffi::Array<tvm::tirx::Stmt>{emission.before, std::move(body)});
            }
            return tvm::tirx::SeqStmt::Flatten(tvm::ffi::Array<tvm::tirx::Stmt>{emission.before, std::move(body), _synchronize(emission.after)});
        }
        auto logical = loop->annotations.count(logical_parallel_annotation) != 0u;
        auto elements = loop->annotations.count(independent_elements_annotation) != 0u;
        if (logical || elements) { validate_domain(loop, _diagnostic); }
        if (_diagnostic.failed()) { return tvm::ffi::GetRef<tvm::tirx::For>(loop); }
        if (auto constraint = loop->annotations.Get(execution_scope_annotation)) {
            auto scope = constraint.value().as<tvm::ffi::String>();
            if (!logical || !scope || scope.value() != "worker" || _lane_depth != 0u) {
                auto name = scope ? std::string{scope.value()} : std::string{"<invalid>"};
                _diagnostic.set_error(luisa::format("nested execution scope '{}' in a cooperative group requires an available, unfactored worker level", name));
                return tvm::ffi::GetRef<tvm::tirx::For>(loop);
            }
        }
        if (elements && _lane_depth == 0u && _cooperative_matrix) {
            MatrixDistribution distribution;
            if (auto iter = _matrix_indices.find(loop); iter != _matrix_indices.end()) { distribution = _plan.matrices.at(iter->second); }
            auto emission = _active_accumulator != nullptr && _active_accumulator->matrix == loop ? _loop_emission : nullptr;
            auto matrix = try_metal_matrix(tvm::ffi::GetRef<tvm::tirx::For>(loop), _thread, _threads, [this](tvm::tirx::BufferVar buffer) {
                                               // Only owned allocations or explicitly proved noalias
                                               // read-only inputs can authorize a matrix access.
                                               if (auto iter = _buffers.find(buffer.get()); iter != _buffers.end()) { return iter->second; }
                                               for (auto &&input : _readonly_inputs) { if (buffer.same_as(input)) { return buffer; } }
                                               return tvm::tirx::BufferVar{}; }, _diagnostic, distribution, emission, _plan.metal_mpp, _ancestors);
            if (_diagnostic.failed()) { return tvm::ffi::GetRef<tvm::tirx::For>(loop); }
            if (emission != nullptr && !matrix.defined()) {
                _diagnostic.set_error("planned matrix recurrence failed emission verification");
                return tvm::ffi::GetRef<tvm::tirx::For>(loop);
            }
            if (matrix.defined()) { return _synchronize(std::move(matrix)); }
        }
        if (elements && _lane_depth == 0u && _cooperative_reductions) {
            auto reduction = try_metal_reduction_tile(tvm::ffi::GetRef<tvm::tirx::For>(loop), _thread, _threads,
                                                      [this](tvm::tirx::BufferVar buffer) { return _buffer(std::move(buffer)); });
            if (_diagnostic.failed()) { return tvm::ffi::GetRef<tvm::tirx::For>(loop); }
            if (reduction.defined()) { return _synchronize(std::move(reduction)); }
        }
        if ((logical || elements) && _lane_depth == 0u) { return _distribute(loop); }
        auto statement = StmtExprMutator::VisitStmt_(loop);
        if (_diagnostic.failed()) { return tvm::ffi::GetRef<tvm::tirx::For>(loop); }
        auto result = statement.as_or_throw<tvm::tirx::For>();
        auto node = result.CopyOnWrite();
        node->annotations.erase(logical_parallel_annotation);
        node->annotations.erase(logical_program_shape_annotation);
        node->annotations.erase(execution_scope_annotation);
        node->annotations.erase(independent_elements_annotation);
        node->annotations.erase(mma_annotation);
        node->annotations.erase(deferred_pipeline_annotation);
        node->annotations.erase(reduction_contract_annotation);
        node->annotations.erase(reduction_policy_annotation);
        return result;
    }

    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::AllocBufferNode *allocation) final {
        auto buffer = allocation->buffer;
        if (_elided_buffers.contains(buffer.get()) || (_active_accumulator != nullptr && buffer.same_as(_active_accumulator->carry.result))) {
            // Retain a proof-only buffer identity for the matrix matcher. The
            // emitted recurrence has no D accesses and needs no D allocation.
            auto type = tvm::tirx::BufferType{"shared", buffer->dtype, buffer->shape, {}, buffer->elem_offset, buffer->data_alignment, buffer->offset_factor};
            _buffers.emplace(buffer.get(), tvm::tirx::BufferVar{buffer.name() + "_elided", std::move(type), buffer.span()});
            return tvm::tirx::Evaluate{tvm::IntImm::Int32(0)};
        }
        auto annotations = allocation->annotations;
        annotations.erase(manual_memory_annotation);
        if (auto constraint = annotations.Get(memory_resource_annotation)) {
            auto resource = constraint.value().as<tvm::ffi::String>();
            auto expected = _lane_depth == 0u ? "shared" : "private";
            if (!resource || resource.value() != expected) {
                auto name = resource ? std::string{resource.value()} : std::string{"<invalid>"};
                _diagnostic.set_error(luisa::format("Memory resource '{}' cannot realize this logical owner in cooperative Metal execution", name));
                return tvm::ffi::GetRef<tvm::tirx::AllocBuffer>(allocation);
            }
            annotations.erase(memory_resource_annotation);
        }
        if (_lane_depth != 0u) {
            auto statement = StmtExprMutator::VisitStmt_(allocation);
            if (_diagnostic.failed()) { return tvm::ffi::GetRef<tvm::tirx::AllocBuffer>(allocation); }
            auto result = statement.as_or_throw<tvm::tirx::AllocBuffer>();
            result.CopyOnWrite()->annotations = std::move(annotations);
            _buffers.emplace(buffer.get(), buffer);
            return result;
        }
        auto offset = buffer->elem_offset.as<tvm::IntImmNode>();
        if (buffer.scope() != "local" || !buffer->strides.empty() || buffer->layout || !buffer->allocated_addr.empty() ||
            offset == nullptr || offset->value != 0) {
            _diagnostic.set_error("cooperative Tile storage requires an unplaced compact compiler temporary");
            return tvm::ffi::GetRef<tvm::tirx::AllocBuffer>(allocation);
        }
        auto empty = std::any_of(buffer->shape.begin(), buffer->shape.end(), [](auto &&dimension) noexcept {
            auto extent = dimension.template as<tvm::IntImmNode>();
            return extent != nullptr && extent->value == 0;
        });
        auto bytes = empty ? uint64_t{0u} : static_cast<uint64_t>((buffer->dtype.bits() * buffer->dtype.lanes() + 7) / 8);
        for (auto &&dimension : buffer->shape) {
            auto extent = static_extent(dimension, _diagnostic);
            if (_diagnostic.failed()) { return tvm::ffi::GetRef<tvm::tirx::AllocBuffer>(allocation); }
            if (extent != 0u && bytes > std::numeric_limits<uint64_t>::max() / extent) {
                _diagnostic.set_error("cooperative Tile storage size exceeds uint64 range");
                return tvm::ffi::GetRef<tvm::tirx::AllocBuffer>(allocation);
            }
            bytes *= extent;
        }
        if (bytes > _shared_memory_limit - _shared_memory_used) {
            _diagnostic.set_error("cooperative Tile storage exceeds target shared-memory capacity");
            return tvm::ffi::GetRef<tvm::tirx::AllocBuffer>(allocation);
        }
        _shared_memory_used += bytes;
        auto type = tvm::tirx::BufferType{"shared", buffer->dtype, buffer->shape, {}, buffer->elem_offset, buffer->data_alignment, buffer->offset_factor};
        auto shared = tvm::tirx::BufferVar{buffer.name() + "_shared", std::move(type), buffer.span()};
        _buffers.emplace(buffer.get(), shared);
        _shared_allocations.emplace_back(shared);
        return tvm::tirx::AllocBuffer{std::move(shared), std::move(annotations), allocation->span};
    }

    [[nodiscard]] tvm::Expr VisitExpr_(const tvm::tirx::BufferLoadNode *load) final {
        auto buffer = _buffer(load->buffer);
        if (_diagnostic.failed()) { return tvm::ffi::GetRef<tvm::tirx::BufferLoad>(load); }
        auto indices = load->indices.Map([this](const tvm::PrimExpr &index) { return VisitPrimExpr(index); });
        auto predicate = _predicate(load->predicate);
        if (_diagnostic.failed()) { return tvm::ffi::GetRef<tvm::tirx::BufferLoad>(load); }
        return tvm::tirx::BufferLoad{std::move(buffer), std::move(indices), std::move(predicate), load->span};
    }

    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::BufferStoreNode *store) final {
        auto buffer = _buffer(store->buffer);
        if (_diagnostic.failed()) { return tvm::ffi::GetRef<tvm::tirx::BufferStore>(store); }
        auto value = VisitPrimExpr(store->value);
        auto indices = store->indices.Map([this](const tvm::PrimExpr &index) { return VisitPrimExpr(index); });
        auto predicate = _predicate(store->predicate);
        if (_diagnostic.failed()) { return tvm::ffi::GetRef<tvm::tirx::BufferStore>(store); }
        auto statement = tvm::tirx::BufferStore{std::move(buffer), std::move(value), std::move(indices), std::move(predicate), store->span};
        if (_lane_depth != 0u) { return statement; }
        // A scalar effect at group scope has one logical invocation, not one
        // copy per hardware thread. Publish it before any worker consumes it.
        return _synchronize(tvm::tirx::IfThenElse{tvm::equal(_thread, tvm::IntImm::Int64(0)), std::move(statement)});
    }

    [[nodiscard]] tvm::Expr VisitExpr_(const tvm::tirx::VarNode *variable) final {
        if (_buffers.contains(variable)) {
            _diagnostic.set_error("cooperative Tile storage cannot escape through an opaque buffer use");
            return tvm::ffi::GetRef<tvm::tirx::Var>(variable);
        }
        return StmtExprMutator::VisitExpr_(variable);
    }

public:
    CooperativeGroupMapper(tvm::tirx::PrimVar thread, uint64_t threads, uint64_t shared_memory_limit, bool cooperative_matrix,
                           const MatrixPlanIndices &matrix_indices, GroupPlan &plan, const AccumulatorLoops &accumulators, uint32_t prefetch_budget,
                           luisa::span<const tvm::tirx::BufferVar> readonly_inputs, const tvm::tirx::ForNode *root, Diagnostic &diagnostic)
        : DiagnosticStmtExprMutator{diagnostic}, _thread{std::move(thread)}, _ancestors{root}, _threads{threads}, _shared_memory_limit{shared_memory_limit}, _prefetch_budget{prefetch_budget},
          _cooperative_matrix{cooperative_matrix}, _matrix_indices{matrix_indices}, _plan{plan}, _accumulators{accumulators}, _readonly_inputs{readonly_inputs} {
        for (auto &&[loop, accumulator] : _accumulators) {
            if (_plan.matrices[accumulator.matrix_index].direct_accumulator_store) {
                if (!accumulator.direct) {
                    _diagnostic.set_error("direct matrix store lacks a proved initializer and sink");
                    return;
                }
                _elided_buffers.emplace(accumulator.carry.initial.get());
                _elided_initializers.emplace(accumulator.direct->initial);
                for (auto allocation : accumulator.direct->temporaries) { _elided_buffers.emplace(allocation->buffer.get()); }
                if (auto &epilogue = accumulator.direct->destination.epilogue) {
                    for (auto &binding : epilogue->bindings) { _elided_initializers.emplace(binding.producer); }
                }
            }
        }
    }

    [[nodiscard]] tvm::tirx::Stmt map(const tvm::tirx::Stmt &body, const PlannerOptions &options) {
        if (_diagnostic.failed()) { return body; }
        _cooperative_reductions = options.enabled && options.metal_subgroup_reductions;
        auto result = VisitStmt(body);
        if (_diagnostic.failed()) { return body; }
        return coalesce_group_barriers(std::move(result), _compiler_barrier, _shared_allocations,
                                       options.enabled && options.coalesce_group_barriers, options.elide_independent_subgroup_barriers, _plan,
                                       _subgroup_private_operations, _subgroup_output_stores);
    }
};

// This is a target-representation audit, NOT an alias/dependence proof of
// parallel instances or independent element domains. Those are semantic
// contracts. The new mapping must preserve each program's ordered phases,
// handle its storage and avoid replicating an opaque side effect per worker.
class AutomaticGroupAudit final : public tvm::tirx::StmtExprVisitor {
protected:
    void VisitStmt(const tvm::tirx::Stmt &statement) final {
        if (!valid) { return; }
        if (!statement.as<tvm::tirx::SeqStmtNode>() && !statement.as<tvm::tirx::ForNode>() &&
            !statement.as<tvm::tirx::AllocBufferNode>() && !statement.as<tvm::tirx::BufferStoreNode>() &&
            !statement.as<tvm::tirx::IfThenElseNode>() && !statement.as<tvm::tirx::EvaluateNode>() &&
            !statement.as<tvm::tirx::BindNode>()) {
            valid = false;
            return;
        }
        StmtExprVisitor::VisitStmt(statement);
    }
    void VisitStmt_(const tvm::tirx::ForNode *loop) final {
        // A nested explicit hierarchy needs its own coordinate factorization;
        // do not reinterpret a requested worker or another logical parallel.
        if (loop->annotations.count(execution_scope_annotation) || loop->annotations.count(logical_parallel_annotation)) {
            valid = false;
            return;
        }
        for (auto &&[name, value] : loop->annotations) {
            valid &= name == independent_elements_annotation || name == mma_annotation ||
                     name == materialized_pure_tile_annotation || name == reduction_contract_annotation ||
                     name == reduction_policy_annotation ||
                     name == deferred_pipeline_annotation;
        }
        if (auto permission = loop->annotations.Get(mma_annotation)) {
            auto value = permission.value().as<tvm::IntImmNode>();
            has_matrix |= value != nullptr && value->value == 1;
        }
        StmtExprVisitor::VisitStmt_(loop);
    }
    void VisitStmt_(const tvm::tirx::AllocBufferNode *allocation) final {
        // Preserve expert resource constraints and explicit materialization.
        valid &= allocation->annotations.empty();
        StmtExprVisitor::VisitStmt_(allocation);
    }
    void VisitExpr_(const tvm::CallNode *call) final {
        static auto effects = tvm::Op::GetAttrMap<tvm::tirx::TCallEffectKind>("TCallEffectKind");
        auto op = call->op.as<tvm::Op>();
        valid &= op && effects.count(op.value()) &&
                 effects[op.value()] <= static_cast<int64_t>(tvm::tirx::CallEffectKind::kPure) &&
                 !call->op.same_as(tvm::tirx::builtin::address_of());
        StmtExprVisitor::VisitExpr_(call);
    }
    void VisitExpr_(const tvm::tirx::ProducerLoadNode *) final { valid = false; }

public:
    bool valid{true};
    bool has_matrix{false};
};

}// namespace

tvm::tirx::Stmt map_metal_cooperative_group(const tvm::tirx::For &loop, uint32_t max_threads, uint64_t shared_memory_limit,
                                            bool cooperative_matrix, bool metal_mpp, const PlannerOptions &options, luisa::vector<GroupPlan> &plans,
                                            luisa::span<const tvm::tirx::BufferVar> readonly_inputs, Diagnostic &diagnostic) {
    if (diagnostic.failed()) { return {}; }
    validate_domain(loop.get(), diagnostic);
    if (diagnostic.failed()) { return {}; }
    auto groups = static_extent(loop->extent, diagnostic);
    if (diagnostic.failed()) { return {}; }
    GroupWorkloadAnalysis analysis{cooperative_matrix, metal_mpp, options.fuse_matrix_epilogues, options.enabled && options.metal_subgroup_reductions, loop.get(), readonly_inputs, diagnostic};
    analysis.workload.programs = groups;
    analysis(loop->body);
    if (diagnostic.failed()) { return {}; }
    auto planned = plan_group(analysis.workload, ExecutionLimits{max_threads, 32u, shared_memory_limit}, options,
                              metal_mpp ? MatrixCostBasis::METAL_MPP_MEMORY : MatrixCostBasis::SIMDGROUP_REFERENCE);
    if (!planned) {
        diagnostic.set_error(planned.error);
        return {};
    }
    auto &plan = planned.plan;
    plan.metal_mpp = metal_mpp;
    plan.name = std::string{loop->loop_var->name};
    auto threads = plan.threads;
    auto thread = tvm::tirx::PrimVar{loop->loop_var->name + "_worker", tvm::PrimType::Int(64)};
    auto group = tvm::tirx::PrimVar{loop->loop_var->name + "_group", tvm::PrimType::Int(64)};
    auto logical_group = tvm::PrimExpr{group};
    auto reordered = options.program_order_rows != 1u || options.program_order_columns != 1u;
    auto shape = loop->annotations.Get(logical_program_shape_annotation);
    if (reordered && !shape) {
        diagnostic.set_error("program traversal requires the original parallel shape");
        return {};
    }
    if (shape) {
        auto dimensions = shape.value().cast<tvm::ffi::Array<tvm::PrimExpr>>();
        auto volume = uint64_t{1u};
        for (auto &&dimension : dimensions) {
            auto extent = dimension.as<tvm::IntImmNode>();
            if (extent == nullptr || extent->value < 0 ||
                (extent->value != 0 && volume > static_cast<uint64_t>(INT64_MAX / extent->value))) {
                diagnostic.set_error("invalid parallel program shape");
                return {};
            }
            volume *= static_cast<uint64_t>(extent->value);
        }
        if (volume != groups || (reordered && dimensions.size() < 2u)) {
            diagnostic.set_error("program traversal requires a matching rank-two-or-higher parallel shape");
            return {};
        }
        if (dimensions.size() >= 2u) {
            auto rows = static_cast<uint64_t>(dimensions[dimensions.size() - 2u].as<tvm::IntImmNode>()->value);
            auto columns = static_cast<uint64_t>(dimensions.back().as<tvm::IntImmNode>()->value);
            plan.program_grid_rows = rows;
            plan.program_grid_columns = columns;
            if (reordered && groups != 0u) {
                auto area = tvm::IntImm::Int64(static_cast<int64_t>(rows * columns));
                auto mapped = rectangular_program_ordinal(tvm::floormod(group, area), rows, columns,
                                                          options.program_order_rows, options.program_order_columns);
                if (!mapped) {
                    diagnostic.set_error(mapped.error);
                    return {};
                }
                logical_group = tvm::floordiv(group, area) * area + mapped.value[0];
            }
        }
    }
    plan.program_order_rows = options.program_order_rows;
    plan.program_order_columns = options.program_order_columns;
    auto body = CooperativeGroupMapper{thread, threads, shared_memory_limit, cooperative_matrix, analysis.matrices, plan, analysis.accumulators,
                                       options.enabled && !metal_mpp ? options.max_pipeline_prefetch_scalars_per_lane : 0u, readonly_inputs, loop.get(), diagnostic}
                    .map(loop->body, options);
    if (diagnostic.failed()) { return {}; }
    // Empty domains are no-ops, but must not hide unsupported descendants.
    if (groups == 0u) {
        auto result = tvm::tirx::Evaluate{tvm::IntImm::Int32(0)};
        plans.emplace_back(std::move(plan));
        return result;
    }
    body = tvm::tirx::Substitute(std::move(body),
                                 tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>{{loop->loop_var, logical_group + loop->min}});
    auto zero = tvm::IntImm::Int64(0);
    auto worker_count = tvm::IntImm::Int64(static_cast<int64_t>(threads));
    auto worker_axis = tvm::tirx::IterVar{tvm::Range::FromMinExtent(zero, worker_count), thread,
                                          tvm::tirx::IterVarType::kThreadIndex, "threadIdx.x"};
    body = tvm::tirx::For{thread, zero, worker_count, tvm::tirx::ForKind::kThreadBinding, std::move(body), std::move(worker_axis)};
    auto group_axis = tvm::tirx::IterVar{tvm::Range::FromMinExtent(zero, loop->extent), group,
                                         tvm::tirx::IterVarType::kThreadIndex, "blockIdx.x"};
    auto result = tvm::tirx::For{group, zero, loop->extent, tvm::tirx::ForKind::kThreadBinding, std::move(body), std::move(group_axis)};
    plans.emplace_back(std::move(plan));
    return result;
}

tvm::tirx::Stmt try_map_metal_cooperative_program(
    const tvm::tirx::For &loop, uint32_t max_threads, uint64_t shared_memory_limit,
    bool cooperative_matrix, bool metal_mpp, const PlannerOptions &options,
    luisa::vector<GroupPlan> &plans, luisa::span<const tvm::tirx::BufferVar> readonly_inputs) {
    if (!options.enabled || !options.map_gpu_cooperative_programs || !cooperative_matrix ||
        loop->annotations.count(execution_scope_annotation) ||
        options.reduction_programs_per_group != 0u || options.reduction_unroll_factor != 1u ||
        options.reduction_lane_elements != 1u || options.cache_reduction_inputs) { return {}; }
    AutomaticGroupAudit audit;
    audit(loop->body);
    // Keep the existing pointwise and row-reduction families. This candidate
    // composes matrix work, ordinary Tile phases and ordered loop-carried state.
    if (!audit.valid || !audit.has_matrix) { return {}; }
    try {
        Diagnostic diagnostic;
        luisa::vector<GroupPlan> candidate;
        auto result = map_metal_cooperative_group(loop, max_threads, shared_memory_limit,
                                                  cooperative_matrix, metal_mpp, options, candidate, readonly_inputs, diagnostic);
        if (diagnostic.failed() || !result.defined()) { return {}; }
        for (auto &plan : candidate) {
            plan.automatic_cooperative = true;
            plans.emplace_back(std::move(plan));
        }
        return result;
    } catch (const std::exception &) {
        // An optional resource/target candidate must not erase the valid
        // reference program or leak a partially constructed plan.
        return {};
    }
}

}// namespace luisa::compute::tile::bridge::tirx::detail
