#include <algorithm>
#include <utility>

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>

#include <luisa/core/stl/unordered_map.h>

#include "execution.h"

namespace luisa::compute::tile::bridge::tirx::detail {

namespace {

using Statements = tvm::ffi::Array<tvm::tirx::Stmt>;
using Buffers = luisa::unordered_set<const tvm::tirx::VarNode *>;

// A whole-group proof, not a local decision that a matrix call "looks pure".
// Only emitted private operations and one terminal, partitioned global store
// are accepted. A shared access, a second sink, an output in a loop, an escape,
// an explicit barrier or any unknown statement keeps the reference fences.
class SubgroupIsolation final {
private:
    const tvm::tirx::Stmt &_barrier;
    const tvm::tirx::Stmt &_output;
    luisa::unordered_set<const tvm::tirx::StmtNode *> _private;
    bool _finished{false};

    [[nodiscard]] bool _visit(const tvm::tirx::Stmt &statement, uint32_t depth) {
        if (statement.same_as(_barrier)) { return true; }
        if (auto evaluate = statement.as<tvm::tirx::EvaluateNode>()) {
            if (auto constant = evaluate->value.as<tvm::IntImmNode>(); constant != nullptr && constant->value == 0) { return true; }
        }
        if (auto sequence = statement.as<tvm::tirx::SeqStmtNode>()) {
            return std::all_of(sequence->seq.begin(), sequence->seq.end(), [&](auto &&child) { return _visit(child, depth); });
        }
        if (_finished) { return false; }
        if (statement.same_as(_output)) {
            _finished = depth == 0u;
            return _finished;
        }
        if (_private.contains(statement.get())) { return true; }
        if (auto loop = statement.as<tvm::tirx::ForNode>()) {
            auto minimum = loop->min.as<tvm::IntImmNode>();
            auto extent = loop->extent.as<tvm::IntImmNode>();
            auto step = loop->step ? loop->step.value().as<tvm::IntImmNode>() : nullptr;
            if (loop->kind != tvm::tirx::ForKind::kSerial || loop->thread_binding || !loop->annotations.empty() ||
                minimum == nullptr || extent == nullptr || extent->value <= 0 ||
                (loop->step && (step == nullptr || step->value != 1))) { return false; }
            return _visit(loop->body, depth + 1u);
        }
        return false;
    }

public:
    SubgroupIsolation(const tvm::tirx::Stmt &barrier, const tvm::tirx::Stmt &output,
                      luisa::span<const tvm::tirx::Stmt> private_operations)
        : _barrier{barrier}, _output{output} {
        for (auto &&statement : private_operations) { _private.emplace(statement.get()); }
    }

    [[nodiscard]] bool prove(const tvm::tirx::Stmt &body) { return _visit(body, 0u) && _finished; }
};

class IsolatedBarrierRemoval final : public tvm::tirx::StmtMutator {
private:
    const tvm::tirx::Stmt &_barrier;

protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt(const tvm::tirx::Stmt &statement) final {
        if (statement.same_as(_barrier)) { return tvm::tirx::Evaluate{tvm::IntImm::Int32(0)}; }
        return StmtMutator::VisitStmt(statement);
    }

public:
    explicit IsolatedBarrierRemoval(const tvm::tirx::Stmt &barrier) noexcept : _barrier{barrier} {}
};

struct Access {
    Buffers reads, writes;
    bool external_read{false};
    bool external_write{false};
    bool opaque{false};

    void merge(const Access &other) {
        reads.insert(other.reads.begin(), other.reads.end());
        writes.insert(other.writes.begin(), other.writes.end());
        external_read |= other.external_read;
        external_write |= other.external_write;
        opaque |= other.opaque;
    }
};

// Effects on distinct fresh shared allocations cannot alias. All global
// accesses share one conservative alias class, regardless of parameter names
// or constness. Other storage and calls are opaque, including matrix atoms:
// this first pass need not understand their ABI to merge two input copies.
class AccessAnalysis final : public tvm::tirx::StmtExprVisitor {
private:
    const Buffers &_shared;
    Access _access;

    void _record(const tvm::tirx::BufferVar &buffer, bool write) {
        if (_shared.contains(buffer.get())) {
            (write ? _access.writes : _access.reads).emplace(buffer.get());
        } else if (buffer.scope() == "global") {
            (write ? _access.external_write : _access.external_read) = true;
        } else {
            _access.opaque = true;
        }
    }

protected:
    void VisitExpr_(const tvm::tirx::BufferLoadNode *load) final {
        _record(load->buffer, false);
        StmtExprVisitor::VisitExpr_(load);
    }
    void VisitStmt_(const tvm::tirx::BufferStoreNode *store) final {
        _record(store->buffer, true);
        StmtExprVisitor::VisitStmt_(store);
    }
    void VisitExpr_(const tvm::CallNode *call) final {
        // Bounded scalar loads keep their original short-circuit predicate.
        // No other call is assumed pure merely because it returns a value.
        _access.opaque |= !call->op.same_as(tvm::tirx::builtin::if_then_else());
        StmtExprVisitor::VisitExpr_(call);
    }
    void VisitStmt_(const tvm::tirx::AllocBufferNode *allocation) final {
        _access.opaque |= !allocation->buffer->allocated_addr.empty() || !allocation->annotations.empty();
        StmtExprVisitor::VisitStmt_(allocation);
    }
    void VisitStmt_(const tvm::tirx::DeclBufferNode *) final { _access.opaque = true; }
    void VisitStmt_(const tvm::tirx::AttrStmtNode *) final { _access.opaque = true; }
    void VisitStmt_(const tvm::tirx::WhileNode *) final { _access.opaque = true; }
    void VisitStmt_(const tvm::tirx::ReturnNode *) final { _access.opaque = true; }
    void VisitStmt_(const tvm::tirx::BreakNode *) final { _access.opaque = true; }
    void VisitStmt_(const tvm::tirx::ContinueNode *) final { _access.opaque = true; }
    void VisitStmt_(const tvm::tirx::AssertStmtNode *) final { _access.opaque = true; }
    void VisitStmt_(const tvm::tirx::SBlockNode *) final { _access.opaque = true; }
    void VisitStmt_(const tvm::tirx::ScopeIdDefStmtNode *) final { _access.opaque = true; }
    void VisitStmt_(const tvm::tirx::TilePrimitiveCallNode *) final { _access.opaque = true; }

public:
    explicit AccessAnalysis(const Buffers &shared) noexcept : _shared{shared} {}
    [[nodiscard]] Access analyze(const Statements &statements) {
        for (auto &&statement : statements) { (*this)(statement); }
        return std::move(_access);
    }
};

[[nodiscard]] bool independent(const Access &left, const Access &right) {
    if (left.opaque || right.opaque ||
        (left.external_write && (right.external_read || right.external_write)) ||
        (right.external_write && (left.external_read || left.external_write))) { return false; }
    auto intersects = [](const Buffers &a, const Buffers &b) {
        return std::any_of(a.begin(), a.end(), [&](auto buffer) { return b.contains(buffer); });
    };
    return !intersects(left.writes, right.reads) && !intersects(left.writes, right.writes) && !intersects(left.reads, right.writes);
}

class BarrierCounter final : public tvm::tirx::StmtVisitor {
private:
    const tvm::tirx::Stmt &_barrier;

protected:
    void VisitStmt_(const tvm::tirx::EvaluateNode *statement) final {
        sites += statement == _barrier.get();
        StmtVisitor::VisitStmt_(statement);
    }

public:
    uint64_t sites{0u};
    explicit BarrierCounter(const tvm::tirx::Stmt &barrier) noexcept : _barrier{barrier} {}
    using StmtVisitor::operator();
};

class BarrierCoalescer final : public tvm::tirx::StmtMutator {
private:
    const tvm::tirx::Stmt &_barrier;
    const Buffers &_shared;

protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::SeqStmtNode *sequence) final {
        Statements statements;
        for (auto &&statement : sequence->seq) {
            auto rewritten = VisitStmt(statement);
            if (auto nested = rewritten.as<tvm::tirx::SeqStmtNode>()) {
                for (auto &&child : nested->seq) { statements.push_back(child); }
            } else {
                statements.push_back(std::move(rewritten));
            }
        }
        Statements result, segment;
        tvm::tirx::Stmt previous_barrier;
        Access pending;
        for (auto &&statement : statements) {
            if (statement.same_as(_barrier)) {
                auto access = AccessAnalysis{_shared}.analyze(segment);
                if (previous_barrier.defined() && !independent(pending, access)) {
                    result.push_back(previous_barrier);
                    pending = {};
                }
                // Accumulate since the last KEPT barrier. Checking only the
                // adjacent operation misses A-write, unrelated-B, A-read.
                pending.merge(access);
                for (auto &&child : segment) { result.push_back(child); }
                segment.clear();
                previous_barrier = statement;
            } else {
                segment.push_back(statement);
            }
        }
        // Do not remove the last fence, even before a pure trailing statement.
        // It may publish reads/writes to the next loop iteration or enclosing
        // region. We never pull a fence out of a loop/branch or move an effect.
        if (previous_barrier.defined()) { result.push_back(previous_barrier); }
        for (auto &&statement : segment) { result.push_back(statement); }
        return tvm::tirx::SeqStmt::Flatten(result);
    }

public:
    BarrierCoalescer(const tvm::tirx::Stmt &barrier, const Buffers &shared) noexcept
        : _barrier{barrier}, _shared{shared} {}
    using StmtMutator::operator();
};

}// namespace

tvm::tirx::Stmt coalesce_group_barriers(
    tvm::tirx::Stmt body, const tvm::tirx::Stmt &compiler_barrier,
    luisa::span<const tvm::tirx::BufferVar> shared_allocations,
    bool enabled, bool elide_independent_subgroups, GroupPlan &plan,
    luisa::span<const tvm::tirx::Stmt> subgroup_private_operations,
    luisa::span<const tvm::tirx::Stmt> subgroup_output_stores) {
    BarrierCounter before{compiler_barrier};
    before(body);
    plan.group_barrier_sites_before = before.sites;
    plan.independent_subgroups = shared_allocations.empty() && subgroup_output_stores.size() == 1u &&
                                 SubgroupIsolation{compiler_barrier, subgroup_output_stores.front(), subgroup_private_operations}.prove(body);
    if (enabled) {
        if (elide_independent_subgroups && plan.independent_subgroups) {
            body = IsolatedBarrierRemoval{compiler_barrier}(body);
        } else {
            Buffers shared;
            for (auto &&buffer : shared_allocations) { shared.emplace(buffer.get()); }
            body = BarrierCoalescer{compiler_barrier, shared}(body);
        }
    }
    BarrierCounter after{compiler_barrier};
    after(body);
    plan.group_barrier_sites_after = after.sites;
    return body;
}

}// namespace luisa::compute::tile::bridge::tirx::detail
