#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>

#include <luisa/core/mathematics.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

#include "execution.h"

namespace luisa::compute::tile::bridge::tirx::detail {

namespace {

constexpr auto subgroup_size = uint64_t{32u};
using BufferKey = const tvm::tirx::VarNode *;

// Maximum private slots owned by any worker under the blocked-cyclic map
// i = (chunk * workers + worker) * lane_elements + element. The partial
// final pack needs only its live prefix, including when elements < workers.
[[nodiscard]] uint64_t stripe_slots(uint64_t elements, uint64_t workers,
                                    uint64_t lane_elements) noexcept {
    auto stride = workers * lane_elements;
    return elements / stride * lane_elements +
           std::min(elements % stride, lane_elements);
}

[[nodiscard]] std::optional<uint64_t> static_extent(
    const tvm::PrimExpr &expression, bool positive = false) noexcept {
    auto value = expression.as<tvm::IntImmNode>();
    if (value == nullptr || value->value < (positive ? 1 : 0)) { return std::nullopt; }
    return static_cast<uint64_t>(value->value);
}

[[nodiscard]] bool unit_serial_loop(const tvm::tirx::ForNode *loop) noexcept {
    auto step = loop->step ? loop->step.value().as<tvm::IntImmNode>() : nullptr;
    return loop->kind == tvm::tirx::ForKind::kSerial && !loop->thread_binding &&
           (!loop->step || (step != nullptr && step->value == 1));
}

[[nodiscard]] bool zero_index(
    const tvm::ffi::Array<tvm::PrimExpr> &indices) noexcept {
    auto value = indices.size() == 1u ? indices[0u].as<tvm::IntImmNode>() : nullptr;
    return value != nullptr && value->value == 0;
}

[[nodiscard]] bool compact_local_scalar(
    const tvm::tirx::BufferVar &buffer) noexcept {
    auto extent = buffer->shape.size() == 1u ? buffer->shape[0u].as<tvm::IntImmNode>() : nullptr;
    auto offset = buffer->elem_offset.as<tvm::IntImmNode>();
    return buffer.scope() == "local" && buffer->dtype == tvm::PrimType::Float(32) &&
           extent != nullptr && extent->value == 1 && buffer->strides.empty() &&
           !buffer->layout && buffer->allocated_addr.empty() &&
           offset != nullptr && offset->value == 0;
}

void flatten_sequence(
    const tvm::tirx::Stmt &statement,
    tvm::ffi::Array<tvm::tirx::Stmt> &result) {
    if (auto sequence = statement.as<tvm::tirx::SeqStmtNode>()) {
        for (auto &&child : sequence->seq) { flatten_sequence(child, result); }
    } else {
        result.push_back(statement);
    }
}

struct ElementDomain {
    luisa::vector<const tvm::tirx::ForNode *> axes;
    tvm::tirx::Stmt body;
    uint64_t count{1u};
};

[[nodiscard]] std::optional<ElementDomain> element_domain(
    const tvm::tirx::ForNode *outer) noexcept {
    auto annotation = outer->annotations.Get(independent_elements_annotation);
    auto rank = annotation ? annotation.value().as<tvm::IntImmNode>() : nullptr;
    if (rank == nullptr || rank->value <= 0) { return std::nullopt; }
    ElementDomain result;
    auto loop = outer;
    for (auto i = int64_t{0}; i < rank->value; i++) {
        if (loop == nullptr || !unit_serial_loop(loop) ||
            (i != 0 && !loop->annotations.empty()) ||
            loop->min.as<tvm::IntImmNode>() == nullptr) {
            return std::nullopt;
        }
        auto extent = static_extent(loop->extent);
        if (!extent || (*extent != 0u &&
                        result.count > std::numeric_limits<uint64_t>::max() / *extent)) {
            return std::nullopt;
        }
        result.count *= *extent;
        result.axes.emplace_back(loop);
        result.body = loop->body;
        loop = loop->body.as<tvm::tirx::ForNode>();
    }
    return result;
}

struct ReductionMatch {
    tvm::tirx::BufferVar carry;
    tvm::PrimExpr contribution;
    const tvm::tirx::BufferStoreNode *update{nullptr};
    const tvm::tirx::BufferStoreNode *initializer{nullptr};
    const tvm::tirx::AllocBufferNode *allocation{nullptr};
    int64_t kind{0};
    uint64_t elements{0u};
};

struct StripedMaterialization {
    tvm::tirx::BufferVar buffer;
    const tvm::tirx::BufferStoreNode *store{nullptr};
    uint64_t elements{0u};
};

[[nodiscard]] std::optional<StripedMaterialization>
match_striped_materialization(const tvm::tirx::ForNode *outer) {
    auto contract =
        outer->annotations.Get(materialized_pure_tile_annotation);
    auto version = contract ? contract.value().as<tvm::IntImmNode>() : nullptr;
    auto domain = element_domain(outer);
    if (version == nullptr || version->value != 1 || !domain ||
        outer->annotations.size() != 2u) {
        return std::nullopt;
    }
    auto store = domain->body.as<tvm::tirx::BufferStoreNode>();
    if (store == nullptr || store->predicate ||
        store->indices.size() != domain->axes.size()) {
        return std::nullopt;
    }
    auto buffer = store->buffer;
    auto offset = buffer->elem_offset.as<tvm::IntImmNode>();
    if (buffer.scope() != "local" ||
        buffer->dtype != tvm::PrimType::Float(32) ||
        buffer->shape.size() != domain->axes.size() ||
        !buffer->strides.empty() || buffer->layout ||
        !buffer->allocated_addr.empty() || offset == nullptr ||
        offset->value != 0) {
        return std::nullopt;
    }
    auto equal = tvm::ffi::StructuralEqual{};
    auto elements = uint64_t{1u};
    for (auto i = size_t{0u}; i < domain->axes.size(); i++) {
        auto dimension = static_extent(buffer->shape[i], true);
        auto extent = static_extent(domain->axes[i]->extent, true);
        if (!dimension || !extent || *dimension != *extent ||
            !equal(store->indices[i], domain->axes[i]->loop_var) ||
            elements > std::numeric_limits<uint64_t>::max() / *dimension) {
            return std::nullopt;
        }
        elements *= *dimension;
    }
    auto pure = true;
    static auto effects =
        tvm::Op::GetAttrMap<tvm::tirx::TCallEffectKind>("TCallEffectKind");
    tvm::tirx::PostOrderVisit(store->value,
                              [&](const tvm::ffi::ObjectRef &node) {
                                  if (auto load = node.as<tvm::tirx::BufferLoadNode>()) {
                                      pure &= !load->buffer.same_as(buffer);
                                  } else if (auto call = node.as<tvm::CallNode>()) {
                                      auto op = call->op.as<tvm::Op>();
                                      pure &= op && effects.count(op.value()) != 0u &&
                                              effects[op.value()] <= static_cast<int64_t>(
                                                                         tvm::tirx::CallEffectKind::kPure);
                                  } else if (auto variable = node.as<tvm::tirx::VarNode>()) {
                                      pure &= variable != buffer.get();
                                  }
                              });
    if (!pure || elements != domain->count) { return std::nullopt; }
    return StripedMaterialization{
        std::move(buffer), store, elements};
}

[[nodiscard]] bool pure_contribution(
    const tvm::PrimExpr &expression,
    const tvm::tirx::BufferVar &carry,
    const tvm::tirx::BufferVar &temporary) {
    static auto effects =
        tvm::Op::GetAttrMap<tvm::tirx::TCallEffectKind>("TCallEffectKind");
    auto valid = expression.ty() == tvm::PrimType::Float(32);
    tvm::tirx::PostOrderVisit(expression, [&](const tvm::ffi::ObjectRef &node) {
        if (auto load = node.as<tvm::tirx::BufferLoadNode>()) {
            valid &= !load->buffer.same_as(carry) &&
                     !load->buffer.same_as(temporary);
        } else if (auto call = node.as<tvm::CallNode>()) {
            auto op = call->op.as<tvm::Op>();
            valid &= op && effects.count(op.value()) != 0u &&
                     effects[op.value()] <=
                         static_cast<int64_t>(tvm::tirx::CallEffectKind::kPure);
        } else if (node.as<tvm::tirx::ProducerLoadNode>() != nullptr) {
            valid = false;
        } else if (auto variable = node.as<tvm::tirx::VarNode>()) {
            valid &= variable != carry.get() && variable != temporary.get();
        }
    });
    return valid;
}

[[nodiscard]] std::optional<ReductionMatch> match_reduction(
    const tvm::tirx::ForNode *loop) {
    auto contract = loop->annotations.Get(reduction_contract_annotation);
    auto kind = contract ? contract.value().as<tvm::IntImmNode>() : nullptr;
    auto minimum = loop->min.as<tvm::IntImmNode>();
    auto elements = static_extent(loop->extent, true);
    if (loop->annotations.size() != 1u || kind == nullptr ||
        (kind->value != reduction_add_contract &&
         kind->value != reduction_max_contract &&
         kind->value != reduction_min_contract) ||
        !unit_serial_loop(loop) || minimum == nullptr || minimum->value != 0 ||
        !elements || loop->loop_var.ty() != tvm::PrimType::Int(64)) {
        return std::nullopt;
    }

    tvm::ffi::Array<tvm::tirx::Stmt> statements;
    flatten_sequence(loop->body, statements);
    if (statements.size() != 3u) { return std::nullopt; }
    auto temporary_allocation = statements[0u].as<tvm::tirx::AllocBufferNode>();
    auto combine_store = statements[1u].as<tvm::tirx::BufferStoreNode>();
    auto update_store = statements[2u].as<tvm::tirx::BufferStoreNode>();
    if (temporary_allocation == nullptr || combine_store == nullptr ||
        update_store == nullptr || !temporary_allocation->annotations.empty() ||
        combine_store->predicate || update_store->predicate ||
        !compact_local_scalar(temporary_allocation->buffer) ||
        !combine_store->buffer.same_as(temporary_allocation->buffer) ||
        !zero_index(combine_store->indices) || !zero_index(update_store->indices) ||
        !compact_local_scalar(update_store->buffer)) {
        return std::nullopt;
    }
    auto forwarded = update_store->value.as<tvm::tirx::BufferLoadNode>();
    if (forwarded == nullptr || forwarded->predicate ||
        !forwarded->buffer.same_as(temporary_allocation->buffer) ||
        !zero_index(forwarded->indices)) {
        return std::nullopt;
    }

    tvm::PrimExpr lhs;
    tvm::PrimExpr rhs;
    if (kind->value == reduction_add_contract) {
        auto combine = combine_store->value.as<tvm::tirx::AddNode>();
        if (combine == nullptr) { return std::nullopt; }
        lhs = combine->a;
        rhs = combine->b;
    } else if (kind->value == reduction_max_contract) {
        auto combine = combine_store->value.as<tvm::tirx::MaxNode>();
        if (combine == nullptr) { return std::nullopt; }
        lhs = combine->a;
        rhs = combine->b;
    } else {
        auto combine = combine_store->value.as<tvm::tirx::MinNode>();
        if (combine == nullptr) { return std::nullopt; }
        lhs = combine->a;
        rhs = combine->b;
    }
    auto carry = update_store->buffer;
    auto is_carry = [&](const tvm::PrimExpr &value) noexcept {
        auto load = value.as<tvm::tirx::BufferLoadNode>();
        return load != nullptr && !load->predicate &&
               load->buffer.same_as(carry) && zero_index(load->indices);
    };
    tvm::PrimExpr contribution;
    if (is_carry(lhs)) {
        contribution = rhs;
    } else if (is_carry(rhs)) {
        contribution = lhs;
    } else {
        return std::nullopt;
    }
    if (!pure_contribution(contribution, carry, temporary_allocation->buffer)) {
        return std::nullopt;
    }
    return ReductionMatch{std::move(carry), std::move(contribution),
                          update_store, nullptr, nullptr, kind->value, *elements};
}

[[nodiscard]] bool identity_initializer(
    const tvm::tirx::BufferStoreNode *store,
    const ReductionMatch &match) noexcept {
    if (store == nullptr || store->predicate ||
        !store->buffer.same_as(match.carry) || !zero_index(store->indices)) {
        return false;
    }
    auto value = store->value.as<tvm::FloatImmNode>();
    if (value == nullptr || store->value.ty() != tvm::PrimType::Float(32)) {
        return false;
    }
    if (match.kind == reduction_add_contract) {
        return value->value == 0.0 && !std::signbit(value->value);
    }
    return std::isinf(value->value) &&
           (match.kind == reduction_max_contract ? value->value < 0.0 :
                                                   value->value > 0.0);
}

[[nodiscard]] tvm::PrimExpr reduction_identity(int64_t kind) {
    if (kind == reduction_add_contract) {
        return tvm::FloatImm{tvm::PrimType::Float(32), 0.0};
    }
    return tvm::FloatImm{
        tvm::PrimType::Float(32),
        kind == reduction_max_contract ?
            -std::numeric_limits<float>::infinity() :
            std::numeric_limits<float>::infinity()};
}

[[nodiscard]] tvm::tirx::Stmt shared_barrier() {
    return tvm::tirx::Evaluate{tvm::Call{
        tvm::PrimType::Int(32), tvm::tirx::builtin::tvm_storage_sync(), {tvm::tirx::StringImm{"shared"}}}};
}

class ReductionAnalysis final : public tvm::tirx::StmtVisitor {
private:
    void VisitStmt_(const tvm::tirx::ForNode *loop) final {
        annotated_reductions += loop->annotations.count(reduction_contract_annotation);
        StmtVisitor::VisitStmt_(loop);
    }

    void VisitStmt_(const tvm::tirx::SeqStmtNode *sequence) final {
        for (auto i = size_t{0u}; i < sequence->seq.size(); i++) {
            auto loop = sequence->seq[i].as<tvm::tirx::ForNode>();
            if (loop == nullptr ||
                !loop->annotations.count(reduction_contract_annotation)) {
                continue;
            }
            auto match = match_reduction(loop);
            auto allocation = i >= 2u ?
                                  sequence->seq[i - 2u].as<tvm::tirx::AllocBufferNode>() :
                                  nullptr;
            auto initializer = i >= 1u ?
                                   sequence->seq[i - 1u].as<tvm::tirx::BufferStoreNode>() :
                                   nullptr;
            if (!match || allocation == nullptr ||
                !allocation->buffer.same_as(match->carry) ||
                !allocation->annotations.empty() ||
                !identity_initializer(initializer, *match)) {
                valid = false;
                continue;
            }
            match->allocation = allocation;
            match->initializer = initializer;
            if (reductions.emplace(loop, std::move(*match)).second) {
                reduction_order.emplace_back(loop);
            } else {
                valid = false;
            }
        }
        StmtVisitor::VisitStmt_(sequence);
    }

public:
    luisa::unordered_map<const tvm::tirx::ForNode *, ReductionMatch> reductions;
    luisa::vector<const tvm::tirx::ForNode *> reduction_order;
    luisa::unordered_set<const tvm::tirx::ForNode *> replicated_elements;
    uint64_t annotated_reductions{0u};
    uint64_t reduction_elements{0u};
    uint64_t max_reduction_elements{0u};
    uint64_t independent_elements{0u};
    luisa::vector<uint64_t> independent_domains;
    bool valid{true};

    void finish(const tvm::tirx::Stmt &body) {
        valid &= annotated_reductions != 0u &&
                 annotated_reductions == reductions.size();
        if (!valid) { return; }
        for (auto loop : reduction_order) {
            auto &reduction = reductions.at(loop);
            reduction_elements += reduction.elements;
            max_reduction_elements =
                std::max(max_reduction_elements, reduction.elements);
        }
        tvm::tirx::PostOrderVisit(body, [&](const tvm::ffi::ObjectRef &node) {
            auto loop = node.as<tvm::tirx::ForNode>();
            if (loop == nullptr ||
                !loop->annotations.count(independent_elements_annotation)) {
                return;
            }
            auto domain = element_domain(loop);
            if (!domain) {
                valid = false;
                return;
            }
            auto contains_reduction = false;
            tvm::tirx::PostOrderVisit(loop->body, [&](const tvm::ffi::ObjectRef &child) {
                auto nested = child.as<tvm::tirx::ForNode>();
                contains_reduction |= nested != nullptr && reductions.contains(nested);
            });
            if (contains_reduction) {
                replicated_elements.emplace(loop);
            } else {
                independent_domains.emplace_back(domain->count);
                independent_elements += std::min(
                    domain->count,
                    std::numeric_limits<uint64_t>::max() - independent_elements);
            }
        });
    }
};

void add_access_demand(ReductionAccessDemand &total,
                       const ReductionAccessDemand &value, double scale) noexcept {
    total.global_read_bytes += value.global_read_bytes * scale;
    total.global_write_bytes += value.global_write_bytes * scale;
    total.private_read_bytes += value.private_read_bytes * scale;
    total.private_write_bytes += value.private_write_bytes * scale;
}

// Cost facts only: this does not rewrite/CSE code or grant memory legality.
// Loads are deduplicated within one evaluation, never across a store or a
// traversal. Both sides of lazy branches count as potential demand. Unknown
// constructs leave the feature unavailable instead of reporting partial data.
class PayloadAccessCounter {
private:
    void _access(const tvm::tirx::BufferVar &buffer, bool read) {
        auto type = buffer->dtype;
        if (type.IsScalableVector() || type.lanes() != 1 || type.bits() == 0 || type.bits() % 8 != 0) {
            known = false;
            return;
        }
        auto bytes = static_cast<double>(type.bits() / 8);
        if (buffer.scope() == "global") {
            (read ? demand.global_read_bytes : demand.global_write_bytes) += bytes;
        } else if (buffer.scope() == "local") {
            (read ? demand.private_read_bytes : demand.private_write_bytes) += bytes;
        } else {
            known = false;
        }
    }

    void _reads(const tvm::ffi::Array<tvm::PrimExpr> &expressions) {
        luisa::vector<tvm::tirx::BufferLoad> seen;
        auto equal = tvm::ffi::StructuralEqual{};
        for (auto &&expression : expressions) {
            tvm::tirx::PostOrderVisit(expression, [&](const tvm::ffi::ObjectRef &node) {
                if (auto load = node.as<tvm::tirx::BufferLoadNode>()) {
                    auto value = tvm::ffi::GetRef<tvm::tirx::BufferLoad>(load);
                    if (value.ty() != load->buffer->dtype) { known = false; }
                    if (std::none_of(seen.begin(), seen.end(), [&](const auto &other) { return equal(value, other); })) {
                        seen.emplace_back(value);
                        _access(load->buffer, true);
                    }
                } else if (node.as<tvm::tirx::ProducerLoadNode>()) {
                    known = false;
                }
            });
        }
    }

public:
    ReductionAccessDemand demand;
    bool known{true};

    void expression(const tvm::Expr &value) {
        if (auto primitive = value.as<tvm::PrimExpr>()) {
            _reads({primitive.value()});
        } else {
            known = false;
        }
    }

    void statement(const tvm::tirx::Stmt &value) {
        if (auto sequence = value.as<tvm::tirx::SeqStmtNode>()) {
            for (auto &&child : sequence->seq) { statement(child); }
        } else if (auto store = value.as<tvm::tirx::BufferStoreNode>()) {
            auto expressions = store->indices;
            expressions.push_back(store->value);
            if (store->predicate) { expressions.push_back(store->predicate.value()); }
            _reads(expressions);
            if (store->value.ty() != store->buffer->dtype) { known = false; }
            _access(store->buffer, false);
        } else if (auto evaluate = value.as<tvm::tirx::EvaluateNode>()) {
            expression(evaluate->value);
        } else if (auto bind = value.as<tvm::tirx::BindNode>()) {
            expression(bind->value);
        } else if (auto branch = value.as<tvm::tirx::IfThenElseNode>()) {
            expression(branch->condition);
            statement(branch->then_case);
            if (branch->else_case) { statement(branch->else_case.value()); }
        } else if (auto loop = value.as<tvm::tirx::ForNode>()) {
            auto count = static_extent(loop->extent);
            if (!count || !unit_serial_loop(loop)) {
                known = false;
                return;
            }
            PayloadAccessCounter body;
            body.statement(loop->body);
            known &= body.known;
            add_access_demand(demand, body.demand, static_cast<double>(*count));
        } else if (!value.as<tvm::tirx::AllocBufferNode>()) {
            known = false;
        }
    }
};

class DistributedAccessAnalysis final : public tvm::tirx::StmtVisitor {
private:
    const ReductionAnalysis &_analysis;
    struct Domain {
        uint64_t elements;
        double repetitions;
        ReductionAccessDemand accesses;
    };
    luisa::vector<Domain> _domains;
    double _repetitions{1.0};

    void VisitStmt_(const tvm::tirx::ForNode *loop) final {
        PayloadAccessCounter counter;
        auto elements = uint64_t{0u};
        if (auto iter = _analysis.reductions.find(loop); iter != _analysis.reductions.end()) {
            elements = iter->second.elements;
            // Carry traffic is part of scalar recurrence/collective service,
            // not a load from the logical payload. Do not count its scaffolding.
            counter.expression(iter->second.contribution);
        } else if (loop->annotations.count(independent_elements_annotation) &&
                   !_analysis.replicated_elements.contains(loop)) {
            auto domain = element_domain(loop);
            if (!domain) {
                known = false;
                return;
            }
            elements = domain->count;
            counter.statement(domain->body);
        } else {
            auto count = static_extent(loop->extent);
            if (!count || !unit_serial_loop(loop)) {
                known = false;
                return;
            }
            auto previous = _repetitions;
            _repetitions *= static_cast<double>(*count);
            if (!std::isfinite(_repetitions)) { known = false; }
            StmtVisitor::VisitStmt_(loop);
            _repetitions = previous;
            return;
        }
        known &= counter.known;
        _domains.emplace_back(Domain{elements, _repetitions, counter.demand});
    }

public:
    bool known{true};
    explicit DistributedAccessAnalysis(const ReductionAnalysis &analysis) noexcept : _analysis{analysis} {}

    void finish() noexcept {
        auto value = demand();
        known &= std::isfinite(value.global_read_bytes) && std::isfinite(value.global_write_bytes) &&
                 std::isfinite(value.private_read_bytes) && std::isfinite(value.private_write_bytes);
    }

    [[nodiscard]] ReductionAccessDemand demand(uint64_t workers = 0u, uint64_t lane_elements = 1u) const noexcept {
        ReductionAccessDemand result;
        if (known) {
            for (auto &&domain : _domains) {
                auto elements = workers ? stripe_slots(domain.elements, workers, lane_elements) : domain.elements;
                add_access_demand(result, domain.accesses, static_cast<double>(elements) * domain.repetitions);
            }
        }
        return result;
    }
};

struct StripedAccess {
    uint64_t allocations{0u};
    uint64_t stores{0u};
    uint64_t loads{0u};
    bool valid{true};
};

class StripedMaterializationAudit final
    : public tvm::tirx::StmtExprVisitor {
private:
    const ReductionAnalysis &_reductions;
    const luisa::unordered_map<BufferKey, StripedMaterialization> &_candidates;
    luisa::vector<const tvm::tirx::ForNode *> _domain;
    std::optional<tvm::PrimExpr> _owner;

    [[nodiscard]] bool _owned_access(
        const StripedMaterialization &candidate,
        const tvm::ffi::Array<tvm::PrimExpr> &indices) const {
        if (!_owner || indices.size() != candidate.buffer->shape.size()) {
            return false;
        }
        tvm::PrimExpr linear = tvm::IntImm::Int64(0);
        for (auto i = size_t{0u}; i < indices.size(); i++) {
            linear = linear * candidate.buffer->shape[i] + indices[i];
        }
        return prove_in_loop_domain(tvm::equal(linear, _owner.value()),
                                    _domain);
    }

protected:
    void VisitStmt_(const tvm::tirx::ForNode *loop) final {
        _domain.emplace_back(loop);
        auto previous_owner = _owner;
        if (_reductions.reductions.contains(loop)) {
            _owner = loop->loop_var - loop->min;
        } else if (loop->annotations.count(independent_elements_annotation) &&
                   !_reductions.replicated_elements.contains(loop)) {
            if (auto element = element_domain(loop)) {
                tvm::PrimExpr linear = tvm::IntImm::Int64(0);
                for (auto axis : element->axes) {
                    linear = linear * axis->extent +
                             (axis->loop_var - axis->min);
                }
                _owner = std::move(linear);
            } else {
                _owner.reset();
            }
        }
        StmtExprVisitor::VisitStmt_(loop);
        _owner = std::move(previous_owner);
        _domain.pop_back();
    }

    void VisitStmt_(const tvm::tirx::AllocBufferNode *allocation) final {
        if (auto iter = _candidates.find(allocation->buffer.get());
            iter != _candidates.end()) {
            auto &record = access[iter->first];
            record.allocations++;
            record.valid &= allocation->annotations.empty();
        }
        StmtExprVisitor::VisitStmt_(allocation);
    }

    void VisitStmt_(const tvm::tirx::BufferStoreNode *store) final {
        if (auto iter = _candidates.find(store->buffer.get());
            iter != _candidates.end()) {
            auto &record = access[iter->first];
            record.stores++;
            record.valid &= store == iter->second.store &&
                            _owned_access(iter->second, store->indices);
        }
        StmtExprVisitor::VisitStmt_(store);
    }

    void VisitExpr_(const tvm::tirx::BufferLoadNode *load) final {
        if (auto iter = _candidates.find(load->buffer.get());
            iter != _candidates.end()) {
            auto &record = access[iter->first];
            record.loads++;
            record.valid &= _owned_access(iter->second, load->indices);
        }
        StmtExprVisitor::VisitExpr_(load);
    }

public:
    luisa::unordered_map<BufferKey, StripedAccess> access;

    StripedMaterializationAudit(
        const ReductionAnalysis &reductions,
        const luisa::unordered_map<BufferKey,
                                   StripedMaterialization> &candidates) noexcept
        : _reductions{reductions}, _candidates{candidates} {}
};

[[nodiscard]] luisa::unordered_map<BufferKey, StripedMaterialization>
striped_materializations(const tvm::tirx::Stmt &body,
                         const ReductionAnalysis &reductions) {
    luisa::unordered_map<BufferKey, StripedMaterialization> candidates;
    luisa::unordered_set<BufferKey> duplicates;
    tvm::tirx::PostOrderVisit(body, [&](const tvm::ffi::ObjectRef &node) {
        auto loop = node.as<tvm::tirx::ForNode>();
        if (loop == nullptr ||
            !loop->annotations.count(materialized_pure_tile_annotation)) {
            return;
        }
        if (auto matched = match_striped_materialization(loop)) {
            auto key = matched->buffer.get();
            if (!candidates.emplace(key, std::move(*matched)).second) {
                duplicates.emplace(key);
            }
        }
    });
    for (auto key : duplicates) { candidates.erase(key); }
    StripedMaterializationAudit audit{reductions, candidates};
    audit(body);
    luisa::vector<BufferKey> rejected;
    for (auto &&[key, candidate] : candidates) {
        static_cast<void>(candidate);
        auto iter = audit.access.find(key);
        if (iter == audit.access.end() || !iter->second.valid ||
            iter->second.allocations != 1u || iter->second.stores != 1u ||
            iter->second.loads == 0u) {
            rejected.emplace_back(key);
        }
    }
    for (auto key : rejected) { candidates.erase(key); }
    return candidates;
}

[[nodiscard]] bool contains_reduction(
    const tvm::tirx::Stmt &statement,
    const ReductionAnalysis &analysis) {
    auto found = false;
    tvm::tirx::PostOrderVisit(statement, [&](const tvm::ffi::ObjectRef &node) {
        auto loop = node.as<tvm::tirx::ForNode>();
        found |= loop != nullptr && analysis.reductions.contains(loop);
    });
    return found;
}

class ProgramAudit final : public tvm::tirx::StmtExprVisitor {
private:
    const ReductionAnalysis &_analysis;
    uint32_t _distributed_depth{0u};

protected:
    void VisitStmt_(const tvm::tirx::ForNode *loop) final {
        if (loop->annotations.count(logical_parallel_annotation)) {
            valid = false;
        }
        auto distributed =
            loop->annotations.count(independent_elements_annotation) &&
            !_analysis.replicated_elements.contains(loop);
        _distributed_depth += distributed;
        StmtExprVisitor::VisitStmt_(loop);
        _distributed_depth -= distributed;
    }

    void VisitStmt_(const tvm::tirx::IfThenElseNode *branch) final {
        if (contains_reduction(tvm::ffi::GetRef<tvm::tirx::IfThenElse>(branch),
                               _analysis)) {
            valid = false;
        }
        StmtExprVisitor::VisitStmt_(branch);
    }

    void VisitStmt_(const tvm::tirx::BufferStoreNode *store) final {
        if (store->buffer.scope() != "local" && _distributed_depth == 0u) {
            valid = false;
        }
        for (auto &&[loop, reduction] : _analysis.reductions) {
            static_cast<void>(loop);
            if (store->buffer.same_as(reduction.carry) &&
                store != reduction.initializer && store != reduction.update) {
                valid = false;
            }
        }
        StmtExprVisitor::VisitStmt_(store);
    }

    void VisitStmt_(const tvm::tirx::AllocBufferNode *allocation) final {
        if (auto constraint =
                allocation->annotations.Get(memory_resource_annotation)) {
            auto resource = constraint.value().as<tvm::ffi::String>();
            valid &= resource && resource.value() == "private";
        }
        StmtExprVisitor::VisitStmt_(allocation);
    }

    void VisitExpr_(const tvm::CallNode *call) final {
        static auto effects =
            tvm::Op::GetAttrMap<tvm::tirx::TCallEffectKind>("TCallEffectKind");
        auto op = call->op.as<tvm::Op>();
        valid &= op && effects.count(op.value()) != 0u &&
                 effects[op.value()] <=
                     static_cast<int64_t>(tvm::tirx::CallEffectKind::kPure);
        StmtExprVisitor::VisitExpr_(call);
    }

    void VisitStmt_(const tvm::tirx::WhileNode *loop) final {
        valid = false;
        StmtExprVisitor::VisitStmt_(loop);
    }
    void VisitStmt_(const tvm::tirx::BreakNode *) final { valid = false; }
    void VisitStmt_(const tvm::tirx::ContinueNode *) final { valid = false; }
    void VisitStmt_(const tvm::tirx::ReturnNode *) final { valid = false; }
    void VisitStmt_(const tvm::tirx::AssertStmtNode *statement) final {
        valid = false;
        StmtExprVisitor::VisitStmt_(statement);
    }
    void VisitStmt_(const tvm::tirx::TilePrimitiveCallNode *) final {
        valid = false;
    }

public:
    bool valid{true};
    explicit ProgramAudit(const ReductionAnalysis &analysis) noexcept
        : _analysis{analysis} {}
};

struct DistributedLocalAccess {
    uint64_t allocations{0u};
    uint64_t distributed_stores{0u};
    uint64_t loads{0u};
    bool stores_owned{true};
    bool loads_owned{true};
};

// A local Tile is private to a physical worker. Once an element-domain store
// is distributed, another worker cannot read that element from its own private
// allocation. Prove the compact row-major address to be the current logical
// owner for every later use. This deliberately rejects permutations and
// opaque/dynamic ownership rather than silently compiling a cross-worker read.
class DistributedLocalAudit final : public tvm::tirx::StmtExprVisitor {
private:
    const ReductionAnalysis &_reductions;
    luisa::vector<const tvm::tirx::ForNode *> _domain;
    std::optional<tvm::PrimExpr> _owner;
    luisa::unordered_map<BufferKey, DistributedLocalAccess> _access;

    [[nodiscard]] static bool _requires_ownership(
        const tvm::tirx::BufferVar &buffer) noexcept {
        if (buffer.scope() != "local" || buffer->shape.empty()) { return false; }
        return std::any_of(
            buffer->shape.begin(), buffer->shape.end(),
            [](const tvm::PrimExpr &dimension) noexcept {
                auto extent = dimension.as<tvm::IntImmNode>();
                return extent == nullptr || extent->value != 1;
            });
    }

    [[nodiscard]] bool _owned_access(
        const tvm::tirx::BufferVar &buffer,
        const tvm::ffi::Array<tvm::PrimExpr> &indices) const {
        auto offset = buffer->elem_offset.as<tvm::IntImmNode>();
        if (!_owner || indices.size() != buffer->shape.size() ||
            !buffer->strides.empty() || buffer->layout ||
            !buffer->allocated_addr.empty() || offset == nullptr ||
            offset->value != 0) {
            return false;
        }
        tvm::PrimExpr linear = tvm::IntImm::Int64(0);
        for (auto i = size_t{0u}; i < indices.size(); i++) {
            linear = linear * buffer->shape[i] + indices[i];
        }
        return prove_in_loop_domain(tvm::equal(linear, _owner.value()),
                                    _domain);
    }

protected:
    void VisitStmt_(const tvm::tirx::ForNode *loop) final {
        _domain.emplace_back(loop);
        auto previous_owner = _owner;
        if (_reductions.reductions.contains(loop)) {
            _owner = loop->loop_var - loop->min;
        } else if (!_owner &&
                   loop->annotations.count(independent_elements_annotation) &&
                   !_reductions.replicated_elements.contains(loop)) {
            if (auto element = element_domain(loop)) {
                tvm::PrimExpr linear = tvm::IntImm::Int64(0);
                for (auto axis : element->axes) {
                    linear = linear * axis->extent +
                             (axis->loop_var - axis->min);
                }
                _owner = std::move(linear);
            }
        }
        StmtExprVisitor::VisitStmt_(loop);
        _owner = std::move(previous_owner);
        _domain.pop_back();
    }

    void VisitStmt_(const tvm::tirx::AllocBufferNode *allocation) final {
        if (_requires_ownership(allocation->buffer)) {
            _access[allocation->buffer.get()].allocations++;
        }
        StmtExprVisitor::VisitStmt_(allocation);
    }

    void VisitStmt_(const tvm::tirx::BufferStoreNode *store) final {
        if (_requires_ownership(store->buffer) && _owner) {
            auto &record = _access[store->buffer.get()];
            record.distributed_stores++;
            record.stores_owned &= _owned_access(store->buffer, store->indices);
        }
        StmtExprVisitor::VisitStmt_(store);
    }

    void VisitExpr_(const tvm::tirx::BufferLoadNode *load) final {
        if (_requires_ownership(load->buffer)) {
            auto &record = _access[load->buffer.get()];
            record.loads++;
            record.loads_owned &= _owned_access(load->buffer, load->indices);
        }
        StmtExprVisitor::VisitExpr_(load);
    }

public:
    explicit DistributedLocalAudit(
        const ReductionAnalysis &reductions) noexcept
        : _reductions{reductions} {}

    [[nodiscard]] bool valid() const noexcept {
        return std::all_of(
            _access.begin(), _access.end(),
            [](const auto &item) noexcept {
                auto &record = item.second;
                return record.distributed_stores == 0u ||
                       (record.allocations == 1u && record.stores_owned &&
                        (record.loads == 0u || record.loads_owned));
            });
    }
};

class ReductionProgramMapper final : public tvm::tirx::StmtExprMutator {
private:
    tvm::PrimExpr _worker;
    tvm::PrimExpr _lane;
    tvm::PrimExpr _subgroup;
    uint64_t _workers;
    uint64_t _subgroups;
    uint32_t _unroll_factor;
    uint32_t _lane_elements;
    const ReductionAnalysis &_analysis;
    const luisa::unordered_map<const tvm::tirx::ForNode *,
                               tvm::tirx::BufferVar> &_partials;
    const luisa::unordered_map<BufferKey, tvm::tirx::BufferVar>
        &_striped_buffers;
    std::optional<tvm::PrimExpr> _striped_slot;
    uint32_t _lane_depth{0u};

    [[nodiscard]] tvm::tirx::Stmt _stripe_loop(const tvm::tirx::PrimVar &chunk, uint64_t chunks,
                                               tvm::tirx::Stmt body) const {
        auto zero = tvm::IntImm::Int64(0);
        if (chunks == 0u) { return tvm::tirx::Evaluate{zero}; }
        if (_unroll_factor == 1u) {
            return tvm::tirx::For{chunk, zero, tvm::IntImm::Int64(static_cast<int64_t>(chunks)),
                                  tvm::tirx::ForKind::kSerial, std::move(body)};
        }
        auto factor = std::min<uint64_t>(_unroll_factor, chunks);
        auto pack = tvm::tirx::PrimVar{chunk->name + "_pack", tvm::PrimType::Int(64)};
        auto slot = tvm::tirx::PrimVar{chunk->name + "_slot", tvm::PrimType::Int(64)};
        auto width = tvm::IntImm::Int64(static_cast<int64_t>(factor));
        auto inner = tvm::tirx::Substitute(body, tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>{{chunk, pack * width + slot}});
        inner = tvm::tirx::For{slot, zero, width, tvm::tirx::ForKind::kUnrolled, std::move(inner)};
        inner = tvm::tirx::For{pack, zero, tvm::IntImm::Int64(static_cast<int64_t>(chunks / factor)),
                               tvm::tirx::ForKind::kSerial, std::move(inner)};
        if (chunks % factor == 0u) { return inner; }
        auto tail = tvm::tirx::For{chunk, tvm::IntImm::Int64(static_cast<int64_t>(chunks / factor * factor)),
                                   tvm::IntImm::Int64(static_cast<int64_t>(chunks % factor)),
                                   tvm::tirx::ForKind::kUnrolled, std::move(body)};
        return tvm::tirx::SeqStmt::Flatten(tvm::ffi::Array<tvm::tirx::Stmt>{std::move(inner), std::move(tail)});
    }

    [[nodiscard]] tvm::ffi::Optional<tvm::PrimExpr> _predicate(
        const tvm::ffi::Optional<tvm::PrimExpr> &predicate) {
        return predicate ? VisitPrimExpr(predicate.value()) :
                           tvm::ffi::Optional<tvm::PrimExpr>{};
    }

    [[nodiscard]] tvm::tirx::Stmt _element_pack(
        const tvm::tirx::PrimVar &element, tvm::tirx::Stmt body) const {
        if (_lane_elements == 1u) {
            return tvm::tirx::Substitute(std::move(body),
                                         tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>{{element, tvm::IntImm::Int64(0)}});
        }
        return tvm::tirx::For{element, tvm::IntImm::Int64(0),
                              tvm::IntImm::Int64(_lane_elements),
                              tvm::tirx::ForKind::kUnrolled, std::move(body)};
    }

    [[nodiscard]] tvm::tirx::Stmt _distributed_loop(
        const tvm::tirx::PrimVar &chunk, const tvm::tirx::PrimVar &element,
        const tvm::PrimExpr &linear, uint64_t elements,
        tvm::tirx::Stmt body) const {
        auto stride = _workers * _lane_elements;
        auto complete_chunks = elements / stride;
        tvm::ffi::Array<tvm::tirx::Stmt> statements;
        if (complete_chunks != 0u) {
            statements.push_back(_stripe_loop(chunk, complete_chunks, _element_pack(element, body)));
        }
        // Only the last, partial pack carries a bounds check. Keeping the
        // complete domain separate exposes consecutive accesses to codegen
        // without speculatively evaluating a tail load or private-array use.
        if (elements % stride != 0u) {
            auto tail = _element_pack(element, tvm::tirx::IfThenElse{
                                                   linear < tvm::IntImm::Int64(static_cast<int64_t>(elements)),
                                                   std::move(body)});
            statements.push_back(tvm::tirx::Substitute(std::move(tail),
                                                       tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>{{chunk, tvm::IntImm::Int64(static_cast<int64_t>(complete_chunks))}}));
        }
        if (statements.empty()) { return tvm::tirx::Evaluate{tvm::IntImm::Int32(0)}; }
        return tvm::tirx::SeqStmt::Flatten(statements);
    }

    [[nodiscard]] tvm::tirx::Stmt _reduction(
        const tvm::tirx::ForNode *loop,
        const ReductionMatch &match) {
        auto chunk = tvm::tirx::PrimVar{
            loop->loop_var->name + "_subgroup_chunk", tvm::PrimType::Int(64)};
        auto element = tvm::tirx::PrimVar{
            loop->loop_var->name + "_lane_element", tvm::PrimType::Int(64)};
        auto width = tvm::IntImm::Int64(_lane_elements);
        auto linear = (chunk * tvm::IntImm::Int64(static_cast<int64_t>(_workers)) + _worker) * width + element;
        auto previous_slot = std::move(_striped_slot);
        _striped_slot = chunk * width + element;
        auto contribution = tvm::tirx::Substitute(
            VisitPrimExpr(match.contribution),
            tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>{
                {loop->loop_var, linear}});
        _striped_slot = std::move(previous_slot);
        auto current = tvm::tirx::BufferLoad{
            match.carry, {tvm::IntImm::Int64(0)}};
        tvm::PrimExpr combined;
        if (match.kind == reduction_add_contract) {
            combined = current + contribution;
        } else if (match.kind == reduction_max_contract) {
            combined = tvm::max(current, contribution);
        } else {
            combined = tvm::min(current, contribution);
        }
        tvm::tirx::Stmt update = tvm::tirx::BufferStore{
            match.carry, std::move(combined), {tvm::IntImm::Int64(0)}};
        auto striped = _distributed_loop(chunk, element, linear, match.elements, std::move(update));
        auto intrinsic = match.kind == reduction_add_contract ?
                             "simd_sum" :
                         match.kind == reduction_max_contract ?
                             "simd_max" :
                             "simd_min";
        tvm::PrimExpr collective = tvm::Call{
            tvm::PrimType::Float(32),
            tvm::tirx::builtin::call_pure_extern(),
            {tvm::tirx::StringImm{intrinsic},
             tvm::tirx::BufferLoad{
                 match.carry, {tvm::IntImm::Int64(0)}}}};
        tvm::ffi::Array<tvm::tirx::Stmt> statements{std::move(striped)};
        if (_subgroups == 1u) {
            statements.push_back(tvm::tirx::BufferStore{
                match.carry, std::move(collective), {tvm::IntImm::Int64(0)}});
            return tvm::tirx::SeqStmt::Flatten(statements);
        }
        auto partial = _partials.at(loop);
        statements.push_back(tvm::tirx::BufferStore{
            match.carry, std::move(collective), {tvm::IntImm::Int64(0)}});
        statements.push_back(tvm::tirx::IfThenElse{
            tvm::equal(_lane, tvm::IntImm::Int64(0)),
            tvm::tirx::BufferStore{
                partial,
                tvm::tirx::BufferLoad{
                    match.carry, {tvm::IntImm::Int64(0)}},
                {_subgroup}}});
        statements.push_back(shared_barrier());
        auto input = tvm::if_then_else(
            _lane < tvm::IntImm::Int64(static_cast<int64_t>(_subgroups)),
            tvm::tirx::BufferLoad{partial, {_lane}},
            reduction_identity(match.kind));
        auto total = tvm::Call{
            tvm::PrimType::Float(32),
            tvm::tirx::builtin::call_pure_extern(),
            {tvm::tirx::StringImm{intrinsic}, std::move(input)}};
        statements.push_back(tvm::tirx::BufferStore{
            match.carry, std::move(total), {tvm::IntImm::Int64(0)}});
        return tvm::tirx::SeqStmt::Flatten(statements);
    }

    [[nodiscard]] tvm::tirx::Stmt _distributed_elements(
        const tvm::tirx::ForNode *loop, const ElementDomain &domain) {
        auto chunk = tvm::tirx::PrimVar{
            loop->loop_var->name + "_subgroup_chunk", tvm::PrimType::Int(64)};
        auto element = tvm::tirx::PrimVar{
            loop->loop_var->name + "_lane_element", tvm::PrimType::Int(64)};
        auto width = tvm::IntImm::Int64(_lane_elements);
        auto previous_slot = std::move(_striped_slot);
        _striped_slot = chunk * width + element;
        _lane_depth++;
        auto body = VisitStmt(domain.body);
        _lane_depth--;
        _striped_slot = std::move(previous_slot);
        if (domain.count == 0u) {
            return tvm::tirx::Evaluate{tvm::IntImm::Int32(0)};
        }
        auto linear = (chunk * tvm::IntImm::Int64(static_cast<int64_t>(_workers)) + _worker) * width + element;
        tvm::ffi::Map<tvm::tirx::Var, tvm::Expr> coordinates;
        auto trailing = domain.count;
        for (auto axis : domain.axes) {
            auto extent = *static_extent(axis->extent);
            trailing /= extent;
            tvm::PrimExpr coordinate = linear;
            if (domain.axes.size() != 1u) {
                coordinate = tvm::floormod(
                    tvm::floordiv(coordinate,
                                  tvm::IntImm::Int64(
                                      static_cast<int64_t>(trailing))),
                    axis->extent);
            }
            coordinates.Set(axis->loop_var, axis->min + coordinate);
        }
        body = tvm::tirx::Substitute(std::move(body), coordinates);
        return _distributed_loop(chunk, element, linear, domain.count, std::move(body));
    }

protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(
        const tvm::tirx::ForNode *loop) final {
        if (auto iter = _analysis.reductions.find(loop);
            iter != _analysis.reductions.end()) {
            return _reduction(loop, iter->second);
        }
        if (loop->annotations.count(independent_elements_annotation)) {
            if (_lane_depth == 0u &&
                !_analysis.replicated_elements.contains(loop)) {
                auto domain = element_domain(loop);
                if (!domain) {
                    throw std::runtime_error{
                        "validated SIMD-group element domain became invalid"};
                }
                return _distributed_elements(loop, *domain);
            }
            auto result = StmtExprMutator::VisitStmt_(loop)
                              .as_or_throw<tvm::tirx::For>();
            auto node = result.CopyOnWrite();
            node->annotations.erase(independent_elements_annotation);
            node->annotations.erase(materialized_pure_tile_annotation);
            node->annotations.erase(mma_annotation);
            return result;
        }
        auto result = StmtExprMutator::VisitStmt_(loop)
                          .as_or_throw<tvm::tirx::For>();
        auto node = result.CopyOnWrite();
        node->annotations.erase(deferred_pipeline_annotation);
        node->annotations.erase(reduction_contract_annotation);
        node->annotations.erase(materialized_pure_tile_annotation);
        return result;
    }

    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(
        const tvm::tirx::AllocBufferNode *allocation) final {
        auto buffer = allocation->buffer;
        if (auto iter = _striped_buffers.find(buffer.get());
            iter != _striped_buffers.end()) {
            buffer = iter->second;
        }
        auto result = tvm::tirx::AllocBuffer{
            std::move(buffer), allocation->annotations, allocation->span};
        auto node = result.CopyOnWrite();
        node->annotations.erase(manual_memory_annotation);
        node->annotations.erase(memory_resource_annotation);
        return result;
    }

    [[nodiscard]] tvm::Expr VisitExpr_(
        const tvm::tirx::BufferLoadNode *load) final {
        if (auto iter = _striped_buffers.find(load->buffer.get());
            iter != _striped_buffers.end()) {
            if (!_striped_slot) {
                throw std::runtime_error{
                    "proved striped Tile storage escaped its element domain"};
            }
            return tvm::tirx::BufferLoad{
                iter->second, {_striped_slot.value()}, _predicate(load->predicate), load->span};
        }
        return StmtExprMutator::VisitExpr_(load);
    }

    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(
        const tvm::tirx::BufferStoreNode *store) final {
        if (auto iter = _striped_buffers.find(store->buffer.get());
            iter != _striped_buffers.end()) {
            if (!_striped_slot) {
                throw std::runtime_error{
                    "proved striped Tile storage escaped its element domain"};
            }
            return tvm::tirx::BufferStore{
                iter->second, VisitPrimExpr(store->value), {_striped_slot.value()}, _predicate(store->predicate), store->span};
        }
        return StmtExprMutator::VisitStmt_(store);
    }

    [[nodiscard]] tvm::Expr VisitExpr_(
        const tvm::tirx::VarNode *variable) final {
        if (_striped_buffers.contains(variable)) {
            throw std::runtime_error{
                "proved striped Tile storage escaped through an opaque use"};
        }
        return StmtExprMutator::VisitExpr_(variable);
    }

public:
    ReductionProgramMapper(
        tvm::PrimExpr worker, tvm::PrimExpr lane,
        tvm::PrimExpr subgroup, uint64_t workers, uint64_t subgroups, uint32_t unroll_factor, uint32_t lane_elements,
        const ReductionAnalysis &analysis,
        const luisa::unordered_map<const tvm::tirx::ForNode *,
                                   tvm::tirx::BufferVar> &partials,
        const luisa::unordered_map<BufferKey,
                                   tvm::tirx::BufferVar> &striped_buffers) noexcept
        : _worker{std::move(worker)}, _lane{std::move(lane)},
          _subgroup{std::move(subgroup)}, _workers{workers},
          _subgroups{subgroups}, _unroll_factor{unroll_factor}, _lane_elements{lane_elements}, _analysis{analysis}, _partials{partials},
          _striped_buffers{striped_buffers} {}
};

}// namespace

tvm::tirx::Stmt try_map_metal_subgroup_reduction(
    const tvm::tirx::For &loop, uint32_t max_threads,
    uint64_t shared_memory_limit,
    const PlannerOptions &options, luisa::vector<GroupPlan> &plans) {
    auto groups = static_extent(loop->extent, true);
    auto minimum = loop->min.as<tvm::IntImmNode>();
    auto scope = loop->annotations.Get(execution_scope_annotation);
    auto scope_name = scope ? scope.value().as<tvm::ffi::String>() :
                              tvm::ffi::Optional<tvm::ffi::String>{};
    if (!options.metal_subgroup_reductions ||
        options.max_reduction_striped_scalars_per_worker == 0u ||
        !unit_serial_loop(loop.get()) ||
        !groups || minimum == nullptr || loop->loop_var.ty() != tvm::PrimType::Int(64) ||
        (scope && (!scope_name || scope_name.value() != "subgroup")) ||
        max_threads < subgroup_size) {
        return {};
    }

    ReductionAnalysis analysis;
    analysis(loop->body);
    analysis.finish(loop->body);
    if (!analysis.valid) { return {}; }
    ProgramAudit audit{analysis};
    audit(loop->body);
    if (!audit.valid) { return {}; }
    DistributedLocalAudit ownership{analysis};
    ownership(loop->body);
    if (!ownership.valid()) { return {}; }
    auto materializations = striped_materializations(loop->body, analysis);
    DistributedAccessAnalysis accesses{analysis};
    accesses(loop->body);
    accesses.finish();

    // The second collective reads one partial per lane, so it can combine
    // up to subgroup_size subgroups. This is an algorithmic bound, distinct
    // from the target's thread limit or the automatic search budget.
    auto maximum_subgroups = std::min<uint64_t>(
        subgroup_size, max_threads / subgroup_size);
    auto default_policy = AnalyticExecutionCostPolicy{};
    auto &policy = options.cost_policy ? *options.cost_policy : default_policy;
    auto model = policy.coefficients(
        ExecutionLimits{max_threads, subgroup_size, shared_memory_limit},
        MatrixCostBasis::SIMDGROUP_REFERENCE, options.cost);
    auto scalar_round_cost = model.subgroup_reduction_scalar_round;
    auto collective_cost = model.subgroup_reduction_collective;
    auto group_setup_cost = model.subgroup_reduction_group_setup;
    if (!std::isfinite(scalar_round_cost) || scalar_round_cost < 0.0 ||
        !std::isfinite(collective_cost) || collective_cost < 0.0 ||
        !std::isfinite(group_setup_cost) || group_setup_cost < 0.0 ||
        !std::isfinite(model.subgroup_reduction_global_access_byte) || model.subgroup_reduction_global_access_byte < 0.0 ||
        !std::isfinite(model.subgroup_reduction_private_access_byte) || model.subgroup_reduction_private_access_byte < 0.0 ||
        options.max_thread_candidates == 0u) {
        throw std::runtime_error{"invalid reduction cost coefficients or search budget"};
    }
    struct Candidate {
        uint64_t subgroups{0u};
        uint64_t packed_programs{0u};
        uint64_t threads{0u};
        uint64_t partial_bytes{0u};
        uint64_t striped_storage_scalars{0u};
        double scalar_rounds{0.0};
        double lane_utilization{0.0};
        ReductionCost cost{0.0, 1.0, std::numeric_limits<double>::infinity()};
    };
    luisa::vector<uint64_t> widths;
    auto exact_packing = options.reduction_programs_per_group;
    if (exact_packing > 1u) {
        if (exact_packing > maximum_subgroups ||
            (options.threads_per_group != 0u &&
             options.threads_per_group != exact_packing * subgroup_size)) {
            return {};
        }
        widths.emplace_back(1u);
    } else if (options.threads_per_group != 0u) {
        if (options.threads_per_group > max_threads ||
            options.threads_per_group % subgroup_size != 0u) {
            return {};
        }
        widths.emplace_back(options.threads_per_group / subgroup_size);
    } else {
        if (maximum_subgroups > options.max_thread_candidates) {
            throw std::runtime_error{"reduction thread candidate budget exceeded; increase the budget or request an exact width"};
        }
        for (auto subgroups = uint64_t{1u};
             subgroups <= maximum_subgroups; subgroups++) {
            widths.emplace_back(subgroups);
        }
    }
    auto best = Candidate{};
    auto candidates_considered = uint64_t{0u};
    auto candidates_rejected = uint64_t{0u};
    for (auto subgroups : widths) {
        auto multi = subgroups > 1u;
        auto partial_bytes = multi ?
                                 analysis.reductions.size() * subgroups *
                                     sizeof(float) :
                                 0u;
        auto workers = subgroups * subgroup_size;
        auto striped_storage_scalars = uint64_t{0u};
        auto striped_storage_valid = true;
        auto striped_storage_budget = static_cast<uint64_t>(
            options.max_reduction_striped_scalars_per_worker);
        for (auto &&[key, materialization] : materializations) {
            static_cast<void>(key);
            auto slots = stripe_slots(materialization.elements, workers, options.reduction_lane_elements);
            if (slots > striped_storage_budget ||
                striped_storage_scalars > striped_storage_budget - slots) {
                striped_storage_valid = false;
                break;
            }
            striped_storage_scalars += slots;
        }
        if (subgroups == 0u || subgroups > maximum_subgroups ||
            partial_bytes > shared_memory_limit || !striped_storage_valid) {
            candidates_rejected++;
            continue;
        }
        auto scalar_rounds = 0.0;
        auto scalar_elements = 0.0;
        for (auto elements : analysis.independent_domains) {
            scalar_elements += static_cast<double>(elements);
            scalar_rounds +=
                static_cast<double>(stripe_slots(elements, workers, options.reduction_lane_elements));
        }
        for (auto reduction : analysis.reduction_order) {
            auto elements = analysis.reductions.at(reduction).elements;
            scalar_elements += static_cast<double>(elements);
            scalar_rounds +=
                static_cast<double>(stripe_slots(elements, workers, options.reduction_lane_elements));
        }
        // Packing independent programs is a separate search dimension from
        // the cooperating width of one program. Multiple-subgroup programs
        // retain a whole group: their barriers must not be guarded by a tail.
        auto packing_begin = exact_packing ? static_cast<uint64_t>(exact_packing) : 1u;
        auto packing_end = !multi && !exact_packing && options.threads_per_group == 0u ?
                               std::min({*groups, maximum_subgroups, uint64_t{8u}}) :
                               packing_begin;
        auto lane_utilization = scalar_rounds == 0.0 ? 0.0 :
                                                       scalar_elements / (scalar_rounds * static_cast<double>(workers));
        for (auto packed = packing_begin; packed <= packing_end; packed++) {
            candidates_considered++;
            auto threads = (multi ? subgroups : packed) * subgroup_size;
            auto features = ReductionCandidate{
                *groups, static_cast<uint32_t>(threads),
                static_cast<uint32_t>(subgroups), static_cast<uint32_t>(packed),
                partial_bytes, striped_storage_scalars, analysis.reductions.size(), scalar_rounds, options.reduction_unroll_factor, options.reduction_lane_elements,
                luisa::ceil_div(*groups, packed), scalar_elements, lane_utilization,
                accesses.known, accesses.demand(), accesses.demand(workers, options.reduction_lane_elements)};
            auto cost = policy.reduction_cost(features, model);
            if (!std::isfinite(cost.program_score) || cost.program_score < 0.0 ||
                !std::isfinite(cost.concurrent_waves) || cost.concurrent_waves < 1.0 ||
                !std::isfinite(cost.kernel_score) || cost.kernel_score < 0.0) {
                throw std::runtime_error{"reduction cost policy returned a nonfinite or negative score"};
            }
            if (cost.kernel_score < best.cost.kernel_score) {
                best = Candidate{subgroups, packed, threads,
                                 partial_bytes, striped_storage_scalars, scalar_rounds, lane_utilization, cost};
            }
        }
    }
    if (best.subgroups == 0u) { return {}; }
    auto subgroups_per_program = best.subgroups;
    auto multi_subgroup = subgroups_per_program > 1u;
    auto partial_bytes = best.partial_bytes;
    auto packed_programs = best.packed_programs;
    auto threads = best.threads;
    auto blocks = multi_subgroup ?
                      *groups :
                      luisa::ceil_div(*groups, packed_programs);
    auto block = tvm::tirx::PrimVar{
        loop->loop_var->name + "_subgroup_block", tvm::PrimType::Int(64)};
    auto thread = tvm::tirx::PrimVar{
        loop->loop_var->name + "_subgroup_thread", tvm::PrimType::Int(64)};
    auto lane = tvm::tirx::PrimVar{
        loop->loop_var->name + "_subgroup_lane", tvm::PrimType::Int(64)};
    auto subgroup = tvm::floordiv(
        thread, tvm::IntImm::Int64(static_cast<int64_t>(subgroup_size)));
    tvm::PrimExpr logical = block;
    if (!multi_subgroup) {
        logical = block *
                      tvm::IntImm::Int64(
                          static_cast<int64_t>(packed_programs)) +
                  subgroup;
    }
    luisa::unordered_map<const tvm::tirx::ForNode *, tvm::tirx::BufferVar>
        partials;
    tvm::ffi::Array<tvm::tirx::Stmt> allocations;
    if (multi_subgroup) {
        for (auto reduction : analysis.reduction_order) {
            auto partial = tvm::tirx::decl_buffer(
                {tvm::IntImm::Int64(
                    static_cast<int64_t>(subgroups_per_program))},
                tvm::PrimType::Float(32),
                loop->loop_var->name + "_subgroup_partials_" +
                    std::to_string(partials.size()),
                "shared");
            partials.emplace(reduction, partial);
            allocations.push_back(tvm::tirx::AllocBuffer{std::move(partial)});
        }
    }
    tvm::PrimExpr worker = multi_subgroup ? tvm::PrimExpr{thread} :
                                            tvm::PrimExpr{lane};
    auto program_workers = multi_subgroup ? threads : subgroup_size;
    luisa::unordered_map<BufferKey, tvm::tirx::BufferVar> striped_buffers;
    auto striped_storage_scalars = uint64_t{0u};
    for (auto &&[key, materialization] : materializations) {
        auto slots =
            stripe_slots(materialization.elements, program_workers, options.reduction_lane_elements);
        striped_storage_scalars += slots;
        auto buffer = tvm::tirx::decl_buffer(
            {tvm::IntImm::Int64(static_cast<int64_t>(slots))},
            materialization.buffer->dtype,
            materialization.buffer.name() + "_worker_stripe", "local");
        striped_buffers.emplace(key, std::move(buffer));
    }
    if (striped_storage_scalars != best.striped_storage_scalars) {
        throw std::runtime_error{
            "reduction stripe resource accounting changed after planning"};
    }
    auto body = ReductionProgramMapper{
        std::move(worker), lane, subgroup,
        program_workers, multi_subgroup ? subgroups_per_program : 1u, options.reduction_unroll_factor, options.reduction_lane_elements,
        analysis, partials, striped_buffers}(loop->body);
    if (!allocations.empty()) {
        allocations.push_back(std::move(body));
        body = tvm::tirx::SeqStmt::Flatten(allocations);
    }
    body = tvm::tirx::Substitute(
        std::move(body),
        tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>{
            {loop->loop_var, loop->min + logical},
            {lane, tvm::floormod(
                       thread,
                       tvm::IntImm::Int64(
                           static_cast<int64_t>(subgroup_size)))}});
    if (!multi_subgroup && blocks * packed_programs != *groups) {
        body = tvm::tirx::IfThenElse{
            logical < loop->extent, std::move(body)};
    }
    auto zero = tvm::IntImm::Int64(0);
    auto thread_count = tvm::IntImm::Int64(static_cast<int64_t>(threads));
    auto thread_axis = tvm::tirx::IterVar{
        tvm::Range::FromMinExtent(zero, thread_count), thread,
        tvm::tirx::IterVarType::kThreadIndex, "threadIdx.x"};
    body = tvm::tirx::For{
        thread, zero, thread_count, tvm::tirx::ForKind::kThreadBinding,
        std::move(body), std::move(thread_axis)};
    auto block_count = tvm::IntImm::Int64(static_cast<int64_t>(blocks));
    auto block_axis = tvm::tirx::IterVar{
        tvm::Range::FromMinExtent(zero, block_count), block,
        tvm::tirx::IterVarType::kThreadIndex, "blockIdx.x"};
    body = tvm::tirx::For{
        block, zero, block_count, tvm::tirx::ForKind::kThreadBinding,
        std::move(body), std::move(block_axis)};

    GroupPlan plan;
    plan.name = std::string{loop->loop_var->name};
    plan.programs = *groups;
    plan.threads = static_cast<uint32_t>(threads);
    plan.shared_memory_bytes = partial_bytes;
    plan.candidates_considered = candidates_considered + candidates_rejected;
    plan.candidates_rejected = candidates_rejected;
    plan.reduction_subgroups_per_program =
        static_cast<uint32_t>(subgroups_per_program);
    plan.reduction_programs_per_group = static_cast<uint32_t>(packed_programs);
    plan.reduction_unroll_factor = options.reduction_unroll_factor;
    plan.reduction_lane_elements = options.reduction_lane_elements;
    plan.reduction_threadgroups = blocks;
    plan.reduction_scalar_rounds = best.scalar_rounds;
    plan.reduction_lane_utilization = best.lane_utilization;
    plan.reduction_payload_accesses_known = accesses.known;
    plan.reduction_payload_accesses_per_program = accesses.demand();
    plan.reduction_payload_accesses_per_worker = accesses.demand(program_workers, options.reduction_lane_elements);
    plan.striped_storage_scalars_per_worker = striped_storage_scalars;
    plan.reduction_operations = analysis.reductions.size();
    plan.reduction_elements = analysis.reduction_elements;
    plan.group_barrier_sites_before =
        multi_subgroup ? analysis.reductions.size() : 0u;
    plan.group_barrier_sites_after = plan.group_barrier_sites_before;
    plan.independent_subgroups = !multi_subgroup;
    plan.optimized = true;
    plan.cost.independent_elements =
        static_cast<double>(analysis.independent_elements);
    plan.cost.score = best.cost.program_score;
    plan.cost.concurrent_waves = best.cost.concurrent_waves;
    plan.cost.kernel_score = best.cost.kernel_score;
    plans.emplace_back(std::move(plan));
    return body;
}

}// namespace luisa::compute::tile::bridge::tirx::detail
