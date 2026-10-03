#include <algorithm>
#include <array>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>

#include <luisa/core/mathematics.h>
#include <luisa/core/platform.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/functional.h>
#include <luisa/core/stl/optional.h>

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

[[nodiscard]] luisa::optional<uint64_t> static_extent(
    const tvm::PrimExpr &expression, bool positive = false) noexcept {
    auto value = expression.as<tvm::IntImmNode>();
    if (value == nullptr || value->value < (positive ? 1 : 0)) { return luisa::nullopt; }
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

[[nodiscard]] luisa::optional<ElementDomain> element_domain(
    const tvm::tirx::ForNode *outer) noexcept {
    auto annotation = outer->annotations.Get(independent_elements_annotation);
    auto rank = annotation ? annotation.value().as<tvm::IntImmNode>() : nullptr;
    if (rank == nullptr || rank->value <= 0) { return luisa::nullopt; }
    ElementDomain result;
    auto loop = outer;
    for (auto i = int64_t{0}; i < rank->value; i++) {
        if (loop == nullptr || !unit_serial_loop(loop) ||
            (i != 0 && !loop->annotations.empty()) ||
            loop->min.as<tvm::IntImmNode>() == nullptr) {
            return luisa::nullopt;
        }
        auto extent = static_extent(loop->extent);
        if (!extent || (*extent != 0u &&
                        result.count > std::numeric_limits<uint64_t>::max() / *extent)) {
            return luisa::nullopt;
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
    luisa::optional<uint64_t> maximum_index_domain_elements;
};

[[nodiscard]] luisa::optional<StripedMaterialization>
match_striped_materialization(const tvm::tirx::ForNode *outer, bool allow_narrow_storage) {
    auto contract =
        outer->annotations.Get(materialized_pure_tile_annotation);
    auto version = contract ? contract.value().as<tvm::IntImmNode>() : nullptr;
    auto domain = element_domain(outer);
    if (version == nullptr || version->value != 1 || !domain ||
        outer->annotations.size() != 2u) {
        return luisa::nullopt;
    }
    auto store = domain->body.as<tvm::tirx::BufferStoreNode>();
    if (store == nullptr || store->predicate ||
        store->indices.size() != domain->axes.size()) {
        return luisa::nullopt;
    }
    auto buffer = store->buffer;
    auto supported_type = buffer->dtype == tvm::PrimType::Float(32) ||
                          (allow_narrow_storage &&
                           (buffer->dtype == tvm::PrimType::Float(16) ||
                            buffer->dtype == tvm::PrimType::BFloat(16)));
    auto offset = buffer->elem_offset.as<tvm::IntImmNode>();
    if (buffer.scope() != "local" || !supported_type ||
        store->value.ty() != buffer->dtype ||
        buffer->shape.size() != domain->axes.size() ||
        !buffer->strides.empty() || buffer->layout ||
        !buffer->allocated_addr.empty() || offset == nullptr ||
        offset->value != 0) {
        return luisa::nullopt;
    }
    auto equal = tvm::ffi::StructuralEqual{};
    auto elements = uint64_t{1u};
    for (auto i = size_t{0u}; i < domain->axes.size(); i++) {
        auto dimension = static_extent(buffer->shape[i], true);
        auto extent = static_extent(domain->axes[i]->extent, true);
        if (!dimension || !extent || *dimension != *extent ||
            !equal(store->indices[i], domain->axes[i]->loop_var) ||
            elements > std::numeric_limits<uint64_t>::max() / *dimension) {
            return luisa::nullopt;
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
    if (!pure || elements != domain->count) { return luisa::nullopt; }
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

[[nodiscard]] luisa::optional<ReductionMatch> match_reduction(
    const tvm::tirx::ForNode *loop) {
    auto contract = loop->annotations.Get(reduction_contract_annotation);
    auto kind = contract ? contract.value().as<tvm::IntImmNode>() : nullptr;
    auto minimum = loop->min.as<tvm::IntImmNode>();
    auto elements = static_extent(loop->extent, true);
    if (loop->annotations.size() != 2u || !permits_unordered_reduction(loop) || kind == nullptr ||
        (kind->value != reduction_add_contract &&
         kind->value != reduction_max_contract &&
         kind->value != reduction_min_contract) ||
        !unit_serial_loop(loop) || minimum == nullptr || minimum->value != 0 ||
        !elements || loop->loop_var.ty() != tvm::PrimType::Int(64)) {
        return luisa::nullopt;
    }

    tvm::ffi::Array<tvm::tirx::Stmt> statements;
    flatten_sequence(loop->body, statements);
    if (statements.size() != 3u) { return luisa::nullopt; }
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
        return luisa::nullopt;
    }
    auto forwarded = update_store->value.as<tvm::tirx::BufferLoadNode>();
    if (forwarded == nullptr || forwarded->predicate ||
        !forwarded->buffer.same_as(temporary_allocation->buffer) ||
        !zero_index(forwarded->indices)) {
        return luisa::nullopt;
    }

    tvm::PrimExpr lhs;
    tvm::PrimExpr rhs;
    if (kind->value == reduction_add_contract) {
        auto combine = combine_store->value.as<tvm::tirx::AddNode>();
        if (combine == nullptr) { return luisa::nullopt; }
        lhs = combine->a;
        rhs = combine->b;
    } else if (kind->value == reduction_max_contract) {
        auto combine = combine_store->value.as<tvm::tirx::MaxNode>();
        if (combine == nullptr) { return luisa::nullopt; }
        lhs = combine->a;
        rhs = combine->b;
    } else {
        auto combine = combine_store->value.as<tvm::tirx::MinNode>();
        if (combine == nullptr) { return luisa::nullopt; }
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
        return luisa::nullopt;
    }
    if (!pure_contribution(contribution, carry, temporary_allocation->buffer)) {
        return luisa::nullopt;
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
    luisa::optional<uint64_t> maximum_index_domain_elements{0u};
};

class StripedMaterializationAudit final
    : public tvm::tirx::StmtExprVisitor {
private:
    const ReductionAnalysis &_reductions;
    const luisa::unordered_map<BufferKey, StripedMaterialization> &_candidates;
    luisa::vector<const tvm::tirx::ForNode *> _domain;
    luisa::optional<tvm::PrimExpr> _owner;
    luisa::optional<uint64_t> _owner_elements;

    void _index_domain(StripedAccess &record) const noexcept {
        if (_owner_elements && record.maximum_index_domain_elements) {
            record.maximum_index_domain_elements = std::max(
                *record.maximum_index_domain_elements, *_owner_elements);
        } else {
            record.maximum_index_domain_elements.reset();
        }
    }

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
        auto previous_elements = _owner_elements;
        if (_reductions.reductions.contains(loop)) {
            _owner = loop->loop_var - loop->min;
            _owner_elements = _reductions.reductions.at(loop).elements;
        } else if (loop->annotations.count(independent_elements_annotation) &&
                   !_reductions.replicated_elements.contains(loop)) {
            if (auto element = element_domain(loop)) {
                tvm::PrimExpr linear = tvm::IntImm::Int64(0);
                for (auto axis : element->axes) {
                    linear = linear * axis->extent +
                             (axis->loop_var - axis->min);
                }
                _owner = std::move(linear);
                _owner_elements = element->count;
            } else {
                _owner.reset();
                _owner_elements.reset();
            }
        }
        StmtExprVisitor::VisitStmt_(loop);
        _owner = std::move(previous_owner);
        _owner_elements = previous_elements;
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
            _index_domain(record);
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
            _index_domain(record);
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
                         const ReductionAnalysis &reductions, bool allow_narrow_storage) {
    luisa::unordered_map<BufferKey, StripedMaterialization> candidates;
    luisa::unordered_set<BufferKey> duplicates;
    tvm::tirx::PostOrderVisit(body, [&](const tvm::ffi::ObjectRef &node) {
        auto loop = node.as<tvm::tirx::ForNode>();
        if (loop == nullptr ||
            !loop->annotations.count(materialized_pure_tile_annotation)) {
            return;
        }
        if (auto matched = match_striped_materialization(loop, allow_narrow_storage)) {
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
        } else {
            candidate.maximum_index_domain_elements =
                iter->second.maximum_index_domain_elements;
        }
    }
    for (auto key : rejected) { candidates.erase(key); }
    return candidates;
}

// This mirrors _stripe_loop, not a register-allocation model. For J full
// chunks and U > 1, factor=min(J,U), with floor(J/factor) serial packs;
// the remainder is explicitly unrolled. One pack simplifies to constant zero.
// U=1 uses the original serial loop, constant only for J <= 1. Therefore the
// least sufficient integer U is floor(J/2)+1, with 0/1 both requiring one.
// A literal partial final worker pack needs no further unrolling.
[[nodiscard]] luisa::optional<uint64_t> constant_striped_index_min_unroll(
    const luisa::unordered_map<BufferKey, StripedMaterialization> &materializations,
    uint64_t workers, uint64_t lane_elements) noexcept {
    if (workers == 0u || lane_elements == 0u ||
        workers > std::numeric_limits<uint64_t>::max() / lane_elements) {
        return luisa::nullopt;
    }
    auto stride = workers * lane_elements;
    auto required = uint64_t{1u};
    for (auto &&[key, materialization] : materializations) {
        static_cast<void>(key);
        if (!materialization.maximum_index_domain_elements) { return luisa::nullopt; }
        auto chunks = *materialization.maximum_index_domain_elements / stride;
        required = std::max(required, chunks / 2u + 1u);
    }
    return required;
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

// Packing cooperating programs introduces group fences between otherwise
// independent rows. Every program must execute the same fence sequence. For
// a packed tail, inactive programs replay the last valid coordinate and only
// suppress external stores: this keeps even data-dependent input addresses
// valid, without ever placing a group fence inside a program predicate.
// Replay is safe only when external writes cannot feed a subsequent read.
// Only unit enclosing loops are admitted here: reusing a partial allocation
// across iterations needs an additional read-before-next-write fence proof.
class PackedProgramAudit final : public tvm::tirx::StmtExprVisitor {
private:
    const ReductionAnalysis &_analysis;
    luisa::unordered_set<BufferKey> _reads;
    luisa::unordered_set<BufferKey> _writes;

protected:
    void VisitStmt_(const tvm::tirx::ForNode *loop) final {
        if (contains_reduction(loop->body, _analysis)) {
            uniform_fences &= loop->min.as<tvm::IntImmNode>() != nullptr &&
                              static_extent(loop->extent) == 1u &&
                              unit_serial_loop(loop);
        }
        StmtExprVisitor::VisitStmt_(loop);
    }

    void VisitExpr_(const tvm::tirx::BufferLoadNode *load) final {
        if (load->buffer.scope() != "local") { _reads.emplace(load->buffer.get()); }
        StmtExprVisitor::VisitExpr_(load);
    }

    void VisitStmt_(const tvm::tirx::BufferStoreNode *store) final {
        if (store->buffer.scope() != "local") { _writes.emplace(store->buffer.get()); }
        StmtExprVisitor::VisitStmt_(store);
    }

public:
    bool uniform_fences{true};
    explicit PackedProgramAudit(const ReductionAnalysis &analysis) noexcept
        : _analysis{analysis} {}

    [[nodiscard]] bool replayable_tail() const noexcept {
        return std::none_of(_writes.begin(), _writes.end(),
                            [&](auto buffer) noexcept { return _reads.contains(buffer); });
    }
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
    luisa::optional<tvm::PrimExpr> _owner;
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

// This private probe recognizes independent, compact element packs.
// Buffer metadata alone never authorizes alignment of a caller's final pointer.
// Private diagnostic facts. No policy consumes these values. Each record is
// one proved global access in one full, active warp/independent phase.
// Counts describe declared IR memory grouping, not emitted ISA or DRAM traffic.
// Only scalar storage types with existing CUDA vector representations.
// Global packs remain at most 16 bytes; compute/reduction types are unchanged.
[[nodiscard]] uint32_t vector_pack_storage_bytes(tvm::PrimType type) noexcept {
    if (type == tvm::PrimType::Float(16) || type == tvm::PrimType::BFloat(16)) { return 2u; }
    return type == tvm::PrimType::Float(32) ? 4u : 0u;
}

struct VectorPackMemoryAccess {
    tvm::tirx::BufferVar root;
    tvm::PrimExpr first;
    uint32_t storage_bytes;
    bool store;
    bool varying;
};

struct MemoryCountBounds {
    uint64_t lower{std::numeric_limits<uint64_t>::max()};
    uint64_t upper{0u};
    void include(uint64_t n) noexcept {
        lower = std::min(lower, n);
        upper = std::max(upper, n);
    }
};

struct VectorPackMemoryCounts {
    bool known{false};
    const char *reason{"unclassified"};
    uint64_t scalar_issues{0u};
    uint64_t vector_issues{0u};
    MemoryCountBounds scalar_root32, scalar_abi, vector_root32, vector_guard;
};

[[nodiscard]] uint32_t memory_mod32(int64_t n) noexcept {
    return static_cast<uint32_t>((n % 32 + 32) % 32);
}

// One memory operation, preserving duplicate sector requests across operations.
// No lane has more than 16 contiguous bytes in this first global-access subset.
[[nodiscard]] luisa::optional<uint64_t> memory_sector_count(
    const std::array<uint64_t, 32u> &lane_offsets, uint32_t origin,
    uint32_t bytes) noexcept {
    if (bytes == 0u || bytes > 16u) { return {}; }
    std::array<uint64_t, 64u> sectors{};
    auto count = size_t{0u};
    for (auto offset : lane_offsets) {
        if (offset > std::numeric_limits<uint64_t>::max() - origin - bytes) { return {}; }
        auto first = (offset + origin) / 32u;
        auto last = (offset + origin + bytes - 1u) / 32u;
        for (auto sector = first; sector <= last; sector++) {
            if (std::find(sectors.begin(), sectors.begin() + count, sector) == sectors.begin() + count) {
                if (count == sectors.size()) { return {}; }
                sectors[count++] = sector;
            }
        }
    }
    return static_cast<uint64_t>(count);
}

// Keep root residue and derived origin separate. The set is an overapproximation
// of actual program residues, not a claim that all extremes can occur together.
[[nodiscard]] uint32_t memory_origin_residues(const tvm::PrimExpr &first,
                                             uint32_t bytes,
                                             tvm::arith::Analyzer &analyzer) {
    auto modular = analyzer->modular_set(first);
    auto base = memory_mod32(modular->base) * bytes % 32u;
    auto stride = std::gcd(32u, memory_mod32(modular->coeff) * bytes % 32u);
    auto mask = uint32_t{0u};
    for (auto i = uint32_t{0u}; i < 32u / stride; i++) {
        mask |= uint32_t{1u} << ((base + i * stride) % 32u);
    }
    return mask;
}

[[nodiscard]] VectorPackMemoryCounts vector_pack_memory_counts(
    const VectorPackMemoryAccess &access, uint32_t width, uint64_t chunks,
    const tvm::tirx::ForNode *thread_domain,
    const tvm::tirx::PrimVar &chunk, const tvm::PrimExpr &lane,
    bool root_guarded, luisa::span<const tvm::tirx::ForNode *const> proof_domain) {
    auto result = VectorPackMemoryCounts{};
    result.reason = "unsupported-width-or-domain";
    if ((width != 2u && width != 4u && width != 8u) || chunks == 0u ||
        thread_domain == nullptr || !lane.as<tvm::tirx::VarNode>() ||
        access.storage_bytes == 0u || access.storage_bytes > 16u ||
        (access.storage_bytes & (access.storage_bytes - 1u)) != 0u ||
        (access.varying && access.storage_bytes * width > 16u)) { return result; }
    auto threads = static_extent(thread_domain->extent, true);
    auto thread_min = static_extent(thread_domain->min);
    if (!threads || !thread_min || *thread_min != 0u || *threads > 1024u ||
        *threads % 32u != 0u || chunks > std::numeric_limits<uint64_t>::max() / width) { return result; }
    result.scalar_issues = chunks * width;
    result.vector_issues = chunks;
    tvm::arith::Analyzer analyzer;
    // Count under the same element/worker/program domain that admitted this
    // access. This only simplifies diagnostic addresses, never runtime guards.
    result.reason = "unknown-proof-domain";
    for (auto loop : proof_domain) {
        if (loop == nullptr || !unit_serial_loop(loop)) { return result; }
        auto minimum = loop->min.as<tvm::IntImmNode>();
        auto extent = loop->extent.as<tvm::IntImmNode>();
        if (minimum == nullptr || extent == nullptr || extent->value <= 0 ||
            minimum->value > std::numeric_limits<int64_t>::max() - (extent->value - 1)) { return result; }
        analyzer->Bind(loop->loop_var, tvm::Range::FromMinExtent(loop->min, loop->extent));
    }
    auto lane_var = tvm::ffi::GetRef<tvm::tirx::Var>(lane.as<tvm::tirx::VarNode>());
    auto replace = [&](uint64_t thread, uint64_t lane_index) {
        return tvm::tirx::Substitute(access.first,
            tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>{
                {thread_domain->loop_var, tvm::IntImm::Int64(static_cast<int64_t>(thread))},
                {lane_var, tvm::IntImm::Int64(static_cast<int64_t>(lane_index))}});
    };
    auto chunk_at = [&](const tvm::PrimExpr &value, int64_t index) {
        return tvm::tirx::Substitute(value,
            tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>{{chunk, tvm::IntImm::Int64(index)}});
    };
    auto residue_cases = uint64_t{0u};
    for (auto warp = uint64_t{0u}; warp < *threads / 32u; warp++) {
        auto first = analyzer->Simplify(replace(warp * 32u, 0u));
        auto base = analyzer->Simplify(chunk_at(first, 0));
        auto step = analyzer->Simplify(chunk_at(first, 1) - base);
        auto coefficient = step.as<tvm::IntImmNode>();
        auto affine = analyzer->Simplify(first - base - chunk * step);
        auto zero = affine.as<tvm::IntImmNode>();
        // Full chunks must repeat the same sector residue. Otherwise defer to
        // a later periodic counter, rather than multiplying an arbitrary chunk.
        result.reason = "unknown-chunk-affine-or-residue";
        if (coefficient == nullptr || zero == nullptr || zero->value != 0 ||
            memory_mod32(coefficient->value) * access.storage_bytes % 32u != 0u) {
            return result;
        }
        std::array<uint64_t, 32u> offsets{};
        for (auto i = uint64_t{0u}; i < 32u; i++) {
            auto difference = analyzer->Simplify(replace(warp * 32u + i, i) - first);
            auto delta = difference.as<tvm::IntImmNode>();
            result.reason = "unknown-or-negative-lane-delta";
            if (delta == nullptr || delta->value < 0 ||
                static_cast<uint64_t>(delta->value) > std::numeric_limits<uint64_t>::max() / access.storage_bytes) { return result; }
            offsets[i] = static_cast<uint64_t>(delta->value) * access.storage_bytes;
        }
        auto derived = memory_origin_residues(base, access.storage_bytes, analyzer);
        auto vector_bytes = access.storage_bytes * (access.varying ? width : 1u);
        auto guard_alignment = access.storage_bytes * (root_guarded ? width : 1u);
        for (auto root_residue = uint32_t{0u}; root_residue < 32u; root_residue += access.storage_bytes) {
            for (auto origin = uint32_t{0u}; origin < 32u; origin++) {
                if ((derived & (uint32_t{1u} << origin)) == 0u) { continue; }
                result.reason = "residue-enumeration-budget";
                if (++residue_cases > 4096u) { return result; }
                auto scalar_sectors = uint64_t{0u};
                for (auto element = uint32_t{0u}; element < width; element++) {
                    auto at = root_residue + origin + (access.varying ? element * access.storage_bytes : 0u);
                    auto count = memory_sector_count(offsets, at, access.storage_bytes);
                    result.reason = "sector-count-overflow";
                    if (!count || scalar_sectors > std::numeric_limits<uint64_t>::max() - *count) { return result; }
                    scalar_sectors += *count;
                }
                auto vector_sectors = memory_sector_count(offsets, root_residue + origin, vector_bytes);
                if (!vector_sectors || scalar_sectors > std::numeric_limits<uint64_t>::max() / chunks ||
                    *vector_sectors > std::numeric_limits<uint64_t>::max() / chunks) { return result; }
                scalar_sectors *= chunks;
                auto vector_total = *vector_sectors * chunks;
                result.scalar_abi.include(scalar_sectors);
                if (root_residue == 0u) {
                    result.scalar_root32.include(scalar_sectors);
                    result.vector_root32.include(vector_total);
                }
                if (root_residue % guard_alignment == 0u) {
                    result.vector_guard.include(vector_total);
                }
            }
        }
    }
    result.reason = "known-ir-grouping-envelope";
    result.known = true;
    return result;
}

struct VectorPackAccessMemoryFacts {
    tvm::tirx::BufferVar root;
    tvm::PrimExpr first;
    uint32_t storage_bytes;
    bool store;
    bool varying;
    bool root_guarded;
    VectorPackMemoryCounts counts;
};

struct VectorPackPhaseMemoryFacts {
    uint64_t phase;
    uint64_t elements;
    uint64_t workers;
    uint32_t width;
    const char *scope;
    tvm::ffi::Optional<tvm::PrimExpr> actual_guard;
    luisa::vector<VectorPackAccessMemoryFacts> accesses;
    const char *kind{"independent"};
};

[[nodiscard]] VectorPackPhaseMemoryFacts collect_vector_pack_memory(
    luisa::span<const VectorPackMemoryAccess> accesses,
    const tvm::ffi::Array<tvm::tirx::BufferVar> &guarded_roots,
    uint64_t phase, uint64_t elements, uint64_t workers, uint32_t width,
    const tvm::tirx::ForNode *thread_domain,
    const tvm::tirx::PrimVar &chunk, const tvm::PrimExpr &lane,
    bool unconditional, const tvm::PrimExpr &actual_guard,
    luisa::span<const tvm::tirx::ForNode *const> proof_domain) {
    auto result = VectorPackPhaseMemoryFacts{
        phase, elements, workers, width, "predicate-or-partial-domain", actual_guard, {}};
    auto complete_domain = unconditional && width != 0u && workers != 0u &&
                           workers <= std::numeric_limits<uint64_t>::max() / width &&
                           elements != 0u && elements % (workers * width) == 0u;
    if (complete_domain) { result.scope = "one-full-active-warp-phase"; }
    // Preserve actual access identities even when participation is unknown.
    // The default counters are invalid unless known is true.
    for (auto &&access : accesses) {
        auto guarded = std::any_of(guarded_roots.begin(), guarded_roots.end(),
            [&](const auto &root) { return root.same_as(access.root); });
        auto counts = VectorPackMemoryCounts{};
        counts.reason = "predicate-or-partial-domain";
        if (complete_domain) {
            auto chunks = elements / (workers * width);
            counts = vector_pack_memory_counts(access, width, chunks, thread_domain,
                                               chunk, lane, guarded, proof_domain);
        }
        result.accesses.emplace_back(VectorPackAccessMemoryFacts{
            access.root, access.first, access.storage_bytes, access.store,
            access.varying, guarded, std::move(counts)});
    }
    return result;
}

void report_prepared_memory(luisa::span<const VectorPackPhaseMemoryFacts> phases) {
    for (auto &&phase : phases) {
        LUISA_INFO("Private CUDA prepared memory scope: phase={} kind={}; not whole-kernel coverage.", phase.phase, phase.kind);
        if (phase.accesses.empty()) {
            LUISA_INFO("Private CUDA prepared memory: phase={} known=false reason={}; no zero substitution.", phase.phase, phase.scope);
            continue;
        }
        for (auto i = size_t{0u}; i < phase.accesses.size(); i++) {
            auto &&access = phase.accesses[i];
            auto &&counts = access.counts;
            auto root_name = std::string{access.root.name().data(), access.root.name().size()};
            if (!counts.known) {
                LUISA_INFO("Private CUDA prepared memory: phase={} access={} root={} write={} known=false reason={}; no zero substitution.",
                           phase.phase, i, root_name, access.store, counts.reason);
                continue;
            }
            LUISA_INFO("Private CUDA prepared memory: phase={} access={} root={} write={} basis=IR-grouping unit={} pack={} scalar-ops={} vector-ops={} scalar-sectors-root32=[{},{}] scalar-sectors-ABI=[{},{}] vector-sectors-root32=[{},{}] vector-sectors-guard=[{},{}]; not-ISA-not-DRAM.",
                       phase.phase, i, root_name, access.store, phase.scope, phase.width,
                       counts.scalar_issues, counts.vector_issues,
                       counts.scalar_root32.lower, counts.scalar_root32.upper,
                       counts.scalar_abi.lower, counts.scalar_abi.upper,
                       counts.vector_root32.lower, counts.vector_root32.upper,
                       counts.vector_guard.lower, counts.vector_guard.upper);
        }
    }
}


class VectorPackPhaseAudit final : public tvm::tirx::StmtExprVisitor {
private:
    const tvm::tirx::PrimVar &_element;
    uint32_t _width;
    const luisa::unordered_set<BufferKey> &_allocated;
    luisa::span<const tvm::tirx::ForNode *const> _domain;
    tvm::ffi::Array<tvm::tirx::BufferVar> _roots;
    luisa::vector<VectorPackMemoryAccess> _memory_accesses;
    bool _memory_unconditional{true};

    void _record_memory(const tvm::tirx::BufferVar &buffer,
                        const tvm::PrimExpr &first, bool store, bool varying) {
        if (buffer.scope() == "global" || buffer.scope().empty()) {
            // Access occurrences are deliberately not claimed as final CSE or
            // machine instruction counts. Do not aggregate different roots.
            auto bits = buffer->dtype.bits();
            auto bytes = !buffer->dtype.IsScalableVector() && buffer->dtype.lanes() == 1 && bits > 0 && bits % 8 == 0 ?
                             static_cast<uint32_t>(bits / 8) : 0u;
            _memory_accesses.push_back({buffer, first, bytes, store, varying});
        }
    }

    [[nodiscard]] bool _depends(const tvm::Expr &expression) const {
        auto result = false;
        tvm::tirx::PostOrderVisit(expression, [&](const tvm::ffi::ObjectRef &node) {
            result |= node.get() == _element.get();
        });
        return result;
    }

    void _access(const tvm::tirx::BufferVar &buffer,
                 const tvm::ffi::Array<tvm::PrimExpr> &indices, bool store) {
        if (!valid) { return; }
        auto offset = buffer->elem_offset.as<tvm::IntImmNode>();
        if (indices.size() != buffer->shape.size() || indices.empty() ||
            !buffer->strides.empty() || buffer->layout || !buffer->allocated_addr.empty() ||
            offset == nullptr || offset->value != 0 ||
            (buffer.scope() != "local" && buffer.scope() != "global" && !buffer.scope().empty())) {
            valid = false;
            return;
        }
        auto volume = uint64_t{1u};
        tvm::PrimExpr linear = tvm::IntImm::Int64(0);
        for (auto i = size_t{0u}; i < indices.size(); i++) {
            auto extent = static_extent(buffer->shape[i], true);
            if (!extent || volume > static_cast<uint64_t>(INT64_MAX) / *extent ||
                indices[i].ty() != tvm::PrimType::Int(64)) {
                valid = false;
                return;
            }
            volume *= *extent;
            linear = linear * buffer->shape[i] + indices[i];
        }
        auto at = [&](int64_t i) {
            return tvm::tirx::Substitute(linear,
                tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>{{_element, tvm::IntImm::Int64(i)}});
        };
        auto first = at(0);
        auto last = at(static_cast<int64_t>(_width - 1u));
        // Prove every lane, not just the first step of a possibly nonlinear index.
        // Only a read may broadcast one scalar across the independent pack.
        if (prove_in_loop_domain(tvm::equal(linear, first), _domain)) {
            valid &= !store;
            if (valid) { _record_memory(buffer, first, store, false); }
            return;
        }
        if (!prove_in_loop_domain(tvm::equal(linear, first + _element), _domain) ||
            !prove_in_loop_domain(tvm::equal(tvm::floormod(first, tvm::IntImm::Int64(_width)),
                                             tvm::IntImm::Int64(0)), _domain) ||
            !prove_in_loop_domain(first >= tvm::IntImm::Int64(0) &&
                                     last < tvm::IntImm::Int64(static_cast<int64_t>(volume)), _domain)) {
            valid = false;
            return;
        }
        auto storage_bytes = vector_pack_storage_bytes(buffer->dtype);
        if (buffer.scope() == "local") {
            // These are real owned allocations, not an external alignment hint.
            // BF16ComputeLegalize may promote a private BF16 allocation to F32.
            // Prove that wider allocation's alignment too, without changing it.
            auto local_bytes = buffer->dtype == tvm::PrimType::BFloat(16) ? 4u : storage_bytes;
            auto bytes = static_cast<int32_t>(local_bytes * _width);
            valid &= bytes != 0 && _allocated.contains(buffer.get()) &&
                     buffer->data_alignment >= bytes && buffer->data_alignment % bytes == 0;
            return;
        }
        if ((buffer.scope() != "global" && !buffer.scope().empty()) ||
            _allocated.contains(buffer.get()) || storage_bytes == 0u ||
            storage_bytes * _width > 16u) {
            valid = false;
            return;
        }
        if (std::none_of(_roots.begin(), _roots.end(), [&](const auto &root) {
                return root.same_as(buffer);
            })) {
            _roots.push_back(buffer);
        }
        _record_memory(buffer, first, store, true);
    }

protected:
    void VisitStmt(const tvm::tirx::Stmt &statement) final {
        if (!statement.as<tvm::tirx::SeqStmtNode>() &&
            !statement.as<tvm::tirx::IfThenElseNode>() &&
            !statement.as<tvm::tirx::BufferStoreNode>()) {
            valid = false;
            return;
        }
        StmtExprVisitor::VisitStmt(statement);
    }

    void VisitStmt_(const tvm::tirx::IfThenElseNode *branch) final {
        // Keep lazy, per-element bounds/masks scalar. Whole-row predicates stay.
        valid &= !_depends(branch->condition) || prove_in_loop_domain(branch->condition, _domain);
        // Keep all statements and guards unchanged. A nontrivial branch makes
        // this initial flat occurrence counter unknown, even when warp-uniform.
        auto condition = branch->condition.as<tvm::IntImmNode>();
        _memory_unconditional &= condition != nullptr && condition->value != 0 && !branch->else_case;
        StmtExprVisitor::VisitStmt_(branch);
    }

    void VisitStmt_(const tvm::tirx::BufferStoreNode *store) final {
        valid &= !store->predicate;
        _access(store->buffer, store->indices, true);
        StmtExprVisitor::VisitStmt_(store);
    }

    void VisitExpr_(const tvm::tirx::BufferLoadNode *load) final {
        valid &= !load->predicate;
        _access(load->buffer, load->indices, false);
        StmtExprVisitor::VisitExpr_(load);
    }

    void VisitExpr_(const tvm::CallNode *call) final {
        static auto vectorizable = tvm::Op::GetAttrMap<tvm::tirx::TVectorizable>("TVectorizable");
        if (call->op.same_as(tvm::tirx::builtin::if_then_else())) {
            // Both arms are visited by this audit. Leave demand unknown rather
            // than counting two mutually exclusive memory paths as one.
            _memory_unconditional = false;
        }
        auto op = call->op.as<tvm::Op>();
        if (_depends(tvm::ffi::GetRef<tvm::Call>(call))) {
            valid &= op && vectorizable.count(op.value()) != 0u && vectorizable[op.value()];
            if (call->op.same_as(tvm::tirx::builtin::if_then_else())) {
                valid &= !_depends(call->args[0u]) ||
                         prove_in_loop_domain(call->args[0u].as_or_throw<tvm::PrimExpr>(), _domain);
            }
        }
        StmtExprVisitor::VisitExpr_(call);
    }

public:
    bool valid{true};
    VectorPackPhaseAudit(const tvm::tirx::PrimVar &element, uint32_t width,
                        const luisa::unordered_set<BufferKey> &allocated,
                        luisa::span<const tvm::tirx::ForNode *const> domain) noexcept
        : _element{element}, _width{width}, _allocated{allocated}, _domain{domain} {}

    [[nodiscard]] VectorPackPhaseMemoryFacts memory_phase(
        uint64_t phase, uint64_t elements, uint64_t workers,
        const tvm::tirx::ForNode *thread_domain, const tvm::tirx::PrimVar &chunk,
        const tvm::PrimExpr &lane, const tvm::PrimExpr &actual_guard) const {
        return collect_vector_pack_memory(_memory_accesses, _roots, phase, elements,
                                          workers, _width, thread_domain, chunk, lane,
                                          _memory_unconditional, actual_guard, _domain);
    }

    [[nodiscard]] tvm::ffi::Optional<tvm::PrimExpr> guard() const {
        if (!valid || _roots.empty()) { return {}; }
        tvm::ffi::Optional<tvm::PrimExpr> condition;
        for (auto &&root : _roots) {
            tvm::ffi::Array<tvm::PrimExpr> indices;
            for (auto &&extent : root->shape) {
                static_cast<void>(extent);
                indices.push_back(tvm::IntImm::Int64(0));
            }
            tvm::Expr pointer = tvm::Call{root.DataPointerType(), tvm::tirx::builtin::address_of(),
                                        {tvm::tirx::BufferLoad{root, std::move(indices)}}};
            tvm::Type integer_type = tvm::PrimType::UInt(64);
            auto address = tvm::reinterpret(std::move(integer_type), std::move(pointer))
                               .as_or_throw<tvm::PrimExpr>();
            auto bytes = vector_pack_storage_bytes(root->dtype) * _width;
            auto aligned = tvm::equal(tvm::bitwise_and(address, tvm::IntImm{tvm::PrimType::UInt(64), bytes - 1u}),
                                      tvm::IntImm{tvm::PrimType::UInt(64), 0});
            condition = condition ? condition.value() && aligned : aligned;
        }
        return condition;
    }
};


// Only specialize coordinate predicates in a proved full contribution chunk.
// This does not simplify floating arithmetic or assume anything about data/NaNs.
class FullChunkPredicateSpecializer final : public tvm::tirx::StmtExprMutator {
private:
    luisa::span<const tvm::tirx::ForNode *const> _domain;

    [[nodiscard]] luisa::optional<bool> _truth(const tvm::PrimExpr &condition) const {
        auto coordinate = condition.ty() == tvm::PrimType::Bool();
        tvm::tirx::PostOrderVisit(condition, [&](const tvm::ffi::ObjectRef &object) {
            auto primitive = object.as<tvm::PrimExpr>();
            if (!primitive || !primitive.value().ty().IsScalar() ||
                !primitive.value().ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt, DLDataTypeCode::kDLBool)) {
                coordinate = false;
            } else if (auto variable = object.as<tvm::tirx::VarNode>()) {
                coordinate &= std::any_of(_domain.begin(), _domain.end(), [&](auto loop) {
                    return loop->loop_var.get() == variable;
                });
            } else if (auto division = object.as<tvm::tirx::FloorDivNode>()) {
                auto divisor = division->b.as<tvm::IntImmNode>();
                coordinate &= divisor && divisor->value > 0;
            } else if (auto modulo = object.as<tvm::tirx::FloorModNode>()) {
                auto divisor = modulo->b.as<tvm::IntImmNode>();
                coordinate &= divisor && divisor->value > 0;
            } else {
                coordinate &= object.as<tvm::IntImmNode>() || object.as<tvm::tirx::CastNode>() ||
                              object.as<tvm::tirx::AddNode>() || object.as<tvm::tirx::SubNode>() ||
                              object.as<tvm::tirx::MulNode>() || object.as<tvm::tirx::MinNode>() ||
                              object.as<tvm::tirx::MaxNode>() || object.as<tvm::tirx::LTNode>() ||
                              object.as<tvm::tirx::LENode>() || object.as<tvm::tirx::GTNode>() ||
                              object.as<tvm::tirx::GENode>() || object.as<tvm::tirx::EQNode>() ||
                              object.as<tvm::tirx::NENode>() || object.as<tvm::tirx::AndNode>() ||
                              object.as<tvm::tirx::OrNode>() || object.as<tvm::tirx::NotNode>();
            }
        });
        if (coordinate) {
            if (prove_in_loop_domain(condition, _domain)) { return true; }
            if (prove_in_loop_domain(!condition, _domain)) { return false; }
        }
        return luisa::nullopt;
    }

    [[nodiscard]] tvm::Expr VisitExpr_(const tvm::CallNode *call) final {
        if (call->op.same_as(tvm::tirx::builtin::if_then_else()) && call->args.size() == 3u) {
            if (auto truth = _truth(call->args[0u].as_or_throw<tvm::PrimExpr>())) {
                // Recurse into the selected lazy arm only; never evaluate the other.
                return VisitExpr(call->args[*truth ? 1u : 2u]);
            }
        }
        return StmtExprMutator::VisitExpr_(call);
    }

    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::IfThenElseNode *branch) final {
        if (auto truth = _truth(branch->condition)) {
            if (*truth) { return VisitStmt(branch->then_case); }
            return branch->else_case ? VisitStmt(branch->else_case.value()) :
                                       tvm::tirx::Evaluate{tvm::IntImm::Int32(0)};
        }
        return StmtExprMutator::VisitStmt_(branch);
    }

public:
    explicit FullChunkPredicateSpecializer(luisa::span<const tvm::tirx::ForNode *const> domain) noexcept
        : _domain{domain} {}
};

// Private terminal-suffix proof. No output-name/shape or operation-name matching.
// The marker is the actual second collective store constructed by the mapper.
class TerminalRowSuffix {
private:
    using Stmt = tvm::tirx::Stmt;
    using Array = tvm::ffi::Array<Stmt>;
    const tvm::tirx::BufferStoreNode *_marker;
    tvm::tirx::BufferVar _carry;
    tvm::PrimExpr _worker;
    tvm::arith::Analyzer _analyzer;
    luisa::unordered_set<const tvm::tirx::VarNode *> _coordinates;
    luisa::unordered_map<BufferKey, uint64_t> _allocations, _reads, _writes;
    luisa::unordered_map<BufferKey, uint64_t> _suffix_reads;
    luisa::unordered_set<BufferKey> _available;
    luisa::unordered_map<BufferKey, tvm::PrimExpr> _ready;
    tvm::PrimExpr _path{tvm::IntImm{tvm::PrimType::Bool(), 1}};
    uint64_t _outputs{0u};
    bool _valid{true};

    [[nodiscard]] bool _contains(const Stmt &body) const {
        auto found = false;
        tvm::tirx::PostOrderVisit(body, [&](const tvm::ffi::ObjectRef &node) {
            found |= node.get() == _marker;
        });
        return found;
    }

    // Only the path to the marker is opened. A unit loop executes exactly once;
    // substitute its bound variable in BOTH pieces before lifting its local cells.
    [[nodiscard]] bool _split(const Stmt &body, Array &before, Array &after) const {
        if (body.get() == _marker) { after.push_back(body); return true; }
        if (auto sequence = body.as<tvm::tirx::SeqStmtNode>()) {
            auto found = false;
            for (auto &&statement : sequence->seq) {
                if (found) { flatten_sequence(statement, after); }
                else if (_contains(statement)) {
                    if (!_split(statement, before, after)) { return false; }
                    found = true;
                } else { flatten_sequence(statement, before); }
            }
            return found;
        }
        if (auto loop = body.as<tvm::tirx::ForNode>(); loop && unit_serial_loop(loop) &&
            static_extent(loop->extent) == 1u && loop->min.as<tvm::IntImmNode>() && loop->annotations.empty()) {
            Array prefix, suffix;
            if (!_split(loop->body, prefix, suffix)) { return false; }
            auto append = [&](const Array &source, Array &destination) {
                for (auto &&statement : source) {
                    auto replaced = tvm::tirx::Substitute(statement,
                        tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>{{loop->loop_var, loop->min}});
                    flatten_sequence(replaced, destination);
                }
            };
            append(prefix, before);
            append(suffix, after);
            return true;
        }
        return false; // Never move a collective through a predicate/repeated loop.
    }

    [[nodiscard]] bool _implied(const tvm::PrimExpr &condition) {
        // Canonicalize both sides under the same physical-thread bounds before
        // adding the path constraint. A single packed program can simplify
        // worker=thread%workers to thread; retaining only the former constraint
        // would fail to prove a later canonicalized singleton-cell index.
        auto path = _analyzer->Simplify(_path);
        auto canonical = _analyzer->Simplify(condition);
        tvm::With<tvm::arith::ConstraintContext> context{_analyzer, path};
        return _analyzer->CanProve(canonical);
    }

    [[nodiscard]] bool _cell(const tvm::tirx::BufferVar &buffer,
                             const tvm::ffi::Array<tvm::PrimExpr> &indices) {
        auto offset = buffer->elem_offset.as<tvm::IntImmNode>();
        if (buffer.scope() != "local" || !buffer->dtype.IsScalar() || buffer->layout ||
            !buffer->strides.empty() || !buffer->allocated_addr.empty() || !offset || offset->value != 0 ||
            buffer->shape.size() != indices.size() || _allocations[buffer.get()] != 1u) { return false; }
        for (size_t i = 0u; i < indices.size(); i++) {
            bool tainted = false;
            if (!_expression(indices[i], true, tainted)) { return false; }
            if (static_extent(buffer->shape[i]) != 1u ||
                !_implied(tvm::equal(indices[i], tvm::IntImm::Int64(0)))) { return false; }
        }
        return !indices.empty();
    }

    // Strict scalar expression whitelist. No pointer expressions, unknown calls,
    // real-memory reads, shared cells, or tainted address/predicate reads.
    [[nodiscard]] bool _expression(const tvm::PrimExpr &expression, bool coordinate, bool &tainted) {
        auto valid = true;
        class Visit final : public tvm::tirx::ExprVisitor {
        private:
            TerminalRowSuffix &_owner;
            bool _coordinate;
            bool &_tainted;
            void VisitExpr_(const tvm::tirx::BufferLoadNode *load) final {
                auto found = _owner._ready.find(load->buffer.get());
                if (_coordinate || load->predicate || !_owner._cell(load->buffer, load->indices) ||
                    found == _owner._ready.end() || !_owner._implied(found->second)) {
                    valid = false;
                    return;
                }
                _owner._suffix_reads[load->buffer.get()]++;
                _tainted = true;
            }
            void VisitExpr_(const tvm::tirx::VarNode *variable) final {
                valid &= _owner._coordinates.contains(variable);
            }
            void VisitExpr_(const tvm::CallNode *call) final {
                if (_coordinate) { valid = false; return; }
                auto known = call->op.same_as(tvm::tirx::builtin::if_then_else()) ||
                             call->op.same_as(tvm::tirx::builtin::reinterpret());
                // BF16 RNE uses only total scalar UInt32 operations. Do not
                // infer purity from an arbitrary intrinsic or admit shifts
                // with unknown/out-of-range counts. Recurse into both operands
                // so metadata, address and non-dominating reads stay rejected.
                auto uint32 = [](const tvm::Expr &value) noexcept {
                    auto type = value->ty.as<tvm::PrimType>();
                    return type && type.value() == tvm::PrimType::UInt(32);
                };
                auto result_type = call->ty.as<tvm::PrimType>();
                if (!known && result_type && result_type.value() == tvm::PrimType::UInt(32) &&
                    call->args.size() == 2u && uint32(call->args[0]) && uint32(call->args[1])) {
                    known = call->op.same_as(tvm::tirx::builtin::bitwise_and()) ||
                            call->op.same_as(tvm::tirx::builtin::bitwise_or());
                    if (call->op.same_as(tvm::tirx::builtin::shift_right())) {
                        auto count = call->args[1].as<tvm::IntImmNode>();
                        known = count && count->value >= 0 && count->value < 32;
                    }
                }
                if (!known) { valid = false; return; }
                ExprVisitor::VisitExpr_(call);
            }
            void VisitExpr_(const tvm::tirx::ProducerLoadNode *) final { valid = false; }
        public:
            bool valid{true};
            Visit(TerminalRowSuffix &owner, bool coordinate, bool &tainted) noexcept
                : _owner{owner}, _coordinate{coordinate}, _tainted{tainted} {}
            void VisitExpr(const tvm::Expr &expr) final {
                auto primitive = expr.as<tvm::PrimExpr>();
                if (!primitive || !primitive.value().ty().IsScalar() ||
                    (_coordinate && !primitive.value().ty().MatchesCode(DLDataTypeCode::kDLInt,
                        DLDataTypeCode::kDLUInt, DLDataTypeCode::kDLBool))) { valid = false; return; }
                // Allow the ordinary scalar arithmetic node family, but never
                // let an unclassified expression (Let, address, shuffle) through.
                auto ordinary = expr.as<tvm::IntImmNode>() || expr.as<tvm::FloatImmNode>() ||
                    expr.as<tvm::tirx::VarNode>() || expr.as<tvm::tirx::BufferLoadNode>() ||
                    expr.as<tvm::tirx::CastNode>() || expr.as<tvm::tirx::AddNode>() || expr.as<tvm::tirx::SubNode>() ||
                    expr.as<tvm::tirx::MulNode>() || expr.as<tvm::tirx::DivNode>() || expr.as<tvm::tirx::FloorDivNode>() ||
                    expr.as<tvm::tirx::FloorModNode>() || expr.as<tvm::tirx::MinNode>() || expr.as<tvm::tirx::MaxNode>() ||
                    expr.as<tvm::tirx::LTNode>() || expr.as<tvm::tirx::LENode>() || expr.as<tvm::tirx::GTNode>() ||
                    expr.as<tvm::tirx::GENode>() || expr.as<tvm::tirx::EQNode>() || expr.as<tvm::tirx::NENode>() ||
                    expr.as<tvm::tirx::AndNode>() || expr.as<tvm::tirx::OrNode>() || expr.as<tvm::tirx::NotNode>() ||
                    expr.as<tvm::CallNode>();
                if (!ordinary) { valid = false; return; }
                ExprVisitor::VisitExpr(expr);
            }
        } visitor{*this, coordinate, tainted};
        visitor(expression);
        valid &= visitor.valid;
        return valid;
    }

    void _visit(const Stmt &body) {
        if (!_valid) { return; }
        if (auto sequence = body.as<tvm::tirx::SeqStmtNode>()) {
            for (auto &&child : sequence->seq) { _visit(child); }
        } else if (auto allocation = body.as<tvm::tirx::AllocBufferNode>()) {
            tvm::ffi::Array<tvm::PrimExpr> zeros;
            for (size_t i = 0; i < allocation->buffer->shape.size(); i++) { zeros.push_back(tvm::IntImm::Int64(0)); }
            auto unconditional = _path.as<tvm::IntImmNode>();
            _valid &= unconditional && unconditional->value != 0 && allocation->annotations.empty() && _cell(allocation->buffer, zeros);
            _available.emplace(allocation->buffer.get());
        } else if (auto loop = body.as<tvm::tirx::ForNode>()) {
            if (!unit_serial_loop(loop) || static_extent(loop->extent) != 1u ||
                !loop->min.as<tvm::IntImmNode>() || !loop->annotations.empty()) { _valid = false; return; }
            _visit(tvm::tirx::Substitute(loop->body,
                tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>{{loop->loop_var, loop->min}}));
        } else if (auto branch = body.as<tvm::tirx::IfThenElseNode>()) {
            bool tainted = false;
            if (branch->else_case || !_expression(branch->condition, true, tainted)) { _valid = false; return; }
            auto saved = _path;
            _path = _path && branch->condition;
            _visit(branch->then_case);
            _path = std::move(saved);
        } else if (auto store = body.as<tvm::tirx::BufferStoreNode>()) {
            bool tainted = false;
            if (store->predicate || !_expression(store->value, false, tainted) || !tainted) { _valid = false; return; }
            if (store->buffer.scope() == "local") {
                if (!_available.contains(store->buffer.get()) || !_cell(store->buffer, store->indices) || _ready.contains(store->buffer.get()) ||
                    _writes[store->buffer.get()] != 1u) { _valid = false; return; }
                _ready.emplace(store->buffer.get(), _path);
            } else {
                auto offset = store->buffer->elem_offset.as<tvm::IntImmNode>();
                auto simple_root = store->buffer.scope() == "global" && _allocations[store->buffer.get()] == 0u &&
                    store->buffer->dtype.IsScalar() && !store->buffer->layout && store->buffer->strides.empty() &&
                    store->buffer->allocated_addr.empty() && offset && offset->value == 0;
                for (auto &&extent : store->buffer->shape) { simple_root &= static_extent(extent, true).has_value(); }
                if (!simple_root || ++_outputs != 1u ||
                    !_implied(tvm::equal(_worker, tvm::IntImm::Int64(0)))) { _valid = false; return; }
                for (auto &&index : store->indices) {
                    bool index_tainted = false;
                    _valid &= _expression(index, true, index_tainted);
                }
            }
        } else if (auto evaluation = body.as<tvm::tirx::EvaluateNode>()) {
            auto zero = evaluation->value.as<tvm::IntImmNode>();
            _valid &= zero && zero->value == 0;
        } else { _valid = false; }
    }

public:
    TerminalRowSuffix(const tvm::tirx::BufferStoreNode *marker, tvm::PrimExpr worker,
                      const tvm::tirx::PrimVar &thread, uint64_t threads,
                      const tvm::tirx::PrimVar &lane, const tvm::tirx::PrimVar &block, const tvm::tirx::ForNode *program)
        : _marker{marker}, _carry{marker->buffer}, _worker{std::move(worker)} {
        _coordinates.emplace(thread.get());
        _coordinates.emplace(lane.get());
        _coordinates.emplace(block.get());
        _coordinates.emplace(program->loop_var.get());
        _analyzer->Bind(thread, tvm::Range::FromMinExtent(tvm::IntImm::Int64(0), tvm::IntImm::Int64(static_cast<int64_t>(threads))));
        _analyzer->Bind(lane, tvm::Range::FromMinExtent(tvm::IntImm::Int64(0), tvm::IntImm::Int64(32)));
    }

    [[nodiscard]] tvm::ffi::Optional<Stmt> rewrite(const Stmt &body, const tvm::PrimExpr &subgroup) {
        Array prefix, suffix;
        if (!_split(body, prefix, suffix) || suffix.empty()) { return {}; }
        // Count occurrences, not unique DAG nodes: an identical BufferLoad may
        // occur before and after the marker and must not conceal an early use.
        class Census final : public tvm::tirx::StmtExprVisitor {
        private:
            TerminalRowSuffix &_owner;
            void VisitStmt_(const tvm::tirx::AllocBufferNode *alloc) final {
                _owner._allocations[alloc->buffer.get()]++;
                metadata_plain &= alloc->annotations.empty();
                buffers.emplace(alloc->buffer.get(), alloc->buffer);
            }
            void VisitExpr_(const tvm::tirx::BufferLoadNode *load) final {
                _owner._reads[load->buffer.get()]++;
                buffers.emplace(load->buffer.get(), load->buffer);
                StmtExprVisitor::VisitExpr_(load);
            }
            void VisitStmt_(const tvm::tirx::BufferStoreNode *store) final {
                _owner._writes[store->buffer.get()]++;
                buffers.emplace(store->buffer.get(), store->buffer);
                external_stores += store->buffer.scope() == "global";
                StmtExprVisitor::VisitStmt_(store);
            }
        public:
            uint64_t external_stores{0u};
            bool metadata_plain{true};
            luisa::unordered_map<BufferKey, tvm::tirx::BufferVar> buffers;
            explicit Census(TerminalRowSuffix &owner) noexcept : _owner{owner} {}
        } census{*this};
        census(body);
        // Types/annotations are not ordinary expression children. Audit them
        // explicitly rather than allowing a hidden carry read or local alias.
        if (!census.metadata_plain) { return {}; }
        for (auto &&[key, buffer] : census.buffers) {
            if (buffer->layout || !buffer->allocated_addr.empty()) { return {}; }
            bool tainted = false;
            if (!_expression(buffer->elem_offset, true, tainted)) { return {}; }
            for (auto &&extent : buffer->shape) { if (!_expression(extent, true, tainted)) { return {}; } }
            for (auto &&stride : buffer->strides) { if (!_expression(stride, true, tainted)) { return {}; } }
        }
        for (auto &&statement : prefix) {
            if (auto allocation = statement.as<tvm::tirx::AllocBufferNode>()) {
                _available.emplace(allocation->buffer.get());
            }
        }
        if (census.external_stores != 1u || !_available.contains(_carry.get()) ||
            !_cell(_carry, {tvm::IntImm::Int64(0)})) { return {}; }
        _ready.emplace(_carry.get(), tvm::IntImm{tvm::PrimType::Bool(), 1});
        // The first item is the mapper-owned second collective itself, already
        // proved by the reducer match. Its shared read is the sole exception.
        for (size_t i = 1u; i < suffix.size(); i++) { _visit(suffix[i]); }
        if (!_valid || _outputs != 1u) { return {}; }
        for (auto &&[key, guard] : _ready) {
            if (key != _carry.get() && _reads[key] != _suffix_reads[key]) { return {}; }
        }
        // A cell's address may not escape even in a prefix statement. Ordinary
        // load/store operands do not expose their BufferVar as a pointer value.
        class Escape final : public tvm::tirx::StmtExprVisitor {
        private:
            const luisa::unordered_map<BufferKey, tvm::PrimExpr> &_cells;
            void VisitExpr_(const tvm::tirx::VarNode *var) final { valid &= !_cells.contains(var); }
            void VisitExpr_(const tvm::tirx::BufferLoadNode *load) final {
                for (auto &&index : load->indices) { VisitExpr(index); }
                if (load->predicate) { VisitExpr(load->predicate.value()); }
            }
            void VisitStmt_(const tvm::tirx::BufferStoreNode *store) final {
                VisitExpr(store->value);
                for (auto &&index : store->indices) { VisitExpr(index); }
                if (store->predicate) { VisitExpr(store->predicate.value()); }
            }
            void VisitStmt_(const tvm::tirx::AllocBufferNode *) final {}
        public:
            bool valid{true};
            explicit Escape(const luisa::unordered_map<BufferKey, tvm::PrimExpr> &cells) noexcept : _cells{cells} {}
        } escape{_ready};
        escape(body);
        if (!escape.valid) { return {}; }
        prefix.push_back(tvm::tirx::IfThenElse{
            tvm::equal(subgroup, tvm::IntImm::Int64(0)), tvm::tirx::SeqStmt::Flatten(suffix)});
        return tvm::tirx::SeqStmt::Flatten(prefix);
    }
};

class ReductionProgramMapper final : public DiagnosticStmtExprMutator {
private:
    tvm::PrimExpr _worker;
    tvm::PrimExpr _lane;
    tvm::PrimExpr _subgroup;
    tvm::PrimExpr _partial_base;
    tvm::ffi::Optional<tvm::PrimExpr> _program_active;
    uint64_t _workers;
    uint64_t _subgroups;
    uint32_t _unroll_factor;
    uint32_t _lane_elements;
    bool _vector_packs;
    const tvm::tirx::ForNode *_program_domain;
    const tvm::tirx::ForNode *_thread_domain;
    // Only loops retained by this mapper belong here. Distributed/reduction
    // axes are replaced by the phase's synthetic chunk/element coordinates.
    luisa::vector<const tvm::tirx::ForNode *> _retained_domain;
    luisa::unordered_set<BufferKey> _vector_allocated;
    luisa::vector<VectorPackPhaseMemoryFacts> *_memory_facts{nullptr};
    uint64_t _vector_phases{0u};
    uint64_t _scalar_phases{0u};
    uint64_t _contribution_storage_budget;
    uint64_t _contribution_storage_scalars{0u};
    SubgroupReductionTarget _target;
    const ReductionAnalysis &_analysis;
    const luisa::unordered_map<const tvm::tirx::ForNode *,
                               tvm::tirx::BufferVar> &_partials;
    const luisa::unordered_map<BufferKey, tvm::tirx::BufferVar>
        &_striped_buffers;
    luisa::optional<tvm::PrimExpr> _striped_slot;
    uint32_t _lane_depth{0u};
    const tvm::tirx::BufferStoreNode *_second_collective{nullptr};

    [[nodiscard]] luisa::vector<const tvm::tirx::ForNode *> _phase_proof_domain(
        const tvm::tirx::ForNode *chunk, const tvm::tirx::ForNode *element) const {
        luisa::vector<const tvm::tirx::ForNode *> domain;
        domain.reserve(_retained_domain.size() + 4u);
        domain.emplace_back(_program_domain);
        domain.emplace_back(_thread_domain);
        domain.insert(domain.end(), _retained_domain.begin(), _retained_domain.end());
        domain.emplace_back(chunk);
        domain.emplace_back(element);
        return domain;
    }

    [[nodiscard]] tvm::tirx::For _visit_retained_loop(const tvm::tirx::ForNode *loop) {
        if (_vector_packs) { _retained_domain.emplace_back(loop); }
        auto result = StmtExprMutator::VisitStmt_(loop).as_or_throw<tvm::tirx::For>();
        if (_vector_packs) { _retained_domain.pop_back(); }
        return result;
    }

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
        const tvm::tirx::PrimVar &element, tvm::tirx::Stmt body, uint64_t count, bool vector_pack = false) const {
        if (count == 1u) {
            return tvm::tirx::Substitute(std::move(body),
                                         tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>{{element, tvm::IntImm::Int64(0)}});
        }
        return tvm::tirx::For{element, tvm::IntImm::Int64(0),
                              tvm::IntImm::Int64(static_cast<int64_t>(count)),
                              vector_pack ? tvm::tirx::ForKind::kVectorized : tvm::tirx::ForKind::kUnrolled, std::move(body)};
    }

    [[nodiscard]] tvm::tirx::Stmt _distributed_loop(
        const tvm::tirx::PrimVar &chunk, const tvm::tirx::PrimVar &element,
        uint64_t elements, tvm::tirx::Stmt body, bool vector_full_packs = false) const {
        auto stride = _workers * _lane_elements;
        auto complete_chunks = elements / stride;
        tvm::ffi::Array<tvm::tirx::Stmt> statements;
        if (complete_chunks != 0u) {
            statements.push_back(_stripe_loop(chunk, complete_chunks, _element_pack(element, body, _lane_elements, vector_full_packs)));
        }
        // The final chunk has full worker packs followed by at most one
        // partial pack. Guard a whole pack, rather than each unrolled scalar.
        // Keep the partial worker separate: it must not speculate a load or
        // private-stripe access beyond its domain. Each worker's recurrence
        // order and storage slots are identical to the element-guarded form.
        if (auto remaining = elements % stride; remaining != 0u) {
            auto complete_workers = remaining / _lane_elements;
            auto partial_elements = remaining % _lane_elements;
            auto boundary = tvm::IntImm::Int64(static_cast<int64_t>(complete_workers));
            tvm::ffi::Array<tvm::tirx::Stmt> tail;
            if (complete_workers != 0u) {
                tail.push_back(tvm::tirx::IfThenElse{
                    _worker < boundary, _element_pack(element, body, _lane_elements, vector_full_packs)});
            }
            if (partial_elements != 0u) {
                tail.push_back(tvm::tirx::IfThenElse{
                    tvm::equal(_worker, boundary), _element_pack(element, std::move(body), partial_elements)});
            }
            statements.push_back(tvm::tirx::Substitute(tvm::tirx::SeqStmt::Flatten(std::move(tail)),
                                                       tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>{{chunk, tvm::IntImm::Int64(static_cast<int64_t>(complete_chunks))}}));
        }
        if (statements.empty()) { return tvm::tirx::Evaluate{tvm::IntImm::Int32(0)}; }
        return tvm::tirx::SeqStmt::Flatten(statements);
    }

    // This only batches independent contributions. Each carry update keeps
    // its original chunk/element order and the unchanged FP32 combine helper.
    [[nodiscard]] tvm::ffi::Optional<tvm::tirx::Stmt> _contribution_packs(
        const tvm::tirx::PrimVar &chunk, const tvm::tirx::PrimVar &element,
        const ReductionMatch &match, const tvm::PrimExpr &contribution,
        const tvm::tirx::Stmt &scalar_update,
        const tvm::tirx::BufferVar &scratch, const tvm::tirx::Stmt &staged_update) {
        auto stride = _workers * _lane_elements;
        auto full_chunks = match.elements / stride;
        if (!_vector_packs || _target != SubgroupReductionTarget::CUDA ||
            full_chunks == 0u || _contribution_storage_scalars > _contribution_storage_budget ||
            _lane_elements > _contribution_storage_budget - _contribution_storage_scalars) {
            return {};
        }
        auto zero = tvm::IntImm::Int64(0);
        auto chunk_domain = tvm::tirx::For{chunk, zero,
            tvm::IntImm::Int64(static_cast<int64_t>(full_chunks)),
            tvm::tirx::ForKind::kSerial, tvm::tirx::Evaluate{zero}};
        auto element_domain = tvm::tirx::For{element, zero, tvm::IntImm::Int64(_lane_elements),
            tvm::tirx::ForKind::kSerial, tvm::tirx::Evaluate{zero}};
        auto proof_domain = _phase_proof_domain(chunk_domain.get(), element_domain.get());
        auto owned = _vector_allocated;
        owned.emplace(scratch.get());
        tvm::tirx::Stmt load_pack = tvm::tirx::BufferStore{scratch, contribution, {element}};
        // Fold only on this full-chunk copy, before BOTH proof and emission.
        // scalar_update and the complete scalar tail retain their original guards.
        load_pack = FullChunkPredicateSpecializer{proof_domain}(load_pack);
        VectorPackPhaseAudit audit{element, _lane_elements, owned, proof_domain};
        audit(load_pack);
        auto guard = audit.guard();
        if (!guard) { return {}; }
        // The exact audit that admits the emitted load produces its facts.
        // This only covers full chunks; the entire remaining chunk stays scalar.
        if (_memory_facts != nullptr) {
            auto facts = audit.memory_phase(_vector_phases + _scalar_phases,
                full_chunks * stride, _workers, _thread_domain, chunk, _lane, guard.value());
            facts.kind = "contribution-staging-full-chunks";
            _memory_facts->emplace_back(std::move(facts));
        }
        auto full_pack = tvm::tirx::SeqStmt::Flatten(tvm::ffi::Array<tvm::tirx::Stmt>{
            _element_pack(element, std::move(load_pack), _lane_elements, true),
            _element_pack(element, staged_update, _lane_elements)});
        tvm::ffi::Array<tvm::tirx::Stmt> vector_statements{
            tvm::tirx::AllocBuffer{scratch},
            _stripe_loop(chunk, full_chunks, std::move(full_pack))};
        if (auto remaining = match.elements % stride; remaining != 0u) {
            auto shifted = tvm::tirx::Substitute(scalar_update,
                tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>{{chunk,
                    chunk + tvm::IntImm::Int64(static_cast<int64_t>(full_chunks))}});
            vector_statements.push_back(_distributed_loop(chunk, element, remaining, std::move(shifted)));
        }
        auto vector = tvm::tirx::SeqStmt::Flatten(std::move(vector_statements));
        auto scalar = _distributed_loop(chunk, element, match.elements, scalar_update);
        // Each successful reduction owns a different allocation. Charge their
        // sum, not a hypothetical reused peak. This is a structural budget,
        // neither registers nor local-memory traffic, and does not rescore.
        _contribution_storage_scalars += _lane_elements;
        _vector_phases++;
        return tvm::tirx::IfThenElse{guard.value(), std::move(vector), std::move(scalar)};
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
        auto contribution = VisitPrimExpr(match.contribution);
        _striped_slot = std::move(previous_slot);
        if (_diagnostic.failed()) { return tvm::ffi::GetRef<tvm::tirx::For>(loop); }
        contribution = tvm::tirx::Substitute(
            contribution, tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>{{loop->loop_var, linear}});
        auto current = tvm::tirx::BufferLoad{
            match.carry, {tvm::IntImm::Int64(0)}};
        auto make_update = [&](const tvm::PrimExpr &value) {
            tvm::PrimExpr combined;
            if (_target == SubgroupReductionTarget::CUDA) {
                // Reduction arithmetic keeps subnormals even when the separately
                // compiled pointwise math uses FTZ. The helper owns this contract.
                auto combine = match.kind == reduction_add_contract ? "__luisa_tile_cuda_reduce_add" :
                               match.kind == reduction_max_contract ? "__luisa_tile_cuda_reduce_max" :
                                                                      "__luisa_tile_cuda_reduce_min";
                combined = tvm::Call{tvm::PrimType::Float(32), tvm::tirx::builtin::call_pure_extern(), {tvm::tirx::StringImm{combine}, current, value}};
            } else if (match.kind == reduction_add_contract) {
                combined = current + value;
            } else if (match.kind == reduction_max_contract) {
                combined = tvm::max(current, value);
            } else {
                combined = tvm::min(current, value);
            }
            return tvm::tirx::BufferStore{
                match.carry, std::move(combined), {tvm::IntImm::Int64(0)}};
        };
        tvm::tirx::Stmt update = make_update(contribution);
        tvm::ffi::Optional<tvm::tirx::Stmt> staged;
        if (_vector_packs) {
            auto scratch = tvm::tirx::decl_buffer({width}, tvm::PrimType::Float(32),
                loop->loop_var->name + "_contribution_pack", "local");
            // This is a new actual allocation; no external or existing buffer
            // receives a stronger alignment promise.
            auto scratch_type = tvm::tirx::CopyBufferType(scratch);
            scratch_type->data_alignment = static_cast<int32_t>(sizeof(float) * _lane_elements);
            scratch = tvm::tirx::RebuildBufferVar(scratch, std::move(scratch_type));
            staged = _contribution_packs(chunk, element, match, contribution, update, scratch,
                make_update(tvm::tirx::BufferLoad{scratch, {element}}));
        }
        auto striped = staged ? staged.value() :
            _distributed_loop(chunk, element, match.elements, std::move(update));
        auto intrinsic = _target == SubgroupReductionTarget::CUDA ?
                             (match.kind == reduction_add_contract ? "__luisa_tile_cuda_warp_sum" :
                              match.kind == reduction_max_contract ? "__luisa_tile_cuda_warp_max" :
                                                                     "__luisa_tile_cuda_warp_min") :
                             (match.kind == reduction_add_contract ? "simd_sum" :
                              match.kind == reduction_max_contract ? "simd_max" :
                                                                     "simd_min");
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
                {_partial_base + _subgroup}}});
        statements.push_back(shared_barrier());
        auto input = tvm::if_then_else(
            _lane < tvm::IntImm::Int64(static_cast<int64_t>(_subgroups)),
            tvm::tirx::BufferLoad{partial, {_partial_base + _lane}},
            reduction_identity(match.kind));
        auto total = tvm::Call{
            tvm::PrimType::Float(32),
            tvm::tirx::builtin::call_pure_extern(),
            {tvm::tirx::StringImm{intrinsic}, std::move(input)}};
        auto second = tvm::tirx::BufferStore{
            match.carry, std::move(total), {tvm::IntImm::Int64(0)}};
        _second_collective = second.get();
        statements.push_back(std::move(second));
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
        if (_diagnostic.failed()) { return tvm::ffi::GetRef<tvm::tirx::For>(loop); }
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
        if (_vector_packs && domain.count >= _lane_elements) {
            // These loops are proof inputs only. The original bounds/lazy guards
            // are retained in both emitted branches; common simplification may
            // remove a condition only when it proves it for the real domain.
            auto zero = tvm::IntImm::Int64(0);
            auto chunks = luisa::ceil_div(domain.count, _workers * _lane_elements);
            auto chunk_domain = tvm::tirx::For{chunk, zero,
                tvm::IntImm::Int64(static_cast<int64_t>(chunks)),
                tvm::tirx::ForKind::kSerial, tvm::tirx::Evaluate{zero}};
            auto element_domain = tvm::tirx::For{element, zero, tvm::IntImm::Int64(_lane_elements),
                tvm::tirx::ForKind::kSerial, tvm::tirx::Evaluate{zero}};
            auto proof_domain = _phase_proof_domain(chunk_domain.get(), element_domain.get());
            VectorPackPhaseAudit audit{element, _lane_elements, _vector_allocated, proof_domain};
            audit(body);
            if (auto guard = audit.guard()) {
                // Retain facts from precisely the proof/guard that emits this
                // body's branches. The record owns its expression references.
                if (_memory_facts != nullptr) {
                    _memory_facts->emplace_back(audit.memory_phase(
                        _vector_phases + _scalar_phases, domain.count, _workers,
                        _thread_domain, chunk, _lane, guard.value()));
                }
                auto scalar = _distributed_loop(chunk, element, domain.count, body);
                auto vector = _distributed_loop(chunk, element, domain.count, std::move(body), true);
                _vector_phases++;
                return tvm::tirx::IfThenElse{guard.value(), std::move(vector), std::move(scalar)};
            }
        }
        if (_memory_facts != nullptr) {
            _memory_facts->emplace_back(VectorPackPhaseMemoryFacts{
                _vector_phases + _scalar_phases, domain.count, _workers,
                _lane_elements, "no-proven-vector-phase", {}, {}});
        }
        _scalar_phases += _vector_packs;
        return _distributed_loop(chunk, element, domain.count, std::move(body));
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
                    return _diagnostic.reject("validated SIMD-group element domain became invalid", tvm::ffi::GetRef<tvm::tirx::For>(loop));
                }
                return _distributed_elements(loop, *domain);
            }
            auto result = _visit_retained_loop(loop);
            auto node = result.CopyOnWrite();
            node->annotations.erase(independent_elements_annotation);
            node->annotations.erase(materialized_pure_tile_annotation);
            node->annotations.erase(mma_annotation);
            return result;
        }
        auto result = _visit_retained_loop(loop);
        auto node = result.CopyOnWrite();
        node->annotations.erase(deferred_pipeline_annotation);
        node->annotations.erase(reduction_contract_annotation);
        node->annotations.erase(reduction_policy_annotation);
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
        if (_vector_packs &&
            (buffer.scope() != "local" ||
             !allocation->annotations.count(tvm::tirx::attr::buffer_data_alignment))) {
            // CUDA codegen lets this annotation override buffer data_alignment.
            // Unknown local overrides cannot prove an aligned vector access.
            // Non-local allocations still remain excluded from root pointers.
            _vector_allocated.emplace(buffer.get());
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
                return _diagnostic.reject("proved striped Tile storage escaped its element domain", tvm::ffi::GetRef<tvm::tirx::BufferLoad>(load));
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
                return _diagnostic.reject("proved striped Tile storage escaped its element domain", tvm::ffi::GetRef<tvm::tirx::BufferStore>(store));
            }
            return tvm::tirx::BufferStore{
                iter->second, VisitPrimExpr(store->value), {_striped_slot.value()}, _predicate(store->predicate), store->span};
        }
        auto result = StmtExprMutator::VisitStmt_(store);
        if (_program_active && store->buffer.scope() != "local") {
            result = tvm::tirx::IfThenElse{_program_active.value(), std::move(result)};
        }
        return result;
    }

    [[nodiscard]] tvm::Expr VisitExpr_(
        const tvm::tirx::VarNode *variable) final {
        if (_striped_buffers.contains(variable)) {
            return _diagnostic.reject("proved striped Tile storage escaped through an opaque use", tvm::ffi::GetRef<tvm::tirx::Var>(variable));
        }
        return StmtExprMutator::VisitExpr_(variable);
    }

public:
    ReductionProgramMapper(
        tvm::PrimExpr worker, tvm::PrimExpr lane,
        tvm::PrimExpr subgroup, tvm::PrimExpr partial_base,
        tvm::ffi::Optional<tvm::PrimExpr> program_active,
        uint64_t workers, uint64_t subgroups, uint32_t unroll_factor, uint32_t lane_elements,
        bool vector_packs, uint64_t contribution_storage_budget,
        const tvm::tirx::ForNode *program_domain, const tvm::tirx::ForNode *thread_domain,
        SubgroupReductionTarget target, const ReductionAnalysis &analysis,
        const luisa::unordered_map<const tvm::tirx::ForNode *,
                                   tvm::tirx::BufferVar> &partials,
        const luisa::unordered_map<BufferKey,
                                   tvm::tirx::BufferVar> &striped_buffers,
        Diagnostic &diagnostic,
        luisa::vector<VectorPackPhaseMemoryFacts> *memory_facts = nullptr) noexcept
        : DiagnosticStmtExprMutator{diagnostic}, _worker{std::move(worker)}, _lane{std::move(lane)},
          _subgroup{std::move(subgroup)}, _partial_base{std::move(partial_base)},
          _program_active{std::move(program_active)}, _workers{workers},
          _subgroups{subgroups}, _unroll_factor{unroll_factor}, _lane_elements{lane_elements},
          _vector_packs{vector_packs}, _program_domain{program_domain}, _thread_domain{thread_domain},
          _memory_facts{memory_facts},
          _contribution_storage_budget{contribution_storage_budget},
          _target{target}, _analysis{analysis}, _partials{partials}, _striped_buffers{striped_buffers} {}

    [[nodiscard]] const tvm::tirx::BufferStoreNode *second_collective() const noexcept { return _second_collective; }
    [[nodiscard]] uint64_t contribution_storage_scalars() const noexcept { return _contribution_storage_scalars; }
    [[nodiscard]] uint64_t vector_phases() const noexcept { return _vector_phases; }
    [[nodiscard]] uint64_t scalar_phases() const noexcept { return _scalar_phases; }
};

struct ReductionTileMatch {
    ElementDomain domain;
    ReductionMatch reduction;
    const tvm::tirx::ForNode *loop;
    const tvm::tirx::BufferStoreNode *output;
};

[[nodiscard]] luisa::optional<ReductionTileMatch> match_reduction_tile(const tvm::tirx::For &loop) {
    auto domain = element_domain(loop.get());
    if (!domain || domain->count == 0u || domain->count > INT64_MAX ||
        loop->annotations.size() != 1u) { return {}; }
    tvm::ffi::Array<tvm::tirx::Stmt> statements;
    flatten_sequence(domain->body, statements);
    if (statements.size() != 4u) { return {}; }
    auto allocation = statements[0u].as<tvm::tirx::AllocBufferNode>();
    auto initializer = statements[1u].as<tvm::tirx::BufferStoreNode>();
    auto reduction = statements[2u].as<tvm::tirx::ForNode>();
    auto output = statements[3u].as<tvm::tirx::BufferStoreNode>();
    if (!allocation || !initializer || !reduction || !output || output->predicate || !allocation->annotations.empty()) { return {}; }
    auto match = match_reduction(reduction);
    auto result = output->value.as<tvm::tirx::BufferLoadNode>();
    if (!match || !allocation->buffer.same_as(match->carry) || !identity_initializer(initializer, *match) ||
        !result || result->predicate || !result->buffer.same_as(match->carry) || !zero_index(result->indices) ||
        output->buffer.same_as(match->carry)) { return {}; }
    return ReductionTileMatch{std::move(*domain), std::move(*match), reduction, output};
}

}// namespace

luisa::optional<uint64_t> metal_reduction_tile_output_count(const tvm::tirx::For &loop) {
    auto match = match_reduction_tile(loop);
    return match ? luisa::optional{match->domain.count} : luisa::nullopt;
}

tvm::tirx::Stmt try_metal_reduction_tile(
    const tvm::tirx::For &loop, const tvm::tirx::PrimVar &thread, uint64_t threads,
    const luisa::function<tvm::tirx::BufferVar(tvm::tirx::BufferVar)> &map_buffer) {
    if (threads < subgroup_size || threads % subgroup_size != 0u) { return {}; }
    auto tile = match_reduction_tile(loop);
    if (!tile) { return {}; }
    auto domain = &tile->domain;
    auto match = &tile->reduction;
    auto reduction = tile->loop;
    auto output = tile->output;

    // Reuse the semantic reduction matcher, but do not run the whole-program
    // mapper: surrounding phases may contain matrices and shared resources.
    // Independence of distinct outputs is the enclosing element contract.
    class AccessMapper final : public tvm::tirx::StmtExprMutator {
    private:
        const luisa::function<tvm::tirx::BufferVar(tvm::tirx::BufferVar)> &_map;
    protected:
        tvm::Expr VisitExpr_(const tvm::tirx::BufferLoadNode *load) final {
            return tvm::tirx::BufferLoad{_map(load->buffer),
                                         load->indices.Map([this](auto &&index) { return VisitPrimExpr(index); }),
                                         load->predicate ? tvm::ffi::Optional<tvm::PrimExpr>{VisitPrimExpr(load->predicate.value())} : luisa::nullopt,
                                         load->span};
        }
    public:
        explicit AccessMapper(const luisa::function<tvm::tirx::BufferVar(tvm::tirx::BufferVar)> &map) noexcept : _map{map} {}
        tvm::PrimExpr expression(const tvm::PrimExpr &value) { return VisitPrimExpr(value); }
    } mapper{map_buffer};

    auto zero = tvm::IntImm::Int64(0);
    auto width = tvm::IntImm::Int64(static_cast<int64_t>(subgroup_size));
    auto lane = tvm::floormod(thread, width);
    auto subgroup = tvm::floordiv(thread, width);
    auto subgroups = threads / subgroup_size;
    auto batch = tvm::tirx::PrimVar{loop->loop_var->name + "_reduction_batch", tvm::PrimType::Int(64)};
    auto row = batch * tvm::IntImm::Int64(static_cast<int64_t>(subgroups)) + subgroup;
    tvm::ffi::Map<tvm::tirx::Var, tvm::Expr> coordinates;
    auto trailing = domain->count;
    for (auto axis : domain->axes) {
        auto extent = *static_extent(axis->extent, true);
        trailing /= extent;
        auto coordinate = tvm::floormod(tvm::floordiv(row, tvm::IntImm::Int64(static_cast<int64_t>(trailing))), axis->extent);
        coordinates.Set(axis->loop_var, axis->min + coordinate);
    }
    auto chunk = tvm::tirx::PrimVar{reduction->loop_var->name + "_reduction_chunk", tvm::PrimType::Int(64)};
    auto index = chunk * width + lane;
    coordinates.Set(reduction->loop_var, index);
    auto contribution = tvm::tirx::Substitute(mapper.expression(match->contribution), coordinates);
    auto carry = tvm::tirx::decl_buffer({tvm::IntImm::Int64(1)}, tvm::PrimType::Float(32),
                                        reduction->loop_var->name + "_lane_carry", "local");
    auto current = tvm::tirx::BufferLoad{carry, {zero}};
    auto combine = [&](tvm::PrimExpr value) -> tvm::PrimExpr {
        if (match->kind == reduction_add_contract) { return current + value; }
        if (match->kind == reduction_max_contract) { return tvm::max(current, value); }
        return tvm::min(current, value);
    };
    auto chunks = luisa::ceil_div(match->elements, subgroup_size);
    tvm::tirx::Stmt update = tvm::tirx::BufferStore{carry, combine(std::move(contribution)), {zero}};
    if (match->elements % subgroup_size != 0u) {
        update = tvm::tirx::IfThenElse{index < reduction->extent, std::move(update)};
    }
    auto intrinsic = match->kind == reduction_add_contract ? "simd_sum" :
                     match->kind == reduction_max_contract ? "simd_max" :
                                                             "simd_min";
    auto collective = tvm::Call{tvm::PrimType::Float(32), tvm::tirx::builtin::call_pure_extern(), {tvm::tirx::StringImm{intrinsic}, current}};
    auto reduced = tvm::tirx::PrimVar{reduction->loop_var->name + "_subgroup_value", tvm::PrimType::Float(32)};
    auto indices = output->indices.Map([&](auto &&value) { return tvm::tirx::Substitute(mapper.expression(value), coordinates); });
    tvm::tirx::Stmt body = tvm::tirx::SeqStmt::Flatten(tvm::ffi::Array<tvm::tirx::Stmt>{
        tvm::tirx::AllocBuffer{carry},
        tvm::tirx::BufferStore{carry, reduction_identity(match->kind), {zero}},
        tvm::tirx::For{chunk, zero, tvm::IntImm::Int64(static_cast<int64_t>(chunks)), tvm::tirx::ForKind::kSerial, std::move(update)},
        // All lanes enter the collective, including identity-padded tails.
        // The leader predicate controls only publication, never participation.
        tvm::tirx::Bind{reduced, std::move(collective)},
        tvm::tirx::IfThenElse{tvm::equal(lane, zero), tvm::tirx::BufferStore{map_buffer(output->buffer), reduced, std::move(indices)}}});
    if (domain->count % subgroups != 0u) {
        body = tvm::tirx::IfThenElse{row < tvm::IntImm::Int64(static_cast<int64_t>(domain->count)), std::move(body)};
    }
    return tvm::tirx::For{batch, zero, tvm::IntImm::Int64(static_cast<int64_t>(luisa::ceil_div(domain->count, subgroups))),
                          tvm::tirx::ForKind::kSerial, std::move(body)};
}

tvm::tirx::Stmt try_map_subgroup_reduction(
    const tvm::tirx::For &loop, uint32_t max_threads,
    uint64_t shared_memory_limit,
    const PlannerOptions &options, luisa::vector<GroupPlan> &plans, Diagnostic &diagnostic,
    SubgroupReductionTarget target) {
    auto groups = static_extent(loop->extent, true);
    auto minimum = loop->min.as<tvm::IntImmNode>();
    auto scope = loop->annotations.Get(execution_scope_annotation);
    auto scope_name = scope ? scope.value().as<tvm::ffi::String>() :
                              tvm::ffi::Optional<tvm::ffi::String>{};
    auto cuda = target == SubgroupReductionTarget::CUDA;
    auto enabled = cuda ? options.cuda_subgroup_reductions : options.metal_subgroup_reductions;
    auto vector_packs = false;
    auto row_only = false;
    if (cuda) {
        if (auto value = luisa::get_environment_variable("LUISA_DIAGNOSTIC_TIRX_ROW_ONLY_COLLECTIVE")) {
            if (*value == "1") { row_only = true; }
            else if (*value != "0") {
                return diagnostic.reject("private CUDA row-only collective must be exactly 0 or 1", tvm::tirx::Stmt{});
            }
        }
        if (auto value = luisa::get_environment_variable("LUISA_DIAGNOSTIC_TIRX_VECTOR_PACKS")) {
            if (*value == "1") {
                vector_packs = true;
            } else if (*value != "0") {
                return diagnostic.reject("private CUDA vector packs must be exactly 0 or 1", tvm::tirx::Stmt{});
            }
        }
        if (vector_packs && (!enabled ||
                             (options.reduction_lane_elements != 2u &&
                              options.reduction_lane_elements != 4u &&
                              options.reduction_lane_elements != 8u))) {
            return diagnostic.reject("private CUDA vector packs require CUDA subgroups and lane_elements=2, 4 or 8", tvm::tirx::Stmt{});
        }
        if (vector_packs) {
            if (auto chains = luisa::get_environment_variable("LUISA_DIAGNOSTIC_TIRX_SUM_LOCAL_CHAINS");
                chains && *chains != "1") {
                return diagnostic.reject("private CUDA vector packs cannot combine with the local-chains experiment", tvm::tirx::Stmt{});
            }
        }
    }
    if (!enabled ||
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
    PackedProgramAudit packing_audit{analysis};
    packing_audit(loop->body);
    DistributedLocalAudit ownership{analysis};
    ownership(loop->body);
    if (!ownership.valid()) { return {}; }
    auto materializations = striped_materializations(loop->body, analysis, cuda);
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
        return diagnostic.reject("invalid reduction cost coefficients or search budget", tvm::tirx::Stmt{});
    }
    struct Candidate {
        uint64_t subgroups{0u};
        uint64_t packed_programs{0u};
        uint64_t threads{0u};
        uint64_t partial_bytes{0u};
        uint64_t striped_storage_scalars{0u};
        uint32_t unroll_factor{1u};
        double scalar_rounds{0.0};
        double lane_utilization{0.0};
        ReductionCost cost{0.0, 1.0, std::numeric_limits<double>::infinity()};
    };
    luisa::vector<uint64_t> widths;
    auto exact_packing = options.reduction_programs_per_group;
    auto requested_packing = std::max<uint64_t>(1u, exact_packing);
    auto maximum_program_subgroups = maximum_subgroups / requested_packing;
    if (maximum_program_subgroups == 0u) { return {}; }
    if (options.threads_per_group != 0u) {
        if (options.threads_per_group > max_threads ||
            options.threads_per_group % (subgroup_size * requested_packing) != 0u) {
            return {};
        }
        widths.emplace_back(options.threads_per_group / (subgroup_size * requested_packing));
    } else {
        if (maximum_program_subgroups > options.max_thread_candidates) {
            return diagnostic.reject("reduction thread candidate budget exceeded; increase the budget or request an exact width", tvm::tirx::Stmt{});
        }
        for (auto subgroups = uint64_t{1u};
             subgroups <= maximum_program_subgroups; subgroups++) {
            widths.emplace_back(subgroups);
        }
    }
    struct PreparedCandidate {
        tvm::tirx::Stmt body;
        luisa::vector<VectorPackPhaseMemoryFacts> memory;
        luisa::string error;
        uint64_t striped_storage_scalars{0u};
        uint64_t contribution_storage_scalars{0u};
        uint64_t vector_phases{0u};
        uint64_t scalar_phases{0u};
        bool vector_unsupported{false};
        bool row_only_applied{false};
        // Deliberately incomplete. Empty records cannot mean free memory.
        const char *memory_coverage{"partial-independent-and-staged-load-phases"};
        bool whole_kernel_memory_known{false};
        bool primitive_ir_counts_known{false};
        bool scalar_independent_coverage_complete{false};
        bool reduction_contribution_coverage_complete{false};
        bool launch_participation_coverage_complete{false};
    };
    // Default selects first and calls this exactly once. The explicit private
    // CUDA vector experiment prepares before scoring and keeps the winning IR.
    auto prepare = [&](const Candidate &candidate) -> PreparedCandidate {
        PreparedCandidate prepared;
        Diagnostic local_diagnostic;
        auto subgroups_per_program = candidate.subgroups;
        auto multi_subgroup = subgroups_per_program > 1u;
        auto packed_programs = candidate.packed_programs;
        auto threads = candidate.threads;
        auto blocks = luisa::ceil_div(*groups, packed_programs);
        auto program_workers = subgroups_per_program * subgroup_size;
        auto block = tvm::tirx::PrimVar{
            loop->loop_var->name + "_subgroup_block", tvm::PrimType::Int(64)};
        auto thread = tvm::tirx::PrimVar{
            loop->loop_var->name + "_subgroup_thread", tvm::PrimType::Int(64)};
        auto lane = tvm::tirx::PrimVar{
            loop->loop_var->name + "_subgroup_lane", tvm::PrimType::Int(64)};
        auto packed_index = tvm::floordiv(thread, tvm::IntImm::Int64(static_cast<int64_t>(program_workers)));
        // Vector proof and its emitted phase must share the actual worker
        // coordinate. A separate free lane variable is substituted only after
        // mapping and cannot prove S1 bounds here. Keep the legacy default path.
        tvm::PrimExpr worker = (multi_subgroup || vector_packs) ?
                                   tvm::floormod(thread, tvm::IntImm::Int64(static_cast<int64_t>(program_workers))) :
                                   tvm::PrimExpr{lane};
        auto subgroup = tvm::floordiv(worker, tvm::IntImm::Int64(static_cast<int64_t>(subgroup_size)));
        auto partial_base = packed_index * tvm::IntImm::Int64(static_cast<int64_t>(subgroups_per_program));
        auto logical = block * tvm::IntImm::Int64(static_cast<int64_t>(packed_programs)) + packed_index;
        auto packed_tail = blocks * packed_programs != *groups;
        auto replay_tail = multi_subgroup && packed_tail;
        tvm::ffi::Optional<tvm::PrimExpr> program_active;
        tvm::PrimExpr mapped_logical = logical;
        if (replay_tail) {
            program_active = logical < loop->extent;
            mapped_logical = tvm::min(logical, loop->extent - tvm::IntImm::Int64(1));
        }
        luisa::unordered_map<const tvm::tirx::ForNode *, tvm::tirx::BufferVar>
            partials;
        tvm::ffi::Array<tvm::tirx::Stmt> allocations;
        if (multi_subgroup) {
            for (auto reduction : analysis.reduction_order) {
                auto partial = tvm::tirx::decl_buffer(
                    {tvm::IntImm::Int64(
                        static_cast<int64_t>(subgroups_per_program * packed_programs))},
                    tvm::PrimType::Float(32),
                    loop->loop_var->name + "_subgroup_partials_" +
                        std::to_string(partials.size()),
                    "shared");
                partials.emplace(reduction, partial);
                allocations.push_back(tvm::tirx::AllocBuffer{std::move(partial)});
            }
        }
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
        if (striped_storage_scalars != candidate.striped_storage_scalars) {
            prepared.error = "reduction stripe resource accounting changed after planning";
            return prepared;
        }
        tvm::ffi::Optional<tvm::tirx::For> vector_thread_domain;
        if (vector_packs) {
            auto zero = tvm::IntImm::Int64(0);
            vector_thread_domain = tvm::tirx::For{thread, zero,
                tvm::IntImm::Int64(static_cast<int64_t>(threads)), tvm::tirx::ForKind::kSerial,
                tvm::tirx::Evaluate{zero}};
        }
        auto mapper = ReductionProgramMapper{
            worker, lane, subgroup, partial_base, program_active,
            program_workers, multi_subgroup ? subgroups_per_program : 1u, candidate.unroll_factor, options.reduction_lane_elements,
            vector_packs, options.max_reduction_striped_scalars_per_worker - candidate.striped_storage_scalars,
            loop.get(), vector_thread_domain ? vector_thread_domain.value().get() : nullptr,
            target, analysis, partials, striped_buffers, local_diagnostic,
            vector_packs ? &prepared.memory : nullptr};
        auto body = mapper(loop->body);
        if (local_diagnostic.failed()) { prepared.error = local_diagnostic.error(); return prepared; }
        if (row_only && multi_subgroup && analysis.reductions.size() == 1u && mapper.second_collective()) {
            TerminalRowSuffix suffix{mapper.second_collective(), worker, thread, threads, lane, block, loop.get()};
            if (auto terminal = suffix.rewrite(body, subgroup)) {
                body = terminal.value();
                prepared.row_only_applied = true;
            }
        }
        if (vector_packs) {
            prepared.vector_unsupported = mapper.vector_phases() == 0u;
        }
        if (!allocations.empty()) {
            allocations.push_back(std::move(body));
            body = tvm::tirx::SeqStmt::Flatten(allocations);
        }
        body = tvm::tirx::Substitute(
            std::move(body),
            tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>{
                {loop->loop_var, loop->min + mapped_logical},
                {lane, tvm::floormod(
                           thread,
                           tvm::IntImm::Int64(
                               static_cast<int64_t>(subgroup_size)))}});
        if (!multi_subgroup && packed_tail) {
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

        prepared.body = std::move(body);
        prepared.striped_storage_scalars = striped_storage_scalars;
        prepared.contribution_storage_scalars = mapper.contribution_storage_scalars();
        prepared.vector_phases = mapper.vector_phases();
        prepared.scalar_phases = mapper.scalar_phases();
        return prepared;
    };
    auto score_prepared = [&](const ReductionCandidate &features,
                              const PreparedCandidate *prepared) {
        // The private cost boundary now receives the actual prepared facts.
        // This patch intentionally leaves all ranking/coefficients unchanged.
        // Experimental compatibility: the score retains original graph facts.
        // Staged scratch, scalar tails and ordered-combine demand are incomplete;
        // do not substitute their missing costs with zero or exclude L1.
        if (prepared != nullptr) {
            LUISA_INFO("Private CUDA prepared candidate: T={} S={} P={} L={} U={} vector-supported={}; old policy score unchanged.",
                       features.threads, features.subgroups_per_program,
                       features.programs_per_group, features.lane_elements,
                       features.unroll_factor, !prepared->vector_unsupported);
            LUISA_INFO("Private CUDA prepared coverage: scope={} whole-kernel-known={} scalar-independent-complete={} reduction-contribution-complete={} launch-participation-complete={}; no zero substitution.",
                       prepared->memory_coverage, prepared->whole_kernel_memory_known,
                       prepared->scalar_independent_coverage_complete,
                       prepared->reduction_contribution_coverage_complete,
                       prepared->launch_participation_coverage_complete);
            LUISA_INFO("Private CUDA prepared staging: extra-scratch-scalars={} primitive-IR-counts-known={}; experimental score retains original pre-staging features.",
                       prepared->contribution_storage_scalars, prepared->primitive_ir_counts_known);
            report_prepared_memory(prepared->memory);
        }
        return policy.reduction_cost(features, model);
    };
    auto best = Candidate{};
    std::optional<PreparedCandidate> best_prepared;
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
        auto minimum_unroll = cuda ?
                                  constant_striped_index_min_unroll(materializations, workers, options.reduction_lane_elements) :
                                  luisa::optional<uint64_t>{};
        auto resolved_unroll = options.reduction_unroll_factor;
        if (resolved_unroll == 0u) {
            // Explicit CUDA-only structural choice. Do not silently clamp an
            // unknown or unsatisfiable requirement, or raise storage budgets.
            if (!cuda || !minimum_unroll || *minimum_unroll > 64u) {
                candidates_rejected++;
                continue;
            }
            resolved_unroll = static_cast<uint32_t>(*minimum_unroll);
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
        // Packing and cooperating width are independent dimensions. Retain
        // the automatic family's incumbent while explicit packing/JIT search
        // can explore several cooperating programs in one physical group.
        auto packing_begin = exact_packing ? static_cast<uint64_t>(exact_packing) : 1u;
        auto packing_end = !multi && !exact_packing && options.threads_per_group == 0u ?
                               std::min({*groups, maximum_subgroups, uint64_t{8u}}) :
                               packing_begin;
        auto lane_utilization = scalar_rounds == 0.0 ? 0.0 :
                                                       scalar_elements / (scalar_rounds * static_cast<double>(workers));
        for (auto packed = packing_begin; packed <= packing_end; packed++) {
            auto threads = subgroups * packed * subgroup_size;
            if (threads > max_threads ||
                partial_bytes > shared_memory_limit / packed ||
                // Every cooperating CUDA program must finish reading its
                // shared partials before a later iteration could overwrite
                // them. The current mapper emits only the publication fence.
                (cuda && multi && !packing_audit.uniform_fences) ||
                (multi && packed > 1u &&
                 (!packing_audit.uniform_fences ||
                  (*groups % packed != 0u && !packing_audit.replayable_tail())))) {
                candidates_rejected++;
                continue;
            }
            candidates_considered++;
            auto group_partial_bytes = partial_bytes * packed;
            auto features = ReductionCandidate{
                *groups, static_cast<uint32_t>(threads),
                static_cast<uint32_t>(subgroups), static_cast<uint32_t>(packed),
                group_partial_bytes, striped_storage_scalars, analysis.reductions.size(), scalar_rounds, resolved_unroll, options.reduction_lane_elements,
                luisa::ceil_div(*groups, packed), scalar_elements, lane_utilization,
                accesses.known, accesses.demand(), accesses.demand(workers, options.reduction_lane_elements)};
            features.source_constant_striped_index_min_unroll = minimum_unroll;
            auto candidate = Candidate{subgroups, packed, threads,
                                       group_partial_bytes, striped_storage_scalars,
                                       resolved_unroll, scalar_rounds, lane_utilization, {}};
            std::optional<PreparedCandidate> current_prepared;
            if (vector_packs) {
                current_prepared.emplace(prepare(candidate));
                if (!current_prepared->error.empty()) {
                    return diagnostic.reject(current_prepared->error, tvm::tirx::Stmt{});
                }
            }
            auto cost = score_prepared(features, current_prepared ? &*current_prepared : nullptr);
            candidate.cost = cost;
            if (!std::isfinite(cost.program_score) || cost.program_score < 0.0 ||
                !std::isfinite(cost.concurrent_waves) || cost.concurrent_waves < 1.0 ||
                !std::isfinite(cost.kernel_score) || cost.kernel_score < 0.0) {
                return diagnostic.reject("reduction cost policy returned a nonfinite or negative score", tvm::tirx::Stmt{});
            }
            if (cost.kernel_score < best.cost.kernel_score) {
                best = candidate;
                best_prepared = std::move(current_prepared);
            }
        }
    }
    if (best.subgroups == 0u) { return {}; }
    // The original/default path prepares only its selected winner. The private
    // vector path already owns that exact body; never remap or re-prove it.
    if (!best_prepared) { best_prepared.emplace(prepare(best)); }
    if (!best_prepared->error.empty()) {
        return diagnostic.reject(best_prepared->error, tvm::tirx::Stmt{});
    }
    if (best_prepared->vector_unsupported) {
        return diagnostic.reject("private CUDA vector packs unsupported: no eligible independent F16/BF16/F32 storage pack phase", tvm::tirx::Stmt{});
    }
    if (vector_packs) {
        LUISA_INFO("Private CUDA vector packs: {} guarded phases, {} scalar phases; emitted vector instructions remain unverified.",
                   best_prepared->vector_phases, best_prepared->scalar_phases);
    }
    if (row_only) {
        LUISA_INFO("Private CUDA row-only collective: applied={}; unchanged full tree, uniform publication barrier; cost remains original.",
                   best_prepared->row_only_applied);
        // Explicit diagnostic requests need an admission receipt even when a
        // benchmark suppresses ordinary informational logging.
        std::fprintf(stderr, "[luisa-tile-row-only] applied=%u\n",
                     best_prepared->row_only_applied ? 1u : 0u);
    }
    auto body = std::move(best_prepared->body);
    auto subgroups_per_program = best.subgroups;
    auto multi_subgroup = subgroups_per_program > 1u;
    auto partial_bytes = best.partial_bytes;
    auto packed_programs = best.packed_programs;
    auto threads = best.threads;
    auto blocks = luisa::ceil_div(*groups, packed_programs);
    auto program_workers = subgroups_per_program * subgroup_size;
    auto striped_storage_scalars = best_prepared->striped_storage_scalars;

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
    plan.reduction_unroll_factor = best.unroll_factor;
    plan.reduction_lane_elements = options.reduction_lane_elements;
    plan.reduction_threadgroups = blocks;
    plan.reduction_scalar_rounds = best.scalar_rounds;
    plan.reduction_lane_utilization = best.lane_utilization;
    // Existing demands describe the original graph. The new explicit staging
    // loads/stores are not included; do not advertise incomplete counts as known.
    plan.reduction_payload_accesses_known = accesses.known && best_prepared->contribution_storage_scalars == 0u && !best_prepared->row_only_applied;
    plan.reduction_payload_accesses_per_program = accesses.demand();
    plan.reduction_payload_accesses_per_worker = accesses.demand(program_workers, options.reduction_lane_elements);
    plan.striped_storage_scalars_per_worker = striped_storage_scalars + best_prepared->contribution_storage_scalars;
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
