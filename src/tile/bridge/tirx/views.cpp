#include <algorithm>

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>

#include <luisa/core/stl/unordered_map.h>

#include "execution.h"

namespace luisa::compute::tile::bridge::tirx::detail {

namespace {

using BufferKey = const tvm::tirx::VarNode *;
using Domain = luisa::vector<const tvm::tirx::ForNode *>;

struct BufferAccess {
    uint64_t allocations{0u};
    uint64_t stores{0u};
    uint64_t loads{0u};
    bool escapes{false};
};

// A Tile load is a snapshot. Delaying its reads is legal only if the source
// cannot change anywhere in this invocation, including another logical group.
// Distinct external pointers are disjoint only under the caller's noalias
// contract. Unknown effects/aliases fail closed for the entire function.
class InputAccess final : public tvm::tirx::StmtExprVisitor {
public:
    luisa::unordered_map<BufferKey, BufferAccess> buffers;
    bool opaque{false};

protected:
    void VisitBufferDef(const tvm::tirx::BufferVar &buffer, bool allocate) final {
        opaque |= !allocate;
        buffers[buffer.get()].allocations++;
        StmtExprVisitor::VisitBufferDef(buffer, allocate);
    }
    void VisitExpr_(const tvm::tirx::VarNode *variable) final { buffers[variable].escapes = true; }
    void VisitExpr_(const tvm::tirx::ProducerLoadNode *) final { opaque = true; }
    void VisitExpr_(const tvm::tirx::BufferLoadNode *load) final {
        buffers[load->buffer.get()].loads++;
        StmtExprVisitor::VisitExpr_(load);
        if (load->predicate) { VisitExpr(load->predicate.value()); }
    }
    void VisitStmt_(const tvm::tirx::BufferStoreNode *store) final {
        buffers[store->buffer.get()].stores++;
        StmtExprVisitor::VisitStmt_(store);
        if (store->predicate) { VisitExpr(store->predicate.value()); }
    }
    void VisitExpr_(const tvm::CallNode *call) final {
        static auto effects = tvm::Op::GetAttrMap<tvm::tirx::TCallEffectKind>("TCallEffectKind");
        auto op = call->op.as<tvm::Op>();
        opaque |= call->op.same_as(tvm::tirx::builtin::address_of()) || !op || effects.count(op.value()) == 0u ||
                  effects[op.value()] > static_cast<int64_t>(tvm::tirx::CallEffectKind::kPure);
        StmtExprVisitor::VisitExpr_(call);
    }
    void VisitStmt_(const tvm::tirx::AttrStmtNode *attribute) final {
        opaque |= attribute->attr_key != pipeline_stage_annotation;
        StmtExprVisitor::VisitStmt_(attribute);
    }
    void VisitStmt_(const tvm::tirx::ForNode *loop) final {
        opaque |= loop->kind != tvm::tirx::ForKind::kSerial || loop->thread_binding.has_value();
        for (auto &&[key, value] : loop->annotations) {
            opaque |= key != logical_parallel_annotation && key != execution_scope_annotation &&
                      key != independent_elements_annotation && key != mma_annotation &&
                      key != materialized_pure_tile_annotation && key != reduction_contract_annotation &&
                      key != logical_pipeline_annotation && key != pipeline_window_annotation && key != pipeline_interval_annotation;
        }
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

[[nodiscard]] bool compact_buffer(const tvm::tirx::BufferVar &buffer) {
    auto offset = buffer->elem_offset.as<tvm::IntImmNode>();
    if (buffer->dtype != tvm::PrimType::Float(32) || buffer->shape.empty() || !buffer->strides.empty() ||
        buffer->layout || !buffer->allocated_addr.empty() || offset == nullptr || offset->value != 0) { return false; }
    return std::all_of(buffer->shape.begin(), buffer->shape.end(), [](auto &&dimension) {
        auto extent = dimension.template as<tvm::IntImmNode>();
        return extent != nullptr && extent->value > 0;
    });
}

[[nodiscard]] tvm::PrimExpr in_bounds(const tvm::tirx::BufferVar &buffer, const tvm::ffi::Array<tvm::PrimExpr> &indices) {
    tvm::PrimExpr predicate = tvm::IntImm::Bool(indices.size() == buffer->shape.size());
    if (indices.size() == buffer->shape.size()) {
        for (auto i = 0u; i < indices.size(); i++) { predicate = predicate && indices[i] >= 0 && indices[i] < buffer->shape[i]; }
    }
    return predicate;
}

// A guard may already contain every required bound, with a different AND
// association/order or extra masks. Native arithmetic simplification does not
// always recognize (!G || G) for mixed-radix coordinates. This small Boolean
// proof does not infer facts from an OR branch or rename free variables.
// The caller separately proves that all reads in the expression are stable.
[[nodiscard]] bool guard_contains(const tvm::PrimExpr &guard, const tvm::PrimExpr &required) {
    if (tvm::ffi::StructuralEqual{}(guard, required)) { return true; }
    if (auto literal = required.as<tvm::IntImmNode>(); literal != nullptr && literal->value == 1 && required.ty() == tvm::PrimType::Bool()) { return true; }
    if (auto conjunction = required.as<tvm::tirx::AndNode>()) {
        return guard_contains(guard, conjunction->a) && guard_contains(guard, conjunction->b);
    }
    if (auto conjunction = guard.as<tvm::tirx::AndNode>()) {
        return guard_contains(conjunction->a, required) || guard_contains(conjunction->b, required);
    }
    return false;
}

struct ForwardedView {
    tvm::tirx::BufferLoad source;
    tvm::PrimExpr value;
    tvm::ffi::Array<tvm::tirx::PrimVar> axes;
};

// Stage cuts order sibling phases but do not create lexical storage scopes.
// Flatten only these grouping nodes for the dominance audit, not loops or
// conditionals. The rewrite below preserves the original stage attributes.
void sequence_parts(const tvm::tirx::Stmt &statement, luisa::vector<tvm::tirx::Stmt> &parts) {
    if (auto sequence = statement.as<tvm::tirx::SeqStmtNode>()) {
        for (auto &&child : sequence->seq) { sequence_parts(child, parts); }
    } else if (auto attribute = statement.as<tvm::tirx::AttrStmtNode>(); attribute != nullptr && attribute->attr_key == pipeline_stage_annotation) {
        sequence_parts(attribute->body, parts);
    } else {
        parts.emplace_back(statement);
    }
}

class ConsumerBounds final : public tvm::tirx::StmtExprVisitor {
private:
    tvm::tirx::BufferVar _buffer;
    Domain _domain;
    tvm::PrimExpr _guard{tvm::IntImm::Bool(true)};

protected:
    void VisitStmt_(const tvm::tirx::ForNode *loop) final {
        _domain.emplace_back(loop);
        StmtExprVisitor::VisitStmt_(loop);
        _domain.pop_back();
    }
    void VisitExpr_(const tvm::tirx::BufferLoadNode *load) final {
        if (load->buffer.same_as(_buffer)) {
            loads++;
            valid &= !load->predicate && load->indices.size() == _buffer->shape.size();
            if (valid) {
                for (auto i = 0u; i < load->indices.size(); i++) {
                    auto lower = load->indices[i] >= 0;
                    auto upper = load->indices[i] < _buffer->shape[i];
                    valid &= (guard_contains(_guard, lower) || prove_in_loop_domain(lower, _domain) ||
                              prove_in_loop_domain(!_guard || lower, _domain)) &&
                             (guard_contains(_guard, upper) || prove_in_loop_domain(upper, _domain) ||
                              prove_in_loop_domain(!_guard || upper, _domain));
                }
            }
        }
        StmtExprVisitor::VisitExpr_(load);
    }
    void VisitExpr_(const tvm::CallNode *call) final {
        if (!call->op.same_as(tvm::tirx::builtin::if_then_else()) ||
            call->args.size() != 3u) {
            StmtExprVisitor::VisitExpr_(call);
            return;
        }
        auto condition = call->args[0].as_or_throw<tvm::PrimExpr>();
        VisitExpr(condition);
        auto outer_guard = _guard;
        _guard = outer_guard && condition;
        VisitExpr(call->args[1].as_or_throw<tvm::PrimExpr>());
        _guard = outer_guard && !condition;
        VisitExpr(call->args[2].as_or_throw<tvm::PrimExpr>());
        _guard = std::move(outer_guard);
    }

public:
    uint64_t loads{0u};
    bool valid{true};
    ConsumerBounds(tvm::tirx::BufferVar buffer, Domain domain) : _buffer{std::move(buffer)}, _domain{std::move(domain)} {}
};

class ViewAnalysis final : public tvm::tirx::StmtVisitor {
private:
    const InputAccess &_access;
    const luisa::unordered_set<BufferKey> &_inputs;
    Domain _domain;
    bool _preserve_guards;

    [[nodiscard]] std::optional<ForwardedView> _copy(const tvm::tirx::Stmt &statement, const tvm::tirx::BufferVar &buffer) const {
        auto body = statement;
        auto domain = _domain;
        tvm::ffi::Array<tvm::tirx::PrimVar> axes;
        for (auto i = 0u; i < buffer->shape.size(); i++) {
            auto loop = body.as<tvm::tirx::ForNode>();
            if (loop == nullptr) { return {}; }
            auto minimum = loop->min.as<tvm::IntImmNode>();
            auto step = loop->step ? loop->step.value().as<tvm::IntImmNode>() : nullptr;
            if (loop->kind != tvm::tirx::ForKind::kSerial || loop->thread_binding || minimum == nullptr || minimum->value != 0 ||
                (loop->step && (step == nullptr || step->value != 1)) || !tvm::ffi::StructuralEqual{}(loop->extent, buffer->shape[i])) { return {}; }
            if (i == 0u) {
                auto rank = loop->annotations.Get(independent_elements_annotation);
                auto literal = rank ? rank.value().as<tvm::IntImmNode>() : nullptr;
                if (loop->annotations.size() != 1u || literal == nullptr || literal->value != static_cast<int64_t>(buffer->shape.size())) { return {}; }
            } else if (!loop->annotations.empty()) {
                return {};
            }
            axes.push_back(loop->loop_var);
            domain.emplace_back(loop);
            body = loop->body;
        }
        auto store = body.as<tvm::tirx::BufferStoreNode>();
        if (store == nullptr || store->predicate || !store->buffer.same_as(buffer) || store->indices.size() != axes.size()) { return {}; }
        for (auto i = 0u; i < axes.size(); i++) {
            if (!store->indices[i].same_as(axes[i])) { return {}; }
        }
        auto value = store->value;
        tvm::PrimExpr guard = tvm::IntImm::Bool(true);
        if (auto conditional = value.as<tvm::CallNode>(); conditional != nullptr && conditional->op.same_as(tvm::tirx::builtin::if_then_else()) && conditional->args.size() == 3u) {
            guard = conditional->args[0].as_or_throw<tvm::PrimExpr>();
            value = conditional->args[1].as_or_throw<tvm::PrimExpr>();
        }
        auto source = value.as<tvm::tirx::BufferLoadNode>();
        if (source == nullptr || source->predicate || !_inputs.contains(source->buffer.get()) || !compact_buffer(source->buffer)) { return {}; }
        auto source_bounds = in_bounds(source->buffer, source->indices);
        auto unconditional = prove_in_loop_domain(guard && source_bounds, domain);
        auto expression = unconditional ? value : store->value;
        if (!unconditional) {
            // Keep the original lazy conditional and fill value; do not turn
            // a padded Tile into an unguarded pointer view. Source, predicate,
            // fill, and address calculations must all remain immutable until
            // every consumer. A mutable indirect index/guard is not safe just
            // because the primary tensor itself is read-only.
            if (!_preserve_guards || (!guard_contains(guard, source_bounds) && !prove_in_loop_domain(!guard || source_bounds, domain))) { return {}; }
        }
        auto immutable = true;
        tvm::tirx::PostOrderVisit(expression, [&](const tvm::ffi::ObjectRef &node) {
            if (auto load = node.as<tvm::tirx::BufferLoadNode>()) { immutable &= _inputs.contains(load->buffer.get()); }
        });
        if (!immutable) { return {}; }
        return ForwardedView{tvm::ffi::GetRef<tvm::tirx::BufferLoad>(source), std::move(expression), std::move(axes)};
    }

protected:
    void VisitStmt_(const tvm::tirx::ForNode *loop) final {
        _domain.emplace_back(loop);
        StmtVisitor::VisitStmt_(loop);
        _domain.pop_back();
    }
    void VisitStmt_(const tvm::tirx::SeqStmtNode *sequence) final {
        luisa::vector<tvm::tirx::Stmt> parts;
        sequence_parts(tvm::ffi::GetRef<tvm::tirx::SeqStmt>(sequence), parts);
        for (auto i = 0u; i < parts.size(); i++) {
            auto allocation = parts[i].as<tvm::tirx::AllocBufferNode>();
            if (allocation == nullptr || !allocation->annotations.empty()) { continue; }
            auto buffer = allocation->buffer;
            if (buffer.scope() != "local" || !compact_buffer(buffer) || views.contains(buffer.get())) { continue; }
            auto &access = _access.buffers.at(buffer.get());
            if (access.allocations != 1u || access.stores != 1u || access.loads == 0u || access.escapes) { continue; }
            for (auto j = i + 1u; j < parts.size(); j++) {
                auto copy = _copy(parts[j], buffer);
                if (!copy) { continue; }
                ConsumerBounds consumers{buffer, _domain};
                for (auto k = j + 1u; k < parts.size(); k++) { consumers(parts[k]); }
                // Count occurrences, not just node identities: a shared Expr
                // node may also occur before initialization or outside here.
                if (consumers.valid && consumers.loads == access.loads) {
                    views.emplace(buffer.get(), std::move(*copy));
                    removed.emplace(allocation);
                    removed.emplace(parts[j].get());
                }
                break;
            }
        }
        StmtVisitor::VisitStmt_(sequence);
    }

public:
    luisa::unordered_map<BufferKey, ForwardedView> views;
    luisa::unordered_set<const tvm::tirx::StmtNode *> removed;
    ViewAnalysis(const InputAccess &access, const luisa::unordered_set<BufferKey> &inputs, bool preserve_guards)
        : _access{access}, _inputs{inputs}, _preserve_guards{preserve_guards} {}
};

class ViewRewriter final : public tvm::tirx::StmtExprMutator {
private:
    const ViewAnalysis &_analysis;

protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt(const tvm::tirx::Stmt &statement) final {
        if (_analysis.removed.contains(statement.get())) { return tvm::tirx::Evaluate{tvm::IntImm::Int32(0)}; }
        return StmtExprMutator::VisitStmt(statement);
    }
    [[nodiscard]] tvm::Expr VisitExpr_(const tvm::tirx::BufferLoadNode *load) final {
        if (auto iter = _analysis.views.find(load->buffer.get()); iter != _analysis.views.end()) {
            auto &view = iter->second;
            tvm::ffi::Map<tvm::tirx::Var, tvm::Expr> coordinates;
            for (auto i = 0u; i < view.axes.size(); i++) { coordinates.Set(view.axes[i], VisitPrimExpr(load->indices[i])); }
            return tvm::tirx::Substitute(view.value, coordinates);
        }
        return StmtExprMutator::VisitExpr_(load);
    }

public:
    explicit ViewRewriter(const ViewAnalysis &analysis) : _analysis{analysis} {}
};

}// namespace

ReadonlyViews forward_readonly_tile_loads(const tvm::tirx::PrimFunc &function, bool noalias, bool preserve_guards) {
    ReadonlyViews result{function->body, {}};
    if (!noalias) { return result; }
    auto current = function;
    luisa::unordered_set<BufferKey> forwarded_inputs;
    for (;;) {
        InputAccess access;
        access(current->body);
        if (access.opaque) { break; }
        luisa::unordered_set<BufferKey> inputs;
        for (auto &&parameter : current->params) {
            if (parameter->ty.as<tvm::tirx::BufferTypeNode>() == nullptr) { continue; }
            auto buffer = tvm::tirx::BufferVar{parameter};
            auto iter = access.buffers.find(buffer.get());
            if (buffer.scope() == "global" && compact_buffer(buffer) && iter != access.buffers.end() &&
                iter->second.allocations == 0u && iter->second.stores == 0u && !iter->second.escapes) {
                inputs.emplace(buffer.get());
            }
        }
        ViewAnalysis analysis{access, inputs, preserve_guards};
        analysis(current->body);
        if (analysis.views.empty()) { break; }
        for (auto &&[buffer, view] : analysis.views) { forwarded_inputs.emplace(view.source->buffer.get()); }
        auto body = ViewRewriter{analysis}(current->body);
        current.CopyOnWrite()->body = std::move(body);
        // Axis relabeling and other value copies may expose another complete
        // immutable-input snapshot. Recheck all effects, bounds and dominance
        // on the new body; each successful round removes at least one unique
        // allocation, so this fixed point is bounded by the original IR size.
    }
    for (auto &&parameter : function->params) {
        if (forwarded_inputs.contains(parameter.get())) { result.inputs.emplace_back(tvm::tirx::BufferVar{parameter}); }
    }
    result.body = current->body;
    return result;
}

}// namespace luisa::compute::tile::bridge::tirx::detail
