#include <luisa/coro/coro_graph.h>
#include <luisa/ast/function_builder.h>
#include <luisa/ast/op.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/memory.h>
#include <luisa/dsl/coro_frame.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_materialize.h>
#include <luisa/xir/passes/coro_split.h>

namespace luisa::compute::coro {

namespace {

static void append_unique(luisa::vector<size_t> &fields, size_t field) noexcept {
    if (std::find(fields.begin(), fields.end(), field) == fields.end()) {
        fields.emplace_back(field);
    }
}

static void append_reserved_fields(luisa::vector<size_t> &fields) noexcept {
    for (auto i = 0u; i < CoroFrameDesc::reserved_field_count; i++) {
        append_unique(fields, i);
    }
}

[[nodiscard]] static const Type *projected_child_type(
    const Type *type, uint32_t index) noexcept {
    LUISA_ASSERT(type != nullptr,
                 "Cannot project a null coroutine binding type.");
    switch (type->tag()) {
        case Type::Tag::STRUCTURE: {
            auto members = type->members();
            LUISA_ASSERT(index < members.size(),
                         "Coroutine binding structure access is out of range.");
            return members[index];
        }
        case Type::Tag::ARRAY:
        case Type::Tag::VECTOR:
            LUISA_ASSERT(index < type->dimension(),
                         "Coroutine binding aggregate access is out of range.");
            return type->element();
        case Type::Tag::MATRIX:
            LUISA_ASSERT(index < type->dimension(),
                         "Coroutine binding matrix access is out of range.");
            return Type::vector(type->element(), type->dimension());
        default:
            LUISA_ERROR_WITH_LOCATION(
                "Cannot project scalar coroutine binding type '{}'.",
                type->description());
    }
}

[[nodiscard]] static const Expression *project_static_access(
    detail::FunctionBuilder *builder, const Type *root_type,
    const Expression *root,
    luisa::span<const uint32_t> access_chain) noexcept {
    auto *type = root_type;
    auto *expression = root;
    for (auto index : access_chain) {
        auto *child = projected_child_type(type, index);
        if (type->tag() == Type::Tag::STRUCTURE) {
            expression = builder->member(child, expression, index);
        } else {
            expression = builder->access(
                child, expression,
                builder->literal(Type::of<uint>(), index));
        }
        type = child;
    }
    return expression;
}

[[nodiscard]] static luisa::vector<CoroSuspendExtensionPtr>
clone_extensions(const xir::CoroSuspendExtensionOwner &owner) noexcept {
    luisa::vector<CoroSuspendExtensionPtr> extensions;
    extensions.reserve(owner.extensions.size());
    for (auto &&extension : owner.extensions) {
        extensions.emplace_back(
            extension == nullptr ? nullptr : extension->clone());
    }
    return extensions;
}

}// namespace

const Expression *CoroSlotAccess::_read(CoroFrame &frame) const noexcept {
    LUISA_ASSERT(materialized(),
                 "Coroutine binding has no materialized frame access.");
    LUISA_ASSERT(frame.desc() != nullptr && _type != nullptr,
                 "Coroutine binding read has no frame/type metadata.");
    auto *builder = detail::FunctionBuilder::current();
    auto *result = builder->local(_type);
    builder->assign(result, builder->call(_type, CallOp::ZERO, {}));
    for (auto &piece : _pieces) {
        LUISA_ASSERT(
            piece.field_index < frame.desc()->frame_field_count() &&
                frame.desc()->frame_field_type(piece.field_index) ==
                    piece.physical_type,
            "Coroutine binding frame field {} does not match its access plan.",
            piece.field_index);
        auto *field = builder->member(
            piece.physical_type, frame.expression(), piece.field_index);
        const Expression *logical = field;
        if (piece.bit_offset) {
            LUISA_ASSERT(piece.logical_type == Type::of<bool>() &&
                             piece.physical_type == Type::of<uint>() &&
                             *piece.bit_offset < 32u,
                         "Invalid packed-boolean coroutine binding piece.");
            auto mask_value = uint{1u} << *piece.bit_offset;
            auto *mask = builder->literal(
                Type::of<uint>(), mask_value);
            auto *zero = builder->literal(Type::of<uint>(), uint{0u});
            auto *masked = builder->binary(
                Type::of<uint>(), BinaryOp::BIT_AND, field, mask);
            logical = builder->binary(
                Type::of<bool>(), BinaryOp::NOT_EQUAL, masked, zero);
        }
        auto *destination = project_static_access(
            builder, _type, result, piece.access_chain);
        LUISA_ASSERT(destination->type() == piece.logical_type &&
                         logical->type() == piece.logical_type,
                     "Coroutine binding read projection type mismatch.");
        builder->assign(destination, logical);
    }
    return result;
}

void CoroSlotAccess::_write(
    CoroFrame &frame, const Expression *value) const noexcept {
    LUISA_ASSERT(materialized(),
                 "Coroutine binding has no materialized frame access.");
    LUISA_ASSERT(frame.desc() != nullptr && value != nullptr &&
                     value->type() == _type,
                 "Coroutine binding write has invalid frame/value metadata.");
    auto *builder = detail::FunctionBuilder::current();
    for (auto &piece : _pieces) {
        LUISA_ASSERT(
            piece.field_index < frame.desc()->frame_field_count() &&
                frame.desc()->frame_field_type(piece.field_index) ==
                    piece.physical_type,
            "Coroutine binding frame field {} does not match its access plan.",
            piece.field_index);
        auto *field = builder->member(
            piece.physical_type, frame.expression(), piece.field_index);
        auto *logical = project_static_access(
            builder, _type, value, piece.access_chain);
        LUISA_ASSERT(logical->type() == piece.logical_type,
                     "Coroutine binding write projection type mismatch.");
        if (piece.bit_offset) {
            LUISA_ASSERT(piece.logical_type == Type::of<bool>() &&
                             piece.physical_type == Type::of<uint>() &&
                             *piece.bit_offset < 32u,
                         "Invalid packed-boolean coroutine binding piece.");
            auto mask_value = uint{1u} << *piece.bit_offset;
            auto clear_value = ~mask_value;
            auto *mask = builder->literal(
                Type::of<uint>(), mask_value);
            auto *clear = builder->literal(
                Type::of<uint>(), clear_value);
            auto *zero = builder->literal(Type::of<uint>(), uint{0u});
            auto *cleared = builder->binary(
                Type::of<uint>(), BinaryOp::BIT_AND, field, clear);
            auto *encoded = builder->call(
                Type::of<uint>(), CallOp::SELECT,
                {zero, mask, logical});
            auto *merged = builder->binary(
                Type::of<uint>(), BinaryOp::BIT_OR, cleared, encoded);
            builder->assign(field, merged);
        } else {
            builder->assign(field, logical);
        }
    }
}

[[nodiscard]] size_t CoroGraph::node_count() const noexcept {
    return _nodes.size();
}

[[nodiscard]] const CoroGraph::Node &CoroGraph::node(size_t index) const noexcept {
    return _nodes[index];
}

[[nodiscard]] const CoroGraph::Node *CoroGraph::node_by_token(size_t token) const noexcept {
    auto it = _token_to_index.find(token);
    if (it == _token_to_index.end()) { return nullptr; }
    return &_nodes[it->second];
}

[[nodiscard]] const CoroGraph::Node *CoroGraph::node_by_name(luisa::string_view name) const noexcept {
    auto it = _name_to_index.find(luisa::string{name});
    if (it == _name_to_index.end()) { return nullptr; }
    return &_nodes[it->second];
}

[[nodiscard]] size_t CoroGraph::edge_count() const noexcept {
    return _edges.size();
}

[[nodiscard]] const CoroGraph::Edge *CoroGraph::edge(size_t from, size_t to) const noexcept {
    for (auto &e : _edges) {
        if (e.from_index == from && e.to_index == to) { return &e; }
    }
    return nullptr;
}

[[nodiscard]] luisa::string CoroGraph::dump() const noexcept {
    luisa::string s;
    for (auto &node : _nodes) {
        auto name = node.name.empty() ? luisa::string{"<entry>"} : node.name;
        s.append(luisa::format("Node {} '{}' token={} terminal={}\n",
                               node.index, name, node.token, node.is_terminal));
        s.append(luisa::format("  Input Fields: {}\n", node.input_fields));
        s.append(luisa::format("  Output Fields: {}\n", node.output_fields));
        s.append(luisa::format("  Relocation Fields: {}\n", node.relocation_fields));
        s.append(luisa::format("  Transition Targets: {}\n", node.targets));
    }
    for (auto &edge : _edges) {
        s.append(luisa::format("Edge {} -> {} load={} store={}\n",
                               edge.from_index, edge.to_index,
                               edge.load_fields, edge.store_fields));
    }
    for (auto &boundary : _boundaries) {
        s.append(luisa::format(
            "Boundary {}: {} -> {} token={} extensions={} bindings={}\n",
            boundary.index, boundary.from_index, boundary.to_index,
            boundary.token, boundary.extensions.size(),
            boundary.bindings.size()));
        for (size_t i = 0u; i < boundary.extensions.size(); ++i) {
            auto &&extension = boundary.extensions[i];
            s.append(luisa::format(
                "  Extension {}: schema='{}' version={} annotation={}\n",
                i, extension == nullptr ? "<null>" : extension->schema(),
                extension == nullptr ? 0u : extension->version(),
                extension != nullptr && extension->is_annotation()));
        }
    }
    return s;
}

[[nodiscard]] CoroGraph CoroGraph::from_module(
    xir::Module &m, const xir::CoroMaterializeInfo &info,
    const xir::CoroCfgDistillResult &cfg) noexcept {

    static_cast<void>(m);
    xir::CoroSplitInfo split;
    return from_module(m, info, cfg, split);
}

[[nodiscard]] CoroGraph CoroGraph::from_module(
    xir::Module &m, const xir::CoroMaterializeInfo &info,
    const xir::CoroCfgDistillResult &cfg,
    const xir::CoroSplitInfo &split) noexcept {

    static_cast<void>(m);
    CoroGraph graph;

    luisa::vector<const xir::CallableFunction *> callables(cfg.scopes.size(), nullptr);
    for (auto &subroutine : split.subroutines) {
        LUISA_ASSERT(
            subroutine.scope_index < callables.size() &&
                callables[subroutine.scope_index] == nullptr &&
                subroutine.callable != nullptr &&
                subroutine.trigger_token ==
                    cfg.scopes[subroutine.scope_index].trigger_token,
            "CoroGraph received inconsistent split metadata for scope {}.",
            subroutine.scope_index);
        callables[subroutine.scope_index] = subroutine.callable;
    }
    if (!split.subroutines.empty()) {
        LUISA_ASSERT(
            split.subroutines.size() == cfg.scopes.size(),
            "CoroGraph received {} callable(s) for {} scope(s).",
            split.subroutines.size(), cfg.scopes.size());
        for (size_t scope_index = 0u;
             scope_index < callables.size(); ++scope_index) {
            LUISA_ASSERT(
                callables[scope_index] != nullptr,
                "CoroGraph is missing the callable for scope {}.",
                scope_index);
        }
    }

    // --- Build nodes from cfg-distill scopes ---
    for (size_t i = 0u; i < cfg.scopes.size(); ++i) {
        auto &scope = cfg.scopes[i];
        Node node;
        node.index = i;
        node.is_terminal = scope.is_terminal;
        node.callable = (i < callables.size()) ? callables[i] : nullptr;
        node.token = scope.trigger_token;
        node.name = scope.trigger_name.has_value() ? *scope.trigger_name : luisa::string{};

        graph._nodes.push_back(std::move(node));

        // Build lookup maps (use the stored node, not the moved-from local)
        auto &stored = graph._nodes.back();
        auto [_, token_inserted] =
            graph._token_to_index.emplace(stored.token, i);
        LUISA_ASSERT(token_inserted,
                     "CoroGraph received duplicate trigger token {}.",
                     stored.token);
        if (!stored.name.empty()) {
            graph._name_to_index.emplace(stored.name, i);
        }
    }

    for (auto &te : info.edges) {
        Edge *edge_ptr = nullptr;
        for (auto &edge : graph._edges) {
            if (edge.from_index == te.from_scope && edge.to_index == te.to_scope) {
                edge_ptr = &edge;
                break;
            }
        }
        if (edge_ptr == nullptr) {
            auto &edge = graph._edges.emplace_back();
            edge.from_index = te.from_scope;
            edge.to_index = te.to_scope;
            edge_ptr = &edge;
        }
        for (auto field : te.load_fields) {
            append_unique(edge_ptr->load_fields, field);
        }
        for (auto field : te.store_fields) {
            append_unique(edge_ptr->store_fields, field);
        }
    }

    // Preserve every static suspend boundary independently. Edges above are
    // allowed to merge load/store transport between the same scopes; semantic
    // Extension objects are not. Each owner binding is projected through the
    // already-colored logical frame values into a typed CoroSlotAccess.
    for (auto &transition : cfg.transition_edges) {
        if (!transition.is_suspend) { continue; }
        LUISA_ASSERT(
            transition.extension_owner.binding_values.size() ==
                    transition.extension_binding_frame_value_indices.size() &&
                transition.extension_owner.binding_values.size() ==
                    transition.extension_binding_access_chains.size(),
            "Coroutine extension binding projection count mismatch.");
        auto &boundary = graph._boundaries.emplace_back();
        boundary.index = graph._boundaries.size() - 1u;
        boundary.from_index = transition.from_scope;
        boundary.to_index = transition.to_scope;
        boundary.token = transition.token;
        boundary.extensions = clone_extensions(
            transition.extension_owner);

        auto binding_count =
            transition.extension_owner.binding_values.size();
        luisa::vector<const CoroSuspendBinding *> descriptors(
            binding_count, nullptr);
        for (auto &&extension :
             transition.extension_owner.extensions) {
            LUISA_ASSERT(extension != nullptr,
                         "Coroutine boundary contains a null Extension.");
            for (auto &&binding : extension->bindings()) {
                LUISA_ASSERT(binding.index < descriptors.size() &&
                                 descriptors[binding.index] == nullptr,
                             "Coroutine boundary contains an invalid or "
                             "duplicate binding index {}.",
                             binding.index);
                descriptors[binding.index] = &binding;
            }
        }
        boundary.bindings.reserve(binding_count);
        for (size_t binding_index = 0u;
             binding_index < binding_count; ++binding_index) {
            auto *descriptor = descriptors[binding_index];
            auto *binding_value =
                transition.extension_owner.binding_values[binding_index];
            LUISA_ASSERT(descriptor != nullptr && binding_value != nullptr &&
                             binding_value->type() != nullptr,
                         "Coroutine boundary binding {} is incomplete.",
                         binding_index);
            auto &base =
                transition.extension_binding_access_chains[binding_index];
            luisa::vector<CoroSlotAccess::Piece> pieces;
            for (auto frame_value_index :
                 transition.extension_binding_frame_value_indices[
                     binding_index]) {
                LUISA_ASSERT(
                    frame_value_index < cfg.frame_values.size(),
                    "Coroutine boundary binding {} references out-of-range "
                    "frame value {}.",
                    binding_index, frame_value_index);
                auto &frame_value = cfg.frame_values[frame_value_index];
                LUISA_ASSERT(
                    frame_value.slot < cfg.frame_slots.size() &&
                        base.size() <= frame_value.access_chain.size() &&
                        std::equal(base.begin(), base.end(),
                                   frame_value.access_chain.begin()),
                    "Coroutine boundary binding {} has an invalid frame "
                    "projection.",
                    binding_index);
                auto relative = luisa::vector<uint32_t>{
                    frame_value.access_chain.begin() + base.size(),
                    frame_value.access_chain.end()};
                pieces.emplace_back(CoroSlotAccess::Piece{
                    .frame_value_index = frame_value_index,
                    .field_index = CoroFrameDesc::reserved_field_count +
                                   frame_value.slot,
                    .access_chain = std::move(relative),
                    .logical_type = frame_value.type,
                    .physical_type =
                        cfg.frame_slots[frame_value.slot].type,
                    .bit_offset = frame_value.bit_offset});
            }
            LUISA_ASSERT(
                descriptor->lifetime ==
                        CoroSuspendBindingLifetime::boundary ||
                    !pieces.empty(),
                "Queued/resumed coroutine binding {} has no frame access.",
                binding_index);
            boundary.bindings.emplace_back(CoroSlotAccess{
                binding_value->type(), descriptor->access,
                descriptor->lifetime, std::move(pieces)});
        }
    }

    for (auto &node : graph._nodes) {
        append_reserved_fields(node.input_fields);
        append_reserved_fields(node.output_fields);
    }

    // cfg-distill has already solved the backward may-liveness equation
    //
    //   L(s) = External(s) union U_(s -> t) (L(t) - K(s -> t))
    //
    // to its least fixed point. Every transition's live values are exactly
    // L(target), so project that existing analysis certificate through the
    // interference-colored physical-slot map instead of reconstructing an
    // approximation from materialized load/store fields. In particular,
    // load_fields only denotes values evaluated by the immediate callable;
    // it deliberately omits dormant values that remain resident until a
    // later continuation.
    luisa::vector<uint8_t> has_relocation_certificate(graph._nodes.size(), 0u);
    for (auto &transition : cfg.transition_edges) {
        LUISA_ASSERT(
            transition.to_scope < graph._nodes.size(),
            "Coroutine transition {} -> {} has an out-of-range target.",
            transition.from_scope, transition.to_scope);
        auto projected = luisa::vector<size_t>{};
        append_reserved_fields(projected);
        for (auto frame_value_index :
             transition.live_frame_value_indices) {
            LUISA_ASSERT(
                frame_value_index < cfg.frame_values.size(),
                "Coroutine transition {} -> {} references out-of-range "
                "frame value {} (count {}).",
                transition.from_scope, transition.to_scope,
                frame_value_index, cfg.frame_values.size());
            auto slot = cfg.frame_values[frame_value_index].slot;
            LUISA_ASSERT(
                slot < cfg.frame_slots.size(),
                "Coroutine frame value {} references out-of-range slot {} "
                "(count {}).",
                frame_value_index, slot, cfg.frame_slots.size());
            append_unique(
                projected,
                CoroFrameDesc::reserved_field_count + slot);
        }
        luisa::sort(projected.begin(), projected.end());
        auto &target = graph._nodes[transition.to_scope];
        if (has_relocation_certificate[transition.to_scope] == 0u) {
            target.relocation_fields = std::move(projected);
            has_relocation_certificate[transition.to_scope] = 1u;
        } else {
            // Ordinary L(target) is identical for every predecessor, while
            // boundary-local Extension operands form a tagged union. Preserve
            // the union of their already-colored physical fields; this adds no
            // slots and lets one token queue relocate frames from all legal
            // incoming suspension sites.
            for (auto field : projected) {
                append_unique(target.relocation_fields, field);
            }
            luisa::sort(target.relocation_fields.begin(),
                        target.relocation_fields.end());
        }
    }
    for (auto &edge : graph._edges) {
        if (edge.from_index < graph._nodes.size()) {
            auto &from_node = graph._nodes[edge.from_index];
            append_unique(from_node.targets, edge.to_index);
            for (auto field : edge.store_fields) {
                append_unique(from_node.output_fields, field);
            }
        }
        if (edge.to_index < graph._nodes.size()) {
            auto &to_node = graph._nodes[edge.to_index];
            for (auto field : edge.load_fields) {
                append_unique(to_node.input_fields, field);
            }
        }
    }
    for (auto &node : graph._nodes) {
        luisa::sort(node.input_fields.begin(), node.input_fields.end());
        luisa::sort(node.output_fields.begin(), node.output_fields.end());
        luisa::sort(node.relocation_fields.begin(), node.relocation_fields.end());
        luisa::sort(node.targets.begin(), node.targets.end());
    }

    return graph;
}

}// namespace luisa::compute::coro
