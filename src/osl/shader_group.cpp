#include <algorithm>
#include <array>
#include <utility>

#include <luisa/core/stl/format.h>
#include <luisa/core/stl/queue.h>
#include <luisa/osl/shader_group.h>
#include "translator_impl.h"

namespace luisa::compute::osl {

ParameterValue ParameterValue::from_osl_row_major(const std::array<float, 16> &m) noexcept {
    return ParameterValue{detail::osl_row_major_to_xir_matrix(m.data())};
}

uint32_t ShaderGroup::add_layer(luisa::string name, const Shader *shader) noexcept {
    auto idx = static_cast<uint32_t>(_layers.size());
    _layers.emplace_back(ShaderLayer{
        .name = std::move(name),
        .shader = shader,
        .overrides = {}});
    return idx;
}

uint32_t ShaderGroup::add_layer(const Shader *shader) noexcept {
    auto name = shader != nullptr ?
                    luisa::format("{}_{}", shader->identifier(), _layers.size()) :
                    luisa::format("layer_{}", _layers.size());
    return add_layer(std::move(name), shader);
}

void ShaderGroup::set_parameter(uint32_t layer, luisa::string_view param_name, ParameterValue value) noexcept {
    if (layer >= _layers.size()) { return; }
    _layers[layer].overrides.insert_or_assign(luisa::string{param_name}, std::move(value));
}

void ShaderGroup::connect_shaders(uint32_t src_layer, luisa::string_view src_param,
                                  uint32_t dst_layer, luisa::string_view dst_param) noexcept {
    _connections.emplace_back(ShaderConnection{
        .src_layer = src_layer,
        .src_param = luisa::string{src_param},
        .dst_layer = dst_layer,
        .dst_param = luisa::string{dst_param}});
}

void ShaderGroup::mark_output(uint32_t layer, luisa::string_view param_name, bool want_deriv) noexcept {
    _output_requests.emplace_back(ShaderOutputRequest{
        .layer = layer,
        .param = luisa::string{param_name},
        .want_deriv = want_deriv});
}

void ShaderGroup::set_root_layer(uint32_t layer) noexcept {
    _root_layer = layer;
}

luisa::optional<uint32_t> ShaderGroup::effective_root_layer() const noexcept {
    if (_layers.empty()) { return luisa::nullopt; }
    if (_root_layer.has_value()) { return _root_layer; }
    return static_cast<uint32_t>(_layers.size() - 1u);
}

namespace detail {

namespace {

[[nodiscard]] const Symbol *find_shader_symbol(const Shader &shader, luisa::string_view name) noexcept {
    for (auto &&sym : shader.symbols()) {
        if (sym->identifier() == name) {
            return sym.get();
        }
    }
    return nullptr;
}

[[nodiscard]] bool is_param_symbol(const Symbol *sym) noexcept {
    if (sym == nullptr) { return false; }
    auto tag = sym->root()->tag();
    return tag == Symbol::Tag::SYM_PARAM || tag == Symbol::Tag::SYM_OUTPUT_PARAM;
}

[[nodiscard]] bool is_output_param_symbol(const Symbol *sym) noexcept {
    if (sym == nullptr) { return false; }
    return sym->root()->tag() == Symbol::Tag::SYM_OUTPUT_PARAM;
}

[[nodiscard]] bool types_connect_compatible(const Type *src, const Type *dst) noexcept {
    if (src == nullptr || dst == nullptr) { return false; }
    if (src == dst) { return true; }
    if (src->tag() != dst->tag()) { return false; }
    if (src->tag() == Type::Tag::CLOSURE) {
        return true;
    }
    if (src->tag() == Type::Tag::STRUCT) {
        auto *s1 = static_cast<const StructType *>(src);
        auto *s2 = static_cast<const StructType *>(dst);
        if (s1->fields().size() != s2->fields().size()) { return false; }
        for (size_t i = 0u; i < s1->fields().size(); ++i) {
            if (s1->fields()[i].array_length != s2->fields()[i].array_length ||
                !types_connect_compatible(s1->fields()[i].type, s2->fields()[i].type)) {
                return false;
            }
        }
        return true;
    }
    auto p1 = static_cast<const SimpleType *>(src)->primitive();
    auto p2 = static_cast<const SimpleType *>(dst)->primitive();
    if (p1 == p2) { return true; }
    auto is_scalar_num = [](SimpleType::Primitive p) noexcept {
        return p == SimpleType::Primitive::INT || p == SimpleType::Primitive::FLOAT;
    };
    auto is_triple = [](SimpleType::Primitive p) noexcept {
        return p == SimpleType::Primitive::POINT ||
               p == SimpleType::Primitive::VECTOR ||
               p == SimpleType::Primitive::NORMAL ||
               p == SimpleType::Primitive::COLOR;
    };
    if (is_scalar_num(p1) && is_scalar_num(p2)) { return true; }
    if (is_scalar_num(p1) && is_triple(p2)) { return true; }
    if (is_triple(p1) && is_triple(p2)) { return true; }
    if (is_scalar_num(p1) && p2 == SimpleType::Primitive::MATRIX) { return true; }
    return false;
}

[[nodiscard]] bool override_compatible_with_symbol(const Symbol *sym, const ParameterValue &val) noexcept {
    if (sym == nullptr || sym->type() == nullptr) { return false; }
    auto *t = sym->type();
    if (t->tag() != Type::Tag::SIMPLE) { return false; }
    auto prim = static_cast<const SimpleType *>(t)->primitive();
    if (!sym->is_array()) {
        switch (prim) {
            case SimpleType::Primitive::INT:
                return luisa::holds_alternative<int32_t>(val.value) ||
                       luisa::holds_alternative<float>(val.value);
            case SimpleType::Primitive::FLOAT:
                return luisa::holds_alternative<float>(val.value) ||
                       luisa::holds_alternative<int32_t>(val.value);
            case SimpleType::Primitive::POINT:
            case SimpleType::Primitive::NORMAL:
            case SimpleType::Primitive::VECTOR:
            case SimpleType::Primitive::COLOR:
                return luisa::holds_alternative<luisa::float3>(val.value) ||
                       luisa::holds_alternative<float>(val.value) ||
                       luisa::holds_alternative<int32_t>(val.value);
            case SimpleType::Primitive::MATRIX:
                return luisa::holds_alternative<luisa::float4x4>(val.value) ||
                       luisa::holds_alternative<float>(val.value) ||
                       luisa::holds_alternative<int32_t>(val.value);
            case SimpleType::Primitive::STRING:
                return luisa::holds_alternative<luisa::string>(val.value);
        }
        return false;
    }
    switch (prim) {
        case SimpleType::Primitive::INT:
            return luisa::holds_alternative<luisa::vector<int32_t>>(val.value);
        case SimpleType::Primitive::FLOAT:
            return luisa::holds_alternative<luisa::vector<float>>(val.value) ||
                   luisa::holds_alternative<luisa::vector<int32_t>>(val.value);
        case SimpleType::Primitive::POINT:
        case SimpleType::Primitive::NORMAL:
        case SimpleType::Primitive::VECTOR:
        case SimpleType::Primitive::COLOR:
            return luisa::holds_alternative<luisa::vector<luisa::float3>>(val.value);
        case SimpleType::Primitive::MATRIX:
            return luisa::holds_alternative<luisa::vector<luisa::float4x4>>(val.value);
        case SimpleType::Primitive::STRING:
            return luisa::holds_alternative<luisa::vector<luisa::string>>(val.value);
    }
    return false;
}

[[nodiscard]] int override_array_size(const ParameterValue &val) noexcept {
    return luisa::visit(
        [](auto &&v) noexcept -> int {
            using T = std::remove_cvref_t<decltype(v)>;
            if constexpr (std::is_same_v<T, luisa::vector<int32_t>> ||
                          std::is_same_v<T, luisa::vector<float>> ||
                          std::is_same_v<T, luisa::vector<luisa::float3>> ||
                          std::is_same_v<T, luisa::vector<luisa::float4x4>> ||
                          std::is_same_v<T, luisa::vector<luisa::string>>) {
                return static_cast<int>(v.size());
            } else {
                return 0;
            }
        },
        val.value);
}

[[nodiscard]] bool shader_writes_globals_or_messages(const Shader &shader) noexcept {
    for (auto &&inst : shader.instructions()) {
        if (inst->opcode() == "setmessage") {
            return true;
        }
        auto rw = classify_inst_rw(*inst);
        const auto args = inst->args();
        for (size_t i = 0u; i < args.size(); ++i) {
            if (!rw.writes_arg(i)) { continue; }
            const Symbol *s = args[i]->root();
            if (s->tag() == Symbol::Tag::SYM_GLOBAL) {
                if (auto gs = parse_global_symbol(s->identifier())) {
                    if (is_global_writable(*gs)) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

}// namespace

bool link_shader_group(
    const ShaderGroup &group,
    bool prune_dead_layers,
    LinkedGroup &out_group,
    luisa::vector<TranslationError> &errors) noexcept {
    out_group = {};
    auto add_error = [&](luisa::string_view layer, luisa::string msg) {
        errors.emplace_back(TranslationError{
            .layer_name = luisa::string{layer},
            .message = std::move(msg)});
    };

    if (group.layers().empty()) {
        add_error({}, "ShaderGroup contains no layers.");
        return false;
    }

    const auto num_layers = static_cast<uint32_t>(group.layers().size());
    if (auto root = group.effective_root_layer(); !root.has_value() || *root >= num_layers) {
        add_error({}, "ShaderGroup root layer index is out of range.");
        return false;
    }

    for (uint32_t i = 0u; i < num_layers; ++i) {
        const auto &layer = group.layers()[i];
        if (layer.shader == nullptr) {
            add_error(layer.name, "ShaderLayer has null Shader pointer.");
            return false;
        }
        for (const auto &[pname, pval] : layer.overrides) {
            const Symbol *sym = find_shader_symbol(*layer.shader, pname);
            if (sym == nullptr) {
                add_error(layer.name, luisa::format("Parameter override '{}' not found in shader '{}'.",
                                                    pname, layer.shader->identifier()));
                return false;
            }
            if (!is_param_symbol(sym)) {
                add_error(layer.name, luisa::format("Symbol '{}' in shader '{}' is not a parameter or output parameter.",
                                                    pname, layer.shader->identifier()));
                return false;
            }
            if (!override_compatible_with_symbol(sym, pval)) {
                add_error(layer.name, luisa::format("Parameter override for '{}' has incompatible type.", pname));
                return false;
            }
            if (sym->is_array() && !sym->is_unbounded() &&
                override_array_size(pval) != sym->array_length()) {
                add_error(layer.name, luisa::format("Array parameter override '{}' size mismatch: expected {}, got {}.",
                                                    pname, sym->array_length(), override_array_size(pval)));
                return false;
            }
        }
    }

    // Validate connections & check duplicate destination connections
    luisa::vector<luisa::vector<uint32_t>> adj_rev(num_layers);
    luisa::vector<luisa::vector<uint32_t>> adj_fwd(num_layers);
    luisa::vector<uint32_t> in_degree(num_layers, 0u);
    luisa::vector<luisa::vector<const Symbol *>> dst_connected_per_layer(num_layers);

    struct ValidatedConn {
        uint32_t src_layer;
        const Symbol *src_sym;
        uint32_t dst_layer;
        const Symbol *dst_sym;
    };
    luisa::vector<ValidatedConn> validated_conns;
    validated_conns.reserve(group.connections().size());

    for (const auto &conn : group.connections()) {
        if (conn.src_layer >= num_layers || conn.dst_layer >= num_layers) {
            add_error({}, luisa::format("Connection layer index out of range ({} -> {}).",
                                        conn.src_layer, conn.dst_layer));
            return false;
        }
        if (conn.src_layer == conn.dst_layer) {
            add_error(group.layers()[conn.src_layer].name,
                      "Self-connection within the same ShaderLayer is invalid.");
            return false;
        }
        const auto &src_layer = group.layers()[conn.src_layer];
        const auto &dst_layer = group.layers()[conn.dst_layer];
        const Symbol *src_sym = find_shader_symbol(*src_layer.shader, conn.src_param);
        const Symbol *dst_sym = find_shader_symbol(*dst_layer.shader, conn.dst_param);
        if (src_sym == nullptr) {
            add_error(src_layer.name, luisa::format("Source parameter '{}' not found.", conn.src_param));
            return false;
        }
        if (dst_sym == nullptr) {
            add_error(dst_layer.name, luisa::format("Destination parameter '{}' not found.", conn.dst_param));
            return false;
        }
        if (!is_output_param_symbol(src_sym)) {
            add_error(src_layer.name, luisa::format("Source symbol '{}' in connection must be an output parameter (oparam).",
                                                    conn.src_param));
            return false;
        }
        if (!is_param_symbol(dst_sym) || is_output_param_symbol(dst_sym)) {
            add_error(dst_layer.name, luisa::format("Destination symbol '{}' in connection must be an input parameter (param).",
                                                    conn.dst_param));
            return false;
        }
        if (src_sym->is_array() != dst_sym->is_array()) {
            add_error(dst_layer.name, luisa::format("Array mismatch connecting '{}.{}' to '{}.{}'.",
                                                    src_layer.name, conn.src_param, dst_layer.name, conn.dst_param));
            return false;
        }
        if (src_sym->is_array() && !src_sym->is_unbounded() && !dst_sym->is_unbounded() &&
            src_sym->array_length() != dst_sym->array_length()) {
            add_error(dst_layer.name, luisa::format("Fixed array length mismatch connecting '{}.{}' [{}] to '{}.{}' [{}].",
                                                    src_layer.name, conn.src_param, src_sym->array_length(),
                                                    dst_layer.name, conn.dst_param, dst_sym->array_length()));
            return false;
        }
        if (!types_connect_compatible(src_sym->type(), dst_sym->type())) {
            add_error(dst_layer.name, luisa::format("Incompatible types connecting '{}.{}' ({}) to '{}.{}' ({}).",
                                                    src_layer.name, conn.src_param, src_sym->type()->identifier(),
                                                    dst_layer.name, conn.dst_param, dst_sym->type()->identifier()));
            return false;
        }
        auto is_ancestor = [](const Symbol *anc, const Symbol *desc) noexcept {
            for (const Symbol *p = desc->parent(); p != nullptr; p = p->parent()) {
                if (p == anc) { return true; }
            }
            return false;
        };
        for (const Symbol *prev_dst : dst_connected_per_layer[conn.dst_layer]) {
            if (prev_dst == dst_sym) {
                add_error(dst_layer.name, luisa::format("Destination parameter '{}' has multiple incoming connections.",
                                                        conn.dst_param));
                return false;
            }
            if (is_ancestor(prev_dst, dst_sym) || is_ancestor(dst_sym, prev_dst)) {
                add_error(dst_layer.name, luisa::format(
                    "Conflicting parent and field connections involving '{}.{}' and '{}'.",
                    dst_layer.name, conn.dst_param, prev_dst->identifier()));
                return false;
            }
        }
        dst_connected_per_layer[conn.dst_layer].emplace_back(dst_sym);
        adj_rev[conn.dst_layer].emplace_back(conn.src_layer);
        adj_fwd[conn.src_layer].emplace_back(conn.dst_layer);
        in_degree[conn.dst_layer]++;
        validated_conns.emplace_back(ValidatedConn{
            .src_layer = conn.src_layer,
            .src_sym = src_sym,
            .dst_layer = conn.dst_layer,
            .dst_sym = dst_sym});
    }

    // Cycle detection across whole graph
    {
        luisa::vector<uint32_t> deg = in_degree;
        luisa::queue<uint32_t> q;
        for (uint32_t i = 0u; i < num_layers; ++i) {
            if (deg[i] == 0u) { q.push(i); }
        }
        uint32_t visited = 0u;
        while (!q.empty()) {
            auto u = q.front();
            q.pop();
            visited++;
            for (auto v : adj_fwd[u]) {
                if (--deg[v] == 0u) { q.push(v); }
            }
        }
        if (visited != num_layers) {
            add_error({}, "ShaderGroup contains a cyclic connection dependency.");
            return false;
        }
    }

    // Validate output requests
    for (const auto &req : group.output_requests()) {
        if (req.layer >= num_layers) {
            add_error({}, luisa::format("Output request layer {} is out of range.", req.layer));
            return false;
        }
        const Symbol *sym = find_shader_symbol(*group.layers()[req.layer].shader, req.param);
        if (sym == nullptr) {
            add_error(group.layers()[req.layer].name,
                      luisa::format("Requested output parameter '{}' not found.", req.param));
            return false;
        }
    }

    // Reachability (dead layer pruning)
    luisa::vector<bool> active(num_layers, !prune_dead_layers);
    if (prune_dead_layers) {
        luisa::queue<uint32_t> q;
        auto mark_root = [&](uint32_t idx) {
            if (!active[idx]) {
                active[idx] = true;
                q.push(idx);
            }
        };
        if (auto root = group.effective_root_layer()) {
            mark_root(*root);
        }
        for (const auto &req : group.output_requests()) {
            mark_root(req.layer);
        }
        for (uint32_t i = 0u; i < num_layers; ++i) {
            if (shader_writes_globals_or_messages(*group.layers()[i].shader)) {
                mark_root(i);
            }
        }
        while (!q.empty()) {
            auto dst = q.front();
            q.pop();
            for (auto src : adj_rev[dst]) {
                mark_root(src);
            }
        }
    }

    // Topological sort of active layers (preserving original layer index order as tie-breaker)
    luisa::vector<uint32_t> active_in_deg(num_layers, 0u);
    for (const auto &vc : validated_conns) {
        if (active[vc.src_layer] && active[vc.dst_layer]) {
            active_in_deg[vc.dst_layer]++;
        }
    }
    luisa::vector<uint32_t> ready;
    for (uint32_t i = 0u; i < num_layers; ++i) {
        if (active[i] && active_in_deg[i] == 0u) {
            ready.emplace_back(i);
        }
    }
    luisa::vector<uint32_t> order;
    while (!ready.empty()) {
        std::sort(ready.begin(), ready.end());
        auto u = ready.front();
        ready.erase(ready.begin());
        order.emplace_back(u);
        for (auto v : adj_fwd[u]) {
            if (active[v] && --active_in_deg[v] == 0u) {
                ready.emplace_back(v);
            }
        }
    }

    luisa::vector<uint32_t> group_to_active(num_layers, ~0u);
    out_group.layers.resize(order.size());
    for (uint32_t ai = 0u; ai < static_cast<uint32_t>(order.size()); ++ai) {
        uint32_t gi = order[ai];
        group_to_active[gi] = ai;
        auto &ll = out_group.layers[ai];
        ll.group_layer_index = gi;
        ll.active_index = ai;
        ll.layer = &group.layers()[gi];
        ll.shader = ll.layer->shader;

        for (const auto &[pname, pval] : ll.layer->overrides) {
            const Symbol *sym = find_shader_symbol(*ll.shader, pname);
            ll.param_overrides.emplace(sym, &pval);
            ll.skip_init_params.emplace(sym->root());
        }
    }
    out_group.eliminated_layer_count = num_layers - order.size();

    // Populate active connections
    for (const auto &vc : validated_conns) {
        if (!active[vc.src_layer] || !active[vc.dst_layer]) { continue; }
        uint32_t src_ai = group_to_active[vc.src_layer];
        uint32_t dst_ai = group_to_active[vc.dst_layer];
        auto conn_idx = static_cast<uint32_t>(out_group.connections.size());
        out_group.connections.emplace_back(LinkedConnection{
            .src_layer = src_ai,
            .src_symbol = vc.src_sym,
            .dst_layer = dst_ai,
            .dst_symbol = vc.dst_sym});
        out_group.layers[dst_ai].incoming_connections.emplace_back(conn_idx);
        out_group.layers[dst_ai].skip_init_params.emplace(vc.dst_sym->root());
    }

    // Resolve effective array lengths in topological order (so upstream unbounded lengths propagate)
    for (auto &ll : out_group.layers) {
        for (auto &&sym_ptr : ll.shader->symbols()) {
            const Symbol *sym = sym_ptr.get();
            if (!sym->is_array()) {
                ll.effective_array_length.emplace(sym, 0);
                continue;
            }
            if (!sym->is_unbounded()) {
                ll.effective_array_length.emplace(sym, sym->array_length());
                continue;
            }
            int resolved = 0;
            // 1. Check incoming connection on sym or root
            for (uint32_t ci : ll.incoming_connections) {
                const auto &c = out_group.connections[ci];
                if (c.dst_symbol == sym || c.dst_symbol == sym->root()) {
                    const auto &src_ll = out_group.layers[c.src_layer];
                    auto it = src_ll.effective_array_length.find(c.src_symbol);
                    if (it != src_ll.effective_array_length.end()) {
                        resolved = it->second;
                    }
                    break;
                }
            }
            // 2. Check parameter override
            if (resolved == 0) {
                if (auto it = ll.param_overrides.find(sym); it != ll.param_overrides.end()) {
                    resolved = override_array_size(*it->second);
                }
            }
            // 3. Check .oso initial values (e.g. param color[] colormap 0 0 0 ...)
            if (resolved == 0 && !sym->initial_values().empty()) {
                auto stride = literal_stride(sym->type());
                if (stride > 0u) {
                    resolved = static_cast<int>(sym->initial_values().size() / stride);
                }
            }
            ll.effective_array_length.emplace(sym, resolved);
        }
    }

    // Populate output bindings
    if (!group.output_requests().empty()) {
        for (const auto &req : group.output_requests()) {
            uint32_t ai = group_to_active[req.layer];
            const Symbol *sym = find_shader_symbol(*out_group.layers[ai].shader, req.param);
            out_group.outputs.emplace_back(LinkedOutput{
                .layer_index = ai,
                .symbol = sym,
                .want_deriv = req.want_deriv});
        }
    } else if (auto root = group.effective_root_layer()) {
        uint32_t ai = group_to_active[*root];
        const Shader *root_shader = out_group.layers[ai].shader;
        for (auto &&sym : root_shader->symbols()) {
            if (sym->tag() == Symbol::Tag::SYM_OUTPUT_PARAM && sym->is_root()) {
                out_group.outputs.emplace_back(LinkedOutput{
                    .layer_index = ai,
                    .symbol = sym.get(),
                    .want_deriv = false});
            }
        }
    }

    return true;
}

}// namespace detail

}// namespace luisa::compute::osl
