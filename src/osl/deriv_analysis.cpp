#include <algorithm>

#include "translator_impl.h"

namespace luisa::compute::osl::detail {

namespace {

[[nodiscard]] bool is_zero_deriv_op(luisa::string_view op) noexcept {
    return op == "Dx" || op == "Dy" || op == "Dz" ||
           op == "floor" || op == "ceil" || op == "trunc" || op == "round" ||
           op == "sign" || op == "step" ||
           op == "cellnoise" || op == "hashnoise" || op == "hash" ||
           op == "isnan" || op == "isinf" || op == "isfinite" ||
           op == "eq" || op == "neq" || op == "lt" || op == "gt" || op == "le" || op == "ge" ||
           op == "and" || op == "or" || op == "not" ||
           op == "bitand" || op == "bitor" || op == "xor" || op == "compl" || op == "shl" || op == "shr" ||
           op == "arraylength" || op == "isconnected" || op == "backfacing" || op == "raytype" ||
           op == "strlen" || op == "getchar" || op == "stoi" || op == "stof" || op == "split" ||
           op == "startswith" || op == "endswith" || op == "trace" ||
           op == "determinant" || op == "gettextureinfo" || op == "getmatrix";
}

[[nodiscard]] bool is_deriv_seed_op(luisa::string_view op) noexcept {
    return op == "Dx" || op == "Dy" || op == "area" || op == "filterwidth" || op == "calculatenormal";
}

[[nodiscard]] bool is_sym_differentiable(const Symbol *sym) noexcept {
    return sym != nullptr && is_differentiable_type(sym->type());
}

[[nodiscard]] bool has_source_deriv_hook(luisa::string_view op) noexcept {
    return op == "getattribute" || op == "getmessage" ||
           op == "texture" || op == "texture3d" || op == "environment" ||
           op == "noise" || op == "pnoise" || op == "snoise" || op == "psnoise";
}

void classify_inst_rw_vecs(const Instruction &inst,
                           luisa::vector<bool> &is_write,
                           luisa::vector<bool> &is_read) noexcept {
    const size_t n = inst.args().size();
    is_write.assign(n, false);
    is_read.assign(n, false);
    auto rw = parse_argrw_hint(inst);
    if (rw.has_value() && rw->size() == n) {
        for (size_t i = 0u; i < n; ++i) {
            char c = (*rw)[i];
            if (c == 'w') { is_write[i] = true; }
            else if (c == 'W') { is_write[i] = true; is_read[i] = true; }
            else { is_read[i] = true; }
        }
        return;
    }
    auto irw = classify_inst_rw(inst);
    auto op = inst.opcode();
    if (op == "aassign" || op == "compassign" || op == "mxcompassign") {
        if (n >= 1u) { is_write[0] = true; is_read[0] = true; }
        for (size_t i = 1u; i < n; ++i) { is_read[i] = true; }
        return;
    }
    if (op == "split" && n >= 3u) {
        is_write[0] = true;
        is_read[1] = true;
        is_write[2] = true;
        is_read[2] = true;
        for (size_t i = 3u; i < n; ++i) { is_read[i] = true; }
        return;
    }
    for (size_t i = 0u; i < n; ++i) {
        if (irw.writes_arg(i)) {
            is_write[i] = true;
        } else {
            is_read[i] = true;
        }
    }
}

[[nodiscard]] bool tree_contains_any(
    const luisa::unordered_set<const Symbol *> &set,
    const Symbol *sym) noexcept {
    if (sym == nullptr) { return false; }
    if (sym->children().empty()) {
        return set.contains(sym);
    }
    for (const Symbol *c : sym->children()) {
        if (tree_contains_any(set, c)) { return true; }
    }
    return false;
}

void mark_struct_tree(luisa::unordered_set<const Symbol *> &set, const Symbol *sym, bool &changed) noexcept {
    if (sym == nullptr) { return; }
    if (sym->children().empty()) {
        if (is_sym_differentiable(sym) && set.emplace(sym).second) {
            changed = true;
        }
        return;
    }
    for (const Symbol *c : sym->children()) {
        mark_struct_tree(set, c, changed);
    }
}

}// namespace

InstArgRW classify_inst_rw(const Instruction &inst) noexcept {
    InstArgRW out{};
    const size_t n = inst.args().size();
    auto set_write = [&](size_t idx) noexcept {
        if (idx < 64u) {
            out.write_mask |= (uint64_t{1u} << idx);
        } else {
            out.write_overflow = true;
        }
    };
    auto rw = parse_argrw_hint(inst);
    if (rw.has_value() && rw->size() == n) {
        for (size_t i = 0u; i < n; ++i) {
            char c = (*rw)[i];
            if (c == 'w' || c == 'W') { set_write(i); }
        }
        return out;
    }
    auto op = inst.opcode();
    if (op == "sincos" && n == 3u) {
        set_write(1u);
        set_write(2u);
        return out;
    }
    if (op == "split" && n >= 3u) {
        set_write(0u);
        set_write(2u);
        return out;
    }
    if (op == "getattribute" || op == "getmessage" || op == "gettextureinfo" || op == "getmatrix") {
        if (n >= 2u) {
            set_write(0u);
            set_write(n - 1u);
        }
        return out;
    }
    if (op == "setmessage" || op == "if" || op == "for" || op == "while" || op == "dowhile" ||
        op == "printf" || op == "warning" || op == "error" ||
        op == "jump" || op == "functioncall" || op == "return" || op == "exit" ||
        op == "break" || op == "continue" || op == "nop") {
        return out;
    }
    if (n >= 1u) {
        set_write(0u);
    }
    return out;
}

void analyze_group_derivatives(
    const LinkedGroup &group,
    const GroupStringAnalysis &str_analysis,
    GroupDerivAnalysis &out_derivs) noexcept {

    const size_t n_layers = group.layers.size();
    luisa::vector<luisa::unordered_set<const Symbol *>> d_need(n_layers);
    luisa::vector<luisa::unordered_set<const Symbol *>> d_src(n_layers);

    luisa::unordered_set<StringToken> msg_need_tokens;
    luisa::unordered_set<StringToken> msg_src_tokens;
    bool msg_need_all = false;
    bool msg_src_all = false;

    // Seed backward demand D_need
    for (const auto &out : group.outputs) {
        if (out.want_deriv && out.layer_index < n_layers) {
            bool dummy = false;
            mark_struct_tree(d_need[out.layer_index], out.symbol, dummy);
        }
    }

    luisa::vector<bool> is_write, is_read;
    for (size_t li = 0u; li < n_layers; ++li) {
        const auto &ll = group.layers[li];
        const auto &la = str_analysis.layers[li];
        for (uint32_t pc = 0u; pc < ll.shader->instructions().size(); ++pc) {
            if (!la.instructions[pc].reachable) { continue; }
            const auto &inst = *ll.shader->instructions()[pc];
            auto op = inst.opcode();
            const auto args = inst.args();
            classify_inst_rw_vecs(inst, is_write, is_read);

            if (is_deriv_seed_op(op) && args.size() >= 2u) {
                bool dummy = false;
                mark_struct_tree(d_need[li], args[1], dummy);
            }
            for (size_t i = 0u; i < args.size(); ++i) {
                if (is_read[i] && instruction_has_argderivs(inst, i + 1u)) {
                    bool dummy = false;
                    mark_struct_tree(d_need[li], args[i], dummy);
                }
            }
            if ((op == "texture" || op == "texture3d" || op == "environment") && args.size() >= 3u) {
                // Coords without explicit dPdx/dPdy require derivatives
                // texture(res, fname, u, v, ...), texture3d/env(res, fname, P, ...)
                size_t pos_end = (op == "texture") ? 4u : 3u;
                bool has_explicit_derivs = false;
                if (op == "texture" && args.size() >= 8u &&
                    !is_simple_type(args[4]->type(), SimpleType::Primitive::STRING)) {
                    has_explicit_derivs = true;
                } else if ((op == "texture3d" || op == "environment") && args.size() >= 5u &&
                           !is_simple_type(args[3]->type(), SimpleType::Primitive::STRING)) {
                    has_explicit_derivs = true;
                }
                if (!has_explicit_derivs) {
                    bool dummy = false;
                    for (size_t i = 2u; i < std::min(args.size(), pos_end); ++i) {
                        mark_struct_tree(d_need[li], args[i], dummy);
                    }
                }
            }
        }
    }

    // Backward fixed point for D_need
    bool changed = true;
    while (changed) {
        changed = false;
        // Across connections: if dst param is needed, mark src oparam
        for (const auto &conn : group.connections) {
            auto check_conn = [&](const Symbol *src_s, const Symbol *dst_s, auto &self) -> void {
                if (!src_s->children().empty() && !dst_s->children().empty()) {
                    for (size_t k = 0u; k < std::min(src_s->children().size(), dst_s->children().size()); ++k) {
                        self(src_s->children()[k], dst_s->children()[k], self);
                    }
                    return;
                }
                if (d_need[conn.dst_layer].contains(dst_s)) {
                    mark_struct_tree(d_need[conn.src_layer], src_s, changed);
                }
            };
            check_conn(conn.src_symbol, conn.dst_symbol, check_conn);
        }

        for (size_t li = n_layers; li-- > 0u;) {
            const auto &ll = group.layers[li];
            const auto &la = str_analysis.layers[li];
            for (uint32_t pc = static_cast<uint32_t>(ll.shader->instructions().size()); pc-- > 0u;) {
                if (!la.instructions[pc].reachable) { continue; }
                const auto &inst = *ll.shader->instructions()[pc];
                auto op = inst.opcode();
                if (is_zero_deriv_op(op)) { continue; }
                const auto args = inst.args();
                classify_inst_rw_vecs(inst, is_write, is_read);

                // Intra-group getmessage -> setmessage demand propagation
                if (op == "getmessage" && args.size() >= 3u) {
                    bool dst_needed = tree_contains_any(d_need[li], args.back());
                    if (dst_needed && args.size() == 3u && !msg_need_all) {
                        const auto &sl = la.instructions[pc].arg_strings[1];
                        if (sl.is_set()) {
                            for (StringToken t : sl.tokens) {
                                if (msg_need_tokens.emplace(t).second) {
                                    changed = true;
                                }
                            }
                        } else {
                            msg_need_all = true;
                            changed = true;
                        }
                    }
                    continue;
                }
                if (op == "setmessage" && args.size() >= 2u) {
                    bool needed = msg_need_all;
                    if (!needed) {
                        const auto &sl = la.instructions[pc].arg_strings[0];
                        if (!sl.is_set()) {
                            needed = true;
                        } else {
                            for (StringToken t : sl.tokens) {
                                if (msg_need_tokens.contains(t)) {
                                    needed = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (needed) {
                        mark_struct_tree(d_need[li], args[1], changed);
                    }
                    continue;
                }

                bool out_needed = false;
                for (size_t i = 0u; i < args.size(); ++i) {
                    if (is_write[i] && d_need[li].contains(args[i])) {
                        out_needed = true;
                        break;
                    }
                }
                if (!out_needed) { continue; }

                for (size_t i = 0u; i < args.size(); ++i) {
                    if (is_read[i] && is_sym_differentiable(args[i])) {
                        mark_struct_tree(d_need[li], args[i], changed);
                    }
                }
            }
        }
    }

    // Seed forward derivative sources D_src
    for (size_t li = 0u; li < n_layers; ++li) {
        const auto &ll = group.layers[li];
        const auto &la = str_analysis.layers[li];
        for (auto &&sym : ll.shader->symbols()) {
            if (sym->tag() == Symbol::Tag::SYM_GLOBAL) {
                if (auto g = parse_global_symbol(sym->identifier()); g && is_global_differentiable(*g)) {
                    bool dummy = false;
                    mark_struct_tree(d_src[li], sym.get(), dummy);
                }
            }
        }
        for (uint32_t pc = 0u; pc < ll.shader->instructions().size(); ++pc) {
            if (!la.instructions[pc].reachable) { continue; }
            const auto &inst = *ll.shader->instructions()[pc];
            auto op = inst.opcode();
            const auto args = inst.args();
            if (has_source_deriv_hook(op) && !args.empty()) {
                bool dummy = false;
                if (op == "getattribute" || op == "getmessage") {
                    if (args.size() >= 2u) {
                        if (op == "getattribute" || (args.size() == 4u)) {
                            mark_struct_tree(d_src[li], args.back(), dummy);
                        }
                    }
                } else {
                    classify_inst_rw_vecs(inst, is_write, is_read);
                    for (size_t i = 0u; i < args.size(); ++i) {
                        if (is_write[i]) {
                            mark_struct_tree(d_src[li], args[i], dummy);
                        }
                    }
                }
            }
        }
    }

    // Forward fixed point for D_src
    changed = true;
    while (changed) {
        changed = false;
        for (size_t li = 0u; li < n_layers; ++li) {
            const auto &ll = group.layers[li];
            const auto &la = str_analysis.layers[li];

            // Incoming layer connections
            for (uint32_t ci : ll.incoming_connections) {
                const auto &conn = group.connections[ci];
                auto prop_conn = [&](const Symbol *src_s, const Symbol *dst_s, auto &self) -> void {
                    if (!src_s->children().empty() && !dst_s->children().empty()) {
                        for (size_t k = 0u; k < std::min(src_s->children().size(), dst_s->children().size()); ++k) {
                            self(src_s->children()[k], dst_s->children()[k], self);
                        }
                        return;
                    }
                    if (d_src[conn.src_layer].contains(src_s)) {
                        mark_struct_tree(d_src[li], dst_s, changed);
                    }
                };
                prop_conn(conn.src_symbol, conn.dst_symbol, prop_conn);
            }

            for (uint32_t pc = 0u; pc < ll.shader->instructions().size(); ++pc) {
                if (!la.instructions[pc].reachable) { continue; }
                const auto &inst = *ll.shader->instructions()[pc];
                auto op = inst.opcode();
                if (is_zero_deriv_op(op)) { continue; }
                const auto args = inst.args();
                classify_inst_rw_vecs(inst, is_write, is_read);

                if (op == "setmessage" && args.size() >= 2u) {
                    if (tree_contains_any(d_src[li], args[1]) && !msg_src_all) {
                        const auto &sl = la.instructions[pc].arg_strings[0];
                        if (sl.is_set()) {
                            for (StringToken t : sl.tokens) {
                                if (msg_src_tokens.emplace(t).second) {
                                    changed = true;
                                }
                            }
                        } else {
                            msg_src_all = true;
                            changed = true;
                        }
                    }
                    continue;
                }
                if (op == "getmessage" && args.size() == 3u) {
                    bool has_src = msg_src_all;
                    if (!has_src) {
                        const auto &sl = la.instructions[pc].arg_strings[1];
                        if (!sl.is_set()) {
                            has_src = true;
                        } else {
                            for (StringToken t : sl.tokens) {
                                if (msg_src_tokens.contains(t)) {
                                    has_src = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (has_src) {
                        mark_struct_tree(d_src[li], args[2], changed);
                    }
                    continue;
                }

                bool any_src_deriv = false;
                for (size_t i = 0u; i < args.size(); ++i) {
                    if (is_read[i] && d_src[li].contains(args[i])) {
                        any_src_deriv = true;
                        break;
                    }
                }
                if (!any_src_deriv) { continue; }

                for (size_t i = 0u; i < args.size(); ++i) {
                    if (is_write[i] && is_sym_differentiable(args[i])) {
                        mark_struct_tree(d_src[li], args[i], changed);
                    }
                }
            }
        }
    }

    // Intersection: dual_symbols = D_need \cap D_src
    out_derivs.dual_symbols.clear();
    out_derivs.dual_symbols.resize(n_layers);
    out_derivs.total_dual_symbols = 0u;
    for (size_t li = 0u; li < n_layers; ++li) {
        for (const Symbol *sym : d_need[li]) {
            if (d_src[li].contains(sym)) {
                if (out_derivs.dual_symbols[li].emplace(sym).second) {
                    out_derivs.total_dual_symbols++;
                }
            }
        }
    }
}

}// namespace luisa::compute::osl::detail
