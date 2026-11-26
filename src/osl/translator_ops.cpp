#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <utility>

#include <luisa/ast/type_registry.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/xir/op.h>
#include "translator_impl.h"

namespace luisa::compute::osl {

namespace {

[[nodiscard]] const Type *osl_int() noexcept {
    static const SimpleType t{SimpleType::Primitive::INT};
    return &t;
}
[[nodiscard]] const Type *osl_float() noexcept {
    static const SimpleType t{SimpleType::Primitive::FLOAT};
    return &t;
}
[[nodiscard]] const Type *osl_point() noexcept {
    static const SimpleType t{SimpleType::Primitive::POINT};
    return &t;
}
[[nodiscard]] const Type *osl_vector() noexcept {
    static const SimpleType t{SimpleType::Primitive::VECTOR};
    return &t;
}
[[nodiscard]] const Type *osl_color() noexcept {
    static const SimpleType t{SimpleType::Primitive::COLOR};
    return &t;
}

template<typename Fn>
void dispatch_strings_1(
    OSLTranslator::Impl &impl,
    const OSLTranslator::Impl::StringSlotSpec &s0,
    Fn &&fn) noexcept {
    const OSLTranslator::Impl::StringSlotSpec arr[1]{s0};
    impl.dispatch_dynamic_strings(arr, std::forward<Fn>(fn));
}

template<typename Fn>
void dispatch_strings_2(
    OSLTranslator::Impl &impl,
    const OSLTranslator::Impl::StringSlotSpec &s0,
    const OSLTranslator::Impl::StringSlotSpec &s1,
    Fn &&fn) noexcept {
    const OSLTranslator::Impl::StringSlotSpec arr[2]{s0, s1};
    impl.dispatch_dynamic_strings(arr, std::forward<Fn>(fn));
}

[[nodiscard]] xir::Value *splat_scalar_for(
    OSLTranslator::Impl &impl, const compute::Type *xt, float scalar) noexcept {
    if (xt == compute::Type::of<luisa::float3>()) {
        return impl.const_float3(luisa::float3{scalar});
    }
    return impl.const_float(scalar);
}

[[nodiscard]] const compute::Type *bool_type_for(const compute::Type *xt) noexcept {
    if (xt == compute::Type::of<luisa::float3>()) {
        return compute::Type::of<luisa::bool3>();
    }
    return compute::Type::of<bool>();
}

[[nodiscard]] xir::Value *make_bool_mask(
    OSLTranslator::Impl &impl, xir::Value *cond_bool, const compute::Type *elem_xt) noexcept {
    if (elem_xt == compute::Type::of<luisa::float3>()) {
        if (cond_bool->type() == compute::Type::of<luisa::bool3>()) {
            return cond_bool;
        }
        return impl.builder.call(
            compute::Type::of<luisa::bool3>(), xir::ArithmeticOp::AGGREGATE,
            {cond_bool, cond_bool, cond_bool});
    }
    return cond_bool;
}

[[nodiscard]] const Type *widen_osl_arithmetic_type(
    const Type *a, const Type *b, const Type *preferred_dst) noexcept {
    if (preferred_dst != nullptr && detail::is_triple_type(preferred_dst)) { return preferred_dst; }
    if (detail::is_triple_type(a)) { return a; }
    if (detail::is_triple_type(b)) { return b; }
    if (preferred_dst != nullptr && detail::is_simple_type(preferred_dst, SimpleType::Primitive::FLOAT)) {
        return preferred_dst;
    }
    if (detail::is_simple_type(a, SimpleType::Primitive::FLOAT) ||
        detail::is_simple_type(b, SimpleType::Primitive::FLOAT)) {
        return preferred_dst != nullptr && detail::is_simple_type(preferred_dst, SimpleType::Primitive::FLOAT)
                   ? preferred_dst
                   : (detail::is_simple_type(a, SimpleType::Primitive::FLOAT) ? a : b);
    }
    return preferred_dst != nullptr ? preferred_dst : a;
}

[[nodiscard]] bool dst_wants_deriv(
    const OSLTranslator::Impl &impl, uint32_t active_layer, const Symbol *dst) noexcept {
    if (dst == nullptr) { return false; }
    if (dst->tag() == Symbol::Tag::SYM_GLOBAL) {
        if (auto g = detail::parse_global_symbol(dst->identifier())) {
            return detail::is_global_differentiable(*g);
        }
        return false;
    }
    const auto *st = impl.find_storage(active_layer, dst);
    return st != nullptr && st->is_dual;
}

[[nodiscard]] const detail::InstructionAnalysis *get_inst_analysis(
    const OSLTranslator::Impl &impl, uint32_t active_layer, uint32_t pc) noexcept {
    const auto &la = impl.string_analysis.layers[active_layer];
    if (pc < la.instructions.size()) {
        return &la.instructions[pc];
    }
    return nullptr;
}

[[nodiscard]] detail::StringLattice get_arg_string_lattice(
    OSLTranslator::Impl &impl,
    const detail::InstructionAnalysis *ia,
    size_t arg_idx,
    const Symbol *sym) noexcept {
    if (ia != nullptr && arg_idx < ia->arg_strings.size() && !ia->arg_strings[arg_idx].is_bot()) {
        return ia->arg_strings[arg_idx];
    }
    if (sym->tag() == Symbol::Tag::SYM_CONST && !sym->is_array() &&
        detail::is_simple_type(sym->type(), SimpleType::Primitive::STRING)) {
        StringToken tok = sym->initial_values().empty()
                              ? empty_string_token
                              : impl.string_table.intern(sym->initial_values()[0].as_string());
        return detail::StringLattice::singleton(tok);
    }
    return detail::StringLattice::top();
}

[[nodiscard]] xir::Value *get_arg_string_token(
    OSLTranslator::Impl &impl,
    uint32_t active_layer,
    const detail::InstructionAnalysis *ia,
    size_t arg_idx,
    const Symbol *sym) noexcept {
    auto lat = get_arg_string_lattice(impl, ia, arg_idx, sym);
    if (auto s = lat.as_singleton()) {
        return impl.token_for_string(impl.string_table.lookup(*s));
    }
    return impl.load_string_token(active_layer, sym);
}

[[nodiscard]] OSLTranslator::Impl::StringSlotSpec make_string_slot(
    OSLTranslator::Impl &impl,
    uint32_t active_layer,
    const detail::InstructionAnalysis *ia,
    size_t arg_idx,
    const Symbol *sym) noexcept {
    OSLTranslator::Impl::StringSlotSpec spec;
    spec.lattice = get_arg_string_lattice(impl, ia, arg_idx, sym);
    if (!spec.lattice.is_singleton() && !spec.lattice.is_bot() && !spec.lattice.is_top()) {
        spec.runtime_token = impl.load_string_token(active_layer, sym);
    }
    return spec;
}

[[nodiscard]] luisa::optional<ConstantOptionValue> extract_arg_constant(
    OSLTranslator::Impl &impl,
    const detail::InstructionAnalysis *ia,
    size_t arg_idx,
    const Symbol *sym) noexcept {
    if (sym == nullptr || sym->is_array()) { return luisa::nullopt; }
    if (detail::is_simple_type(sym->type(), SimpleType::Primitive::STRING)) {
        auto lat = get_arg_string_lattice(impl, ia, arg_idx, sym);
        if (auto tok = lat.as_singleton()) {
            return ConstantOptionValue{impl.string_table.lookup(*tok)};
        }
        return luisa::nullopt;
    }
    if (ia != nullptr && arg_idx < ia->arg_scalars.size()) {
        const auto &sc = ia->arg_scalars[arg_idx];
        if (sc.is_int()) {
            return ConstantOptionValue{sc.i};
        }
        if (sc.is_float()) {
            if (detail::is_simple_type(sym->type(), SimpleType::Primitive::FLOAT)) {
                return ConstantOptionValue{sc.f};
            }
            if (detail::is_triple_type(sym->type())) {
                return ConstantOptionValue{luisa::float3{sc.f}};
            }
        }
    }
    if (sym->tag() == Symbol::Tag::SYM_CONST) {
        const auto &vals = sym->initial_values();
        if (detail::is_simple_type(sym->type(), SimpleType::Primitive::INT)) {
            return ConstantOptionValue{vals.empty() ? 0 : static_cast<int32_t>(vals[0].as_int())};
        }
        if (detail::is_simple_type(sym->type(), SimpleType::Primitive::FLOAT)) {
            return ConstantOptionValue{vals.empty() ? 0.0f : vals[0].as_float()};
        }
        if (detail::is_triple_type(sym->type())) {
            luisa::float3 v{0.0f};
            if (vals.size() >= 3u) {
                v = luisa::float3{vals[0].as_float(), vals[1].as_float(), vals[2].as_float()};
            } else if (vals.size() == 1u) {
                v = luisa::float3{vals[0].as_float()};
            }
            return ConstantOptionValue{v};
        }
        if (detail::is_simple_type(sym->type(), SimpleType::Primitive::MATRIX)) {
            luisa::float4x4 m = luisa::make_float4x4(0.0f);
            if (vals.size() >= 16u) {
                float raw[16];
                for (size_t i = 0u; i < 16u; ++i) { raw[i] = vals[i].as_float(); }
                m = detail::osl_row_major_to_xir_matrix(raw);
            } else if (vals.size() == 1u) {
                m = luisa::make_float4x4(vals[0].as_float());
            }
            return ConstantOptionValue{m};
        }
    }
    return luisa::nullopt;
}

[[nodiscard]] bool arg_is_written_output(luisa::optional<luisa::string_view> argrw, size_t arg_idx) noexcept {
    if (!argrw.has_value() || arg_idx >= argrw->size()) { return false; }
    char c = (*argrw)[arg_idx];
    return c == 'w' || c == 'W';
}

struct PendingStringOptEntry {
    size_t key_arg_idx{0u};
    size_t val_arg_idx{0u};
    size_t key_slot_idx{~0u};
    size_t val_slot_idx{~0u};
    bool is_output{false};
    DualValue loaded_val{};
    xir::Value *out_val_ptr{nullptr};
    xir::Value *out_dx_ptr{nullptr};
    xir::Value *out_dy_ptr{nullptr};
    luisa::optional<ConstantOptionValue> cached_const{};
};

void collect_named_options(
    OSLTranslator::Impl &impl,
    uint32_t active_layer,
    const detail::InstructionAnalysis *ia,
    const Instruction &inst,
    size_t start_idx,
    size_t end_idx,
    bool want_opt_derivs,
    luisa::vector<OSLTranslator::Impl::StringSlotSpec> &slots,
    luisa::vector<PendingStringOptEntry> &pending_opts) noexcept {

    const auto args = inst.args();
    auto argrw = detail::parse_argrw_hint(inst);

    for (size_t k = start_idx; k + 1u < end_idx; k += 2u) {
        const Symbol *key_sym = args[k];
        const Symbol *val_sym = args[k + 1u];
        if (!detail::is_simple_type(key_sym->type(), SimpleType::Primitive::STRING)) {
            impl.report_error(luisa::format(
                "{}: expected string option key at argument {}.", inst.opcode(), k));
            return;
        }

        PendingStringOptEntry entry;
        entry.key_arg_idx = k;
        entry.val_arg_idx = k + 1u;
        entry.key_slot_idx = slots.size();
        slots.emplace_back(make_string_slot(impl, active_layer, ia, k, key_sym));

        bool is_out = arg_is_written_output(argrw, k + 1u) &&
                      val_sym->tag() != Symbol::Tag::SYM_CONST;
        entry.is_output = is_out;

        if (is_out) {
            if (val_sym->tag() == Symbol::Tag::SYM_GLOBAL) {
                impl.report_error(luisa::format(
                    "{}: output option writing directly to global '{}' is not supported.",
                    inst.opcode(), val_sym->identifier()));
                return;
            }
            const auto *st = impl.find_storage(active_layer, val_sym);
            if (st == nullptr || st->val_ptr == nullptr) {
                impl.report_error(luisa::format(
                    "{}: unallocated output option destination '{}'.",
                    inst.opcode(), val_sym->identifier()));
                return;
            }
            entry.out_val_ptr = st->val_ptr;
            entry.out_dx_ptr = st->dx_ptr;
            entry.out_dy_ptr = st->dy_ptr;
        } else if (detail::is_simple_type(val_sym->type(), SimpleType::Primitive::STRING)) {
            entry.val_slot_idx = slots.size();
            slots.emplace_back(make_string_slot(impl, active_layer, ia, k + 1u, val_sym));
            entry.loaded_val = {.val = get_arg_string_token(impl, active_layer, ia, k + 1u, val_sym)};
        } else if (val_sym->type()->tag() != Type::Tag::CLOSURE && !val_sym->is_array()) {
            entry.loaded_val = impl.load_symbol(active_layer, val_sym, want_opt_derivs);
            entry.cached_const = extract_arg_constant(impl, ia, k + 1u, val_sym);
        } else if (val_sym->is_array()) {
            if (const auto *st = impl.find_storage(active_layer, val_sym)) {
                entry.loaded_val = {.val = st->val_ptr};
            }
        }
        pending_opts.emplace_back(std::move(entry));
    }
    if ((end_idx - start_idx) % 2u != 0u) {
        impl.report_error(luisa::format(
            "{}: trailing unpaired named option argument.", inst.opcode()));
    }
}

[[nodiscard]] luisa::vector<NamedOption> materialize_named_options(
    const Instruction &inst,
    const OSLTranslator::Impl &impl,
    uint32_t active_layer,
    luisa::span<const PendingStringOptEntry> pending_opts,
    luisa::span<const luisa::string_view> resolved_strings) noexcept {
    luisa::vector<NamedOption> opts;
    opts.reserve(pending_opts.size());
    const auto args = inst.args();
    for (const auto &p : pending_opts) {
        const Symbol *val_sym = args[p.val_arg_idx];
        NamedOption no;
        no.key = resolved_strings[p.key_slot_idx];
        no.osl_type = val_sym->type();
        no.array_length = impl.effective_array_length(active_layer, val_sym);
        no.is_output = p.is_output;
        no.value = p.loaded_val;
        no.out_val_ptr = p.out_val_ptr;
        no.out_dx_ptr = p.out_dx_ptr;
        no.out_dy_ptr = p.out_dy_ptr;
        if (p.val_slot_idx != ~0u) {
            no.constant = ConstantOptionValue{resolved_strings[p.val_slot_idx]};
        } else {
            no.constant = p.cached_const;
        }
        opts.emplace_back(no);
    }
    return opts;
}

template<typename Fn>
void conditional_update_symbol(
    OSLTranslator::Impl &impl,
    uint32_t active_layer,
    const Symbol *dst,
    xir::Value *ok_int,
    bool want_deriv,
    Fn &&produce_value) noexcept {

    impl.store_symbol(active_layer, dst, impl.load_symbol(active_layer, dst, want_deriv));
    xir::Value *ok_bool = impl.to_bool(ok_int);
    xir::BasicBlock *then_bb = impl.function->create_basic_block();
    xir::BasicBlock *merge_bb = impl.function->create_basic_block();
    impl.builder.cond_br(ok_bool, then_bb, merge_bb);
    impl.builder.set_insertion_point(then_bb);
    DualValue new_v = produce_value();
    if (!impl.errors.empty()) {
        if (!impl.builder.is_insertion_point_terminator()) { impl.builder.br(merge_bb); }
        impl.builder.set_insertion_point(merge_bb);
        return;
    }
    if (dst->is_array()) {
        if (const auto *dst_st = impl.find_storage(active_layer, dst)) {
            if (new_v.val != nullptr && new_v.val->type() == dst_st->val_ptr->type()) {
                impl.store_symbol(active_layer, dst, new_v);
            }
        }
    } else if (new_v.valid()) {
        if (detail::is_simple_type(dst->type(), SimpleType::Primitive::STRING)) {
            impl.store_string_token(active_layer, dst, new_v.val);
        } else {
            impl.store_symbol(active_layer, dst, new_v);
        }
    }
    if (!impl.builder.is_insertion_point_terminator()) {
        impl.builder.br(merge_bb);
    }
    impl.builder.set_insertion_point(merge_bb);
}

[[nodiscard]] xir::Value *map_matrix_cols(
    OSLTranslator::Impl &impl,
    xir::Value *mat_a,
    xir::Value *mat_b,
    xir::ArithmeticOp op) noexcept {
    xir::Value *cols[4]{};
    auto *f4_t = compute::Type::of<luisa::float4>();
    for (uint32_t c = 0u; c < 4u; ++c) {
        xir::Value *idx = impl.const_uint(c);
        xir::Value *ca = impl.builder.call(f4_t, xir::ArithmeticOp::EXTRACT, {mat_a, idx});
        if (mat_b != nullptr) {
            xir::Value *cb = impl.builder.call(f4_t, xir::ArithmeticOp::EXTRACT, {mat_b, idx});
            cols[c] = impl.builder.call(f4_t, op, {ca, cb});
        } else {
            cols[c] = impl.builder.call(f4_t, op, {ca});
        }
    }
    return impl.builder.call(compute::Type::of<luisa::float4x4>(), xir::ArithmeticOp::AGGREGATE,
                             {cols[0], cols[1], cols[2], cols[3]});
}

[[nodiscard]] xir::Value *scale_matrix_scalar(
    OSLTranslator::Impl &impl, xir::Value *mat, xir::Value *s) noexcept {
    auto *f4_t = compute::Type::of<luisa::float4>();
    xir::Value *sf = impl.to_float(s);
    xir::Value *s4 = impl.builder.call(f4_t, xir::ArithmeticOp::AGGREGATE, {sf, sf, sf, sf});
    xir::Value *cols[4]{};
    for (uint32_t c = 0u; c < 4u; ++c) {
        xir::Value *ca = impl.builder.call(f4_t, xir::ArithmeticOp::EXTRACT, {mat, impl.const_uint(c)});
        cols[c] = impl.builder.call(f4_t, xir::ArithmeticOp::BINARY_MUL, {ca, s4});
    }
    return impl.builder.call(compute::Type::of<luisa::float4x4>(), xir::ArithmeticOp::AGGREGATE,
                             {cols[0], cols[1], cols[2], cols[3]});
}

[[nodiscard]] xir::Value *matrix_equal(
    OSLTranslator::Impl &impl, xir::Value *mat_a, xir::Value *mat_b) noexcept {
    auto *f4_t = compute::Type::of<luisa::float4>();
    auto *b4_t = compute::Type::of<luisa::bool4>();
    auto *b_t = compute::Type::of<bool>();
    xir::Value *acc = impl.const_bool(true);
    for (uint32_t c = 0u; c < 4u; ++c) {
        xir::Value *idx = impl.const_uint(c);
        xir::Value *ca = impl.builder.call(f4_t, xir::ArithmeticOp::EXTRACT, {mat_a, idx});
        xir::Value *cb = impl.builder.call(f4_t, xir::ArithmeticOp::EXTRACT, {mat_b, idx});
        xir::Value *cmp4 = impl.builder.call(b4_t, xir::ArithmeticOp::BINARY_EQUAL, {ca, cb});
        xir::Value *eq_col = impl.builder.call(b_t, xir::ArithmeticOp::ALL, {cmp4});
        acc = impl.builder.call(b_t, xir::ArithmeticOp::BINARY_BIT_AND, {acc, eq_col});
    }
    return acc;
}

[[nodiscard]] DualValue make_sign_dual(
    OSLTranslator::Impl &impl, xir::Value *x, const compute::Type *xt) noexcept {
    xir::Value *z = splat_scalar_for(impl, xt, 0.0f);
    xir::Value *one = splat_scalar_for(impl, xt, 1.0f);
    xir::Value *neg_one = splat_scalar_for(impl, xt, -1.0f);
    const compute::Type *bt = bool_type_for(xt);
    xir::Value *gt0 = impl.builder.call(bt, xir::ArithmeticOp::BINARY_GREATER, {x, z});
    xir::Value *lt0 = impl.builder.call(bt, xir::ArithmeticOp::BINARY_LESS, {x, z});
    xir::Value *pos_or_zero = impl.builder.call(xt, xir::ArithmeticOp::SELECT, {z, one, gt0});
    xir::Value *res = impl.builder.call(xt, xir::ArithmeticOp::SELECT, {pos_or_zero, neg_one, lt0});
    return {.val = res};
}

[[nodiscard]] xir::Value *safe_div_val(
    OSLTranslator::Impl &impl, xir::Value *a, xir::Value *b) noexcept {
    const compute::Type *xt = a->type();
    if (xt == compute::Type::of<int32_t>()) {
        auto *bt = compute::Type::of<bool>();
        xir::Value *z = impl.const_int(0);
        xir::Value *is_zero = impl.builder.call(bt, xir::ArithmeticOp::BINARY_EQUAL, {b, z});
        xir::Value *is_min = impl.builder.call(bt, xir::ArithmeticOp::BINARY_EQUAL, {a, impl.const_int(INT32_MIN)});
        xir::Value *is_m1 = impl.builder.call(bt, xir::ArithmeticOp::BINARY_EQUAL, {b, impl.const_int(-1)});
        xir::Value *is_min_m1 = impl.builder.call(bt, xir::ArithmeticOp::BINARY_BIT_AND, {is_min, is_m1});
        xir::Value *bad = impl.builder.call(bt, xir::ArithmeticOp::BINARY_BIT_OR, {is_zero, is_min_m1});
        xir::Value *safe_b = impl.builder.call(xt, xir::ArithmeticOp::SELECT, {b, impl.const_int(1), bad});
        xir::Value *q = impl.builder.call(xt, xir::ArithmeticOp::BINARY_DIV, {a, safe_b});
        xir::Value *q_safe = impl.builder.call(xt, xir::ArithmeticOp::SELECT, {q, z, is_zero});
        return impl.builder.call(xt, xir::ArithmeticOp::SELECT, {q_safe, impl.const_int(INT32_MIN), is_min_m1});
    }
    xir::Value *z = splat_scalar_for(impl, xt, 0.0f);
    xir::Value *one = splat_scalar_for(impl, xt, 1.0f);
    const compute::Type *bt = bool_type_for(xt);
    xir::Value *nz = impl.builder.call(bt, xir::ArithmeticOp::BINARY_NOT_EQUAL, {b, z});
    xir::Value *safe_b = impl.builder.call(xt, xir::ArithmeticOp::SELECT, {one, b, nz});
    xir::Value *q = impl.builder.call(xt, xir::ArithmeticOp::BINARY_DIV, {a, safe_b});
    return impl.builder.call(xt, xir::ArithmeticOp::SELECT, {z, q, nz});
}

[[nodiscard]] DualValue scale_dual_by_val(
    OSLTranslator::Impl &impl, const DualValue &dv, xir::Value *factor, const compute::Type *xt) noexcept {
    DualValue out;
    out.val = dv.val;
    if (dv.dx) { out.dx = impl.builder.call(xt, xir::ArithmeticOp::BINARY_MUL, {dv.dx, factor}); }
    if (dv.dy) { out.dy = impl.builder.call(xt, xir::ArithmeticOp::BINARY_MUL, {dv.dy, factor}); }
    return out;
}

[[nodiscard]] DualValue unary_float_math(
    OSLTranslator::Impl &impl,
    luisa::string_view op,
    const DualValue &src,
    bool want_deriv) noexcept {

    xir::Value *u = src.val;
    const compute::Type *xt = u->type();
    const compute::Type *bt = bool_type_for(xt);
    auto u_call = [&](xir::ArithmeticOp aop, xir::Value *v) {
        return impl.builder.call(xt, aop, {v});
    };
    auto b_call = [&](xir::ArithmeticOp aop, xir::Value *a, xir::Value *b) {
        return impl.builder.call(xt, aop, {a, b});
    };
    auto cmp_call = [&](xir::ArithmeticOp aop, xir::Value *a, xir::Value *b) {
        return impl.builder.call(bt, aop, {a, b});
    };
    auto splat = [&](float s) { return splat_scalar_for(impl, xt, s); };

    DualValue out;
    xir::Value *deriv_factor = nullptr;

    if (op == "neg") {
        out.val = u_call(xir::ArithmeticOp::UNARY_MINUS, u);
        if (want_deriv && src.has_derivs()) {
            if (src.dx) { out.dx = u_call(xir::ArithmeticOp::UNARY_MINUS, src.dx); }
            if (src.dy) { out.dy = u_call(xir::ArithmeticOp::UNARY_MINUS, src.dy); }
        }
        return out;
    }
    if (op == "abs" || op == "fabs") {
        out.val = u_call(xir::ArithmeticOp::ABS, u);
        if (want_deriv && src.has_derivs()) {
            deriv_factor = make_sign_dual(impl, u, xt).val;
        }
    } else if (op == "sign") {
        out.val = make_sign_dual(impl, u, xt).val;
        return out;
    } else if (op == "floor" || op == "ceil" || op == "round" || op == "trunc") {
        xir::ArithmeticOp aop = xir::ArithmeticOp::FLOOR;
        if (op == "ceil") { aop = xir::ArithmeticOp::CEIL; }
        else if (op == "round") { aop = xir::ArithmeticOp::ROUND; }
        else if (op == "trunc") { aop = xir::ArithmeticOp::TRUNC; }
        out.val = u_call(aop, u);
        return out;
    } else if (op == "sin") {
        out.val = u_call(xir::ArithmeticOp::SIN, u);
        if (want_deriv && src.has_derivs()) { deriv_factor = u_call(xir::ArithmeticOp::COS, u); }
    } else if (op == "cos") {
        out.val = u_call(xir::ArithmeticOp::COS, u);
        if (want_deriv && src.has_derivs()) {
            deriv_factor = u_call(xir::ArithmeticOp::UNARY_MINUS, u_call(xir::ArithmeticOp::SIN, u));
        }
    } else if (op == "tan") {
        out.val = u_call(xir::ArithmeticOp::TAN, u);
        if (want_deriv && src.has_derivs()) {
            xir::Value *c = u_call(xir::ArithmeticOp::COS, u);
            xir::Value *c2 = b_call(xir::ArithmeticOp::BINARY_MUL, c, c);
            deriv_factor = safe_div_val(impl, splat(1.0f), c2);
        }
    } else if (op == "asin") {
        xir::Value *uc = impl.builder.call(xt, xir::ArithmeticOp::CLAMP, {u, splat(-1.0f), splat(1.0f)});
        out.val = u_call(xir::ArithmeticOp::ASIN, uc);
        if (want_deriv && src.has_derivs()) {
            xir::Value *u2 = b_call(xir::ArithmeticOp::BINARY_MUL, uc, uc);
            xir::Value *inside = u_call(xir::ArithmeticOp::SQRT,
                                        b_call(xir::ArithmeticOp::MAX,
                                               b_call(xir::ArithmeticOp::BINARY_SUB, splat(1.0f), u2),
                                               splat(1e-12f)));
            deriv_factor = b_call(xir::ArithmeticOp::BINARY_DIV, splat(1.0f), inside);
        }
    } else if (op == "acos") {
        xir::Value *uc = impl.builder.call(xt, xir::ArithmeticOp::CLAMP, {u, splat(-1.0f), splat(1.0f)});
        out.val = u_call(xir::ArithmeticOp::ACOS, uc);
        if (want_deriv && src.has_derivs()) {
            xir::Value *u2 = b_call(xir::ArithmeticOp::BINARY_MUL, uc, uc);
            xir::Value *inside = u_call(xir::ArithmeticOp::SQRT,
                                        b_call(xir::ArithmeticOp::MAX,
                                               b_call(xir::ArithmeticOp::BINARY_SUB, splat(1.0f), u2),
                                               splat(1e-12f)));
            deriv_factor = b_call(xir::ArithmeticOp::BINARY_DIV, splat(-1.0f), inside);
        }
    } else if (op == "atan") {
        out.val = u_call(xir::ArithmeticOp::ATAN, u);
        if (want_deriv && src.has_derivs()) {
            xir::Value *u2 = b_call(xir::ArithmeticOp::BINARY_MUL, u, u);
            deriv_factor = b_call(xir::ArithmeticOp::BINARY_DIV, splat(1.0f),
                                  b_call(xir::ArithmeticOp::BINARY_ADD, splat(1.0f), u2));
        }
    } else if (op == "sinh") {
        out.val = u_call(xir::ArithmeticOp::SINH, u);
        if (want_deriv && src.has_derivs()) { deriv_factor = u_call(xir::ArithmeticOp::COSH, u); }
    } else if (op == "cosh") {
        out.val = u_call(xir::ArithmeticOp::COSH, u);
        if (want_deriv && src.has_derivs()) { deriv_factor = u_call(xir::ArithmeticOp::SINH, u); }
    } else if (op == "tanh") {
        out.val = u_call(xir::ArithmeticOp::TANH, u);
        if (want_deriv && src.has_derivs()) {
            xir::Value *c = u_call(xir::ArithmeticOp::COSH, u);
            deriv_factor = safe_div_val(impl, splat(1.0f), b_call(xir::ArithmeticOp::BINARY_MUL, c, c));
        }
    } else if (op == "exp") {
        out.val = u_call(xir::ArithmeticOp::EXP, u);
        if (want_deriv && src.has_derivs()) { deriv_factor = out.val; }
    } else if (op == "exp2") {
        out.val = u_call(xir::ArithmeticOp::EXP2, u);
        if (want_deriv && src.has_derivs()) {
            deriv_factor = b_call(xir::ArithmeticOp::BINARY_MUL, out.val, splat(0.69314718056f));
        }
    } else if (op == "expm1") {
        out.val = impl.emit_portable_expm1(u);
        if (want_deriv && src.has_derivs()) { deriv_factor = u_call(xir::ArithmeticOp::EXP, u); }
    } else if (op == "log") {
        out.val = u_call(xir::ArithmeticOp::LOG, u);
        if (want_deriv && src.has_derivs()) {
            xir::Value *pos = cmp_call(xir::ArithmeticOp::BINARY_GREATER, u, splat(0.0f));
            deriv_factor = impl.builder.call(xt, xir::ArithmeticOp::SELECT,
                                             {splat(0.0f), safe_div_val(impl, splat(1.0f), u), pos});
        }
    } else if (op == "log2") {
        out.val = u_call(xir::ArithmeticOp::LOG2, u);
        if (want_deriv && src.has_derivs()) {
            xir::Value *pos = cmp_call(xir::ArithmeticOp::BINARY_GREATER, u, splat(0.0f));
            xir::Value *inv = safe_div_val(impl, splat(1.44269504089f), u);
            deriv_factor = impl.builder.call(xt, xir::ArithmeticOp::SELECT, {splat(0.0f), inv, pos});
        }
    } else if (op == "log10") {
        out.val = u_call(xir::ArithmeticOp::LOG10, u);
        if (want_deriv && src.has_derivs()) {
            xir::Value *pos = cmp_call(xir::ArithmeticOp::BINARY_GREATER, u, splat(0.0f));
            xir::Value *inv = safe_div_val(impl, splat(0.43429448190f), u);
            deriv_factor = impl.builder.call(xt, xir::ArithmeticOp::SELECT, {splat(0.0f), inv, pos});
        }
    } else if (op == "logb") {
        xir::Value *au = b_call(xir::ArithmeticOp::MAX, u_call(xir::ArithmeticOp::ABS, u), splat(1e-38f));
        xir::Value *nz = cmp_call(xir::ArithmeticOp::BINARY_NOT_EQUAL, u, splat(0.0f));
        xir::Value *fl = u_call(xir::ArithmeticOp::FLOOR, u_call(xir::ArithmeticOp::LOG2, au));
        out.val = impl.builder.call(xt, xir::ArithmeticOp::SELECT, {splat(0.0f), fl, nz});
        return out;
    } else if (op == "sqrt") {
        xir::Value *pos = cmp_call(xir::ArithmeticOp::BINARY_GREATER, u, splat(0.0f));
        xir::Value *su = u_call(xir::ArithmeticOp::SQRT, b_call(xir::ArithmeticOp::MAX, u, splat(0.0f)));
        out.val = impl.builder.call(xt, xir::ArithmeticOp::SELECT, {splat(0.0f), su, pos});
        if (want_deriv && src.has_derivs()) {
            xir::Value *d = safe_div_val(impl, splat(0.5f), su);
            deriv_factor = impl.builder.call(xt, xir::ArithmeticOp::SELECT, {splat(0.0f), d, pos});
        }
    } else if (op == "inversesqrt") {
        xir::Value *pos = cmp_call(xir::ArithmeticOp::BINARY_GREATER, u, splat(0.0f));
        xir::Value *safe_u = impl.builder.call(xt, xir::ArithmeticOp::SELECT, {splat(1.0f), u, pos});
        xir::Value *ru = u_call(xir::ArithmeticOp::RSQRT, safe_u);
        out.val = impl.builder.call(xt, xir::ArithmeticOp::SELECT, {splat(0.0f), ru, pos});
        if (want_deriv && src.has_derivs()) {
            xir::Value *d = safe_div_val(impl, b_call(xir::ArithmeticOp::BINARY_MUL, splat(-0.5f), ru), safe_u);
            deriv_factor = impl.builder.call(xt, xir::ArithmeticOp::SELECT, {splat(0.0f), d, pos});
        }
    } else if (op == "cbrt") {
        xir::Value *au = u_call(xir::ArithmeticOp::ABS, u);
        xir::Value *nz = cmp_call(xir::ArithmeticOp::BINARY_GREATER, au, splat(0.0f));
        xir::Value *safe_au = b_call(xir::ArithmeticOp::MAX, au, splat(1e-30f));
        xir::Value *mag = b_call(xir::ArithmeticOp::POW, safe_au, splat(1.0f / 3.0f));
        xir::Value *sg = make_sign_dual(impl, u, xt).val;
        out.val = impl.builder.call(xt, xir::ArithmeticOp::SELECT,
                                    {splat(0.0f), b_call(xir::ArithmeticOp::BINARY_MUL, sg, mag), nz});
        if (want_deriv && src.has_derivs()) {
            xir::Value *v2 = b_call(xir::ArithmeticOp::BINARY_MUL, mag, mag);
            xir::Value *d = safe_div_val(impl, splat(1.0f / 3.0f), v2);
            deriv_factor = impl.builder.call(xt, xir::ArithmeticOp::SELECT, {splat(0.0f), d, nz});
        }
    } else if (op == "erf" || op == "erfc") {
        bool is_erfc = (op == "erfc");
        out.val = is_erfc ? impl.emit_portable_erfc(u) : impl.emit_portable_erf(u);
        if (want_deriv && src.has_derivs()) {
            constexpr float two_over_sqrt_pi = 1.1283791670955126f;
            xir::Value *neg_u2 = u_call(xir::ArithmeticOp::UNARY_MINUS, b_call(xir::ArithmeticOp::BINARY_MUL, u, u));
            xir::Value *e = u_call(xir::ArithmeticOp::EXP, neg_u2);
            deriv_factor = b_call(xir::ArithmeticOp::BINARY_MUL,
                                  splat(is_erfc ? -two_over_sqrt_pi : two_over_sqrt_pi), e);
        }
    }

    if (want_deriv && deriv_factor != nullptr && src.has_derivs()) {
        if (src.dx) { out.dx = b_call(xir::ArithmeticOp::BINARY_MUL, deriv_factor, src.dx); }
        if (src.dy) { out.dy = b_call(xir::ArithmeticOp::BINARY_MUL, deriv_factor, src.dy); }
    }
    return out;
}

[[nodiscard]] xir::Value *any_component_nonzero(OSLTranslator::Impl &impl, xir::Value *b) noexcept {
    if (b->type() == compute::Type::of<luisa::bool3>()) {
        return impl.builder.call(compute::Type::of<bool>(), xir::ArithmeticOp::ANY, {b});
    }
    return b;
}

[[nodiscard]] xir::Value *all_components_true(OSLTranslator::Impl &impl, xir::Value *b) noexcept {
    if (b->type() == compute::Type::of<luisa::bool3>()) {
        return impl.builder.call(compute::Type::of<bool>(), xir::ArithmeticOp::ALL, {b});
    }
    return b;
}

}// namespace

xir::Value *OSLTranslator::Impl::emit_portable_erf(xir::Value *x) noexcept {
    const compute::Type *xt = x->type();
    const compute::Type *bt = bool_type_for(xt);
    auto splat = [&](float s) { return splat_scalar_for(*this, xt, s); };
    auto bin = [&](xir::ArithmeticOp op, xir::Value *a, xir::Value *b) {
        return builder.call(xt, op, {a, b});
    };

    xir::Value *ax = builder.call(xt, xir::ArithmeticOp::ABS, {x});
    xir::Value *t = bin(xir::ArithmeticOp::BINARY_DIV, splat(1.0f),
                        bin(xir::ArithmeticOp::BINARY_ADD, splat(1.0f),
                            bin(xir::ArithmeticOp::BINARY_MUL, splat(0.3275911f), ax)));

    // Horner polynomial: a1..a5 (Abramowitz & Stegun 7.1.26)
    xir::Value *p = splat(1.061405429f);
    p = builder.call(xt, xir::ArithmeticOp::FMA, {p, t, splat(-1.453152027f)});
    p = builder.call(xt, xir::ArithmeticOp::FMA, {p, t, splat(1.421413741f)});
    p = builder.call(xt, xir::ArithmeticOp::FMA, {p, t, splat(-0.284496736f)});
    p = builder.call(xt, xir::ArithmeticOp::FMA, {p, t, splat(0.254829592f)});
    p = bin(xir::ArithmeticOp::BINARY_MUL, p, t);

    xir::Value *neg_ax2 = builder.call(xt, xir::ArithmeticOp::UNARY_MINUS,
                                       {bin(xir::ArithmeticOp::BINARY_MUL, ax, ax)});
    xir::Value *e = builder.call(xt, xir::ArithmeticOp::EXP, {neg_ax2});
    xir::Value *y = bin(xir::ArithmeticOp::BINARY_SUB, splat(1.0f),
                        bin(xir::ArithmeticOp::BINARY_MUL, p, e));
    xir::Value *neg_y = builder.call(xt, xir::ArithmeticOp::UNARY_MINUS, {y});
    xir::Value *ge0 = builder.call(bt, xir::ArithmeticOp::BINARY_GREATER_EQUAL, {x, splat(0.0f)});
    return builder.call(xt, xir::ArithmeticOp::SELECT, {neg_y, y, ge0});
}

xir::Value *OSLTranslator::Impl::emit_portable_erfc(xir::Value *x) noexcept {
    const compute::Type *xt = x->type();
    const compute::Type *bt = bool_type_for(xt);
    auto splat = [&](float s) { return splat_scalar_for(*this, xt, s); };
    auto bin = [&](xir::ArithmeticOp op, xir::Value *a, xir::Value *b) {
        return builder.call(xt, op, {a, b});
    };

    xir::Value *ax = builder.call(xt, xir::ArithmeticOp::ABS, {x});
    xir::Value *t = bin(xir::ArithmeticOp::BINARY_DIV, splat(1.0f),
                        bin(xir::ArithmeticOp::BINARY_ADD, splat(1.0f),
                            bin(xir::ArithmeticOp::BINARY_MUL, splat(0.3275911f), ax)));

    xir::Value *p = splat(1.061405429f);
    p = builder.call(xt, xir::ArithmeticOp::FMA, {p, t, splat(-1.453152027f)});
    p = builder.call(xt, xir::ArithmeticOp::FMA, {p, t, splat(1.421413741f)});
    p = builder.call(xt, xir::ArithmeticOp::FMA, {p, t, splat(-0.284496736f)});
    p = builder.call(xt, xir::ArithmeticOp::FMA, {p, t, splat(0.254829592f)});
    p = bin(xir::ArithmeticOp::BINARY_MUL, p, t);

    xir::Value *neg_ax2 = builder.call(xt, xir::ArithmeticOp::UNARY_MINUS,
                                       {bin(xir::ArithmeticOp::BINARY_MUL, ax, ax)});
    xir::Value *e = builder.call(xt, xir::ArithmeticOp::EXP, {neg_ax2});
    xir::Value *erfc_abs = bin(xir::ArithmeticOp::BINARY_MUL, p, e);
    xir::Value *erfc_neg = bin(xir::ArithmeticOp::BINARY_SUB, splat(2.0f), erfc_abs);
    xir::Value *ge0 = builder.call(bt, xir::ArithmeticOp::BINARY_GREATER_EQUAL, {x, splat(0.0f)});
    return builder.call(xt, xir::ArithmeticOp::SELECT, {erfc_neg, erfc_abs, ge0});
}

xir::Value *OSLTranslator::Impl::emit_portable_expm1(xir::Value *x) noexcept {
    const compute::Type *xt = x->type();
    const compute::Type *bt = bool_type_for(xt);
    auto splat = [&](float s) { return splat_scalar_for(*this, xt, s); };
    auto bin = [&](xir::ArithmeticOp op, xir::Value *a, xir::Value *b) {
        return builder.call(xt, op, {a, b});
    };

    xir::Value *ax = builder.call(xt, xir::ArithmeticOp::ABS, {x});
    xir::Value *small = builder.call(bt, xir::ArithmeticOp::BINARY_LESS, {ax, splat(1e-4f)});

    // Taylor: x * (1 + x * (0.5 + x * (1/6 + x/24)))
    xir::Value *poly = builder.call(xt, xir::ArithmeticOp::FMA, {x, splat(1.0f / 24.0f), splat(1.0f / 6.0f)});
    poly = builder.call(xt, xir::ArithmeticOp::FMA, {x, poly, splat(0.5f)});
    poly = builder.call(xt, xir::ArithmeticOp::FMA, {x, poly, splat(1.0f)});
    xir::Value *taylor = bin(xir::ArithmeticOp::BINARY_MUL, x, poly);

    xir::Value *direct = bin(xir::ArithmeticOp::BINARY_SUB,
                             builder.call(xt, xir::ArithmeticOp::EXP, {x}), splat(1.0f));
    return builder.call(xt, xir::ArithmeticOp::SELECT, {direct, taylor, small});
}

bool OSLTranslator::Impl::try_emit_instruction(
    uint32_t active_layer, uint32_t pc, const Instruction &inst) noexcept {

    const auto op = inst.opcode();
    const auto args = inst.args();
    const detail::InstructionAnalysis *ia = get_inst_analysis(*this, active_layer, pc);

    // 1. Control-flow / metadata / debug print opcodes (CFG already lowered in emit_code_range)
    if (op == "nop" || op == "useparam" || op == "end" ||
        op == "if" || op == "for" || op == "while" || op == "dowhile" ||
        op == "jump" || op == "break" || op == "continue" ||
        op == "return" || op == "exit" || op == "functioncall" ||
        op == "printf" || op == "warning" || op == "error") {
        return true;
    }

    // 2. assign
    if (op == "assign" && args.size() == 2u) {
        const Symbol *dst = args[0];
        const Symbol *src = args[1];

        if (dst->type()->tag() == Type::Tag::CLOSURE) {
            if (src->type()->tag() == Type::Tag::CLOSURE) {
                store_closure(active_layer, dst, load_closure(active_layer, src));
                return true;
            }
            if (detail::is_simple_type(src->type(), SimpleType::Primitive::INT) ||
                detail::is_simple_type(src->type(), SimpleType::Primitive::FLOAT) ||
                detail::is_triple_type(src->type())) {
                store_closure(active_layer, dst, owner->emit_closure_zero());
                return true;
            }
            return false;
        }
        if (dst->type()->tag() == Type::Tag::STRUCT && src->type()->tag() == Type::Tag::STRUCT) {
            auto assign_struct = [&](const Symbol *ds, const Symbol *ss, auto &self) -> bool {
                if (ds->children().size() != ss->children().size()) { return false; }
                for (size_t k = 0u; k < ds->children().size(); ++k) {
                    const Symbol *dc = ds->children()[k];
                    const Symbol *sc = ss->children()[k];
                    if (dc->type()->tag() == Type::Tag::STRUCT && sc->type()->tag() == Type::Tag::STRUCT) {
                        if (!self(dc, sc, self)) { return false; }
                        continue;
                    }
                    if (dc->is_array() && sc->is_array()) {
                        int n = std::min(effective_array_length(active_layer, dc),
                                         effective_array_length(active_layer, sc));
                        bool cwd = dst_wants_deriv(*this, active_layer, dc);
                        for (int j = 0; j < n; ++j) {
                            xir::Value *idx = const_int(j);
                            if (dc->type()->tag() == Type::Tag::CLOSURE) {
                                store_closure_array_element(active_layer, dc, idx,
                                                            load_closure_array_element(active_layer, sc, idx));
                            } else if (detail::is_simple_type(dc->type(), SimpleType::Primitive::STRING)) {
                                store_string_array_element(active_layer, dc, idx,
                                                           load_string_array_element(active_layer, sc, idx));
                            } else {
                                DualValue ev = load_array_element(active_layer, sc, idx, cwd);
                                store_array_element(active_layer, dc, idx,
                                                    convert_dual(ev, sc->type(), dc->type(), cwd));
                            }
                        }
                        continue;
                    }
                    if (dc->type()->tag() == Type::Tag::CLOSURE && sc->type()->tag() == Type::Tag::CLOSURE) {
                        store_closure(active_layer, dc, load_closure(active_layer, sc));
                    } else if (detail::is_simple_type(dc->type(), SimpleType::Primitive::STRING) &&
                               detail::is_simple_type(sc->type(), SimpleType::Primitive::STRING)) {
                        store_string_token(active_layer, dc, load_string_token(active_layer, sc));
                    } else {
                        bool cwd = dst_wants_deriv(*this, active_layer, dc);
                        DualValue sv = load_symbol(active_layer, sc, cwd);
                        store_symbol(active_layer, dc, convert_dual(sv, sc->type(), dc->type(), cwd));
                    }
                }
                return true;
            };
            return assign_struct(dst, src, assign_struct);
        }
        if (dst->is_array() && src->is_array()) {
            int dst_len = effective_array_length(active_layer, dst);
            int src_len = effective_array_length(active_layer, src);
            int n = std::min(dst_len, src_len);
            bool wd = dst_wants_deriv(*this, active_layer, dst);
            if (dst->type()->tag() == Type::Tag::CLOSURE) {
                const auto *dst_st = find_storage(active_layer, dst);
                const auto *src_st = find_storage(active_layer, src);
                const compute::Type *ct = owner->closure_xir_type();
                if (dst_st && src_st && dst_st->val_ptr && src_st->val_ptr) {
                    for (int k = 0; k < n; ++k) {
                        auto *sv = builder.load(ct, builder.gep(ct, src_st->val_ptr, {const_int(k)}));
                        builder.store(builder.gep(ct, dst_st->val_ptr, {const_int(k)}), sv);
                    }
                    return true;
                }
                return n == 0;
            }
            if (detail::is_simple_type(dst->type(), SimpleType::Primitive::STRING)) {
                const auto *dst_st = find_storage(active_layer, dst);
                const auto *src_st = find_storage(active_layer, src);
                auto *u_t = compute::Type::of<uint32_t>();
                if (dst_st && src_st && dst_st->val_ptr && src_st->val_ptr) {
                    for (int k = 0; k < n; ++k) {
                        auto *sv = builder.load(u_t, builder.gep(u_t, src_st->val_ptr, {const_int(k)}));
                        builder.store(builder.gep(u_t, dst_st->val_ptr, {const_int(k)}), sv);
                    }
                    return true;
                }
                return n == 0;
            }
            for (int k = 0; k < n; ++k) {
                DualValue sv = load_array_element(active_layer, src, const_int(k), wd);
                DualValue cv = convert_dual(sv, src->type(), dst->type(), wd);
                store_array_element(active_layer, dst, const_int(k), cv);
            }
            return true;
        }
        if (detail::is_simple_type(dst->type(), SimpleType::Primitive::STRING)) {
            if (!detail::is_simple_type(src->type(), SimpleType::Primitive::STRING)) { return false; }
            store_string_token(active_layer, dst, get_arg_string_token(*this, active_layer, ia, 1u, src));
            return true;
        }
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        DualValue sv = load_symbol(active_layer, src, wd);
        store_symbol(active_layer, dst, convert_dual(sv, src->type(), dst->type(), wd));
        return true;
    }

    // 3. Triple constructors: point / vector / normal / color
    if (op == "point" || op == "vector" || op == "normal" || op == "color") {
        if (args.size() < 2u || args.size() > 5u) { return false; }
        const Symbol *dst = args[0];
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        bool has_space = detail::is_simple_type(args[1]->type(), SimpleType::Primitive::STRING);
        size_t num_start = has_space ? 2u : 1u;
        size_t num_count = args.size() - num_start;
        if (num_count != 1u && num_count != 3u) { return false; }

        auto make_triple = [&]() -> DualValue {
            if (num_count == 1u) {
                DualValue s = load_symbol(active_layer, args[num_start], wd);
                return convert_dual(s, args[num_start]->type(), dst->type(), wd);
            }
            DualValue c0 = convert_dual(load_symbol(active_layer, args[num_start + 0u], wd),
                                        args[num_start + 0u]->type(), osl_float(), wd);
            DualValue c1 = convert_dual(load_symbol(active_layer, args[num_start + 1u], wd),
                                        args[num_start + 1u]->type(), osl_float(), wd);
            DualValue c2 = convert_dual(load_symbol(active_layer, args[num_start + 2u], wd),
                                        args[num_start + 2u]->type(), osl_float(), wd);
            auto *f3_t = compute::Type::of<luisa::float3>();
            DualValue out;
            out.val = builder.call(f3_t, xir::ArithmeticOp::AGGREGATE, {c0.val, c1.val, c2.val});
            if (wd && (c0.has_derivs() || c1.has_derivs() || c2.has_derivs())) {
                if (c0.dx || c1.dx || c2.dx) {
                    out.dx = builder.call(f3_t, xir::ArithmeticOp::AGGREGATE,
                                          {c0.dx ? c0.dx : const_float(0.0f),
                                           c1.dx ? c1.dx : const_float(0.0f),
                                           c2.dx ? c2.dx : const_float(0.0f)});
                }
                if (c0.dy || c1.dy || c2.dy) {
                    out.dy = builder.call(f3_t, xir::ArithmeticOp::AGGREGATE,
                                          {c0.dy ? c0.dy : const_float(0.0f),
                                           c1.dy ? c1.dy : const_float(0.0f),
                                           c2.dy ? c2.dy : const_float(0.0f)});
                }
            }
            return out;
        };

        DualValue v = make_triple();
        if (!has_space) {
            store_symbol(active_layer, dst, v);
            return true;
        }
        StringSlotSpec space_slot = make_string_slot(*this, active_layer, ia, 1u, args[1]);
        dispatch_strings_1(
            *this, space_slot,
            [&](luisa::span<const luisa::string_view> resolved) {
                DualValue tr;
                if (op == "color") {
                    tr = owner->emit_transform_color(resolved[0], "rgb", v, wd);
                } else {
                    TransformKind tk = TransformKind::POINT;
                    if (op == "vector") { tk = TransformKind::VECTOR; }
                    else if (op == "normal") { tk = TransformKind::NORMAL; }
                    tr = owner->emit_transform_space(resolved[0], "common", tk, v, wd);
                }
                if (errors.empty() && tr.valid()) {
                    store_symbol(active_layer, dst, tr);
                }
            });
        return errors.empty();
    }

    // 4. Matrix constructor
    if (op == "matrix") {
        if (args.size() < 2u) { return false; }
        const Symbol *dst = args[0];
        if (args.size() == 3u &&
            detail::is_simple_type(args[1]->type(), SimpleType::Primitive::STRING) &&
            detail::is_simple_type(args[2]->type(), SimpleType::Primitive::STRING)) {
            StringSlotSpec s_from = make_string_slot(*this, active_layer, ia, 1u, args[1]);
            StringSlotSpec s_to = make_string_slot(*this, active_layer, ia, 2u, args[2]);
            dispatch_strings_2(
                *this, s_from, s_to,
                [&](luisa::span<const luisa::string_view> resolved) {
                    xir::Value *mat_L = nullptr;
                    xir::Value *ok = owner->emit_getmatrix(resolved[0], resolved[1], mat_L);
                    if (!errors.empty()) { return; }
                    xir::Value *safe_m = mat_L ? mat_L : const_float4x4(luisa::make_float4x4(0.0f));
                    xir::Value *ok_b = to_bool(ok);
                    xir::Value *z_m = const_float4x4(luisa::make_float4x4(0.0f));
                    auto *f4_t = compute::Type::of<luisa::float4>();
                    xir::Value *cols[4]{};
                    for (uint32_t c = 0u; c < 4u; ++c) {
                        xir::Value *idx = const_uint(c);
                        xir::Value *zc = builder.call(f4_t, xir::ArithmeticOp::EXTRACT, {z_m, idx});
                        xir::Value *mc = builder.call(f4_t, xir::ArithmeticOp::EXTRACT, {safe_m, idx});
                        cols[c] = builder.call(f4_t, xir::ArithmeticOp::SELECT, {zc, mc, ok_b});
                    }
                    xir::Value *sel = builder.call(compute::Type::of<luisa::float4x4>(), xir::ArithmeticOp::AGGREGATE,
                                                   {cols[0], cols[1], cols[2], cols[3]});
                    store_symbol(active_layer, dst, {.val = sel});
                });
            return errors.empty();
        }
        bool has_space = detail::is_simple_type(args[1]->type(), SimpleType::Primitive::STRING);
        size_t num_start = has_space ? 2u : 1u;
        size_t num_count = args.size() - num_start;
        if (num_count != 1u && num_count != 16u) { return false; }

        xir::Value *mat_L = nullptr;
        if (num_count == 1u) {
            mat_L = to_float4x4(load_symbol(active_layer, args[num_start], false).val);
        } else {
            auto *f4_t = compute::Type::of<luisa::float4>();
            xir::Value *cols[4]{};
            for (size_t r = 0u; r < 4u; ++r) {
                xir::Value *c0 = to_float(load_symbol(active_layer, args[num_start + r * 4u + 0u], false).val);
                xir::Value *c1 = to_float(load_symbol(active_layer, args[num_start + r * 4u + 1u], false).val);
                xir::Value *c2 = to_float(load_symbol(active_layer, args[num_start + r * 4u + 2u], false).val);
                xir::Value *c3 = to_float(load_symbol(active_layer, args[num_start + r * 4u + 3u], false).val);
                cols[r] = builder.call(f4_t, xir::ArithmeticOp::AGGREGATE, {c0, c1, c2, c3});
            }
            mat_L = builder.call(compute::Type::of<luisa::float4x4>(), xir::ArithmeticOp::AGGREGATE,
                                 {cols[0], cols[1], cols[2], cols[3]});
        }

        if (!has_space) {
            store_symbol(active_layer, dst, {.val = mat_L});
            return true;
        }
        StringSlotSpec space_slot = make_string_slot(*this, active_layer, ia, 1u, args[1]);
        dispatch_strings_1(
            *this, space_slot,
            [&](luisa::span<const luisa::string_view> resolved) {
                DualValue tr = owner->emit_transform_space(resolved[0], "common", TransformKind::MATRIX, {.val = mat_L}, false);
                if (errors.empty() && tr.valid()) {
                    store_symbol(active_layer, dst, tr);
                }
            });
        return errors.empty();
    }

    // 5. Closure algebra on add / mul / mix
    if (args.size() >= 3u && args[0]->type()->tag() == Type::Tag::CLOSURE) {
        const Symbol *dst = args[0];
        if (op == "add" && args.size() == 3u) {
            ClosureValue c0 = load_closure(active_layer, args[1]);
            ClosureValue c1 = load_closure(active_layer, args[2]);
            store_closure(active_layer, dst, owner->emit_closure_add(c0, c1));
            return errors.empty();
        }
        if (op == "mul" && args.size() == 3u) {
            ClosureValue cl{};
            const Symbol *w_sym = nullptr;
            if (args[1]->type()->tag() == Type::Tag::CLOSURE) {
                cl = load_closure(active_layer, args[1]);
                w_sym = args[2];
            } else if (args[2]->type()->tag() == Type::Tag::CLOSURE) {
                cl = load_closure(active_layer, args[2]);
                w_sym = args[1];
            }
            if (w_sym != nullptr) {
                xir::Value *w = to_float3(load_symbol(active_layer, w_sym, false).val);
                store_closure(active_layer, dst, owner->emit_closure_scale(cl, w));
                return errors.empty();
            }
        }
        if (op == "mix" && args.size() == 4u) {
            ClosureValue c0 = load_closure(active_layer, args[1]);
            ClosureValue c1 = load_closure(active_layer, args[2]);
            xir::Value *a = to_float3(load_symbol(active_layer, args[3], false).val);
            xir::Value *one = const_float3(luisa::float3{1.0f});
            xir::Value *inv_a = builder.call(compute::Type::of<luisa::float3>(),
                                             xir::ArithmeticOp::BINARY_SUB, {one, a});
            ClosureValue sc0 = owner->emit_closure_scale(c0, inv_a);
            ClosureValue sc1 = owner->emit_closure_scale(c1, a);
            store_closure(active_layer, dst, owner->emit_closure_add(sc0, sc1));
            return errors.empty();
        }
    }

    // 6. Basic binary arithmetic (add, sub, mul, div, mod, fmod)
    if ((op == "add" || op == "sub" || op == "mul" || op == "div" || op == "mod" || op == "fmod") &&
        args.size() == 3u) {
        const Symbol *dst = args[0];
        const Symbol *sa = args[1];
        const Symbol *sb = args[2];
        bool wd = dst_wants_deriv(*this, active_layer, dst);

        // Matrix special cases
        bool dst_is_mat = detail::is_simple_type(dst->type(), SimpleType::Primitive::MATRIX);
        bool a_is_mat = detail::is_simple_type(sa->type(), SimpleType::Primitive::MATRIX);
        bool b_is_mat = detail::is_simple_type(sb->type(), SimpleType::Primitive::MATRIX);
        if (dst_is_mat || a_is_mat || b_is_mat) {
            if (!dst_is_mat) { return false; }
            xir::Value *va = load_symbol(active_layer, sa, false).val;
            xir::Value *vb = load_symbol(active_layer, sb, false).val;
            xir::Value *res = nullptr;
            if (op == "add") {
                res = map_matrix_cols(*this, a_is_mat ? va : to_float4x4(va),
                                      b_is_mat ? vb : to_float4x4(vb), xir::ArithmeticOp::BINARY_ADD);
            } else if (op == "sub") {
                res = map_matrix_cols(*this, a_is_mat ? va : to_float4x4(va),
                                      b_is_mat ? vb : to_float4x4(vb), xir::ArithmeticOp::BINARY_SUB);
            } else if (op == "mul") {
                if (a_is_mat && b_is_mat) {
                    // (A * B)^T = B^T * A^T
                    res = builder.call(compute::Type::of<luisa::float4x4>(),
                                       xir::ArithmeticOp::MATRIX_LINALG_MUL, {vb, va});
                } else if (a_is_mat) {
                    res = scale_matrix_scalar(*this, va, vb);
                } else {
                    res = scale_matrix_scalar(*this, vb, va);
                }
            } else if (op == "div") {
                if (a_is_mat && b_is_mat) {
                    xir::Value *inv_b = builder.call(compute::Type::of<luisa::float4x4>(),
                                                     xir::ArithmeticOp::MATRIX_INVERSE, {vb});
                    res = builder.call(compute::Type::of<luisa::float4x4>(),
                                       xir::ArithmeticOp::MATRIX_LINALG_MUL, {inv_b, va});
                } else if (a_is_mat) {
                    xir::Value *inv_s = safe_div_val(*this, const_float(1.0f), to_float(vb));
                    res = scale_matrix_scalar(*this, va, inv_s);
                } else {
                    xir::Value *inv_m = builder.call(compute::Type::of<luisa::float4x4>(),
                                                     xir::ArithmeticOp::MATRIX_INVERSE, {vb});
                    res = scale_matrix_scalar(*this, inv_m, va);
                }
            } else {
                return false;
            }
            store_symbol(active_layer, dst, {.val = res});
            return true;
        }

        // Pure integer arithmetic
        bool all_int = detail::is_simple_type(dst->type(), SimpleType::Primitive::INT) &&
                       detail::is_simple_type(sa->type(), SimpleType::Primitive::INT) &&
                       detail::is_simple_type(sb->type(), SimpleType::Primitive::INT);
        if (all_int && op != "fmod") {
            xir::Value *ia_v = load_symbol(active_layer, sa, false).val;
            xir::Value *ib_v = load_symbol(active_layer, sb, false).val;
            auto *i_t = compute::Type::of<int32_t>();
            xir::Value *r = nullptr;
            if (op == "add") { r = builder.call(i_t, xir::ArithmeticOp::BINARY_ADD, {ia_v, ib_v}); }
            else if (op == "sub") { r = builder.call(i_t, xir::ArithmeticOp::BINARY_SUB, {ia_v, ib_v}); }
            else if (op == "mul") { r = builder.call(i_t, xir::ArithmeticOp::BINARY_MUL, {ia_v, ib_v}); }
            else if (op == "div") { r = safe_div_val(*this, ia_v, ib_v); }
            else if (op == "mod") {
                auto *b_t = compute::Type::of<bool>();
                xir::Value *z = const_int(0);
                xir::Value *is_zero = builder.call(b_t, xir::ArithmeticOp::BINARY_EQUAL, {ib_v, z});
                xir::Value *is_min = builder.call(b_t, xir::ArithmeticOp::BINARY_EQUAL, {ia_v, const_int(INT32_MIN)});
                xir::Value *is_m1 = builder.call(b_t, xir::ArithmeticOp::BINARY_EQUAL, {ib_v, const_int(-1)});
                xir::Value *is_min_m1 = builder.call(b_t, xir::ArithmeticOp::BINARY_BIT_AND, {is_min, is_m1});
                xir::Value *bad = builder.call(b_t, xir::ArithmeticOp::BINARY_BIT_OR, {is_zero, is_min_m1});
                xir::Value *safe_b = builder.call(i_t, xir::ArithmeticOp::SELECT, {ib_v, const_int(1), bad});
                xir::Value *m = builder.call(i_t, xir::ArithmeticOp::BINARY_MOD, {ia_v, safe_b});
                r = builder.call(i_t, xir::ArithmeticOp::SELECT, {m, z, bad});
            }
            store_symbol(active_layer, dst, {.val = r});
            return true;
        }

        // Float / Triple arithmetic (widen inputs symmetrically before XIR op)
        const Type *common_osl = widen_osl_arithmetic_type(sa->type(), sb->type(), dst->type());
        if (detail::is_simple_type(common_osl, SimpleType::Primitive::INT)) {
            common_osl = osl_float();
        }
        DualValue da = convert_dual(load_symbol(active_layer, sa, wd), sa->type(), common_osl, wd);
        DualValue db = convert_dual(load_symbol(active_layer, sb, wd), sb->type(), common_osl, wd);
        const compute::Type *xt = da.val->type();
        auto bin = [&](xir::ArithmeticOp aop, xir::Value *x, xir::Value *y) {
            return builder.call(xt, aop, {x, y});
        };

        DualValue res;
        if (op == "add") {
            res.val = bin(xir::ArithmeticOp::BINARY_ADD, da.val, db.val);
            if (wd) {
                if (da.dx && db.dx) { res.dx = bin(xir::ArithmeticOp::BINARY_ADD, da.dx, db.dx); }
                else { res.dx = da.dx ? da.dx : db.dx; }
                if (da.dy && db.dy) { res.dy = bin(xir::ArithmeticOp::BINARY_ADD, da.dy, db.dy); }
                else { res.dy = da.dy ? da.dy : db.dy; }
            }
        } else if (op == "sub") {
            res.val = bin(xir::ArithmeticOp::BINARY_SUB, da.val, db.val);
            if (wd) {
                if (da.dx && db.dx) { res.dx = bin(xir::ArithmeticOp::BINARY_SUB, da.dx, db.dx); }
                else if (da.dx) { res.dx = da.dx; }
                else if (db.dx) { res.dx = builder.call(xt, xir::ArithmeticOp::UNARY_MINUS, {db.dx}); }
                if (da.dy && db.dy) { res.dy = bin(xir::ArithmeticOp::BINARY_SUB, da.dy, db.dy); }
                else if (da.dy) { res.dy = da.dy; }
                else if (db.dy) { res.dy = builder.call(xt, xir::ArithmeticOp::UNARY_MINUS, {db.dy}); }
            }
        } else if (op == "mul") {
            res.val = bin(xir::ArithmeticOp::BINARY_MUL, da.val, db.val);
            if (wd) {
                auto prod_deriv = [&](xir::Value *dxa, xir::Value *dxb) -> xir::Value * {
                    xir::Value *t1 = dxa ? bin(xir::ArithmeticOp::BINARY_MUL, dxa, db.val) : nullptr;
                    xir::Value *t2 = dxb ? bin(xir::ArithmeticOp::BINARY_MUL, da.val, dxb) : nullptr;
                    if (t1 && t2) { return bin(xir::ArithmeticOp::BINARY_ADD, t1, t2); }
                    return t1 ? t1 : t2;
                };
                res.dx = prod_deriv(da.dx, db.dx);
                res.dy = prod_deriv(da.dy, db.dy);
            }
        } else if (op == "div") {
            res.val = safe_div_val(*this, da.val, db.val);
            if (wd && (da.has_derivs() || db.has_derivs())) {
                xir::Value *b2 = bin(xir::ArithmeticOp::BINARY_MUL, db.val, db.val);
                auto quot_deriv = [&](xir::Value *dxa, xir::Value *dxb) -> xir::Value * {
                    if (!dxa && !dxb) { return nullptr; }
                    xir::Value *num = nullptr;
                    if (dxa && dxb) {
                        num = bin(xir::ArithmeticOp::BINARY_SUB,
                                  bin(xir::ArithmeticOp::BINARY_MUL, dxa, db.val),
                                  bin(xir::ArithmeticOp::BINARY_MUL, da.val, dxb));
                    } else if (dxa) {
                        num = bin(xir::ArithmeticOp::BINARY_MUL, dxa, db.val);
                    } else {
                        num = builder.call(xt, xir::ArithmeticOp::UNARY_MINUS,
                                           {bin(xir::ArithmeticOp::BINARY_MUL, da.val, dxb)});
                    }
                    return safe_div_val(*this, num, b2);
                };
                res.dx = quot_deriv(da.dx, db.dx);
                res.dy = quot_deriv(da.dy, db.dy);
            }
        } else {
            // mod (floor) / fmod (trunc)
            xir::Value *q = safe_div_val(*this, da.val, db.val);
            xir::Value *iq = builder.call(
                xt, op == "mod" ? xir::ArithmeticOp::FLOOR : xir::ArithmeticOp::TRUNC, {q});
            xir::Value *rem = bin(xir::ArithmeticOp::BINARY_SUB, da.val,
                                  bin(xir::ArithmeticOp::BINARY_MUL, db.val, iq));
            xir::Value *z = splat_scalar_for(*this, xt, 0.0f);
            xir::Value *nz = builder.call(bool_type_for(xt), xir::ArithmeticOp::BINARY_NOT_EQUAL, {db.val, z});
            res.val = builder.call(xt, xir::ArithmeticOp::SELECT, {z, rem, nz});
            if (wd) {
                auto mod_deriv = [&](xir::Value *dxa, xir::Value *dxb) -> xir::Value * {
                    if (!dxa && !dxb) { return nullptr; }
                    xir::Value *d = dxa ? dxa : z;
                    if (dxb) {
                        d = bin(xir::ArithmeticOp::BINARY_SUB, d, bin(xir::ArithmeticOp::BINARY_MUL, dxb, iq));
                    }
                    return builder.call(xt, xir::ArithmeticOp::SELECT, {z, d, nz});
                };
                res.dx = mod_deriv(da.dx, db.dx);
                res.dy = mod_deriv(da.dy, db.dy);
            }
        }
        store_symbol(active_layer, dst, convert_dual(res, common_osl, dst->type(), wd));
        return true;
    }

    // 7. Matrix unary neg / determinant / transpose
    if ((op == "neg" || op == "determinant" || op == "transpose") && args.size() == 2u &&
        detail::is_simple_type(args[1]->type(), SimpleType::Primitive::MATRIX)) {
        const Symbol *dst = args[0];
        xir::Value *mat_L = load_symbol(active_layer, args[1], false).val;
        if (op == "neg") {
            store_symbol(active_layer, dst, {.val = map_matrix_cols(*this, mat_L, nullptr, xir::ArithmeticOp::UNARY_MINUS)});
            return true;
        }
        if (op == "determinant") {
            xir::Value *det = builder.call(compute::Type::of<float>(), xir::ArithmeticOp::MATRIX_DETERMINANT, {mat_L});
            store_symbol(active_layer, dst, convert_dual({.val = det}, osl_float(), dst->type(), false));
            return true;
        }
        if (op == "transpose") {
            xir::Value *tr = builder.call(compute::Type::of<luisa::float4x4>(), xir::ArithmeticOp::MATRIX_TRANSPOSE, {mat_L});
            store_symbol(active_layer, dst, {.val = tr});
            return true;
        }
    }

    // 8. Bitwise & logical operations
    if ((op == "bitand" || op == "bitor" || op == "xor" || op == "shl" || op == "shr" ||
         op == "and" || op == "or") &&
        args.size() == 3u) {
        const Symbol *dst = args[0];
        if (op == "and" || op == "or") {
            xir::Value *b0 = to_bool(load_symbol(active_layer, args[1], false).val);
            xir::Value *b1 = to_bool(load_symbol(active_layer, args[2], false).val);
            xir::Value *br = builder.call(
                compute::Type::of<bool>(),
                op == "and" ? xir::ArithmeticOp::BINARY_BIT_AND : xir::ArithmeticOp::BINARY_BIT_OR,
                {b0, b1});
            store_symbol(active_layer, dst, convert_dual({.val = to_int(br)}, osl_int(), dst->type(), false));
            return true;
        }
        xir::Value *i0 = to_int(load_symbol(active_layer, args[1], false).val);
        xir::Value *i1 = to_int(load_symbol(active_layer, args[2], false).val);
        xir::ArithmeticOp aop = xir::ArithmeticOp::BINARY_BIT_AND;
        if (op == "bitor") { aop = xir::ArithmeticOp::BINARY_BIT_OR; }
        else if (op == "xor") { aop = xir::ArithmeticOp::BINARY_BIT_XOR; }
        else if (op == "shl") { aop = xir::ArithmeticOp::BINARY_SHIFT_LEFT; }
        else if (op == "shr") { aop = xir::ArithmeticOp::BINARY_SHIFT_RIGHT; }
        xir::Value *r = builder.call(compute::Type::of<int32_t>(), aop, {i0, i1});
        store_symbol(active_layer, dst, convert_dual({.val = r}, osl_int(), dst->type(), false));
        return true;
    }
    if ((op == "compl" || op == "not") && args.size() == 2u) {
        const Symbol *dst = args[0];
        if (op == "not") {
            xir::Value *b = to_bool(load_symbol(active_layer, args[1], false).val);
            xir::Value *nb = builder.call(compute::Type::of<bool>(), xir::ArithmeticOp::BINARY_EQUAL,
                                          {b, const_bool(false)});
            store_symbol(active_layer, dst, convert_dual({.val = to_int(nb)}, osl_int(), dst->type(), false));
            return true;
        }
        xir::Value *iv = to_int(load_symbol(active_layer, args[1], false).val);
        xir::Value *r = builder.call(compute::Type::of<int32_t>(), xir::ArithmeticOp::UNARY_BIT_NOT, {iv});
        store_symbol(active_layer, dst, convert_dual({.val = r}, osl_int(), dst->type(), false));
        return true;
    }

    // 9. Comparisons (eq, neq, lt, gt, le, ge)
    if ((op == "eq" || op == "neq" || op == "lt" || op == "gt" || op == "le" || op == "ge") &&
        args.size() == 3u) {
        const Symbol *dst = args[0];
        const Symbol *sa = args[1];
        const Symbol *sb = args[2];

        if (detail::is_simple_type(sa->type(), SimpleType::Primitive::STRING) ||
            detail::is_simple_type(sb->type(), SimpleType::Primitive::STRING)) {
            if (op == "eq" || op == "neq") {
                xir::Value *ta = get_arg_string_token(*this, active_layer, ia, 1u, sa);
                xir::Value *tb = get_arg_string_token(*this, active_layer, ia, 2u, sb);
                xir::Value *b = builder.call(
                    compute::Type::of<bool>(),
                    op == "eq" ? xir::ArithmeticOp::BINARY_EQUAL : xir::ArithmeticOp::BINARY_NOT_EQUAL,
                    {ta, tb});
                store_symbol(active_layer, dst, convert_dual({.val = to_int(b)}, osl_int(), dst->type(), false));
                return true;
            }
            if (op == "lt" || op == "gt" || op == "le" || op == "ge") {
                StringSlotSpec slot_a = make_string_slot(*this, active_layer, ia, 1u, sa);
                StringSlotSpec slot_b = make_string_slot(*this, active_layer, ia, 2u, sb);
                dispatch_strings_2(
                    *this, slot_a, slot_b,
                    [&](luisa::span<const luisa::string_view> resolved) {
                        bool res = false;
                        if (op == "lt") { res = resolved[0] < resolved[1]; }
                        else if (op == "gt") { res = resolved[0] > resolved[1]; }
                        else if (op == "le") { res = resolved[0] <= resolved[1]; }
                        else { res = resolved[0] >= resolved[1]; }
                        store_symbol(active_layer, dst,
                                     convert_dual({.val = const_int(res ? 1 : 0)}, osl_int(), dst->type(), false));
                    });
                return errors.empty();
            }
            return false;
        }
        if (detail::is_simple_type(sa->type(), SimpleType::Primitive::MATRIX) ||
            detail::is_simple_type(sb->type(), SimpleType::Primitive::MATRIX)) {
            if (op != "eq" && op != "neq") { return false; }
            xir::Value *ma = to_float4x4(load_symbol(active_layer, sa, false).val);
            xir::Value *mb = to_float4x4(load_symbol(active_layer, sb, false).val);
            xir::Value *eq = matrix_equal(*this, ma, mb);
            xir::Value *b = (op == "eq") ? eq
                                         : builder.call(compute::Type::of<bool>(), xir::ArithmeticOp::BINARY_EQUAL,
                                                        {eq, const_bool(false)});
            store_symbol(active_layer, dst, convert_dual({.val = to_int(b)}, osl_int(), dst->type(), false));
            return true;
        }
        if (sa->type()->tag() == Type::Tag::CLOSURE || sb->type()->tag() == Type::Tag::CLOSURE) {
            // In OSL, closure == 0 / closure != 0 tests empty closure or equality via weight == 0?
            // Not commonly emitted, fail closed if unexpected.
            return false;
        }

        const Type *common_osl = widen_osl_arithmetic_type(sa->type(), sb->type(), nullptr);
        xir::Value *va = convert_dual(load_symbol(active_layer, sa, false), sa->type(), common_osl, false).val;
        xir::Value *vb = convert_dual(load_symbol(active_layer, sb, false), sb->type(), common_osl, false).val;
        const compute::Type *bt = bool_type_for(va->type());
        xir::ArithmeticOp aop = xir::ArithmeticOp::BINARY_EQUAL;
        if (op == "neq") { aop = xir::ArithmeticOp::BINARY_NOT_EQUAL; }
        else if (op == "lt") { aop = xir::ArithmeticOp::BINARY_LESS; }
        else if (op == "gt") { aop = xir::ArithmeticOp::BINARY_GREATER; }
        else if (op == "le") { aop = xir::ArithmeticOp::BINARY_LESS_EQUAL; }
        else if (op == "ge") { aop = xir::ArithmeticOp::BINARY_GREATER_EQUAL; }
        xir::Value *cmp_v = builder.call(bt, aop, {va, vb});
        xir::Value *b = (op == "neq") ? any_component_nonzero(*this, cmp_v)
                                      : all_components_true(*this, cmp_v);
        store_symbol(active_layer, dst, convert_dual({.val = to_int(b)}, osl_int(), dst->type(), false));
        return true;
    }

    // 10. Unary math + float/int abs/neg/sign/floor/ceil/round/trunc/trig/exp/log/sqrt/erf
    if ((op == "neg" || op == "abs" || op == "fabs" || op == "sign" ||
         op == "floor" || op == "ceil" || op == "round" || op == "trunc" ||
         op == "sin" || op == "cos" || op == "tan" ||
         op == "asin" || op == "acos" || op == "atan" ||
         op == "sinh" || op == "cosh" || op == "tanh" ||
         op == "exp" || op == "exp2" || op == "expm1" ||
         op == "log" || op == "log2" || op == "log10" || op == "logb" ||
         op == "sqrt" || op == "inversesqrt" || op == "cbrt" ||
         op == "erf" || op == "erfc") &&
        args.size() == 2u) {
        const Symbol *dst = args[0];
        const Symbol *src = args[1];
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        if ((op == "neg" || op == "abs" || op == "fabs" || op == "sign") &&
            detail::is_simple_type(dst->type(), SimpleType::Primitive::INT) &&
            detail::is_simple_type(src->type(), SimpleType::Primitive::INT)) {
            xir::Value *iv = load_symbol(active_layer, src, false).val;
            auto *i_t = compute::Type::of<int32_t>();
            auto *b_t = compute::Type::of<bool>();
            xir::Value *r = nullptr;
            if (op == "neg") { r = builder.call(i_t, xir::ArithmeticOp::UNARY_MINUS, {iv}); }
            else if (op == "abs" || op == "fabs") { r = builder.call(i_t, xir::ArithmeticOp::ABS, {iv}); }
            else {
                xir::Value *gt0 = builder.call(b_t, xir::ArithmeticOp::BINARY_GREATER, {iv, const_int(0)});
                xir::Value *lt0 = builder.call(b_t, xir::ArithmeticOp::BINARY_LESS, {iv, const_int(0)});
                r = builder.call(i_t, xir::ArithmeticOp::SELECT,
                                 {builder.call(i_t, xir::ArithmeticOp::SELECT, {const_int(0), const_int(1), gt0}),
                                  const_int(-1), lt0});
            }
            store_symbol(active_layer, dst, {.val = r});
            return true;
        }
        const Type *eval_osl = detail::is_simple_type(dst->type(), SimpleType::Primitive::INT)
                                   ? osl_float()
                                   : dst->type();
        DualValue s = convert_dual(load_symbol(active_layer, src, wd), src->type(), eval_osl, wd);
        DualValue r = unary_float_math(*this, op, s, wd);
        store_symbol(active_layer, dst, convert_dual(r, eval_osl, dst->type(), wd));
        return true;
    }

    // 11. sincos (angle, out_sin, out_cos)
    if (op == "sincos" && args.size() == 3u) {
        const Symbol *angle = args[0];
        const Symbol *out_s = args[1];
        const Symbol *out_c = args[2];
        bool wd_s = dst_wants_deriv(*this, active_layer, out_s);
        bool wd_c = dst_wants_deriv(*this, active_layer, out_c);
        bool wd = wd_s || wd_c;
        const Type *eval_osl = detail::is_triple_type(angle->type()) ? angle->type() : osl_float();
        DualValue a = convert_dual(load_symbol(active_layer, angle, wd), angle->type(), eval_osl, wd);
        DualValue sv = unary_float_math(*this, "sin", a, wd_s);
        DualValue cv = unary_float_math(*this, "cos", a, wd_c);
        store_symbol(active_layer, out_s, convert_dual(sv, eval_osl, out_s->type(), wd_s));
        store_symbol(active_layer, out_c, convert_dual(cv, eval_osl, out_c->type(), wd_c));
        return true;
    }

    // 12. atan2 & pow & step
    if ((op == "atan2" || op == "pow" || op == "step") && args.size() == 3u) {
        const Symbol *dst = args[0];
        const Symbol *sa = args[1];
        const Symbol *sb = args[2];
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        const Type *eval_osl = widen_osl_arithmetic_type(sa->type(), sb->type(), dst->type());
        if (detail::is_simple_type(eval_osl, SimpleType::Primitive::INT)) { eval_osl = osl_float(); }
        DualValue da = convert_dual(load_symbol(active_layer, sa, wd), sa->type(), eval_osl, wd);
        DualValue db = convert_dual(load_symbol(active_layer, sb, wd), sb->type(), eval_osl, wd);
        const compute::Type *xt = da.val->type();
        const compute::Type *bt = bool_type_for(xt);
        auto bin = [&](xir::ArithmeticOp aop, xir::Value *x, xir::Value *y) {
            return builder.call(xt, aop, {x, y});
        };
        auto splat = [&](float s) { return splat_scalar_for(*this, xt, s); };

        DualValue out;
        if (op == "step") {
            out.val = bin(xir::ArithmeticOp::STEP, da.val, db.val);
        } else if (op == "atan2") {
            out.val = bin(xir::ArithmeticOp::ATAN2, da.val, db.val);
            if (wd && (da.has_derivs() || db.has_derivs())) {
                xir::Value *r2 = bin(xir::ArithmeticOp::BINARY_ADD,
                                     bin(xir::ArithmeticOp::BINARY_MUL, da.val, da.val),
                                     bin(xir::ArithmeticOp::BINARY_MUL, db.val, db.val));
                auto deriv = [&](xir::Value *dy_a, xir::Value *dx_b) -> xir::Value * {
                    if (!dy_a && !dx_b) { return nullptr; }
                    xir::Value *t1 = dy_a ? bin(xir::ArithmeticOp::BINARY_MUL, db.val, dy_a) : nullptr;
                    xir::Value *t2 = dx_b ? bin(xir::ArithmeticOp::BINARY_MUL, da.val, dx_b) : nullptr;
                    xir::Value *num = (t1 && t2) ? bin(xir::ArithmeticOp::BINARY_SUB, t1, t2)
                                                 : (t1 ? t1 : builder.call(xt, xir::ArithmeticOp::UNARY_MINUS, {t2}));
                    return safe_div_val(*this, num, r2);
                };
                out.dx = deriv(da.dx, db.dx);
                out.dy = deriv(da.dy, db.dy);
            }
        } else {
            // Portable GPU-safe pow(a, b): avoid negative-base or 0^0 NaN in DXC/SPIR-V
            xir::Value *abs_a = builder.call(xt, xir::ArithmeticOp::ABS, {da.val});
            xir::Value *safe_pos = bin(xir::ArithmeticOp::MAX, abs_a, splat(1e-30f));
            xir::Value *p_abs = bin(xir::ArithmeticOp::POW, safe_pos, db.val);

            xir::Value *a_gt0 = builder.call(bt, xir::ArithmeticOp::BINARY_GREATER, {da.val, splat(0.0f)});
            xir::Value *a_eq0 = builder.call(bt, xir::ArithmeticOp::BINARY_EQUAL, {da.val, splat(0.0f)});
            xir::Value *b_eq0 = builder.call(bt, xir::ArithmeticOp::BINARY_EQUAL, {db.val, splat(0.0f)});
            xir::Value *b_gt0 = builder.call(bt, xir::ArithmeticOp::BINARY_GREATER, {db.val, splat(0.0f)});
            xir::Value *zero_nonneg_val = builder.call(xt, xir::ArithmeticOp::SELECT, {splat(0.0f), splat(1.0f), b_eq0});
            xir::Value *b_ge0 = builder.call(bt, xir::ArithmeticOp::BINARY_BIT_OR, {b_eq0, b_gt0});
            xir::Value *zero_res = builder.call(xt, xir::ArithmeticOp::SELECT, {p_abs, zero_nonneg_val, b_ge0});

            xir::Value *b_trunc = builder.call(xt, xir::ArithmeticOp::TRUNC, {db.val});
            xir::Value *is_int = builder.call(bt, xir::ArithmeticOp::BINARY_EQUAL, {db.val, b_trunc});
            xir::Value *half_b = bin(xir::ArithmeticOp::BINARY_MUL, db.val, splat(0.5f));
            xir::Value *half_trunc = builder.call(xt, xir::ArithmeticOp::TRUNC, {half_b});
            xir::Value *is_odd = builder.call(bt, xir::ArithmeticOp::BINARY_NOT_EQUAL, {half_b, half_trunc});
            xir::Value *neg_p = builder.call(xt, xir::ArithmeticOp::UNARY_MINUS, {p_abs});
            xir::Value *int_neg_res = builder.call(xt, xir::ArithmeticOp::SELECT, {p_abs, neg_p, is_odd});
            xir::Value *neg_res = builder.call(xt, xir::ArithmeticOp::SELECT, {splat(0.0f), int_neg_res, is_int});

            xir::Value *nonpos_res = builder.call(xt, xir::ArithmeticOp::SELECT, {neg_res, zero_res, a_eq0});
            out.val = builder.call(xt, xir::ArithmeticOp::SELECT, {nonpos_res, p_abs, a_gt0});

            if (wd && (da.has_derivs() || db.has_derivs())) {
                xir::Value *safe_a_div = builder.call(xt, xir::ArithmeticOp::SELECT,
                                                      {da.val, splat(1.0f), a_eq0});
                xir::Value *dp_da_exact = safe_div_val(*this, bin(xir::ArithmeticOp::BINARY_MUL, db.val, out.val), safe_a_div);
                xir::Value *valid_neg = builder.call(bt, xir::ArithmeticOp::BINARY_BIT_AND,
                                                     {builder.call(bt, xir::ArithmeticOp::BINARY_LESS, {da.val, splat(0.0f)}),
                                                      is_int});
                xir::Value *valid_a = builder.call(bt, xir::ArithmeticOp::BINARY_BIT_OR, {a_gt0, valid_neg});
                xir::Value *dp_da = builder.call(xt, xir::ArithmeticOp::SELECT, {splat(0.0f), dp_da_exact, valid_a});
                xir::Value *log_a = builder.call(xt, xir::ArithmeticOp::LOG, {safe_pos});
                auto deriv = [&](xir::Value *dxa, xir::Value *dxb) -> xir::Value * {
                    if (!dxa && !dxb) { return nullptr; }
                    xir::Value *term1 = dxa ? bin(xir::ArithmeticOp::BINARY_MUL, dp_da, dxa) : nullptr;
                    xir::Value *term2 = dxb
                                            ? builder.call(xt, xir::ArithmeticOp::SELECT,
                                                           {splat(0.0f),
                                                            bin(xir::ArithmeticOp::BINARY_MUL,
                                                                bin(xir::ArithmeticOp::BINARY_MUL, out.val, log_a), dxb),
                                                            a_gt0})
                                            : nullptr;
                    if (term1 && term2) { return bin(xir::ArithmeticOp::BINARY_ADD, term1, term2); }
                    return term1 ? term1 : term2;
                };
                out.dx = deriv(da.dx, db.dx);
                out.dy = deriv(da.dy, db.dy);
            }
        }
        store_symbol(active_layer, dst, convert_dual(out, eval_osl, dst->type(), wd));
        return true;
    }

    // 13. min / max / clamp / mix / select / smoothstep
    if ((op == "min" || op == "max") && args.size() == 3u) {
        const Symbol *dst = args[0];
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        bool all_int = detail::is_simple_type(dst->type(), SimpleType::Primitive::INT) &&
                       detail::is_simple_type(args[1]->type(), SimpleType::Primitive::INT) &&
                       detail::is_simple_type(args[2]->type(), SimpleType::Primitive::INT);
        if (all_int) {
            xir::Value *a = load_symbol(active_layer, args[1], false).val;
            xir::Value *b = load_symbol(active_layer, args[2], false).val;
            xir::Value *r = builder.call(compute::Type::of<int32_t>(),
                                         op == "min" ? xir::ArithmeticOp::MIN : xir::ArithmeticOp::MAX, {a, b});
            store_symbol(active_layer, dst, {.val = r});
            return true;
        }
        const Type *eval_osl = widen_osl_arithmetic_type(args[1]->type(), args[2]->type(), dst->type());
        if (detail::is_simple_type(eval_osl, SimpleType::Primitive::INT)) { eval_osl = osl_float(); }
        DualValue da = convert_dual(load_symbol(active_layer, args[1], wd), args[1]->type(), eval_osl, wd);
        DualValue db = convert_dual(load_symbol(active_layer, args[2], wd), args[2]->type(), eval_osl, wd);
        const compute::Type *xt = da.val->type();
        const compute::Type *bt = bool_type_for(xt);
        DualValue out;
        out.val = builder.call(xt, op == "min" ? xir::ArithmeticOp::MIN : xir::ArithmeticOp::MAX, {da.val, db.val});
        if (wd && (da.has_derivs() || db.has_derivs())) {
            xir::Value *take_a = builder.call(
                bt, op == "min" ? xir::ArithmeticOp::BINARY_LESS : xir::ArithmeticOp::BINARY_GREATER,
                {da.val, db.val});
            xir::Value *z = splat_scalar_for(*this, xt, 0.0f);
            if (da.dx || db.dx) {
                out.dx = builder.call(xt, xir::ArithmeticOp::SELECT, {db.dx ? db.dx : z, da.dx ? da.dx : z, take_a});
            }
            if (da.dy || db.dy) {
                out.dy = builder.call(xt, xir::ArithmeticOp::SELECT, {db.dy ? db.dy : z, da.dy ? da.dy : z, take_a});
            }
        }
        store_symbol(active_layer, dst, convert_dual(out, eval_osl, dst->type(), wd));
        return true;
    }

    if (op == "clamp" && args.size() == 4u) {
        const Symbol *dst = args[0];
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        bool all_int = detail::is_simple_type(dst->type(), SimpleType::Primitive::INT) &&
                       detail::is_simple_type(args[1]->type(), SimpleType::Primitive::INT) &&
                       detail::is_simple_type(args[2]->type(), SimpleType::Primitive::INT) &&
                       detail::is_simple_type(args[3]->type(), SimpleType::Primitive::INT);
        if (all_int) {
            xir::Value *x = load_symbol(active_layer, args[1], false).val;
            xir::Value *lo = load_symbol(active_layer, args[2], false).val;
            xir::Value *hi = load_symbol(active_layer, args[3], false).val;
            store_symbol(active_layer, dst,
                         {.val = builder.call(compute::Type::of<int32_t>(), xir::ArithmeticOp::CLAMP, {x, lo, hi})});
            return true;
        }
        const Type *eval_osl = widen_osl_arithmetic_type(args[1]->type(), widen_osl_arithmetic_type(args[2]->type(), args[3]->type(), dst->type()), dst->type());
        if (detail::is_simple_type(eval_osl, SimpleType::Primitive::INT)) { eval_osl = osl_float(); }
        DualValue dx = convert_dual(load_symbol(active_layer, args[1], wd), args[1]->type(), eval_osl, wd);
        DualValue dlo = convert_dual(load_symbol(active_layer, args[2], wd), args[2]->type(), eval_osl, wd);
        DualValue dhi = convert_dual(load_symbol(active_layer, args[3], wd), args[3]->type(), eval_osl, wd);
        const compute::Type *xt = dx.val->type();
        const compute::Type *bt = bool_type_for(xt);
        DualValue out;
        out.val = builder.call(xt, xir::ArithmeticOp::CLAMP, {dx.val, dlo.val, dhi.val});
        if (wd && (dx.has_derivs() || dlo.has_derivs() || dhi.has_derivs())) {
            xir::Value *lt_lo = builder.call(bt, xir::ArithmeticOp::BINARY_LESS, {dx.val, dlo.val});
            xir::Value *gt_hi = builder.call(bt, xir::ArithmeticOp::BINARY_GREATER, {dx.val, dhi.val});
            xir::Value *z = splat_scalar_for(*this, xt, 0.0f);
            auto sel_d = [&](xir::Value *vx, xir::Value *vlo, xir::Value *vhi) -> xir::Value * {
                if (!vx && !vlo && !vhi) { return nullptr; }
                xir::Value *m = builder.call(xt, xir::ArithmeticOp::SELECT, {vx ? vx : z, vlo ? vlo : z, lt_lo});
                return builder.call(xt, xir::ArithmeticOp::SELECT, {m, vhi ? vhi : z, gt_hi});
            };
            out.dx = sel_d(dx.dx, dlo.dx, dhi.dx);
            out.dy = sel_d(dx.dy, dlo.dy, dhi.dy);
        }
        store_symbol(active_layer, dst, convert_dual(out, eval_osl, dst->type(), wd));
        return true;
    }

    if (op == "mix" && args.size() == 4u) {
        const Symbol *dst = args[0];
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        const Type *eval_osl = widen_osl_arithmetic_type(args[1]->type(), widen_osl_arithmetic_type(args[2]->type(), args[3]->type(), dst->type()), dst->type());
        if (detail::is_simple_type(eval_osl, SimpleType::Primitive::INT)) { eval_osl = osl_float(); }
        DualValue dx = convert_dual(load_symbol(active_layer, args[1], wd), args[1]->type(), eval_osl, wd);
        DualValue dy = convert_dual(load_symbol(active_layer, args[2], wd), args[2]->type(), eval_osl, wd);
        DualValue da = convert_dual(load_symbol(active_layer, args[3], wd), args[3]->type(), eval_osl, wd);
        const compute::Type *xt = dx.val->type();
        auto bin = [&](xir::ArithmeticOp aop, xir::Value *x, xir::Value *y) {
            return builder.call(xt, aop, {x, y});
        };
        DualValue out;
        out.val = builder.call(xt, xir::ArithmeticOp::LERP, {dx.val, dy.val, da.val});
        if (wd && (dx.has_derivs() || dy.has_derivs() || da.has_derivs())) {
            xir::Value *one = splat_scalar_for(*this, xt, 1.0f);
            xir::Value *oma = bin(xir::ArithmeticOp::BINARY_SUB, one, da.val);
            xir::Value *diff = bin(xir::ArithmeticOp::BINARY_SUB, dy.val, dx.val);
            auto mix_d = [&](xir::Value *px, xir::Value *py, xir::Value *pa) -> xir::Value * {
                if (!px && !py && !pa) { return nullptr; }
                xir::Value *acc = nullptr;
                auto add_term = [&](xir::Value *t) {
                    acc = acc ? bin(xir::ArithmeticOp::BINARY_ADD, acc, t) : t;
                };
                if (px) { add_term(bin(xir::ArithmeticOp::BINARY_MUL, oma, px)); }
                if (py) { add_term(bin(xir::ArithmeticOp::BINARY_MUL, da.val, py)); }
                if (pa) { add_term(bin(xir::ArithmeticOp::BINARY_MUL, diff, pa)); }
                return acc;
            };
            out.dx = mix_d(dx.dx, dy.dx, da.dx);
            out.dy = mix_d(dx.dy, dy.dy, da.dy);
        }
        store_symbol(active_layer, dst, convert_dual(out, eval_osl, dst->type(), wd));
        return true;
    }

    if (op == "select" && args.size() == 4u) {
        const Symbol *dst = args[0];
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        bool all_int = detail::is_simple_type(dst->type(), SimpleType::Primitive::INT) &&
                       detail::is_simple_type(args[1]->type(), SimpleType::Primitive::INT) &&
                       detail::is_simple_type(args[2]->type(), SimpleType::Primitive::INT) &&
                       !detail::is_triple_type(args[3]->type());
        if (all_int) {
            xir::Value *x = load_symbol(active_layer, args[1], false).val;
            xir::Value *y = load_symbol(active_layer, args[2], false).val;
            xir::Value *c = to_bool(load_symbol(active_layer, args[3], false).val);
            store_symbol(active_layer, dst,
                         {.val = builder.call(compute::Type::of<int32_t>(), xir::ArithmeticOp::SELECT, {x, y, c})});
            return true;
        }
        const Type *eval_osl = widen_osl_arithmetic_type(args[1]->type(), args[2]->type(), dst->type());
        if (detail::is_simple_type(eval_osl, SimpleType::Primitive::INT)) { eval_osl = osl_float(); }
        DualValue dx = convert_dual(load_symbol(active_layer, args[1], wd), args[1]->type(), eval_osl, wd);
        DualValue dy = convert_dual(load_symbol(active_layer, args[2], wd), args[2]->type(), eval_osl, wd);
        const compute::Type *xt = dx.val->type();
        xir::Value *cond_raw = load_symbol(active_layer, args[3], false).val;
        xir::Value *cond_mask = nullptr;
        if (cond_raw->type() == compute::Type::of<luisa::float3>()) {
            cond_mask = builder.call(compute::Type::of<luisa::bool3>(), xir::ArithmeticOp::BINARY_NOT_EQUAL,
                                     {cond_raw, const_float3(luisa::float3{0.0f})});
        } else {
            cond_mask = make_bool_mask(*this, to_bool(cond_raw), xt);
        }
        DualValue out;
        out.val = builder.call(xt, xir::ArithmeticOp::SELECT, {dx.val, dy.val, cond_mask});
        if (wd && (dx.has_derivs() || dy.has_derivs())) {
            xir::Value *z = splat_scalar_for(*this, xt, 0.0f);
            if (dx.dx || dy.dx) {
                out.dx = builder.call(xt, xir::ArithmeticOp::SELECT, {dx.dx ? dx.dx : z, dy.dx ? dy.dx : z, cond_mask});
            }
            if (dx.dy || dy.dy) {
                out.dy = builder.call(xt, xir::ArithmeticOp::SELECT, {dx.dy ? dx.dy : z, dy.dy ? dy.dy : z, cond_mask});
            }
        }
        store_symbol(active_layer, dst, convert_dual(out, eval_osl, dst->type(), wd));
        return true;
    }

    if (op == "smoothstep" && args.size() == 4u) {
        const Symbol *dst = args[0];
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        const Type *eval_osl = widen_osl_arithmetic_type(args[1]->type(), widen_osl_arithmetic_type(args[2]->type(), args[3]->type(), dst->type()), dst->type());
        if (detail::is_simple_type(eval_osl, SimpleType::Primitive::INT)) { eval_osl = osl_float(); }
        DualValue de0 = convert_dual(load_symbol(active_layer, args[1], wd), args[1]->type(), eval_osl, wd);
        DualValue de1 = convert_dual(load_symbol(active_layer, args[2], wd), args[2]->type(), eval_osl, wd);
        DualValue dx = convert_dual(load_symbol(active_layer, args[3], wd), args[3]->type(), eval_osl, wd);
        const compute::Type *xt = dx.val->type();
        const compute::Type *bt = bool_type_for(xt);
        auto bin = [&](xir::ArithmeticOp aop, xir::Value *a, xir::Value *b) {
            return builder.call(xt, aop, {a, b});
        };
        auto splat = [&](float s) { return splat_scalar_for(*this, xt, s); };
        DualValue out;
        out.val = builder.call(xt, xir::ArithmeticOp::SMOOTHSTEP, {de0.val, de1.val, dx.val});
        if (wd && (de0.has_derivs() || de1.has_derivs() || dx.has_derivs())) {
            xir::Value *span = bin(xir::ArithmeticOp::BINARY_SUB, de1.val, de0.val);
            xir::Value *num = bin(xir::ArithmeticOp::BINARY_SUB, dx.val, de0.val);
            xir::Value *u = builder.call(xt, xir::ArithmeticOp::CLAMP,
                                         {safe_div_val(*this, num, span), splat(0.0f), splat(1.0f)});
            xir::Value *slope_u = bin(xir::ArithmeticOp::BINARY_MUL, splat(6.0f),
                                      bin(xir::ArithmeticOp::BINARY_MUL, u,
                                          bin(xir::ArithmeticOp::BINARY_SUB, splat(1.0f), u)));
            xir::Value *in_range = builder.call(
                bt, xir::ArithmeticOp::BINARY_BIT_AND,
                {builder.call(bt, xir::ArithmeticOp::BINARY_GREATER, {dx.val, de0.val}),
                 builder.call(bt, xir::ArithmeticOp::BINARY_LESS, {dx.val, de1.val})});
            xir::Value *valid_span = builder.call(bt, xir::ArithmeticOp::BINARY_GREATER, {span, splat(0.0f)});
            in_range = builder.call(bt, xir::ArithmeticOp::BINARY_BIT_AND, {in_range, valid_span});
            xir::Value *span2 = bin(xir::ArithmeticOp::BINARY_MUL, span, span);
            auto sm_d = [&](xir::Value *px, xir::Value *pe0, xir::Value *pe1) -> xir::Value * {
                if (!px && !pe0 && !pe1) { return nullptr; }
                xir::Value *d_num = (px && pe0) ? bin(xir::ArithmeticOp::BINARY_SUB, px, pe0)
                                                : (px ? px : builder.call(xt, xir::ArithmeticOp::UNARY_MINUS, {pe0}));
                xir::Value *d_span = (pe1 && pe0) ? bin(xir::ArithmeticOp::BINARY_SUB, pe1, pe0)
                                                  : (pe1 ? pe1 : (pe0 ? builder.call(xt, xir::ArithmeticOp::UNARY_MINUS, {pe0}) : nullptr));
                xir::Value *du_num = d_num ? bin(xir::ArithmeticOp::BINARY_MUL, d_num, span) : splat(0.0f);
                if (d_span) {
                    xir::Value *t = bin(xir::ArithmeticOp::BINARY_MUL, num, d_span);
                    du_num = d_num ? bin(xir::ArithmeticOp::BINARY_SUB, du_num, t)
                                   : builder.call(xt, xir::ArithmeticOp::UNARY_MINUS, {t});
                }
                xir::Value *du = safe_div_val(*this, du_num, span2);
                xir::Value *d = bin(xir::ArithmeticOp::BINARY_MUL, slope_u, du);
                return builder.call(xt, xir::ArithmeticOp::SELECT, {splat(0.0f), d, in_range});
            };
            out.dx = sm_d(dx.dx, de0.dx, de1.dx);
            out.dy = sm_d(dx.dy, de0.dy, de1.dy);
        }
        store_symbol(active_layer, dst, convert_dual(out, eval_osl, dst->type(), wd));
        return true;
    }

    // 14. isnan / isinf / isfinite
    if ((op == "isnan" || op == "isinf" || op == "isfinite") && args.size() == 2u) {
        const Symbol *dst = args[0];
        xir::Value *v = load_symbol(active_layer, args[1], false).val;
        if (detail::is_simple_type(args[1]->type(), SimpleType::Primitive::INT)) {
            int32_t res = (op == "isfinite") ? 1 : 0;
            store_symbol(active_layer, dst, convert_dual({.val = const_int(res)}, osl_int(), dst->type(), false));
            return true;
        }
        const compute::Type *bt = bool_type_for(v->type());
        xir::Value *b = nullptr;
        if (op == "isnan") {
            b = any_component_nonzero(*this, builder.call(bt, xir::ArithmeticOp::ISNAN, {v}));
        } else if (op == "isinf") {
            b = any_component_nonzero(*this, builder.call(bt, xir::ArithmeticOp::ISINF, {v}));
        } else {
            xir::Value *nan_b = any_component_nonzero(*this, builder.call(bt, xir::ArithmeticOp::ISNAN, {v}));
            xir::Value *inf_b = any_component_nonzero(*this, builder.call(bt, xir::ArithmeticOp::ISINF, {v}));
            xir::Value *bad = builder.call(compute::Type::of<bool>(), xir::ArithmeticOp::BINARY_BIT_OR, {nan_b, inf_b});
            b = builder.call(compute::Type::of<bool>(), xir::ArithmeticOp::BINARY_EQUAL, {bad, const_bool(false)});
        }
        store_symbol(active_layer, dst, convert_dual({.val = to_int(b)}, osl_int(), dst->type(), false));
        return true;
    }

    // 15. Vector ops: dot, cross, length, distance, normalize, faceforward, reflect, refract
    if (op == "dot" && args.size() == 3u) {
        const Symbol *dst = args[0];
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        DualValue a = load_symbol(active_layer, args[1], wd);
        DualValue b = load_symbol(active_layer, args[2], wd);
        xir::Value *va = to_float3(a.val);
        xir::Value *vb = to_float3(b.val);
        auto *f_t = compute::Type::of<float>();
        DualValue out;
        out.val = builder.call(f_t, xir::ArithmeticOp::DOT, {va, vb});
        if (wd && (a.has_derivs() || b.has_derivs())) {
            auto dot_d = [&](xir::Value *da, xir::Value *db) -> xir::Value * {
                xir::Value *t1 = da ? builder.call(f_t, xir::ArithmeticOp::DOT, {to_float3(da), vb}) : nullptr;
                xir::Value *t2 = db ? builder.call(f_t, xir::ArithmeticOp::DOT, {va, to_float3(db)}) : nullptr;
                if (t1 && t2) { return builder.call(f_t, xir::ArithmeticOp::BINARY_ADD, {t1, t2}); }
                return t1 ? t1 : t2;
            };
            out.dx = dot_d(a.dx, b.dx);
            out.dy = dot_d(a.dy, b.dy);
        }
        store_symbol(active_layer, dst, convert_dual(out, osl_float(), dst->type(), wd));
        return true;
    }

    if (op == "cross" && args.size() == 3u) {
        const Symbol *dst = args[0];
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        DualValue a = load_symbol(active_layer, args[1], wd);
        DualValue b = load_symbol(active_layer, args[2], wd);
        xir::Value *va = to_float3(a.val);
        xir::Value *vb = to_float3(b.val);
        auto *f3_t = compute::Type::of<luisa::float3>();
        DualValue out;
        out.val = builder.call(f3_t, xir::ArithmeticOp::CROSS, {va, vb});
        if (wd && (a.has_derivs() || b.has_derivs())) {
            auto cross_d = [&](xir::Value *da, xir::Value *db) -> xir::Value * {
                xir::Value *t1 = da ? builder.call(f3_t, xir::ArithmeticOp::CROSS, {to_float3(da), vb}) : nullptr;
                xir::Value *t2 = db ? builder.call(f3_t, xir::ArithmeticOp::CROSS, {va, to_float3(db)}) : nullptr;
                if (t1 && t2) { return builder.call(f3_t, xir::ArithmeticOp::BINARY_ADD, {t1, t2}); }
                return t1 ? t1 : t2;
            };
            out.dx = cross_d(a.dx, b.dx);
            out.dy = cross_d(a.dy, b.dy);
        }
        store_symbol(active_layer, dst, out);
        return true;
    }

    if ((op == "length" && args.size() == 2u) || (op == "distance" && args.size() == 3u)) {
        const Symbol *dst = args[0];
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        auto *f_t = compute::Type::of<float>();
        auto *f3_t = compute::Type::of<luisa::float3>();
        DualValue dv;
        if (op == "length") {
            dv = load_symbol(active_layer, args[1], wd);
            dv.val = to_float3(dv.val);
        } else {
            DualValue pa = load_symbol(active_layer, args[1], wd);
            DualValue pb = load_symbol(active_layer, args[2], wd);
            dv.val = builder.call(f3_t, xir::ArithmeticOp::BINARY_SUB, {to_float3(pa.val), to_float3(pb.val)});
            if (wd) {
                auto sub_d = [&](xir::Value *a, xir::Value *b) -> xir::Value * {
                    if (a && b) { return builder.call(f3_t, xir::ArithmeticOp::BINARY_SUB, {a, b}); }
                    if (a) { return a; }
                    if (b) { return builder.call(f3_t, xir::ArithmeticOp::UNARY_MINUS, {b}); }
                    return nullptr;
                };
                dv.dx = sub_d(pa.dx, pb.dx);
                dv.dy = sub_d(pa.dy, pb.dy);
            }
        }
        DualValue out;
        out.val = builder.call(f_t, xir::ArithmeticOp::LENGTH, {dv.val});
        if (wd && dv.has_derivs()) {
            xir::Value *pos = builder.call(compute::Type::of<bool>(), xir::ArithmeticOp::BINARY_GREATER,
                                           {out.val, const_float(0.0f)});
            xir::Value *safe_L = builder.call(f_t, xir::ArithmeticOp::SELECT, {const_float(1.0f), out.val, pos});
            auto len_d = [&](xir::Value *d) -> xir::Value * {
                if (!d) { return nullptr; }
                xir::Value *q = builder.call(f_t, xir::ArithmeticOp::BINARY_DIV,
                                             {builder.call(f_t, xir::ArithmeticOp::DOT, {dv.val, to_float3(d)}), safe_L});
                return builder.call(f_t, xir::ArithmeticOp::SELECT, {const_float(0.0f), q, pos});
            };
            out.dx = len_d(dv.dx);
            out.dy = len_d(dv.dy);
        }
        store_symbol(active_layer, dst, convert_dual(out, osl_float(), dst->type(), wd));
        return true;
    }

    if (op == "normalize" && args.size() == 2u) {
        const Symbol *dst = args[0];
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        DualValue v = load_symbol(active_layer, args[1], wd);
        xir::Value *vv = to_float3(v.val);
        auto *f_t = compute::Type::of<float>();
        auto *f3_t = compute::Type::of<luisa::float3>();
        xir::Value *len = builder.call(f_t, xir::ArithmeticOp::LENGTH, {vv});
        xir::Value *pos = builder.call(compute::Type::of<bool>(), xir::ArithmeticOp::BINARY_GREATER,
                                       {len, const_float(0.0f)});
        xir::Value *pos3 = make_bool_mask(*this, pos, f3_t);
        xir::Value *nv = builder.call(f3_t, xir::ArithmeticOp::NORMALIZE, {vv});
        DualValue out;
        out.val = builder.call(f3_t, xir::ArithmeticOp::SELECT, {const_float3(luisa::float3{0.0f}), nv, pos3});
        if (wd && v.has_derivs()) {
            xir::Value *safe_L = builder.call(f_t, xir::ArithmeticOp::SELECT, {const_float(1.0f), len, pos});
            xir::Value *inv_L3 = to_float3(builder.call(f_t, xir::ArithmeticOp::BINARY_DIV, {const_float(1.0f), safe_L}));
            auto norm_d = [&](xir::Value *d) -> xir::Value * {
                if (!d) { return nullptr; }
                xir::Value *dv = to_float3(d);
                xir::Value *ndotd = to_float3(builder.call(f_t, xir::ArithmeticOp::DOT, {out.val, dv}));
                xir::Value *proj = builder.call(f3_t, xir::ArithmeticOp::BINARY_SUB,
                                                {dv, builder.call(f3_t, xir::ArithmeticOp::BINARY_MUL, {out.val, ndotd})});
                xir::Value *res = builder.call(f3_t, xir::ArithmeticOp::BINARY_MUL, {proj, inv_L3});
                return builder.call(f3_t, xir::ArithmeticOp::SELECT,
                                    {const_float3(luisa::float3{0.0f}), res, pos3});
            };
            out.dx = norm_d(v.dx);
            out.dy = norm_d(v.dy);
        }
        store_symbol(active_layer, dst, out);
        return true;
    }

    if (op == "faceforward" && (args.size() == 3u || args.size() == 4u)) {
        const Symbol *dst = args[0];
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        DualValue N = load_symbol(active_layer, args[1], wd);
        xir::Value *I = to_float3(load_symbol(active_layer, args[2], false).val);
        xir::Value *Nref = (args.size() == 4u)
                               ? to_float3(load_symbol(active_layer, args[3], false).val)
                               : to_float3(owner->emit_global_read(GlobalSymbol::Ng, false).val);
        if (!errors.empty()) { return false; }
        auto *f3_t = compute::Type::of<luisa::float3>();
        xir::Value *dp = builder.call(compute::Type::of<float>(), xir::ArithmeticOp::DOT, {I, Nref});
        xir::Value *flip = builder.call(compute::Type::of<bool>(), xir::ArithmeticOp::BINARY_GREATER, {dp, const_float(0.0f)});
        xir::Value *flip3 = make_bool_mask(*this, flip, f3_t);
        DualValue out;
        out.val = builder.call(f3_t, xir::ArithmeticOp::SELECT,
                               {N.val, builder.call(f3_t, xir::ArithmeticOp::UNARY_MINUS, {N.val}), flip3});
        if (wd && N.has_derivs()) {
            if (N.dx) {
                out.dx = builder.call(f3_t, xir::ArithmeticOp::SELECT,
                                      {N.dx, builder.call(f3_t, xir::ArithmeticOp::UNARY_MINUS, {N.dx}), flip3});
            }
            if (N.dy) {
                out.dy = builder.call(f3_t, xir::ArithmeticOp::SELECT,
                                      {N.dy, builder.call(f3_t, xir::ArithmeticOp::UNARY_MINUS, {N.dy}), flip3});
            }
        }
        store_symbol(active_layer, dst, out);
        return true;
    }

    if (op == "reflect" && args.size() == 3u) {
        const Symbol *dst = args[0];
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        DualValue I = convert_dual(load_symbol(active_layer, args[1], wd), args[1]->type(), osl_vector(), wd);
        DualValue N = convert_dual(load_symbol(active_layer, args[2], wd), args[2]->type(), osl_vector(), wd);
        auto *f_t = compute::Type::of<float>();
        auto *f3_t = compute::Type::of<luisa::float3>();
        DualValue out;
        out.val = builder.call(f3_t, xir::ArithmeticOp::REFLECT, {I.val, N.val});
        if (wd && (I.has_derivs() || N.has_derivs())) {
            xir::Value *IdotN = builder.call(f_t, xir::ArithmeticOp::DOT, {I.val, N.val});
            auto refl_d = [&](xir::Value *dI, xir::Value *dN) -> xir::Value * {
                if (!dI && !dN) { return nullptr; }
                xir::Value *di = dI ? dI : zero_of(f3_t);
                xir::Value *dn = dN ? dN : zero_of(f3_t);
                xir::Value *ddot = builder.call(
                    f_t, xir::ArithmeticOp::BINARY_ADD,
                    {builder.call(f_t, xir::ArithmeticOp::DOT, {di, N.val}),
                     builder.call(f_t, xir::ArithmeticOp::DOT, {I.val, dn})});
                xir::Value *term = builder.call(
                    f3_t, xir::ArithmeticOp::BINARY_ADD,
                    {builder.call(f3_t, xir::ArithmeticOp::BINARY_MUL, {to_float3(ddot), N.val}),
                     builder.call(f3_t, xir::ArithmeticOp::BINARY_MUL, {to_float3(IdotN), dn})});
                xir::Value *two_t = builder.call(f3_t, xir::ArithmeticOp::BINARY_MUL,
                                                 {const_float3(luisa::float3{2.0f}), term});
                return builder.call(f3_t, xir::ArithmeticOp::BINARY_SUB, {di, two_t});
            };
            out.dx = refl_d(I.dx, N.dx);
            out.dy = refl_d(I.dy, N.dy);
        }
        store_symbol(active_layer, dst, convert_dual(out, osl_vector(), dst->type(), wd));
        return true;
    }

    if (op == "refract" && args.size() == 4u) {
        const Symbol *dst = args[0];
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        DualValue I = convert_dual(load_symbol(active_layer, args[1], wd), args[1]->type(), osl_vector(), wd);
        DualValue N = convert_dual(load_symbol(active_layer, args[2], wd), args[2]->type(), osl_vector(), wd);
        DualValue eta = convert_dual(load_symbol(active_layer, args[3], wd), args[3]->type(), osl_float(), wd);
        auto *f_t = compute::Type::of<float>();
        auto *f3_t = compute::Type::of<luisa::float3>();
        xir::Value *IdotN = builder.call(f_t, xir::ArithmeticOp::DOT, {I.val, N.val});
        xir::Value *one = const_float(1.0f);
        xir::Value *eta2 = builder.call(f_t, xir::ArithmeticOp::BINARY_MUL, {eta.val, eta.val});
        xir::Value *c2 = builder.call(f_t, xir::ArithmeticOp::BINARY_MUL, {IdotN, IdotN});
        xir::Value *one_minus_c2 = builder.call(f_t, xir::ArithmeticOp::BINARY_SUB, {one, c2});
        xir::Value *k = builder.call(f_t, xir::ArithmeticOp::BINARY_SUB,
                                     {one, builder.call(f_t, xir::ArithmeticOp::BINARY_MUL, {eta2, one_minus_c2})});
        xir::Value *k_ok = builder.call(compute::Type::of<bool>(), xir::ArithmeticOp::BINARY_GREATER_EQUAL, {k, const_float(0.0f)});
        xir::Value *safe_k = builder.call(f_t, xir::ArithmeticOp::MAX, {k, const_float(0.0f)});
        xir::Value *sq_k = builder.call(f_t, xir::ArithmeticOp::SQRT, {safe_k});
        xir::Value *scale_N = builder.call(f_t, xir::ArithmeticOp::BINARY_ADD,
                                           {builder.call(f_t, xir::ArithmeticOp::BINARY_MUL, {eta.val, IdotN}), sq_k});
        xir::Value *t = builder.call(f3_t, xir::ArithmeticOp::BINARY_SUB,
                                     {builder.call(f3_t, xir::ArithmeticOp::BINARY_MUL, {to_float3(eta.val), I.val}),
                                      builder.call(f3_t, xir::ArithmeticOp::BINARY_MUL, {N.val, to_float3(scale_N)})});
        xir::Value *k_mask3 = make_bool_mask(*this, k_ok, f3_t);
        DualValue out;
        out.val = builder.call(f3_t, xir::ArithmeticOp::SELECT,
                               {const_float3(luisa::float3{0.0f}), t, k_mask3});
        if (wd && (I.has_derivs() || N.has_derivs() || eta.has_derivs())) {
            xir::Value *safe_sq = builder.call(f_t, xir::ArithmeticOp::MAX, {sq_k, const_float(1e-12f)});
            auto refr_d = [&](xir::Value *dI, xir::Value *dN, xir::Value *dEta) -> xir::Value * {
                if (!dI && !dN && !dEta) { return nullptr; }
                xir::Value *di = dI ? dI : zero_of(f3_t);
                xir::Value *dn = dN ? dN : zero_of(f3_t);
                xir::Value *de = dEta ? dEta : const_float(0.0f);
                xir::Value *dc = builder.call(
                    f_t, xir::ArithmeticOp::BINARY_ADD,
                    {builder.call(f_t, xir::ArithmeticOp::DOT, {di, N.val}),
                     builder.call(f_t, xir::ArithmeticOp::DOT, {I.val, dn})});
                xir::Value *num_ds = builder.call(
                    f_t, xir::ArithmeticOp::BINARY_SUB,
                    {builder.call(f_t, xir::ArithmeticOp::BINARY_MUL,
                                  {builder.call(f_t, xir::ArithmeticOp::BINARY_MUL, {eta2, IdotN}), dc}),
                     builder.call(f_t, xir::ArithmeticOp::BINARY_MUL,
                                  {builder.call(f_t, xir::ArithmeticOp::BINARY_MUL, {eta.val, one_minus_c2}), de})});
                xir::Value *ds = builder.call(f_t, xir::ArithmeticOp::BINARY_DIV, {num_ds, safe_sq});
                xir::Value *d_scale = builder.call(
                    f_t, xir::ArithmeticOp::BINARY_ADD,
                    {builder.call(f_t, xir::ArithmeticOp::BINARY_ADD,
                                  {builder.call(f_t, xir::ArithmeticOp::BINARY_MUL, {de, IdotN}),
                                   builder.call(f_t, xir::ArithmeticOp::BINARY_MUL, {eta.val, dc})}),
                     ds});
                xir::Value *d_etaI = builder.call(
                    f3_t, xir::ArithmeticOp::BINARY_ADD,
                    {builder.call(f3_t, xir::ArithmeticOp::BINARY_MUL, {to_float3(de), I.val}),
                     builder.call(f3_t, xir::ArithmeticOp::BINARY_MUL, {to_float3(eta.val), di})});
                xir::Value *d_sN = builder.call(
                    f3_t, xir::ArithmeticOp::BINARY_ADD,
                    {builder.call(f3_t, xir::ArithmeticOp::BINARY_MUL, {to_float3(d_scale), N.val}),
                     builder.call(f3_t, xir::ArithmeticOp::BINARY_MUL, {to_float3(scale_N), dn})});
                xir::Value *dt = builder.call(f3_t, xir::ArithmeticOp::BINARY_SUB, {d_etaI, d_sN});
                return builder.call(f3_t, xir::ArithmeticOp::SELECT,
                                    {const_float3(luisa::float3{0.0f}), dt, k_mask3});
            };
            out.dx = refr_d(I.dx, N.dx, eta.dx);
            out.dy = refr_d(I.dy, N.dy, eta.dy);
        }
        store_symbol(active_layer, dst, convert_dual(out, osl_vector(), dst->type(), wd));
        return true;
    }

    // 16. Derivatives: Dx, Dy, Dz, area, filterwidth, calculatenormal
    if (op == "Dz" && args.size() == 2u) {
        const Symbol *dst = args[0];
        const compute::Type *xt = detail::osl_type_to_xir_type(dst->type(), 0, owner->closure_xir_type());
        if (xt == nullptr) { return false; }
        store_symbol(active_layer, dst, {.val = zero_of(xt)});
        return true;
    }

    if ((op == "Dx" || op == "Dy") && args.size() == 2u) {
        const Symbol *dst = args[0];
        DualValue src = load_symbol(active_layer, args[1], true);
        const compute::Type *xt = detail::osl_type_to_xir_type(dst->type(), 0, owner->closure_xir_type());
        xir::Value *d = (op == "Dx") ? src.dx : src.dy;
        DualValue res{.val = d ? d : zero_of(xt)};
        store_symbol(active_layer, dst, convert_dual(res, args[1]->type(), dst->type(), false));
        return true;
    }

    if (op == "area" && args.size() == 2u) {
        const Symbol *dst = args[0];
        DualValue p = load_symbol(active_layer, args[1], true);
        xir::Value *res = const_float(0.0f);
        if (p.dx != nullptr && p.dy != nullptr) {
            xir::Value *cr = builder.call(compute::Type::of<luisa::float3>(), xir::ArithmeticOp::CROSS,
                                          {to_float3(p.dx), to_float3(p.dy)});
            res = builder.call(compute::Type::of<float>(), xir::ArithmeticOp::LENGTH, {cr});
        }
        store_symbol(active_layer, dst, convert_dual({.val = res}, osl_float(), dst->type(), false));
        return true;
    }

    if (op == "filterwidth" && args.size() == 2u) {
        const Symbol *dst = args[0];
        DualValue x = load_symbol(active_layer, args[1], true);
        const compute::Type *xt = detail::osl_type_to_xir_type(args[1]->type(), 0, owner->closure_xir_type());
        xir::Value *res = zero_of(xt);
        if (x.dx != nullptr || x.dy != nullptr) {
            auto sq = [&](xir::Value *v) {
                return v ? builder.call(xt, xir::ArithmeticOp::BINARY_MUL, {v, v}) : zero_of(xt);
            };
            xir::Value *s2 = builder.call(xt, xir::ArithmeticOp::BINARY_ADD, {sq(x.dx), sq(x.dy)});
            res = builder.call(xt, xir::ArithmeticOp::SQRT, {s2});
        }
        store_symbol(active_layer, dst, convert_dual({.val = res}, args[1]->type(), dst->type(), false));
        return true;
    }

    if (op == "calculatenormal" && args.size() == 2u) {
        const Symbol *dst = args[0];
        DualValue p = load_symbol(active_layer, args[1], true);
        xir::Value *res = const_float3(luisa::float3{0.0f});
        if (p.dx != nullptr && p.dy != nullptr) {
            res = builder.call(compute::Type::of<luisa::float3>(), xir::ArithmeticOp::CROSS,
                               {to_float3(p.dx), to_float3(p.dy)});
        }
        store_symbol(active_layer, dst, {.val = res});
        return true;
    }

    // 17. Color builtins: luminance, blackbody, wavelength_color
    if (op == "luminance" && args.size() == 2u) {
        const Symbol *dst = args[0];
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        DualValue c = convert_dual(load_symbol(active_layer, args[1], wd), args[1]->type(), osl_color(), wd);
        DualValue lum = emit_luminance(c, wd);
        store_symbol(active_layer, dst, convert_dual(lum, osl_float(), dst->type(), wd));
        return true;
    }

    if ((op == "blackbody" || op == "wavelength_color") && args.size() == 2u) {
        const Symbol *dst = args[0];
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        DualValue arg = convert_dual(load_symbol(active_layer, args[1], wd), args[1]->type(), osl_float(), wd);
        DualValue col = (op == "blackbody") ? emit_blackbody(arg, wd) : emit_wavelength_color(arg, wd);
        store_symbol(active_layer, dst, col);
        return true;
    }

    // 18. Component & array indexing & queries
    if (op == "compref" && args.size() == 3u) {
        const Symbol *dst = args[0];
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        DualValue tv = load_symbol(active_layer, args[1], wd);
        xir::Value *idx = clamp_index(load_symbol(active_layer, args[2], false).val, 3);
        auto pick_comp = [&](xir::Value *vec3) -> xir::Value * {
            if (vec3 == nullptr) { return nullptr; }
            xir::Value *c0 = extract_vec_comp(vec3, 0u);
            xir::Value *c1 = extract_vec_comp(vec3, 1u);
            xir::Value *c2 = extract_vec_comp(vec3, 2u);
            auto *b_t = compute::Type::of<bool>();
            auto *f_t = compute::Type::of<float>();
            xir::Value *is1 = builder.call(b_t, xir::ArithmeticOp::BINARY_EQUAL, {idx, const_int(1)});
            xir::Value *is2 = builder.call(b_t, xir::ArithmeticOp::BINARY_EQUAL, {idx, const_int(2)});
            xir::Value *m = builder.call(f_t, xir::ArithmeticOp::SELECT, {c0, c1, is1});
            return builder.call(f_t, xir::ArithmeticOp::SELECT, {m, c2, is2});
        };
        DualValue out;
        out.val = pick_comp(tv.val);
        if (wd) {
            out.dx = pick_comp(tv.dx);
            out.dy = pick_comp(tv.dy);
        }
        store_symbol(active_layer, dst, convert_dual(out, osl_float(), dst->type(), wd));
        return true;
    }

    if (op == "compassign" && args.size() == 3u) {
        const Symbol *dst = args[0];
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        DualValue cur = load_symbol(active_layer, dst, wd);
        xir::Value *idx = clamp_index(load_symbol(active_layer, args[1], false).val, 3);
        DualValue elem = convert_dual(load_symbol(active_layer, args[2], wd), args[2]->type(), osl_float(), wd);
        auto assign_comp = [&](xir::Value *vec3, xir::Value *ev) -> xir::Value * {
            if (vec3 == nullptr && ev == nullptr) { return nullptr; }
            xir::Value *base = vec3 ? vec3 : const_float3(luisa::float3{0.0f});
            xir::Value *fv = ev ? ev : const_float(0.0f);
            xir::Value *c[3]{};
            for (uint32_t k = 0u; k < 3u; ++k) {
                xir::Value *orig = extract_vec_comp(base, k);
                xir::Value *eq = builder.call(compute::Type::of<bool>(), xir::ArithmeticOp::BINARY_EQUAL,
                                              {idx, const_int(static_cast<int32_t>(k))});
                c[k] = builder.call(compute::Type::of<float>(), xir::ArithmeticOp::SELECT, {orig, fv, eq});
            }
            return builder.call(compute::Type::of<luisa::float3>(), xir::ArithmeticOp::AGGREGATE, {c[0], c[1], c[2]});
        };
        DualValue out;
        out.val = assign_comp(cur.val, elem.val);
        if (wd && (cur.has_derivs() || elem.has_derivs())) {
            out.dx = assign_comp(cur.dx, elem.dx);
            out.dy = assign_comp(cur.dy, elem.dy);
        }
        store_symbol(active_layer, dst, out);
        return true;
    }

    if (op == "mxcompref" && args.size() == 4u) {
        const Symbol *dst = args[0];
        xir::Value *mat_L = load_symbol(active_layer, args[1], false).val;
        xir::Value *row = clamp_index(load_symbol(active_layer, args[2], false).val, 4);
        xir::Value *col = clamp_index(load_symbol(active_layer, args[3], false).val, 4);
        // L = M_osl^T, so OSL row corresponds to XIR column `row`, element `col`
        auto *f4_t = compute::Type::of<luisa::float4>();
        auto *f_t = compute::Type::of<float>();
        auto *b_t = compute::Type::of<bool>();
        xir::Value *sel_col = builder.call(f4_t, xir::ArithmeticOp::EXTRACT, {mat_L, const_uint(0u)});
        for (uint32_t r = 1u; r < 4u; ++r) {
            xir::Value *cr = builder.call(f4_t, xir::ArithmeticOp::EXTRACT, {mat_L, const_uint(r)});
            xir::Value *eq = builder.call(b_t, xir::ArithmeticOp::BINARY_EQUAL, {row, const_int(static_cast<int32_t>(r))});
            sel_col = builder.call(f4_t, xir::ArithmeticOp::SELECT, {sel_col, cr, eq});
        }
        xir::Value *elem = builder.call(f_t, xir::ArithmeticOp::EXTRACT, {sel_col, const_uint(0u)});
        for (uint32_t c = 1u; c < 4u; ++c) {
            xir::Value *ec = builder.call(f_t, xir::ArithmeticOp::EXTRACT, {sel_col, const_uint(c)});
            xir::Value *eq = builder.call(b_t, xir::ArithmeticOp::BINARY_EQUAL, {col, const_int(static_cast<int32_t>(c))});
            elem = builder.call(f_t, xir::ArithmeticOp::SELECT, {elem, ec, eq});
        }
        store_symbol(active_layer, dst, convert_dual({.val = elem}, osl_float(), dst->type(), false));
        return true;
    }

    if (op == "mxcompassign" && args.size() == 4u) {
        const Symbol *dst = args[0];
        xir::Value *mat_L = load_symbol(active_layer, dst, false).val;
        xir::Value *row = clamp_index(load_symbol(active_layer, args[1], false).val, 4);
        xir::Value *col = clamp_index(load_symbol(active_layer, args[2], false).val, 4);
        xir::Value *fv = to_float(load_symbol(active_layer, args[3], false).val);
        auto *f4_t = compute::Type::of<luisa::float4>();
        auto *f_t = compute::Type::of<float>();
        auto *b_t = compute::Type::of<bool>();
        xir::Value *cols[4]{};
        for (uint32_t r = 0u; r < 4u; ++r) {
            xir::Value *cr = builder.call(f4_t, xir::ArithmeticOp::EXTRACT, {mat_L, const_uint(r)});
            xir::Value *row_eq = builder.call(b_t, xir::ArithmeticOp::BINARY_EQUAL, {row, const_int(static_cast<int32_t>(r))});
            xir::Value *elems[4]{};
            for (uint32_t c = 0u; c < 4u; ++c) {
                xir::Value *orig = builder.call(f_t, xir::ArithmeticOp::EXTRACT, {cr, const_uint(c)});
                xir::Value *col_eq = builder.call(b_t, xir::ArithmeticOp::BINARY_EQUAL, {col, const_int(static_cast<int32_t>(c))});
                xir::Value *hit = builder.call(b_t, xir::ArithmeticOp::BINARY_BIT_AND, {row_eq, col_eq});
                elems[c] = builder.call(f_t, xir::ArithmeticOp::SELECT, {orig, fv, hit});
            }
            cols[r] = builder.call(f4_t, xir::ArithmeticOp::AGGREGATE, {elems[0], elems[1], elems[2], elems[3]});
        }
        xir::Value *updated = builder.call(compute::Type::of<luisa::float4x4>(), xir::ArithmeticOp::AGGREGATE,
                                           {cols[0], cols[1], cols[2], cols[3]});
        store_symbol(active_layer, dst, {.val = updated});
        return true;
    }

    if (op == "aref" && args.size() == 3u) {
        const Symbol *dst = args[0];
        const Symbol *arr = args[1];
        if (!arr->is_array()) { return false; }
        xir::Value *raw_idx = load_symbol(active_layer, args[2], false).val;
        if (dst->type()->tag() == Type::Tag::CLOSURE) {
            store_closure(active_layer, dst, load_closure_array_element(active_layer, arr, raw_idx));
            return errors.empty();
        }
        if (detail::is_simple_type(dst->type(), SimpleType::Primitive::STRING)) {
            store_string_token(active_layer, dst, load_string_array_element(active_layer, arr, raw_idx));
            return errors.empty();
        }
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        DualValue ev = load_array_element(active_layer, arr, raw_idx, wd);
        if (!errors.empty()) { return false; }
        store_symbol(active_layer, dst, convert_dual(ev, arr->type(), dst->type(), wd));
        return true;
    }

    if (op == "aassign" && args.size() == 3u) {
        const Symbol *arr = args[0];
        if (!arr->is_array()) { return false; }
        xir::Value *raw_idx = load_symbol(active_layer, args[1], false).val;
        const Symbol *src = args[2];
        if (arr->type()->tag() == Type::Tag::CLOSURE) {
            ClosureValue cv = (src->type()->tag() == Type::Tag::CLOSURE)
                                  ? load_closure(active_layer, src)
                                  : owner->emit_closure_zero();
            store_closure_array_element(active_layer, arr, raw_idx, cv);
            return errors.empty();
        }
        if (detail::is_simple_type(arr->type(), SimpleType::Primitive::STRING)) {
            xir::Value *tok = get_arg_string_token(*this, active_layer, ia, 2u, src);
            store_string_array_element(active_layer, arr, raw_idx, tok);
            return errors.empty();
        }
        bool wd = dst_wants_deriv(*this, active_layer, arr);
        DualValue sv = load_symbol(active_layer, src, wd);
        store_array_element(active_layer, arr, raw_idx, convert_dual(sv, src->type(), arr->type(), wd));
        return errors.empty();
    }

    if (op == "arraylength" && args.size() == 2u) {
        int len = effective_array_length(active_layer, args[1]);
        store_symbol(active_layer, args[0],
                     convert_dual({.val = const_int(len)}, osl_int(), args[0]->type(), false));
        return true;
    }

    if (op == "isconnected" && args.size() == 2u) {
        const Symbol *q = args[1]->root();
        bool conn = false;
        const auto &ll = linked_group.layers[active_layer];
        for (uint32_t ci : ll.incoming_connections) {
            if (linked_group.connections[ci].dst_symbol->root() == q) {
                conn = true;
                break;
            }
        }
        if (!conn) {
            for (const auto &[dyn_sym, _] : ll.dynamic_inputs) {
                if (dyn_sym->root() == q) {
                    conn = true;
                    break;
                }
            }
        }
        store_symbol(active_layer, args[0],
                     convert_dual({.val = const_int(conn ? 1 : 0)}, osl_int(), args[0]->type(), false));
        return true;
    }

    // 19. String operations (constant-folded or bounded switch evaluation)
    if (op == "strlen" && args.size() == 2u) {
        StringSlotSpec s = make_string_slot(*this, active_layer, ia, 1u, args[1]);
        dispatch_strings_1(
            *this, s,
            [&](luisa::span<const luisa::string_view> resolved) {
                int32_t len = static_cast<int32_t>(resolved[0].size());
                store_symbol(active_layer, args[0],
                             convert_dual({.val = const_int(len)}, osl_int(), args[0]->type(), false));
            });
        return errors.empty();
    }

    if ((op == "startswith" || op == "endswith") && args.size() == 3u) {
        StringSlotSpec s = make_string_slot(*this, active_layer, ia, 1u, args[1]);
        StringSlotSpec p = make_string_slot(*this, active_layer, ia, 2u, args[2]);
        dispatch_strings_2(
            *this, s, p,
            [&](luisa::span<const luisa::string_view> resolved) {
                bool match = (op == "startswith")
                                 ? resolved[0].starts_with(resolved[1])
                                 : resolved[0].ends_with(resolved[1]);
                store_symbol(active_layer, args[0],
                             convert_dual({.val = const_int(match ? 1 : 0)}, osl_int(), args[0]->type(), false));
            });
        return errors.empty();
    }

    if ((op == "stoi" || op == "stof") && args.size() == 2u) {
        StringSlotSpec s = make_string_slot(*this, active_layer, ia, 1u, args[1]);
        dispatch_strings_1(
            *this, s,
            [&](luisa::span<const luisa::string_view> resolved) {
                if (op == "stoi") {
                    int32_t iv = detail::eval_osl_stoi(resolved[0]);
                    store_symbol(active_layer, args[0],
                                 convert_dual({.val = const_int(iv)}, osl_int(), args[0]->type(), false));
                } else {
                    float fv = detail::eval_osl_stof(resolved[0]);
                    store_symbol(active_layer, args[0],
                                 convert_dual({.val = const_float(fv)}, osl_float(), args[0]->type(), false));
                }
            });
        return errors.empty();
    }

    if (op == "getchar" && args.size() == 3u) {
        StringSlotSpec s = make_string_slot(*this, active_layer, ia, 1u, args[1]);
        auto c_idx = extract_arg_constant(*this, ia, 2u, args[2]);
        xir::Value *dyn_idx = (!c_idx.has_value() || !luisa::holds_alternative<int32_t>(*c_idx))
                                  ? to_int(load_symbol(active_layer, args[2], false).val)
                                  : nullptr;
        dispatch_strings_1(
            *this, s,
            [&](luisa::span<const luisa::string_view> resolved) {
                if (dyn_idx == nullptr) {
                    int32_t idx = luisa::get<int32_t>(*c_idx);
                    int32_t ch = detail::eval_osl_getchar(resolved[0], idx);
                    store_symbol(active_layer, args[0],
                                 convert_dual({.val = const_int(ch)}, osl_int(), args[0]->type(), false));
                    return;
                }
                luisa::string_view sv = resolved[0];
                const int32_t n = static_cast<int32_t>(sv.size());
                if (n == 0) {
                    store_symbol(active_layer, args[0],
                                 convert_dual({.val = const_int(0)}, osl_int(), args[0]->type(), false));
                    return;
                }
                auto *b_t = compute::Type::of<bool>();
                auto *i_t = compute::Type::of<int32_t>();
                xir::Value *ge0 = builder.call(b_t, xir::ArithmeticOp::BINARY_GREATER_EQUAL, {dyn_idx, const_int(0)});
                xir::Value *ltn = builder.call(b_t, xir::ArithmeticOp::BINARY_LESS, {dyn_idx, const_int(n)});
                xir::Value *inb = builder.call(b_t, xir::ArithmeticOp::BINARY_BIT_AND, {ge0, ltn});
                xir::Value *ch_val = nullptr;
                if (n <= 8) {
                    ch_val = const_int(0);
                    for (int32_t k = 0; k < n; ++k) {
                        xir::Value *eq = builder.call(b_t, xir::ArithmeticOp::BINARY_EQUAL, {dyn_idx, const_int(k)});
                        int32_t c = static_cast<int32_t>(static_cast<unsigned char>(sv[static_cast<size_t>(k)]));
                        ch_val = builder.call(i_t, xir::ArithmeticOp::SELECT, {ch_val, const_int(c), eq});
                    }
                } else {
                    auto *u_t = compute::Type::of<uint32_t>();
                    size_t nwords = (static_cast<size_t>(n) + 3u) / 4u;
                    const compute::Type *arr_t = compute::Type::array(u_t, nwords);
                    xir::Value *packed = builder.alloca_local(arr_t);
                    for (size_t w = 0u; w < nwords; ++w) {
                        uint32_t word = 0u;
                        for (size_t b = 0u; b < 4u; ++b) {
                            size_t pos = w * 4u + b;
                            if (pos < sv.size()) {
                                word |= static_cast<uint32_t>(static_cast<unsigned char>(sv[pos])) << (b * 8u);
                            }
                        }
                        builder.store(builder.gep(u_t, packed, {const_int(static_cast<int32_t>(w))}),
                                      const_uint(word));
                    }
                    xir::Value *clamped = builder.call(i_t, xir::ArithmeticOp::CLAMP,
                                                       {dyn_idx, const_int(0), const_int(n - 1)});
                    xir::Value *u_idx = builder.static_cast_(u_t, clamped);
                    xir::Value *w_idx = builder.call(u_t, xir::ArithmeticOp::BINARY_SHIFT_RIGHT, {u_idx, const_uint(2u)});
                    xir::Value *word_v = builder.load(u_t, builder.gep(u_t, packed, {w_idx}));
                    xir::Value *byte_in_w = builder.call(u_t, xir::ArithmeticOp::BINARY_BIT_AND, {u_idx, const_uint(3u)});
                    xir::Value *shift = builder.call(u_t, xir::ArithmeticOp::BINARY_SHIFT_LEFT, {byte_in_w, const_uint(3u)});
                    xir::Value *shifted = builder.call(u_t, xir::ArithmeticOp::BINARY_SHIFT_RIGHT, {word_v, shift});
                    xir::Value *byte_u = builder.call(u_t, xir::ArithmeticOp::BINARY_BIT_AND, {shifted, const_uint(0xFFu)});
                    xir::Value *byte_i = builder.static_cast_(i_t, byte_u);
                    ch_val = builder.call(i_t, xir::ArithmeticOp::SELECT, {const_int(0), byte_i, inb});
                }
                store_symbol(active_layer, args[0],
                             convert_dual({.val = ch_val}, osl_int(), args[0]->type(), false));
            });
        return errors.empty();
    }

    if (op == "split" && args.size() >= 3u) {
        const Symbol *dst_count = args[0];
        const Symbol *src_str = args[1];
        const Symbol *out_arr = args[2];
        if (!out_arr->is_array() || !detail::is_simple_type(out_arr->type(), SimpleType::Primitive::STRING) ||
            !detail::is_simple_type(src_str->type(), SimpleType::Primitive::STRING)) {
            return false;
        }
        int arr_len = effective_array_length(active_layer, out_arr);
        int32_t maxsplit = -1;
        if (args.size() >= 5u) {
            auto c_ms = extract_arg_constant(*this, ia, 4u, args[4]);
            if (!c_ms.has_value() || !luisa::holds_alternative<int32_t>(*c_ms)) {
                report_error("split(): dynamic runtime maxsplit cannot be statically evaluated into StringTable.");
                return false;
            }
            maxsplit = luisa::get<int32_t>(*c_ms);
        }
        luisa::vector<StringSlotSpec> slots;
        slots.emplace_back(make_string_slot(*this, active_layer, ia, 1u, src_str));
        bool has_sep = args.size() >= 4u;
        if (has_sep) {
            slots.emplace_back(make_string_slot(*this, active_layer, ia, 3u, args[3]));
        }
        dispatch_dynamic_strings(
            slots,
            [&](luisa::span<const luisa::string_view> resolved) {
                luisa::string_view sep = has_sep ? resolved[1] : luisa::string_view{};
                auto pieces = detail::eval_osl_split(resolved[0], sep, maxsplit);
                int stored = std::min(arr_len, static_cast<int>(pieces.size()));
                for (int k = 0; k < stored; ++k) {
                    store_string_array_element(active_layer, out_arr, const_int(k),
                                               token_for_string(pieces[static_cast<size_t>(k)]));
                }
                store_symbol(active_layer, dst_count,
                             convert_dual({.val = const_int(stored)}, osl_int(), dst_count->type(), false));
            });
        return errors.empty();
    }

    if (op == "substr" && (args.size() == 3u || args.size() == 4u)) {
        StringSlotSpec s = make_string_slot(*this, active_layer, ia, 1u, args[1]);
        auto c_start = extract_arg_constant(*this, ia, 2u, args[2]);
        luisa::optional<ConstantOptionValue> c_len = (args.size() == 4u)
                                                         ? extract_arg_constant(*this, ia, 3u, args[3])
                                                         : luisa::optional<ConstantOptionValue>{ConstantOptionValue{INT32_MAX}};
        if (!c_start || !c_len || !luisa::holds_alternative<int32_t>(*c_start) ||
            !luisa::holds_alternative<int32_t>(*c_len)) {
            report_error("substr(): dynamic start/length cannot be statically evaluated into StringTable.");
            return false;
        }
        int32_t start_i = luisa::get<int32_t>(*c_start);
        int32_t len_i = luisa::get<int32_t>(*c_len);
        dispatch_strings_1(
            *this, s,
            [&](luisa::span<const luisa::string_view> resolved) {
                int32_t n = (args.size() == 4u) ? len_i : static_cast<int32_t>(resolved[0].size());
                luisa::string sub = detail::eval_osl_substr(resolved[0], start_i, n);
                store_string_token(active_layer, args[0], token_for_string(sub));
            });
        return errors.empty();
    }

    if (op == "concat" && args.size() >= 3u) {
        luisa::vector<StringSlotSpec> slots;
        slots.reserve(args.size() - 1u);
        for (size_t k = 1u; k < args.size(); ++k) {
            slots.emplace_back(make_string_slot(*this, active_layer, ia, k, args[k]));
        }
        dispatch_dynamic_strings(
            slots,
            [&](luisa::span<const luisa::string_view> resolved) {
                luisa::string out;
                for (auto sv : resolved) { out.append(sv); }
                store_string_token(active_layer, args[0], token_for_string(out));
            });
        return errors.empty();
    }

    if (op == "format" && args.size() >= 2u) {
        luisa::vector<StringSlotSpec> slots;
        luisa::vector<size_t> str_slot_for_arg(args.size(), ~0u);
        slots.emplace_back(make_string_slot(*this, active_layer, ia, 1u, args[1]));
        str_slot_for_arg[1] = 0u;
        size_t arg_limit = args.size();
        if (slots[0].lattice.is_set() && !slots[0].lattice.tokens.empty()) {
            auto count_specs = [](luisa::string_view f) noexcept -> luisa::optional<size_t> {
                size_t cnt = 0u;
                for (size_t i = 0u; i < f.size(); ++i) {
                    if (f[i] != '%') { continue; }
                    if (i + 1u >= f.size()) { return luisa::nullopt; }
                    if (f[i + 1u] == '%') { ++i; continue; }
                    size_t j = i + 1u;
                    while (j < f.size() && (f[j] == '-' || f[j] == '+' || f[j] == ' ' || f[j] == '#' || f[j] == '0')) { ++j; }
                    while (j < f.size() && std::isdigit(static_cast<unsigned char>(f[j]))) { ++j; }
                    if (j < f.size() && f[j] == '.') {
                        ++j;
                        while (j < f.size() && std::isdigit(static_cast<unsigned char>(f[j]))) { ++j; }
                    }
                    if (j >= f.size()) { return luisa::nullopt; }
                    ++cnt;
                    i = j;
                }
                return cnt;
            };
            size_t max_specs = 0u;
            bool specs_ok = true;
            for (StringToken ftok : slots[0].lattice.tokens) {
                auto c = count_specs(string_table.lookup(ftok));
                if (!c.has_value()) { specs_ok = false; break; }
                max_specs = std::max(max_specs, *c);
            }
            if (specs_ok) {
                arg_limit = std::min(args.size(), 2u + max_specs);
            }
        }
        luisa::vector<luisa::optional<ConstantOptionValue>> static_args(args.size());
        for (size_t k = 2u; k < arg_limit; ++k) {
            if (detail::is_simple_type(args[k]->type(), SimpleType::Primitive::STRING)) {
                str_slot_for_arg[k] = slots.size();
                slots.emplace_back(make_string_slot(*this, active_layer, ia, k, args[k]));
            } else {
                static_args[k] = extract_arg_constant(*this, ia, k, args[k]);
                if (!static_args[k].has_value()) {
                    report_error(luisa::format(
                        "format(): non-constant runtime argument '{}' prevents static string evaluation.",
                        args[k]->identifier()));
                    return false;
                }
            }
        }
        dispatch_dynamic_strings(
            slots,
            [&](luisa::span<const luisa::string_view> resolved) {
                luisa::vector<detail::HostFormatArg> fmt_args;
                fmt_args.reserve(arg_limit - 2u);
                for (size_t k = 2u; k < arg_limit; ++k) {
                    if (str_slot_for_arg[k] != ~0u) {
                        fmt_args.emplace_back(resolved[str_slot_for_arg[k]]);
                    } else {
                        const auto &cv = *static_args[k];
                        if (auto *p = luisa::get_if<int32_t>(&cv)) { fmt_args.emplace_back(*p); }
                        else if (auto *p = luisa::get_if<float>(&cv)) { fmt_args.emplace_back(*p); }
                        else if (auto *p = luisa::get_if<luisa::float3>(&cv)) { fmt_args.emplace_back(*p); }
                        else if (auto *p = luisa::get_if<luisa::float4x4>(&cv)) { fmt_args.emplace_back(*p); }
                    }
                }
                auto formatted = detail::eval_osl_format(resolved[0], fmt_args);
                if (!formatted.has_value()) {
                    report_error("format(): malformed OSL format string or argument count mismatch.");
                    return;
                }
                store_string_token(active_layer, args[0], token_for_string(*formatted));
            });
        return errors.empty();
    }

    if (op == "hash" && args.size() == 2u) {
        const Symbol *dst = args[0];
        const Symbol *src = args[1];
        if (detail::is_simple_type(src->type(), SimpleType::Primitive::STRING)) {
            StringSlotSpec s = make_string_slot(*this, active_layer, ia, 1u, src);
            dispatch_strings_1(
                *this, s,
                [&](luisa::span<const luisa::string_view> resolved) {
                    int32_t hv = detail::eval_osl_string_hash(resolved[0]);
                    store_symbol(active_layer, dst,
                                 convert_dual({.val = const_int(hv)}, osl_int(), dst->type(), false));
                });
            return errors.empty();
        }
        // Numeric deterministic 32-bit hash in XIR
        auto mix_u32 = [&](xir::Value *h, xir::Value *x) {
            auto *u_t = compute::Type::of<uint32_t>();
            xir::Value *k = builder.call(u_t, xir::ArithmeticOp::BINARY_BIT_XOR, {h, x});
            k = builder.call(u_t, xir::ArithmeticOp::BINARY_MUL, {k, const_uint(0x85ebca6bu)});
            k = builder.call(u_t, xir::ArithmeticOp::BINARY_BIT_XOR,
                             {k, builder.call(u_t, xir::ArithmeticOp::BINARY_SHIFT_RIGHT, {k, const_uint(13u)})});
            k = builder.call(u_t, xir::ArithmeticOp::BINARY_MUL, {k, const_uint(0xc2b2ae35u)});
            return builder.call(u_t, xir::ArithmeticOp::BINARY_BIT_XOR,
                                {k, builder.call(u_t, xir::ArithmeticOp::BINARY_SHIFT_RIGHT, {k, const_uint(16u)})});
        };
        auto *u_t = compute::Type::of<uint32_t>();
        xir::Value *sv = load_symbol(active_layer, src, false).val;
        xir::Value *h = const_uint(0x9e3779b9u);
        if (sv->type() == compute::Type::of<int32_t>()) {
            h = mix_u32(h, builder.bit_cast_(u_t, sv));
        } else if (sv->type() == compute::Type::of<float>()) {
            h = mix_u32(h, builder.bit_cast_(u_t, sv));
        } else if (sv->type() == compute::Type::of<luisa::float3>()) {
            for (uint32_t c = 0u; c < 3u; ++c) {
                h = mix_u32(h, builder.bit_cast_(u_t, extract_vec_comp(sv, c)));
            }
        } else {
            return false;
        }
        xir::Value *hi = builder.static_cast_(compute::Type::of<int32_t>(), h);
        store_symbol(active_layer, dst, convert_dual({.val = hi}, osl_int(), dst->type(), false));
        return true;
    }

    // 20. Coordinate / color space transforms
    if ((op == "transform" || op == "transformv" || op == "transformn") &&
        (args.size() == 3u || args.size() == 4u)) {
        const Symbol *dst = args[0];
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        TransformKind tk = TransformKind::POINT;
        if (op == "transformv") { tk = TransformKind::VECTOR; }
        else if (op == "transformn") { tk = TransformKind::NORMAL; }

        if (args.size() == 3u) {
            if (detail::is_simple_type(args[1]->type(), SimpleType::Primitive::MATRIX)) {
                xir::Value *mat_L = load_symbol(active_layer, args[1], false).val;
                DualValue v = convert_dual(load_symbol(active_layer, args[2], wd), args[2]->type(), dst->type(), wd);
                store_symbol(active_layer, dst, transform_dual_by_matrix(mat_L, tk, v, wd));
                return true;
            }
            if (detail::is_simple_type(args[1]->type(), SimpleType::Primitive::STRING)) {
                StringSlotSpec s_to = make_string_slot(*this, active_layer, ia, 1u, args[1]);
                DualValue v = convert_dual(load_symbol(active_layer, args[2], wd), args[2]->type(), dst->type(), wd);
                dispatch_strings_1(
                    *this, s_to,
                    [&](luisa::span<const luisa::string_view> resolved) {
                        DualValue tr = owner->emit_transform_space("common", resolved[0], tk, v, wd);
                        if (errors.empty() && tr.valid()) {
                            store_symbol(active_layer, dst, tr);
                        }
                    });
                return errors.empty();
            }
            return false;
        }
        StringSlotSpec s_from = make_string_slot(*this, active_layer, ia, 1u, args[1]);
        StringSlotSpec s_to = make_string_slot(*this, active_layer, ia, 2u, args[2]);
        DualValue v = convert_dual(load_symbol(active_layer, args[3], wd), args[3]->type(), dst->type(), wd);
        dispatch_strings_2(
            *this, s_from, s_to,
            [&](luisa::span<const luisa::string_view> resolved) {
                DualValue tr = owner->emit_transform_space(resolved[0], resolved[1], tk, v, wd);
                if (errors.empty() && tr.valid()) {
                    store_symbol(active_layer, dst, tr);
                }
            });
        return errors.empty();
    }

    if (op == "transformc" && args.size() == 4u) {
        const Symbol *dst = args[0];
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        StringSlotSpec s_from = make_string_slot(*this, active_layer, ia, 1u, args[1]);
        StringSlotSpec s_to = make_string_slot(*this, active_layer, ia, 2u, args[2]);
        DualValue v = convert_dual(load_symbol(active_layer, args[3], wd), args[3]->type(), osl_color(), wd);
        dispatch_strings_2(
            *this, s_from, s_to,
            [&](luisa::span<const luisa::string_view> resolved) {
                DualValue tr = owner->emit_transform_color(resolved[0], resolved[1], v, wd);
                if (errors.empty() && tr.valid()) {
                    store_symbol(active_layer, dst, tr);
                }
            });
        return errors.empty();
    }

    if (op == "getmatrix" && args.size() == 4u) {
        const Symbol *dst = args[0];
        StringSlotSpec s_from = make_string_slot(*this, active_layer, ia, 1u, args[1]);
        StringSlotSpec s_to = make_string_slot(*this, active_layer, ia, 2u, args[2]);
        const Symbol *out_m = args[3];
        dispatch_strings_2(
            *this, s_from, s_to,
            [&](luisa::span<const luisa::string_view> resolved) {
                xir::Value *mat_L = nullptr;
                xir::Value *ok = owner->emit_getmatrix(resolved[0], resolved[1], mat_L);
                if (!errors.empty()) { return; }
                store_symbol(active_layer, dst, convert_dual({.val = ok}, osl_int(), dst->type(), false));
                conditional_update_symbol(*this, active_layer, out_m, ok, false, [&]() -> DualValue {
                    return {.val = mat_L ? mat_L : const_float4x4(luisa::make_float4x4(0.0f))};
                });
            });
        return errors.empty();
    }

    // 21. Noise family: noise, pnoise, snoise, psnoise, cellnoise, hashnoise
    if ((op == "noise" || op == "pnoise" || op == "snoise" ||
         op == "psnoise" || op == "cellnoise" || op == "hashnoise") &&
        args.size() >= 2u) {
        const Symbol *dst = args[0];
        bool wd = dst_wants_deriv(*this, active_layer, dst);
        bool explicit_type = (op == "noise" || op == "pnoise") &&
                             detail::is_simple_type(args[1]->type(), SimpleType::Primitive::STRING);

        NoiseOp nop = NoiseOp::NOISE;
        luisa::string_view implicit_type = "uperlin";
        if (op == "pnoise") { nop = NoiseOp::PNOISE; }
        else if (op == "snoise") { nop = NoiseOp::SNOISE; implicit_type = "perlin"; }
        else if (op == "psnoise") { nop = NoiseOp::PSNOISE; implicit_type = "perlin"; }
        else if (op == "cellnoise") { nop = NoiseOp::CELLNOISE; implicit_type = "cell"; }
        else if (op == "hashnoise") { nop = NoiseOp::HASHNOISE; implicit_type = "hash"; }

        size_t pos_start = explicit_type ? 2u : 1u;
        size_t opt_start = pos_start;
        while (opt_start < args.size() &&
               !detail::is_simple_type(args[opt_start]->type(), SimpleType::Primitive::STRING)) {
            ++opt_start;
        }
        size_t num_pos = opt_start - pos_start;
        if (num_pos == 0u) { return false; }

        bool is_periodic = (nop == NoiseOp::PNOISE || nop == NoiseOp::PSNOISE);
        if (is_periodic && (num_pos % 2u != 0u)) { return false; }
        size_t coord_count = is_periodic ? (num_pos / 2u) : num_pos;
        if (coord_count > 2u) { return false; }

        luisa::vector<DualValue> coords;
        luisa::vector<DualValue> periods;
        coords.reserve(coord_count);
        for (size_t k = 0u; k < coord_count; ++k) {
            const Symbol *cs = args[pos_start + k];
            DualValue dv = load_symbol(active_layer, cs, wd);
            const Type *target_t = detail::is_triple_type(cs->type()) ? osl_point() : osl_float();
            coords.emplace_back(convert_dual(dv, cs->type(), target_t, wd));
        }
        if (is_periodic) {
            periods.reserve(coord_count);
            for (size_t k = 0u; k < coord_count; ++k) {
                const Symbol *ps = args[pos_start + coord_count + k];
                DualValue dv = load_symbol(active_layer, ps, false);
                const Type *target_t = detail::is_triple_type(ps->type()) ? osl_point() : osl_float();
                periods.emplace_back(convert_dual(dv, ps->type(), target_t, false));
            }
        }

        luisa::vector<StringSlotSpec> slots;
        size_t type_slot = ~0u;
        if (explicit_type) {
            type_slot = slots.size();
            slots.emplace_back(make_string_slot(*this, active_layer, ia, 1u, args[1]));
        }
        luisa::vector<PendingStringOptEntry> pending_opts;
        collect_named_options(*this, active_layer, ia, inst, opt_start, args.size(),
                              wd, slots, pending_opts);
        if (!errors.empty()) { return false; }

        dispatch_dynamic_strings(
            slots,
            [&](luisa::span<const luisa::string_view> resolved) {
                luisa::string_view ntype = explicit_type ? resolved[type_slot] : implicit_type;
                auto opts = materialize_named_options(inst, *this, active_layer, pending_opts, resolved);
                DualValue r = owner->emit_noise(ntype, nop, coords, periods, dst->type(), wd, opts);
                if (errors.empty() && r.valid()) {
                    store_symbol(active_layer, dst, r);
                }
            });
        return errors.empty();
    }

    // 22. Shading point flags & ray queries
    if (op == "backfacing" && args.size() == 1u) {
        xir::Value *bf = owner->emit_backfacing();
        if (!errors.empty() || bf == nullptr) { return false; }
        store_symbol(active_layer, args[0], convert_dual({.val = bf}, osl_int(), args[0]->type(), false));
        return true;
    }

    if (op == "raytype" && args.size() == 2u) {
        StringSlotSpec s = make_string_slot(*this, active_layer, ia, 1u, args[1]);
        dispatch_strings_1(
            *this, s,
            [&](luisa::span<const luisa::string_view> resolved) {
                xir::Value *rt = owner->emit_raytype(resolved[0]);
                if (errors.empty() && rt != nullptr) {
                    store_symbol(active_layer, args[0], convert_dual({.val = rt}, osl_int(), args[0]->type(), false));
                }
            });
        return errors.empty();
    }

    if (op == "trace" && args.size() >= 3u) {
        const Symbol *dst = args[0];
        DualValue pos = convert_dual(load_symbol(active_layer, args[1], false), args[1]->type(), osl_point(), false);
        DualValue dir = convert_dual(load_symbol(active_layer, args[2], false), args[2]->type(), osl_vector(), false);
        luisa::vector<StringSlotSpec> slots;
        luisa::vector<PendingStringOptEntry> pending_opts;
        collect_named_options(*this, active_layer, ia, inst, 3u, args.size(),
                              true, slots, pending_opts);
        if (!errors.empty()) { return false; }
        dispatch_dynamic_strings(
            slots,
            [&](luisa::span<const luisa::string_view> resolved) {
                auto opts = materialize_named_options(inst, *this, active_layer, pending_opts, resolved);
                xir::Value *hit = owner->emit_trace(pos, dir, opts);
                if (errors.empty() && hit != nullptr) {
                    store_symbol(active_layer, dst, convert_dual({.val = hit}, osl_int(), dst->type(), false));
                }
            });
        return errors.empty();
    }

    // 23. Attributes & messages
    if ((op == "getattribute" || op == "getmessage") && (args.size() == 3u || args.size() == 4u)) {
        const Symbol *dst = args[0];
        bool has_first_str = (args.size() == 4u);
        const Symbol *first_sym = has_first_str ? args[1] : nullptr;
        const Symbol *name_sym = has_first_str ? args[2] : args[1];
        const Symbol *out_sym = has_first_str ? args[3] : args[2];
        if ((has_first_str && !detail::is_simple_type(first_sym->type(), SimpleType::Primitive::STRING)) ||
            !detail::is_simple_type(name_sym->type(), SimpleType::Primitive::STRING)) {
            return false;
        }
        bool wd = dst_wants_deriv(*this, active_layer, out_sym);
        int out_arr_len = effective_array_length(active_layer, out_sym);

        luisa::vector<StringSlotSpec> slots;
        if (has_first_str) {
            slots.emplace_back(make_string_slot(*this, active_layer, ia, 1u, first_sym));
        }
        slots.emplace_back(make_string_slot(*this, active_layer, ia, has_first_str ? 2u : 1u, name_sym));

        dispatch_dynamic_strings(
            slots,
            [&](luisa::span<const luisa::string_view> resolved) {
                luisa::optional<luisa::string_view> opt_first =
                    has_first_str ? luisa::optional<luisa::string_view>{resolved[0]} : luisa::nullopt;
                luisa::string_view attr_or_msg = has_first_str ? resolved[1] : resolved[0];
                DualValue queried{};
                xir::Value *ok = (op == "getattribute")
                                     ? owner->emit_getattribute(opt_first, attr_or_msg, out_sym->type(), out_arr_len, wd, queried)
                                     : owner->emit_getmessage(opt_first, attr_or_msg, out_sym->type(), out_arr_len, wd, queried);
                if (!errors.empty() || ok == nullptr) { return; }
                store_symbol(active_layer, dst, convert_dual({.val = ok}, osl_int(), dst->type(), false));
                conditional_update_symbol(*this, active_layer, out_sym, ok, wd, [&]() -> DualValue {
                    return queried;
                });
            });
        return errors.empty();
    }

    if (op == "setmessage" && args.size() == 2u) {
        const Symbol *name_sym = args[0];
        const Symbol *val_sym = args[1];
        if (!detail::is_simple_type(name_sym->type(), SimpleType::Primitive::STRING)) { return false; }
        StringSlotSpec s_name = make_string_slot(*this, active_layer, ia, 0u, name_sym);
        int arr_len = effective_array_length(active_layer, val_sym);
        DualValue dv;
        if (detail::is_simple_type(val_sym->type(), SimpleType::Primitive::STRING)) {
            dv = {.val = get_arg_string_token(*this, active_layer, ia, 1u, val_sym)};
        } else {
            dv = load_symbol(active_layer, val_sym, detail::is_differentiable_type(val_sym->type()));
        }
        dispatch_strings_1(
            *this, s_name,
            [&](luisa::span<const luisa::string_view> resolved) {
                owner->emit_setmessage(resolved[0], val_sym->type(), arr_len, dv);
            });
        return errors.empty();
    }

    // 24. Texture & gettextureinfo
    if ((op == "texture" || op == "texture3d" || op == "environment") && args.size() >= 3u) {
        const Symbol *dst = args[0];
        const Symbol *fname = args[1];
        if (!detail::is_simple_type(fname->type(), SimpleType::Primitive::STRING)) { return false; }
        bool wd = dst_wants_deriv(*this, active_layer, dst);

        TextureOp top = TextureOp::TEXTURE_2D;
        if (op == "texture3d") { top = TextureOp::TEXTURE_3D; }
        else if (op == "environment") { top = TextureOp::ENVIRONMENT; }

        size_t pos_coord_count = (top == TextureOp::TEXTURE_2D) ? 2u : 1u;
        size_t explicit_deriv_count = (top == TextureOp::TEXTURE_2D) ? 4u : 2u;
        if (args.size() < 2u + pos_coord_count) { return false; }

        size_t first_str_after = 2u + pos_coord_count;
        while (first_str_after < args.size() &&
               !detail::is_simple_type(args[first_str_after]->type(), SimpleType::Primitive::STRING)) {
            ++first_str_after;
        }
        size_t extra_nums = first_str_after - (2u + pos_coord_count);
        bool has_explicit_derivs = (extra_nums == explicit_deriv_count);
        size_t opt_start = (2u + pos_coord_count) + (has_explicit_derivs ? explicit_deriv_count : 0u);
        if (first_str_after != opt_start) { return false; }

        luisa::vector<DualValue> coords;
        for (size_t k = 0u; k < pos_coord_count; ++k) {
            const Symbol *cs = args[2u + k];
            bool need_c_deriv = !has_explicit_derivs &&
                                (wd || cs->tag() == Symbol::Tag::SYM_GLOBAL ||
                                 (find_storage(active_layer, cs) && find_storage(active_layer, cs)->is_dual));
            DualValue dv = load_symbol(active_layer, cs, need_c_deriv);
            const Type *target_t = (top == TextureOp::TEXTURE_2D) ? osl_float() : osl_vector();
            coords.emplace_back(convert_dual(dv, cs->type(), target_t, need_c_deriv));
        }

        luisa::vector<DualValue> exp_derivs;
        if (has_explicit_derivs) {
            for (size_t k = 0u; k < explicit_deriv_count; ++k) {
                const Symbol *ds = args[2u + pos_coord_count + k];
                DualValue dv = load_symbol(active_layer, ds, false);
                const Type *target_t = (top == TextureOp::TEXTURE_2D) ? osl_float() : osl_vector();
                exp_derivs.emplace_back(convert_dual(dv, ds->type(), target_t, false));
            }
        }

        luisa::vector<StringSlotSpec> slots;
        slots.emplace_back(make_string_slot(*this, active_layer, ia, 1u, fname));
        luisa::vector<PendingStringOptEntry> pending_opts;
        collect_named_options(*this, active_layer, ia, inst, opt_start, args.size(),
                              true, slots, pending_opts);
        if (!errors.empty()) { return false; }

        dispatch_dynamic_strings(
            slots,
            [&](luisa::span<const luisa::string_view> resolved) {
                auto opts = materialize_named_options(inst, *this, active_layer, pending_opts, resolved);
                DualValue tex = owner->emit_texture(
                    top, resolved[0], coords, exp_derivs, dst->type(), wd, opts);
                if (errors.empty() && tex.valid()) {
                    store_symbol(active_layer, dst, convert_dual(tex, dst->type(), dst->type(), wd));
                }
            });
        return errors.empty();
    }

    if (op == "gettextureinfo" && args.size() == 4u) {
        const Symbol *dst = args[0];
        const Symbol *fname = args[1];
        const Symbol *dataname = args[2];
        const Symbol *out_sym = args[3];
        if (!detail::is_simple_type(fname->type(), SimpleType::Primitive::STRING) ||
            !detail::is_simple_type(dataname->type(), SimpleType::Primitive::STRING)) {
            return false;
        }
        StringSlotSpec s_fn = make_string_slot(*this, active_layer, ia, 1u, fname);
        StringSlotSpec s_dn = make_string_slot(*this, active_layer, ia, 2u, dataname);
        int out_arr_len = effective_array_length(active_layer, out_sym);
        dispatch_strings_2(
            *this, s_fn, s_dn,
            [&](luisa::span<const luisa::string_view> resolved) {
                DualValue out_v{};
                xir::Value *ok = owner->emit_gettextureinfo(
                    resolved[0], resolved[1], out_sym->type(), out_arr_len, out_v);
                if (!errors.empty() || ok == nullptr) { return; }
                store_symbol(active_layer, dst, convert_dual({.val = ok}, osl_int(), dst->type(), false));
                conditional_update_symbol(*this, active_layer, out_sym, ok, false, [&]() -> DualValue {
                    return out_v;
                });
            });
        return errors.empty();
    }

    // 25. Splines: spline & splineinverse
    if ((op == "spline" || op == "splineinverse") && args.size() >= 4u) {
        const Symbol *dst = args[0];
        const Symbol *basis_sym = args[1];
        const Symbol *x_sym = args[2];
        if (!detail::is_simple_type(basis_sym->type(), SimpleType::Primitive::STRING)) {
            return false;
        }
        const bool is_inv = (op == "splineinverse");
        if (is_inv && !detail::is_simple_type(dst->type(), SimpleType::Primitive::FLOAT) &&
            !detail::is_simple_type(dst->type(), SimpleType::Primitive::INT)) {
            return false;
        }

        const bool is_array_4 = (args.size() == 4u);
        const bool is_array_5 = (args.size() == 5u && args[4]->is_array());
        const bool is_array_knots = is_array_4 || is_array_5;
        const Symbol *knot_arr_sym = is_array_4 ? args[3] : (is_array_5 ? args[4] : nullptr);
        int nknots = 0;
        if (is_array_knots) {
            if (!knot_arr_sym->is_array()) { return false; }
            int eff_len = effective_array_length(active_layer, knot_arr_sym);
            if (is_array_5) {
                auto c_nk = extract_arg_constant(*this, ia, 3u, args[3]);
                if (!c_nk.has_value() || !luisa::holds_alternative<int32_t>(*c_nk)) {
                    report_error(luisa::format("{}: dynamic runtime nknots is not supported.", op));
                    return false;
                }
                nknots = luisa::get<int32_t>(*c_nk);
                if (nknots < 0 || nknots > eff_len) {
                    report_error(luisa::format("{}: nknots ({}) exceeds array length ({}).", op, nknots, eff_len));
                    return false;
                }
            } else {
                nknots = eff_len;
            }
        } else {
            nknots = static_cast<int>(args.size() - 3u);
        }

        const Type *eval_osl = is_inv
                                   ? osl_float()
                                   : (detail::is_triple_type(dst->type()) ? dst->type() : osl_float());
        const compute::Type *xt = detail::osl_type_to_xir_type(eval_osl, 0, owner->closure_xir_type());
        const bool wd = dst_wants_deriv(*this, active_layer, dst);
        const bool knot_deriv_possible = !is_inv || wd;

        // Preload variadic knots once (if not array-form)
        luisa::vector<DualValue> variadic_knots;
        if (!is_array_knots) {
            variadic_knots.reserve(static_cast<size_t>(nknots));
            for (size_t k = 3u; k < args.size(); ++k) {
                variadic_knots.emplace_back(
                    convert_dual(load_symbol(active_layer, args[k], wd && knot_deriv_possible),
                                 args[k]->type(), eval_osl, wd && knot_deriv_possible));
            }
        }

        DualValue x_or_v = convert_dual(load_symbol(active_layer, x_sym, wd), x_sym->type(), osl_float(), wd);
        StringSlotSpec s_basis = make_string_slot(*this, active_layer, ia, 1u, basis_sym);

        dispatch_strings_1(
            *this, s_basis,
            [&](luisa::span<const luisa::string_view> resolved) {
                luisa::string_view basis = resolved[0];
                int step = 1;
                float mat[4][4]{};
                if (basis == "catmull-rom") {
                    step = 1;
                    float m[4][4] = {
                        {-0.5f, 1.5f, -1.5f, 0.5f},
                        {1.0f, -2.5f, 2.0f, -0.5f},
                        {-0.5f, 0.0f, 0.5f, 0.0f},
                        {0.0f, 1.0f, 0.0f, 0.0f}};
                    std::memcpy(mat, m, sizeof(mat));
                } else if (basis == "bspline") {
                    step = 1;
                    float m[4][4] = {
                        {-1.0f / 6.0f, 3.0f / 6.0f, -3.0f / 6.0f, 1.0f / 6.0f},
                        {3.0f / 6.0f, -6.0f / 6.0f, 3.0f / 6.0f, 0.0f},
                        {-3.0f / 6.0f, 0.0f, 3.0f / 6.0f, 0.0f},
                        {1.0f / 6.0f, 4.0f / 6.0f, 1.0f / 6.0f, 0.0f}};
                    std::memcpy(mat, m, sizeof(mat));
                } else if (basis == "bezier") {
                    step = 3;
                    float m[4][4] = {
                        {-1.0f, 3.0f, -3.0f, 1.0f},
                        {3.0f, -6.0f, 3.0f, 0.0f},
                        {-3.0f, 3.0f, 0.0f, 0.0f},
                        {1.0f, 0.0f, 0.0f, 0.0f}};
                    std::memcpy(mat, m, sizeof(mat));
                } else if (basis == "hermite") {
                    step = 2;
                    float m[4][4] = {
                        {2.0f, 1.0f, -2.0f, 1.0f},
                        {-3.0f, -2.0f, 3.0f, -1.0f},
                        {0.0f, 1.0f, 0.0f, 0.0f},
                        {1.0f, 0.0f, 0.0f, 0.0f}};
                    std::memcpy(mat, m, sizeof(mat));
                } else if (basis == "linear") {
                    step = 1;
                    float m[4][4] = {
                        {0.0f, 0.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 0.0f, 0.0f},
                        {0.0f, -1.0f, 1.0f, 0.0f},
                        {0.0f, 1.0f, 0.0f, 0.0f}};
                    std::memcpy(mat, m, sizeof(mat));
                } else if (basis == "constant") {
                    step = 1;
                    float m[4][4] = {
                        {0.0f, 0.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 0.0f, 0.0f},
                        {0.0f, 1.0f, 0.0f, 0.0f}};
                    std::memcpy(mat, m, sizeof(mat));
                } else {
                    report_error(luisa::format("{}: unknown spline basis '{}'.", op, basis));
                    return;
                }

                if (nknots < 4 || ((nknots - 4) % step) != 0) {
                    report_error(luisa::format(
                        "{}('{}'): invalid knot count {} (requires >= 4 and (n - 4) % {} == 0).",
                        op, basis, nknots, step));
                    return;
                }
                const int nsegs = (nknots - 4) / step + 1;
                auto *f_t = compute::Type::of<float>();
                auto *i_t = compute::Type::of<int32_t>();
                auto *b_t = compute::Type::of<bool>();

                auto load_knot = [&](xir::Value *kidx, bool need_d) -> DualValue {
                    if (is_array_knots) {
                        DualValue dv = load_array_element(active_layer, knot_arr_sym, kidx, need_d);
                        return convert_dual(dv, knot_arr_sym->type(), eval_osl, need_d);
                    }
                    // Static or dynamic selection from preloaded variadic knots
                    DualValue acc{.val = variadic_knots[0].val};
                    if (need_d) {
                        acc.dx = variadic_knots[0].dx ? variadic_knots[0].dx : zero_of(xt);
                        acc.dy = variadic_knots[0].dy ? variadic_knots[0].dy : zero_of(xt);
                    }
                    const compute::Type *sel_bt = bool_type_for(xt);
                    for (int k = 1; k < nknots; ++k) {
                        xir::Value *eq = builder.call(b_t, xir::ArithmeticOp::BINARY_EQUAL, {kidx, const_int(k)});
                        xir::Value *mask = make_bool_mask(*this, eq, xt);
                        acc.val = builder.call(xt, xir::ArithmeticOp::SELECT,
                                               {acc.val, variadic_knots[static_cast<size_t>(k)].val, mask});
                        if (need_d) {
                            xir::Value *kdx = variadic_knots[static_cast<size_t>(k)].dx
                                                  ? variadic_knots[static_cast<size_t>(k)].dx
                                                  : zero_of(xt);
                            xir::Value *kdy = variadic_knots[static_cast<size_t>(k)].dy
                                                  ? variadic_knots[static_cast<size_t>(k)].dy
                                                  : zero_of(xt);
                            acc.dx = builder.call(xt, xir::ArithmeticOp::SELECT, {acc.dx, kdx, mask});
                            acc.dy = builder.call(xt, xir::ArithmeticOp::SELECT, {acc.dy, kdy, mask});
                        }
                        (void)sel_bt;
                    }
                    return acc;
                };

                auto linear_comb = [&](const float row[4], xir::Value *k0, xir::Value *k1,
                                       xir::Value *k2, xir::Value *k3) -> xir::Value * {
                    xir::Value *sum = nullptr;
                    xir::Value *ks[4] = {k0, k1, k2, k3};
                    for (int j = 0; j < 4; ++j) {
                        if (row[j] == 0.0f) { continue; }
                        xir::Value *term = (row[j] == 1.0f)
                                               ? ks[j]
                                               : builder.call(xt, xir::ArithmeticOp::BINARY_MUL,
                                                              {ks[j], splat_scalar_for(*this, xt, row[j])});
                        sum = sum ? builder.call(xt, xir::ArithmeticOp::BINARY_ADD, {sum, term}) : term;
                    }
                    return sum ? sum : zero_of(xt);
                };

                struct SplineEvalOut {
                    xir::Value *val{nullptr};
                    xir::Value *dval_du{nullptr};
                    xir::Value *dk_dx{nullptr};
                    xir::Value *dk_dy{nullptr};
                };

                auto eval_spline_at = [&](xir::Value *x_val, bool need_du, bool need_dk) -> SplineEvalOut {
                    xir::Value *xc = builder.call(f_t, xir::ArithmeticOp::CLAMP,
                                                  {x_val, const_float(0.0f), const_float(1.0f)});
                    xir::Value *u = xc;
                    xir::Value *base_idx = const_int(0);
                    if (nsegs > 1) {
                        xir::Value *scaled = builder.call(f_t, xir::ArithmeticOp::BINARY_MUL,
                                                          {xc, const_float(static_cast<float>(nsegs))});
                        xir::Value *fl = builder.call(f_t, xir::ArithmeticOp::FLOOR, {scaled});
                        xir::Value *seg = builder.call(i_t, xir::ArithmeticOp::CLAMP,
                                                       {to_int(fl), const_int(0), const_int(nsegs - 1)});
                        u = builder.call(f_t, xir::ArithmeticOp::BINARY_SUB, {scaled, to_float(seg)});
                        base_idx = (step == 1) ? seg
                                               : builder.call(i_t, xir::ArithmeticOp::BINARY_MUL, {seg, const_int(step)});
                    }
                    DualValue k[4];
                    for (int j = 0; j < 4; ++j) {
                        xir::Value *idx = (j == 0) ? base_idx
                                                   : builder.call(i_t, xir::ArithmeticOp::BINARY_ADD, {base_idx, const_int(j)});
                        k[j] = load_knot(idx, need_dk);
                    }
                    xir::Value *u_t = (xt == f_t) ? u : to_float3(u);
                    xir::Value *a = linear_comb(mat[0], k[0].val, k[1].val, k[2].val, k[3].val);
                    xir::Value *b = linear_comb(mat[1], k[0].val, k[1].val, k[2].val, k[3].val);
                    xir::Value *c = linear_comb(mat[2], k[0].val, k[1].val, k[2].val, k[3].val);
                    xir::Value *d = linear_comb(mat[3], k[0].val, k[1].val, k[2].val, k[3].val);
                    auto horner = [&](xir::Value *c3, xir::Value *c2, xir::Value *c1, xir::Value *c0) {
                        xir::Value *v = builder.call(xt, xir::ArithmeticOp::BINARY_MUL, {c3, u_t});
                        v = builder.call(xt, xir::ArithmeticOp::BINARY_ADD, {v, c2});
                        v = builder.call(xt, xir::ArithmeticOp::BINARY_MUL, {v, u_t});
                        v = builder.call(xt, xir::ArithmeticOp::BINARY_ADD, {v, c1});
                        v = builder.call(xt, xir::ArithmeticOp::BINARY_MUL, {v, u_t});
                        return builder.call(xt, xir::ArithmeticOp::BINARY_ADD, {v, c0});
                    };
                    SplineEvalOut sout;
                    sout.val = horner(a, b, c, d);
                    if (need_du) {
                        xir::Value *a3 = builder.call(xt, xir::ArithmeticOp::BINARY_MUL, {a, splat_scalar_for(*this, xt, 3.0f)});
                        xir::Value *b2 = builder.call(xt, xir::ArithmeticOp::BINARY_MUL, {b, splat_scalar_for(*this, xt, 2.0f)});
                        xir::Value *dv = builder.call(xt, xir::ArithmeticOp::BINARY_MUL, {a3, u_t});
                        dv = builder.call(xt, xir::ArithmeticOp::BINARY_ADD, {dv, b2});
                        dv = builder.call(xt, xir::ArithmeticOp::BINARY_MUL, {dv, u_t});
                        sout.dval_du = builder.call(xt, xir::ArithmeticOp::BINARY_ADD, {dv, c});
                    }
                    if (need_dk) {
                        auto knot_d = [&](xir::Value *dk0, xir::Value *dk1, xir::Value *dk2, xir::Value *dk3) -> xir::Value * {
                            if (!dk0 && !dk1 && !dk2 && !dk3) { return nullptr; }
                            xir::Value *z = zero_of(xt);
                            xir::Value *p0 = dk0 ? dk0 : z;
                            xir::Value *p1 = dk1 ? dk1 : z;
                            xir::Value *p2 = dk2 ? dk2 : z;
                            xir::Value *p3 = dk3 ? dk3 : z;
                            return horner(linear_comb(mat[0], p0, p1, p2, p3),
                                          linear_comb(mat[1], p0, p1, p2, p3),
                                          linear_comb(mat[2], p0, p1, p2, p3),
                                          linear_comb(mat[3], p0, p1, p2, p3));
                        };
                        sout.dk_dx = knot_d(k[0].dx, k[1].dx, k[2].dx, k[3].dx);
                        sout.dk_dy = knot_d(k[0].dy, k[1].dy, k[2].dy, k[3].dy);
                    }
                    return sout;
                };

                if (!is_inv) {
                    bool need_du = wd && x_or_v.has_derivs();
                    SplineEvalOut sout = eval_spline_at(x_or_v.val, need_du, wd);
                    DualValue res{.val = sout.val};
                    if (wd) {
                        xir::Value *gt0 = builder.call(b_t, xir::ArithmeticOp::BINARY_GREATER, {x_or_v.val, const_float(0.0f)});
                        xir::Value *lt1 = builder.call(b_t, xir::ArithmeticOp::BINARY_LESS, {x_or_v.val, const_float(1.0f)});
                        xir::Value *in01 = builder.call(b_t, xir::ArithmeticOp::BINARY_BIT_AND, {gt0, lt1});
                        xir::Value *nseg_f = const_float(static_cast<float>(nsegs));
                        auto combine_d = [&](xir::Value *dx_x, xir::Value *dk_x) -> xir::Value * {
                            xir::Value *term_x = nullptr;
                            if (dx_x && sout.dval_du) {
                                xir::Value *dx_masked = builder.call(f_t, xir::ArithmeticOp::SELECT, {const_float(0.0f), dx_x, in01});
                                xir::Value *du = builder.call(f_t, xir::ArithmeticOp::BINARY_MUL, {dx_masked, nseg_f});
                                xir::Value *du_t = (xt == f_t) ? du : to_float3(du);
                                term_x = builder.call(xt, xir::ArithmeticOp::BINARY_MUL, {sout.dval_du, du_t});
                            }
                            if (term_x && dk_x) { return builder.call(xt, xir::ArithmeticOp::BINARY_ADD, {term_x, dk_x}); }
                            return term_x ? term_x : dk_x;
                        };
                        res.dx = combine_d(x_or_v.dx, sout.dk_dx);
                        res.dy = combine_d(x_or_v.dy, sout.dk_dy);
                    }
                    store_symbol(active_layer, dst, convert_dual(res, eval_osl, dst->type(), wd));
                    return;
                }

                // splineinverse(basis, v, knots...) via 24-step XIR loop bisection
                xir::Value *v_target = x_or_v.val;
                xir::Value *s0 = eval_spline_at(const_float(0.0f), false, false).val;
                xir::Value *s1 = eval_spline_at(const_float(1.0f), false, false).val;
                xir::Value *ascending = builder.call(b_t, xir::ArithmeticOp::BINARY_GREATER_EQUAL, {s1, s0});

                xir::Value *le_s0 = builder.call(b_t, xir::ArithmeticOp::BINARY_LESS_EQUAL, {v_target, s0});
                xir::Value *ge_s0 = builder.call(b_t, xir::ArithmeticOp::BINARY_GREATER_EQUAL, {v_target, s0});
                xir::Value *clamp_lo = builder.call(b_t, xir::ArithmeticOp::SELECT, {ge_s0, le_s0, ascending});

                xir::Value *ge_s1 = builder.call(b_t, xir::ArithmeticOp::BINARY_GREATER_EQUAL, {v_target, s1});
                xir::Value *le_s1 = builder.call(b_t, xir::ArithmeticOp::BINARY_LESS_EQUAL, {v_target, s1});
                xir::Value *clamp_hi = builder.call(b_t, xir::ArithmeticOp::SELECT, {le_s1, ge_s1, ascending});
                xir::Value *in_range = builder.call(
                    b_t, xir::ArithmeticOp::BINARY_EQUAL,
                    {builder.call(b_t, xir::ArithmeticOp::BINARY_BIT_OR, {clamp_lo, clamp_hi}), const_bool(false)});

                xir::Value *lo_ptr = builder.alloca_local(f_t);
                xir::Value *hi_ptr = builder.alloca_local(f_t);
                xir::Value *it_ptr = builder.alloca_local(i_t);
                builder.store(lo_ptr, const_float(0.0f));
                builder.store(hi_ptr, const_float(1.0f));
                builder.store(it_ptr, const_int(0));

                xir::BasicBlock *cond_bb = function->create_basic_block();
                xir::BasicBlock *body_bb = function->create_basic_block();
                xir::BasicBlock *after_bb = function->create_basic_block();
                builder.cond_br(in_range, cond_bb, after_bb);

                builder.set_insertion_point(cond_bb);
                xir::Value *cur_it = builder.load(i_t, it_ptr);
                xir::Value *more = builder.call(b_t, xir::ArithmeticOp::BINARY_LESS, {cur_it, const_int(24)});
                builder.cond_br(more, body_bb, after_bb);

                builder.set_insertion_point(body_bb);
                xir::Value *lo_v = builder.load(f_t, lo_ptr);
                xir::Value *hi_v = builder.load(f_t, hi_ptr);
                xir::Value *mid = builder.call(f_t, xir::ArithmeticOp::BINARY_MUL,
                                               {builder.call(f_t, xir::ArithmeticOp::BINARY_ADD, {lo_v, hi_v}),
                                                const_float(0.5f)});
                xir::Value *s_mid = eval_spline_at(mid, false, false).val;
                xir::Value *lt_v = builder.call(b_t, xir::ArithmeticOp::BINARY_LESS, {s_mid, v_target});
                xir::Value *gt_v = builder.call(b_t, xir::ArithmeticOp::BINARY_GREATER, {s_mid, v_target});
                xir::Value *go_right = builder.call(b_t, xir::ArithmeticOp::SELECT, {gt_v, lt_v, ascending});
                builder.store(lo_ptr, builder.call(f_t, xir::ArithmeticOp::SELECT, {lo_v, mid, go_right}));
                builder.store(hi_ptr, builder.call(f_t, xir::ArithmeticOp::SELECT, {mid, hi_v, go_right}));
                builder.store(it_ptr, builder.call(i_t, xir::ArithmeticOp::BINARY_ADD, {cur_it, const_int(1)}));
                builder.br(cond_bb);

                builder.set_insertion_point(after_bb);
                xir::Value *mid_res = builder.call(
                    f_t, xir::ArithmeticOp::BINARY_MUL,
                    {builder.call(f_t, xir::ArithmeticOp::BINARY_ADD,
                                  {builder.load(f_t, lo_ptr), builder.load(f_t, hi_ptr)}),
                     const_float(0.5f)});
                xir::Value *oob_res = builder.call(f_t, xir::ArithmeticOp::SELECT, {const_float(1.0f), const_float(0.0f), clamp_lo});
                DualValue inv_out;
                inv_out.val = builder.call(f_t, xir::ArithmeticOp::SELECT, {oob_res, mid_res, in_range});
                if (wd) {
                    SplineEvalOut s_deriv = eval_spline_at(inv_out.val, true, true);
                    xir::Value *dS_dx = builder.call(f_t, xir::ArithmeticOp::BINARY_MUL,
                                                     {s_deriv.dval_du, const_float(static_cast<float>(nsegs))});
                    auto inv_d = [&](xir::Value *dv_d, xir::Value *dk_d) -> xir::Value * {
                        if (!dv_d && !dk_d) { return nullptr; }
                        xir::Value *rhs = dv_d ? dv_d : const_float(0.0f);
                        xir::Value *num = dk_d ? builder.call(f_t, xir::ArithmeticOp::BINARY_SUB, {rhs, dk_d}) : rhs;
                        xir::Value *raw = safe_div_val(*this, num, dS_dx);
                        return builder.call(f_t, xir::ArithmeticOp::SELECT, {const_float(0.0f), raw, in_range});
                    };
                    inv_out.dx = inv_d(x_or_v.dx, s_deriv.dk_dx);
                    inv_out.dy = inv_d(x_or_v.dy, s_deriv.dk_dy);
                }
                store_symbol(active_layer, dst, convert_dual(inv_out, osl_float(), dst->type(), wd));
            });
        return errors.empty();
    }

    // 26. Closure constructor
    if (op == "closure" && args.size() >= 2u) {
        const Symbol *dst = args[0];
        const Symbol *name_sym = args[1];
        if (dst->type()->tag() != Type::Tag::CLOSURE ||
            !detail::is_simple_type(name_sym->type(), SimpleType::Primitive::STRING)) {
            return false;
        }

        luisa::vector<const Type *> arg_types;
        arg_types.reserve(args.size() - 2u);
        for (size_t k = 2u; k < args.size(); ++k) {
            arg_types.emplace_back(args[k]->type());
        }

        StringSlotSpec name_slot = make_string_slot(*this, active_layer, ia, 1u, name_sym);
        dispatch_strings_1(
            *this, name_slot,
            [&](luisa::span<const luisa::string_view> name_resolved) {
                luisa::string_view cl_name = name_resolved[0];
                size_t pos_count = std::min<size_t>(
                    arg_types.size(), owner->closure_positional_arg_count(cl_name, arg_types));
                size_t opt_start = 2u + pos_count;

                luisa::vector<StringSlotSpec> opt_slots;
                struct PendingHookArg {
                    size_t arg_idx{0u};
                    size_t str_slot{~0u};
                    DualValue val{};
                    ClosureValue cl{};
                    luisa::optional<ConstantOptionValue> cval{};
                };
                luisa::vector<PendingHookArg> hook_args;
                for (size_t k = 2u; k < opt_start; ++k) {
                    const Symbol *as = args[k];
                    PendingHookArg ha;
                    ha.arg_idx = k;
                    if (as->type()->tag() == Type::Tag::CLOSURE) {
                        ha.cl = load_closure(active_layer, as);
                    } else if (detail::is_simple_type(as->type(), SimpleType::Primitive::STRING)) {
                        ha.str_slot = opt_slots.size();
                        opt_slots.emplace_back(make_string_slot(*this, active_layer, ia, k, as));
                        ha.val = {.val = get_arg_string_token(*this, active_layer, ia, k, as)};
                    } else if (!as->is_array() && as->type()->tag() != Type::Tag::STRUCT) {
                        ha.val = load_symbol(active_layer, as, false);
                        ha.cval = extract_arg_constant(*this, ia, k, as);
                    } else if (as->is_array()) {
                        if (const auto *st = find_storage(active_layer, as)) {
                            ha.val = {.val = st->val_ptr};
                        }
                    }
                    hook_args.emplace_back(std::move(ha));
                }

                luisa::vector<PendingStringOptEntry> pending_opts;
                collect_named_options(*this, active_layer, ia, inst, opt_start, args.size(),
                                      false, opt_slots, pending_opts);
                if (!errors.empty()) { return; }

                dispatch_dynamic_strings(
                    opt_slots,
                    [&](luisa::span<const luisa::string_view> resolved) {
                        luisa::vector<HookArg> mat_args;
                        mat_args.reserve(hook_args.size());
                        for (const auto &ha : hook_args) {
                            const Symbol *as = args[ha.arg_idx];
                            HookArg out_a;
                            out_a.osl_type = as->type();
                            out_a.array_length = effective_array_length(active_layer, as);
                            out_a.value = ha.val;
                            out_a.closure = ha.cl;
                            if (ha.str_slot != ~0u) {
                                out_a.constant = ConstantOptionValue{resolved[ha.str_slot]};
                            } else {
                                out_a.constant = ha.cval;
                            }
                            mat_args.emplace_back(out_a);
                        }
                        auto opts = materialize_named_options(inst, *this, active_layer, pending_opts, resolved);
                        ClosureValue cv = owner->emit_closure(cl_name, mat_args, opts);
                        if (errors.empty() && cv.valid()) {
                            store_closure(active_layer, dst, cv);
                        }
                    });
            });
        return errors.empty();
    }

    return false;
}

}// namespace luisa::compute::osl
