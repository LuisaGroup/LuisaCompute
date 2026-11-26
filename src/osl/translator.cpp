#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <utility>

#include <luisa/ast/type_registry.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/queue.h>
#include <luisa/xir/passes/algebraic_simplify.h>
#include <luisa/xir/passes/const_fold.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/lower_irreducible_cfg.h>
#include <luisa/xir/passes/mem2reg.h>
#include <luisa/xir/passes/reg2mem.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/passes/sccp.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <luisa/xir/verifier.h>
#include "translator_impl.h"

namespace luisa::compute::osl {

namespace detail {

bool is_simple_type(const Type *t, SimpleType::Primitive p) noexcept {
    return t != nullptr && t->tag() == Type::Tag::SIMPLE &&
           static_cast<const SimpleType *>(t)->primitive() == p;
}

bool is_triple_type(const Type *t) noexcept {
    if (t == nullptr || t->tag() != Type::Tag::SIMPLE) { return false; }
    auto p = static_cast<const SimpleType *>(t)->primitive();
    return p == SimpleType::Primitive::POINT ||
           p == SimpleType::Primitive::NORMAL ||
           p == SimpleType::Primitive::VECTOR ||
           p == SimpleType::Primitive::COLOR;
}

bool is_differentiable_type(const Type *t) noexcept {
    if (t == nullptr) { return false; }
    if (t->tag() == Type::Tag::SIMPLE) {
        auto p = static_cast<const SimpleType *>(t)->primitive();
        return p == SimpleType::Primitive::FLOAT ||
               p == SimpleType::Primitive::POINT ||
               p == SimpleType::Primitive::NORMAL ||
               p == SimpleType::Primitive::VECTOR ||
               p == SimpleType::Primitive::COLOR;
    }
    if (t->tag() == Type::Tag::STRUCT) {
        for (const auto &f : static_cast<const StructType *>(t)->fields()) {
            if (is_differentiable_type(f.type)) { return true; }
        }
    }
    return false;
}

size_t literal_stride(const Type *t) noexcept {
    if (t == nullptr || t->tag() != Type::Tag::SIMPLE) { return 0u; }
    switch (static_cast<const SimpleType *>(t)->primitive()) {
        case SimpleType::Primitive::INT:
        case SimpleType::Primitive::FLOAT:
        case SimpleType::Primitive::STRING:
            return 1u;
        case SimpleType::Primitive::POINT:
        case SimpleType::Primitive::NORMAL:
        case SimpleType::Primitive::VECTOR:
        case SimpleType::Primitive::COLOR:
            return 3u;
        case SimpleType::Primitive::MATRIX:
            return 16u;
    }
    return 0u;
}

const compute::Type *osl_type_to_xir_type(
    const Type *t, int array_len, const compute::Type *closure_type) noexcept {
    if (t == nullptr) { return nullptr; }
    const compute::Type *elem = nullptr;
    switch (t->tag()) {
        case Type::Tag::SIMPLE: {
            switch (static_cast<const SimpleType *>(t)->primitive()) {
                case SimpleType::Primitive::INT: elem = compute::Type::of<int32_t>(); break;
                case SimpleType::Primitive::FLOAT: elem = compute::Type::of<float>(); break;
                case SimpleType::Primitive::POINT:
                case SimpleType::Primitive::NORMAL:
                case SimpleType::Primitive::VECTOR:
                case SimpleType::Primitive::COLOR: elem = compute::Type::of<luisa::float3>(); break;
                case SimpleType::Primitive::MATRIX: elem = compute::Type::of<luisa::float4x4>(); break;
                case SimpleType::Primitive::STRING: elem = compute::Type::of<uint32_t>(); break;
            }
            break;
        }
        case Type::Tag::CLOSURE:
            elem = closure_type;
            break;
        case Type::Tag::STRUCT:
            return nullptr;
    }
    if (elem == nullptr) { return nullptr; }
    if (array_len > 0) {
        return compute::Type::array(elem, static_cast<size_t>(array_len));
    }
    return elem;
}

luisa::string_view unquote_hint_arg(luisa::string_view s) noexcept {
    if (s.size() >= 2u && s.front() == '"' && s.back() == '"') {
        return s.substr(1u, s.size() - 2u);
    }
    return s;
}

luisa::optional<luisa::string_view> parse_argrw_hint(const Instruction &inst) noexcept {
    for (const auto &h : inst.hints()) {
        if (h.identifier() == "argrw" && !h.args().empty()) {
            return unquote_hint_arg(h.args()[0]);
        }
    }
    return luisa::nullopt;
}

bool instruction_has_argderivs(const Instruction &inst, size_t arg_index_1based) noexcept {
    for (const auto &h : inst.hints()) {
        if (h.identifier() == "argderivs") {
            for (const auto &a : h.args()) {
                char *end = nullptr;
                long v = std::strtol(a.c_str(), &end, 10);
                if (end != a.c_str() && static_cast<size_t>(v) == arg_index_1based) {
                    return true;
                }
            }
        }
    }
    return false;
}

luisa::optional<GlobalSymbol> parse_global_symbol(luisa::string_view name) noexcept {
    if (name == "P") { return GlobalSymbol::P; }
    if (name == "I") { return GlobalSymbol::I; }
    if (name == "N") { return GlobalSymbol::N; }
    if (name == "Ng") { return GlobalSymbol::Ng; }
    if (name == "u") { return GlobalSymbol::u; }
    if (name == "v") { return GlobalSymbol::v; }
    if (name == "dPdu") { return GlobalSymbol::dPdu; }
    if (name == "dPdv") { return GlobalSymbol::dPdv; }
    if (name == "Ps") { return GlobalSymbol::Ps; }
    if (name == "time") { return GlobalSymbol::time; }
    if (name == "dtime") { return GlobalSymbol::dtime; }
    if (name == "dPdtime") { return GlobalSymbol::dPdtime; }
    if (name == "Ci") { return GlobalSymbol::Ci; }
    return luisa::nullopt;
}

bool is_global_writable(GlobalSymbol g) noexcept {
    return g == GlobalSymbol::P || g == GlobalSymbol::N || g == GlobalSymbol::Ci;
}

bool is_global_differentiable(GlobalSymbol g) noexcept {
    return g != GlobalSymbol::dtime && g != GlobalSymbol::Ci;
}

luisa::float4x4 osl_row_major_to_xir_matrix(const float *vals) noexcept {
    // Store L = M_osl^T so Luisa column r holds OSL row r
    return luisa::make_float4x4(
        vals[0], vals[1], vals[2], vals[3],
        vals[4], vals[5], vals[6], vals[7],
        vals[8], vals[9], vals[10], vals[11],
        vals[12], vals[13], vals[14], vals[15]);
}

}// namespace detail

// --- OSLTranslator::Impl helpers ---

xir::Value *OSLTranslator::Impl::const_int(int32_t v) noexcept {
    return module->create_constant(compute::Type::of<int32_t>(), &v);
}

xir::Value *OSLTranslator::Impl::const_uint(uint32_t v) noexcept {
    return module->create_constant(compute::Type::of<uint32_t>(), &v);
}

xir::Value *OSLTranslator::Impl::const_float(float v) noexcept {
    return module->create_constant(compute::Type::of<float>(), &v);
}

xir::Value *OSLTranslator::Impl::const_bool(bool v) noexcept {
    return module->create_constant(compute::Type::of<bool>(), &v);
}

xir::Value *OSLTranslator::Impl::const_float3(luisa::float3 v) noexcept {
    return module->create_constant(compute::Type::of<luisa::float3>(), &v);
}

xir::Value *OSLTranslator::Impl::const_float4(luisa::float4 v) noexcept {
    return module->create_constant(compute::Type::of<luisa::float4>(), &v);
}

xir::Value *OSLTranslator::Impl::const_float4x4(luisa::float4x4 v) noexcept {
    return module->create_constant(compute::Type::of<luisa::float4x4>(), &v);
}

xir::Value *OSLTranslator::Impl::zero_of(const compute::Type *t) noexcept {
    if (t == nullptr) { return nullptr; }
    if (t->is_structure()) {
        luisa::vector<xir::Value *> fields;
        fields.reserve(t->members().size());
        for (const compute::Type *m : t->members()) {
            fields.emplace_back(zero_of(m));
        }
        return builder.call(t, xir::ArithmeticOp::AGGREGATE, fields);
    }
    if (t->is_array()) {
        xir::Value *elem_zero = zero_of(t->element());
        luisa::vector<xir::Value *> elems(t->dimension(), elem_zero);
        return builder.call(t, xir::ArithmeticOp::AGGREGATE, elems);
    }
    return module->create_constant_zero(t);
}

void OSLTranslator::Impl::set_location(const detail::LinkedLayer &layer, uint32_t pc, const Instruction &inst) noexcept {
    current_layer_name = layer.layer->name;
    current_pc = pc;
    current_opcode = luisa::string{inst.opcode()};
    current_filename.clear();
    current_line = -1;
    for (const auto &h : inst.hints()) {
        if (h.identifier() == "filename" && !h.args().empty()) {
            current_filename = luisa::string{detail::unquote_hint_arg(h.args()[0])};
        } else if (h.identifier() == "line" && !h.args().empty()) {
            current_line = static_cast<int>(std::strtol(h.args()[0].c_str(), nullptr, 10));
        }
    }
}

void OSLTranslator::Impl::clear_location() noexcept {
    current_pc = ~0u;
    current_opcode.clear();
    current_filename.clear();
    current_line = -1;
}

void OSLTranslator::Impl::report_error(luisa::string msg) noexcept {
    errors.emplace_back(TranslationError{
        .layer_name = current_layer_name,
        .instruction_index = current_pc,
        .opcode = current_opcode,
        .filename = current_filename,
        .line = current_line,
        .message = std::move(msg)});
}

const OSLTranslator::Impl::SymbolStorage *OSLTranslator::Impl::find_storage(
    uint32_t active_layer, const Symbol *sym) const noexcept {
    const auto &st = layer_runtimes[active_layer].storage;
    auto it = st.find(sym);
    return it == st.end() ? nullptr : &it->second;
}

int OSLTranslator::Impl::effective_array_length(uint32_t active_layer, const Symbol *sym) const noexcept {
    const auto &ll = linked_group.layers[active_layer];
    auto it = ll.effective_array_length.find(sym);
    return it == ll.effective_array_length.end() ? 0 : it->second;
}

xir::Value *OSLTranslator::Impl::extract_vec_comp(xir::Value *vec3, uint32_t idx) noexcept {
    return builder.call(compute::Type::of<float>(), xir::ArithmeticOp::EXTRACT, {vec3, const_uint(idx)});
}

xir::Value *OSLTranslator::Impl::insert_vec_comp(xir::Value *vec3, xir::Value *elem, uint32_t idx) noexcept {
    return builder.call(compute::Type::of<luisa::float3>(), xir::ArithmeticOp::INSERT, {vec3, elem, const_uint(idx)});
}

xir::Value *OSLTranslator::Impl::to_bool(xir::Value *v) noexcept {
    if (v == nullptr) { return const_bool(false); }
    auto *t = v->type();
    if (t == compute::Type::of<bool>()) { return v; }
    if (t == compute::Type::of<int32_t>()) {
        return builder.call(compute::Type::of<bool>(), xir::ArithmeticOp::BINARY_NOT_EQUAL, {v, const_int(0)});
    }
    if (t == compute::Type::of<uint32_t>()) {
        return builder.call(compute::Type::of<bool>(), xir::ArithmeticOp::BINARY_NOT_EQUAL, {v, const_uint(0u)});
    }
    if (t == compute::Type::of<float>()) {
        return builder.call(compute::Type::of<bool>(), xir::ArithmeticOp::BINARY_NOT_EQUAL, {v, const_float(0.0f)});
    }
    if (t == compute::Type::of<luisa::float3>()) {
        auto *nz = builder.call(compute::Type::of<luisa::bool3>(), xir::ArithmeticOp::BINARY_NOT_EQUAL, {v, const_float3(luisa::float3{0.0f})});
        return builder.call(compute::Type::of<bool>(), xir::ArithmeticOp::ANY, {nz});
    }
    report_error(luisa::format("Cannot convert type '{}' to bool.", t->description()));
    return const_bool(false);
}

xir::Value *OSLTranslator::Impl::to_int(xir::Value *v) noexcept {
    if (v == nullptr) { return const_int(0); }
    if (v->type() == compute::Type::of<int32_t>()) { return v; }
    if (v->type() == compute::Type::of<bool>()) {
        return builder.call(compute::Type::of<int32_t>(), xir::ArithmeticOp::SELECT, {const_int(0), const_int(1), v});
    }
    if (v->type() == compute::Type::of<uint32_t>() || v->type() == compute::Type::of<float>()) {
        return builder.static_cast_(compute::Type::of<int32_t>(), v);
    }
    report_error(luisa::format("Cannot convert type '{}' to int.", v->type()->description()));
    return const_int(0);
}

xir::Value *OSLTranslator::Impl::to_float(xir::Value *v) noexcept {
    if (v == nullptr) { return const_float(0.0f); }
    if (v->type() == compute::Type::of<float>()) { return v; }
    if (v->type() == compute::Type::of<bool>()) {
        return builder.call(compute::Type::of<float>(), xir::ArithmeticOp::SELECT, {const_float(0.0f), const_float(1.0f), v});
    }
    if (v->type() == compute::Type::of<int32_t>() || v->type() == compute::Type::of<uint32_t>()) {
        return builder.static_cast_(compute::Type::of<float>(), v);
    }
    report_error(luisa::format("Cannot convert type '{}' to float.", v->type()->description()));
    return const_float(0.0f);
}

xir::Value *OSLTranslator::Impl::to_float3(xir::Value *v) noexcept {
    if (v == nullptr) { return const_float3(luisa::float3{0.0f}); }
    if (v->type() == compute::Type::of<luisa::float3>()) { return v; }
    xir::Value *f = to_float(v);
    return builder.call(compute::Type::of<luisa::float3>(), xir::ArithmeticOp::AGGREGATE, {f, f, f});
}

xir::Value *OSLTranslator::Impl::to_float4x4(xir::Value *v) noexcept {
    if (v == nullptr) { return const_float4x4(luisa::make_float4x4(0.0f)); }
    if (v->type() == compute::Type::of<luisa::float4x4>()) { return v; }
    xir::Value *f = to_float(v);
    xir::Value *z = const_float(0.0f);
    auto *f4_t = compute::Type::of<luisa::float4>();
    auto *c0 = builder.call(f4_t, xir::ArithmeticOp::AGGREGATE, {f, z, z, z});
    auto *c1 = builder.call(f4_t, xir::ArithmeticOp::AGGREGATE, {z, f, z, z});
    auto *c2 = builder.call(f4_t, xir::ArithmeticOp::AGGREGATE, {z, z, f, z});
    auto *c3 = builder.call(f4_t, xir::ArithmeticOp::AGGREGATE, {z, z, z, f});
    return builder.call(compute::Type::of<luisa::float4x4>(), xir::ArithmeticOp::AGGREGATE, {c0, c1, c2, c3});
}

DualValue OSLTranslator::Impl::convert_dual(
    const DualValue &src, const Type *src_type, const Type *dst_type, bool want_deriv) noexcept {
    if (src_type == dst_type) {
        if (!want_deriv) { return {.val = src.val}; }
        return src;
    }
    if (detail::is_simple_type(dst_type, SimpleType::Primitive::INT)) {
        return {.val = to_int(src.val)};
    }
    if (detail::is_simple_type(dst_type, SimpleType::Primitive::FLOAT)) {
        xir::Value *v = to_float(src.val);
        if (!want_deriv || detail::is_simple_type(src_type, SimpleType::Primitive::INT)) {
            return {.val = v};
        }
        return {.val = v, .dx = src.dx, .dy = src.dy};
    }
    if (detail::is_triple_type(dst_type)) {
        xir::Value *v = to_float3(src.val);
        if (!want_deriv || detail::is_simple_type(src_type, SimpleType::Primitive::INT)) {
            return {.val = v};
        }
        xir::Value *dx = src.dx ? to_float3(src.dx) : nullptr;
        xir::Value *dy = src.dy ? to_float3(src.dy) : nullptr;
        return {.val = v, .dx = dx, .dy = dy};
    }
    if (detail::is_simple_type(dst_type, SimpleType::Primitive::MATRIX)) {
        return {.val = to_float4x4(src.val)};
    }
    if (detail::is_simple_type(dst_type, SimpleType::Primitive::STRING) &&
        detail::is_simple_type(src_type, SimpleType::Primitive::STRING)) {
        return {.val = src.val};
    }
    report_error(luisa::format("Unsupported type conversion from '{}' to '{}'.",
                               src_type ? src_type->identifier() : "null",
                               dst_type ? dst_type->identifier() : "null"));
    return {.val = zero_of(detail::osl_type_to_xir_type(dst_type, 0, owner->closure_xir_type()))};
}

xir::Value *OSLTranslator::Impl::clamp_index(xir::Value *idx, int len) noexcept {
    if (len <= 0) {
        report_error("Indexing into empty or zero-length array.");
        return const_int(0);
    }
    xir::Value *i = to_int(idx);
    if (options.bounds_check == BoundsCheckMode::CLAMP) {
        return builder.call(compute::Type::of<int32_t>(), xir::ArithmeticOp::CLAMP,
                            {i, const_int(0), const_int(len - 1)});
    }
    // GUARD mode: clamp + assume or clamp to 0
    return builder.call(compute::Type::of<int32_t>(), xir::ArithmeticOp::CLAMP,
                        {i, const_int(0), const_int(len - 1)});
}

DualValue OSLTranslator::Impl::make_constant_dual(const Symbol *sym) noexcept {
    const auto &vals = sym->initial_values();
    if (detail::is_simple_type(sym->type(), SimpleType::Primitive::INT)) {
        int32_t v = vals.empty() ? 0 : static_cast<int32_t>(vals[0].as_int());
        return {.val = const_int(v)};
    }
    if (detail::is_simple_type(sym->type(), SimpleType::Primitive::FLOAT)) {
        float v = vals.empty() ? 0.0f : vals[0].as_float();
        return {.val = const_float(v)};
    }
    if (detail::is_triple_type(sym->type())) {
        luisa::float3 v{0.0f};
        if (vals.size() >= 3u) {
            v = luisa::float3{vals[0].as_float(), vals[1].as_float(), vals[2].as_float()};
        } else if (vals.size() == 1u) {
            v = luisa::float3{vals[0].as_float()};
        }
        return {.val = const_float3(v)};
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
        return {.val = const_float4x4(m)};
    }
    if (detail::is_simple_type(sym->type(), SimpleType::Primitive::STRING)) {
        StringToken tok = vals.empty() ? empty_string_token : string_table.intern(vals[0].as_string());
        return {.val = const_uint(tok)};
    }
    report_error(luisa::format("Unsupported constant symbol type '{}'.", sym->type()->identifier()));
    return {};
}

DualValue OSLTranslator::Impl::load_symbol(uint32_t active_layer, const Symbol *sym, bool want_deriv) noexcept {
    if (sym->tag() == Symbol::Tag::SYM_CONST && !sym->is_array()) {
        return make_constant_dual(sym);
    }
    if (sym->tag() == Symbol::Tag::SYM_GLOBAL) {
        if (auto g = detail::parse_global_symbol(sym->identifier())) {
            if (*g == GlobalSymbol::Ci) {
                report_error("Cannot read closure global 'Ci' as numeric DualValue.");
                return {};
            }
            return owner->emit_global_read(*g, want_deriv);
        }
        report_error(luisa::format("Unknown OSL global symbol '{}'.", sym->identifier()));
        return {};
    }
    const auto *st = find_storage(active_layer, sym);
    if (st == nullptr || st->val_ptr == nullptr) {
        report_error(luisa::format("Unallocated symbol '{}'.", sym->identifier()));
        return {};
    }
    const compute::Type *xt = st->val_ptr->type();
    DualValue out;
    out.val = builder.load(xt, st->val_ptr);
    if (want_deriv && st->is_dual) {
        out.dx = st->dx_ptr ? builder.load(xt, st->dx_ptr) : nullptr;
        out.dy = st->dy_ptr ? builder.load(xt, st->dy_ptr) : nullptr;
    }
    return out;
}

void OSLTranslator::Impl::store_symbol(uint32_t active_layer, const Symbol *sym, const DualValue &val) noexcept {
    if (sym->tag() == Symbol::Tag::SYM_GLOBAL) {
        if (auto g = detail::parse_global_symbol(sym->identifier())) {
            if (!detail::is_global_writable(*g)) {
                report_error(luisa::format("Global symbol '{}' is read-only.", sym->identifier()));
                return;
            }
            owner->emit_global_write(*g, val);
            return;
        }
        report_error(luisa::format("Unknown global symbol '{}'.", sym->identifier()));
        return;
    }
    const auto *st = find_storage(active_layer, sym);
    if (st == nullptr || st->val_ptr == nullptr) {
        report_error(luisa::format("Unallocated destination symbol '{}'.", sym->identifier()));
        return;
    }
    builder.store(st->val_ptr, val.val);
    if (st->is_dual) {
        const compute::Type *xt = st->val_ptr->type();
        if (st->dx_ptr) { builder.store(st->dx_ptr, val.dx ? val.dx : zero_of(xt)); }
        if (st->dy_ptr) { builder.store(st->dy_ptr, val.dy ? val.dy : zero_of(xt)); }
    }
}

DualValue OSLTranslator::Impl::load_array_element(
    uint32_t active_layer, const Symbol *arr, xir::Value *index, bool want_deriv) noexcept {
    int len = effective_array_length(active_layer, arr);
    const auto *st = find_storage(active_layer, arr);
    const compute::Type *elem_t =
        (st != nullptr && st->val_ptr != nullptr)
            ? st->val_ptr->type()->element()
            : detail::osl_type_to_xir_type(arr->type(), 0, owner->closure_xir_type());
    if (len <= 0) {
        if (options.bounds_check != BoundsCheckMode::GUARD || elem_t == nullptr) {
            report_error("Indexing into empty or zero-length array.");
            return {};
        }
        DualValue out{.val = zero_of(elem_t)};
        if (want_deriv && detail::is_differentiable_type(arr->type())) {
            out.dx = zero_of(elem_t);
            out.dy = zero_of(elem_t);
        }
        return out;
    }
    if (st == nullptr || st->val_ptr == nullptr) {
        return {};
    }
    if (options.bounds_check == BoundsCheckMode::CLAMP) {
        xir::Value *ci = clamp_index(index, len);
        auto *vp = builder.gep(elem_t, st->val_ptr, {ci});
        DualValue out;
        out.val = builder.load(elem_t, vp);
        if (want_deriv && st->is_dual) {
            if (st->dx_ptr) { out.dx = builder.load(elem_t, builder.gep(elem_t, st->dx_ptr, {ci})); }
            if (st->dy_ptr) { out.dy = builder.load(elem_t, builder.gep(elem_t, st->dy_ptr, {ci})); }
        }
        return out;
    }

    // BoundsCheckMode::GUARD: runtime bounds-guarded read (OOB returns 0 without aliasing)
    xir::Value *iv = to_int(index);
    auto *b_t = compute::Type::of<bool>();
    xir::Value *ge0 = builder.call(b_t, xir::ArithmeticOp::BINARY_GREATER_EQUAL, {iv, const_int(0)});
    xir::Value *lt_n = builder.call(b_t, xir::ArithmeticOp::BINARY_LESS, {iv, const_int(len)});
    xir::Value *in_bounds = builder.call(b_t, xir::ArithmeticOp::BINARY_BIT_AND, {ge0, lt_n});

    bool emit_d = want_deriv && st->is_dual;
    xir::Value *res_val_ptr = builder.alloca_local(elem_t);
    xir::Value *res_dx_ptr = (emit_d && st->dx_ptr) ? builder.alloca_local(elem_t) : nullptr;
    xir::Value *res_dy_ptr = (emit_d && st->dy_ptr) ? builder.alloca_local(elem_t) : nullptr;

    xir::BasicBlock *in_bb = function->create_basic_block();
    xir::BasicBlock *oob_bb = function->create_basic_block();
    xir::BasicBlock *merge_bb = function->create_basic_block();
    builder.cond_br(in_bounds, in_bb, oob_bb);

    builder.set_insertion_point(in_bb);
    builder.store(res_val_ptr, builder.load(elem_t, builder.gep(elem_t, st->val_ptr, {iv})));
    if (res_dx_ptr) { builder.store(res_dx_ptr, builder.load(elem_t, builder.gep(elem_t, st->dx_ptr, {iv}))); }
    if (res_dy_ptr) { builder.store(res_dy_ptr, builder.load(elem_t, builder.gep(elem_t, st->dy_ptr, {iv}))); }
    builder.br(merge_bb);

    builder.set_insertion_point(oob_bb);
    builder.store(res_val_ptr, zero_of(elem_t));
    if (res_dx_ptr) { builder.store(res_dx_ptr, zero_of(elem_t)); }
    if (res_dy_ptr) { builder.store(res_dy_ptr, zero_of(elem_t)); }
    builder.br(merge_bb);

    builder.set_insertion_point(merge_bb);
    DualValue out;
    out.val = builder.load(elem_t, res_val_ptr);
    if (res_dx_ptr) { out.dx = builder.load(elem_t, res_dx_ptr); }
    if (res_dy_ptr) { out.dy = builder.load(elem_t, res_dy_ptr); }
    return out;
}

void OSLTranslator::Impl::store_array_element(
    uint32_t active_layer, const Symbol *arr, xir::Value *index, const DualValue &val) noexcept {
    int len = effective_array_length(active_layer, arr);
    const auto *st = find_storage(active_layer, arr);
    if (len <= 0) {
        if (options.bounds_check != BoundsCheckMode::GUARD) {
            report_error("Indexing into empty or zero-length array.");
        }
        return;
    }
    if (st == nullptr || st->val_ptr == nullptr) {
        return;
    }
    const compute::Type *elem_t = st->val_ptr->type()->element();
    if (options.bounds_check == BoundsCheckMode::CLAMP) {
        xir::Value *ci = clamp_index(index, len);
        builder.store(builder.gep(elem_t, st->val_ptr, {ci}), val.val);
        if (st->is_dual) {
            if (st->dx_ptr) {
                builder.store(builder.gep(elem_t, st->dx_ptr, {ci}), val.dx ? val.dx : zero_of(elem_t));
            }
            if (st->dy_ptr) {
                builder.store(builder.gep(elem_t, st->dy_ptr, {ci}), val.dy ? val.dy : zero_of(elem_t));
            }
        }
        return;
    }

    // BoundsCheckMode::GUARD: runtime bounds-guarded write (OOB is a no-op)
    xir::Value *iv = to_int(index);
    auto *b_t = compute::Type::of<bool>();
    xir::Value *ge0 = builder.call(b_t, xir::ArithmeticOp::BINARY_GREATER_EQUAL, {iv, const_int(0)});
    xir::Value *lt_n = builder.call(b_t, xir::ArithmeticOp::BINARY_LESS, {iv, const_int(len)});
    xir::Value *in_bounds = builder.call(b_t, xir::ArithmeticOp::BINARY_BIT_AND, {ge0, lt_n});

    xir::BasicBlock *in_bb = function->create_basic_block();
    xir::BasicBlock *merge_bb = function->create_basic_block();
    builder.cond_br(in_bounds, in_bb, merge_bb);

    builder.set_insertion_point(in_bb);
    builder.store(builder.gep(elem_t, st->val_ptr, {iv}), val.val);
    if (st->is_dual) {
        if (st->dx_ptr) {
            builder.store(builder.gep(elem_t, st->dx_ptr, {iv}), val.dx ? val.dx : zero_of(elem_t));
        }
        if (st->dy_ptr) {
            builder.store(builder.gep(elem_t, st->dy_ptr, {iv}), val.dy ? val.dy : zero_of(elem_t));
        }
    }
    builder.br(merge_bb);
    builder.set_insertion_point(merge_bb);
}

ClosureValue OSLTranslator::Impl::load_closure_array_element(
    uint32_t active_layer, const Symbol *arr, xir::Value *index) noexcept {
    int len = effective_array_length(active_layer, arr);
    const auto *st = find_storage(active_layer, arr);
    const compute::Type *closure_t = owner->closure_xir_type();
    if (len <= 0) {
        if (options.bounds_check != BoundsCheckMode::GUARD) {
            report_error("Indexing into empty or zero-length closure array.");
        }
        return owner->emit_closure_zero();
    }
    if (st == nullptr || st->val_ptr == nullptr) {
        return owner->emit_closure_zero();
    }
    if (options.bounds_check == BoundsCheckMode::CLAMP) {
        xir::Value *ci = clamp_index(index, len);
        return {.value = builder.load(closure_t, builder.gep(closure_t, st->val_ptr, {ci}))};
    }
    xir::Value *iv = to_int(index);
    auto *b_t = compute::Type::of<bool>();
    xir::Value *ge0 = builder.call(b_t, xir::ArithmeticOp::BINARY_GREATER_EQUAL, {iv, const_int(0)});
    xir::Value *lt_n = builder.call(b_t, xir::ArithmeticOp::BINARY_LESS, {iv, const_int(len)});
    xir::Value *in_bounds = builder.call(b_t, xir::ArithmeticOp::BINARY_BIT_AND, {ge0, lt_n});
    xir::Value *res_ptr = builder.alloca_local(closure_t);
    xir::BasicBlock *in_bb = function->create_basic_block();
    xir::BasicBlock *oob_bb = function->create_basic_block();
    xir::BasicBlock *merge_bb = function->create_basic_block();
    builder.cond_br(in_bounds, in_bb, oob_bb);
    builder.set_insertion_point(in_bb);
    builder.store(res_ptr, builder.load(closure_t, builder.gep(closure_t, st->val_ptr, {iv})));
    builder.br(merge_bb);
    builder.set_insertion_point(oob_bb);
    builder.store(res_ptr, owner->emit_closure_zero().value);
    builder.br(merge_bb);
    builder.set_insertion_point(merge_bb);
    return {.value = builder.load(closure_t, res_ptr)};
}

void OSLTranslator::Impl::store_closure_array_element(
    uint32_t active_layer, const Symbol *arr, xir::Value *index, const ClosureValue &c) noexcept {
    int len = effective_array_length(active_layer, arr);
    const auto *st = find_storage(active_layer, arr);
    const compute::Type *closure_t = owner->closure_xir_type();
    if (len <= 0) {
        if (options.bounds_check != BoundsCheckMode::GUARD) {
            report_error("Indexing into empty or zero-length closure array.");
        }
        return;
    }
    if (st == nullptr || st->val_ptr == nullptr) { return; }
    if (options.bounds_check == BoundsCheckMode::CLAMP) {
        xir::Value *ci = clamp_index(index, len);
        builder.store(builder.gep(closure_t, st->val_ptr, {ci}), c.value);
        return;
    }
    xir::Value *iv = to_int(index);
    auto *b_t = compute::Type::of<bool>();
    xir::Value *ge0 = builder.call(b_t, xir::ArithmeticOp::BINARY_GREATER_EQUAL, {iv, const_int(0)});
    xir::Value *lt_n = builder.call(b_t, xir::ArithmeticOp::BINARY_LESS, {iv, const_int(len)});
    xir::Value *in_bounds = builder.call(b_t, xir::ArithmeticOp::BINARY_BIT_AND, {ge0, lt_n});
    xir::BasicBlock *in_bb = function->create_basic_block();
    xir::BasicBlock *merge_bb = function->create_basic_block();
    builder.cond_br(in_bounds, in_bb, merge_bb);
    builder.set_insertion_point(in_bb);
    builder.store(builder.gep(closure_t, st->val_ptr, {iv}), c.value);
    builder.br(merge_bb);
    builder.set_insertion_point(merge_bb);
}

ClosureValue OSLTranslator::Impl::load_closure(uint32_t active_layer, const Symbol *sym) noexcept {
    const auto *st = find_storage(active_layer, sym);
    if (st == nullptr || !st->is_closure || st->val_ptr == nullptr) {
        report_error(luisa::format("Symbol '{}' is not a closure.", sym->identifier()));
        return owner->emit_closure_zero();
    }
    return {.value = builder.load(st->val_ptr->type(), st->val_ptr)};
}

void OSLTranslator::Impl::store_closure(uint32_t active_layer, const Symbol *sym, const ClosureValue &c) noexcept {
    if (sym->tag() == Symbol::Tag::SYM_GLOBAL && sym->identifier() == "Ci") {
        owner->emit_closure_output(c);
        return;
    }
    const auto *st = find_storage(active_layer, sym);
    if (st == nullptr || !st->is_closure || st->val_ptr == nullptr) {
        report_error(luisa::format("Destination symbol '{}' is not a closure.", sym->identifier()));
        return;
    }
    builder.store(st->val_ptr, c.value);
}

xir::Value *OSLTranslator::Impl::load_string_token(uint32_t active_layer, const Symbol *sym) noexcept {
    if (sym->tag() == Symbol::Tag::SYM_CONST) {
        luisa::string_view sv = sym->initial_values().empty()
                                    ? luisa::string_view{}
                                    : sym->initial_values()[0].as_string();
        return token_for_string(sv);
    }
    const auto *st = find_storage(active_layer, sym);
    if (st == nullptr || st->val_ptr == nullptr) {
        report_error(luisa::format("Unallocated string symbol '{}'.", sym->identifier()));
        return token_for_string({});
    }
    return builder.load(compute::Type::of<uint32_t>(), st->val_ptr);
}

void OSLTranslator::Impl::store_string_token(uint32_t active_layer, const Symbol *sym, xir::Value *token) noexcept {
    const auto *st = find_storage(active_layer, sym);
    if (st == nullptr || st->val_ptr == nullptr) {
        report_error(luisa::format("Unallocated string destination '{}'.", sym->identifier()));
        return;
    }
    builder.store(st->val_ptr, token);
}

xir::Value *OSLTranslator::Impl::load_string_array_element(
    uint32_t active_layer, const Symbol *arr, xir::Value *index) noexcept {
    int len = effective_array_length(active_layer, arr);
    const auto *st = find_storage(active_layer, arr);
    auto *u_t = compute::Type::of<uint32_t>();
    if (len <= 0) {
        if (options.bounds_check != BoundsCheckMode::GUARD) {
            report_error("Indexing into empty or zero-length string array.");
        }
        return token_for_string({});
    }
    if (st == nullptr || st->val_ptr == nullptr) {
        return token_for_string({});
    }
    if (options.bounds_check == BoundsCheckMode::CLAMP) {
        xir::Value *ci = clamp_index(index, len);
        return builder.load(u_t, builder.gep(u_t, st->val_ptr, {ci}));
    }
    xir::Value *iv = to_int(index);
    auto *b_t = compute::Type::of<bool>();
    xir::Value *ge0 = builder.call(b_t, xir::ArithmeticOp::BINARY_GREATER_EQUAL, {iv, const_int(0)});
    xir::Value *lt_n = builder.call(b_t, xir::ArithmeticOp::BINARY_LESS, {iv, const_int(len)});
    xir::Value *in_bounds = builder.call(b_t, xir::ArithmeticOp::BINARY_BIT_AND, {ge0, lt_n});
    xir::Value *res_ptr = builder.alloca_local(u_t);
    xir::BasicBlock *in_bb = function->create_basic_block();
    xir::BasicBlock *oob_bb = function->create_basic_block();
    xir::BasicBlock *merge_bb = function->create_basic_block();
    builder.cond_br(in_bounds, in_bb, oob_bb);
    builder.set_insertion_point(in_bb);
    builder.store(res_ptr, builder.load(u_t, builder.gep(u_t, st->val_ptr, {iv})));
    builder.br(merge_bb);
    builder.set_insertion_point(oob_bb);
    builder.store(res_ptr, token_for_string({}));
    builder.br(merge_bb);
    builder.set_insertion_point(merge_bb);
    return builder.load(u_t, res_ptr);
}

void OSLTranslator::Impl::store_string_array_element(
    uint32_t active_layer, const Symbol *arr, xir::Value *index, xir::Value *token) noexcept {
    int len = effective_array_length(active_layer, arr);
    const auto *st = find_storage(active_layer, arr);
    auto *u_t = compute::Type::of<uint32_t>();
    if (len <= 0) {
        if (options.bounds_check != BoundsCheckMode::GUARD) {
            report_error("Indexing into empty or zero-length string array.");
        }
        return;
    }
    if (st == nullptr || st->val_ptr == nullptr) { return; }
    if (options.bounds_check == BoundsCheckMode::CLAMP) {
        xir::Value *ci = clamp_index(index, len);
        builder.store(builder.gep(u_t, st->val_ptr, {ci}), token);
        return;
    }
    xir::Value *iv = to_int(index);
    auto *b_t = compute::Type::of<bool>();
    xir::Value *ge0 = builder.call(b_t, xir::ArithmeticOp::BINARY_GREATER_EQUAL, {iv, const_int(0)});
    xir::Value *lt_n = builder.call(b_t, xir::ArithmeticOp::BINARY_LESS, {iv, const_int(len)});
    xir::Value *in_bounds = builder.call(b_t, xir::ArithmeticOp::BINARY_BIT_AND, {ge0, lt_n});
    xir::BasicBlock *in_bb = function->create_basic_block();
    xir::BasicBlock *merge_bb = function->create_basic_block();
    builder.cond_br(in_bounds, in_bb, merge_bb);
    builder.set_insertion_point(in_bb);
    builder.store(builder.gep(u_t, st->val_ptr, {iv}), token);
    builder.br(merge_bb);
    builder.set_insertion_point(merge_bb);
}

// --- Matrix transformation helpers (XIR matrix stores L = M_osl^T) ---

xir::Value *OSLTranslator::Impl::transform_point_by_matrix(xir::Value *mat_L, xir::Value *pt) noexcept {
    auto *f4_t = compute::Type::of<luisa::float4>();
    auto *f3_t = compute::Type::of<luisa::float3>();
    xir::Value *x = extract_vec_comp(pt, 0u);
    xir::Value *y = extract_vec_comp(pt, 1u);
    xir::Value *z = extract_vec_comp(pt, 2u);
    xir::Value *p4 = builder.call(f4_t, xir::ArithmeticOp::AGGREGATE, {x, y, z, const_float(1.0f)});
    xir::Value *r4 = builder.call(f4_t, xir::ArithmeticOp::MATRIX_LINALG_MUL, {mat_L, p4});
    xir::Value *rx = extract_vec_comp(r4, 0u);
    xir::Value *ry = extract_vec_comp(r4, 1u);
    xir::Value *rz = extract_vec_comp(r4, 2u);
    xir::Value *rw = extract_vec_comp(r4, 3u);
    xir::Value *r3 = builder.call(f3_t, xir::ArithmeticOp::AGGREGATE, {rx, ry, rz});
    // Perspective divide when w != 1 and w != 0
    xir::Value *w_ok = builder.call(compute::Type::of<bool>(), xir::ArithmeticOp::BINARY_NOT_EQUAL, {rw, const_float(0.0f)});
    xir::Value *w_not1 = builder.call(compute::Type::of<bool>(), xir::ArithmeticOp::BINARY_NOT_EQUAL, {rw, const_float(1.0f)});
    xir::Value *do_div = builder.call(compute::Type::of<bool>(), xir::ArithmeticOp::BINARY_BIT_AND, {w_ok, w_not1});
    xir::Value *w3 = builder.call(f3_t, xir::ArithmeticOp::AGGREGATE, {rw, rw, rw});
    xir::Value *div3 = builder.call(f3_t, xir::ArithmeticOp::BINARY_DIV, {r3, w3});
    return builder.call(f3_t, xir::ArithmeticOp::SELECT, {r3, div3, do_div});
}

xir::Value *OSLTranslator::Impl::transform_vector_by_matrix(xir::Value *mat_L, xir::Value *vec) noexcept {
    auto *f4_t = compute::Type::of<luisa::float4>();
    auto *f3_t = compute::Type::of<luisa::float3>();
    xir::Value *x = extract_vec_comp(vec, 0u);
    xir::Value *y = extract_vec_comp(vec, 1u);
    xir::Value *z = extract_vec_comp(vec, 2u);
    xir::Value *v4 = builder.call(f4_t, xir::ArithmeticOp::AGGREGATE, {x, y, z, const_float(0.0f)});
    xir::Value *r4 = builder.call(f4_t, xir::ArithmeticOp::MATRIX_LINALG_MUL, {mat_L, v4});
    return builder.call(f3_t, xir::ArithmeticOp::AGGREGATE,
                        {extract_vec_comp(r4, 0u), extract_vec_comp(r4, 1u), extract_vec_comp(r4, 2u)});
}

xir::Value *OSLTranslator::Impl::transform_normal_by_matrix(xir::Value *mat_L, xir::Value *nrm) noexcept {
    auto *f3x3_t = compute::Type::of<luisa::float3x3>();
    auto *f3_t = compute::Type::of<luisa::float3>();
    auto col3 = [&](uint32_t c) {
        auto *c4 = builder.call(compute::Type::of<luisa::float4>(), xir::ArithmeticOp::EXTRACT, {mat_L, const_uint(c)});
        return builder.call(f3_t, xir::ArithmeticOp::AGGREGATE,
                            {extract_vec_comp(c4, 0u), extract_vec_comp(c4, 1u), extract_vec_comp(c4, 2u)});
    };
    auto *m3 = builder.call(f3x3_t, xir::ArithmeticOp::AGGREGATE, {col3(0u), col3(1u), col3(2u)});
    auto *inv3 = builder.call(f3x3_t, xir::ArithmeticOp::MATRIX_INVERSE, {m3});
    auto *invT3 = builder.call(f3x3_t, xir::ArithmeticOp::MATRIX_TRANSPOSE, {inv3});
    return builder.call(f3_t, xir::ArithmeticOp::MATRIX_LINALG_MUL, {invT3, nrm});
}

DualValue OSLTranslator::Impl::transform_dual_by_matrix(
    xir::Value *mat_L, TransformKind kind, const DualValue &src, bool want_deriv) noexcept {
    DualValue out;
    switch (kind) {
        case TransformKind::POINT: {
            auto *f4_t = compute::Type::of<luisa::float4>();
            auto *f3_t = compute::Type::of<luisa::float3>();
            auto *b_t = compute::Type::of<bool>();
            xir::Value *p4 = builder.call(f4_t, xir::ArithmeticOp::AGGREGATE,
                                          {extract_vec_comp(src.val, 0u),
                                           extract_vec_comp(src.val, 1u),
                                           extract_vec_comp(src.val, 2u),
                                           const_float(1.0f)});
            xir::Value *r4 = builder.call(f4_t, xir::ArithmeticOp::MATRIX_LINALG_MUL, {mat_L, p4});
            xir::Value *r3 = builder.call(f3_t, xir::ArithmeticOp::AGGREGATE,
                                          {extract_vec_comp(r4, 0u), extract_vec_comp(r4, 1u), extract_vec_comp(r4, 2u)});
            xir::Value *rw = extract_vec_comp(r4, 3u);
            xir::Value *w_ok = builder.call(b_t, xir::ArithmeticOp::BINARY_NOT_EQUAL, {rw, const_float(0.0f)});
            xir::Value *w_not1 = builder.call(b_t, xir::ArithmeticOp::BINARY_NOT_EQUAL, {rw, const_float(1.0f)});
            xir::Value *do_div = builder.call(b_t, xir::ArithmeticOp::BINARY_BIT_AND, {w_ok, w_not1});
            xir::Value *safe_w = builder.call(compute::Type::of<float>(), xir::ArithmeticOp::SELECT,
                                              {const_float(1.0f), rw, do_div});
            xir::Value *w3 = builder.call(f3_t, xir::ArithmeticOp::AGGREGATE, {safe_w, safe_w, safe_w});
            xir::Value *q_div = builder.call(f3_t, xir::ArithmeticOp::BINARY_DIV, {r3, w3});
            out.val = builder.call(f3_t, xir::ArithmeticOp::SELECT, {r3, q_div, do_div});
            if (want_deriv && src.has_derivs()) {
                auto transform_pt_deriv = [&](xir::Value *dp) -> xir::Value * {
                    xir::Value *d4 = builder.call(f4_t, xir::ArithmeticOp::AGGREGATE,
                                                  {extract_vec_comp(dp, 0u),
                                                   extract_vec_comp(dp, 1u),
                                                   extract_vec_comp(dp, 2u),
                                                   const_float(0.0f)});
                    xir::Value *dr4 = builder.call(f4_t, xir::ArithmeticOp::MATRIX_LINALG_MUL, {mat_L, d4});
                    xir::Value *dr3 = builder.call(f3_t, xir::ArithmeticOp::AGGREGATE,
                                                   {extract_vec_comp(dr4, 0u), extract_vec_comp(dr4, 1u), extract_vec_comp(dr4, 2u)});
                    xir::Value *dw = extract_vec_comp(dr4, 3u);
                    xir::Value *dw3 = builder.call(f3_t, xir::ArithmeticOp::AGGREGATE, {dw, dw, dw});
                    xir::Value *q_dw = builder.call(f3_t, xir::ArithmeticOp::BINARY_MUL, {out.val, dw3});
                    xir::Value *num = builder.call(f3_t, xir::ArithmeticOp::BINARY_SUB, {dr3, q_dw});
                    xir::Value *dq_proj = builder.call(f3_t, xir::ArithmeticOp::BINARY_DIV, {num, w3});
                    return builder.call(f3_t, xir::ArithmeticOp::SELECT, {dr3, dq_proj, do_div});
                };
                if (src.dx) { out.dx = transform_pt_deriv(src.dx); }
                if (src.dy) { out.dy = transform_pt_deriv(src.dy); }
            }
            break;
        }
        case TransformKind::VECTOR:
            out.val = transform_vector_by_matrix(mat_L, src.val);
            if (want_deriv && src.has_derivs()) {
                if (src.dx) { out.dx = transform_vector_by_matrix(mat_L, src.dx); }
                if (src.dy) { out.dy = transform_vector_by_matrix(mat_L, src.dy); }
            }
            break;
        case TransformKind::NORMAL:
            out.val = transform_normal_by_matrix(mat_L, src.val);
            if (want_deriv && src.has_derivs()) {
                if (src.dx) { out.dx = transform_normal_by_matrix(mat_L, src.dx); }
                if (src.dy) { out.dy = transform_normal_by_matrix(mat_L, src.dy); }
            }
            break;
        case TransformKind::MATRIX:
            out.val = builder.call(compute::Type::of<luisa::float4x4>(),
                                   xir::ArithmeticOp::MATRIX_LINALG_MUL, {mat_L, src.val});
            break;
    }
    return out;
}

// --- Automatic controlled switch(token) expansion ---

uint32_t OSLTranslator::Impl::external_id_for_string(luisa::string_view sv) noexcept {
    uint32_t ext = owner->intern_string(sv);
    auto [it, inserted] = external_string_reverse.emplace(ext, luisa::string{sv});
    if (!inserted && it->second != sv) {
        report_error(luisa::format(
            "String token collision in intern_string(): '{}' and '{}' both map to {}.",
            it->second, sv, ext));
    }
    return ext;
}

xir::Value *OSLTranslator::Impl::token_for_string(luisa::string_view sv) noexcept {
    return const_uint(external_id_for_string(sv));
}

void OSLTranslator::Impl::dispatch_dynamic_strings(
    luisa::span<const StringSlotSpec> slots,
    luisa::function<void(luisa::span<const luisa::string_view>)> emit_leaf) noexcept {

    size_t total_cases = 1u;
    for (const auto &s : slots) {
        if (s.lattice.is_bot()) {
            report_error("Uninitialized string passed to static-string hook.");
            return;
        }
        if (s.lattice.is_top()) {
            report_error("Dynamic string cannot be statically enumerated for static-string hook.");
            return;
        }
        total_cases *= s.lattice.tokens.size();
        if (total_cases > options.max_string_switch_cases) {
            report_error(luisa::format("Dynamic string switch Cartesian expansion ({}) exceeds limit ({}).",
                                       total_cases, options.max_string_switch_cases));
            return;
        }
    }

    luisa::vector<luisa::string_view> resolved(slots.size());
    bool any_dynamic = false;
    for (size_t i = 0u; i < slots.size(); ++i) {
        if (slots[i].lattice.is_singleton()) {
            resolved[i] = string_table.lookup(*slots[i].lattice.as_singleton());
        } else {
            any_dynamic = true;
        }
    }

    if (!any_dynamic) {
        emit_leaf(resolved);
        return;
    }

    // Build nested IndexedBranchInst switch tree
    string_switch_count++;
    xir::BasicBlock *merge_bb = function->create_basic_block();

    auto expand = [&](size_t idx, auto &self) -> void {
        while (idx < slots.size() && slots[idx].lattice.is_singleton()) {
            resolved[idx] = string_table.lookup(*slots[idx].lattice.as_singleton());
            ++idx;
        }
        if (idx == slots.size()) {
            emit_leaf(resolved);
            if (!builder.is_insertion_point_terminator()) {
                builder.br(merge_bb);
            }
            return;
        }

        const auto &slot = slots[idx];
        xir::Value *tok_val = slot.runtime_token;
        if (tok_val == nullptr) {
            report_error("Missing runtime token for dynamic string switch.");
            return;
        }
        auto *sw = builder.indexed_branch(tok_val);
        xir::BasicBlock *def_bb = function->create_basic_block();
        sw->set_default_block(def_bb);
        luisa::vector<xir::BasicBlock *> case_bbs;
        case_bbs.reserve(slot.lattice.tokens.size());
        for (StringToken tok : slot.lattice.tokens) {
            luisa::string_view sv = string_table.lookup(tok);
            uint32_t ext_id = external_id_for_string(sv);
            if (!errors.empty()) { return; }
            xir::BasicBlock *cb = function->create_basic_block();
            sw->add_case(static_cast<uint64_t>(ext_id), cb);
            case_bbs.emplace_back(cb);
        }

        builder.set_insertion_point(def_bb);
        builder.unreachable_("Unhandled dynamic string token in OSL switch.");

        for (size_t k = 0u; k < slot.lattice.tokens.size(); ++k) {
            resolved[idx] = string_table.lookup(slot.lattice.tokens[k]);
            builder.set_insertion_point(case_bbs[k]);
            self(idx + 1u, self);
        }
    };

    expand(0u, expand);
    builder.set_insertion_point(merge_bb);
}

// --- Storage allocation & group linking codegen ---

void OSLTranslator::Impl::allocate_group_storage() noexcept {
    layer_runtimes.resize(linked_group.layers.size());
    const compute::Type *closure_t = owner->closure_xir_type();
    if (closure_t == nullptr) {
        report_error("closure_xir_type() returned nullptr.");
        return;
    }

    for (size_t li = 0u; li < linked_group.layers.size(); ++li) {
        const auto &ll = linked_group.layers[li];
        auto &rt = layer_runtimes[li];
        const auto &dual_set = deriv_analysis.dual_symbols[li];

        for (auto &&sym_ptr : ll.shader->symbols()) {
            const Symbol *sym = sym_ptr.get();
            if (sym->tag() == Symbol::Tag::SYM_CONST && !sym->is_array()) { continue; }
            if (sym->tag() == Symbol::Tag::SYM_GLOBAL) { continue; }
            if (sym->type()->tag() == Type::Tag::STRUCT) { continue; }

            SymbolStorage stor;
            stor.array_length = effective_array_length(static_cast<uint32_t>(li), sym);
            stor.is_string = detail::is_simple_type(sym->type(), SimpleType::Primitive::STRING);
            stor.is_closure = (sym->type()->tag() == Type::Tag::CLOSURE);
            stor.is_const = (sym->tag() == Symbol::Tag::SYM_CONST);
            stor.is_dual = dual_set.contains(sym) && detail::is_differentiable_type(sym->type());

            if (sym->is_array() && stor.array_length <= 0) {
                // Empty unbounded array: no XIR array alloca (arraylength=0)
                rt.storage.emplace(sym, stor);
                continue;
            }

            const compute::Type *xt = detail::osl_type_to_xir_type(sym->type(), stor.array_length, closure_t);
            if (xt == nullptr) {
                report_error(luisa::format("Unsupported symbol type '{}' for '{}'.",
                                           sym->type()->identifier(), sym->identifier()));
                return;
            }

            stor.val_ptr = builder.alloca_local(xt);
            if (stor.is_dual) {
                stor.dx_ptr = builder.alloca_local(xt);
                stor.dy_ptr = builder.alloca_local(xt);
            }

            // Initialize storage from parameter override or .oso initial values
            const ParameterValue *ov = nullptr;
            if (auto it = ll.param_overrides.find(sym); it != ll.param_overrides.end()) {
                ov = it->second;
            }

            if (stor.is_closure) {
                ClosureValue z = owner->emit_closure_zero();
                if (stor.array_length == 0) {
                    builder.store(stor.val_ptr, z.value);
                } else {
                    for (int k = 0; k < stor.array_length; ++k) {
                        auto *ep = builder.gep(closure_t, stor.val_ptr, {const_int(k)});
                        builder.store(ep, z.value);
                    }
                }
            } else if (!sym->is_array()) {
                xir::Value *init_v = nullptr;
                if (ov != nullptr) {
                    if (stor.is_string) {
                        auto s = luisa::get_if<luisa::string>(&ov->value);
                        init_v = token_for_string(s ? luisa::string_view{*s} : luisa::string_view{});
                    } else if (detail::is_simple_type(sym->type(), SimpleType::Primitive::INT)) {
                        int32_t v = 0;
                        if (auto *p = luisa::get_if<int32_t>(&ov->value)) { v = *p; }
                        else if (auto *pf = luisa::get_if<float>(&ov->value)) { v = static_cast<int32_t>(*pf); }
                        init_v = const_int(v);
                    } else if (detail::is_simple_type(sym->type(), SimpleType::Primitive::FLOAT)) {
                        float v = 0.0f;
                        if (auto *p = luisa::get_if<float>(&ov->value)) { v = *p; }
                        else if (auto *pi = luisa::get_if<int32_t>(&ov->value)) { v = static_cast<float>(*pi); }
                        init_v = const_float(v);
                    } else if (detail::is_triple_type(sym->type())) {
                        luisa::float3 v{0.0f};
                        if (auto *p = luisa::get_if<luisa::float3>(&ov->value)) { v = *p; }
                        else if (auto *pf = luisa::get_if<float>(&ov->value)) { v = luisa::float3{*pf}; }
                        else if (auto *pi = luisa::get_if<int32_t>(&ov->value)) { v = luisa::float3{static_cast<float>(*pi)}; }
                        init_v = const_float3(v);
                    } else if (detail::is_simple_type(sym->type(), SimpleType::Primitive::MATRIX)) {
                        luisa::float4x4 m = luisa::make_float4x4(0.0f);
                        if (auto *p = luisa::get_if<luisa::float4x4>(&ov->value)) { m = *p; }
                        else if (auto *pf = luisa::get_if<float>(&ov->value)) { m = luisa::make_float4x4(*pf); }
                        else if (auto *pi = luisa::get_if<int32_t>(&ov->value)) { m = luisa::make_float4x4(static_cast<float>(*pi)); }
                        init_v = const_float4x4(m);
                    }
                } else if (!sym->initial_values().empty()) {
                    init_v = stor.is_string
                                 ? token_for_string(sym->initial_values()[0].as_string())
                                 : make_constant_dual(sym).val;
                } else {
                    init_v = stor.is_string ? token_for_string({}) : zero_of(xt);
                }
                builder.store(stor.val_ptr, init_v);
                if (stor.dx_ptr) { builder.store(stor.dx_ptr, zero_of(xt)); }
                if (stor.dy_ptr) { builder.store(stor.dy_ptr, zero_of(xt)); }
            } else {
                // Array initialization
                if (stor.is_string) {
                    xir::Value *empty_tok = token_for_string({});
                    auto *elem_t = compute::Type::of<uint32_t>();
                    for (int k = 0; k < stor.array_length; ++k) {
                        builder.store(builder.gep(elem_t, stor.val_ptr, {const_int(k)}), empty_tok);
                    }
                } else {
                    builder.store(stor.val_ptr, zero_of(xt));
                }
                if (stor.dx_ptr) { builder.store(stor.dx_ptr, zero_of(xt)); }
                if (stor.dy_ptr) { builder.store(stor.dy_ptr, zero_of(xt)); }

                const compute::Type *elem_t = xt->element();
                auto write_elem = [&](int idx, xir::Value *ev) {
                    builder.store(builder.gep(elem_t, stor.val_ptr, {const_int(idx)}), ev);
                };
                if (ov != nullptr) {
                    if (auto *vi = luisa::get_if<luisa::vector<int32_t>>(&ov->value)) {
                        for (int k = 0; k < std::min<int>(stor.array_length, static_cast<int>(vi->size())); ++k) {
                            write_elem(k, detail::is_simple_type(sym->type(), SimpleType::Primitive::FLOAT) ?
                                              const_float(static_cast<float>((*vi)[k])) :
                                              const_int((*vi)[k]));
                        }
                    } else if (auto *vf = luisa::get_if<luisa::vector<float>>(&ov->value)) {
                        for (int k = 0; k < std::min<int>(stor.array_length, static_cast<int>(vf->size())); ++k) {
                            write_elem(k, const_float((*vf)[k]));
                        }
                    } else if (auto *v3 = luisa::get_if<luisa::vector<luisa::float3>>(&ov->value)) {
                        for (int k = 0; k < std::min<int>(stor.array_length, static_cast<int>(v3->size())); ++k) {
                            write_elem(k, const_float3((*v3)[k]));
                        }
                    } else if (auto *vm = luisa::get_if<luisa::vector<luisa::float4x4>>(&ov->value)) {
                        for (int k = 0; k < std::min<int>(stor.array_length, static_cast<int>(vm->size())); ++k) {
                            write_elem(k, const_float4x4((*vm)[k]));
                        }
                    } else if (auto *vs = luisa::get_if<luisa::vector<luisa::string>>(&ov->value)) {
                        for (int k = 0; k < std::min<int>(stor.array_length, static_cast<int>(vs->size())); ++k) {
                            write_elem(k, token_for_string((*vs)[k]));
                        }
                    }
                } else if (!sym->initial_values().empty()) {
                    const auto &iv = sym->initial_values();
                    size_t stride = detail::literal_stride(sym->type());
                    int count = std::min<int>(stor.array_length, static_cast<int>(iv.size() / std::max<size_t>(1u, stride)));
                    for (int k = 0; k < count; ++k) {
                        size_t base = static_cast<size_t>(k) * stride;
                        if (stor.is_string) {
                            write_elem(k, token_for_string(iv[base].as_string()));
                        } else if (detail::is_simple_type(sym->type(), SimpleType::Primitive::INT)) {
                            write_elem(k, const_int(static_cast<int32_t>(iv[base].as_int())));
                        } else if (detail::is_simple_type(sym->type(), SimpleType::Primitive::FLOAT)) {
                            write_elem(k, const_float(iv[base].as_float()));
                        } else if (detail::is_triple_type(sym->type())) {
                            write_elem(k, const_float3(luisa::float3{iv[base].as_float(), iv[base + 1u].as_float(), iv[base + 2u].as_float()}));
                        } else if (detail::is_simple_type(sym->type(), SimpleType::Primitive::MATRIX)) {
                            float raw[16];
                            for (size_t j = 0u; j < 16u; ++j) { raw[j] = iv[base + j].as_float(); }
                            write_elem(k, const_float4x4(detail::osl_row_major_to_xir_matrix(raw)));
                        }
                    }
                }
            }
            rt.storage.emplace(sym, stor);
        }
    }
}

void OSLTranslator::Impl::emit_group_body(const ShaderGroup &group) noexcept {
    owner->on_begin_group(group);
    if (!errors.empty()) { return; }

    allocate_group_storage();
    if (!errors.empty()) { return; }

    for (size_t li = 0u; li < linked_group.layers.size(); ++li) {
        emit_layer(static_cast<uint32_t>(li));
        if (!errors.empty()) { return; }
    }

    // Collect output bindings (including arrays and structs recursively)
    auto populate_output = [&](uint32_t active_layer_idx, uint32_t group_layer_idx,
                               luisa::string_view layer_nm, const Symbol *sym,
                               bool want_deriv, OutputBinding &b, auto &self) -> void {
        b.layer_index = group_layer_idx;
        b.layer_name = layer_nm;
        b.symbol = sym;
        b.field_name = sym->identifier();
        b.type = sym->type();
        b.array_length = effective_array_length(active_layer_idx, sym);
        if (sym->type()->tag() == Type::Tag::STRUCT) {
            b.fields.resize(sym->children().size());
            for (size_t k = 0u; k < sym->children().size(); ++k) {
                self(active_layer_idx, group_layer_idx, layer_nm,
                     sym->children()[k], want_deriv, b.fields[k], self);
            }
            return;
        }
        if (sym->is_array()) {
            int n = std::max(0, b.array_length);
            b.elements.resize(static_cast<size_t>(n));
            for (int k = 0; k < n; ++k) {
                auto &eb = b.elements[static_cast<size_t>(k)];
                eb.layer_index = group_layer_idx;
                eb.layer_name = layer_nm;
                eb.symbol = sym;
                eb.field_name = sym->identifier();
                eb.type = sym->type();
                if (sym->type()->tag() == Type::Tag::CLOSURE) {
                    eb.closure = load_closure_array_element(active_layer_idx, sym, const_int(k));
                } else if (detail::is_simple_type(sym->type(), SimpleType::Primitive::STRING)) {
                    eb.value = {.val = load_string_array_element(active_layer_idx, sym, const_int(k))};
                } else {
                    eb.value = load_array_element(active_layer_idx, sym, const_int(k), want_deriv);
                }
            }
            return;
        }
        if (sym->type()->tag() == Type::Tag::CLOSURE) {
            b.closure = load_closure(active_layer_idx, sym);
        } else if (detail::is_simple_type(sym->type(), SimpleType::Primitive::STRING)) {
            b.value = {.val = load_string_token(active_layer_idx, sym)};
        } else {
            b.value = load_symbol(active_layer_idx, sym, want_deriv);
        }
    };

    luisa::vector<OutputBinding> out_bindings;
    out_bindings.reserve(linked_group.outputs.size());
    for (const auto &out_req : linked_group.outputs) {
        const auto &ll = linked_group.layers[out_req.layer_index];
        OutputBinding b;
        populate_output(out_req.layer_index, ll.group_layer_index, ll.layer->name,
                        out_req.symbol, out_req.want_deriv, b, populate_output);
        out_bindings.emplace_back(std::move(b));
    }

    clear_location();
    owner->on_end_group(group, out_bindings);
    if (!errors.empty()) { return; }

    if (!builder.is_insertion_point_terminator()) {
        if (function->type() == nullptr) {
            builder.return_void();
        }
    }
}

void OSLTranslator::Impl::emit_layer(uint32_t active_layer) noexcept {
    const auto &ll = linked_group.layers[active_layer];
    current_layer_name = ll.layer->name;

    // 1. Copy incoming connections into destination parameter storage
    for (uint32_t ci : ll.incoming_connections) {
        const auto &conn = linked_group.connections[ci];
        auto copy_sym = [&](const Symbol *src_s, const Symbol *dst_s, auto &self) -> void {
            if (!src_s->children().empty() && !dst_s->children().empty()) {
                for (size_t k = 0u; k < std::min(src_s->children().size(), dst_s->children().size()); ++k) {
                    self(src_s->children()[k], dst_s->children()[k], self);
                }
                return;
            }
            if (dst_s->is_array()) {
                int src_len = effective_array_length(conn.src_layer, src_s);
                int dst_len = effective_array_length(active_layer, dst_s);
                int n = std::min(src_len, dst_len);
                const auto *dst_st = find_storage(active_layer, dst_s);
                bool dst_deriv = dst_st != nullptr && dst_st->is_dual;
                for (int k = 0; k < n; ++k) {
                    if (dst_s->type()->tag() == Type::Tag::CLOSURE) {
                        const auto *src_st = find_storage(conn.src_layer, src_s);
                        auto *sv = builder.load(owner->closure_xir_type(),
                                                builder.gep(owner->closure_xir_type(), src_st->val_ptr, {const_int(k)}));
                        builder.store(builder.gep(owner->closure_xir_type(), dst_st->val_ptr, {const_int(k)}), sv);
                    } else if (detail::is_simple_type(dst_s->type(), SimpleType::Primitive::STRING)) {
                        const auto *src_st = find_storage(conn.src_layer, src_s);
                        auto *sv = builder.load(compute::Type::of<uint32_t>(),
                                                builder.gep(compute::Type::of<uint32_t>(), src_st->val_ptr, {const_int(k)}));
                        builder.store(builder.gep(compute::Type::of<uint32_t>(), dst_st->val_ptr, {const_int(k)}), sv);
                    } else {
                        DualValue dv = load_array_element(conn.src_layer, src_s, const_int(k), dst_deriv);
                        DualValue cv = convert_dual(dv, src_s->type(), dst_s->type(), dst_deriv);
                        store_array_element(active_layer, dst_s, const_int(k), cv);
                    }
                }
                return;
            }
            if (dst_s->type()->tag() == Type::Tag::CLOSURE) {
                store_closure(active_layer, dst_s, load_closure(conn.src_layer, src_s));
            } else if (detail::is_simple_type(dst_s->type(), SimpleType::Primitive::STRING)) {
                store_string_token(active_layer, dst_s, load_string_token(conn.src_layer, src_s));
            } else {
                const auto *dst_st = find_storage(active_layer, dst_s);
                bool dst_deriv = dst_st != nullptr && dst_st->is_dual;
                DualValue dv = load_symbol(conn.src_layer, src_s, dst_deriv);
                DualValue cv = convert_dual(dv, src_s->type(), dst_s->type(), dst_deriv);
                store_symbol(active_layer, dst_s, cv);
            }
        };
        copy_sym(conn.src_symbol, conn.dst_symbol, copy_sym);
    }

    // 2. Emit active CodeRanges in order
    xir::BasicBlock *layer_exit_bb = function->create_basic_block();
    layer_runtimes[active_layer].layer_exit_block = layer_exit_bb;

    xir::BasicBlock *cur_bb = builder.insertion_point()->parent_block();
    bool stopped_early = false;
    for (const auto &range : ll.ranges) {
        if (!range.is_main && range.param_symbol != nullptr &&
            ll.skip_init_params.contains(range.param_symbol->root())) {
            continue;
        }
        xir::BasicBlock *next_bb = range.is_main ? layer_exit_bb : function->create_basic_block();
        bool reached_exit = emit_code_range(active_layer, range, next_bb);
        if (!errors.empty()) { return; }
        builder.set_insertion_point(next_bb);
        cur_bb = next_bb;
        if (!range.is_main && !reached_exit) {
            builder.unreachable_("Unreachable after unconditional layer exit.");
            stopped_early = true;
            break;
        }
    }
    if (!stopped_early && cur_bb != layer_exit_bb && !builder.is_insertion_point_terminator()) {
        builder.br(layer_exit_bb);
    }
    builder.set_insertion_point(layer_exit_bb);
}

bool OSLTranslator::Impl::emit_code_range(
    uint32_t active_layer, const detail::OSOCodeRange &range, xir::BasicBlock *next_after_range) noexcept {
    if (range.blocks.empty()) {
        if (!builder.is_insertion_point_terminator()) {
            builder.br(next_after_range);
        }
        return true;
    }

    const auto &ll = linked_group.layers[active_layer];
    const auto &la = string_analysis.layers[active_layer];

    // Discover blocks reachable along non-pruned CFG edges
    luisa::vector<xir::BasicBlock *> xir_blocks(range.blocks.size(), nullptr);
    xir_blocks[range.entry_block] = builder.insertion_point()->parent_block();
    xir_blocks[range.exit_block] = next_after_range;

    luisa::vector<bool> visited(range.blocks.size(), false);
    luisa::queue<uint32_t> q;
    visited[range.entry_block] = true;
    q.push(range.entry_block);

    auto get_xir_block = [&](uint32_t bid) -> xir::BasicBlock * {
        if (xir_blocks[bid] == nullptr) {
            xir_blocks[bid] = function->create_basic_block();
        }
        return xir_blocks[bid];
    };

    while (!q.empty()) {
        uint32_t bid = q.front();
        q.pop();
        if (bid == range.exit_block) { continue; }
        const auto &blk = range.blocks[bid];
        xir::BasicBlock *xbb = get_xir_block(bid);
        builder.set_insertion_point(xbb);

        for (uint32_t pc : blk.pcs) {
            emit_instruction(active_layer, pc, *ll.shader->instructions()[pc]);
            if (!errors.empty()) { return false; }
        }
        if (builder.is_insertion_point_terminator()) { continue; }

        auto enqueue = [&](uint32_t succ_id) {
            if (succ_id != ~0u && succ_id < range.blocks.size() && !visited[succ_id]) {
                visited[succ_id] = true;
                q.push(succ_id);
            }
        };

        if (blk.term == detail::OSOTermKind::BR) {
            uint32_t target = blk.succ[0];
            if (target == ~0u) {
                builder.unreachable_("Unterminated OSL CFG block.");
            } else {
                builder.br(get_xir_block(target));
                enqueue(target);
            }
        } else if (blk.term == detail::OSOTermKind::EXIT_LAYER) {
            builder.br(layer_runtimes[active_layer].layer_exit_block);
        } else if (blk.term == detail::OSOTermKind::COND_BR) {
            const auto &ia = la.instructions[blk.cond_pc];
            if (ia.folded_branch.has_value()) {
                uint32_t chosen = *ia.folded_branch ? blk.succ[0] : blk.succ[1];
                builder.br(get_xir_block(chosen));
                enqueue(chosen);
            } else {
                set_location(ll, blk.cond_pc, *ll.shader->instructions()[blk.cond_pc]);
                DualValue cond_val = load_symbol(active_layer, blk.cond_symbol, false);
                xir::Value *cond_bool = to_bool(cond_val.val);
                xir::BasicBlock *tbb = get_xir_block(blk.succ[0]);
                xir::BasicBlock *fbb = get_xir_block(blk.succ[1]);
                builder.cond_br(cond_bool, tbb, fbb);
                enqueue(blk.succ[0]);
                enqueue(blk.succ[1]);
            }
        } else {
            builder.unreachable_("Unreachable OSL block.");
        }
    }
    return visited[range.exit_block];
}

void OSLTranslator::Impl::emit_instruction(
    uint32_t active_layer, uint32_t pc, const Instruction &inst) noexcept {
    set_location(linked_group.layers[active_layer], pc, inst);
    if (!try_emit_instruction(active_layer, pc, inst)) {
        if (errors.empty() || errors.back().instruction_index != pc) {
            report_error(luisa::format("Unsupported or malformed OSL instruction '{}'.", inst.opcode()));
        }
    }
}

// --- Intra-group message slots ---

void OSLTranslator::Impl::emit_intra_group_setmessage(
    luisa::string_view name, const Type *src_type, int src_array_len, const DualValue &value) noexcept {
    if (src_type == nullptr || src_type->tag() != Type::Tag::SIMPLE) {
        report_error(luisa::format("setmessage('{}'): unsupported message type.", name));
        return;
    }
    const compute::Type *xt = detail::osl_type_to_xir_type(src_type, src_array_len, owner->closure_xir_type());
    if (xt == nullptr) {
        report_error(luisa::format("setmessage('{}'): failed to map message type.", name));
        return;
    }
    auto &slot = group_messages[luisa::string{name}];
    if (slot.has_value_ptr == nullptr) {
        // Place alloca in function entry block
        xir::Instruction *saved_ip = builder.insertion_point();
        builder.set_insertion_point(function->body_block()->instructions().head_sentinel());
        slot.osl_type = src_type;
        slot.array_length = src_array_len;
        slot.has_value_ptr = builder.alloca_local(compute::Type::of<bool>());
        builder.store(slot.has_value_ptr, const_bool(false));
        auto *vp = builder.alloca_local(xt);
        builder.store(vp, zero_of(xt));
        slot.val_ptrs.emplace_back(vp);
        if (detail::is_differentiable_type(src_type)) {
            auto *dxp = builder.alloca_local(xt);
            auto *dyp = builder.alloca_local(xt);
            builder.store(dxp, zero_of(xt));
            builder.store(dyp, zero_of(xt));
            slot.dx_ptrs.emplace_back(dxp);
            slot.dy_ptrs.emplace_back(dyp);
        }
        builder.set_insertion_point(saved_ip);
    }
    if (slot.val_ptrs.front()->type() != xt) {
        report_error(luisa::format("setmessage('{}'): conflicting message types within ShaderGroup.", name));
        return;
    }
    builder.store(slot.has_value_ptr, const_bool(true));
    builder.store(slot.val_ptrs.front(), value.val);
    if (!slot.dx_ptrs.empty()) {
        builder.store(slot.dx_ptrs.front(), value.dx ? value.dx : zero_of(xt));
        builder.store(slot.dy_ptrs.front(), value.dy ? value.dy : zero_of(xt));
    }
}

xir::Value *OSLTranslator::Impl::emit_intra_group_getmessage(
    luisa::string_view name, const Type *dst_type, int dst_array_len, bool want_deriv, DualValue &out_value) noexcept {
    auto it = group_messages.find(name);
    if (it == group_messages.end() || it->second.has_value_ptr == nullptr) {
        return const_int(0);
    }
    const auto &slot = it->second;
    if (slot.array_length != dst_array_len) {
        return const_int(0);
    }
    const compute::Type *src_xt = slot.val_ptrs.front()->type();
    xir::Value *loaded_val = builder.load(src_xt, slot.val_ptrs.front());
    DualValue src_dv{.val = loaded_val};
    if (want_deriv && !slot.dx_ptrs.empty()) {
        src_dv.dx = builder.load(src_xt, slot.dx_ptrs.front());
        src_dv.dy = builder.load(src_xt, slot.dy_ptrs.front());
    }
    if (dst_array_len == 0) {
        out_value = convert_dual(src_dv, slot.osl_type, dst_type, want_deriv);
    } else {
        if (src_xt != detail::osl_type_to_xir_type(dst_type, dst_array_len, owner->closure_xir_type())) {
            return const_int(0);
        }
        out_value = src_dv;
    }
    xir::Value *ok_bool = builder.load(compute::Type::of<bool>(), slot.has_value_ptr);
    return to_int(ok_bool);
}

// --- OSLTranslator public & protected API ---

OSLTranslator::OSLTranslator() noexcept
    : _impl{luisa::make_unique<Impl>()} {
    _impl->owner = this;
}

OSLTranslator::~OSLTranslator() noexcept = default;

TranslationResult OSLTranslator::translate(
    const Shader &shader,
    xir::FunctionDefinition *target_function,
    const TranslatorOptions &options) noexcept {
    ShaderGroup group;
    group.add_layer(luisa::string{shader.identifier()}, &shader);
    return translate(group, target_function, options);
}

TranslationResult OSLTranslator::translate(
    const ShaderGroup &group,
    xir::FunctionDefinition *target_function,
    const TranslatorOptions &options) noexcept {
    // Reset Impl per translation call
    _impl->module = nullptr;
    _impl->function = nullptr;
    _impl->options = options;
    _impl->string_table.clear();
    _impl->external_string_reverse.clear();
    _impl->errors.clear();
    _impl->linked_group = {};
    _impl->string_analysis = {};
    _impl->deriv_analysis = {};
    _impl->layer_runtimes.clear();
    _impl->group_messages.clear();
    _impl->string_switch_count = 0u;
    _impl->current_layer_name.clear();
    _impl->clear_location();

    auto finalize_result = [&]() noexcept -> TranslationResult {
        TranslationResult res;
        res.errors = std::move(_impl->errors);
        res.string_table = _impl->string_table;
        res.active_layer_count = _impl->linked_group.layers.size();
        res.eliminated_layer_count = _impl->linked_group.eliminated_layer_count;
        res.dual_symbol_count = _impl->deriv_analysis.total_dual_symbols;
        res.string_switch_count = _impl->string_switch_count;
        return res;
    };

    if (target_function == nullptr || target_function->parent_module() == nullptr) {
        _impl->report_error("Target XIR FunctionDefinition or its parent Module is null.");
        return finalize_result();
    }
    _impl->function = target_function;
    _impl->module = target_function->parent_module();

    if (target_function->body_block() == nullptr) {
        static_cast<void>(target_function->create_body_block());
    }
    if (target_function->body_block()->is_terminated()) {
        _impl->report_error("Target XIR FunctionDefinition body block is already terminated.");
        return finalize_result();
    }
    _impl->builder.set_insertion_point(target_function->body_block());

    if (!detail::link_shader_group(group, options.prune_dead_layers, _impl->linked_group, _impl->errors)) {
        return finalize_result();
    }

    detail::analyze_group_strings_and_constants(
        _impl->linked_group, _impl->string_table,
        options.max_string_lattice_size, _impl->string_analysis, _impl->errors);
    if (!_impl->errors.empty()) {
        return finalize_result();
    }

    detail::analyze_group_derivatives(
        _impl->linked_group, _impl->string_analysis, _impl->deriv_analysis);

    _impl->emit_group_body(group);
    if (!_impl->errors.empty()) {
        return finalize_result();
    }

    if (options.verify_xir) {
        auto ver = xir::xir_verify_module(_impl->module);
        if (!ver.succeeded()) {
            for (const auto &e : ver.errors) {
                _impl->errors.emplace_back(TranslationError{
                    .message = luisa::format("XIR verification failed: {}", e.message)});
            }
            return finalize_result();
        }
    }

    if (options.run_xir_passes) {
        static_cast<void>(xir::simplify_cfg_pass_run_on_module(_impl->module));
        static_cast<void>(xir::mem2reg_pass_run_on_module(_impl->module));
        static_cast<void>(xir::sccp_pass_run_on_module(_impl->module));
        static_cast<void>(xir::const_fold_pass_run_on_module(_impl->module));
        static_cast<void>(xir::algebraic_simplify_pass_run_on_module(_impl->module));
        static_cast<void>(xir::dce_pass_run_on_module(_impl->module));
        static_cast<void>(xir::simplify_cfg_pass_run_on_module(_impl->module));
    }

    if (options.restructure_for_ast) {
        static_cast<void>(xir::reg2mem_pass_run_on_module(_impl->module));
        auto irred = xir::lower_irreducible_cfg_pass_run_on_module(_impl->module);
        if (!irred.succeeded()) {
            _impl->errors.emplace_back(TranslationError{
                .message = "XIR lower_irreducible_cfg pass failed."});
            return finalize_result();
        }
        auto rest = xir::restructure_cfg_pass_run_on_module(_impl->module);
        if (!rest.succeeded()) {
            _impl->errors.emplace_back(TranslationError{
                .message = "XIR restructure_cfg pass failed."});
            return finalize_result();
        }
        static_cast<void>(xir::reg2mem_pass_run_on_module(_impl->module));
        if (options.verify_xir) {
            auto ver = xir::xir_verify_module(
                _impl->module,
                {.require_no_phi = true,
                 .require_canonical_break_continue_targets = true});
            if (!ver.succeeded()) {
                for (const auto &e : ver.errors) {
                    _impl->errors.emplace_back(TranslationError{
                        .message = luisa::format("Post-restructure XIR verification failed: {}", e.message)});
                }
                return finalize_result();
            }
        }
    }

    return finalize_result();
}

// --- Protected accessors & helpers ---

xir::Module *OSLTranslator::xir_module() const noexcept { return _impl->module; }
xir::FunctionDefinition *OSLTranslator::xir_function() const noexcept { return _impl->function; }
xir::XIRBuilder &OSLTranslator::xir_builder() noexcept { return _impl->builder; }
const StringTable &OSLTranslator::string_table() const noexcept { return _impl->string_table; }
void OSLTranslator::report_error(luisa::string message) noexcept { _impl->report_error(std::move(message)); }

void OSLTranslator::store_named_option_output(const NamedOption &opt, const DualValue &value) noexcept {
    if (!opt.is_output || opt.out_val_ptr == nullptr || !value.valid()) { return; }
    const compute::Type *xt = opt.out_val_ptr->type();
    _impl->builder.store(opt.out_val_ptr, value.val);
    if (opt.out_dx_ptr != nullptr) {
        _impl->builder.store(opt.out_dx_ptr, value.dx ? value.dx : _impl->zero_of(xt));
    }
    if (opt.out_dy_ptr != nullptr) {
        _impl->builder.store(opt.out_dy_ptr, value.dy ? value.dy : _impl->zero_of(xt));
    }
}

xir::Value *OSLTranslator::const_int(int32_t v) noexcept { return _impl->const_int(v); }
xir::Value *OSLTranslator::const_uint(uint32_t v) noexcept { return _impl->const_uint(v); }
xir::Value *OSLTranslator::const_float(float v) noexcept { return _impl->const_float(v); }
xir::Value *OSLTranslator::const_bool(bool v) noexcept { return _impl->const_bool(v); }
xir::Value *OSLTranslator::const_float3(luisa::float3 v) noexcept { return _impl->const_float3(v); }
xir::Value *OSLTranslator::const_float4(luisa::float4 v) noexcept { return _impl->const_float4(v); }
xir::Value *OSLTranslator::const_float4x4(luisa::float4x4 v) noexcept { return _impl->const_float4x4(v); }
xir::Value *OSLTranslator::zero_of(const compute::Type *type) noexcept { return _impl->zero_of(type); }

// --- Default Template Method hooks (fail-closed where renderer semantics are required) ---

const compute::Type *OSLTranslator::closure_xir_type() const noexcept {
    return compute::Type::of<uint32_t>();
}

DualValue OSLTranslator::emit_global_read(GlobalSymbol global, bool) noexcept {
    report_error(luisa::format("OSLTranslator::emit_global_read not implemented for global #{}.",
                               static_cast<uint32_t>(global)));
    return {};
}

void OSLTranslator::emit_global_write(GlobalSymbol global, const DualValue &) noexcept {
    report_error(luisa::format("OSLTranslator::emit_global_write not implemented for global #{}.",
                               static_cast<uint32_t>(global)));
}

void OSLTranslator::emit_closure_output(const ClosureValue &) noexcept {
    report_error("OSLTranslator::emit_closure_output not implemented for Ci.");
}

xir::Value *OSLTranslator::emit_backfacing() noexcept {
    report_error("OSLTranslator::emit_backfacing not implemented.");
    return nullptr;
}

xir::Value *OSLTranslator::emit_raytype(luisa::string_view type_name) noexcept {
    report_error(luisa::format("OSLTranslator::emit_raytype('{}') not implemented.", type_name));
    return nullptr;
}

ClosureValue OSLTranslator::emit_closure_zero() noexcept {
    return {.value = zero_of(closure_xir_type())};
}

ClosureValue OSLTranslator::emit_closure(
    luisa::string_view name, luisa::span<const HookArg>, luisa::span<const NamedOption>) noexcept {
    report_error(luisa::format("OSLTranslator::emit_closure('{}') not implemented.", name));
    return {};
}

ClosureValue OSLTranslator::emit_closure_scale(const ClosureValue &, xir::Value *) noexcept {
    report_error("OSLTranslator::emit_closure_scale not implemented.");
    return {};
}

ClosureValue OSLTranslator::emit_closure_add(const ClosureValue &, const ClosureValue &) noexcept {
    report_error("OSLTranslator::emit_closure_add not implemented.");
    return {};
}

size_t OSLTranslator::closure_positional_arg_count(
    luisa::string_view name,
    luisa::span<const Type *const> arg_types) const noexcept {
    const size_t n = arg_types.size();
    auto is_str = [&](size_t idx) noexcept {
        return idx < n && detail::is_simple_type(arg_types[idx], SimpleType::Primitive::STRING);
    };
    size_t min_pos = 0u;
    if (name == "emission" || name == "background" || name == "transparent" || name == "holdout") {
        min_pos = 0u;
    } else if (name == "diffuse" || name == "translucent" || name == "debug") {
        min_pos = std::min<size_t>(n, 1u);
    } else if (name == "oren_nayar" || name == "phong" || name == "westin" || name == "refraction") {
        min_pos = std::min<size_t>(n, 2u);
    } else if (name == "reflection") {
        min_pos = (n >= 2u && !is_str(1u)) ? 2u : std::min<size_t>(n, 1u);
    } else if (name == "ward") {
        min_pos = std::min<size_t>(n, 4u);
    } else if (name == "microfacet") {
        // microfacet(dist, N, ...) always takes positional string `dist` then normal `N`
        min_pos = std::min<size_t>(n, 2u);
    } else if (name == "bssrdf") {
        min_pos = std::min<size_t>(n, 4u);
    }
    size_t opt_start = n;
    while (opt_start >= min_pos + 2u && is_str(opt_start - 2u)) {
        opt_start -= 2u;
    }
    return opt_start;
}

xir::Value *OSLTranslator::emit_getattribute(
    luisa::optional<luisa::string_view>, luisa::string_view attr_name,
    const Type *, int, bool, DualValue &) noexcept {
    report_error(luisa::format("OSLTranslator::emit_getattribute('{}') not implemented.", attr_name));
    return nullptr;
}

xir::Value *OSLTranslator::emit_getmessage(
    luisa::optional<luisa::string_view> source, luisa::string_view msg_name,
    const Type *dst_type, int dst_array_len, bool want_deriv, DualValue &out_value) noexcept {
    if (!source.has_value()) {
        return _impl->emit_intra_group_getmessage(msg_name, dst_type, dst_array_len, want_deriv, out_value);
    }
    report_error(luisa::format("OSLTranslator::emit_getmessage('{}', '{}') not implemented.",
                               *source, msg_name));
    return nullptr;
}

void OSLTranslator::emit_setmessage(
    luisa::string_view msg_name, const Type *src_type, int src_array_len, const DualValue &value) noexcept {
    _impl->emit_intra_group_setmessage(msg_name, src_type, src_array_len, value);
}

DualValue OSLTranslator::emit_texture(
    TextureOp, luisa::string_view filename, luisa::span<const DualValue>,
    luisa::span<const DualValue>, const Type *, bool, luisa::span<const NamedOption>) noexcept {
    report_error(luisa::format("OSLTranslator::emit_texture('{}') not implemented.", filename));
    return {};
}

xir::Value *OSLTranslator::emit_gettextureinfo(
    luisa::string_view filename, luisa::string_view dataname,
    const Type *, int, DualValue &) noexcept {
    report_error(luisa::format("OSLTranslator::emit_gettextureinfo('{}', '{}') not implemented.",
                               filename, dataname));
    return nullptr;
}

xir::Value *OSLTranslator::emit_getmatrix(
    luisa::string_view from_space, luisa::string_view to_space, xir::Value *&out_matrix) noexcept {
    if (from_space == to_space) {
        out_matrix = const_float4x4(luisa::make_float4x4(1.0f));
        return const_int(1);
    }
    report_error(luisa::format("OSLTranslator::emit_getmatrix('{}', '{}') not implemented.",
                               from_space, to_space));
    return nullptr;
}

DualValue OSLTranslator::emit_transform_space(
    luisa::string_view from_space, luisa::string_view to_space,
    TransformKind kind, const DualValue &value, bool want_deriv) noexcept {
    if (from_space == to_space) {
        return want_deriv ? value : DualValue{.val = value.val};
    }
    xir::Value *mat_L = nullptr;
    xir::Value *ok = emit_getmatrix(from_space, to_space, mat_L);
    if (!_impl->errors.empty() || ok == nullptr || mat_L == nullptr) {
        return {};
    }
    DualValue transformed = _impl->transform_dual_by_matrix(mat_L, kind, value, want_deriv);
    if (kind == TransformKind::MATRIX) {
        xir::Value *ok_b = _impl->to_bool(ok);
        xir::Value *cols[4]{};
        for (uint c = 0u; c < 4u; ++c) {
            xir::Value *idx = _impl->const_uint(c);
            xir::Value *orig_col = _impl->builder.call(
                compute::Type::of<luisa::float4>(), xir::ArithmeticOp::EXTRACT, {value.val, idx});
            xir::Value *new_col = _impl->builder.call(
                compute::Type::of<luisa::float4>(), xir::ArithmeticOp::EXTRACT, {transformed.val, idx});
            cols[c] = _impl->builder.call(
                compute::Type::of<luisa::float4>(), xir::ArithmeticOp::SELECT, {orig_col, new_col, ok_b});
        }
        transformed.val = _impl->builder.call(
            compute::Type::of<luisa::float4x4>(), xir::ArithmeticOp::AGGREGATE, {cols[0], cols[1], cols[2], cols[3]});
        return transformed;
    }
    xir::Value *ok_b = _impl->to_bool(ok);
    auto *f3_t = compute::Type::of<luisa::float3>();
    transformed.val = _impl->builder.call(f3_t, xir::ArithmeticOp::SELECT, {value.val, transformed.val, ok_b});
    if (want_deriv) {
        if (transformed.dx || value.dx) {
            transformed.dx = _impl->builder.call(
                f3_t, xir::ArithmeticOp::SELECT,
                {value.dx ? value.dx : zero_of(f3_t),
                 transformed.dx ? transformed.dx : zero_of(f3_t),
                 ok_b});
        }
        if (transformed.dy || value.dy) {
            transformed.dy = _impl->builder.call(
                f3_t, xir::ArithmeticOp::SELECT,
                {value.dy ? value.dy : zero_of(f3_t),
                 transformed.dy ? transformed.dy : zero_of(f3_t),
                 ok_b});
        }
    }
    return transformed;
}

DualValue OSLTranslator::emit_transform_color(
    luisa::string_view from_space, luisa::string_view to_space,
    const DualValue &value, bool want_deriv) noexcept {
    return _impl->emit_default_color_transform(from_space, to_space, value, want_deriv);
}

DualValue OSLTranslator::emit_noise(
    luisa::string_view noise_type, NoiseOp op,
    luisa::span<const DualValue> coords, luisa::span<const DualValue> periods,
    const Type *dst_type, bool want_deriv, luisa::span<const NamedOption> opts) noexcept {
    return _impl->emit_default_noise(noise_type, op, coords, periods, dst_type, want_deriv, opts);
}

xir::Value *OSLTranslator::emit_trace(
    const DualValue &, const DualValue &, luisa::span<const NamedOption>) noexcept {
    report_error("OSLTranslator::emit_trace not implemented.");
    return nullptr;
}

uint32_t OSLTranslator::intern_string(luisa::string_view s) noexcept {
    return _impl->string_table.intern(s);
}

}// namespace luisa::compute::osl
