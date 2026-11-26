#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <crtdbg.h>
#include <windows.h>
#endif

#include <luisa/luisa-compute.h>
#include <luisa/core/logging.h>
#include <luisa/osl/oso_parser.h>
#include <luisa/osl/shader_group.h>
#include <luisa/osl/translator.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/translators/xir2ast.h>
#include <luisa/xir/verifier.h>

namespace luisa::test::osl_test {

// Keep all structs 16-byte aligned in descending alignment order so host C++,
// HLSL StructuredBuffer, SPIR-V std430, LLVM Fallback, and SIMD layouts match.
struct TestShaderGlobals {
    luisa::float3 P{};
    luisa::float3 dPdx{1.0f, 0.0f, 0.0f};
    luisa::float3 dPdy{0.0f, 1.0f, 0.0f};
    luisa::float3 I{0.0f, 0.0f, -1.0f};
    luisa::float3 N{0.0f, 0.0f, 1.0f};
    luisa::float3 Ng{0.0f, 0.0f, 1.0f};
    luisa::float3 dPdu{1.0f, 0.0f, 0.0f};
    luisa::float3 dPdv{0.0f, 1.0f, 0.0f};
    float u{0.0f};
    float dudx{1.0f};
    float dudy{0.0f};
    float v{0.0f};
    float dvdx{0.0f};
    float dvdy{1.0f};
    float time{0.0f};
    float dtime{0.01f};
    uint32_t ray_type_mask{1u};// bit0=camera, 1=shadow, 2=diffuse, 3=glossy, 4=reflection, 5=refraction
    int32_t backfacing_flag{0};
    uint32_t pad0{0u};
    uint32_t pad1{0u};
};

struct TestClosure {
    luisa::float3 weight{0.0f};
    luisa::float3 arg0{0.0f};
    uint32_t kind_tag{0u};// 1=diffuse, 2=oren_nayar, 3=microfacet, 4=emission, 5=transparent, 6=reflection, 7=background
    float arg1{0.0f};
    float arg2{0.0f};
    float arg3{0.0f};
};

struct TestShaderOutput {
    luisa::float4x4 m0{luisa::make_float4x4(1.0f)};
    TestClosure ci{};
    luisa::float3 v0{};
    luisa::float3 v1{};
    luisa::float3 v2{};
    luisa::float3 v3{};
    luisa::float3 v4{};
    luisa::float3 v5{};
    luisa::float3 v6{};
    luisa::float3 v7{};
    float f0{0.0f};
    float f1{0.0f};
    float f2{0.0f};
    float f3{0.0f};
    float f4{0.0f};
    float f5{0.0f};
    float f6{0.0f};
    float f7{0.0f};
    int32_t i0{0};
    int32_t i1{0};
    int32_t i2{0};
    int32_t i3{0};
};

}// namespace luisa::test::osl_test

LUISA_STRUCT(luisa::test::osl_test::TestShaderGlobals,
             P, dPdx, dPdy, I, N, Ng, dPdu, dPdv,
             u, dudx, dudy, v, dvdx, dvdy, time, dtime,
             ray_type_mask, backfacing_flag, pad0, pad1){};

LUISA_STRUCT(luisa::test::osl_test::TestClosure,
             weight, arg0, kind_tag, arg1, arg2, arg3){};

LUISA_STRUCT(luisa::test::osl_test::TestShaderOutput,
             m0, ci, v0, v1, v2, v3, v4, v5, v6, v7,
             f0, f1, f2, f3, f4, f5, f6, f7,
             i0, i1, i2, i3){};

namespace luisa::test::osl_test {

// TestOSLTranslator: lightweight reference implementation of all 8 Template Method hook families
class TestOSLTranslator : public compute::osl::OSLTranslator {

private:
    compute::xir::ResourceArgument *_in_buf{nullptr};
    compute::xir::ResourceArgument *_out_buf{nullptr};
    compute::xir::Value *_tid{nullptr};
    compute::xir::Value *_globals_val{nullptr};
    compute::xir::Value *_out_local_ptr{nullptr};

    // Cached trace result per thread for getmessage("trace", ...)
    compute::xir::Value *_trace_hit_ptr{nullptr};
    compute::xir::Value *_trace_dist_ptr{nullptr};
    compute::xir::Value *_trace_p_ptr{nullptr};
    compute::xir::Value *_trace_n_ptr{nullptr};

protected:
    [[nodiscard]] compute::xir::Value *_read_global_field(uint32_t field_idx, const compute::Type *ft) noexcept {
        return xir_builder().call(ft, compute::xir::ArithmeticOp::EXTRACT,
                                  {_globals_val, const_uint(field_idx)});
    }

    void _store_out_field(uint32_t field_idx, compute::xir::Value *val) noexcept {
        const auto *out_t = compute::Type::of<TestShaderOutput>();
        compute::xir::Value *cur = xir_builder().load(out_t, _out_local_ptr);
        compute::xir::Value *next = xir_builder().call(
            out_t, compute::xir::ArithmeticOp::INSERT, {cur, val, const_uint(field_idx)});
        xir_builder().store(_out_local_ptr, next);
    }

    void _finalize_output_buffer() noexcept {
        auto &b = xir_builder();
        const auto *out_t = compute::Type::of<TestShaderOutput>();
        compute::xir::Value *final_out = b.load(out_t, _out_local_ptr);
        b.call(compute::xir::ResourceWriteOp::BUFFER_WRITE, {_out_buf, _tid, final_out});
    }

    [[nodiscard]] static luisa::float4x4 object_to_common_xir() noexcept {
        // Store transposed: L[col=r] = OSL row r
        return luisa::make_float4x4(
            2.0f, 0.5f, 0.0f, 0.0f,
            0.0f, 3.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 4.0f, 0.0f,
            10.0f, -5.0f, 2.0f, 1.0f);
    }

    [[nodiscard]] static luisa::float4x4 shader_to_common_xir() noexcept {
        // Pure uniform scale 0.5 + translation (0, 2, 0)
        return luisa::make_float4x4(
            0.5f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.5f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.5f, 0.0f,
            0.0f, 2.0f, 0.0f, 1.0f);
    }

    [[nodiscard]] static luisa::float4x4 camera_to_common_xir() noexcept {
        // Flip Z axis & translate (0, 0, 5)
        return luisa::make_float4x4(
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, -1.0f, 0.0f,
            0.0f, 0.0f, 5.0f, 1.0f);
    }

protected:
    [[nodiscard]] const compute::Type *closure_xir_type() const noexcept override {
        return compute::Type::of<TestClosure>();
    }

    void on_begin_group(const compute::osl::ShaderGroup &) noexcept override {
        auto *fn = xir_function();
        const auto *in_buf_t = compute::Type::buffer(compute::Type::of<TestShaderGlobals>());
        const auto *out_buf_t = compute::Type::buffer(compute::Type::of<TestShaderOutput>());
        _in_buf = fn->create_resource_argument(in_buf_t);
        _out_buf = fn->create_resource_argument(out_buf_t);

        auto &b = xir_builder();
        auto *did = xir_module()->create_dispatch_id();
        _tid = b.call(compute::Type::of<uint32_t>(), compute::xir::ArithmeticOp::EXTRACT,
                      {did, const_uint(0u)});
        _globals_val = b.call(compute::Type::of<TestShaderGlobals>(),
                              compute::xir::ResourceReadOp::BUFFER_READ, {_in_buf, _tid});

        _out_local_ptr = b.alloca_local(compute::Type::of<TestShaderOutput>());
        b.store(_out_local_ptr, zero_of(compute::Type::of<TestShaderOutput>()));

        _trace_hit_ptr = b.alloca_local(compute::Type::of<int32_t>());
        _trace_dist_ptr = b.alloca_local(compute::Type::of<float>());
        _trace_p_ptr = b.alloca_local(compute::Type::of<luisa::float3>());
        _trace_n_ptr = b.alloca_local(compute::Type::of<luisa::float3>());
        b.store(_trace_hit_ptr, const_int(0));
        b.store(_trace_dist_ptr, const_float(0.0f));
        b.store(_trace_p_ptr, zero_of(compute::Type::of<luisa::float3>()));
        b.store(_trace_n_ptr, zero_of(compute::Type::of<luisa::float3>()));
    }

    void on_end_group(const compute::osl::ShaderGroup &,
                      luisa::span<const compute::osl::OutputBinding> outputs) noexcept override {
        // TestShaderOutput member indices:
        // 0:m0, 1:ci, 2..9:v0..v7, 10..17:f0..f7, 18..21:i0..i3
        bool used_v[8]{};
        bool used_f[8]{};
        bool used_i[4]{};
        bool used_m0 = false;

        auto name_slot = [&](luisa::string_view n, luisa::string_view prefix, int limit) -> int {
            if (n.size() == prefix.size() + 1u && n.starts_with(prefix)) {
                char c = n[prefix.size()];
                if (c >= '0' && c < '0' + limit) { return c - '0'; }
            }
            return -1;
        };

        auto &b = xir_builder();
        auto store_binding = [&](const compute::osl::OutputBinding &binding,
                                 luisa::string_view slot_name,
                                 auto &self) -> void {
            if (!binding.fields.empty()) {
                for (const auto &f : binding.fields) {
                    self(f, f.field_name, self);
                }
                return;
            }
            if (!binding.elements.empty()) {
                for (const auto &el : binding.elements) {
                    self(el, {}, self);
                }
                return;
            }
            const compute::osl::Type *bt = binding.type ? binding.type : (binding.symbol ? binding.symbol->type() : nullptr);
            if (bt == nullptr) { return; }
            if (bt->tag() == compute::osl::Type::Tag::CLOSURE) {
                if (binding.closure.valid()) { _store_out_field(1u, binding.closure.value); }
                return;
            }
            if (!binding.value.valid()) { return; }
            if (bt->tag() == compute::osl::Type::Tag::SIMPLE) {
                auto p = static_cast<const compute::osl::SimpleType *>(bt)->primitive();
                if (p == compute::osl::SimpleType::Primitive::INT ||
                    p == compute::osl::SimpleType::Primitive::STRING) {
                    int idx = name_slot(slot_name, "i", 4);
                    if (idx < 0) {
                        for (int k = 0; k < 4; ++k) {
                            if (!used_i[k]) { idx = k; break; }
                        }
                    }
                    if (idx >= 0 && idx < 4) {
                        used_i[idx] = true;
                        auto *iv = (p == compute::osl::SimpleType::Primitive::STRING)
                                       ? b.static_cast_(compute::Type::of<int32_t>(), binding.value.val)
                                       : binding.value.val;
                        _store_out_field(18u + static_cast<uint32_t>(idx), iv);
                    }
                } else if (p == compute::osl::SimpleType::Primitive::FLOAT) {
                    int idx = name_slot(slot_name, "f", 8);
                    if (idx < 0) {
                        for (int k = 0; k < 8; ++k) {
                            if (!used_f[k]) { idx = k; break; }
                        }
                    }
                    if (idx >= 0 && idx < 8) {
                        used_f[idx] = true;
                        _store_out_field(10u + static_cast<uint32_t>(idx), binding.value.val);
                    }
                } else if (p == compute::osl::SimpleType::Primitive::MATRIX) {
                    if (!used_m0 || slot_name == "m0") {
                        used_m0 = true;
                        _store_out_field(0u, binding.value.val);
                    }
                } else if (p == compute::osl::SimpleType::Primitive::POINT ||
                           p == compute::osl::SimpleType::Primitive::VECTOR ||
                           p == compute::osl::SimpleType::Primitive::NORMAL ||
                           p == compute::osl::SimpleType::Primitive::COLOR) {
                    int idx = name_slot(slot_name, "v", 8);
                    if (idx < 0) {
                        for (int k = 0; k < 8; ++k) {
                            if (!used_v[k]) { idx = k; break; }
                        }
                    }
                    if (idx >= 0 && idx < 8) {
                        used_v[idx] = true;
                        _store_out_field(2u + static_cast<uint32_t>(idx), binding.value.val);
                    }
                }
            }
        };

        for (const auto &out : outputs) {
            if (out.symbol == nullptr) { continue; }
            store_binding(out, out.symbol->identifier(), store_binding);
        }

        _finalize_output_buffer();
    }

    // 1. Shading point global state
    [[nodiscard]] compute::osl::DualValue emit_global_read(
        compute::osl::GlobalSymbol g, bool want_deriv) noexcept override {
        auto *f3_t = compute::Type::of<luisa::float3>();
        auto *f_t = compute::Type::of<float>();
        using GS = compute::osl::GlobalSymbol;
        switch (g) {
            case GS::P:
            case GS::Ps: {
                compute::osl::DualValue dv{.val = _read_global_field(0u, f3_t)};
                if (want_deriv) {
                    dv.dx = _read_global_field(1u, f3_t);
                    dv.dy = _read_global_field(2u, f3_t);
                }
                return dv;
            }
            case GS::I: return {.val = _read_global_field(3u, f3_t)};
            case GS::N: return {.val = _read_global_field(4u, f3_t)};
            case GS::Ng: return {.val = _read_global_field(5u, f3_t)};
            case GS::dPdu: return {.val = _read_global_field(6u, f3_t)};
            case GS::dPdv: return {.val = _read_global_field(7u, f3_t)};
            case GS::u: {
                compute::osl::DualValue dv{.val = _read_global_field(8u, f_t)};
                if (want_deriv) {
                    dv.dx = _read_global_field(9u, f_t);
                    dv.dy = _read_global_field(10u, f_t);
                }
                return dv;
            }
            case GS::v: {
                compute::osl::DualValue dv{.val = _read_global_field(11u, f_t)};
                if (want_deriv) {
                    dv.dx = _read_global_field(12u, f_t);
                    dv.dy = _read_global_field(13u, f_t);
                }
                return dv;
            }
            case GS::time: return {.val = _read_global_field(14u, f_t)};
            case GS::dtime: return {.val = _read_global_field(15u, f_t)};
            case GS::dPdtime: return {.val = const_float3(luisa::float3{0.0f})};
            default:
                report_error("Unsupported global read.");
                return {};
        }
    }

    void emit_global_write(compute::osl::GlobalSymbol g, const compute::osl::DualValue &val) noexcept override {
        if (g == compute::osl::GlobalSymbol::N || g == compute::osl::GlobalSymbol::P) {
            _store_out_field(g == compute::osl::GlobalSymbol::N ? 2u : 3u, val.val);
            return;
        }
        report_error("Unsupported global write.");
    }

    void emit_closure_output(const compute::osl::ClosureValue &ci) noexcept override {
        if (ci.valid()) { _store_out_field(1u, ci.value); }
    }

    [[nodiscard]] compute::xir::Value *emit_backfacing() noexcept override {
        return _read_global_field(17u, compute::Type::of<int32_t>());
    }

    [[nodiscard]] compute::xir::Value *emit_raytype(luisa::string_view type_name) noexcept override {
        uint32_t bit = 0u;
        if (type_name == "camera") { bit = 1u; }
        else if (type_name == "shadow") { bit = 2u; }
        else if (type_name == "diffuse") { bit = 4u; }
        else if (type_name == "glossy") { bit = 8u; }
        else if (type_name == "reflection") { bit = 16u; }
        else if (type_name == "refraction") { bit = 32u; }
        else { return const_int(0); }

        auto &b = xir_builder();
        auto *mask = _read_global_field(16u, compute::Type::of<uint32_t>());
        auto *and_v = b.call(compute::Type::of<uint32_t>(), compute::xir::ArithmeticOp::BINARY_BIT_AND,
                             {mask, const_uint(bit)});
        auto *nz = b.call(compute::Type::of<bool>(), compute::xir::ArithmeticOp::BINARY_NOT_EQUAL,
                          {and_v, const_uint(0u)});
        return b.call(compute::Type::of<int32_t>(), compute::xir::ArithmeticOp::SELECT,
                      {const_int(0), const_int(1), nz});
    }

    // 2. Closure algebra
    [[nodiscard]] compute::osl::ClosureValue emit_closure_zero() noexcept override {
        return {.value = zero_of(compute::Type::of<TestClosure>())};
    }

    [[nodiscard]] compute::osl::ClosureValue emit_closure(
        luisa::string_view name,
        luisa::span<const compute::osl::HookArg> args,
        luisa::span<const compute::osl::NamedOption> opts) noexcept override {
        uint32_t tag = 0u;
        if (name == "diffuse") { tag = 1u; }
        else if (name == "oren_nayar") { tag = 2u; }
        else if (name == "microfacet") { tag = 3u; }
        else if (name == "emission") { tag = 4u; }
        else if (name == "transparent") { tag = 5u; }
        else if (name == "reflection") { tag = 6u; }
        else if (name == "background") { tag = 7u; }
        else {
            report_error(luisa::format("TestOSLTranslator: unknown closure '{}'.", name));
            return {};
        }

        luisa::compute::xir::Value *a0 = const_float3(luisa::float3{0.0f});
        luisa::compute::xir::Value *a1 = const_float(0.0f);
        luisa::compute::xir::Value *a2 = const_float(0.0f);
        luisa::compute::xir::Value *a3 = const_float(0.0f);

        auto to_f = [&](luisa::compute::xir::Value *v) -> luisa::compute::xir::Value * {
            if (v->type() == compute::Type::of<float>()) { return v; }
            if (v->type() == compute::Type::of<int32_t>()) { return xir_builder().static_cast_(compute::Type::of<float>(), v); }
            return const_float(0.0f);
        };
        auto to_f3 = [&](luisa::compute::xir::Value *v) -> luisa::compute::xir::Value * {
            if (v->type() == compute::Type::of<luisa::float3>()) { return v; }
            auto *f = to_f(v);
            return xir_builder().call(compute::Type::of<luisa::float3>(), compute::xir::ArithmeticOp::AGGREGATE, {f, f, f});
        };

        // Pack first triple arg into a0, numeric scalar args into a1..a3
        uint32_t f_slot = 0u;
        bool got_triple = false;
        for (const auto &a : args) {
            if (!a.value.valid()) { continue; }
            auto *t = a.value.val->type();
            if (!got_triple && t == compute::Type::of<luisa::float3>()) {
                a0 = to_f3(a.value.val);
                got_triple = true;
            } else if (t == compute::Type::of<float>() || t == compute::Type::of<int32_t>()) {
                if (f_slot == 0u) { a1 = to_f(a.value.val); }
                else if (f_slot == 1u) { a2 = to_f(a.value.val); }
                else if (f_slot == 2u) { a3 = to_f(a.value.val); }
                ++f_slot;
            }
        }
        for (const auto &opt : opts) {
            if (!opt.is_output && opt.key == "ior" && opt.value.valid()) {
                a3 = to_f(opt.value.val);
            }
        }

        auto *ct = compute::Type::of<TestClosure>();
        return {.value = xir_builder().call(
                    ct, compute::xir::ArithmeticOp::AGGREGATE,
                    {const_float3(luisa::float3{1.0f}), a0, const_uint(tag), a1, a2, a3})};
    }

    [[nodiscard]] compute::osl::ClosureValue emit_closure_scale(
        const compute::osl::ClosureValue &cl,
        compute::xir::Value *weight) noexcept override {
        auto &b = xir_builder();
        auto *ct = compute::Type::of<TestClosure>();
        auto *f3_t = compute::Type::of<luisa::float3>();
        auto *old_w = b.call(f3_t, compute::xir::ArithmeticOp::EXTRACT, {cl.value, const_uint(0u)});
        auto *new_w = b.call(f3_t, compute::xir::ArithmeticOp::BINARY_MUL, {old_w, weight});
        return {.value = b.call(ct, compute::xir::ArithmeticOp::INSERT, {cl.value, new_w, const_uint(0u)})};
    }

    [[nodiscard]] compute::osl::ClosureValue emit_closure_add(
        const compute::osl::ClosureValue &lhs,
        const compute::osl::ClosureValue &rhs) noexcept override {
        auto &b = xir_builder();
        auto *ct = compute::Type::of<TestClosure>();
        auto *f3_t = compute::Type::of<luisa::float3>();
        auto *f_t = compute::Type::of<float>();
        auto *u_t = compute::Type::of<uint32_t>();

        auto ex_f3 = [&](compute::xir::Value *c, uint32_t idx) { return b.call(f3_t, compute::xir::ArithmeticOp::EXTRACT, {c, const_uint(idx)}); };
        auto ex_f = [&](compute::xir::Value *c, uint32_t idx) { return b.call(f_t, compute::xir::ArithmeticOp::EXTRACT, {c, const_uint(idx)}); };
        auto ex_u = [&](compute::xir::Value *c, uint32_t idx) { return b.call(u_t, compute::xir::ArithmeticOp::EXTRACT, {c, const_uint(idx)}); };

        auto *lw = ex_f3(lhs.value, 0u);
        auto *rw = ex_f3(rhs.value, 0u);
        auto *la0 = ex_f3(lhs.value, 1u);
        auto *ra0 = ex_f3(rhs.value, 1u);
        auto *ltag = ex_u(lhs.value, 2u);
        auto *rtag = ex_u(rhs.value, 2u);

        // Combine weights cleanly, average/sum weighted payloads, preserve nonzero tag bitmask/sum
        auto *w_sum = b.call(f3_t, compute::xir::ArithmeticOp::BINARY_ADD, {lw, rw});
        auto *a0_sum = b.call(f3_t, compute::xir::ArithmeticOp::BINARY_ADD, {la0, ra0});
        auto *tag_comb = b.call(u_t, compute::xir::ArithmeticOp::BINARY_BIT_OR, {ltag, b.call(u_t, compute::xir::ArithmeticOp::BINARY_SHIFT_LEFT, {rtag, const_uint(4u)})});
        auto *r_empty = b.call(compute::Type::of<bool>(), compute::xir::ArithmeticOp::BINARY_EQUAL, {rtag, const_uint(0u)});
        auto *l_empty = b.call(compute::Type::of<bool>(), compute::xir::ArithmeticOp::BINARY_EQUAL, {ltag, const_uint(0u)});
        auto *tag_pick = b.call(u_t, compute::xir::ArithmeticOp::SELECT, {tag_comb, ltag, r_empty});
        tag_pick = b.call(u_t, compute::xir::ArithmeticOp::SELECT, {tag_pick, rtag, l_empty});

        auto *a1 = b.call(f_t, compute::xir::ArithmeticOp::BINARY_ADD, {ex_f(lhs.value, 3u), ex_f(rhs.value, 3u)});
        auto *a2 = b.call(f_t, compute::xir::ArithmeticOp::BINARY_ADD, {ex_f(lhs.value, 4u), ex_f(rhs.value, 4u)});
        auto *a3 = b.call(f_t, compute::xir::ArithmeticOp::BINARY_ADD, {ex_f(lhs.value, 5u), ex_f(rhs.value, 5u)});
        return {.value = b.call(ct, compute::xir::ArithmeticOp::AGGREGATE, {w_sum, a0_sum, tag_pick, a1, a2, a3})};
    }

    // 3. Attributes & messages
    [[nodiscard]] compute::xir::Value *emit_getattribute(
        luisa::optional<luisa::string_view> object_name,
        luisa::string_view attr_name,
        const compute::osl::Type *dst_type,
        int,
        bool want_deriv,
        compute::osl::DualValue &out_value) noexcept override {
        auto &b = xir_builder();
        if (object_name.has_value() && *object_name == "objA") {
            if (attr_name == "custom:scale") {
                out_value = {.val = const_float(4.5f)};
                return const_int(1);
            }
            return const_int(0);
        }
        if (attr_name == "object:index") {
            auto *i_tid = b.static_cast_(compute::Type::of<int32_t>(), _tid);
            auto *idx = b.call(compute::Type::of<int32_t>(), compute::xir::ArithmeticOp::BINARY_ADD, {i_tid, const_int(100)});
            if (dst_type->tag() == compute::osl::Type::Tag::SIMPLE &&
                static_cast<const compute::osl::SimpleType *>(dst_type)->primitive() == compute::osl::SimpleType::Primitive::FLOAT) {
                out_value = {.val = b.static_cast_(compute::Type::of<float>(), idx)};
            } else {
                out_value = {.val = idx};
            }
            return const_int(1);
        }
        if (attr_name == "custom:color") {
            auto *p_val = _read_global_field(0u, compute::Type::of<luisa::float3>());
            auto *col = b.call(compute::Type::of<luisa::float3>(), compute::xir::ArithmeticOp::BINARY_MUL,
                               {p_val, const_float3(luisa::float3{0.25f, 0.5f, 1.0f})});
            out_value.val = col;
            if (want_deriv) {
                out_value.dx = b.call(compute::Type::of<luisa::float3>(), compute::xir::ArithmeticOp::BINARY_MUL,
                                      {_read_global_field(1u, compute::Type::of<luisa::float3>()),
                                       const_float3(luisa::float3{0.25f, 0.5f, 1.0f})});
                out_value.dy = b.call(compute::Type::of<luisa::float3>(), compute::xir::ArithmeticOp::BINARY_MUL,
                                      {_read_global_field(2u, compute::Type::of<luisa::float3>()),
                                       const_float3(luisa::float3{0.25f, 0.5f, 1.0f})});
            }
            return const_int(1);
        }
        if (attr_name == "custom:scale") {
            out_value = {.val = const_float(2.25f)};
            return const_int(1);
        }
        if (attr_name == "geom:name") {
            out_value = {.val = const_uint(intern_string("test_mesh"))};
            return const_int(1);
        }
        return const_int(0);
    }

    [[nodiscard]] compute::xir::Value *emit_getmessage(
        luisa::optional<luisa::string_view> source,
        luisa::string_view msg_name,
        const compute::osl::Type *dst_type,
        int dst_array_len,
        bool want_deriv,
        compute::osl::DualValue &out_value) noexcept override {
        if (!source.has_value()) {
            return OSLTranslator::emit_getmessage(source, msg_name, dst_type, dst_array_len, want_deriv, out_value);
        }
        if (*source == "trace") {
            auto &b = xir_builder();
            auto *hit_i = b.load(compute::Type::of<int32_t>(), _trace_hit_ptr);
            if (msg_name == "hit") {
                out_value = {.val = hit_i};
                return const_int(1);
            }
            if (msg_name == "hitdist") {
                out_value = {.val = b.load(compute::Type::of<float>(), _trace_dist_ptr)};
                return hit_i;
            }
            if (msg_name == "P") {
                out_value = {.val = b.load(compute::Type::of<luisa::float3>(), _trace_p_ptr)};
                return hit_i;
            }
            if (msg_name == "N") {
                out_value = {.val = b.load(compute::Type::of<luisa::float3>(), _trace_n_ptr)};
                return hit_i;
            }
            return const_int(0);
        }
        report_error(luisa::format("TestOSLTranslator: unknown getmessage source '{}'.", *source));
        return nullptr;
    }

    // 4. Texture & gettextureinfo
    [[nodiscard]] compute::osl::DualValue emit_texture(
        compute::osl::TextureOp op,
        luisa::string_view filename,
        luisa::span<const compute::osl::DualValue> coords,
        luisa::span<const compute::osl::DualValue> explicit_derivs,
        const compute::osl::Type *dst_type,
        bool want_deriv,
        luisa::span<const compute::osl::NamedOption> opts) noexcept override {
        auto &b = xir_builder();
        auto *f_t = compute::Type::of<float>();
        auto *f3_t = compute::Type::of<luisa::float3>();
        auto *b_t = compute::Type::of<bool>();

        luisa::string_view wrap = "periodic";
        const compute::osl::NamedOption *alpha_opt = nullptr;
        for (const auto &o : opts) {
            if (o.key == "wrap" && o.constant.has_value()) {
                if (auto *sv = luisa::get_if<luisa::string_view>(&*o.constant)) { wrap = *sv; }
            } else if (o.key == "alpha" && o.is_output) {
                alpha_opt = &o;
            }
        }

        auto apply_wrap = [&](compute::xir::Value *x, compute::xir::Value **in_range_flag) -> compute::xir::Value * {
            if (wrap == "clamp") {
                if (in_range_flag) { *in_range_flag = const_bool(true); }
                return b.call(f_t, compute::xir::ArithmeticOp::CLAMP, {x, const_float(0.0f), const_float(1.0f)});
            }
            if (wrap == "black") {
                auto *ge0 = b.call(b_t, compute::xir::ArithmeticOp::BINARY_GREATER_EQUAL, {x, const_float(0.0f)});
                auto *le1 = b.call(b_t, compute::xir::ArithmeticOp::BINARY_LESS_EQUAL, {x, const_float(1.0f)});
                if (in_range_flag) {
                    *in_range_flag = b.call(b_t, compute::xir::ArithmeticOp::BINARY_BIT_AND, {ge0, le1});
                }
                return x;
            }
            if (in_range_flag) { *in_range_flag = const_bool(true); }
            return b.call(f_t, compute::xir::ArithmeticOp::FRACT, {x});
        };

        compute::osl::DualValue result{};
        compute::xir::Value *alpha_v = const_float(1.0f);

        if (op == compute::osl::TextureOp::TEXTURE_2D && coords.size() == 2u) {
            compute::xir::Value *ok_u = nullptr, *ok_v = nullptr;
            compute::xir::Value *wu = apply_wrap(coords[0].val, &ok_u);
            compute::xir::Value *wv = apply_wrap(coords[1].val, &ok_v);
            compute::xir::Value *in_bounds = b.call(b_t, compute::xir::ArithmeticOp::BINARY_BIT_AND, {ok_u, ok_v});

            if (filename == "ramp.tx") {
                // Analytical ramp(s, t) = (0.5*s + 0.25*t, 0.25*s + 0.75*t, 0.1*s + 0.1*t)
                auto *r = b.call(f_t, compute::xir::ArithmeticOp::BINARY_ADD,
                                 {b.call(f_t, compute::xir::ArithmeticOp::BINARY_MUL, {wu, const_float(0.5f)}),
                                  b.call(f_t, compute::xir::ArithmeticOp::BINARY_MUL, {wv, const_float(0.25f)})});
                auto *g = b.call(f_t, compute::xir::ArithmeticOp::BINARY_ADD,
                                 {b.call(f_t, compute::xir::ArithmeticOp::BINARY_MUL, {wu, const_float(0.25f)}),
                                  b.call(f_t, compute::xir::ArithmeticOp::BINARY_MUL, {wv, const_float(0.75f)})});
                auto *bl = b.call(f_t, compute::xir::ArithmeticOp::BINARY_ADD,
                                  {b.call(f_t, compute::xir::ArithmeticOp::BINARY_MUL, {wu, const_float(0.1f)}),
                                   b.call(f_t, compute::xir::ArithmeticOp::BINARY_MUL, {wv, const_float(0.1f)})});
                auto *rgb = b.call(f3_t, compute::xir::ArithmeticOp::AGGREGATE, {r, g, bl});
                auto *b3 = b.call(compute::Type::of<luisa::bool3>(), compute::xir::ArithmeticOp::AGGREGATE,
                                  {in_bounds, in_bounds, in_bounds});
                result.val = b.call(f3_t, compute::xir::ArithmeticOp::SELECT,
                                    {const_float3(luisa::float3{0.0f}), rgb, b3});
                alpha_v = b.call(f_t, compute::xir::ArithmeticOp::SELECT,
                                 {const_float(0.0f),
                                  b.call(f_t, compute::xir::ArithmeticOp::BINARY_MUL, {wu, wv}),
                                  in_bounds});
                if (want_deriv) {
                    compute::xir::Value *dsdx = explicit_derivs.size() == 4u ? explicit_derivs[0].val : coords[0].dx;
                    compute::xir::Value *dtdx = explicit_derivs.size() == 4u ? explicit_derivs[1].val : coords[1].dx;
                    compute::xir::Value *dsdy = explicit_derivs.size() == 4u ? explicit_derivs[2].val : coords[0].dy;
                    compute::xir::Value *dtdy = explicit_derivs.size() == 4u ? explicit_derivs[3].val : coords[1].dy;
                    auto make_dr = [&](compute::xir::Value *ds, compute::xir::Value *dt) -> compute::xir::Value * {
                        if (!ds && !dt) { return nullptr; }
                        auto *s = ds ? ds : const_float(0.0f);
                        auto *t = dt ? dt : const_float(0.0f);
                        auto *dr = b.call(f_t, compute::xir::ArithmeticOp::BINARY_ADD,
                                          {b.call(f_t, compute::xir::ArithmeticOp::BINARY_MUL, {s, const_float(0.5f)}),
                                           b.call(f_t, compute::xir::ArithmeticOp::BINARY_MUL, {t, const_float(0.25f)})});
                        auto *dg = b.call(f_t, compute::xir::ArithmeticOp::BINARY_ADD,
                                          {b.call(f_t, compute::xir::ArithmeticOp::BINARY_MUL, {s, const_float(0.25f)}),
                                           b.call(f_t, compute::xir::ArithmeticOp::BINARY_MUL, {t, const_float(0.75f)})});
                        auto *db = b.call(f_t, compute::xir::ArithmeticOp::BINARY_ADD,
                                          {b.call(f_t, compute::xir::ArithmeticOp::BINARY_MUL, {s, const_float(0.1f)}),
                                           b.call(f_t, compute::xir::ArithmeticOp::BINARY_MUL, {t, const_float(0.1f)})});
                        return b.call(f3_t, compute::xir::ArithmeticOp::AGGREGATE, {dr, dg, db});
                    };
                    result.dx = make_dr(dsdx, dtdx);
                    result.dy = make_dr(dsdy, dtdy);
                }
            } else if (filename == "checker.tx") {
                auto *c = b.call(f_t, compute::xir::ArithmeticOp::BINARY_MUL, {wu, const_float(2.0f)});
                result.val = b.call(f3_t, compute::xir::ArithmeticOp::AGGREGATE, {c, c, c});
                alpha_v = const_float(0.75f);
            } else {
                report_error(luisa::format("TestOSLTranslator: unknown 2D texture '{}'.", filename));
                return {};
            }
        } else if (op == compute::osl::TextureOp::ENVIRONMENT && filename == "env.exr" && !coords.empty()) {
            auto *n = b.call(f3_t, compute::xir::ArithmeticOp::NORMALIZE, {coords[0].val});
            result.val = b.call(f3_t, compute::xir::ArithmeticOp::BINARY_ADD,
                                {b.call(f3_t, compute::xir::ArithmeticOp::BINARY_MUL, {n, const_float3(luisa::float3{0.5f})}),
                                 const_float3(luisa::float3{0.5f})});
        } else if (op == compute::osl::TextureOp::TEXTURE_3D && filename == "vol.vdb" && !coords.empty()) {
            result.val = b.call(f3_t, compute::xir::ArithmeticOp::BINARY_MUL,
                                {coords[0].val, const_float3(luisa::float3{0.5f})});
        } else {
            report_error(luisa::format("TestOSLTranslator: unsupported texture '{}' ({}).",
                                       filename, static_cast<uint32_t>(op)));
            return {};
        }

        if (alpha_opt != nullptr) {
            store_named_option_output(*alpha_opt, {.val = alpha_v});
        }
        if (dst_type != nullptr && dst_type->tag() == compute::osl::Type::Tag::SIMPLE &&
            static_cast<const compute::osl::SimpleType *>(dst_type)->primitive() == compute::osl::SimpleType::Primitive::FLOAT) {
            compute::osl::DualValue s;
            s.val = b.call(f_t, compute::xir::ArithmeticOp::EXTRACT, {result.val, const_uint(0u)});
            if (result.dx) { s.dx = b.call(f_t, compute::xir::ArithmeticOp::EXTRACT, {result.dx, const_uint(0u)}); }
            if (result.dy) { s.dy = b.call(f_t, compute::xir::ArithmeticOp::EXTRACT, {result.dy, const_uint(0u)}); }
            return s;
        }
        return result;
    }

    [[nodiscard]] compute::xir::Value *emit_gettextureinfo(
        luisa::string_view filename,
        luisa::string_view dataname,
        const compute::osl::Type *dst_type,
        int,
        compute::osl::DualValue &out_value) noexcept override {
        if (filename != "ramp.tx" && filename != "checker.tx") {
            return const_int(0);
        }
        if (dataname == "exists") {
            out_value = {.val = const_int(1)};
            return const_int(1);
        }
        if (dataname == "channels") {
            out_value = {.val = const_int(filename == "ramp.tx" ? 4 : 3)};
            return const_int(1);
        }
        if (dataname == "resolution" && dst_type->tag() == compute::osl::Type::Tag::SIMPLE &&
            static_cast<const compute::osl::SimpleType *>(dst_type)->primitive() == compute::osl::SimpleType::Primitive::FLOAT) {
            out_value = {.val = const_float(512.0f)};
            return const_int(1);
        }
        return const_int(0);
    }

    // 5. Coordinate spaces
    [[nodiscard]] compute::xir::Value *emit_getmatrix(
        luisa::string_view from_space,
        luisa::string_view to_space,
        compute::xir::Value *&out_matrix) noexcept override {
        auto space_to_common = [&](luisa::string_view sp, luisa::float4x4 &m) -> bool {
            if (sp == "common" || sp == "world") {
                m = luisa::make_float4x4(1.0f);
                return true;
            }
            if (sp == "object") {
                m = object_to_common_xir();
                return true;
            }
            if (sp == "shader") {
                m = shader_to_common_xir();
                return true;
            }
            if (sp == "camera") {
                m = camera_to_common_xir();
                return true;
            }
            return false;
        };
        luisa::float4x4 mf = luisa::make_float4x4(1.0f), mt = luisa::make_float4x4(1.0f);
        if (!space_to_common(from_space, mf) || !space_to_common(to_space, mt)) {
            out_matrix = const_float4x4(luisa::make_float4x4(0.0f));
            return const_int(0);
        }
        auto &b = xir_builder();
        auto *mf_v = const_float4x4(mf);
        auto *mt_inv = b.call(compute::Type::of<luisa::float4x4>(),
                              compute::xir::ArithmeticOp::MATRIX_INVERSE, {const_float4x4(mt)});
        // XIR stores L = M^T; M_{from->to} = M_from * M_to^{-1} -> L = L_to^{-1} * L_from
        out_matrix = b.call(compute::Type::of<luisa::float4x4>(),
                            compute::xir::ArithmeticOp::MATRIX_LINALG_MUL, {mt_inv, mf_v});
        return const_int(1);
    }

    // 8. Ray query
    [[nodiscard]] compute::xir::Value *emit_trace(
        const compute::osl::DualValue &pos,
        const compute::osl::DualValue &dir,
        luisa::span<const compute::osl::NamedOption> opts) noexcept override {
        auto &b = xir_builder();
        auto *f_t = compute::Type::of<float>();
        auto *f3_t = compute::Type::of<luisa::float3>();

        float mindist = 0.0f;
        float maxdist = 100.0f;
        for (const auto &o : opts) {
            if (!o.is_output && o.key == "mindist" && o.constant.has_value()) {
                if (auto *pf = luisa::get_if<float>(&*o.constant)) { mindist = *pf; }
            }
            if (!o.is_output && o.key == "maxdist" && o.constant.has_value()) {
                if (auto *pf = luisa::get_if<float>(&*o.constant)) { maxdist = *pf; }
            }
        }

        // Plane z = 0 intersection along dir: t = -pos.z / dir.z
        auto *pz = b.call(f_t, compute::xir::ArithmeticOp::EXTRACT, {pos.val, const_uint(2u)});
        auto *dz = b.call(f_t, compute::xir::ArithmeticOp::EXTRACT, {dir.val, const_uint(2u)});
        auto *dz_nz = b.call(compute::Type::of<bool>(), compute::xir::ArithmeticOp::BINARY_NOT_EQUAL, {dz, const_float(0.0f)});
        auto *safe_dz = b.call(f_t, compute::xir::ArithmeticOp::SELECT, {const_float(1.0f), dz, dz_nz});
        auto *t_raw = b.call(f_t, compute::xir::ArithmeticOp::BINARY_DIV,
                             {b.call(f_t, compute::xir::ArithmeticOp::UNARY_MINUS, {pz}), safe_dz});
        auto *ge_min = b.call(compute::Type::of<bool>(), compute::xir::ArithmeticOp::BINARY_GREATER, {t_raw, const_float(mindist)});
        auto *le_max = b.call(compute::Type::of<bool>(), compute::xir::ArithmeticOp::BINARY_LESS, {t_raw, const_float(maxdist)});
        auto *hit_b = b.call(compute::Type::of<bool>(), compute::xir::ArithmeticOp::BINARY_BIT_AND,
                             {dz_nz, b.call(compute::Type::of<bool>(), compute::xir::ArithmeticOp::BINARY_BIT_AND, {ge_min, le_max})});
        auto *hit_i = b.call(compute::Type::of<int32_t>(), compute::xir::ArithmeticOp::SELECT, {const_int(0), const_int(1), hit_b});
        auto *t3 = b.call(f3_t, compute::xir::ArithmeticOp::AGGREGATE, {t_raw, t_raw, t_raw});
        auto *hit_p = b.call(f3_t, compute::xir::ArithmeticOp::BINARY_ADD,
                             {pos.val, b.call(f3_t, compute::xir::ArithmeticOp::BINARY_MUL, {dir.val, t3})});

        b.store(_trace_hit_ptr, hit_i);
        b.store(_trace_dist_ptr, b.call(f_t, compute::xir::ArithmeticOp::SELECT, {const_float(0.0f), t_raw, hit_b}));
        b.store(_trace_p_ptr, hit_p);
        b.store(_trace_n_ptr, const_float3(luisa::float3{0.0f, 0.0f, 1.0f}));
        return hit_i;
    }
};

// CyclesSmokeTranslator: tolerates Cycles BSDFs, arbitrary textures/attributes/spaces/globals for .oso smoke verification
class CyclesSmokeTranslator final : public TestOSLTranslator {
protected:
    [[nodiscard]] size_t closure_positional_arg_count(
        luisa::string_view name,
        luisa::span<const compute::osl::Type *const> arg_types) const noexcept override {
        const size_t n = arg_types.size();
        auto min_clamp = [&](size_t v) { return std::min<size_t>(n, v); };
        size_t known = 0u;
        bool matched = true;
        if (name == "generalized_schlick_bsdf") { known = min_clamp(10u); }
        else if (name == "conductor_bsdf") { known = min_clamp(7u); }
        else if (name == "dielectric_bsdf") { known = min_clamp(6u); }
        else if (name == "microfacet_f82_tint") { known = min_clamp(7u); }
        else if (name == "microfacet_multi_ggx_aniso" || name == "microfacet_multi_ggx_glass") { known = min_clamp(5u); }
        else if (name == "microfacet_multi_ggx") { known = min_clamp(4u); }
        else if (name == "sheen_bsdf") { known = min_clamp(3u); }
        else if (name == "principled_sheen" || name == "principled_diffuse" ||
                 name == "oren_nayar_diffuse_bsdf" || name == "burley_diffuse_bsdf" ||
                 name == "ashikhmin_velvet") { known = min_clamp(2u); }
        else if (name == "diffuse_toon" || name == "specular_toon") { known = min_clamp(3u); }
        else if (name == "diffuse_bsdf" || name == "henyey_greenstein") { known = min_clamp(1u); }
        else if (name == "absorption" || name == "transparent_bsdf" || name == "holdout") { known = 0u; }
        else if (name == "principled_hair") { known = min_clamp(8u); }
        else if (name == "hair_bsdf") { known = min_clamp(5u); }
        else { matched = false; }
        if (!matched) {
            return TestOSLTranslator::closure_positional_arg_count(name, arg_types);
        }
        size_t opt_start = n;
        while (opt_start >= known + 2u &&
               arg_types[opt_start - 2u]->tag() == compute::osl::Type::Tag::SIMPLE &&
               static_cast<const compute::osl::SimpleType *>(arg_types[opt_start - 2u])->primitive() ==
                   compute::osl::SimpleType::Primitive::STRING) {
            opt_start -= 2u;
        }
        return opt_start;
    }

    [[nodiscard]] compute::osl::ClosureValue emit_closure(
        luisa::string_view name,
        luisa::span<const compute::osl::HookArg> args,
        luisa::span<const compute::osl::NamedOption> opts) noexcept override {
        if (name == "diffuse" || name == "oren_nayar" || name == "microfacet" ||
            name == "emission" || name == "transparent" || name == "reflection" || name == "background") {
            return TestOSLTranslator::emit_closure(name, args, opts);
        }
        luisa::compute::xir::Value *a0 = const_float3(luisa::float3{0.0f});
        for (const auto &a : args) {
            if (a.value.valid() && a.value.val->type() == compute::Type::of<luisa::float3>()) {
                a0 = a.value.val;
                break;
            }
        }
        auto *ct = compute::Type::of<TestClosure>();
        return {.value = xir_builder().call(
                    ct, compute::xir::ArithmeticOp::AGGREGATE,
                    {const_float3(luisa::float3{1.0f}), a0, const_uint(99u),
                     const_float(0.0f), const_float(0.0f), const_float(0.0f)})};
    }

    [[nodiscard]] compute::xir::Value *emit_getmessage(
        luisa::optional<luisa::string_view> source,
        luisa::string_view msg_name,
        const compute::osl::Type *dst_type,
        int dst_array_len,
        bool want_deriv,
        compute::osl::DualValue &out_value) noexcept override {
        if (!source.has_value() || *source == "trace") {
            return TestOSLTranslator::emit_getmessage(source, msg_name, dst_type, dst_array_len, want_deriv, out_value);
        }
        return const_int(0);
    }

    [[nodiscard]] compute::osl::DualValue emit_texture(
        compute::osl::TextureOp,
        luisa::string_view,
        luisa::span<const compute::osl::DualValue>,
        luisa::span<const compute::osl::DualValue>,
        const compute::osl::Type *dst_type,
        bool want_deriv,
        luisa::span<const compute::osl::NamedOption>) noexcept override {
        auto zero_for = [&](const compute::osl::Type *t) -> compute::xir::Value * {
            if (t && t->tag() == compute::osl::Type::Tag::SIMPLE &&
                static_cast<const compute::osl::SimpleType *>(t)->primitive() ==
                    compute::osl::SimpleType::Primitive::FLOAT) {
                return const_float(0.0f);
            }
            return const_float3(luisa::float3{0.0f});
        };
        compute::osl::DualValue dv{.val = zero_for(dst_type)};
        if (want_deriv) {
            dv.dx = zero_for(dst_type);
            dv.dy = zero_for(dst_type);
        }
        return dv;
    }
};

inline int g_osl_argc{0};
inline const char **g_osl_argv{nullptr};

inline void set_osl_test_args(int argc, char **argv) noexcept {
    g_osl_argc = argc;
    g_osl_argv = const_cast<const char **>(argv);
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#ifdef _DEBUG
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
#endif
}

[[nodiscard]] inline compute::Device create_osl_test_device(compute::Context &ctx) {
    if (g_osl_argc > 1 && g_osl_argv != nullptr && g_osl_argv[1] != nullptr && g_osl_argv[1][0] != '\0') {
        LUISA_INFO("OSL test device requested: {}", g_osl_argv[1]);
        return ctx.create_device(g_osl_argv[1]);
    }
    auto installed = ctx.installed_backends();
    for (auto preferred : {"fallback", "simd", "vk", "dx"}) {
        for (const auto &b : installed) {
            if (b == preferred) {
                LUISA_INFO("OSL test device auto-selected: {}", b);
                return ctx.create_device(b);
            }
        }
    }
    return ctx.create_default_device();
}

[[nodiscard]] inline compute::Device &get_osl_test_device() {
    struct CachedTestDevice {
        luisa::unique_ptr<compute::Context> ctx;
        luisa::optional<compute::Device> device;
    };
    static CachedTestDevice cached;
    if (!cached.device.has_value()) {
        const char *exe = (g_osl_argc > 0 && g_osl_argv && g_osl_argv[0]) ? g_osl_argv[0] : "";
        cached.ctx = luisa::make_unique<compute::Context>(exe);
        cached.device.emplace(create_osl_test_device(*cached.ctx));
        std::atexit([]() noexcept {
            cached.device.reset();
            cached.ctx.reset();
        });
    }
    return *cached.device;
}

struct ExecutedOSLGroup {
    compute::osl::TranslationResult translation;
    bool ast_ok{false};
    bool shader_compile_ok{false};
    luisa::vector<TestShaderOutput> outputs;
};

[[nodiscard]] inline ExecutedOSLGroup compile_and_run_with_translator(
    compute::osl::OSLTranslator &translator,
    const compute::osl::ShaderGroup &group,
    luisa::vector<TestShaderGlobals> inputs,
    const compute::osl::TranslatorOptions &custom_opts = {}) {

    ExecutedOSLGroup res;
    compute::xir::Module xir_mod;
    auto *kernel = xir_mod.create_kernel();

    auto opts = custom_opts;
    opts.verify_xir = true;
    opts.run_xir_passes = true;
    opts.restructure_for_ast = true;

    res.translation = translator.translate(group, kernel, opts);
    if (!res.translation.succeeded()) {
        for (const auto &err : res.translation.errors) {
            LUISA_WARNING("OSL translate error [{}:pc={} op='{}']: {}",
                          err.layer_name, err.instruction_index, err.opcode, err.message);
        }
        return res;
    }

    auto post_ver = compute::xir::xir_verify_module(
        &xir_mod, {.require_no_phi = true, .require_canonical_break_continue_targets = true});
    if (!post_ver.succeeded()) {
        for (const auto &e : post_ver.errors) {
            LUISA_WARNING("XIR post-verify error: {}", e.message);
            res.translation.errors.emplace_back(compute::osl::TranslationError{.message = e.message});
        }
        return res;
    }

    auto ast = compute::xir::xir_to_ast_translate(*kernel, {.strict = true});
    if (ast == nullptr) {
        LUISA_WARNING("xir_to_ast_translate returned nullptr.");
        return res;
    }
    res.ast_ok = true;

    compute::Device &device = get_osl_test_device();
    compute::Shader1D<compute::Buffer<TestShaderGlobals>, compute::Buffer<TestShaderOutput>> shader{
        device.impl(), ast->function(), compute::ShaderOption{}};
    res.shader_compile_ok = shader.compile_ok();
    if (!res.shader_compile_ok) { return res; }

    const size_t n = std::max<size_t>(1u, inputs.size());
    if (inputs.empty()) { inputs.emplace_back(); }

    auto in_buf = device.create_buffer<TestShaderGlobals>(n);
    auto out_buf = device.create_buffer<TestShaderOutput>(n);
    res.outputs.resize(n);

    auto stream = device.create_stream();
    stream << in_buf.copy_from(inputs.data())
           << shader(in_buf, out_buf).dispatch(static_cast<uint>(n))
           << out_buf.copy_to(res.outputs.data())
           << compute::synchronize();
    return res;
}

[[nodiscard]] inline ExecutedOSLGroup compile_and_run_osl_group(
    const compute::osl::ShaderGroup &group,
    luisa::vector<TestShaderGlobals> inputs,
    const compute::osl::TranslatorOptions &custom_opts = {}) {
    TestOSLTranslator translator;
    return compile_and_run_with_translator(translator, group, std::move(inputs), custom_opts);
}

[[nodiscard]] inline luisa::unique_ptr<compute::osl::Shader> parse_oso_source(luisa::string_view oso_text) {
    return compute::osl::OSOParser::parse(oso_text);
}

[[nodiscard]] inline ExecutedOSLGroup compile_and_run_single_with_translator(
    compute::osl::OSLTranslator &translator,
    luisa::string_view oso_text,
    luisa::vector<TestShaderGlobals> inputs,
    luisa::vector<luisa::string_view> output_names = {},
    const compute::osl::TranslatorOptions &opts = {}) {
    auto shader = parse_oso_source(oso_text);
    if (!shader) { return {}; }
    compute::osl::ShaderGroup g;
    uint32_t main_layer = g.add_layer("main", shader.get());
    if (output_names.empty()) {
        for (auto &&sym : shader->symbols()) {
            if (sym->tag() == compute::osl::Symbol::Tag::SYM_OUTPUT_PARAM && sym->is_root()) {
                g.mark_output(main_layer, sym->identifier(), false);
            }
        }
    } else {
        for (auto name : output_names) {
            g.mark_output(main_layer, name, false);
        }
    }
    return compile_and_run_with_translator(translator, g, std::move(inputs), opts);
}

[[nodiscard]] inline ExecutedOSLGroup compile_and_run_single_oso(
    luisa::string_view oso_text,
    luisa::vector<TestShaderGlobals> inputs,
    luisa::vector<luisa::string_view> output_names = {},
    const compute::osl::TranslatorOptions &opts = {}) {
    auto shader = parse_oso_source(oso_text);
    if (!shader) { return {}; }
    compute::osl::ShaderGroup g;
    uint32_t main_layer = g.add_layer("main", shader.get());
    if (output_names.empty()) {
        for (auto &&sym : shader->symbols()) {
            if (sym->tag() == compute::osl::Symbol::Tag::SYM_OUTPUT_PARAM && sym->is_root()) {
                g.mark_output(main_layer, sym->identifier(), false);
            }
        }
    } else {
        for (auto name : output_names) {
            g.mark_output(main_layer, name, false);
        }
    }
    return compile_and_run_osl_group(g, std::move(inputs), opts);
}

}// namespace luisa::test::osl_test
