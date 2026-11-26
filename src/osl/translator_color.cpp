#include <algorithm>
#include <cmath>

#include <luisa/ast/type_registry.h>
#include <luisa/core/stl/format.h>
#include "translator_impl.h"

namespace luisa::compute::osl {

namespace {

// Lightweight dual-number helper built on OSLTranslator::Impl
struct DualOps {
    OSLTranslator::Impl &impl;
    bool deriv{false};

    [[nodiscard]] xir::Value *f(float v) noexcept { return impl.const_float(v); }
    [[nodiscard]] xir::Value *f3(float x, float y, float z) noexcept {
        return impl.const_float3(luisa::float3{x, y, z});
    }

    [[nodiscard]] DualValue make(xir::Value *v, xir::Value *dx = nullptr, xir::Value *dy = nullptr) noexcept {
        return {.val = v, .dx = deriv ? dx : nullptr, .dy = deriv ? dy : nullptr};
    }

    [[nodiscard]] xir::Value *bin(xir::ArithmeticOp op, xir::Value *a, xir::Value *b) noexcept {
        return impl.builder.call(a->type(), op, {a, b});
    }
    [[nodiscard]] xir::Value *un(xir::ArithmeticOp op, xir::Value *a) noexcept {
        return impl.builder.call(a->type(), op, {a});
    }
    [[nodiscard]] xir::Value *cmp(xir::ArithmeticOp op, xir::Value *a, xir::Value *b) noexcept {
        return impl.builder.call(compute::Type::of<bool>(), op, {a, b});
    }
    [[nodiscard]] xir::Value *sel(xir::Value *fv, xir::Value *tv, xir::Value *cond) noexcept {
        return impl.builder.call(fv->type(), xir::ArithmeticOp::SELECT, {fv, tv, cond});
    }

    [[nodiscard]] DualValue add(const DualValue &a, const DualValue &b) noexcept {
        xir::Value *v = bin(xir::ArithmeticOp::BINARY_ADD, a.val, b.val);
        if (!deriv) { return {.val = v}; }
        auto add_d = [&](xir::Value *da, xir::Value *db) -> xir::Value * {
            if (da && db) { return bin(xir::ArithmeticOp::BINARY_ADD, da, db); }
            return da ? da : db;
        };
        return {.val = v, .dx = add_d(a.dx, b.dx), .dy = add_d(a.dy, b.dy)};
    }

    [[nodiscard]] DualValue sub(const DualValue &a, const DualValue &b) noexcept {
        xir::Value *v = bin(xir::ArithmeticOp::BINARY_SUB, a.val, b.val);
        if (!deriv) { return {.val = v}; }
        auto sub_d = [&](xir::Value *da, xir::Value *db) -> xir::Value * {
            if (da && db) { return bin(xir::ArithmeticOp::BINARY_SUB, da, db); }
            if (da) { return da; }
            if (db) { return un(xir::ArithmeticOp::UNARY_MINUS, db); }
            return nullptr;
        };
        return {.val = v, .dx = sub_d(a.dx, b.dx), .dy = sub_d(a.dy, b.dy)};
    }

    [[nodiscard]] DualValue mul_c(const DualValue &a, float c) noexcept {
        xir::Value *vc = f(c);
        xir::Value *v = bin(xir::ArithmeticOp::BINARY_MUL, a.val, vc);
        if (!deriv) { return {.val = v}; }
        return {.val = v,
                .dx = a.dx ? bin(xir::ArithmeticOp::BINARY_MUL, a.dx, vc) : nullptr,
                .dy = a.dy ? bin(xir::ArithmeticOp::BINARY_MUL, a.dy, vc) : nullptr};
    }

    [[nodiscard]] DualValue mul(const DualValue &a, const DualValue &b) noexcept {
        xir::Value *v = bin(xir::ArithmeticOp::BINARY_MUL, a.val, b.val);
        if (!deriv) { return {.val = v}; }
        auto mul_d = [&](xir::Value *da, xir::Value *db) -> xir::Value * {
            xir::Value *t1 = da ? bin(xir::ArithmeticOp::BINARY_MUL, da, b.val) : nullptr;
            xir::Value *t2 = db ? bin(xir::ArithmeticOp::BINARY_MUL, a.val, db) : nullptr;
            if (t1 && t2) { return bin(xir::ArithmeticOp::BINARY_ADD, t1, t2); }
            return t1 ? t1 : t2;
        };
        return {.val = v, .dx = mul_d(a.dx, b.dx), .dy = mul_d(a.dy, b.dy)};
    }

    [[nodiscard]] DualValue div(const DualValue &a, const DualValue &b) noexcept {
        xir::Value *safe_b = sel(b.val, f(1.0f), cmp(xir::ArithmeticOp::BINARY_EQUAL, b.val, f(0.0f)));
        xir::Value *v = sel(f(0.0f), bin(xir::ArithmeticOp::BINARY_DIV, a.val, safe_b),
                            cmp(xir::ArithmeticOp::BINARY_NOT_EQUAL, b.val, f(0.0f)));
        if (!deriv) { return {.val = v}; }
        xir::Value *b2 = bin(xir::ArithmeticOp::BINARY_MUL, safe_b, safe_b);
        xir::Value *nz = cmp(xir::ArithmeticOp::BINARY_NOT_EQUAL, b.val, f(0.0f));
        auto div_d = [&](xir::Value *da, xir::Value *db) -> xir::Value * {
            if (!da && !db) { return nullptr; }
            xir::Value *num = nullptr;
            xir::Value *t1 = da ? bin(xir::ArithmeticOp::BINARY_MUL, da, safe_b) : nullptr;
            xir::Value *t2 = db ? bin(xir::ArithmeticOp::BINARY_MUL, a.val, db) : nullptr;
            if (t1 && t2) { num = bin(xir::ArithmeticOp::BINARY_SUB, t1, t2); }
            else if (t1) { num = t1; }
            else { num = un(xir::ArithmeticOp::UNARY_MINUS, t2); }
            return sel(f(0.0f), bin(xir::ArithmeticOp::BINARY_DIV, num, b2), nz);
        };
        return {.val = v, .dx = div_d(a.dx, b.dx), .dy = div_d(a.dy, b.dy)};
    }

    [[nodiscard]] DualValue comp(const DualValue &v3, uint32_t idx) noexcept {
        return {.val = impl.extract_vec_comp(v3.val, idx),
                .dx = (deriv && v3.dx) ? impl.extract_vec_comp(v3.dx, idx) : nullptr,
                .dy = (deriv && v3.dy) ? impl.extract_vec_comp(v3.dy, idx) : nullptr};
    }

    [[nodiscard]] DualValue make3(const DualValue &x, const DualValue &y, const DualValue &z) noexcept {
        auto *f3_t = compute::Type::of<luisa::float3>();
        xir::Value *val = impl.builder.call(f3_t, xir::ArithmeticOp::AGGREGATE, {x.val, y.val, z.val});
        if (!deriv || (!x.has_derivs() && !y.has_derivs() && !z.has_derivs())) {
            return {.val = val};
        }
        auto dval = [&](xir::Value *v) { return v ? v : f(0.0f); };
        xir::Value *dx = impl.builder.call(f3_t, xir::ArithmeticOp::AGGREGATE, {dval(x.dx), dval(y.dx), dval(z.dx)});
        xir::Value *dy = impl.builder.call(f3_t, xir::ArithmeticOp::AGGREGATE, {dval(x.dy), dval(y.dy), dval(z.dy)});
        return {.val = val, .dx = dx, .dy = dy};
    }

    [[nodiscard]] DualValue select(const DualValue &fv, const DualValue &tv, xir::Value *cond) noexcept {
        xir::Value *v = sel(fv.val, tv.val, cond);
        if (!deriv || (!fv.has_derivs() && !tv.has_derivs())) {
            return {.val = v};
        }
        auto *zero = impl.zero_of(v->type());
        xir::Value *dx = sel(fv.dx ? fv.dx : zero, tv.dx ? tv.dx : zero, cond);
        xir::Value *dy = sel(fv.dy ? fv.dy : zero, tv.dy ? tv.dy : zero, cond);
        return {.val = v, .dx = dx, .dy = dy};
    }

    [[nodiscard]] DualValue pow_c(const DualValue &base, float exp_val) noexcept {
        xir::Value *pos = cmp(xir::ArithmeticOp::BINARY_GREATER, base.val, f(0.0f));
        xir::Value *safe = sel(f(1.0f), base.val, pos);
        xir::Value *p = bin(xir::ArithmeticOp::POW, safe, f(exp_val));
        xir::Value *v = sel(f(0.0f), p, pos);
        if (!deriv || !base.has_derivs()) { return {.val = v}; }
        xir::Value *dp = bin(xir::ArithmeticOp::BINARY_MUL, f(exp_val),
                             bin(xir::ArithmeticOp::POW, safe, f(exp_val - 1.0f)));
        dp = sel(f(0.0f), dp, pos);
        return {.val = v,
                .dx = base.dx ? bin(xir::ArithmeticOp::BINARY_MUL, dp, base.dx) : nullptr,
                .dy = base.dy ? bin(xir::ArithmeticOp::BINARY_MUL, dp, base.dy) : nullptr};
    }

    [[nodiscard]] DualValue fabs_v(const DualValue &x) noexcept {
        xir::Value *v = un(xir::ArithmeticOp::ABS, x.val);
        if (!deriv || !x.has_derivs()) { return {.val = v}; }
        xir::Value *neg = cmp(xir::ArithmeticOp::BINARY_LESS, x.val, f(0.0f));
        auto sgn_mul = [&](xir::Value *d) -> xir::Value * {
            if (!d) { return nullptr; }
            return sel(d, un(xir::ArithmeticOp::UNARY_MINUS, d), neg);
        };
        return {.val = v, .dx = sgn_mul(x.dx), .dy = sgn_mul(x.dy)};
    }
};

[[nodiscard]] luisa::string_view canonical_colorspace(luisa::string_view s) noexcept {
    if (s == "rgb" || s == "RGB" || s == "linear" || s == "Rec709") { return "rgb"; }
    if (s == "sRGB") { return "sRGB"; }
    if (s == "hsv" || s == "HSV") { return "hsv"; }
    if (s == "hsl" || s == "HSL") { return "hsl"; }
    if (s == "XYZ" || s == "CIE_XYZ") { return "XYZ"; }
    if (s == "xyY") { return "xyY"; }
    if (s == "YIQ") { return "YIQ"; }
    return {};
}

[[nodiscard]] DualValue rgb_matrix3(DualOps &d, const DualValue &c, const float (&m)[9]) noexcept {
    DualValue r = d.comp(c, 0u);
    DualValue g = d.comp(c, 1u);
    DualValue b = d.comp(c, 2u);
    DualValue o0 = d.add(d.add(d.mul_c(r, m[0]), d.mul_c(g, m[1])), d.mul_c(b, m[2]));
    DualValue o1 = d.add(d.add(d.mul_c(r, m[3]), d.mul_c(g, m[4])), d.mul_c(b, m[5]));
    DualValue o2 = d.add(d.add(d.mul_c(r, m[6]), d.mul_c(g, m[7])), d.mul_c(b, m[8]));
    return d.make3(o0, o1, o2);
}

[[nodiscard]] DualValue rgb_to_hsv(DualOps &d, const DualValue &rgb) noexcept {
    DualValue r = d.comp(rgb, 0u);
    DualValue g = d.comp(rgb, 1u);
    DualValue b = d.comp(rgb, 2u);

    xir::Value *rg_max = d.cmp(xir::ArithmeticOp::BINARY_GREATER, r.val, g.val);
    DualValue max_rg = d.select(g, r, rg_max);
    DualValue max_v = d.select(b, max_rg, d.cmp(xir::ArithmeticOp::BINARY_GREATER, max_rg.val, b.val));

    DualValue min_rg = d.select(r, g, rg_max);
    DualValue min_v = d.select(b, min_rg, d.cmp(xir::ArithmeticOp::BINARY_LESS, min_rg.val, b.val));

    DualValue delta = d.sub(max_v, min_v);
    DualValue zero = d.make(d.f(0.0f));
    xir::Value *has_max = d.cmp(xir::ArithmeticOp::BINARY_GREATER, max_v.val, d.f(0.0f));
    DualValue s = d.select(zero, d.div(delta, max_v), has_max);

    xir::Value *has_delta = d.cmp(xir::ArithmeticOp::BINARY_GREATER, delta.val, d.f(0.0f));
    DualValue h_r = d.div(d.sub(g, b), delta);
    DualValue h_g = d.add(d.make(d.f(2.0f)), d.div(d.sub(b, r), delta));
    DualValue h_b = d.add(d.make(d.f(4.0f)), d.div(d.sub(r, g), delta));
    DualValue h = d.select(h_b, h_g, d.cmp(xir::ArithmeticOp::BINARY_EQUAL, max_v.val, g.val));
    h = d.select(h, h_r, d.cmp(xir::ArithmeticOp::BINARY_EQUAL, max_v.val, r.val));
    h = d.mul_c(h, 1.0f / 6.0f);
    h = d.select(h, d.add(h, d.make(d.f(1.0f))), d.cmp(xir::ArithmeticOp::BINARY_LESS, h.val, d.f(0.0f)));
    h = d.select(zero, h, has_delta);
    return d.make3(h, s, max_v);
}

[[nodiscard]] DualValue hsv_to_rgb(DualOps &d, const DualValue &hsv) noexcept {
    DualValue h = d.comp(hsv, 0u);
    DualValue s = d.comp(hsv, 1u);
    DualValue v = d.comp(hsv, 2u);
    // Hue wrap: fract(h) * 6
    xir::Value *hf = d.un(xir::ArithmeticOp::FRACT, h.val);
    DualValue h6 = d.make(d.bin(xir::ArithmeticOp::BINARY_MUL, hf, d.f(6.0f)),
                          (d.deriv && h.dx) ? d.bin(xir::ArithmeticOp::BINARY_MUL, h.dx, d.f(6.0f)) : nullptr,
                          (d.deriv && h.dy) ? d.bin(xir::ArithmeticOp::BINARY_MUL, h.dy, d.f(6.0f)) : nullptr);
    DualValue one = d.make(d.f(1.0f));
    DualValue two = d.make(d.f(2.0f));
    DualValue three = d.make(d.f(3.0f));
    DualValue four = d.make(d.f(4.0f));

    auto clamp01 = [&](const DualValue &x) -> DualValue {
        DualValue z = d.make(d.f(0.0f));
        DualValue c = d.select(x, z, d.cmp(xir::ArithmeticOp::BINARY_LESS, x.val, d.f(0.0f)));
        return d.select(c, one, d.cmp(xir::ArithmeticOp::BINARY_GREATER, c.val, d.f(1.0f)));
    };
    DualValue kr = clamp01(d.sub(d.fabs_v(d.sub(h6, three)), one));
    DualValue kg = clamp01(d.sub(two, d.fabs_v(d.sub(h6, two))));
    DualValue kb = clamp01(d.sub(two, d.fabs_v(d.sub(h6, four))));

    DualValue one_minus_s = d.sub(one, s);
    DualValue r = d.mul(v, d.add(one_minus_s, d.mul(s, kr)));
    DualValue g = d.mul(v, d.add(one_minus_s, d.mul(s, kg)));
    DualValue b = d.mul(v, d.add(one_minus_s, d.mul(s, kb)));
    return d.make3(r, g, b);
}

[[nodiscard]] DualValue rgb_to_hsl(DualOps &d, const DualValue &rgb) noexcept {
    DualValue hsv = rgb_to_hsv(d, rgb);
    DualValue h = d.comp(hsv, 0u);
    DualValue s_hsv = d.comp(hsv, 1u);
    DualValue v = d.comp(hsv, 2u);
    DualValue one = d.make(d.f(1.0f));
    DualValue two = d.make(d.f(2.0f));
    DualValue l = d.mul_c(d.mul(v, d.sub(two, s_hsv)), 0.5f);
    DualValue denom = d.select(d.sub(one, l), l, d.cmp(xir::ArithmeticOp::BINARY_LESS, l.val, d.f(0.5f)));
    denom = d.mul_c(denom, 2.0f);
    xir::Value *pos = d.cmp(xir::ArithmeticOp::BINARY_GREATER, denom.val, d.f(0.0f));
    DualValue s_hsl = d.select(d.make(d.f(0.0f)), d.div(d.sub(v, l), denom), pos);
    return d.make3(h, s_hsl, l);
}

[[nodiscard]] DualValue hsl_to_rgb(DualOps &d, const DualValue &hsl) noexcept {
    DualValue h = d.comp(hsl, 0u);
    DualValue s = d.comp(hsl, 1u);
    DualValue l = d.comp(hsl, 2u);
    DualValue one = d.make(d.f(1.0f));
    DualValue min_l = d.select(d.sub(one, l), l, d.cmp(xir::ArithmeticOp::BINARY_LESS, l.val, d.f(0.5f)));
    DualValue v = d.add(l, d.mul(s, min_l));
    xir::Value *pos = d.cmp(xir::ArithmeticOp::BINARY_GREATER, v.val, d.f(0.0f));
    DualValue s_hsv = d.select(d.make(d.f(0.0f)), d.mul_c(d.div(d.sub(v, l), v), 2.0f), pos);
    return hsv_to_rgb(d, d.make3(h, s_hsv, v));
}

}// namespace

DualValue OSLTranslator::Impl::emit_default_color_transform(
    luisa::string_view from_space, luisa::string_view to_space,
    const DualValue &value, bool want_deriv) noexcept {
    auto from_c = canonical_colorspace(from_space);
    auto to_c = canonical_colorspace(to_space);
    if (from_c.empty()) {
        report_error(luisa::format("Unknown source color space '{}'.", from_space));
        return {};
    }
    if (to_c.empty()) {
        report_error(luisa::format("Unknown target color space '{}'.", to_space));
        return {};
    }
    if (from_c == to_c) {
        return want_deriv ? value : DualValue{.val = value.val};
    }

    DualOps d{*this, want_deriv && value.has_derivs()};

    static constexpr float xyz_to_rgb_m[9] = {
        3.2404542f, -1.5371385f, -0.4985314f,
        -0.9692660f, 1.8760108f, 0.0415560f,
        0.0556434f, -0.2040259f, 1.0572252f};
    static constexpr float rgb_to_xyz_m[9] = {
        0.4124564f, 0.3575761f, 0.1804375f,
        0.2126729f, 0.7151522f, 0.0721750f,
        0.0193339f, 0.1191920f, 0.9503041f};
    static constexpr float rgb_to_yiq_m[9] = {
        0.2990f, 0.5870f, 0.1140f,
        0.595716f, -0.274453f, -0.321263f,
        0.211456f, -0.522591f, 0.311135f};
    static constexpr float yiq_to_rgb_m[9] = {
        1.0f, 0.9562957f, 0.6210244f,
        1.0f, -0.2721221f, -0.6473806f,
        1.0f, -1.1069890f, 1.7046150f};

    auto srgb_comp_to_lin = [&](const DualValue &c) -> DualValue {
        xir::Value *lo_cond = d.cmp(xir::ArithmeticOp::BINARY_LESS_EQUAL, c.val, d.f(0.04045f));
        DualValue lo = d.mul_c(c, 1.0f / 12.92f);
        DualValue hi = d.pow_c(d.mul_c(d.add(c, d.make(d.f(0.055f))), 1.0f / 1.055f), 2.4f);
        return d.select(hi, lo, lo_cond);
    };
    auto lin_comp_to_srgb = [&](const DualValue &c) -> DualValue {
        xir::Value *lo_cond = d.cmp(xir::ArithmeticOp::BINARY_LESS_EQUAL, c.val, d.f(0.0031308f));
        DualValue lo = d.mul_c(c, 12.92f);
        DualValue hi = d.sub(d.mul_c(d.pow_c(c, 1.0f / 2.4f), 1.055f), d.make(d.f(0.055f)));
        return d.select(hi, lo, lo_cond);
    };

    // Convert from_c -> rgb
    DualValue rgb = value;
    if (from_c == "sRGB") {
        rgb = d.make3(srgb_comp_to_lin(d.comp(value, 0u)),
                      srgb_comp_to_lin(d.comp(value, 1u)),
                      srgb_comp_to_lin(d.comp(value, 2u)));
    } else if (from_c == "hsv") {
        rgb = hsv_to_rgb(d, value);
    } else if (from_c == "hsl") {
        rgb = hsl_to_rgb(d, value);
    } else if (from_c == "XYZ") {
        rgb = rgb_matrix3(d, value, xyz_to_rgb_m);
    } else if (from_c == "xyY") {
        DualValue x = d.comp(value, 0u);
        DualValue y = d.comp(value, 1u);
        DualValue Y = d.comp(value, 2u);
        xir::Value *y_pos = d.cmp(xir::ArithmeticOp::BINARY_GREATER, y.val, d.f(0.0f));
        DualValue zero = d.make(d.f(0.0f));
        DualValue X = d.select(zero, d.div(d.mul(x, Y), y), y_pos);
        DualValue z_part = d.sub(d.sub(d.make(d.f(1.0f)), x), y);
        DualValue Z = d.select(zero, d.div(d.mul(z_part, Y), y), y_pos);
        rgb = rgb_matrix3(d, d.make3(X, Y, Z), xyz_to_rgb_m);
    } else if (from_c == "YIQ") {
        rgb = rgb_matrix3(d, value, yiq_to_rgb_m);
    }

    // Convert rgb -> to_c
    if (to_c == "rgb") { return rgb; }
    if (to_c == "sRGB") {
        return d.make3(lin_comp_to_srgb(d.comp(rgb, 0u)),
                       lin_comp_to_srgb(d.comp(rgb, 1u)),
                       lin_comp_to_srgb(d.comp(rgb, 2u)));
    }
    if (to_c == "hsv") { return rgb_to_hsv(d, rgb); }
    if (to_c == "hsl") { return rgb_to_hsl(d, rgb); }
    if (to_c == "XYZ") { return rgb_matrix3(d, rgb, rgb_to_xyz_m); }
    if (to_c == "xyY") {
        DualValue xyz = rgb_matrix3(d, rgb, rgb_to_xyz_m);
        DualValue X = d.comp(xyz, 0u);
        DualValue Y = d.comp(xyz, 1u);
        DualValue Z = d.comp(xyz, 2u);
        DualValue sum = d.add(d.add(X, Y), Z);
        xir::Value *pos = d.cmp(xir::ArithmeticOp::BINARY_GREATER, sum.val, d.f(0.0f));
        DualValue zero = d.make(d.f(0.0f));
        DualValue x = d.select(zero, d.div(X, sum), pos);
        DualValue y = d.select(zero, d.div(Y, sum), pos);
        return d.make3(x, y, Y);
    }
    if (to_c == "YIQ") { return rgb_matrix3(d, rgb, rgb_to_yiq_m); }
    return rgb;
}

DualValue OSLTranslator::Impl::emit_luminance(const DualValue &c, bool want_deriv) noexcept {
    xir::Value *w = const_float3(luisa::float3{0.2126729f, 0.7151522f, 0.0721750f});
    xir::Value *v = builder.call(compute::Type::of<float>(), xir::ArithmeticOp::DOT, {c.val, w});
    if (!want_deriv || !c.has_derivs()) { return {.val = v}; }
    xir::Value *dx = c.dx ? builder.call(compute::Type::of<float>(), xir::ArithmeticOp::DOT, {c.dx, w}) : nullptr;
    xir::Value *dy = c.dy ? builder.call(compute::Type::of<float>(), xir::ArithmeticOp::DOT, {c.dy, w}) : nullptr;
    return {.val = v, .dx = dx, .dy = dy};
}

DualValue OSLTranslator::Impl::emit_blackbody(const DualValue &temp, bool want_deriv) noexcept {
    // Portable Planckian locus approximation in Rec.709 linear RGB (normalized at D65)
    DualOps d{*this, want_deriv && temp.has_derivs()};
    DualValue T = d.select(d.make(d.f(1000.0f)), temp,
                           d.cmp(xir::ArithmeticOp::BINARY_GREATER, temp.val, d.f(1000.0f)));
    DualValue invT = d.div(d.make(d.f(1000.0f)), T);
    DualValue invT2 = d.mul(invT, invT);
    DualValue invT3 = d.mul(invT2, invT);

    // Kang et al. 2002 CIE xy chromaticity fit
    DualValue xc_lo = d.add(d.add(d.mul_c(invT3, -0.2661239f), d.mul_c(invT2, -0.2343589f)),
                            d.add(d.mul_c(invT, 0.8776956f), d.make(d.f(0.179910f))));
    DualValue xc_hi = d.add(d.add(d.mul_c(invT3, -3.0258469f), d.mul_c(invT2, 2.1070379f)),
                            d.add(d.mul_c(invT, 0.2226347f), d.make(d.f(0.240390f))));
    DualValue xc = d.select(xc_hi, xc_lo, d.cmp(xir::ArithmeticOp::BINARY_LESS_EQUAL, T.val, d.f(4000.0f)));
    DualValue xc2 = d.mul(xc, xc);
    DualValue xc3 = d.mul(xc2, xc);

    DualValue yc_a = d.add(d.add(d.mul_c(xc3, -1.1063814f), d.mul_c(xc2, -1.34811020f)),
                           d.add(d.mul_c(xc, 2.18555832f), d.make(d.f(-0.20219683f))));
    DualValue yc_b = d.add(d.add(d.mul_c(xc3, -0.9549476f), d.mul_c(xc2, -1.37418593f)),
                           d.add(d.mul_c(xc, 2.09137015f), d.make(d.f(-0.16748867f))));
    DualValue yc_c = d.add(d.add(d.mul_c(xc3, 3.0817580f), d.mul_c(xc2, -5.87338670f)),
                           d.add(d.mul_c(xc, 3.75112997f), d.make(d.f(-0.37001483f))));
    DualValue yc = d.select(yc_c, yc_b, d.cmp(xir::ArithmeticOp::BINARY_LESS_EQUAL, T.val, d.f(4000.0f)));
    yc = d.select(yc, yc_a, d.cmp(xir::ArithmeticOp::BINARY_LESS_EQUAL, T.val, d.f(2222.0f)));

    DualValue xyy = d.make3(xc, yc, d.make(d.f(1.0f)));
    DualValue rgb = emit_default_color_transform("xyY", "rgb", xyy, want_deriv);
    // Zero if temp <= 0
    xir::Value *pos = d.cmp(xir::ArithmeticOp::BINARY_GREATER, temp.val, d.f(0.0f));
    return d.select(d.make(d.f3(0.0f, 0.0f, 0.0f)), rgb, pos);
}

DualValue OSLTranslator::Impl::emit_wavelength_color(const DualValue &lambda, bool want_deriv) noexcept {
    // Wyman et al. 2013 analytic piecewise Gaussian CIE 1931 XYZ -> Rec.709 RGB
    DualOps d{*this, want_deriv && lambda.has_derivs()};
    auto gauss = [&](float amp, float mu, float s1, float s2) -> DualValue {
        DualValue diff = d.sub(lambda, d.make(d.f(mu)));
        DualValue s = d.select(d.make(d.f(s1)), d.make(d.f(s2)),
                               d.cmp(xir::ArithmeticOp::BINARY_GREATER_EQUAL, diff.val, d.f(0.0f)));
        DualValue t = d.mul(diff, s);
        DualValue e = d.mul_c(d.mul(t, t), -0.5f);
        xir::Value *ev = d.bin(xir::ArithmeticOp::BINARY_MUL, d.f(amp), d.un(xir::ArithmeticOp::EXP, e.val));
        if (!d.deriv || !e.has_derivs()) { return {.val = ev}; }
        return {.val = ev,
                .dx = e.dx ? d.bin(xir::ArithmeticOp::BINARY_MUL, ev, e.dx) : nullptr,
                .dy = e.dy ? d.bin(xir::ArithmeticOp::BINARY_MUL, ev, e.dy) : nullptr};
    };
    DualValue X = d.add(d.add(gauss(1.056f, 599.8f, 0.0264f, 0.0323f),
                              gauss(0.362f, 442.0f, 0.0624f, 0.0374f)),
                        gauss(-0.065f, 501.1f, 0.0490f, 0.0382f));
    DualValue Y = d.add(gauss(0.821f, 568.8f, 0.0213f, 0.0247f),
                        gauss(0.286f, 530.9f, 0.0613f, 0.0322f));
    DualValue Z = d.add(gauss(1.217f, 437.0f, 0.0845f, 0.0278f),
                        gauss(0.681f, 459.0f, 0.0385f, 0.0725f));
    return emit_default_color_transform("XYZ", "rgb", d.make3(X, Y, Z), want_deriv);
}

}// namespace luisa::compute::osl
