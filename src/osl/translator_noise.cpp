#include <algorithm>
#include <cmath>
#include <utility>

#include <luisa/ast/type_registry.h>
#include <luisa/core/stl/format.h>
#include "translator_impl.h"

namespace luisa::compute::osl {

namespace {

// Deterministic pure-XIR procedural noise helpers (1D..4D)

struct NoiseBuilder {
    OSLTranslator::Impl &impl;

    [[nodiscard]] xir::Value *i(int32_t v) noexcept { return impl.const_int(v); }
    [[nodiscard]] xir::Value *u(uint32_t v) noexcept { return impl.const_uint(v); }
    [[nodiscard]] xir::Value *f(float v) noexcept { return impl.const_float(v); }

    [[nodiscard]] xir::Value *bin(xir::ArithmeticOp op, xir::Value *a, xir::Value *b) noexcept {
        return impl.builder.call(a->type(), op, {a, b});
    }
    [[nodiscard]] xir::Value *un(xir::ArithmeticOp op, xir::Value *a) noexcept {
        return impl.builder.call(a->type(), op, {a});
    }
    [[nodiscard]] xir::Value *cmp(xir::ArithmeticOp op, xir::Value *a, xir::Value *b) noexcept {
        return impl.builder.call(compute::Type::of<bool>(), op, {a, b});
    }
    [[nodiscard]] xir::Value *sel(xir::Value *fv, xir::Value *tv, xir::Value *c) noexcept {
        return impl.builder.call(fv->type(), xir::ArithmeticOp::SELECT, {fv, tv, c});
    }
    [[nodiscard]] xir::Value *i2u(xir::Value *v) noexcept {
        return impl.builder.static_cast_(compute::Type::of<uint32_t>(), v);
    }
    [[nodiscard]] xir::Value *u2f(xir::Value *v) noexcept {
        return impl.builder.static_cast_(compute::Type::of<float>(), v);
    }

    // Compact single-pass 32-bit avalanche mix (bounded AST DAG fanout)
    [[nodiscard]] xir::Value *hash_u32(xir::Value *x) noexcept {
        x = bin(xir::ArithmeticOp::BINARY_BIT_XOR, x, bin(xir::ArithmeticOp::BINARY_SHIFT_RIGHT, x, u(16u)));
        x = bin(xir::ArithmeticOp::BINARY_MUL, x, u(0x7feb352du));
        x = bin(xir::ArithmeticOp::BINARY_BIT_XOR, x, bin(xir::ArithmeticOp::BINARY_SHIFT_RIGHT, x, u(15u)));
        return bin(xir::ArithmeticOp::BINARY_MUL, x, u(0x846ca68bu));
    }

    [[nodiscard]] xir::Value *hash_nd(luisa::span<xir::Value *const> coords, uint32_t seed) noexcept {
        xir::Value *h = u(0x9e3779b9u ^ seed);
        static constexpr uint32_t primes[4] = {0x85ebca6bu, 0xc2b2ae35u, 0x27d4eb2fu, 0x165667b1u};
        for (size_t k = 0u; k < coords.size(); ++k) {
            xir::Value *uk = bin(xir::ArithmeticOp::BINARY_MUL, i2u(coords[k]), u(primes[k & 3u]));
            h = bin(xir::ArithmeticOp::BINARY_BIT_XOR, h, uk);
        }
        return hash_u32(h);
    }

    // Map uint32 hash to [0, 1)
    [[nodiscard]] xir::Value *u2unit(xir::Value *h) noexcept {
        xir::Value *top24 = bin(xir::ArithmeticOp::BINARY_SHIFT_RIGHT, h, u(8u));
        return bin(xir::ArithmeticOp::BINARY_MUL, u2f(top24), f(1.0f / 16777216.0f));
    }

    // Map uint32 hash to [-1, 1)
    [[nodiscard]] xir::Value *u2signed(xir::Value *h) noexcept {
        return bin(xir::ArithmeticOp::BINARY_SUB,
                   bin(xir::ArithmeticOp::BINARY_MUL, u2unit(h), f(2.0f)), f(1.0f));
    }

    // Wrap integer coord modulo positive integer period (if periodic)
    [[nodiscard]] xir::Value *wrap_i(xir::Value *ix, xir::Value *period) noexcept {
        if (period == nullptr) { return ix; }
        xir::Value *p_i = impl.builder.static_cast_(compute::Type::of<int32_t>(),
                                                    un(xir::ArithmeticOp::FLOOR, period));
        xir::Value *p_pos = cmp(xir::ArithmeticOp::BINARY_GREATER, p_i, i(0));
        xir::Value *safe_p = sel(i(1), p_i, p_pos);
        xir::Value *m = bin(xir::ArithmeticOp::BINARY_MOD, ix, safe_p);
        xir::Value *neg = cmp(xir::ArithmeticOp::BINARY_LESS, m, i(0));
        m = sel(m, bin(xir::ArithmeticOp::BINARY_ADD, m, safe_p), neg);
        return sel(ix, m, p_pos);
    }

    // Smooth quintic Perlin gradient noise in N dims (1..4) with analytic gradient
    [[nodiscard]] std::pair<xir::Value *, luisa::vector<xir::Value *>>
    perlin_nd(luisa::span<xir::Value *const> coords,
              luisa::span<xir::Value *const> periods,
              uint32_t seed, bool need_grad) noexcept {
        const size_t dim = coords.size();
        luisa::vector<xir::Value *> i0(dim), i1(dim), tx(dim), fade(dim), inv_fade(dim), dfade(dim);
        for (size_t d = 0u; d < dim; ++d) {
            xir::Value *fl = un(xir::ArithmeticOp::FLOOR, coords[d]);
            xir::Value *ix0 = impl.builder.static_cast_(compute::Type::of<int32_t>(), fl);
            xir::Value *ix1 = bin(xir::ArithmeticOp::BINARY_ADD, ix0, i(1));
            xir::Value *per = d < periods.size() ? periods[d] : nullptr;
            i0[d] = wrap_i(ix0, per);
            i1[d] = wrap_i(ix1, per);
            xir::Value *t = bin(xir::ArithmeticOp::BINARY_SUB, coords[d], fl);
            tx[d] = t;
            // Quintic fade: t^3 * (t * (t * 6 - 15) + 10)
            xir::Value *t2 = bin(xir::ArithmeticOp::BINARY_MUL, t, t);
            xir::Value *t3 = bin(xir::ArithmeticOp::BINARY_MUL, t2, t);
            xir::Value *poly = bin(xir::ArithmeticOp::BINARY_ADD,
                                   bin(xir::ArithmeticOp::BINARY_MUL, t,
                                       bin(xir::ArithmeticOp::BINARY_SUB,
                                           bin(xir::ArithmeticOp::BINARY_MUL, t, f(6.0f)), f(15.0f))),
                                   f(10.0f));
            fade[d] = bin(xir::ArithmeticOp::BINARY_MUL, t3, poly);
            inv_fade[d] = bin(xir::ArithmeticOp::BINARY_SUB, f(1.0f), fade[d]);
            if (need_grad) {
                // dfade/dt = 30 * t^2 * (t - 1)^2
                xir::Value *tm1 = bin(xir::ArithmeticOp::BINARY_SUB, t, f(1.0f));
                dfade[d] = bin(xir::ArithmeticOp::BINARY_MUL, f(30.0f),
                               bin(xir::ArithmeticOp::BINARY_MUL, t2, bin(xir::ArithmeticOp::BINARY_MUL, tm1, tm1)));
            }
        }

        const size_t corners = size_t{1u} << dim;
        luisa::vector<xir::Value *> corner_vals(corners);
        luisa::vector<luisa::vector<xir::Value *>> corner_grads(corners);

        static constexpr uint32_t grad_primes[4] = {0x9e3779b1u, 0x243f6a89u, 0xb7e15163u, 0x6a09e667u};
        luisa::vector<xir::Value *> ic(dim);
        for (size_t mask = 0u; mask < corners; ++mask) {
            for (size_t d = 0u; d < dim; ++d) {
                ic[d] = (mask & (size_t{1u} << d)) ? i1[d] : i0[d];
            }
            xir::Value *h = hash_nd(ic, seed);
            xir::Value *dot_v = f(0.0f);
            if (need_grad) { corner_grads[mask].resize(dim); }
            // Derive pseudo-random gradient components linearly from single corner hash (zero exponential AST reuse)
            float scale = dim == 1u ? 1.5f : (dim == 2u ? 1.15f : 1.0f);
            for (size_t d = 0u; d < dim; ++d) {
                xir::Value *hd = bin(xir::ArithmeticOp::BINARY_MUL, h, u(grad_primes[d & 3u]));
                xir::Value *gd = bin(xir::ArithmeticOp::BINARY_MUL, u2signed(hd), f(scale));
                if (need_grad) { corner_grads[mask][d] = gd; }
                xir::Value *delta = (mask & (size_t{1u} << d)) ?
                                        bin(xir::ArithmeticOp::BINARY_SUB, tx[d], f(1.0f)) :
                                        tx[d];
                dot_v = bin(xir::ArithmeticOp::BINARY_ADD, dot_v, bin(xir::ArithmeticOp::BINARY_MUL, gd, delta));
            }
            corner_vals[mask] = dot_v;
        }

        // Multilinear interpolation using (1 - fade) * a + fade * b so heavy corner expressions are referenced once
        struct Node {
            xir::Value *val;
            luisa::vector<xir::Value *> grad;
        };
        luisa::vector<Node> cur(corners);
        for (size_t m = 0u; m < corners; ++m) {
            cur[m].val = corner_vals[m];
            if (need_grad) { cur[m].grad = corner_grads[m]; }
        }
        for (size_t d = 0u; d < dim; ++d) {
            size_t half = cur.size() >> 1u;
            luisa::vector<Node> next(half);
            for (size_t j = 0u; j < half; ++j) {
                const auto &a = cur[2u * j];
                const auto &b = cur[2u * j + 1u];
                next[j].val = bin(xir::ArithmeticOp::BINARY_ADD,
                                  bin(xir::ArithmeticOp::BINARY_MUL, inv_fade[d], a.val),
                                  bin(xir::ArithmeticOp::BINARY_MUL, fade[d], b.val));
                if (need_grad) {
                    next[j].grad.resize(dim);
                    xir::Value *diff = bin(xir::ArithmeticOp::BINARY_SUB, b.val, a.val);
                    for (size_t g = 0u; g < dim; ++g) {
                        xir::Value *dg = bin(xir::ArithmeticOp::BINARY_ADD,
                                             bin(xir::ArithmeticOp::BINARY_MUL, inv_fade[d], a.grad[g]),
                                             bin(xir::ArithmeticOp::BINARY_MUL, fade[d], b.grad[g]));
                        if (g == d) {
                            dg = bin(xir::ArithmeticOp::BINARY_ADD, dg,
                                     bin(xir::ArithmeticOp::BINARY_MUL, dfade[d], diff));
                        }
                        next[j].grad[g] = dg;
                    }
                }
            }
            cur = std::move(next);
        }
        return {cur[0].val, std::move(cur[0].grad)};
    }
};

}// namespace

DualValue OSLTranslator::Impl::emit_default_noise(
    luisa::string_view noise_type, NoiseOp op,
    luisa::span<const DualValue> coords,
    luisa::span<const DualValue> periods,
    const Type *dst_type, bool want_deriv,
    luisa::span<const NamedOption>) noexcept {

    // Validate noise type
    enum struct Mode { PERLIN, UPERLIN, SIMPLEX, USIMPLEX, CELL, HASH };
    Mode mode = Mode::UPERLIN;
    if (op == NoiseOp::CELLNOISE || noise_type == "cell") {
        mode = Mode::CELL;
    } else if (op == NoiseOp::HASHNOISE || noise_type == "hash") {
        mode = Mode::HASH;
    } else if (noise_type == "perlin" || noise_type == "snoise" ||
               (op == NoiseOp::SNOISE && noise_type.empty()) ||
               (op == NoiseOp::PSNOISE && noise_type.empty())) {
        mode = Mode::PERLIN;
    } else if (noise_type == "uperlin" || noise_type == "noise" || noise_type.empty()) {
        mode = Mode::UPERLIN;
    } else if (noise_type == "simplex") {
        mode = Mode::SIMPLEX;
    } else if (noise_type == "usimplex") {
        mode = Mode::USIMPLEX;
    } else {
        report_error(luisa::format("Unknown OSL noise type '{}'.", noise_type));
        return {};
    }

    // Flatten coords and periods into 1..4 scalar DualValues
    luisa::vector<DualValue> c_scalars;
    for (const auto &c : coords) {
        if (c.val->type() == compute::Type::of<luisa::float3>()) {
            for (uint32_t k = 0u; k < 3u; ++k) {
                c_scalars.emplace_back(DualValue{
                    .val = extract_vec_comp(c.val, k),
                    .dx = (want_deriv && c.dx) ? extract_vec_comp(c.dx, k) : nullptr,
                    .dy = (want_deriv && c.dy) ? extract_vec_comp(c.dy, k) : nullptr});
            }
        } else {
            c_scalars.emplace_back(DualValue{
                .val = to_float(c.val),
                .dx = want_deriv ? c.dx : nullptr,
                .dy = want_deriv ? c.dy : nullptr});
        }
    }
    luisa::vector<xir::Value *> p_scalars;
    for (const auto &p : periods) {
        if (p.val->type() == compute::Type::of<luisa::float3>()) {
            for (uint32_t k = 0u; k < 3u; ++k) {
                p_scalars.emplace_back(extract_vec_comp(p.val, k));
            }
        } else {
            p_scalars.emplace_back(to_float(p.val));
        }
    }
    if (c_scalars.empty() || c_scalars.size() > 4u) {
        report_error("Unsupported noise coordinate dimensionality.");
        return {};
    }

    NoiseBuilder nb{*this};
    bool is_triple = detail::is_triple_type(dst_type);
    uint32_t channels = is_triple ? 3u : 1u;

    luisa::vector<xir::Value *> c_vals(c_scalars.size());
    bool any_input_deriv = false;
    for (size_t k = 0u; k < c_scalars.size(); ++k) {
        c_vals[k] = c_scalars[k].val;
        any_input_deriv |= c_scalars[k].has_derivs();
    }
    bool compute_grad = want_deriv && any_input_deriv &&
                        (mode != Mode::CELL && mode != Mode::HASH);

    auto eval_channel = [&](uint32_t ch) -> DualValue {
        uint32_t seed_base = ch * 0x4f1bbcdbu;
        if (mode == Mode::CELL || mode == Mode::HASH) {
            luisa::vector<xir::Value *> keys(c_vals.size());
            for (size_t d = 0u; d < c_vals.size(); ++d) {
                if (mode == Mode::CELL) {
                    xir::Value *ix = builder.static_cast_(
                        compute::Type::of<int32_t>(),
                        nb.un(xir::ArithmeticOp::FLOOR, c_vals[d]));
                    keys[d] = (d < p_scalars.size()) ? nb.wrap_i(ix, p_scalars[d]) : ix;
                } else {
                    // Exact bit-cast float -> int key for hashnoise
                    keys[d] = builder.bit_cast_(compute::Type::of<int32_t>(), c_vals[d]);
                }
            }
            return {.val = nb.u2unit(nb.hash_nd(keys, seed_base))};
        }

        uint32_t noise_seed = seed_base ^ ((mode == Mode::SIMPLEX || mode == Mode::USIMPLEX) ? 0x1234567u : 0u);
        auto [sval, grad] = nb.perlin_nd(c_vals, p_scalars, noise_seed, compute_grad);
        bool is_unsigned = (mode == Mode::UPERLIN || mode == Mode::USIMPLEX);
        xir::Value *val = is_unsigned ?
                              nb.bin(xir::ArithmeticOp::BINARY_ADD,
                                     nb.bin(xir::ArithmeticOp::BINARY_MUL, sval, const_float(0.5f)),
                                     const_float(0.5f)) :
                              sval;
        if (!compute_grad) { return {.val = val}; }

        float gscale = is_unsigned ? 0.5f : 1.0f;
        xir::Value *dx_sum = nullptr;
        xir::Value *dy_sum = nullptr;
        for (size_t d = 0u; d < c_scalars.size(); ++d) {
            xir::Value *gd = gscale == 1.0f ? grad[d] : nb.bin(xir::ArithmeticOp::BINARY_MUL, grad[d], const_float(gscale));
            if (c_scalars[d].dx) {
                xir::Value *term = nb.bin(xir::ArithmeticOp::BINARY_MUL, gd, c_scalars[d].dx);
                dx_sum = dx_sum ? nb.bin(xir::ArithmeticOp::BINARY_ADD, dx_sum, term) : term;
            }
            if (c_scalars[d].dy) {
                xir::Value *term = nb.bin(xir::ArithmeticOp::BINARY_MUL, gd, c_scalars[d].dy);
                dy_sum = dy_sum ? nb.bin(xir::ArithmeticOp::BINARY_ADD, dy_sum, term) : term;
            }
        }
        return {.val = val, .dx = dx_sum, .dy = dy_sum};
    };

    if (!is_triple) {
        return eval_channel(0u);
    }
    DualValue c0 = eval_channel(0u);
    DualValue c1 = eval_channel(1u);
    DualValue c2 = eval_channel(2u);
    auto *f3_t = compute::Type::of<luisa::float3>();
    DualValue out;
    out.val = builder.call(f3_t, xir::ArithmeticOp::AGGREGATE, {c0.val, c1.val, c2.val});
    if (compute_grad && (c0.has_derivs() || c1.has_derivs() || c2.has_derivs())) {
        auto dv = [&](xir::Value *v) { return v ? v : const_float(0.0f); };
        out.dx = builder.call(f3_t, xir::ArithmeticOp::AGGREGATE, {dv(c0.dx), dv(c1.dx), dv(c2.dx)});
        out.dy = builder.call(f3_t, xir::ArithmeticOp::AGGREGATE, {dv(c0.dy), dv(c1.dy), dv(c2.dy)});
    }
    return out;
}

}// namespace luisa::compute::osl
