// OCR engine -- rotary position embedding operator implementations (Phase 2).
#include "ocr/ops/ops_rope.h"

#include <luisa/dsl/sugar.h>

#include "ocr/ops/ops_common.h"
#include "ocr/quantization.h"
#include "ocr/runtime/shader_cache.h"

namespace luisa::ocr {

using namespace luisa::compute;

namespace {

// One thread per (token, head). Loops pairs in thread. `Interleaved` is a
// compile-time host bool selecting the pair layout.
template<bool Interleaved>
void rope_body(const BufferFloat &x, const BufferFloat &out, const BufferInt &positions, const BufferUInt &axis_of_pair,
               Var<uint> tokens, Var<uint> heads, Var<uint> head_dim, Var<uint> axes,
               Var<float> theta) noexcept {
    const auto row = ocr_linear_id();
    const auto total = tokens * heads;
    $if (row < total) {
        const auto t = row / heads;
        const auto pairs = head_dim / 2u;
        const auto base = row * head_dim;
        $for (d, pairs) {
            const auto axis = axis_of_pair.read(d);
            $if (axis < axes) {
                const auto pos = cast<float>(positions.read(t * axes + axis));
                // inv_freq = theta^(-2d/head_dim) = exp2(log2(theta) * -2d/head_dim).
                const auto inv_freq = exp2(log2(theta) * (-2.0f * cast<float>(d) / cast<float>(head_dim)));
                const auto angle = pos * inv_freq;
                const auto c = cos(angle);
                const auto s = sin(angle);
                Var<uint> i0 = def(0u);
                Var<uint> i1 = def(0u);
                if constexpr (Interleaved) {
                    i0 = d;
                    i1 = d + pairs;
                } else {
                    i0 = 2u * d;
                    i1 = 2u * d + 1u;
                }
                const auto a = x.read(base + i0);
                const auto b = x.read(base + i1);
                out.write(base + i0, a * c - b * s);
                out.write(base + i1, a * s + b * c);
            };
        };
    };
}

template<bool Interleaved>
[[nodiscard]] RopeAxialKernel make_rope_axial_impl() noexcept {
    RopeAxialKernel kernel = [](BufferFloat x, BufferFloat out, BufferInt positions, BufferUInt axis_of_pair,
                                Var<uint> tokens, Var<uint> heads, Var<uint> head_dim, Var<uint> axes,
                                Var<float> theta) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_rope");
        rope_body<Interleaved>(x, out, positions, axis_of_pair, tokens, heads, head_dim, axes, theta);
    };
    return kernel;
}

}// namespace

[[nodiscard]] RopeAxialKernel make_rope_1d_kernel() noexcept { return make_rope_axial_impl<false>(); }
[[nodiscard]] RopeAxialKernel make_rope_rotate_half_kernel() noexcept { return make_rope_axial_impl<true>(); }
[[nodiscard]] RopeAxialKernel make_rope_2d_kernel() noexcept { return make_rope_axial_impl<false>(); }
[[nodiscard]] RopeAxialKernel make_rope_mrope_kernel() noexcept { return make_rope_axial_impl<false>(); }
[[nodiscard]] RopeAxialKernel make_rope_interleaved_mrope_kernel() noexcept { return make_rope_axial_impl<true>(); }
[[nodiscard]] RopeAxialKernel make_rope_xd_kernel() noexcept { return make_rope_axial_impl<false>(); }

[[nodiscard]] SinusoidalTableKernel make_sinusoidal_table_kernel() noexcept {
    SinusoidalTableKernel kernel = [](BufferFloat out, Var<uint> len, Var<uint> dim, Var<float> theta) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_sinusoidal_table");
        const auto p = ocr_linear_id();
        $if (p < len) {
            const auto pairs = dim / 2u;
            $for (d, pairs) {
                const auto inv_freq = exp2(log2(theta) * (-2.0f * cast<float>(d) / cast<float>(dim)));
                const auto angle = cast<float>(p) * inv_freq;
                out.write(p * dim + 2u * d, sin(angle));
                out.write(p * dim + 2u * d + 1u, cos(angle));
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<LaunchPlan1D> plan_rope_dispatch(uint64_t tokens, uint64_t heads) {
    uint64_t rows = 0;
    if (!checked_mul(tokens, heads, rows)) {
        return Result<LaunchPlan1D>::failure(StatusCode::invalid_argument, "rope row count overflow", "tokens*heads");
    }
    return plan_threads(rows, 256u);
}

[[nodiscard]] Result<LaunchPlan1D> plan_sinusoidal_dispatch(uint64_t len) {
    return plan_threads(len, 256u);
}

[[nodiscard]] luisa::string rope_variant_key(luisa::string_view variant) {
    return make_shader_key("rope", variant);
}

}// namespace luisa::ocr