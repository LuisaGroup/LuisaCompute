// OCR engine -- vision connector / composition operator implementations (Phase 2).
#include "ocr/ops/ops_vision.h"

#include <luisa/dsl/sugar.h>

#include "ocr/ops/ops_common.h"
#include "ocr/quantization.h"
#include "ocr/runtime/shader_cache.h"

namespace luisa::ocr {

using namespace luisa::compute;

[[nodiscard]] PixelUnshuffleKernel make_pixel_unshuffle_kernel() noexcept {
    PixelUnshuffleKernel kernel = [](BufferFloat x, BufferFloat out, Var<uint> n, Var<uint> c, Var<uint> h,
                                     Var<uint> w, Var<uint> r) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_pixel_unshuffle");
        const auto id = ocr_linear_id();
        $if (id < n * c * h * w) {
            const auto xx = id % w;
            const auto y = (id / w) % h;
            const auto ci = (id / (w * h)) % c;
            const auto ni = id / (w * h * c);
            const auto oh = h / r;
            const auto ow = w / r;
            const auto oc = c * r * r;
            const auto oy = y / r;
            const auto ox = xx / r;
            const auto ry = y % r;
            const auto rx = xx % r;
            const auto oc_idx = ci * r * r + ry * r + rx;
            out.write(((ni * oc + oc_idx) * oh + oy) * ow + ox, x.read(id));
        };
    };
    return kernel;
}

[[nodiscard]] SpatialMergeKernel make_spatial_merge_kernel() noexcept {
    SpatialMergeKernel kernel = [](BufferFloat x, BufferFloat out, Var<uint> T, Var<uint> H,
                                     Var<uint> tokens_w, Var<uint> m) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_spatial_merge");
        const auto id = ocr_linear_id();
        const auto mm = m * m;
        const auto out_tokens = T / mm;
        const auto out_hidden = H * mm;
        $if (id < out_tokens * out_hidden) {
            const auto e = id % out_hidden;
            const auto ot = id / out_hidden;
            const auto blocks_w = tokens_w / m;
            const auto bx = ot % blocks_w;
            const auto by = ot / blocks_w;
            const auto sub = e / H;       // which of the m*m tokens
            const auto d = e % H;         // hidden index
            const auto sy = sub / m;
            const auto sx = sub % m;
            const auto src_token = (by * m + sy) * tokens_w + (bx * m + sx);
            out.write(id, x.read(src_token * H + d));
        };
    };
    return kernel;
}

[[nodiscard]] AddPositionKernel make_add_position_kernel() noexcept {
    AddPositionKernel kernel = [](BufferFloat x, BufferFloat pos, BufferInt pos_ids, BufferFloat out,
                                  Var<uint> tokens, Var<uint> hidden) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_add_position");
        const auto id = ocr_linear_id();
        $if (id < tokens * hidden) {
            const auto t = id / hidden;
            const auto d = id % hidden;
            const auto p = cast<uint>(pos_ids.read(t));
            out.write(id, x.read(id) + pos.read(p * hidden + d));
        };
    };
    return kernel;
}

[[nodiscard]] DeepStackInjectKernel make_deepstack_inject_kernel() noexcept {
    DeepStackInjectKernel kernel = [](BufferFloat x, BufferFloat proj, BufferFloat out, Var<uint> tokens,
                                      Var<uint> hidden, Var<uint> start, Var<uint> count) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_deepstack_inject");
        const auto id = ocr_linear_id();
        $if (id < tokens * hidden) {
            const auto t = id / hidden;
            const auto d = id % hidden;
            Var<float> value = x.read(id);
            $if (t >= start & t < start + count) {
                value += proj.read((t - start) * hidden + d);
            };
            out.write(id, value);
        };
    };
    return kernel;
}

[[nodiscard]] Result<LaunchPlan1D> plan_pixel_unshuffle_dispatch(uint64_t n, uint64_t c, uint64_t h, uint64_t w) {
    uint64_t count = 0;
    if (!checked_mul(n, c, count) || !checked_mul(count, h, count) || !checked_mul(count, w, count)) {
        return Result<LaunchPlan1D>::failure(StatusCode::invalid_argument, "pixel unshuffle count overflow", "n*c*h*w");
    }
    return plan_threads(count, 256u);
}

[[nodiscard]] Result<LaunchPlan1D> plan_spatial_merge_dispatch(uint64_t out_tokens, uint64_t out_hidden) {
    uint64_t count = 0;
    if (!checked_mul(out_tokens, out_hidden, count)) {
        return Result<LaunchPlan1D>::failure(StatusCode::invalid_argument, "spatial merge count overflow", "out_tokens*out_hidden");
    }
    return plan_threads(count, 256u);
}

[[nodiscard]] Result<LaunchPlan1D> plan_add_position_dispatch(uint64_t tokens, uint64_t hidden) {
    uint64_t count = 0;
    if (!checked_mul(tokens, hidden, count)) {
        return Result<LaunchPlan1D>::failure(StatusCode::invalid_argument, "add position count overflow", "tokens*hidden");
    }
    return plan_threads(count, 256u);
}

[[nodiscard]] Result<LaunchPlan1D> plan_deepstack_inject_dispatch(uint64_t tokens, uint64_t hidden) {
    uint64_t count = 0;
    if (!checked_mul(tokens, hidden, count)) {
        return Result<LaunchPlan1D>::failure(StatusCode::invalid_argument, "deepstack count overflow", "tokens*hidden");
    }
    return plan_threads(count, 256u);
}

[[nodiscard]] luisa::string vision_variant_key(luisa::string_view op) {
    return make_shader_key("vision", op);
}

}// namespace luisa::ocr