// OCR engine -- CTC frame operator implementations (Phase 5).
#include "ocr/ops/ops_ctc.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <luisa/dsl/sugar.h>

#include "ocr/ops/ops_common.h"
#include "ocr/quantization.h"
#include "ocr/runtime/shader_cache.h"

namespace luisa::ocr {

using namespace luisa::compute;

[[nodiscard]] Status validate_ctc_log_softmax_inputs(uint64_t rows, uint64_t cols, float temperature) {
    if (rows == 0u || cols == 0u) {
        return Status::error(StatusCode::invalid_argument, "ctc log-softmax dimensions must be nonzero", "rows/cols");
    }
    if (!std::isfinite(temperature) || temperature <= 0.0f) {
        return Status::error(StatusCode::invalid_argument,
                             "ctc temperature must be finite and strictly positive", "temperature");
    }
    uint64_t count = 0;
    if (!checked_mul(rows, cols, count)) {
        return Status::error(StatusCode::invalid_argument, "ctc logit count overflow", "rows*cols");
    }
    return Status::success();
}

[[nodiscard]] Status validate_ctc_frame_inputs(uint64_t rows, uint64_t cols) {
    if (rows == 0u || cols == 0u) {
        return Status::error(StatusCode::invalid_argument, "ctc frame dimensions must be nonzero", "rows/cols");
    }
    uint64_t count = 0;
    if (!checked_mul(rows, cols, count)) {
        return Status::error(StatusCode::invalid_argument, "ctc logit count overflow", "rows*cols");
    }
    return Status::success();
}

[[nodiscard]] CtcLogSoftmaxKernel make_ctc_log_softmax_kernel() noexcept {
    CtcLogSoftmaxKernel kernel = [](BufferFloat logits, BufferFloat out, Var<uint> rows, Var<uint> cols,
                                    Var<float> temperature) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_ctc_log_softmax");
        const auto row = ocr_linear_id();
        $if (row < rows) {
            const auto base = row * cols;
            // Defensive: invalid temperature -> defined zero output, no NaN.
            $if (temperature > 0.0f) {
                Var<float> max_v = def(logits.read(base) / temperature);
                $for (c, 1u, cols) { max_v = max(max_v, logits.read(base + c) / temperature); };
                Var<float> sum = def(0.0f);
                $for (c, cols) { sum += exp(logits.read(base + c) / temperature - max_v); };
                const auto log_sum = log(sum);
                $for (c, cols) {
                    out.write(base + c, logits.read(base + c) / temperature - max_v - log_sum);
                };
            } $else {
                $for (c, cols) { out.write(base + c, 0.0f); };
            };
        };
    };
    return kernel;
}

[[nodiscard]] CtcArgmaxProbsKernel make_ctc_argmax_probs_kernel() noexcept {
    CtcArgmaxProbsKernel kernel = [](BufferFloat log_probs, BufferInt ids, BufferFloat probs,
                                     Var<uint> rows, Var<uint> cols) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_ctc_argmax_probs");
        const auto row = ocr_linear_id();
        $if (row < rows) {
            const auto base = row * cols;
            Var<uint> best = def(0u);
            Var<float> best_v = def(log_probs.read(base));
            $for (c, 1u, cols) {
                const auto v = log_probs.read(base + c);
                $if (v > best_v) {
                    best_v = v;
                    best = c;
                };
            };
            ids.write(row, cast<int>(best));
            probs.write(row, exp(best_v));
        };
    };
    return kernel;
}

[[nodiscard]] SigmoidKernel make_sigmoid_kernel() noexcept {
    SigmoidKernel kernel = [](BufferFloat x, BufferFloat out, Var<uint> count) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_sigmoid");
        const auto id = ocr_linear_id();
        $if (id < count) {
            out.write(id, ocr_sigmoid(x.read(id)));
        };
    };
    return kernel;
}

[[nodiscard]] HeightCollapseKernel make_height_collapse_kernel() noexcept {
    HeightCollapseKernel kernel = [](BufferFloat x, BufferFloat out, Var<uint> n, Var<uint> c,
                                     Var<uint> h, Var<uint> w, Var<uint> r) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_height_collapse");
        const auto id = ocr_linear_id();
        const auto hb = h / r;// remaining height blocks
        const auto frames = n * w * hb;
        const auto channels = c * r;
        $if (r > 0u & h % r == 0u & id < frames * channels) {
            const auto ch_out = id % channels;
            const auto frame = id / channels;
            const auto xcol = frame % w;
            const auto by = (frame / w) % hb;
            const auto ni = frame / (w * hb);
            const auto ry = ch_out / c;
            const auto ch = ch_out % c;
            const auto y = by * r + ry;
            const auto src = ((ni * c + ch) * h + y) * w + xcol;
            out.write(id, x.read(src));
        };
    };
    return kernel;
}

[[nodiscard]] CnnActKernel make_cnn_act_kernel(uint32_t act) noexcept {
    CnnActKernel kernel = [act](BufferFloat x, BufferFloat out, Var<uint> count) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_cnn_act");
        const auto id = ocr_linear_id();
        $if (id < count) {
            const auto v = x.read(id);
            Var<float> activated = def(v);
            switch (act) {
                case 1u: activated = ocr_relu(v); break;
                case 2u: activated = ocr_relu6(v); break;
                case 3u: activated = ocr_hard_swish(v); break;
                default: activated = v; break;
            }
            out.write(id, activated);
        };
    };
    return kernel;
}

[[nodiscard]] FpnResizeKernel make_fpn_resize_kernel() noexcept {
    FpnResizeKernel kernel = [](BufferFloat x, BufferFloat out, Var<uint> n, Var<uint> c,
                                Var<uint> h, Var<uint> w, Var<uint> out_h, Var<uint> out_w) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_fpn_resize");
        const auto id = ocr_linear_id();
        const auto plane = out_h * out_w;
        $if (id < n * c * plane) {
            const auto ox = id % out_w;
            const auto oy = (id / out_w) % out_h;
            const auto ci = (id / plane) % c;
            const auto ni = id / (plane * c);
            // Half-pixel centers, align_corners=false.
            const auto scale_x = cast<float>(w) / cast<float>(out_w);
            const auto scale_y = cast<float>(h) / cast<float>(out_h);
            const auto sx = (cast<float>(ox) + 0.5f) * scale_x - 0.5f;
            const auto sy = (cast<float>(oy) + 0.5f) * scale_y - 0.5f;
            const auto x0f = floor(sx);
            const auto y0f = floor(sy);
            const auto fx = sx - x0f;
            const auto fy = sy - y0f;
            const auto base = (ni * c + ci) * h * w;
            const auto sample = [&](Var<int> px, Var<int> py) noexcept {
                const auto cx = clamp(px, 0, cast<int>(w) - 1);
                const auto cy = clamp(py, 0, cast<int>(h) - 1);
                return x.read(base + cast<uint>(cy) * w + cast<uint>(cx));
            };
            const auto x0 = cast<int>(x0f);
            const auto y0 = cast<int>(y0f);
            const auto v00 = sample(x0, y0);
            const auto v10 = sample(x0 + 1, y0);
            const auto v01 = sample(x0, y0 + 1);
            const auto v11 = sample(x0 + 1, y0 + 1);
            const auto top = v00 + (v10 - v00) * fx;
            const auto bottom = v01 + (v11 - v01) * fx;
            out.write(id, top + (bottom - top) * fy);
        };
    };
    return kernel;
}

[[nodiscard]] PolygonCropKernel make_polygon_crop_kernel() noexcept {
    PolygonCropKernel kernel = [](BufferFloat src, BufferFloat dst, Var<uint> sw, Var<uint> sh,
                                  Var<uint> crop_w, Var<uint> crop_h, Var<float> cx, Var<float> cy,
                                  Var<float> rw, Var<float> rh, Var<float> angle) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_polygon_crop");
        const auto id = ocr_linear_id();
        const auto plane = crop_w * crop_h;
        $if (id < 3u * plane & rw > 0.0f & rh > 0.0f) {
            const auto px = id % crop_w;
            const auto py = (id / crop_w) % crop_h;
            const auto ch = id / plane;
            // Crop-local coordinates centered at the crop center.
            const auto u = (cast<float>(px) + 0.5f) / cast<float>(crop_w) - 0.5f;
            const auto v = (cast<float>(py) + 0.5f) / cast<float>(crop_h) - 0.5f;
            // Rotate by angle (CCW in image coords) and scale to the rect.
            const auto cos_a = cos(angle);
            const auto sin_a = sin(angle);
            const auto rx = u * rw;
            const auto ry = v * rh;
            const auto sx = cx + rx * cos_a - ry * sin_a;
            const auto sy = cy + rx * sin_a + ry * cos_a;
            Var<float> value = def(0.0f);
            $if (sx >= 0.0f & sy >= 0.0f & sx <= cast<float>(sw - 1u) & sy <= cast<float>(sh - 1u)) {
                const auto x0f = floor(sx);
                const auto y0f = floor(sy);
                const auto fx = sx - x0f;
                const auto fy = sy - y0f;
                const auto x0 = cast<int>(x0f);
                const auto y0 = cast<int>(y0f);
                const auto sample = [&](Var<int> ax, Var<int> ay) noexcept {
                    const auto bx = clamp(ax, 0, cast<int>(sw) - 1);
                    const auto by = clamp(ay, 0, cast<int>(sh) - 1);
                    return src.read(ch * sw * sh + cast<uint>(by) * sw + cast<uint>(bx));
                };
                const auto v00 = sample(x0, y0);
                const auto v10 = sample(x0 + 1, y0);
                const auto v01 = sample(x0, y0 + 1);
                const auto v11 = sample(x0 + 1, y0 + 1);
                const auto top = v00 + (v10 - v00) * fx;
                const auto bottom = v01 + (v11 - v01) * fx;
                value = top + (bottom - top) * fy;
            };
            dst.write(id, value);
        };
    };
    return kernel;
}

[[nodiscard]] Result<LaunchPlan1D> plan_ctc_log_softmax_dispatch(uint64_t rows) {
    return plan_threads(rows, 256u);
}

[[nodiscard]] Result<LaunchPlan1D> plan_ctc_argmax_dispatch(uint64_t rows) {
    return plan_threads(rows, 256u);
}

[[nodiscard]] Result<LaunchPlan1D> plan_sigmoid_dispatch(uint64_t count) {
    return plan_threads(count, 256u);
}

[[nodiscard]] Result<LaunchPlan1D> plan_height_collapse_dispatch(uint64_t frames, uint64_t channels) {
    uint64_t count = 0;
    if (!checked_mul(frames, channels, count)) {
        return Result<LaunchPlan1D>::failure(StatusCode::invalid_argument,
                                             "height-collapse count overflow", "frames*channels");
    }
    return plan_threads(count, 256u);
}

[[nodiscard]] Result<LaunchPlan1D> plan_cnn_act_dispatch(uint64_t count) {
    return plan_threads(count, 256u);
}

[[nodiscard]] Result<LaunchPlan1D> plan_fpn_resize_dispatch(uint64_t n, uint64_t c, uint64_t out_h, uint64_t out_w) {
    uint64_t count = 0;
    if (!checked_mul(n, c, count) || !checked_mul(count, out_h, count) || !checked_mul(count, out_w, count)) {
        return Result<LaunchPlan1D>::failure(StatusCode::invalid_argument, "fpn resize count overflow", "n*c*out_h*out_w");
    }
    return plan_threads(count, 256u);
}

[[nodiscard]] Result<LaunchPlan1D> plan_polygon_crop_dispatch(uint64_t crop_w, uint64_t crop_h) {
    uint64_t count = 0;
    if (!checked_mul(crop_w, crop_h, count) || !checked_mul(count, 3u, count)) {
        return Result<LaunchPlan1D>::failure(StatusCode::invalid_argument, "polygon crop count overflow", "crop_w*crop_h*3");
    }
    return plan_threads(count, 256u);
}

[[nodiscard]] luisa::string ctc_variant_key(luisa::string_view op) {
    return make_shader_key("ctc", op);
}

// ---------------------------------------------------------------------------
// Host oracles
// ---------------------------------------------------------------------------

[[nodiscard]] Status ref_ctc_log_softmax(luisa::span<const float> logits, uint32_t rows, uint32_t cols,
                                         float temperature, luisa::vector<float> &out) {
    if (auto s = validate_ctc_log_softmax_inputs(rows, cols, temperature); !s.ok()) { return s; }
    if (logits.size() != static_cast<size_t>(rows) * cols) {
        return Status::error(StatusCode::invalid_argument, "logits span size mismatch", "logits");
    }
    out.resize(static_cast<size_t>(rows) * cols);
    const double t = static_cast<double>(temperature);
    for (uint32_t r = 0; r < rows; r++) {
        const size_t base = static_cast<size_t>(r) * cols;
        double max_v = -std::numeric_limits<double>::infinity();
        for (uint32_t c = 0; c < cols; c++) {
            max_v = std::max(max_v, static_cast<double>(logits[base + c]) / t);
        }
        double sum = 0.0;
        for (uint32_t c = 0; c < cols; c++) {
            sum += std::exp(static_cast<double>(logits[base + c]) / t - max_v);
        }
        const double log_sum = std::log(sum);
        for (uint32_t c = 0; c < cols; c++) {
            out[base + c] = static_cast<float>(static_cast<double>(logits[base + c]) / t - max_v - log_sum);
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_ctc_argmax_probs(luisa::span<const float> log_probs, uint32_t rows, uint32_t cols,
                                          luisa::vector<int32_t> &ids, luisa::vector<float> &probs) {
    if (auto s = validate_ctc_frame_inputs(rows, cols); !s.ok()) { return s; }
    if (log_probs.size() != static_cast<size_t>(rows) * cols) {
        return Status::error(StatusCode::invalid_argument, "log_probs span size mismatch", "log_probs");
    }
    ids.resize(rows);
    probs.resize(rows);
    for (uint32_t r = 0; r < rows; r++) {
        const size_t base = static_cast<size_t>(r) * cols;
        uint32_t best = 0;
        float best_v = log_probs[base];
        for (uint32_t c = 1; c < cols; c++) {
            const float v = log_probs[base + c];
            if (v > best_v) {
                best_v = v;
                best = c;
            }
        }
        ids[r] = static_cast<int32_t>(best);
        probs[r] = static_cast<float>(std::exp(static_cast<double>(best_v)));
    }
    return Status::success();
}

[[nodiscard]] Status ref_sigmoid(luisa::span<const float> x, luisa::vector<float> &out) {
    out.resize(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        out[i] = static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(x[i]))));
    }
    return Status::success();
}

[[nodiscard]] Status ref_height_collapse(luisa::span<const float> x, uint32_t n, uint32_t c,
                                         uint32_t h, uint32_t w, uint32_t r, luisa::vector<float> &out) {
    if (n == 0u || c == 0u || h == 0u || w == 0u || r == 0u || h % r != 0u) {
        return Status::error(StatusCode::invalid_argument, "height-collapse invalid geometry", "geometry");
    }
    if (x.size() != static_cast<size_t>(n) * c * h * w) {
        return Status::error(StatusCode::invalid_argument, "height-collapse input size mismatch", "x");
    }
    const uint32_t hb = h / r;
    const uint64_t frames = static_cast<uint64_t>(n) * w * hb;
    const uint32_t channels = c * r;
    out.resize(static_cast<size_t>(frames) * channels);
    for (uint32_t ni = 0; ni < n; ni++) {
        for (uint32_t by = 0; by < hb; by++) {
            for (uint32_t xcol = 0; xcol < w; xcol++) {
                const uint64_t frame = (static_cast<uint64_t>(ni) * hb + by) * w + xcol;
                for (uint32_t ry = 0; ry < r; ry++) {
                    for (uint32_t ch = 0; ch < c; ch++) {
                        const uint32_t y = by * r + ry;
                        const uint64_t src = ((static_cast<uint64_t>(ni) * c + ch) * h + y) * w + xcol;
                        const uint64_t dst = frame * channels + ry * c + ch;
                        out[static_cast<size_t>(dst)] = x[static_cast<size_t>(src)];
                    }
                }
            }
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_cnn_act(luisa::span<const float> x, uint32_t act, luisa::vector<float> &out) {
    out.resize(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        const double v = x[i];
        double a = v;
        switch (act) {
            case 1u: a = std::max(v, 0.0); break;
            case 2u: a = std::clamp(v, 0.0, 6.0); break;
            case 3u: a = v * std::clamp(v + 3.0, 0.0, 6.0) / 6.0; break;
            default: break;
        }
        out[i] = static_cast<float>(a);
    }
    return Status::success();
}

namespace {

// Double bilinear sample of a CHW plane with clamped edges.
[[nodiscard]] double sample_plane(luisa::span<const float> x, uint32_t w, uint32_t h,
                                  size_t base, double sx, double sy) noexcept {
    const double x0f = std::floor(sx);
    const double y0f = std::floor(sy);
    const int32_t x0 = static_cast<int32_t>(x0f);
    const int32_t y0 = static_cast<int32_t>(y0f);
    const double fx = sx - x0f;
    const double fy = sy - y0f;
    const auto at = [&](int32_t px, int32_t py) -> double {
        const int32_t cx = std::clamp(px, 0, static_cast<int32_t>(w) - 1);
        const int32_t cy = std::clamp(py, 0, static_cast<int32_t>(h) - 1);
        return x[base + static_cast<size_t>(cy) * w + static_cast<uint32_t>(cx)];
    };
    const double v00 = at(x0, y0);
    const double v10 = at(x0 + 1, y0);
    const double v01 = at(x0, y0 + 1);
    const double v11 = at(x0 + 1, y0 + 1);
    const double top = v00 + (v10 - v00) * fx;
    const double bottom = v01 + (v11 - v01) * fx;
    return top + (bottom - top) * fy;
}

}// namespace

[[nodiscard]] Status ref_fpn_resize(luisa::span<const float> x, uint32_t n, uint32_t c, uint32_t h,
                                    uint32_t w, uint32_t out_h, uint32_t out_w, luisa::vector<float> &out) {
    if (n == 0u || c == 0u || h == 0u || w == 0u || out_h == 0u || out_w == 0u) {
        return Status::error(StatusCode::invalid_argument, "fpn resize invalid geometry", "geometry");
    }
    if (x.size() != static_cast<size_t>(n) * c * h * w) {
        return Status::error(StatusCode::invalid_argument, "fpn resize input size mismatch", "x");
    }
    out.resize(static_cast<size_t>(n) * c * out_h * out_w);
    const double scale_x = static_cast<double>(w) / out_w;
    const double scale_y = static_cast<double>(h) / out_h;
    for (uint32_t ni = 0; ni < n; ni++) {
        for (uint32_t ci = 0; ci < c; ci++) {
            const size_t base = (static_cast<size_t>(ni) * c + ci) * h * w;
            const size_t obase = (static_cast<size_t>(ni) * c + ci) * out_h * out_w;
            for (uint32_t oy = 0; oy < out_h; oy++) {
                for (uint32_t ox = 0; ox < out_w; ox++) {
                    const double sx = (ox + 0.5) * scale_x - 0.5;
                    const double sy = (oy + 0.5) * scale_y - 0.5;
                    out[obase + static_cast<size_t>(oy) * out_w + ox] =
                        static_cast<float>(sample_plane(x, w, h, base, sx, sy));
                }
            }
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_polygon_crop(luisa::span<const float> src, uint32_t sw, uint32_t sh,
                                      uint32_t crop_w, uint32_t crop_h, float cx, float cy,
                                      float rw, float rh, float angle, luisa::vector<float> &out) {
    if (sw == 0u || sh == 0u || crop_w == 0u || crop_h == 0u || rw <= 0.0f || rh <= 0.0f) {
        return Status::error(StatusCode::invalid_argument, "polygon crop invalid geometry", "geometry");
    }
    if (src.size() != static_cast<size_t>(3) * sw * sh) {
        return Status::error(StatusCode::invalid_argument, "polygon crop input size mismatch", "src");
    }
    out.resize(static_cast<size_t>(3) * crop_w * crop_h);
    const double cos_a = std::cos(static_cast<double>(angle));
    const double sin_a = std::sin(static_cast<double>(angle));
    for (uint32_t ch = 0; ch < 3u; ch++) {
        const size_t base = static_cast<size_t>(ch) * sw * sh;
        for (uint32_t py = 0; py < crop_h; py++) {
            for (uint32_t px = 0; px < crop_w; px++) {
                const double u = (px + 0.5) / static_cast<double>(crop_w) - 0.5;
                const double v = (py + 0.5) / static_cast<double>(crop_h) - 0.5;
                const double rx = u * rw;
                const double ry = v * rh;
                const double sx = cx + rx * cos_a - ry * sin_a;
                const double sy = cy + rx * sin_a + ry * cos_a;
                double value = 0.0;
                if (sx >= 0.0 && sy >= 0.0 && sx <= static_cast<double>(sw - 1u) &&
                    sy <= static_cast<double>(sh - 1u)) {
                    value = sample_plane(src, sw, sh, base, sx, sy);
                }
                out[static_cast<size_t>(ch) * crop_w * crop_h + static_cast<size_t>(py) * crop_w + px] =
                    static_cast<float>(value);
            }
        }
    }
    return Status::success();
}

}// namespace luisa::ocr