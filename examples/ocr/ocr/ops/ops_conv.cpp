// OCR engine -- convolution / pooling / tensor-shape operator implementations (Phase 2).
#include "ocr/ops/ops_conv.h"

#include <luisa/dsl/sugar.h>

#include "ocr/ops/ops_common.h"
#include "ocr/quantization.h"
#include "ocr/runtime/shader_cache.h"

namespace luisa::ocr {

using namespace luisa::compute;

[[nodiscard]] Result<uint32_t> conv2d_out_dim(uint32_t in, uint32_t k, uint32_t stride, uint32_t pad, uint32_t dil) {
    using R = Result<uint32_t>;
    if (k == 0u || stride == 0u || dil == 0u) {
        return R::failure(StatusCode::invalid_argument, "invalid conv geometry", "geometry");
    }
    const uint64_t eff = static_cast<uint64_t>(k - 1u) * dil + 1u;
    const uint64_t padded = static_cast<uint64_t>(in) + 2u * pad;
    if (padded < eff) {
        return R::failure(StatusCode::invalid_argument, "conv input smaller than kernel", "geometry");
    }
    return checked_u32((padded - eff) / stride + 1u, "conv out dim");
}

[[nodiscard]] Result<uint32_t> conv1d_out_len(uint32_t len, uint32_t k, uint32_t stride, uint32_t pad,
                                              uint32_t dil, bool causal) {
    using R = Result<uint32_t>;
    if (k == 0u || stride == 0u || dil == 0u) {
        return R::failure(StatusCode::invalid_argument, "invalid conv1d geometry", "geometry");
    }
    const uint64_t eff = static_cast<uint64_t>(k - 1u) * dil + 1u;
    const uint64_t pad_left = causal ? eff - 1u : pad;
    const uint64_t pad_right = causal ? 0u : pad;
    const uint64_t padded = static_cast<uint64_t>(len) + pad_left + pad_right;
    if (padded < eff) {
        return R::failure(StatusCode::invalid_argument, "conv1d input smaller than kernel", "geometry");
    }
    return checked_u32((padded - eff) / stride + 1u, "conv1d out len");
}

[[nodiscard]] Status validate_conv2d_inputs(uint32_t n, uint32_t in_c, uint32_t h, uint32_t w,
                                            uint32_t out_c, uint32_t kh, uint32_t kw,
                                            uint32_t stride_h, uint32_t stride_w,
                                            uint32_t dil_h, uint32_t dil_w, uint32_t groups) {
    if (groups == 0u || in_c == 0u || out_c == 0u || in_c % groups != 0u || out_c % groups != 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "conv2d groups must divide both in_c and out_c (all nonzero)", "groups");
    }
    if (n == 0u || h == 0u || w == 0u || kh == 0u || kw == 0u ||
        stride_h == 0u || stride_w == 0u || dil_h == 0u || dil_w == 0u) {
        return Status::error(StatusCode::invalid_argument, "conv2d dimensions must be nonzero", "dims");
    }
    return Status::success();
}

[[nodiscard]] Status validate_conv1d_inputs(uint32_t n, uint32_t c, uint32_t len, uint32_t out_c,
                                            uint32_t k, uint32_t stride, uint32_t dilation, uint32_t groups) {
    if (groups == 0u || c == 0u || out_c == 0u || c % groups != 0u || out_c % groups != 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "conv1d groups must divide both c and out_c (all nonzero)", "groups");
    }
    if (n == 0u || len == 0u || k == 0u || stride == 0u || dilation == 0u) {
        return Status::error(StatusCode::invalid_argument, "conv1d dimensions must be nonzero", "dims");
    }
    return Status::success();
}

[[nodiscard]] Conv2dKernel make_conv2d_kernel() noexcept {
    Conv2dKernel kernel = [](BufferFloat x, BufferFloat weight, BufferFloat bias, BufferFloat out,
                             Var<uint> n, Var<uint> in_c, Var<uint> h, Var<uint> w,
                             Var<uint> out_c, Var<uint> kh, Var<uint> kw,
                             Var<uint> stride_h, Var<uint> stride_w, Var<uint> pad_h, Var<uint> pad_w,
                             Var<uint> dil_h, Var<uint> dil_w, Var<uint> groups,
                             Var<uint> out_h, Var<uint> out_w, Var<uint> has_bias) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_conv2d");
        const auto id = ocr_linear_id();
        $if (id < n * out_c * out_h * out_w) {
            // S5 defensive: invalid groups/divisibility -> defined zero output.
            $if (groups == 0u | in_c % groups != 0u | out_c % groups != 0u | out_c == 0u | in_c == 0u) {
                out.write(id, 0.0f);
            } $else {
            const auto ox = id % out_w;
            const auto oy = (id / out_w) % out_h;
            const auto oc = (id / (out_w * out_h)) % out_c;
            const auto ni = id / (out_w * out_h * out_c);
            const auto icg = in_c / groups;
            const auto ocg = out_c / groups;
            const auto g = oc / ocg;
            Var<float> acc = def(0.0f);
            $if (has_bias != 0u) { acc = bias.read(oc); };
            $for (ic, icg) {
                const auto ic_abs = g * icg + ic;
                $for (ky, kh) {
                    const auto iy = cast<int>(oy * stride_h) + cast<int>(ky * dil_h) - cast<int>(pad_h);
                    $if (iy >= 0 & iy < cast<int>(h)) {
                        $for (kx, kw) {
                            const auto ix = cast<int>(ox * stride_w) + cast<int>(kx * dil_w) - cast<int>(pad_w);
                            $if (ix >= 0 & ix < cast<int>(w)) {
                                const auto xi = ((ni * in_c + ic_abs) * h + cast<uint>(iy)) * w + cast<uint>(ix);
                                const auto wi = ((oc * icg + ic) * kh + ky) * kw + kx;
                                acc += x.read(xi) * weight.read(wi);
                            };
                        };
                    };
                };
            };
            out.write(id, acc);
            };
        };
    };
    return kernel;
}

[[nodiscard]] Conv1dKernel make_conv1d_kernel() noexcept {
    Conv1dKernel kernel = [](BufferFloat x, BufferFloat weight, BufferFloat bias, BufferFloat out,
                             Var<uint> n, Var<uint> c, Var<uint> len, Var<uint> out_c, Var<uint> k,
                             Var<uint> stride, Var<uint> pad, Var<uint> dilation, Var<uint> groups,
                             Var<uint> causal, Var<uint> out_len, Var<uint> has_bias) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_conv1d");
        const auto id = ocr_linear_id();
        $if (id < n * out_c * out_len) {
            // S5 defensive: invalid groups/divisibility -> defined zero output.
            $if (groups == 0u | c % groups != 0u | out_c % groups != 0u | out_c == 0u | c == 0u) {
                out.write(id, 0.0f);
            } $else {
            const auto ot = id % out_len;
            const auto oc = (id / out_len) % out_c;
            const auto ni = id / (out_len * out_c);
            const auto icg = c / groups;
            const auto ocg = out_c / groups;
            const auto g = oc / ocg;
            const auto eff_k = (k - 1u) * dilation + 1u;
            const auto pad_left = select(pad, eff_k - 1u, causal != 0u);
            Var<float> acc = def(0.0f);
            $if (has_bias != 0u) { acc = bias.read(oc); };
            $for (ic, icg) {
                const auto ic_abs = g * icg + ic;
                $for (kt, k) {
                    const auto it = cast<int>(ot * stride) + cast<int>(kt * dilation) - cast<int>(pad_left);
                    $if (it >= 0 & it < cast<int>(len)) {
                        const auto xi = (ni * c + ic_abs) * len + cast<uint>(it);
                        const auto wi = (oc * icg + ic) * k + kt;
                        acc += x.read(xi) * weight.read(wi);
                    };
                };
            };
            out.write(id, acc);
            };
        };
    };
    return kernel;
}

[[nodiscard]] BatchNormKernel make_batch_norm_kernel() noexcept {
    BatchNormKernel kernel = [](BufferFloat x, BufferFloat mean, BufferFloat var, BufferFloat gamma,
                                BufferFloat beta, BufferFloat out, Var<uint> n, Var<uint> c,
                                Var<uint> spatial, Var<float> eps) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_batch_norm");
        const auto id = ocr_linear_id();
        $if (id < n * c * spatial) {
            const auto ci = (id / spatial) % c;
            const auto inv = rsqrt(var.read(ci) + eps);
            out.write(id, (x.read(id) - mean.read(ci)) * inv * gamma.read(ci) + beta.read(ci));
        };
    };
    return kernel;
}

[[nodiscard]] AvgPool2dKernel make_avg_pool2d_kernel() noexcept {
    AvgPool2dKernel kernel = [](BufferFloat x, BufferFloat out, Var<uint> n, Var<uint> c, Var<uint> h, Var<uint> w,
                                Var<uint> kh, Var<uint> kw, Var<uint> sh, Var<uint> sw,
                                Var<uint> ph, Var<uint> pw, Var<uint> out_h, Var<uint> out_w) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_avg_pool2d");
        const auto id = ocr_linear_id();
        $if (id < n * c * out_h * out_w) {
            const auto ox = id % out_w;
            const auto oy = (id / out_w) % out_h;
            const auto ci = (id / (out_w * out_h)) % c;
            const auto ni = id / (out_w * out_h * c);
            Var<float> sum = def(0.0f);
            Var<uint> count = def(0u);
            $for (ky, kh) {
                const auto iy = cast<int>(oy * sh) + cast<int>(ky) - cast<int>(ph);
                $if (iy >= 0 & iy < cast<int>(h)) {
                    $for (kx, kw) {
                        const auto ix = cast<int>(ox * sw) + cast<int>(kx) - cast<int>(pw);
                        $if (ix >= 0 & ix < cast<int>(w)) {
                            sum += x.read(((ni * c + ci) * h + cast<uint>(iy)) * w + cast<uint>(ix));
                            count += 1u;
                        };
                    };
                };
            };
            Var<float> result = def(0.0f);
            $if (count > 0u) { result = sum / cast<float>(count); };
            out.write(id, result);
        };
    };
    return kernel;
}

[[nodiscard]] AdaptiveAvgPool2dKernel make_adaptive_avg_pool2d_kernel() noexcept {
    AdaptiveAvgPool2dKernel kernel = [](BufferFloat x, BufferFloat out, Var<uint> n, Var<uint> c, Var<uint> h,
                                        Var<uint> w, Var<uint> out_h, Var<uint> out_w) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_adaptive_avg_pool2d");
        const auto id = ocr_linear_id();
        $if (id < n * c * out_h * out_w) {
            const auto ox = id % out_w;
            const auto oy = (id / out_w) % out_h;
            const auto ci = (id / (out_w * out_h)) % c;
            const auto ni = id / (out_w * out_h * c);
            const auto y0 = oy * h / out_h;
            const auto y1 = (oy + 1u) * h / out_h;
            const auto x0 = ox * w / out_w;
            const auto x1 = (ox + 1u) * w / out_w;
            Var<float> sum = def(0.0f);
            Var<uint> count = def(0u);
            $for (yy, y0, y1) {
                $for (xx, x0, x1) {
                    sum += x.read(((ni * c + ci) * h + yy) * w + xx);
                    count += 1u;
                };
            };
            Var<float> result = def(0.0f);
            $if (count > 0u) { result = sum / cast<float>(count); };
            out.write(id, result);
        };
    };
    return kernel;
}

[[nodiscard]] GlobalAvgPool2dKernel make_global_avg_pool2d_kernel() noexcept {
    GlobalAvgPool2dKernel kernel = [](BufferFloat x, BufferFloat out, Var<uint> n, Var<uint> c,
                                      Var<uint> spatial) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_global_avg_pool2d");
        const auto id = ocr_linear_id();
        $if (id < n * c) {
            const auto base = id * spatial;
            Var<float> sum = def(0.0f);
            $for (p, spatial) { sum += x.read(base + p); };
            out.write(id, sum / cast<float>(spatial));
        };
    };
    return kernel;
}

[[nodiscard]] ConcatChannelsKernel make_concat_channels_kernel() noexcept {
    ConcatChannelsKernel kernel = [](BufferFloat a, BufferFloat b, BufferFloat out, Var<uint> n,
                                     Var<uint> ca, Var<uint> cb, Var<uint> spatial) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_concat_channels");
        const auto id = ocr_linear_id();
        const auto oc = ca + cb;
        $if (id < n * oc * spatial) {
            const auto p = id % spatial;
            const auto ci = (id / spatial) % oc;
            const auto ni = id / (spatial * oc);
            Var<float> value = def(0.0f);
            $if (ci < ca) {
                value = a.read((ni * ca + ci) * spatial + p);
            } $else {
                value = b.read((ni * cb + (ci - ca)) * spatial + p);
            };
            out.write(id, value);
        };
    };
    return kernel;
}

[[nodiscard]] SplitChannelsKernel make_split_channels_kernel() noexcept {
    SplitChannelsKernel kernel = [](BufferFloat x, BufferFloat out, Var<uint> n, Var<uint> c, Var<uint> c0,
                                    Var<uint> count, Var<uint> spatial) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_split_channels");
        const auto id = ocr_linear_id();
        $if (id < n * count * spatial) {
            const auto p = id % spatial;
            const auto ci = (id / spatial) % count;
            const auto ni = id / (spatial * count);
            out.write(id, x.read((ni * c + (c0 + ci)) * spatial + p));
        };
    };
    return kernel;
}

[[nodiscard]] ChannelShuffleKernel make_channel_shuffle_kernel() noexcept {
    ChannelShuffleKernel kernel = [](BufferFloat x, BufferFloat out, Var<uint> n, Var<uint> c,
                                     Var<uint> spatial, Var<uint> groups) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_channel_shuffle");
        const auto id = ocr_linear_id();
        $if (id < n * c * spatial) {
            const auto p = id % spatial;
            const auto ci = (id / spatial) % c;
            const auto ni = id / (spatial * c);
            const auto cpg = c / groups;
            const auto g = ci / cpg;
            const auto j = ci % cpg;
            const auto dst = j * groups + g;
            out.write((ni * c + dst) * spatial + p, x.read(id));
        };
    };
    return kernel;
}

[[nodiscard]] Result<LaunchPlan1D> plan_conv2d_dispatch(uint64_t n, uint64_t out_c, uint64_t out_h, uint64_t out_w) {
    uint64_t count = 0;
    if (!checked_mul(n, out_c, count) || !checked_mul(count, out_h, count) || !checked_mul(count, out_w, count)) {
        return Result<LaunchPlan1D>::failure(StatusCode::invalid_argument, "conv2d output count overflow", "n*out_c*out_h*out_w");
    }
    return plan_threads(count, 256u);
}

[[nodiscard]] Result<LaunchPlan1D> plan_conv1d_dispatch(uint64_t n, uint64_t out_c, uint64_t out_len) {
    uint64_t count = 0;
    if (!checked_mul(n, out_c, count) || !checked_mul(count, out_len, count)) {
        return Result<LaunchPlan1D>::failure(StatusCode::invalid_argument, "conv1d output count overflow", "n*out_c*out_len");
    }
    return plan_threads(count, 256u);
}

[[nodiscard]] Result<LaunchPlan1D> plan_elementwise_dispatch(uint64_t count) {
    return plan_threads(count, 256u);
}

[[nodiscard]] luisa::string conv_variant_key(luisa::string_view op) {
    return make_shader_key("conv", op);
}

}// namespace luisa::ocr