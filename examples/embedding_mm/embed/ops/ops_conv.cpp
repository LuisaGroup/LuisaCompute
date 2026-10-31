// embedding_mm engine -- Audio 2D Subsampling & Causal Depthwise 1D Convolution implementations.
#include "embedding_mm/embed/ops/ops_conv.h"

#include <luisa/dsl/sugar.h>

namespace luisa::embed_mm {

AudioSubsampleConv0Kernel make_audio_subsample_conv0_kernel() noexcept {
    using namespace luisa::compute;

    AudioSubsampleConv0Kernel kernel = [](BufferFloat in_features,
                                          BufferFloat weight,
                                          BufferFloat norm_weight,
                                          BufferFloat out_h0,
                                          Var<uint> H_in,
                                          Var<uint> W_in,
                                          Var<uint> H_out,
                                          Var<uint> W_out,
                                          Var<float> eps) noexcept {
        set_block_size(128u, 1u, 1u);
        set_name("audio_subsample_conv0");

        Shared<float> red{128u};

        Var tx = thread_x();
        Var w = dispatch_y();
        Var h = dispatch_z();

        Var v = def(0.0f);
        $if (w < W_out & h < H_out) {
            Var in_h_base = h.cast<int>() * 2 - 1;
            Var in_w_base = w.cast<int>() * 2 - 1;

            $for (kh, 3u) {
                $for (kw, 3u) {
                    Var ih = in_h_base + kh.cast<int>();
                    Var iw = in_w_base + kw.cast<int>();
                    $if (ih >= 0 & ih < H_in.cast<int>() & iw >= 0 & iw < W_in.cast<int>()) {
                        Var in_idx = ih.cast<uint>() * W_in + iw.cast<uint>();
                        Var w_idx = (tx * 3u + kh) * 3u + kw;
                        v += in_features.read(in_idx) * weight.read(w_idx);
                    };
                };
            };
        };

        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            for (auto stride = 64u; stride > 0u; stride >>= 1u) {
                $if (tx < stride) {
                    red[tx] += red[tx + stride];
                };
                sync_block();
            };
            Var total = red[0u];
            sync_block();
            return total;
        };

        Var mean = block_sum(v) * (1.0f / 128.0f);
        Var diff = v - mean;
        Var var = block_sum(diff * diff) * (1.0f / 128.0f);
        Var inv_std = rsqrt(var + eps);

        $if (w < W_out & h < H_out) {
            Var normed = diff * inv_std * norm_weight.read(tx);
            Var val = max(0.0f, normed);
            out_h0.write(tx * (H_out * W_out) + h * W_out + w, val);
        };
    };
    return kernel;
}

AudioSubsampleConv1Kernel make_audio_subsample_conv1_kernel() noexcept {
    using namespace luisa::compute;

    AudioSubsampleConv1Kernel kernel = [](BufferFloat in_h0,
                                          BufferFloat weight,
                                          BufferFloat norm_weight,
                                          BufferFloat out_flat,
                                          Var<uint> H_in,
                                          Var<uint> W_in,
                                          Var<uint> H_out,
                                          Var<uint> W_out,
                                          Var<float> eps) noexcept {
        set_block_size(32u, 1u, 1u);
        set_name("audio_subsample_conv1");

        Shared<float> red{32u};

        Var tx = thread_x();
        Var w = dispatch_y();
        Var h = dispatch_z();

        Var v = def(0.0f);
        $if (w < W_out & h < H_out) {
            Var in_h_base = h.cast<int>() * 2 - 1;
            Var in_w_base = w.cast<int>() * 2 - 1;

            $for (in_c, 128u) {
                Var c_offset = in_c * (H_in * W_in);
                Var w_c_offset = (tx * 128u + in_c) * 9u;
                $for (kh, 3u) {
                    $for (kw, 3u) {
                        Var ih = in_h_base + kh.cast<int>();
                        Var iw = in_w_base + kw.cast<int>();
                        $if (ih >= 0 & ih < H_in.cast<int>() & iw >= 0 & iw < W_in.cast<int>()) {
                            Var in_idx = c_offset + ih.cast<uint>() * W_in + iw.cast<uint>();
                            Var w_idx = w_c_offset + kh * 3u + kw;
                            v += in_h0.read(in_idx) * weight.read(w_idx);
                        };
                    };
                };
            };
        };

        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            for (auto stride = 16u; stride > 0u; stride >>= 1u) {
                $if (tx < stride) {
                    red[tx] += red[tx + stride];
                };
                sync_block();
            };
            Var total = red[0u];
            sync_block();
            return total;
        };

        Var mean = block_sum(v) * (1.0f / 32.0f);
        Var diff = v - mean;
        Var var = block_sum(diff * diff) * (1.0f / 32.0f);
        Var inv_std = rsqrt(var + eps);

        $if (w < W_out & h < H_out) {
            Var normed = diff * inv_std * norm_weight.read(tx);
            Var val = max(0.0f, normed);
            out_flat.write(h * 1024u + w * 32u + tx, val);
        };
    };
    return kernel;
}

AudioDepthwiseConv1DKernel make_audio_depthwise_conv1d_kernel() noexcept {
    using namespace luisa::compute;

    AudioDepthwiseConv1DKernel kernel = [](BufferFloat in_x,
                                           BufferFloat weight,
                                           BufferFloat out_y,
                                           Var<uint> T,
                                           Var<uint> channels) noexcept {
        set_block_size(128u, 1u, 1u);
        set_name("audio_depthwise_conv1d");

        Var c = dispatch_x();
        Var t = dispatch_y();

        $if (c < channels & t < T) {
            Var acc = def(0.0f);
            $for (k, 5u) {
                Var src_t = t.cast<int>() - 4 + k.cast<int>();
                $if (src_t >= 0) {
                    Var in_val = in_x.read(src_t.cast<uint>() * channels + c);
                    Var w_val = weight.read(c * 5u + k);
                    acc += in_val * w_val;
                };
            };
            out_y.write(t * channels + c, acc);
        };
    };
    return kernel;
}

}// namespace luisa::embed_mm
