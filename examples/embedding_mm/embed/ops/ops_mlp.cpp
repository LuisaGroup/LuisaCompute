// embedding_mm engine -- Pointwise MLP activations, gating, clamping, and residual implementations.
#include "embedding_mm/embed/ops/ops_mlp.h"

#include <luisa/dsl/sugar.h>

namespace luisa::embed_mm {

GeGLUKernel make_geglu_kernel() noexcept {
    using namespace luisa::compute;

    GeGLUKernel kernel = [](BufferFloat gate, BufferFloat up, BufferFloat out, Var<uint> N) noexcept {
        set_block_size(kMLPBlock, 1u, 1u);
        set_name("geglu");

        Var idx = dispatch_x();
        Var stride = dispatch_size().x;
        $while (idx < N) {
            Var x = gate.read(idx);
            Var u = up.read(idx);

            // gelu_tanh(x) = 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            constexpr float kSqrt2OverPi = 0.7978845608028654f;
            constexpr float kCoeff = 0.044715f;

            Var inner = kSqrt2OverPi * (x + kCoeff * x * x * x);
            Var act = 0.5f * x * (1.0f + tanh(inner));

            out.write(idx, act * u);
            idx += stride;
        };
    };
    return kernel;
}

ResidualAddKernel make_residual_add_kernel() noexcept {
    using namespace luisa::compute;

    ResidualAddKernel kernel = [](BufferFloat x, BufferFloat residual, BufferFloat out, Var<uint> N) noexcept {
        set_block_size(kMLPBlock, 1u, 1u);
        set_name("residual_add");

        Var idx = dispatch_x();
        Var stride = dispatch_size().x;
        $while (idx < N) {
            out.write(idx, x.read(idx) + residual.read(idx));
            idx += stride;
        };
    };
    return kernel;
}

PLEGateKernel make_ple_gate_kernel() noexcept {
    using namespace luisa::compute;

    PLEGateKernel kernel = [](BufferFloat gate, BufferFloat ple_all, BufferFloat out,
                              Var<uint> layer_idx, Var<uint> total_tokens, Var<uint> H) noexcept {
        set_block_size(kMLPBlock, 1u, 1u);
        set_name("ple_gate");

        Var idx = dispatch_x();
        Var stride = dispatch_size().x;
        Var N = total_tokens * H;
        $while (idx < N) {
            Var token_idx = idx / H;
            Var d = idx % H;

            Var g = gate.read(idx);
            Var ple_in = ple_all.read((token_idx * 24u + layer_idx) * H + d);

            constexpr float kSqrt2OverPi = 0.7978845608028654f;
            constexpr float kCoeff = 0.044715f;

            Var inner = kSqrt2OverPi * (g + kCoeff * g * g * g);
            Var act = 0.5f * g * (1.0f + tanh(inner));

            out.write(idx, act * ple_in);
            idx += stride;
        };
    };
    return kernel;
}

ScaleAddKernel make_scale_add_kernel() noexcept {
    using namespace luisa::compute;

    ScaleAddKernel kernel = [](BufferFloat x, BufferFloat residual, BufferFloat out,
                               Var<float> scale, Var<uint> N) noexcept {
        set_block_size(kMLPBlock, 1u, 1u);
        set_name("scale_add");

        Var idx = dispatch_x();
        Var stride = dispatch_size().x;
        $while (idx < N) {
            Var sum = x.read(idx) + residual.read(idx);
            out.write(idx, sum * scale);
            idx += stride;
        };
    };
    return kernel;
}

AudioClampKernel make_audio_clamp_kernel() noexcept {
    using namespace luisa::compute;

    AudioClampKernel kernel = [](BufferFloat x,
                                 BufferFloat y,
                                 Var<float> min_v,
                                 Var<float> max_v,
                                 Var<uint> total_elements) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("audio_clamp");

        Var idx = dispatch_x();
        $if (idx < total_elements) {
            Var val = x.read(idx);
            y.write(idx, clamp(val, min_v, max_v));
        };
    };
    return kernel;
}

AudioSiluKernel make_audio_silu_kernel() noexcept {
    using namespace luisa::compute;

    AudioSiluKernel kernel = [](BufferFloat x,
                                BufferFloat y,
                                Var<uint> total_elements) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("audio_silu");

        Var idx = dispatch_x();
        $if (idx < total_elements) {
            Var val = x.read(idx);
            Var sig = 1.0f / (1.0f + exp(-val));
            y.write(idx, val * sig);
        };
    };
    return kernel;
}

AudioGluKernel make_audio_glu_kernel() noexcept {
    using namespace luisa::compute;

    AudioGluKernel kernel = [](BufferFloat in_x,
                               BufferFloat out_y,
                               Var<uint> T,
                               Var<uint> H) noexcept {
        set_block_size(128u, 1u, 1u);
        set_name("audio_glu");

        Var c = dispatch_x();
        Var t = dispatch_y();

        $if (c < H & t < T) {
            Var in_idx1 = t * (H * 2u) + c;
            Var in_idx2 = t * (H * 2u) + H + c;
            Var x1 = in_x.read(in_idx1);
            Var x2 = in_x.read(in_idx2);
            Var sig = 1.0f / (1.0f + exp(-x2));
            out_y.write(t * H + c, x1 * sig);
        };
    };
    return kernel;
}

AudioAddScaleKernel make_audio_add_scale_kernel() noexcept {
    using namespace luisa::compute;

    AudioAddScaleKernel kernel = [](BufferFloat x,
                                    BufferFloat residual,
                                    BufferFloat y,
                                    Var<float> scale,
                                    Var<uint> total_elements) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("audio_add_scale");

        Var idx = dispatch_x();
        $if (idx < total_elements) {
            y.write(idx, x.read(idx) * scale + residual.read(idx));
        };
    };
    return kernel;
}

AudioAddKernel make_audio_add_kernel() noexcept {
    using namespace luisa::compute;

    AudioAddKernel kernel = [](BufferFloat x,
                               BufferFloat residual,
                               BufferFloat y,
                               Var<uint> total_elements) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("audio_add");

        Var idx = dispatch_x();
        $if (idx < total_elements) {
            y.write(idx, x.read(idx) + residual.read(idx));
        };
    };
    return kernel;
}

}// namespace luisa::embed_mm
