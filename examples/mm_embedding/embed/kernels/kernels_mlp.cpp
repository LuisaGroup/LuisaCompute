// mm_embedding engine -- MLP activation and residual add implementations.
#include "mm_embedding/embed/kernels/kernels_mlp.h"

#include <luisa/dsl/sugar.h>

namespace luisa::mm_embed {

GeGLUKernel make_geglu_kernel() noexcept {
    using namespace luisa::compute;

    GeGLUKernel kernel = [](BufferFloat gate, BufferFloat up, BufferFloat out, Var<uint> N) noexcept {
        set_block_size(kMLPBlock, 1u, 1u);
        set_name("geglu");

        Var idx = dispatch_x();
        $if (idx < N) {
            Var x = gate.read(idx);
            Var u = up.read(idx);

            // gelu_tanh(x) = 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            constexpr float kSqrt2OverPi = 0.7978845608028654f;
            constexpr float kCoeff = 0.044715f;

            Var inner = kSqrt2OverPi * (x + kCoeff * x * x * x);
            Var act = 0.5f * x * (1.0f + tanh(inner));

            out.write(idx, act * u);
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
        $if (idx < N) {
            out.write(idx, x.read(idx) + residual.read(idx));
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
        Var N = total_tokens * H;
        $if (idx < N) {
            Var token_idx = idx / H;
            Var d = idx % H;

            Var g = gate.read(idx);
            Var ple_in = ple_all.read((token_idx * 24u + layer_idx) * H + d);

            constexpr float kSqrt2OverPi = 0.7978845608028654f;
            constexpr float kCoeff = 0.044715f;

            Var inner = kSqrt2OverPi * (g + kCoeff * g * g * g);
            Var act = 0.5f * g * (1.0f + tanh(inner));

            out.write(idx, act * ple_in);
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
        $if (idx < N) {
            Var sum = x.read(idx) + residual.read(idx);
            out.write(idx, sum * scale);
        };
    };
    return kernel;
}

}// namespace luisa::mm_embed
