// OCR engine -- normalization / reduction operator implementations (Phase 2).
#include "ocr/ops/ops_norm.h"

#include <limits>

#include <luisa/dsl/sugar.h>

#include "ocr/ops/ops_common.h"
#include "ocr/runtime/shader_cache.h"

namespace luisa::ocr {

using namespace luisa::compute;

[[nodiscard]] LayerNormKernel make_layer_norm_kernel() noexcept {
    LayerNormKernel kernel = [](BufferFloat x, BufferFloat y, BufferFloat gamma, BufferFloat beta,
                                Var<uint> rows, Var<uint> H, Var<float> eps, Var<uint> has_weight) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_layer_norm");
        const auto row = ocr_linear_id();
        $if (row < rows) {
            const auto base = row * H;
            // FP32 mean (two-pass for numerical stability).
            Var<float> mean = def(0.0f);
            $for (i, H) { mean += x.read(base + i); };
            mean = mean / cast<float>(H);
            Var<float> var = def(0.0f);
            $for (i, H) {
                const auto d = x.read(base + i) - mean;
                var += d * d;
            };
            var = var / cast<float>(H);
            const auto inv = rsqrt(var + eps);
            $for (i, H) {
                Var<float> v = (x.read(base + i) - mean) * inv;
                $if (has_weight != 0u) {
                    v = v * gamma.read(i) + beta.read(i);
                };
                y.write(base + i, v);
            };
        };
    };
    return kernel;
}

[[nodiscard]] LayerNormKernel make_layer_norm_cooperative_kernel() noexcept {
    LayerNormKernel kernel = [](BufferFloat x, BufferFloat y, BufferFloat gamma, BufferFloat beta,
                                Var<uint> rows, Var<uint> H, Var<float> eps, Var<uint> has_weight) noexcept {
        constexpr uint32_t block = 256u;
        set_block_size(block, 1u, 1u);
        set_name("ocr_layer_norm_cooperative");
        Shared<float> reduction{block};
        const auto lane = thread_x();
        const auto width = block_size().x;
        const auto row = dispatch_id().y * (dispatch_size().x / width) + dispatch_id().x / width;
        const auto base = row * H;
        const auto block_sum = [&](Var<float> partial) noexcept {
            reduction[lane] = partial;
            sync_block();
            for (uint32_t stride = block / 2u; stride != 0u; stride >>= 1u) {
                $if (lane < stride) { reduction[lane] += reduction[lane + stride]; };
                sync_block();
            }
            const auto total = reduction[0u];
            sync_block();
            return total;
        };
        Var<float> sum = def(0.0f);
        Var<uint> i = lane;
        $while (i < H) {
            sum += x.read(base + i);
            i += width;
        };
        const auto mean = block_sum(sum) / cast<float>(H);
        Var<float> var_sum = def(0.0f);
        i = lane;
        $while (i < H) {
            const auto delta = x.read(base + i) - mean;
            var_sum += delta * delta;
            i += width;
        };
        const auto inv = rsqrt(block_sum(var_sum) / cast<float>(H) + eps);
        i = lane;
        $while (i < H) {
            Var<float> value = (x.read(base + i) - mean) * inv;
            $if (has_weight != 0u) { value = value * gamma.read(i) + beta.read(i); };
            y.write(base + i, value);
            i += width;
        };
    };
    return kernel;
}

[[nodiscard]] RmsNormKernel make_rms_norm_kernel() noexcept {
    RmsNormKernel kernel = [](BufferFloat x, BufferFloat y, BufferFloat gamma,
                                Var<uint> rows, Var<uint> H, Var<float> eps, Var<uint> has_weight) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_rms_norm");
        const auto row = ocr_linear_id();
        $if (row < rows) {
            const auto base = row * H;
            Var<float> sumsq = def(0.0f);
            $for (i, H) {
                const auto v = x.read(base + i);
                sumsq += v * v;
            };
            const auto inv = rsqrt(sumsq / cast<float>(H) + eps);
            $for (i, H) {
                Var<float> v = x.read(base + i) * inv;
                $if (has_weight != 0u) { v = v * gamma.read(i); };
                y.write(base + i, v);
            };
        };
    };
    return kernel;
}

[[nodiscard]] RmsNormKernel make_rms_norm_cooperative_kernel() noexcept {
    RmsNormKernel kernel = [](BufferFloat x, BufferFloat y, BufferFloat gamma,
                              Var<uint> rows, Var<uint> H, Var<float> eps, Var<uint> has_weight) noexcept {
        constexpr uint32_t block = 256u;
        set_block_size(block, 1u, 1u);
        set_name("ocr_rms_norm_cooperative");
        Shared<float> reduction{block};
        const auto lane = thread_x();
        const auto width = block_size().x;
        const auto row = dispatch_id().y * (dispatch_size().x / width) + dispatch_id().x / width;
        const auto base = row * H;
        const auto block_sum = [&](Var<float> partial) noexcept {
            reduction[lane] = partial;
            sync_block();
            for (uint32_t stride = block / 2u; stride != 0u; stride >>= 1u) {
                $if (lane < stride) { reduction[lane] += reduction[lane + stride]; };
                sync_block();
            }
            const auto total = reduction[0u];
            sync_block();
            return total;
        };
        Var<float> sumsq = def(0.0f);
        Var<uint> i = lane;
        $while (i < H) {
            const auto value = x.read(base + i);
            sumsq += value * value;
            i += width;
        };
        const auto inv = rsqrt(block_sum(sumsq) / cast<float>(H) + eps);
        i = lane;
        $while (i < H) {
            Var<float> value = x.read(base + i) * inv;
            $if (has_weight != 0u) { value *= gamma.read(i); };
            y.write(base + i, value);
            i += width;
        };
    };
    return kernel;
}

[[nodiscard]] ResidualRmsNormKernel make_residual_rms_norm_kernel() noexcept {
    ResidualRmsNormKernel kernel = [](BufferFloat projection, BufferFloat residual, BufferFloat gamma,
                                      BufferFloat residual_out, BufferFloat norm_out,
                                      Var<uint> rows, Var<uint> H, Var<float> eps) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_residual_rms_norm");
        const auto row = ocr_linear_id();
        $if (row < rows) {
            const auto base = row * H;
            Var<float> sumsq = def(0.0f);
            $for (i, H) {
                const auto v = projection.read(base + i) + residual.read(base + i);
                sumsq += v * v;
            };
            const auto inv = rsqrt(sumsq / cast<float>(H) + eps);
            $for (i, H) {
                const auto v = projection.read(base + i) + residual.read(base + i);
                residual_out.write(base + i, v);
                norm_out.write(base + i, v * inv * gamma.read(i));
            };
        };
    };
    return kernel;
}

[[nodiscard]] L2NormKernel make_l2_norm_kernel() noexcept {
    L2NormKernel kernel = [](BufferFloat x, BufferFloat y, Var<uint> rows, Var<uint> H, Var<float> floor) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_l2_norm");
        const auto row = ocr_linear_id();
        $if (row < rows) {
            const auto base = row * H;
            Var<float> sumsq = def(0.0f);
            $for (i, H) {
                const auto v = x.read(base + i);
                sumsq += v * v;
            };
            const auto norm = max(sqrt(sumsq), floor);
            $for (i, H) { y.write(base + i, x.read(base + i) / norm); };
        };
    };
    return kernel;
}

[[nodiscard]] RowSumKernel make_row_sum_kernel() noexcept {
    RowSumKernel kernel = [](BufferFloat x, BufferFloat out, Var<uint> rows, Var<uint> H) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_row_sum");
        const auto row = ocr_linear_id();
        $if (row < rows) {
            const auto base = row * H;
            Var<float> acc = def(0.0f);
            $for (i, H) { acc += x.read(base + i); };
            out.write(row, acc);
        };
    };
    return kernel;
}

[[nodiscard]] ResidualAddKernel make_residual_add_kernel() noexcept {
    ResidualAddKernel kernel = [](BufferFloat a, BufferFloat b, BufferFloat out, Var<uint> count) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_residual_add");
        const auto i = ocr_linear_id();
        $if (i < count) { out.write(i, a.read(i) + b.read(i)); };
    };
    return kernel;
}

[[nodiscard]] Result<LaunchPlan1D> plan_norm_dispatch(uint64_t rows) {
    return plan_threads(rows, 256u);
}

[[nodiscard]] Result<LaunchPlan1D> plan_norm_cooperative_dispatch(uint64_t rows) {
    if (rows > std::numeric_limits<uint64_t>::max() / 256ull) {
        return Result<LaunchPlan1D>::failure(StatusCode::invalid_argument,
                                             "cooperative norm thread count overflow", "rows*256");
    }
    return plan_threads(rows * 256ull, 256u);
}

[[nodiscard]] Result<LaunchPlan1D> plan_residual_add_dispatch(uint64_t count) {
    return plan_threads(count, 256u);
}

[[nodiscard]] luisa::string norm_variant_key(luisa::string_view op) {
    return make_shader_key("norm", op);
}

}// namespace luisa::ocr