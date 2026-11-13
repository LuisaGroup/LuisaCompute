#pragma once
// OCR engine -- Surya device executor (Phase 9). DEVICE-SAFE header.
//
// detector (EfficientViT-Segformer-style CNN + DB head) -> line crops -> vision
// merger -> hybrid GDN decode. The model contract / result types / compose live
// in the host-safe models/surya.h; this header is the device executor (mirrors
// the CNN architecture: runtime/cnn_executor.* vs models/pp_ocrv6.*).
//
// Ownership (runtime/executor.h): borrows an installed immutable WeightStore,
// owns a per-request detector DeviceWorkspace + TypedShaderCaches, and embeds a
// HybridVlmGraph (vision + hybrid decode) reused per line crop. All launches go
// through the safe dispatch adapters; all host transfers use luisa::span. No
// exceptions / RTTI. SYNTHETIC ONLY.
#include <cstdint>

#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "ocr/models/surya.h"
#include "ocr/ops/ops_conv.h"
#include "ocr/ops/ops_ctc.h"
#include "ocr/ops/ops_gdn.h"
#include "ocr/runtime/dispatch_adapter.h"
#include "ocr/runtime/hybrid_gdn_executor.h"
#include "ocr/runtime/shader_cache.h"
#include "ocr/runtime/weights.h"
#include "ocr/runtime/workspace.h"

namespace luisa::ocr {

// The Surya executor: detector -> line crops -> vision -> hybrid decode.
// NOT thread-safe. Reusable across requests (hybrid cache reset per line). The
// stream is driven internally (host DB postprocess + per-line decode need host
// data between device stages). Blank pages produce a valid EMPTY result (no
// lines, no hallucinated text, no fabricated confidence).
class SuryaExecutor final {
public:
    SuryaExecutor() noexcept = default;

    // TOKENIZER LIFETIME CONTRACT (S2): SuryaConfig::tokenizer is a BORROWED
    // const Tokenizer* that the CALLER owns. It is REQUIRED (create() refuses a
    // null pointer or a vocabulary smaller than the decoder's) because Surya is a
    // text recognizer: without it every line's text would be silently empty. It
    // must outlive create() AND every recognize() call; the executor keeps only
    // the non-owning pointer (never copies/frees it) and passes it to
    // HybridVlmGraph::run, which uses it only inside generate() and never retains
    // it. recognize() re-checks it so a cleared pointer fails closed instead of
    // emitting empty content.
    [[nodiscard]] Status create(compute::Device &device, compute::Stream &stream,
                                const SuryaConfig &config, const WeightStore &weights);

    [[nodiscard]] Result<SuryaRunResult> recognize(compute::Stream &stream,
                                                   luisa::span<const SuryaPageInput> pages);

    [[nodiscard]] const SuryaConfig &config() const noexcept { return _config; }
    [[nodiscard]] uint64_t weight_bytes() const noexcept { return _weight_bytes; }
    [[nodiscard]] uint64_t workspace_bytes() const noexcept { return _det_ws.capacity_bytes(); }
    [[nodiscard]] const ShaderKeyRegistry &shader_registry() const noexcept { return _shader_registry; }

    // Layered-golden download (last run's detector probability map; caller drains).
    [[nodiscard]] Status download_detector_prob(compute::Stream &stream, luisa::vector<float> &out) const;
    [[nodiscard]] uint32_t last_map_height() const noexcept { return _last_map_h; }
    [[nodiscard]] uint32_t last_map_width() const noexcept { return _last_map_w; }

private:
    struct ConvWeights {
        compute::BufferView<float> weight, bias;
    };
    [[nodiscard]] Status resolve_weights(const WeightStore &weights);
    [[nodiscard]] Status reserve_workspace();
    [[nodiscard]] Status compile_shaders();
    [[nodiscard]] Status record_conv(compute::CommandList &list, const SuryaConvStage &stage,
                                     const ConvWeights &w, compute::BufferView<float> in,
                                     compute::BufferView<float> out, uint32_t in_h, uint32_t in_w,
                                     uint32_t &out_h, uint32_t &out_w);
    [[nodiscard]] Status record_detector(compute::CommandList &list, compute::BufferView<float> pixels,
                                         uint32_t &map_h, uint32_t &map_w);
    [[nodiscard]] compute::BufferView<float> dws(const WorkspaceAllocation &a) const noexcept {
        return a.view<float>(_det_ws.buffer());
    }

    SuryaConfig _config;
    compute::Device *_device{nullptr};
    DeviceWorkspace _det_ws;
    HybridVlmGraph _line_graph;// vision + hybrid decode per line crop
    uint64_t _weight_bytes{0};
    ConvWeights _stem, _stage1, _stage2, _db;
    WorkspaceAllocation _det_pixels, _det_a, _det_b, _det_prob;
    uint32_t _last_map_h{0}, _last_map_w{0};

    ShaderKeyRegistry _shader_registry;
    TypedShaderCache<Conv2dKernel> _conv_cache;
    TypedShaderCache<CnnActKernel> _act_cache;
    TypedShaderCache<ElementwiseSigmoidKernel> _sigmoid_cache;
    const TypedShaderCache<Conv2dKernel>::ShaderT *_sh_conv{nullptr};
    const TypedShaderCache<CnnActKernel>::ShaderT *_sh_act[4]{nullptr, nullptr, nullptr, nullptr};
    const TypedShaderCache<ElementwiseSigmoidKernel>::ShaderT *_sh_sigmoid{nullptr};
};

}// namespace luisa::ocr
