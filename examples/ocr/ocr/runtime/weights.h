#pragma once
// OCR engine -- immutable per-component device weights (Phase 2).
//
// Loads tensors from a Phase 1 PackageReader into device buffers, keeping
// STORAGE encoding explicit and separate from the FP32 compute view:
//   fp32        -> Buffer<float>  [element_count]
//   fp16/bf16   -> Buffer<uint>   [ceil(element_count/2)] two halves per word,
//                  element 2j in bits[15:0], 2j+1 in bits[31:16] (little-endian
//                  pair, identical to the on-disk byte order -> raw copy).
//   int4_packed -> Buffer<uint>   [ceil(element_count/8)] eight nibbles per
//                  word, element 8j+c in bits[4c+3:4c] (low nibble first, the
//                  Phase 1 convention). Plus Buffer<float> scales[group_count]
//                  and, for asymmetric, Buffer<uint> zero_points[group_count].
//   int8        -> Buffer<uint>   [ceil(element_count/4)] four signed bytes per
//                  word, element 4j+c in bits[8c+7:8c], sign-extended on read.
//                  Plus Buffer<float> scales[group_count].
// No storage format is ever reinterpreted as another; dequantization happens
// in the operator kernels (ops_gemm) with FP32 accumulation.
//
// Upload lifetime is EXPLICIT. A WeightUploadBatch stages host bytes, creates
// device buffers, and records luisa::span copy commands. The staging memory is
// releasable ONLY after the caller proves upload completion (complete() after
// stream.synchronize()); release_staging() refuses before that. A synchronous
// upload_component_sync() convenience wraps prepare->commit->synchronize->
// complete->release->install and is safe because it drains the stream first.
//
// Resident bytes are the ACTUAL device buffer bytes (sum of size_bytes()),
// tracked per component for memory_budget. Weights are immutable after
// install; there is no in-place mutation API.
#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "ocr/package.h"
#include "ocr/runtime/tensor.h"
#include "ocr/status.h"

namespace luisa::ocr {

// One immutable device weight. Only the buffers matching `encoding` are
// populated; the others are default (invalid). `logical` is the FP32 compute
// view of the tensor shape/layout (dtype f32) -- NOT the storage word count.
struct DeviceWeight {
    luisa::string name;
    luisa::string component;
    TensorEncoding encoding{TensorEncoding::fp32};
    QuantKind quant_kind{QuantKind::none};
    uint32_t group_size{0};
    uint64_t element_count{0};
    uint64_t group_count{0};
    TensorDesc logical;// fp32 compute descriptor (shape/layout)

    compute::Buffer<float> dense;       // fp32
    compute::Buffer<uint32_t> packed;   // fp16/bf16/int4/int8 code words
    compute::Buffer<float> scales;      // quantized
    compute::Buffer<uint32_t> zero_points;// int4 asymmetric

    [[nodiscard]] uint64_t resident_bytes() const noexcept;
};

// Host-side device-byte planning for a set of tensors (no allocation). Used
// by memory_budget to preflight the exact upload requirement.
[[nodiscard]] Result<uint64_t> plan_weight_device_bytes(luisa::span<const PackageTensor> tensors);

// A staged upload: host bytes + created device buffers + recorded commands.
// Staging stays alive until complete() + release_staging().
class WeightUploadBatch {
public:
    WeightUploadBatch() noexcept = default;

    // Stage `tensors` from `reader` into host memory and create device
    // buffers. `component` tags every weight. Does NOT touch the stream.
    [[nodiscard]] Status prepare(compute::Device &device, PackageReader &reader,
                                 luisa::string_view component,
                                 luisa::span<const PackageTensor> tensors);

    // Append the upload copy commands (luisa::span based) to `list`.
    [[nodiscard]] Status record(compute::CommandList &list) const;

    // Caller MUST call this only after the stream that consumed record() has
    // synchronized (upload completion proven). Marks staging releasable.
    void complete() noexcept { _completed = true; }
    [[nodiscard]] bool completed() const noexcept { return _completed; }

    // Release host staging. Refuses (invalid_argument) before complete().
    [[nodiscard]] Status release_staging();
    [[nodiscard]] uint64_t staging_bytes() const noexcept { return _staging_bytes; }
    [[nodiscard]] bool staging_released() const noexcept { return _staging_released; }

    [[nodiscard]] luisa::span<const DeviceWeight> weights() const noexcept { return _weights; }
    [[nodiscard]] uint64_t device_bytes() const noexcept;

    // Move the (completed, staging-released) weights out for installation.
    [[nodiscard]] Result<luisa::vector<DeviceWeight>> take_weights();

private:
    struct Staging {
        luisa::vector<float> dense_f32;     // fp32 payload (typed for span upload)
        luisa::vector<uint32_t> packed_u32; // packed code words (typed)
        luisa::vector<float> scales;
        luisa::vector<uint32_t> zero_points;
    };
    luisa::vector<DeviceWeight> _weights;
    luisa::vector<Staging> _staging;
    uint64_t _staging_bytes{0};
    bool _completed{false};
    bool _staging_released{false};
    bool _taken{false};
};

// Immutable weight store, keyed by component then tensor name.
class WeightStore {
public:
    WeightStore() noexcept = default;

    // Install a completed batch (staging must be released). Rejects duplicate
    // tensor names within a component.
    [[nodiscard]] Status install(WeightUploadBatch &batch);

    // Synchronous component load: prepare -> commit -> synchronize ->
    // complete -> release staging -> install. Safe: drains the stream before
    // releasing staging. `names` empty = every tensor in the reader.
    [[nodiscard]] Status upload_component_sync(compute::Device &device, compute::Stream &stream,
                                               PackageReader &reader, luisa::string_view component,
                                               luisa::span<const luisa::string> names = {});

    [[nodiscard]] const DeviceWeight *find(luisa::string_view component, luisa::string_view name) const noexcept;
    [[nodiscard]] uint64_t component_bytes(luisa::string_view component) const noexcept;
    [[nodiscard]] uint64_t total_bytes() const noexcept;
    [[nodiscard]] luisa::vector<luisa::string> components() const;
    // Tensor names installed under one component (empty if absent). Used by
    // executors to reject unexpected tensors against their declared inventory.
    [[nodiscard]] luisa::vector<luisa::string> tensor_names(luisa::string_view component) const;
    // Drop a component's weights (deferred device destruction). Returns bytes freed.
    [[nodiscard]] uint64_t unload_component(luisa::string_view component);

private:
    // component -> (name -> weight)
    luisa::unordered_map<luisa::string, luisa::unordered_map<luisa::string, DeviceWeight>> _components;
};

}// namespace luisa::ocr