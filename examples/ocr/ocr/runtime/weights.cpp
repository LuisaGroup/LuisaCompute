// OCR engine -- immutable per-component device weights (Phase 2).
#include "ocr/runtime/weights.h"

#include <cstring>
#include <utility>

#include "ocr/quantization.h"
#include "ocr/runtime/dispatch.h"

namespace luisa::ocr {

namespace {

// Word counts for the packed storage of `count` logical elements.
[[nodiscard]] Result<uint64_t> packed_word_count(TensorEncoding encoding, uint64_t count) {
    using R = Result<uint64_t>;
    uint64_t per_word = 0;
    switch (encoding) {
        case TensorEncoding::fp16:
        case TensorEncoding::bf16: per_word = 2u; break;
        case TensorEncoding::int4_packed: per_word = 8u; break;
        case TensorEncoding::int8: per_word = 4u; break;
        default:
            return R::failure(StatusCode::invalid_argument, "encoding is not a packed storage", "encoding");
    }
    return R::success((count + per_word - 1u) / per_word);
}

// Device bytes for one tensor's storage (codes + scales + zero points).
[[nodiscard]] Result<uint64_t> tensor_device_bytes(const PackageTensor &tensor) {
    using R = Result<uint64_t>;
    uint64_t total = 0;
    auto add = [&](uint64_t bytes) -> Status {
        uint64_t next = 0;
        if (!checked_add(total, bytes, next)) {
            return Status::error(StatusCode::invalid_argument, "weight byte total overflows uint64", tensor.name);
        }
        total = next;
        return Status::success();
    };
    if (tensor.encoding == TensorEncoding::fp32) {
        auto b = tensor_byte_size(TensorDType::f32, tensor.element_count);
        if (!b.ok()) { return R::failure(b.status); }
        auto s = add(*b.value);
        if (!s.ok()) { return R::failure(s); }
    } else {
        auto words = packed_word_count(tensor.encoding, tensor.element_count);
        if (!words.ok()) { return R::failure(words.status); }
        auto b = tensor_byte_size(TensorDType::u32, *words.value);
        if (!b.ok()) { return R::failure(b.status); }
        auto s = add(*b.value);
        if (!s.ok()) { return R::failure(s); }
        if (is_quantized_encoding(tensor.encoding)) {
            uint64_t groups = 0;
            if (!group_count_for(tensor.element_count, tensor.group_size, groups)) {
                return R::failure(StatusCode::invalid_argument, "group count overflow", tensor.name);
            }
            auto sb = tensor_byte_size(TensorDType::f32, groups);
            if (!sb.ok()) { return R::failure(sb.status); }
            s = add(*sb.value);
            if (!s.ok()) { return R::failure(s); }
            if (tensor.quant_kind == QuantKind::int4_asymmetric) {
                auto zb = tensor_byte_size(TensorDType::u32, groups);
                if (!zb.ok()) { return R::failure(zb.status); }
                s = add(*zb.value);
                if (!s.ok()) { return R::failure(s); }
            }
        }
    }
    return R::success(total);
}

// Logical fp32 compute descriptor from a package tensor.
[[nodiscard]] Result<TensorDesc> logical_desc(const PackageTensor &tensor) {
    using R = Result<TensorDesc>;
    TensorDesc desc;
    desc.dtype = TensorDType::f32;
    desc.shape = tensor.shape;
    desc.strides = tensor.strides;
    switch (tensor.layout) {
        case TensorLayout::kn: desc.layout = TensorLayoutTag::kn; break;
        case TensorLayout::nk: desc.layout = TensorLayoutTag::nk; break;
        default: desc.layout = TensorLayoutTag::row_major; break;
    }
    auto v = desc.normalize();
    if (!v.ok()) { return R::failure(v); }
    v = desc.validate();
    if (!v.ok()) { return R::failure(v); }
    return R::success(std::move(desc));
}

}// namespace

[[nodiscard]] uint64_t DeviceWeight::resident_bytes() const noexcept {
    uint64_t total = 0;
    auto add = [&](uint64_t bytes) {
        uint64_t next = 0;
        if (!checked_add(total, bytes, next)) { total = UINT64_MAX; } else { total = next; }
    };
    if (dense) { add(dense.size_bytes()); }
    if (packed) { add(packed.size_bytes()); }
    if (scales) { add(scales.size_bytes()); }
    if (zero_points) { add(zero_points.size_bytes()); }
    return total;
}

[[nodiscard]] Result<uint64_t> plan_weight_device_bytes(luisa::span<const PackageTensor> tensors) {
    using R = Result<uint64_t>;
    uint64_t total = 0;
    for (const auto &tensor : tensors) {
        auto bytes = tensor_device_bytes(tensor);
        if (!bytes.ok()) { return R::failure(bytes.status); }
        uint64_t next = 0;
        if (!checked_add(total, *bytes.value, next)) {
            return R::failure(StatusCode::invalid_argument, "weight byte total overflows uint64", "tensors");
        }
        total = next;
    }
    return R::success(total);
}

[[nodiscard]] Status WeightUploadBatch::prepare(compute::Device &device, PackageReader &reader,
                                                luisa::string_view component,
                                                luisa::span<const PackageTensor> tensors) {
    if (_taken || !_weights.empty()) {
        return Status::error(StatusCode::invalid_argument, "upload batch already prepared", "batch");
    }
    for (const auto &tensor : tensors) {
        DeviceWeight weight;
        weight.name = tensor.name;
        weight.component = luisa::string{component};
        weight.encoding = tensor.encoding;
        weight.quant_kind = tensor.quant_kind;
        weight.group_size = tensor.group_size;
        weight.element_count = tensor.element_count;
        auto desc = logical_desc(tensor);
        if (!desc.ok()) { return desc.status; }
        weight.logical = std::move(*desc.value);

        Staging staging;
        if (tensor.encoding == TensorEncoding::fp32) {
            staging.dense_f32.resize(static_cast<size_t>(tensor.element_count));
            auto r = reader.read_tensor_f32(tensor, staging.dense_f32);
            if (!r.ok()) { return r; }
            auto words = tensor.element_count;
            if (words == 0u) { words = 1u; }// device buffers must be nonzero
            weight.dense = device.create_buffer<float>(static_cast<size_t>(words));
            if (!weight.dense) {
                return Status::error(StatusCode::out_of_memory, "failed to create weight buffer", tensor.name);
            }
        } else {
            auto words = packed_word_count(tensor.encoding, tensor.element_count);
            if (!words.ok()) { return words.status; }
            // Stage the packed codes as little-endian uint32 words. The on-disk
            // payload is already little-endian, so this is a byte copy with a
            // zero-padded tail word.
            const uint64_t word_count = *words.value == 0u ? 1u : *words.value;
            staging.packed_u32.resize(static_cast<size_t>(word_count), 0u);
            luisa::vector<std::byte> raw(static_cast<size_t>(tensor.payload_size));
            auto r = reader.read_tensor_payload(tensor, raw);
            if (!r.ok()) { return r; }
            std::memcpy(staging.packed_u32.data(), raw.data(), raw.size());
            weight.packed = device.create_buffer<uint32_t>(static_cast<size_t>(word_count));
            if (!weight.packed) {
                return Status::error(StatusCode::out_of_memory, "failed to create weight buffer", tensor.name);
            }
            if (is_quantized_encoding(tensor.encoding)) {
                uint64_t groups = 0;
                if (!group_count_for(tensor.element_count, tensor.group_size, groups)) {
                    return Status::error(StatusCode::invalid_argument, "group count overflow", tensor.name);
                }
                weight.group_count = groups;
                const uint64_t scale_count = groups == 0u ? 1u : groups;
                staging.scales.resize(static_cast<size_t>(groups), 0.0f);
                auto sr = reader.read_tensor_scales(tensor, staging.scales);
                if (!sr.ok()) { return sr; }
                weight.scales = device.create_buffer<float>(static_cast<size_t>(scale_count));
                if (!weight.scales) {
                    return Status::error(StatusCode::out_of_memory, "failed to create scales buffer", tensor.name);
                }
                if (tensor.quant_kind == QuantKind::int4_asymmetric) {
                    staging.zero_points.resize(static_cast<size_t>(groups), 0u);
                    luisa::vector<uint8_t> zp(static_cast<size_t>(groups));
                    auto zr = reader.read_tensor_zero_points(tensor, zp);
                    if (!zr.ok()) { return zr; }
                    for (uint64_t g = 0; g < groups; g++) { staging.zero_points[static_cast<size_t>(g)] = zp[static_cast<size_t>(g)]; }
                    weight.zero_points = device.create_buffer<uint32_t>(static_cast<size_t>(scale_count));
                    if (!weight.zero_points) {
                        return Status::error(StatusCode::out_of_memory, "failed to create zero-point buffer", tensor.name);
                    }
                }
            }
        }
        uint64_t staged = staging.dense_f32.size() * 4u + staging.packed_u32.size() * 4u +
                          staging.scales.size() * 4u + staging.zero_points.size() * 4u;
        uint64_t next = 0;
        if (!checked_add(_staging_bytes, staged, next)) {
            return Status::error(StatusCode::invalid_argument, "staging byte total overflows uint64", tensor.name);
        }
        _staging_bytes = next;
        _weights.emplace_back(std::move(weight));
        _staging.emplace_back(std::move(staging));
    }
    return Status::success();
}

[[nodiscard]] Status WeightUploadBatch::record(compute::CommandList &list) const {
    if (_staging_released) {
        return Status::error(StatusCode::invalid_argument, "staging already released; cannot record upload", "batch");
    }
    if (_weights.size() != _staging.size()) {
        return Status::error(StatusCode::internal_error, "upload batch is inconsistent", "batch");
    }
    for (size_t i = 0; i < _weights.size(); i++) {
        const auto &weight = _weights[i];
        const auto &staging = _staging[i];
        if (weight.encoding == TensorEncoding::fp32) {
            const auto count = weight.element_count;
            if (count != 0u) {
                // Typed staging span (safe-mode compatible; no raw-pointer copy).
                luisa::span<const float> src{staging.dense_f32.data(), static_cast<size_t>(count)};
                list << weight.dense.view(0u, static_cast<size_t>(count)).copy_from(src);
            }
        } else {
            luisa::span<const uint32_t> src{staging.packed_u32};
            list << weight.packed.view().copy_from(src);
            if (weight.scales) {
                list << weight.scales.view().copy_from(luisa::span<const float>{staging.scales});
            }
            if (weight.zero_points) {
                list << weight.zero_points.view().copy_from(luisa::span<const uint32_t>{staging.zero_points});
            }
        }
    }
    return Status::success();
}

[[nodiscard]] Status WeightUploadBatch::release_staging() {
    if (!_completed) {
        return Status::error(StatusCode::invalid_argument,
                             "staging is releasable only after upload completion (complete() after synchronize)", "batch");
    }
    if (!_staging_released) {
        for (auto &staging : _staging) {
            staging.dense_f32.clear();
            staging.dense_f32.shrink_to_fit();
            staging.packed_u32.clear();
            staging.packed_u32.shrink_to_fit();
            staging.scales.clear();
            staging.scales.shrink_to_fit();
            staging.zero_points.clear();
            staging.zero_points.shrink_to_fit();
        }
        _staging_released = true;
        _staging_bytes = 0u;
    }
    return Status::success();
}

[[nodiscard]] uint64_t WeightUploadBatch::device_bytes() const noexcept {
    uint64_t total = 0;
    for (const auto &weight : _weights) {
        uint64_t next = 0;
        if (!checked_add(total, weight.resident_bytes(), next)) { return UINT64_MAX; }
        total = next;
    }
    return total;
}

[[nodiscard]] Result<luisa::vector<DeviceWeight>> WeightUploadBatch::take_weights() {
    using R = Result<luisa::vector<DeviceWeight>>;
    if (!_completed || !_staging_released) {
        return R::failure(StatusCode::invalid_argument,
                          "weights are installable only after completion and staging release", "batch");
    }
    if (_taken) {
        return R::failure(StatusCode::invalid_argument, "weights already taken", "batch");
    }
    _taken = true;
    return R::success(std::move(_weights));
}

[[nodiscard]] Status WeightStore::install(WeightUploadBatch &batch) {
    auto taken = batch.take_weights();
    if (!taken.ok()) { return taken.status; }
    luisa::string component;
    for (auto &weight : *taken.value) {
        if (component.empty()) { component = weight.component; }
        auto &map = _components[component];
        if (map.find(weight.name) != map.end()) {
            return Status::error(StatusCode::invalid_argument,
                                 "duplicate weight name in component", weight.name);
        }
        map.emplace(weight.name, std::move(weight));
    }
    return Status::success();
}

[[nodiscard]] Status WeightStore::upload_component_sync(compute::Device &device, compute::Stream &stream,
                                                        PackageReader &reader, luisa::string_view component,
                                                        luisa::span<const luisa::string> names) {
    // Select tensors (component-selective loading).
    luisa::vector<PackageTensor> selected;
    if (names.empty()) {
        auto all = reader.tensors();
        selected.assign(all.begin(), all.end());
    } else {
        for (const auto &name : names) {
            const auto *tensor = reader.find(name);
            if (tensor == nullptr) {
                return Status::error(StatusCode::not_found,
                                     luisa::string{"weight tensor '"} + name + "' not found in package", name);
            }
            selected.emplace_back(*tensor);
        }
    }
    WeightUploadBatch batch;
    auto p = batch.prepare(device, reader, component, selected);
    if (!p.ok()) { return p; }
    auto list = compute::CommandList::create();
    auto r = batch.record(list);
    if (!r.ok()) { return r; }
    stream << list.commit() << compute::synchronize();
    // Stream drained: upload completion proven, staging now releasable.
    batch.complete();
    auto rel = batch.release_staging();
    if (!rel.ok()) { return rel; }
    return install(batch);
}

[[nodiscard]] const DeviceWeight *WeightStore::find(luisa::string_view component, luisa::string_view name) const noexcept {
    auto it = _components.find(luisa::string{component});
    if (it == _components.end()) { return nullptr; }
    auto wit = it->second.find(luisa::string{name});
    if (wit == it->second.end()) { return nullptr; }
    return &wit->second;
}

[[nodiscard]] uint64_t WeightStore::component_bytes(luisa::string_view component) const noexcept {
    auto it = _components.find(luisa::string{component});
    if (it == _components.end()) { return 0u; }
    uint64_t total = 0;
    for (const auto &entry : it->second) {
        uint64_t next = 0;
        if (!checked_add(total, entry.second.resident_bytes(), next)) { return UINT64_MAX; }
        total = next;
    }
    return total;
}

[[nodiscard]] uint64_t WeightStore::total_bytes() const noexcept {
    uint64_t total = 0;
    for (const auto &entry : _components) {
        uint64_t next = 0;
        if (!checked_add(total, component_bytes(entry.first), next)) { return UINT64_MAX; }
        total = next;
    }
    return total;
}

[[nodiscard]] luisa::vector<luisa::string> WeightStore::components() const {
    luisa::vector<luisa::string> out;
    out.reserve(_components.size());
    for (const auto &entry : _components) { out.emplace_back(entry.first); }
    return out;
}

[[nodiscard]] luisa::vector<luisa::string> WeightStore::tensor_names(luisa::string_view component) const {
    luisa::vector<luisa::string> out;
    auto it = _components.find(luisa::string{component});
    if (it == _components.end()) { return out; }
    out.reserve(it->second.size());
    for (const auto &entry : it->second) { out.emplace_back(entry.first); }
    return out;
}

[[nodiscard]] uint64_t WeightStore::unload_component(luisa::string_view component) {
    auto bytes = component_bytes(component);
    _components.erase(luisa::string{component});
    return bytes;
}

}// namespace luisa::ocr