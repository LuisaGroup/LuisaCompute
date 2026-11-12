// OCR engine -- checked engine-side memory accounting (Phase 2). Host-only.
#include "ocr/runtime/memory_budget.h"

#include <utility>

#include <luisa/core/stl/format.h>
#include <luisa/core/stl/hash.h>

#include "ocr/quantization.h"

namespace luisa::ocr {

[[nodiscard]] luisa::string_view memory_category_name(MemoryCategory category) noexcept {
    switch (category) {
        case MemoryCategory::weights: return "weights";
        case MemoryCategory::workspace: return "workspace";
        case MemoryCategory::kv_cache: return "kv_cache";
        case MemoryCategory::shader_cache: return "shader_cache";
        case MemoryCategory::host_staging: return "host_staging";
        default: return "unknown";
    }
}

[[nodiscard]] uint64_t MemoryBudget::KeyHash::operator()(const Key &key) const noexcept {
    uint64_t h = luisa::hash<luisa::string>{}(key.component);
    const auto category = static_cast<uint32_t>(key.category);
    return luisa::hash<uint32_t>{}(category, h);
}

[[nodiscard]] Status MemoryBudget::require(luisa::string_view component, MemoryCategory category, uint64_t bytes) {
    Key key{luisa::string{component}, category};
    auto it = _requirements.find(key);
    const uint64_t current = it == _requirements.end() ? 0u : it->second;
    uint64_t next = 0;
    if (!checked_add(current, bytes, next)) {
        return Status::error(StatusCode::invalid_argument,
                             "memory requirement overflows uint64", luisa::string{component});
    }
    _requirements[key] = next;
    return Status::success();
}

[[nodiscard]] Status MemoryBudget::set_requirement(luisa::string_view component, MemoryCategory category, uint64_t bytes) {
    Key key{luisa::string{component}, category};
    _requirements[key] = bytes;
    return Status::success();
}

void MemoryBudget::release_component(luisa::string_view component) {
    for (auto it = _requirements.begin(); it != _requirements.end();) {
        if (it->first.component == component) {
            it = _requirements.erase(it);
        } else {
            ++it;
        }
    }
}

[[nodiscard]] uint64_t MemoryBudget::category_bytes(MemoryCategory category) const noexcept {
    uint64_t total = 0;
    for (const auto &entry : _requirements) {
        if (entry.first.category != category) { continue; }
        uint64_t next = 0;
        if (!checked_add(total, entry.second, next)) { return UINT64_MAX; }
        total = next;
    }
    return total;
}

[[nodiscard]] uint64_t MemoryBudget::component_bytes(luisa::string_view component) const noexcept {
    uint64_t total = 0;
    for (const auto &entry : _requirements) {
        if (entry.first.component != component) { continue; }
        uint64_t next = 0;
        if (!checked_add(total, entry.second, next)) { return UINT64_MAX; }
        total = next;
    }
    return total;
}

[[nodiscard]] uint64_t MemoryBudget::device_bytes() const noexcept {
    uint64_t total = 0;
    for (auto category : {MemoryCategory::weights, MemoryCategory::workspace,
                          MemoryCategory::kv_cache, MemoryCategory::shader_cache}) {
        uint64_t next = 0;
        if (!checked_add(total, category_bytes(category), next)) { return UINT64_MAX; }
        total = next;
    }
    return total;
}

[[nodiscard]] uint64_t MemoryBudget::host_bytes() const noexcept {
    return category_bytes(MemoryCategory::host_staging);
}

[[nodiscard]] uint64_t MemoryBudget::total_bytes() const noexcept {
    uint64_t total = 0;
    if (!checked_add(device_bytes(), host_bytes(), total)) { return UINT64_MAX; }
    return total;
}

[[nodiscard]] Status MemoryBudget::preflight() const {
    const auto device = device_bytes();
    const auto host = host_bytes();
    if (_limits.max_device_bytes != 0u && device > _limits.max_device_bytes) {
        return Status::error(StatusCode::out_of_memory,
                             luisa::format("device memory requirement {} bytes exceeds the limit {} bytes",
                                           device, _limits.max_device_bytes),
                             "max_device_bytes");
    }
    if (_limits.max_host_bytes != 0u && host > _limits.max_host_bytes) {
        return Status::error(StatusCode::out_of_memory,
                             luisa::format("host staging requirement {} bytes exceeds the limit {} bytes",
                                           host, _limits.max_host_bytes),
                             "max_host_bytes");
    }
    return Status::success();
}

[[nodiscard]] Status MemoryBudget::preflight_additional_device(uint64_t extra_bytes) const {
    uint64_t required = 0;
    if (!checked_add(device_bytes(), extra_bytes, required)) {
        return Status::error(StatusCode::invalid_argument, "memory requirement overflows uint64", "extra");
    }
    if (_limits.max_device_bytes != 0u && required > _limits.max_device_bytes) {
        return Status::error(StatusCode::out_of_memory,
                             luisa::format("device memory requirement {} bytes (including {} additional) exceeds the limit {} bytes",
                                           required, extra_bytes, _limits.max_device_bytes),
                             "max_device_bytes");
    }
    return Status::success();
}

[[nodiscard]] luisa::vector<MemoryRequirement> MemoryBudget::requirements() const {
    luisa::vector<MemoryRequirement> out;
    out.reserve(_requirements.size());
    for (const auto &entry : _requirements) {
        out.push_back(MemoryRequirement{entry.first.component, entry.first.category, entry.second});
    }
    return out;
}

[[nodiscard]] Result<uint64_t> plan_kv_cache_bytes(uint64_t full_attention_layers, uint64_t requests,
                                                   uint64_t tokens, uint64_t kv_heads, uint64_t head_dim,
                                                   uint64_t dtype_bytes) {
    using R = Result<uint64_t>;
    uint64_t total = 1u;
    for (auto factor : {full_attention_layers, requests, tokens, kv_heads, head_dim, uint64_t{2}, dtype_bytes}) {
        uint64_t next = 0;
        if (!checked_mul(total, factor, next)) {
            return R::failure(StatusCode::invalid_argument, "KV cache byte plan overflows uint64", "kv");
        }
        total = next;
    }
    return R::success(total);
}

}// namespace luisa::ocr