#pragma once
// OCR engine -- checked engine-side memory accounting (Phase 2). Host-only.
//
// Tracks weight / workspace / KV-cache / shader-cache byte requirements per
// component and preflights them against caller limits BEFORE any allocation.
// There is NO invented universal device-memory query: the engine accounts for
// what IT allocates and compares against explicit caller limits (api.h
// MemoryLimits). Backend-specific queries (e.g. CUDA free_memory) may be added
// later as an OPTIONAL refinement, never as the portable baseline.
//
// Every sum is checked 64-bit; a requirement that overflows or exceeds a limit
// returns a useful Status naming the offending component and the required vs
// available bytes, so callers can shrink a batch or raise a limit deliberately.
#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

#include "ocr/status.h"

namespace luisa::ocr {

// Resource limits. 0 = engine default / unset (the engine still enforces its
// own checked budgets). Limits are preflighted before allocation. Defined
// here (the memory module) and re-exported by api.h so device-side runtime
// headers can use the budget without pulling the host-only recipe contracts.
struct MemoryLimits {
    uint64_t max_device_bytes{0};
    uint64_t max_host_bytes{0};
};

enum class MemoryCategory : uint32_t {
    weights = 0,
    workspace = 1,
    kv_cache = 2,
    shader_cache = 3,
    host_staging = 4,
};

[[nodiscard]] luisa::string_view memory_category_name(MemoryCategory category) noexcept;

// One component's requirement in one category.
struct MemoryRequirement {
    luisa::string component;
    MemoryCategory category{MemoryCategory::workspace};
    uint64_t bytes{0};
};

// A checked budget ledger. Add requirements, then preflight against limits.
class MemoryBudget {
public:
    MemoryBudget() noexcept = default;
    explicit MemoryBudget(MemoryLimits limits) noexcept : _limits(limits) {}

    void set_limits(MemoryLimits limits) noexcept { _limits = limits; }
    [[nodiscard]] const MemoryLimits &limits() const noexcept { return _limits; }

    // Record a requirement (checked accumulate per component+category).
    [[nodiscard]] Status require(luisa::string_view component, MemoryCategory category, uint64_t bytes);
    // Replace the requirement for a component+category (e.g. a resized arena).
    [[nodiscard]] Status set_requirement(luisa::string_view component, MemoryCategory category, uint64_t bytes);
    // Drop a component's requirements (unload).
    void release_component(luisa::string_view component);

    [[nodiscard]] uint64_t category_bytes(MemoryCategory category) const noexcept;
    [[nodiscard]] uint64_t component_bytes(luisa::string_view component) const noexcept;
    // Device-side total = weights + workspace + kv_cache + shader_cache.
    [[nodiscard]] uint64_t device_bytes() const noexcept;
    [[nodiscard]] uint64_t host_bytes() const noexcept;// host_staging
    [[nodiscard]] uint64_t total_bytes() const noexcept;

    // Preflight: device_bytes() <= max_device_bytes (when nonzero) and
    // host_bytes() <= max_host_bytes (when nonzero). Returns a Status naming
    // required vs available on refusal. 0 limits mean "no explicit limit".
    [[nodiscard]] Status preflight() const;
    // Preflight an ADDITIONAL device requirement without recording it.
    [[nodiscard]] Status preflight_additional_device(uint64_t extra_bytes) const;

    [[nodiscard]] luisa::vector<MemoryRequirement> requirements() const;

private:
    struct Key {
        luisa::string component;
        MemoryCategory category;
        bool operator==(const Key &other) const noexcept {
            return category == other.category && component == other.component;
        }
    };
    struct KeyHash {
        [[nodiscard]] uint64_t operator()(const Key &key) const noexcept;
    };
    MemoryLimits _limits;
    luisa::unordered_map<Key, uint64_t, KeyHash> _requirements;
};

// KV-cache capacity from the actual checkpoint dimensions (plan §7.2):
//   KV = full_attention_layers * requests * tokens * kv_heads * head_dim * 2 * dtype_bytes
// Checked; `dtype_bytes` is the KV storage element size (4 for the FP32 baseline).
[[nodiscard]] Result<uint64_t> plan_kv_cache_bytes(uint64_t full_attention_layers, uint64_t requests,
                                                   uint64_t tokens, uint64_t kv_heads, uint64_t head_dim,
                                                   uint64_t dtype_bytes);

}// namespace luisa::ocr