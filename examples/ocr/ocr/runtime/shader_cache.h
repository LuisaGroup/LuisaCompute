#pragma once
// OCR engine -- per-owner typed shader cache (Phase 2).
//
// Compiles each device-free kernel factory ONCE per owner/variant and retains
// the shader for the owner's lifetime. Type safety comes from a concrete
// typed template (TypedShaderCache<KernelT>) keyed by an explicit variant key
// -- NO RTTI, no typeid, no type-erased fake caching. Each ops family declares
// a small concrete variant tag (e.g. GemmVariant{epilogue, weight_format}) and
// a stable string key; the registry (ShaderKeyRegistry) records every key ever
// requested so diagnostics can prove "compiled once per variant".
//
// compile_ok() is checked on every compile; a failed compile returns a Status
// and is NOT cached (a later retry may succeed). Shaders are named with the
// stable variant key for GPU debuggers.
#include <cstdint>
#include <utility>

#include <luisa/core/stl/functional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>

#include "ocr/status.h"

namespace luisa::ocr {

// Stable variant key: a family prefix + caller-supplied variant suffix. Keys
// are plain strings (no hashing of pointers / no RTTI), so they are stable
// across runs and usable as shader names.
[[nodiscard]] luisa::string make_shader_key(luisa::string_view family, luisa::string_view variant);

// Records every (key -> compile count) for an owner. Pure diagnostics; the
// typed caches are the source of truth for retention.
class ShaderKeyRegistry {
public:
    void record_compile(luisa::string_view key);
    [[nodiscard]] uint32_t compile_count(luisa::string_view key) const noexcept;
    [[nodiscard]] luisa::vector<luisa::string> keys() const;
    [[nodiscard]] uint64_t total_compiles() const noexcept { return _total; }

private:
    luisa::unordered_map<luisa::string, uint32_t> _counts;
    uint64_t _total{0};
};

// Typed cache for one kernel family. KernelT is a concrete compute::KernelN<...>
// alias. Factory is any callable `KernelT()` (device-free). The cache owns the
// compiled shaders and the device reference; it is move-only and must outlive
// every dispatch that uses its shaders.
//
// STABILITY: shaders are held via luisa::unique_ptr, NOT by value in the map.
// luisa::unordered_map may be ankerl::unordered_dense (xmake), which MOVES
// values on rehash; a by-value Shader would invalidate every previously
// returned pointer. The unique_ptr pointee address is stable across rehashes
// for the cache's lifetime, so callers may retain `const ShaderT *`.
template<typename KernelT>
class TypedShaderCache {
public:
    using ShaderT = decltype(std::declval<compute::Device &>().compile(std::declval<KernelT>()));
    using Factory = luisa::function<KernelT()>;

    TypedShaderCache() noexcept = default;
    TypedShaderCache(compute::Device &device, ShaderKeyRegistry &registry) noexcept
        : _device(&device), _registry(&registry) {}

    // Get (or compile once) the shader for `variant_key`. `factory` is only
    // invoked on a cache miss. Returns nullptr + Status on compile failure.
    // The returned pointer remains valid until the cache is destroyed,
    // including across further get_or_compile calls (rehashes).
    [[nodiscard]] Result<const ShaderT *> get_or_compile(luisa::string_view variant_key,
                                                         const Factory &factory) {
        using R = Result<const ShaderT *>;
        if (_device == nullptr) {
            return R::failure(StatusCode::invalid_argument, "shader cache has no device", "device");
        }
        const luisa::string key{variant_key};
        auto it = _shaders.find(key);
        if (it != _shaders.end()) { return R::success(it->second.get()); }
        KernelT kernel = factory();
        compute::ShaderOption option;
        option.name = key;
        auto shader = luisa::make_unique<ShaderT>(_device->compile(kernel, option));
        if (!*shader || !shader->compile_ok()) {
            return R::failure(StatusCode::internal_error,
                              luisa::string{"shader compile failed for variant '"} + key + "'", key);
        }
        shader->set_name(key);
        if (_registry != nullptr) { _registry->record_compile(key); }
        const ShaderT *stable = shader.get();
        _shaders.emplace(key, std::move(shader));
        return R::success(stable);
    }

    [[nodiscard]] bool contains(luisa::string_view variant_key) const noexcept {
        return _shaders.find(luisa::string{variant_key}) != _shaders.end();
    }
    [[nodiscard]] uint64_t size() const noexcept { return _shaders.size(); }

private:
    compute::Device *_device{nullptr};
    ShaderKeyRegistry *_registry{nullptr};
    luisa::unordered_map<luisa::string, luisa::unique_ptr<ShaderT>> _shaders;
};

}// namespace luisa::ocr