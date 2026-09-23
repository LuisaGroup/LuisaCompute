#pragma once

#include <cuda.h>

#include <luisa/core/dynamic_module.h>
#include <luisa/core/stl/lru_cache.h>
#include <luisa/core/string_scratch.h>
#include <luisa/ast/function.h>
#include <luisa/runtime/context.h>

#include "cuda_shader_metadata.h"

namespace luisa::compute::cuda {

class CUDADevice;

/**
 * @brief Kernel compiler of CUDA
 * 
 */
class CUDACompiler {

public:
    using Cache = LRUCache<uint64_t /* hash */,
                           luisa::vector<std::byte> /* compiled ptx */>;
    static constexpr auto max_cache_item_count = 64u;

private:
    const CUDADevice *_device;
    mutable luisa::unique_ptr<Cache> _cache;
    luisa::string _nvrtc_path;
    uint32_t _nvrtc_version;
    luisa::string _device_library;
    // The device-side software ray-tracing traversal (cuda_builtin/
    // cuda_device_fallback_rtx.h).  It is kept out of `_device_library` on
    // purpose: a shader only gets it when the fallback is in use, so the source
    // of every hardware-path shader - and therefore its hash - is unchanged.
    luisa::string _fallback_rtx_device_library;

public:
    explicit CUDACompiler(const CUDADevice *device) noexcept;
    CUDACompiler(CUDACompiler &&) noexcept = default;
    CUDACompiler(const CUDACompiler &) noexcept = delete;
    CUDACompiler &operator=(CUDACompiler &&) noexcept = default;
    CUDACompiler &operator=(const CUDACompiler &) noexcept = delete;
    [[nodiscard]] auto nvrtc_version() const noexcept { return _nvrtc_version; }
    [[nodiscard]] auto device_library() const noexcept { return luisa::string_view{_device_library}; }
    [[nodiscard]] auto fallback_rtx_device_library() const noexcept {
        return luisa::string_view{_fallback_rtx_device_library};
    }
    [[nodiscard]] luisa::vector<std::byte> compile(const luisa::string &src, const luisa::string &src_filename,
                                                   luisa::span<const char *const> options,
                                                   const CUDAShaderMetadata *metadata = nullptr) const noexcept;
    [[nodiscard]] static uint64_t compute_hash(const luisa::string &src, luisa::span<const char *const> options) noexcept;
    [[nodiscard]] static size_t type_size(const Type *type) noexcept;
    [[nodiscard]] auto device() const noexcept { return _device; }
    static void process_builtin(luisa::string &result, char const *data, size_t size) noexcept;
};

}// namespace luisa::compute::cuda
