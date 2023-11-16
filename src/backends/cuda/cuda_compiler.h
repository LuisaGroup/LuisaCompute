#pragma once

#include <cuda.h>

#include <luisa/core/dynamic_module.h>
#include <luisa/core/stl/lru_cache.h>
#include <luisa/ast/function.h>
#include <luisa/runtime/context.h>

#include "cuda_shader_metadata.h"
#include "cuda_nvrtc.h"

namespace luisa::compute::cuda {

class CUDADevice;

/**
 * @brief Kernel compiler of CUDA
 * 
 */
class CUDACompiler {

public:
    using Cache = LRUCache<uint64_t /* hash */,
                           luisa::string /* compiled ptx */>;
    static constexpr auto max_cache_item_count = 64u;

private:
    using nvrtc_version_func = int();
    using nvrtc_compile_func = LUISA_NVRTC_StringBuffer(
        const char *filename, const char *src,
        const char *const *options, size_t num_options);
    using nvrtc_free_func = void(LUISA_NVRTC_StringBuffer buffer);

private:
    const CUDADevice *_device;
    luisa::string _device_library;
    mutable luisa::unique_ptr<Cache> _cache;
#ifdef LUISA_COMPUTE_STANDALONE_NVRTC_DLL
    DynamicModule _nvrtc_module;
#endif
    nvrtc_version_func *_version_func{};
    nvrtc_compile_func *_compile_func{};
    nvrtc_free_func *_free_func{};
    uint _nvrtc_version{};

public:
    explicit CUDACompiler(const CUDADevice *device) noexcept;
    CUDACompiler(CUDACompiler &&) noexcept = default;
    CUDACompiler(const CUDACompiler &) noexcept = delete;
    CUDACompiler &operator=(CUDACompiler &&) noexcept = default;
    CUDACompiler &operator=(const CUDACompiler &) noexcept = delete;
    [[nodiscard]] auto nvrtc_version() const noexcept { return _nvrtc_version; }
    [[nodiscard]] auto device_library() const noexcept { return luisa::string_view{_device_library}; }
    [[nodiscard]] luisa::string compile(const luisa::string &src, const luisa::string &src_filename,
                                        luisa::span<const char *const> options,
                                        const CUDAShaderMetadata *metadata = nullptr) const noexcept;
    [[nodiscard]] uint64_t compute_hash(const luisa::string &src,
                                        luisa::span<const char *const> options) const noexcept;
    [[nodiscard]] static size_t type_size(const Type *type) noexcept;
    [[nodiscard]] auto device() const noexcept { return _device; }
};

}// namespace luisa::compute::cuda
