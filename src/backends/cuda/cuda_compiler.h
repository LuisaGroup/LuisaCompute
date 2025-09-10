#pragma once

#include <cuda.h>

#include <luisa/core/dynamic_module.h>
#include <luisa/core/stl/lru_cache.h>
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
    luisa::string _device_library;
    mutable luisa::unique_ptr<Cache> _cache;
    luisa::string _nvrtc_path;
    uint32_t _nvrtc_version;

public:
    explicit CUDACompiler(const CUDADevice *device) noexcept;
    CUDACompiler(CUDACompiler &&) noexcept = default;
    CUDACompiler(const CUDACompiler &) noexcept = delete;
    CUDACompiler &operator=(CUDACompiler &&) noexcept = default;
    CUDACompiler &operator=(const CUDACompiler &) noexcept = delete;
    [[nodiscard]] auto nvrtc_version() const noexcept { return _nvrtc_version; }
    [[nodiscard]] auto device_library() const noexcept { return luisa::string_view{_device_library}; }
    [[nodiscard]] luisa::vector<std::byte> compile(const luisa::string &src, const luisa::string &src_filename,
                                                   luisa::span<const char *const> options,
                                                   const CUDAShaderMetadata *metadata = nullptr) const noexcept;
    [[nodiscard]] uint64_t compute_hash(const luisa::string &src,
                                        luisa::span<const char *const> options) const noexcept;
    [[nodiscard]] static size_t type_size(const Type *type) noexcept;
    [[nodiscard]] auto device() const noexcept { return _device; }
};

}// namespace luisa::compute::cuda
