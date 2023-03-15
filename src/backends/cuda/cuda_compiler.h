//
// Created by Mike on 2021/11/8.
//

#pragma once

#include <nvrtc.h>
#include <cuda.h>

#include <core/stl/lru_cache.h>
#include <ast/function.h>
#include <runtime/context.h>

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
    const CUDADevice *_device;
    uint _nvrtc_version;
    uint64_t _library_hash;
    luisa::unique_ptr<Cache> _cache;

private:
    explicit CUDACompiler(const CUDADevice *device) noexcept;

public:
    CUDACompiler(CUDACompiler &&) noexcept = default;
    CUDACompiler(const CUDACompiler &) noexcept = delete;
    CUDACompiler &operator=(CUDACompiler &&) noexcept = default;
    CUDACompiler &operator=(const CUDACompiler &) noexcept = delete;
    [[nodiscard]] auto nvrtc_version() const noexcept { return _nvrtc_version; }
    [[nodiscard]] luisa::string compile(const luisa::string &src,
                                        const ShaderOption &option,
                                        luisa::span<const char *const> extra_options) noexcept;
    [[nodiscard]] static size_t type_size(const Type *type) noexcept;
};

}// namespace luisa::compute::cuda
