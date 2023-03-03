//
// Created by Mike on 2021/11/8.
//

#pragma once

#include <nvrtc.h>
#include <cuda.h>

#include <core/lru_cache.h>
#include <ast/function.h>
#include <runtime/context.h>

namespace luisa::compute::cuda {

/**
 * @brief Kernel compiler of CUDA
 * 
 */
class CUDACompiler {

public:
    using Cache = LRUCache<uint64_t, luisa::string>;
    static constexpr auto max_cache_item_count = 128u;

private:
    luisa::unique_ptr<Cache> _cache;

private:
    CUDACompiler() noexcept : _cache{Cache::create(max_cache_item_count)} {}

public:
    CUDACompiler(CUDACompiler &&) noexcept = delete;
    CUDACompiler(const CUDACompiler &) noexcept = delete;
    CUDACompiler &operator=(CUDACompiler &&) noexcept = delete;
    CUDACompiler &operator=(const CUDACompiler &) noexcept = delete;
    /**
     * @brief Return singleton
     * 
     * @return CUDACompiler& 
     */
    [[nodiscard]] static CUDACompiler &instance() noexcept;
    /**
     * @brief Compile kernel function
     * 
     * @param ctx context
     * @param function function
     * @param sm // TODO
     * @return compile result
     */
    [[nodiscard]] luisa::string compile(const Context &ctx, Function function, uint32_t sm) noexcept;

    [[nodiscard]] size_t type_size(const Type *type) const noexcept;
};

}// namespace luisa::compute::cuda
