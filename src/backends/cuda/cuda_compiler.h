//
// Created by Mike on 2021/11/8.
//

#include <nvrtc.h>
#include <cuda.h>

#include <core/allocator.h>
#include <ast/function.h>
#include <runtime/context.h>

namespace luisa::compute::cuda {

class CUDACompiler {

public:
    static constexpr auto max_cache_item_count = 128u;

private:
    std::mutex _mutex;
    uint64_t _current_timepoint{0u};
    luisa::unordered_map<uint64_t, uint64_t> _function_hash_to_timepoint;
    luisa::map<uint64_t, std::pair<luisa::string, uint64_t>> _timepoint_to_ptx_and_hash;

private:
    CUDACompiler() noexcept = default;

public:
    CUDACompiler(CUDACompiler &&) noexcept = delete;
    CUDACompiler(const CUDACompiler &) noexcept = delete;
    CUDACompiler &operator=(CUDACompiler &&) noexcept = delete;
    CUDACompiler &operator=(const CUDACompiler &) noexcept = delete;
    [[nodiscard]] static CUDACompiler &instance() noexcept;
    [[nodiscard]] luisa::string compile(const Context &ctx, Function function, uint32_t sm) noexcept;
};

}// namespace luisa::compute::cuda
