#pragma once

#include <luisa/runtime/swapchain.h>

namespace luisa::compute::simd {

class SIMDStream;
class SIMDTexture;

class SIMDSwapchain {

private:
    SIMDStream *_bound_stream;
    void *_handle{};
    void *_native_handle{};
    uint2 _size;
    PixelStorage _storage{PixelStorage::BYTE4};

public:
    SIMDSwapchain(SIMDStream *bound_stream, const SwapchainOption &option) noexcept;
    ~SIMDSwapchain() noexcept;
    SIMDSwapchain(const SIMDSwapchain &) = delete;
    SIMDSwapchain &operator=(const SIMDSwapchain &) = delete;

    void present(SIMDStream *stream, const SIMDTexture *frame) noexcept;
    [[nodiscard]] auto native_handle() const noexcept { return _native_handle; }
    [[nodiscard]] auto storage() const noexcept { return _storage; }
};

}// namespace luisa::compute::simd
