//
// Created by Mike on 3/24/2023.
//

#pragma once

#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN

#include <core/basic_types.h>
#include <core/stl/memory.h>
#include <runtime/rhi/pixel.h>

namespace luisa::compute::cuda {

class CUDADevice;
class CUDAStream;
class CUDATexture;

class CUDASwapchain {

public:
    class Impl;

private:
    luisa::unique_ptr<Impl> _impl;

public:
    CUDASwapchain(CUDADevice *device, uint64_t window_handle,
                  uint width, uint height, bool allow_hdr,
                  bool vsync, uint back_buffer_size) noexcept;
    ~CUDASwapchain() noexcept;
    [[nodiscard]] PixelStorage pixel_storage() const noexcept;
    void present(CUDAStream *stream, CUDATexture *image) noexcept;
    void set_name(luisa::string &&name) noexcept;
};

}// namespace luisa::compute::cuda

#endif
