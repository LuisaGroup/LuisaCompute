#include <luisa/core/logging.h>

#include "simd_swapchain.h"
#include "simd_stream.h"
#include "simd_texture.h"

#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
LUISA_EXPORT_API void *luisa_compute_create_cpu_swapchain(
    uint64_t display_handle, uint64_t window_handle,
    uint32_t width, uint32_t height, bool allow_hdr, bool vsync,
    uint32_t back_buffer_count) noexcept;
LUISA_EXPORT_API uint8_t luisa_compute_cpu_swapchain_storage(void *swapchain) noexcept;
LUISA_EXPORT_API void *luisa_compute_cpu_swapchain_native_handle(void *swapchain) noexcept;
LUISA_EXPORT_API void luisa_compute_destroy_cpu_swapchain(void *swapchain) noexcept;
LUISA_EXPORT_API void luisa_compute_cpu_swapchain_present_with_callback(
    void *swapchain, void *context, void (*blit)(void *context, void *mapped_pixels)) noexcept;
#endif

namespace luisa::compute::simd {

SIMDSwapchain::SIMDSwapchain(SIMDStream *bound_stream, const SwapchainOption &option) noexcept
    : _bound_stream{bound_stream}, _size{option.size} {
#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
    LUISA_ASSERT(_bound_stream != nullptr && all(_size > 0u),
                 "SIMD swapchain requires a stream and a nonempty image size.");
    _handle = luisa_compute_create_cpu_swapchain(
        option.display, option.window, _size.x, _size.y,
        option.wants_hdr, option.wants_vsync, option.back_buffer_count);
    LUISA_ASSERT(_handle != nullptr, "Failed to create SIMD Vulkan swapchain.");
    _storage = static_cast<PixelStorage>(luisa_compute_cpu_swapchain_storage(_handle));
    _native_handle = luisa_compute_cpu_swapchain_native_handle(_handle);
    LUISA_ASSERT(_native_handle != nullptr &&
                     (_storage == PixelStorage::BYTE4 || _storage == PixelStorage::HALF4),
                 "SIMD Vulkan swapchain returned an invalid native handle or pixel storage.");
#else
    LUISA_ERROR_WITH_LOCATION("SIMD display requires a GUI build with Vulkan swapchain support.");
#endif
}

SIMDSwapchain::~SIMDSwapchain() noexcept {
#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
    // The bridge waits for its Vulkan work before releasing staging resources.
    luisa_compute_destroy_cpu_swapchain(_handle);
#endif
}

void SIMDSwapchain::present(SIMDStream *stream, const SIMDTexture *frame) noexcept {
#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
    LUISA_ASSERT(stream == _bound_stream, "SIMD swapchain stream mismatch.");
    LUISA_ASSERT(frame != nullptr && frame->dimension() == 2u,
                 "SIMD swapchain requires a two-dimensional image.");
    auto view = frame->view(0u);
    LUISA_ASSERT(all(view.size2d() == _size) && view.size3d().z == 1u,
                 "SIMD swapchain image size mismatch.");
    LUISA_ASSERT(view.storage() == _storage,
                 "SIMD swapchain image storage does not match the negotiated storage.");
    // SIMD dispatch and its worker pool complete synchronously. Keep this
    // boundary explicit before the bridge copies the rendered pixels.
    stream->synchronize();
    // The callback copies the mip's pixel data while staging memory is mapped;
    // it returns before present does, so neither the view nor the image escapes.
    luisa_compute_cpu_swapchain_present_with_callback(
        _handle, &view, [](void *context, void *mapped_pixels) noexcept {
            static_cast<const fallback::FallbackTextureView *>(context)->copy_to(mapped_pixels);
        });
#else
    LUISA_ERROR_WITH_LOCATION("SIMD display requires a GUI build with Vulkan swapchain support.");
#endif
}

}// namespace luisa::compute::simd
