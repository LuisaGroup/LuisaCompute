#pragma once

#include <algorithm>
#include <cstdint>

#include <vulkan/vulkan_core.h>

#include <luisa/runtime/rhi/pixel.h>

namespace lc::vk::detail {

enum class CudaInteropTexturePlanStatus : uint8_t {
    SUCCESS,
    INVALID_DIMENSION,
    INCOMPATIBLE_EXTENT,
    ZERO_EXTENT
};

[[nodiscard]] constexpr const char *
cuda_interop_texture_plan_status_name(
    CudaInteropTexturePlanStatus status) noexcept {
    switch (status) {
        case CudaInteropTexturePlanStatus::SUCCESS: return "success";
        case CudaInteropTexturePlanStatus::INVALID_DIMENSION: return "invalid dimension";
        case CudaInteropTexturePlanStatus::INCOMPATIBLE_EXTENT: return "extent is incompatible with image dimension";
        case CudaInteropTexturePlanStatus::ZERO_EXTENT: return "zero extent";
    }
    return "unknown";
}

struct CudaInteropTexturePlan {
    CudaInteropTexturePlanStatus status;
    VkImageType image_type;
    VkExtent3D extent;
    uint32_t mip_levels;
    VkImageUsageFlags usage;
    uint32_t dimension;
    bool simultaneous_access;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return status == CudaInteropTexturePlanStatus::SUCCESS;
    }
};

[[nodiscard]] constexpr uint32_t
cuda_interop_mip_level_count(
    VkExtent3D extent, uint32_t requested_levels) noexcept {
    auto max_extent = std::max({extent.width, extent.height, extent.depth});
    auto max_levels = 0u;
    while (max_extent != 0u) {
        max_extent >>= 1u;
        max_levels++;
    }
    return requested_levels == 0u ?
               max_levels :
               std::min(requested_levels, max_levels);
}

[[nodiscard]] constexpr CudaInteropTexturePlan
plan_cuda_interop_texture(
    luisa::compute::PixelFormat format,
    uint32_t dimension,
    uint32_t width, uint32_t height, uint32_t depth,
    uint32_t requested_mip_levels,
    bool simultaneous_access,
    bool allow_raster_target) noexcept {
    auto image_type = VK_IMAGE_TYPE_MAX_ENUM;
    auto extent = VkExtent3D{width, height, depth};
    switch (dimension) {
        case 1u:
            image_type = VK_IMAGE_TYPE_1D;
            break;
        case 2u:
            image_type = VK_IMAGE_TYPE_2D;
            break;
        case 3u:
            image_type = VK_IMAGE_TYPE_3D;
            break;
        default:
            return CudaInteropTexturePlan{
                .status = CudaInteropTexturePlanStatus::INVALID_DIMENSION,
                .image_type = image_type,
                .extent = extent,
                .mip_levels = 0u,
                .usage = 0u,
                .dimension = dimension,
                .simultaneous_access = simultaneous_access};
    }
    if ((dimension == 1u &&
         (extent.height != 1u || extent.depth != 1u)) ||
        (dimension == 2u && extent.depth != 1u)) {
        return CudaInteropTexturePlan{
            .status = CudaInteropTexturePlanStatus::INCOMPATIBLE_EXTENT,
            .image_type = image_type,
            .extent = extent,
            .mip_levels = 0u,
            .usage = 0u,
            .dimension = dimension,
            .simultaneous_access = simultaneous_access};
    }
    if (extent.width == 0u || extent.height == 0u ||
        extent.depth == 0u) {
        return CudaInteropTexturePlan{
            .status = CudaInteropTexturePlanStatus::ZERO_EXTENT,
            .image_type = image_type,
            .extent = extent,
            .mip_levels = 0u,
            .usage = 0u,
            .dimension = dimension,
            .simultaneous_access = simultaneous_access};
    }
    auto usage = static_cast<VkImageUsageFlags>(
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT |
        (allow_raster_target ?
             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT :
             0u) |
        ((luisa::compute::is_srgb(format) ||
          luisa::compute::is_block_compressed(format)) ?
             0u :
             VK_IMAGE_USAGE_STORAGE_BIT));
    return CudaInteropTexturePlan{
        .status = CudaInteropTexturePlanStatus::SUCCESS,
        .image_type = image_type,
        .extent = extent,
        .mip_levels = cuda_interop_mip_level_count(
            extent, requested_mip_levels),
        .usage = usage,
        .dimension = dimension,
        .simultaneous_access = simultaneous_access};
}

}// namespace lc::vk::detail
