#include "ut/ut.hpp"

#include "cuda_interop_texture_plan.h"

using namespace boost::ut;
using namespace boost::ut::literals;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "vk_cuda_interop_texture_plan_preserves_dimension_extent_and_flags"_test = [] {
        constexpr auto image = lc::vk::detail::plan_cuda_interop_texture(
            PixelFormat::RGBA8UNorm,
            2u, 64u, 32u, 1u, 4u,
            true, true);
        expect(image.valid());
        expect(image.image_type == VK_IMAGE_TYPE_2D);
        expect(image.extent.width == 64u);
        expect(image.extent.height == 32u);
        expect(image.extent.depth == 1u);
        expect(image.mip_levels == 4u);
        expect(image.dimension == 2u);
        expect(image.simultaneous_access);
        expect((image.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0u);
        expect((image.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0u);
        expect((image.usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0u);
        expect((image.usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0u);
        expect((image.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0u);

        constexpr auto volume = lc::vk::detail::plan_cuda_interop_texture(
            PixelFormat::RGBA8UNorm,
            3u, 16u, 8u, 4u, 3u,
            false, false);
        expect(volume.valid());
        expect(volume.image_type == VK_IMAGE_TYPE_3D);
        expect(volume.extent.width == 16u);
        expect(volume.extent.height == 8u);
        expect(volume.extent.depth == 4u);
        expect(volume.mip_levels == 3u);
        expect(volume.dimension == 3u);
        expect(!volume.simultaneous_access);
        expect((volume.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) == 0u);

        constexpr auto linear = lc::vk::detail::plan_cuda_interop_texture(
            PixelFormat::R32F,
            1u, 8u, 1u, 1u, 1u,
            false, false);
        expect(linear.valid());
        expect(linear.image_type == VK_IMAGE_TYPE_1D);
        expect(linear.extent.width == 8u);
        expect(linear.extent.height == 1u);
        expect(linear.extent.depth == 1u);
    };

    "vk_cuda_interop_texture_plan_normalizes_public_mip_convention"_test = [] {
        constexpr auto full_chain = lc::vk::detail::plan_cuda_interop_texture(
            PixelFormat::RGBA8UNorm,
            3u, 8u, 4u, 2u, 0u,
            false, false);
        expect(full_chain.valid());
        expect(full_chain.mip_levels == 4u)
            << "zero requested levels means the full mip chain";

        constexpr auto clamped = lc::vk::detail::plan_cuda_interop_texture(
            PixelFormat::RGBA8UNorm,
            2u, 8u, 4u, 1u, 99u,
            false, false);
        expect(clamped.valid());
        expect(clamped.mip_levels == 4u)
            << "the Vulkan allocation and runtime wrapper must expose the same clamped chain";
    };

    "vk_cuda_interop_texture_plan_excludes_non_storage_formats"_test = [] {
        constexpr auto srgb = lc::vk::detail::plan_cuda_interop_texture(
            PixelFormat::RGBA8SRGB,
            2u, 4u, 4u, 1u, 1u,
            false, false);
        expect(srgb.valid());
        expect((srgb.usage & VK_IMAGE_USAGE_STORAGE_BIT) == 0u);

        for (auto value = luisa::to_underlying(PixelFormat::BC1UNorm);
             value <= luisa::to_underlying(PixelFormat::BC7SRGB);
             value++) {
            auto format = static_cast<PixelFormat>(value);
            expect(is_block_compressed(format));
            auto compressed = lc::vk::detail::plan_cuda_interop_texture(
                format,
                2u, 16u, 16u, 1u, 1u,
                false, false);
            expect(compressed.valid());
            expect((compressed.usage & VK_IMAGE_USAGE_STORAGE_BIT) == 0u)
                << "block-compressed interop images cannot advertise STORAGE usage";
            expect((compressed.usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0u);
        }
    };

    "vk_cuda_interop_texture_plan_rejects_invalid_shapes"_test = [] {
        using Status = lc::vk::detail::CudaInteropTexturePlanStatus;
        constexpr auto dimension = lc::vk::detail::plan_cuda_interop_texture(
            PixelFormat::RGBA8UNorm,
            4u, 1u, 1u, 1u, 1u,
            false, false);
        expect(!dimension.valid());
        expect(dimension.status == Status::INVALID_DIMENSION);

        constexpr auto incompatible =
            lc::vk::detail::plan_cuda_interop_texture(
                PixelFormat::RGBA8UNorm,
                2u, 4u, 4u, 2u, 1u,
                false, false);
        expect(!incompatible.valid());
        expect(incompatible.status == Status::INCOMPATIBLE_EXTENT);

        constexpr auto extent = lc::vk::detail::plan_cuda_interop_texture(
            PixelFormat::RGBA8UNorm,
            3u, 4u, 4u, 0u, 1u,
            false, false);
        expect(!extent.valid());
        expect(extent.status == Status::ZERO_EXTENT);
    };
}
