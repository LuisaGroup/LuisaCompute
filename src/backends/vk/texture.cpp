#include "texture.h"
#include "device.h"
#include "device_feature_plan.h"
#include "sparse_binding_plan.h"
#include "log.h"
#include <luisa/core/stl/vector.h>
namespace lc::vk {
using namespace luisa::compute;
namespace {
// Validate that the physical device supports the requested image format for
// the requested image usages. On failure, fills `reason` with the missing
// format-feature bits so callers can emit a diagnosable LUISA_ERROR.
[[nodiscard]] bool is_vk_format_supported(
    VkPhysicalDevice physical_device, VkFormat format,
    VkImageUsageFlags usage, luisa::string *reason = nullptr) noexcept {
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(physical_device, format, &props);
    auto features = props.optimalTilingFeatures;
    auto required_features = VkFormatFeatureFlags{0u};
    if (usage & VK_IMAGE_USAGE_SAMPLED_BIT) {
        required_features |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    }
    if (usage & VK_IMAGE_USAGE_STORAGE_BIT) {
        required_features |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
    }
    if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
        required_features |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
    }
    if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) {
        required_features |= VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
    }
    if (usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) {
        required_features |= VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
    }
    if (usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) {
        required_features |= VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    }
    auto missing = required_features & ~features;
    if (missing != 0u && reason != nullptr) {
        luisa::vector<const char *> names;
        if (missing & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) { names.emplace_back("SAMPLED_IMAGE"); }
        if (missing & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) { names.emplace_back("STORAGE_IMAGE"); }
        if (missing & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) { names.emplace_back("COLOR_ATTACHMENT"); }
        if (missing & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) { names.emplace_back("DEPTH_STENCIL_ATTACHMENT"); }
        if (missing & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT) { names.emplace_back("TRANSFER_SRC"); }
        if (missing & VK_FORMAT_FEATURE_TRANSFER_DST_BIT) { names.emplace_back("TRANSFER_DST"); }
        if (missing & VK_FORMAT_FEATURE_BLIT_SRC_BIT) { names.emplace_back("BLIT_SRC"); }
        if (missing & VK_FORMAT_FEATURE_BLIT_DST_BIT) { names.emplace_back("BLIT_DST"); }
        luisa::string joined;
        for (auto i = 0u; i < names.size(); i++) {
            if (i != 0u) { joined.append(", "); }
            joined.append(names[i]);
        }
        *reason = luisa::format(
            "format {} is not supported for the requested usage {:#x}; "
            "missing VkFormatFeatureFlagBits: {} (available {:#x})",
            static_cast<int>(format), static_cast<uint32_t>(usage), joined,
            static_cast<uint32_t>(features));
    }
    return missing == 0u;
}
}// namespace

NativeImageState::NativeImageState(
    VkImage image, VkFormat format, uint dimension, uint3 size,
    uint mip_levels, bool simultaneous_access,
    luisa::shared_ptr<std::atomic_size_t> expiration_counter)
    : image{image},
      format{format},
      size{size},
      mip_levels{mip_levels},
      dimension{dimension},
      simultaneous_access{simultaneous_access},
      _expiration_counter{std::move(expiration_counter)} {
    LUISA_ASSERT(image != VK_NULL_HANDLE,
                 "Cannot track a null Vulkan image.");
    LUISA_ASSERT(mip_levels > 0u,
                 "A Vulkan image must have at least one mip level.");
    _layouts.resize(mip_levels, VK_IMAGE_LAYOUT_UNDEFINED);
}

NativeImageState::~NativeImageState() noexcept {
    // The registry owns only weak references. Keep a lifetime-independent
    // counter so the next acquisition can promptly reclaim expired weak
    // control blocks without calling back through a possibly destroyed Device.
    _expiration_counter->fetch_add(1u, std::memory_order_relaxed);
}

VkImageLayout NativeImageState::layout(uint level) const {
    std::lock_guard lock{_layout_mtx};
    LUISA_ASSERT(level < _layouts.size(),
                 "Vulkan image mip {} is outside [0, {}).",
                 level, _layouts.size());
    return _layouts[level];
}

void NativeImageState::set_layout(
    uint level, VkImageLayout layout) const {
    std::lock_guard lock{_layout_mtx};
    LUISA_ASSERT(level < _layouts.size(),
                 "Vulkan image mip {} is outside [0, {}).",
                 level, _layouts.size());
    _layouts[level] = layout;
}

Texture::Texture(Device *device)
    : Resource(device),
      _vk_img(nullptr),
      _format(static_cast<compute::PixelFormat>(0)),
      _mip(0),
      _dimension(0) {
    _allocation = nullptr;
}

void Texture::_acquire_native_state(VkFormat format) {
    _native_state = device()->acquire_native_image_state(
        _vk_img, format, _dimension, _size, _mip,
        _simultaneous_access);
}

Texture::Texture(
    Device *device,
    VkImage external_image,
    uint dimension,
    compute::PixelFormat format,
    uint3 size,
    uint mip,
    bool simultaneous_access,
    VkDeviceMemory external_memory)
    : Resource(device),
      _vk_img(external_image),
      _format(format),
      _size(size),
      _mip(mip),
      _dimension(dimension),
      _contained{false},
      _simultaneous_access(simultaneous_access) {
    _allocation = nullptr;
    _acquire_native_state(to_vk_format(format));
    if (external_memory) {
        _allocated_memory = external_memory;
        _external_allocation = true;
    }
}

Texture::Texture(
    Device *device,
    VkImage external_image,
    uint dimension,
    VkFormat format,
    uint3 size,
    uint mip,
    bool simultaneous_access,
    VkDeviceMemory external_memory)
    : Resource(device),
      _vk_img(external_image),
      _format(
          static_cast<compute::PixelFormat>(static_cast<uint>(format) | (1u << 31u))),
      _size(size),
      _mip(mip),
      _dimension(dimension),
      _contained{false},
      _simultaneous_access(simultaneous_access) {
    _allocation = nullptr;
    _acquire_native_state(format);
    if (external_memory) {
        _allocated_memory = external_memory;
        _external_allocation = true;
    }
}

Texture::Texture(
    Device *device,
    uint dimension,
    PixelFormat format,
    uint3 size,
    uint mip,
    bool simultaneous_access,
    bool allow_raster_target)
    : Resource(device),
      _format(format),
      _size(size),
      _mip(mip),
      _dimension(dimension),
      _simultaneous_access(simultaneous_access) {
    auto vk_format = to_vk_format(format);
    if (vk_format == VK_FORMAT_UNDEFINED) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Unsupported pixel format {} cannot be mapped to a Vulkan format.",
            static_cast<uint32_t>(format));
    }
    auto image_usage =
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT |
        (allow_raster_target ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0) |
        ((is_srgb(format) || is_block_compressed(format)) ? 0 : VK_IMAGE_USAGE_STORAGE_BIT);
    luisa::string unsupported_reason;
    if (!is_vk_format_supported(
            device->physical_device(), vk_format, image_usage,
            &unsupported_reason)) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("{}", unsupported_reason);
    }
    auto allocation = device->allocator().allocate_image(
        [&]() {
            switch (dimension) {
                case 1:
                    return VK_IMAGE_TYPE_1D;
                case 2:
                    return VK_IMAGE_TYPE_2D;
                case 3:
                    return VK_IMAGE_TYPE_3D;
                default:
                    break;
            }
            LUISA_ERROR_WITH_LOCATION("Invalid texture dimension.");
        }(),
        vk_format,
        size,
        mip,
        image_usage);
    _vk_img = allocation.image;
    _allocation = allocation.allocation;
    _acquire_native_state(vk_format);
}

VkImageAspectFlags Texture::get_aspect_from_format(VkFormat format) {
    switch (format) {
        case VK_FORMAT_D32_SFLOAT:
        case VK_FORMAT_X8_D24_UNORM_PACK32:
        case VK_FORMAT_D16_UNORM:
            return VK_IMAGE_ASPECT_DEPTH_BIT;
        case VK_FORMAT_S8_UINT:
            return VK_IMAGE_ASPECT_STENCIL_BIT;
        case VK_FORMAT_D16_UNORM_S8_UINT:
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return VK_IMAGE_ASPECT_STENCIL_BIT | VK_IMAGE_ASPECT_DEPTH_BIT;
        default:
            return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}
Texture::Texture(
    Device *device,
    compute::DepthFormat format,
    uint2 size)
    : Resource(device),
      _format(static_cast<compute::PixelFormat>(static_cast<uint>(format) | (1u << 16u))),
      _size(make_uint3(size, 1)),
      _mip(1),
      _dimension(2),
      _simultaneous_access(false) {

    auto allocation = device->allocator().allocate_image(
        VK_IMAGE_TYPE_2D,
        to_vk_format(static_cast<compute::PixelFormat>(static_cast<uint>(format) | (1u << 16u))),
        make_uint3(size, 1),
        1,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    _vk_img = allocation.image;
    _allocation = allocation.allocation;
    _acquire_native_state(to_vk_format(_format));
}
Texture::~Texture() {
    // Drop the registry-visible state before destroying an owned VkImage. A
    // concurrently created image may reuse the same handle immediately after
    // vkDestroyImage; it must not inherit this image's stale metadata/layouts.
    std::lock_guard native_state_lock{device()->_native_image_state_mtx};
    _native_state.reset();
    if (_external_allocation) {
        vkDestroyImage(device()->logic_device(), _vk_img, Device::alloc_callbacks());
        vkFreeMemory(device()->logic_device(), _allocated_memory, Device::alloc_callbacks());
    } else if (_allocation)
        device()->allocator().destroy_image({_vk_img, _allocation});
    else if (_contained)
        vkDestroyImage(device()->logic_device(), _vk_img, Device::alloc_callbacks());
}

void Texture::init_as_sparse(
    uint dimension,
    compute::PixelFormat format,
    uint3 size,
    uint mip,
    bool simultaneous_access) {
    auto enabled = device()->enabled_features();
    auto sparse_features = detail::validate_sparse_texture_features(
        {.sparse_binding = enabled.sparseBinding == VK_TRUE,
         .sparse_residency_image_2d =
             enabled.sparseResidencyImage2D == VK_TRUE,
         .sparse_residency_image_3d =
             enabled.sparseResidencyImage3D == VK_TRUE},
        dimension);
    LUISA_ASSERT(
        static_cast<bool>(sparse_features),
        "Vulkan sparse-texture creation is unavailable for dimension {}: {}.",
        dimension,
        detail::sparse_residency_feature_status_name(
            sparse_features.status));
    auto img_type = [&]() {
        switch (dimension) {
            case 1:
                return VK_IMAGE_TYPE_1D;
            case 2:
                return VK_IMAGE_TYPE_2D;
            case 3:
                return VK_IMAGE_TYPE_3D;
            default:
                break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid texture dimension.");
    }();
    VkImageCreateInfo img_create_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
        .imageType = img_type,
        .format = to_vk_format(format),
        .extent = VkExtent3D{size.x, size.y, size.z},
        .mipLevels = mip,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    device()->allocator().apply_queue_sharing(img_create_info);
    if (!(is_srgb(format) || is_block_compressed(format))) {
        img_create_info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    uint32_t sparse_format_property_count{};
    vkGetPhysicalDeviceSparseImageFormatProperties(
        device()->physical_device(), img_create_info.format,
        img_create_info.imageType, img_create_info.samples,
        img_create_info.usage, img_create_info.tiling,
        &sparse_format_property_count, nullptr);
    LUISA_ASSERT(
        sparse_format_property_count != 0u,
        "Vulkan physical device does not support sparse residency for format "
        "{}, dimension {}, and usage flags 0x{:x}.",
        static_cast<uint32_t>(img_create_info.format), dimension,
        img_create_info.usage);
    VK_CHECK_RESULT(vkCreateImage(device()->logic_device(), &img_create_info, Device::alloc_callbacks(), &_vk_img));
    vkGetImageMemoryRequirements(
        device()->logic_device(), _vk_img, &_memory_requirements);
    LUISA_ASSERT(
        _memory_requirements.alignment != 0u,
        "Vulkan reported zero sparse-image block size.");

    uint32_t sparse_requirement_count{};
    vkGetImageSparseMemoryRequirements(
        device()->logic_device(), _vk_img,
        &sparse_requirement_count, nullptr);
    LUISA_ASSERT(
        sparse_requirement_count != 0u,
        "Vulkan created a sparse-resident image without sparse memory requirements.");
    luisa::vector<VkSparseImageMemoryRequirements> sparse_requirements;
    sparse_requirements.resize(sparse_requirement_count);
    vkGetImageSparseMemoryRequirements(
        device()->logic_device(), _vk_img,
        &sparse_requirement_count, sparse_requirements.data());
    sparse_requirements.resize(sparse_requirement_count);
    auto selection = detail::select_sparse_image_requirements(
        std::span<const VkSparseImageMemoryRequirements>{
            sparse_requirements.data(), sparse_requirements.size()});
    LUISA_ASSERT(
        static_cast<bool>(selection),
        "Vulkan sparse-image memory requirements are not representable by "
        "the Luisa tile API (status {}). Opaque metadata bindings and "
        "ambiguous color-aspect layouts are unsupported.",
        static_cast<uint32_t>(selection.status));
    _sparse_memory_requirements =
        sparse_requirements[selection.color_requirement_index];
    auto mip_tail = detail::validate_sparse_image_mip_tail(
        mip, _sparse_memory_requirements.imageMipTailFirstLod);
    LUISA_ASSERT(
        static_cast<bool>(mip_tail),
        "Vulkan sparse image requests {} mip levels, but opaque mip-tail "
        "binding begins at level {}. The Luisa sparse tile API cannot "
        "represent mip tails; every requested level must be below {}.",
        mip, _sparse_memory_requirements.imageMipTailFirstLod,
        _sparse_memory_requirements.imageMipTailFirstLod);
    auto granularity =
        _sparse_memory_requirements.formatProperties.imageGranularity;
    LUISA_ASSERT(
        granularity.width != 0u && granularity.height != 0u &&
            granularity.depth != 0u,
        "Vulkan reported a zero sparse-image granularity ({}, {}, {}).",
        granularity.width, granularity.height, granularity.depth);
    _format = format;
    _size = size;
    _mip = mip;
    _dimension = dimension;
    _simultaneous_access = simultaneous_access;
    _acquire_native_state(to_vk_format(format));
}

VkFormat Texture::to_vk_format(PixelFormat format) {
    // native format
    if ((luisa::to_underlying(format) & (1u << 31u)) != 0) {
        return static_cast<VkFormat>(luisa::to_underlying(format) & ((1u << 31u) - 1u));
    }
    // depth
    else if (luisa::to_underlying(format) > 65535u) {
        auto depth_format = static_cast<compute::DepthFormat>(luisa::to_underlying(format) & 65535u);
        switch (depth_format) {
            case compute::DepthFormat::D16:
                return VK_FORMAT_D16_UNORM;
            case compute::DepthFormat::D24S8:
                return VK_FORMAT_D24_UNORM_S8_UINT;
            case compute::DepthFormat::D32:
                return VK_FORMAT_D32_SFLOAT;
            case compute::DepthFormat::D32S8A24:
                return VK_FORMAT_D32_SFLOAT_S8_UINT;
            default:
                return VK_FORMAT_UNDEFINED;
        }
    }

    switch (format) {
        case PixelFormat::R8SInt:
            return VK_FORMAT_R8_SINT;
        case PixelFormat::R8UInt:
            return VK_FORMAT_R8_UINT;
        case PixelFormat::R8UNorm:
            return VK_FORMAT_R8_UNORM;
        case PixelFormat::RG8SInt:
            return VK_FORMAT_R8G8_SINT;
        case PixelFormat::RG8UInt:
            return VK_FORMAT_R8G8_UINT;
        case PixelFormat::RG8UNorm:
            return VK_FORMAT_R8G8_UNORM;
        case PixelFormat::RGBA8SInt:
            return VK_FORMAT_R8G8B8A8_SINT;
        case PixelFormat::RGBA8UInt:
            return VK_FORMAT_R8G8B8A8_UINT;
        case PixelFormat::RGBA8SRGB:
            return VK_FORMAT_R8G8B8A8_SRGB;
        case PixelFormat::RGBA8UNorm:
            return VK_FORMAT_R8G8B8A8_UNORM;

        case PixelFormat::R16SInt:
            return VK_FORMAT_R16_SINT;
        case PixelFormat::R16UInt:
            return VK_FORMAT_R16_UINT;
        case PixelFormat::R16UNorm:
            return VK_FORMAT_R16_UNORM;
        case PixelFormat::RG16SInt:
            return VK_FORMAT_R16G16_SINT;
        case PixelFormat::RG16UInt:
            return VK_FORMAT_R16G16_UINT;
        case PixelFormat::RG16UNorm:
            return VK_FORMAT_R16G16_UNORM;
        case PixelFormat::RGBA16SInt:
            return VK_FORMAT_R16G16B16A16_SINT;
        case PixelFormat::RGBA16UInt:
            return VK_FORMAT_R16G16B16A16_UINT;
        case PixelFormat::RGBA16UNorm:
            return VK_FORMAT_R16G16B16A16_UNORM;

        case PixelFormat::R32SInt:
            return VK_FORMAT_R32_SINT;
        case PixelFormat::R32UInt:
            return VK_FORMAT_R32_UINT;
        case PixelFormat::RG32SInt:
            return VK_FORMAT_R32G32_SINT;
        case PixelFormat::RG32UInt:
            return VK_FORMAT_R32G32_UINT;
        case PixelFormat::RGBA32SInt:
            return VK_FORMAT_R32G32B32A32_SINT;
        case PixelFormat::RGBA32UInt:
            return VK_FORMAT_R32G32B32A32_UINT;
        case PixelFormat::R16F:
            return VK_FORMAT_R16_SFLOAT;
        case PixelFormat::RG16F:
            return VK_FORMAT_R16G16_SFLOAT;
        case PixelFormat::RGBA16F:
            return VK_FORMAT_R16G16B16A16_SFLOAT;
        case PixelFormat::R32F:
            return VK_FORMAT_R32_SFLOAT;
        case PixelFormat::RG32F:
            return VK_FORMAT_R32G32_SFLOAT;
        case PixelFormat::RGBA32F:
            return VK_FORMAT_R32G32B32A32_SFLOAT;
        case PixelFormat::R10G10B10A2UInt:
            return VK_FORMAT_A2R10G10B10_UINT_PACK32;
        case PixelFormat::R10G10B10A2UNorm:
            return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        case PixelFormat::R11G11B10F:
            return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        case PixelFormat::BC1UNorm:
            return VK_FORMAT_BC1_RGB_UNORM_BLOCK;
        case PixelFormat::BC2UNorm:
            return VK_FORMAT_BC2_UNORM_BLOCK;
        case PixelFormat::BC3UNorm:
            return VK_FORMAT_BC3_UNORM_BLOCK;
        case PixelFormat::BC4UNorm:
            return VK_FORMAT_BC4_UNORM_BLOCK;
        case PixelFormat::BC5UNorm:
            return VK_FORMAT_BC5_UNORM_BLOCK;
        case PixelFormat::BC6HUF16:
            return VK_FORMAT_BC6H_UFLOAT_BLOCK;
        case PixelFormat::BC7SRGB:
            return VK_FORMAT_BC7_SRGB_BLOCK;
        case PixelFormat::BC7UNorm:
            return VK_FORMAT_BC7_UNORM_BLOCK;
        // ASTC block formats (mandatory on Android; BC formats are typically
        // unsupported there). Both UNORM and SRGB variants are mapped.
        case PixelFormat::ASTC_4x4:
            return VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
        case PixelFormat::ASTC_4x4_SRGB:
            return VK_FORMAT_ASTC_4x4_SRGB_BLOCK;
        case PixelFormat::ASTC_5x5:
            return VK_FORMAT_ASTC_5x5_UNORM_BLOCK;
        case PixelFormat::ASTC_5x5_SRGB:
            return VK_FORMAT_ASTC_5x5_SRGB_BLOCK;
        case PixelFormat::ASTC_6x6:
            return VK_FORMAT_ASTC_6x6_UNORM_BLOCK;
        case PixelFormat::ASTC_6x6_SRGB:
            return VK_FORMAT_ASTC_6x6_SRGB_BLOCK;
        case PixelFormat::ASTC_8x8:
            return VK_FORMAT_ASTC_8x8_UNORM_BLOCK;
        case PixelFormat::ASTC_8x8_SRGB:
            return VK_FORMAT_ASTC_8x8_SRGB_BLOCK;
        case PixelFormat::ASTC_10x10:
            return VK_FORMAT_ASTC_10x10_UNORM_BLOCK;
        case PixelFormat::ASTC_10x10_SRGB:
            return VK_FORMAT_ASTC_10x10_SRGB_BLOCK;
        case PixelFormat::ASTC_12x12:
            return VK_FORMAT_ASTC_12x12_UNORM_BLOCK;
        case PixelFormat::ASTC_12x12_SRGB:
            return VK_FORMAT_ASTC_12x12_SRGB_BLOCK;
        default:
            return VK_FORMAT_UNDEFINED;
    }
}
}// namespace lc::vk
