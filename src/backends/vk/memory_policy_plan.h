#pragma once
#include <cstdint>
#include <vulkan/vulkan_core.h>
#include <luisa/runtime/rhi/pixel.h>

namespace lc::vk::detail {

// Host-only planning for the optional device-memory policy extensions:
//
//   * VK_EXT_zero_initialize_device_memory
//   * VK_EXT_memory_decompression
//   * VK_EXT_image_tiling_control
//   * VK_EXT_texture_compression_astc_3d
//   * VK_KHR_extended_flags (capability-only, see the invariant at the bottom)
//
// Everything here is `constexpr` and free of device state so it can be unit
// tested without a Vulkan device. The backend calls these planners and then
// applies the result with the real Vulkan entry points. Nothing in this header
// makes correctness depend on an optional extension: every planner has an
// explicit "extension not enabled" answer that selects the pre-existing
// fallback path.

// ---------------------------------------------------------------------------
// VK_EXT_zero_initialize_device_memory
// ---------------------------------------------------------------------------

// Why a device-memory allocation does or does not request
// VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT.
enum class ZeroInitializeAllocationDecision : uint32_t {
    // The caller asked for the pre-existing, undefined-content allocation.
    kNotRequested = 0u,
    // The extension is unsupported on this physical device, switched off
    // through VulkanDeviceConfigExt, or forced off because the logical device
    // was imported. Pre-existing path.
    kDisabledByConfig,
    // The allocation performs a memory import operation;
    // VUID-VkMemoryAllocateFlagsInfo-flags-10760 forbids the flag.
    kMemoryImportForbidden,
    // The selected memory type index is outside
    // VkPhysicalDeviceMemoryProperties::memoryTypeCount.
    kInvalidMemoryType,
    // The memory type is VK_MEMORY_PROPERTY_PROTECTED_BIT;
    // VUID-VkMemoryAllocateFlagsInfo-flags-10761 forbids the flag.
    kProtectedMemoryType,
    // Lazily allocated memory has no backing store until it is attached to a
    // framebuffer, so the implementation cannot zero it "before the
    // application is able to access it". Never request the flag.
    kLazilyAllocatedMemoryType,
    // Use VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT.
    kDriverZeroInitialize,
};

[[nodiscard]] constexpr const char *zero_initialize_decision_name(
    ZeroInitializeAllocationDecision decision) noexcept {
    switch (decision) {
        case ZeroInitializeAllocationDecision::kNotRequested: return "not-requested";
        case ZeroInitializeAllocationDecision::kDisabledByConfig: return "extension-disabled";
        case ZeroInitializeAllocationDecision::kMemoryImportForbidden: return "memory-import";
        case ZeroInitializeAllocationDecision::kInvalidMemoryType: return "invalid-memory-type";
        case ZeroInitializeAllocationDecision::kProtectedMemoryType: return "protected-memory-type";
        case ZeroInitializeAllocationDecision::kLazilyAllocatedMemoryType: return "lazily-allocated-memory-type";
        case ZeroInitializeAllocationDecision::kDriverZeroInitialize: return "driver-zero-initialize";
    }
    return "unknown";
}

struct ZeroInitializeAllocationPlan {
    ZeroInitializeAllocationDecision decision{
        ZeroInitializeAllocationDecision::kNotRequested};
    // Whether VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT may be added to the
    // VkMemoryAllocateFlagsInfo of this allocation.
    [[nodiscard]] constexpr bool use_flag() const noexcept {
        return decision == ZeroInitializeAllocationDecision::kDriverZeroInitialize;
    }
    // True for every decision that falls back to the pre-existing allocation.
    [[nodiscard]] constexpr bool is_fallback() const noexcept {
        return !use_flag();
    }
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return use_flag();
    }
};

// Decides whether one device-memory allocation may request driver-side zero
// initialization.
//
// `extension_enabled` is the *negotiated* state
// (`Device::optional_extension_enabled(
//      OptionalExtension::ZERO_INITIALIZE_DEVICE_MEMORY)`), which already folds
// in the physical-device extension list, the feature bit, the
// VulkanDeviceConfigExt switch and the imported-device fail-closed rule.
// `memory_type_property_flags` is the VkMemoryType::propertyFlags of the memory
// type the allocation is going to use, and `memory_type_index` /
// `memory_type_count` come from the same VkPhysicalDeviceMemoryProperties.
//
// The checks run in a fixed order so the reported reason is deterministic:
// request, negotiation, import, memory type range, protected, lazily
// allocated.
[[nodiscard]] constexpr ZeroInitializeAllocationPlan
plan_zero_initialize_allocation(
    bool extension_enabled, bool zero_initialize_requested,
    bool memory_import, uint32_t memory_type_index,
    uint32_t memory_type_count,
    uint32_t memory_type_property_flags) noexcept {
    if (!zero_initialize_requested) {
        return {ZeroInitializeAllocationDecision::kNotRequested};
    }
    if (!extension_enabled) {
        return {ZeroInitializeAllocationDecision::kDisabledByConfig};
    }
    if (memory_import) {
        return {ZeroInitializeAllocationDecision::kMemoryImportForbidden};
    }
    if (memory_type_index >= memory_type_count) {
        return {ZeroInitializeAllocationDecision::kInvalidMemoryType};
    }
    if ((memory_type_property_flags & VK_MEMORY_PROPERTY_PROTECTED_BIT) != 0u) {
        return {ZeroInitializeAllocationDecision::kProtectedMemoryType};
    }
    if ((memory_type_property_flags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) != 0u) {
        return {ZeroInitializeAllocationDecision::kLazilyAllocatedMemoryType};
    }
    return {ZeroInitializeAllocationDecision::kDriverZeroInitialize};
}

// ---------------------------------------------------------------------------
// Memory type selection for dedicated (non-VMA) allocations
// ---------------------------------------------------------------------------

// One entry of VkPhysicalDeviceMemoryProperties, flattened so the planner stays
// free of Vulkan handles.
struct MemoryTypeCandidate {
    // VkMemoryType::propertyFlags.
    uint32_t property_flags{0u};
    // VkMemoryHeap::flags of the heap that backs this memory type.
    uint32_t heap_flags{0u};
};

enum class MemoryTypeSelectionStatus : uint32_t {
    kOk = 0u,
    // No memory type in `memory_type_bits` exists on the device.
    kNoMemoryTypeInBits,
    // Every candidate that satisfies the mandatory properties is excluded by
    // `forbidden_flags` (protected/lazily allocated memory).
    kAllCandidatesForbidden,
    // No candidate satisfies `required_flags`.
    kMissingRequiredProperties,
    // `memory_type_count` exceeds the 32 bits a VkMemoryRequirements mask can
    // address.
    kTooManyMemoryTypes,
};

[[nodiscard]] constexpr const char *memory_type_selection_status_name(
    MemoryTypeSelectionStatus status) noexcept {
    switch (status) {
        case MemoryTypeSelectionStatus::kOk: return "ok";
        case MemoryTypeSelectionStatus::kNoMemoryTypeInBits: return "no-memory-type-in-bits";
        case MemoryTypeSelectionStatus::kAllCandidatesForbidden: return "all-candidates-forbidden";
        case MemoryTypeSelectionStatus::kMissingRequiredProperties: return "missing-required-properties";
        case MemoryTypeSelectionStatus::kTooManyMemoryTypes: return "too-many-memory-types";
    }
    return "unknown";
}

struct MemoryTypeSelectionPlan {
    // VK_MAX_MEMORY_TYPES when no type was selected.
    uint32_t memory_type_index{VK_MAX_MEMORY_TYPES};
    MemoryTypeSelectionStatus status{MemoryTypeSelectionStatus::kNoMemoryTypeInBits};
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == MemoryTypeSelectionStatus::kOk;
    }
};

// Picks a memory type for a dedicated allocation. `memory_type_bits` is the
// VkMemoryRequirements::memoryTypeBits mask of the resource, `candidates`
// describes every memory type of the physical device in index order.
//
// The first candidate that satisfies `required_flags` and avoids every
// `forbidden_flags` bit is remembered as the fallback, and the first such
// candidate that additionally satisfies `preferred_flags` wins immediately.
// This mirrors the greedy preference order Vulkan Memory Allocator uses for its
// own allocations, so an explicitly zero-initialized or decompression buffer
// lands in the same memory type the pre-existing path would have chosen.
[[nodiscard]] constexpr MemoryTypeSelectionPlan select_memory_type(
    uint32_t memory_type_bits, uint32_t memory_type_count,
    const MemoryTypeCandidate *candidates,
    uint32_t required_flags, uint32_t preferred_flags,
    uint32_t forbidden_flags) noexcept {
    if (memory_type_count > 32u) {
        return {VK_MAX_MEMORY_TYPES, MemoryTypeSelectionStatus::kTooManyMemoryTypes};
    }
    if (candidates == nullptr || memory_type_count == 0u) {
        return {VK_MAX_MEMORY_TYPES, MemoryTypeSelectionStatus::kNoMemoryTypeInBits};
    }
    auto any_addressable = false;
    auto any_allowed = false;
    auto fallback = VK_MAX_MEMORY_TYPES;
    for (auto i = 0u; i < memory_type_count; i++) {
        if ((memory_type_bits & (1u << i)) == 0u) { continue; }
        any_addressable = true;
        if ((candidates[i].property_flags & forbidden_flags) != 0u) { continue; }
        any_allowed = true;
        if ((candidates[i].property_flags & required_flags) != required_flags) { continue; }
        if ((candidates[i].property_flags & preferred_flags) == preferred_flags) {
            return {i, MemoryTypeSelectionStatus::kOk};
        }
        if (fallback == VK_MAX_MEMORY_TYPES) { fallback = i; }
    }
    if (fallback != VK_MAX_MEMORY_TYPES) {
        return {fallback, MemoryTypeSelectionStatus::kOk};
    }
    if (!any_addressable) {
        return {VK_MAX_MEMORY_TYPES, MemoryTypeSelectionStatus::kNoMemoryTypeInBits};
    }
    if (!any_allowed) {
        return {VK_MAX_MEMORY_TYPES, MemoryTypeSelectionStatus::kAllCandidatesForbidden};
    }
    return {VK_MAX_MEMORY_TYPES, MemoryTypeSelectionStatus::kMissingRequiredProperties};
}

// ---------------------------------------------------------------------------
// VK_EXT_memory_decompression
// ---------------------------------------------------------------------------

// VUID-VkDecompressMemoryInfoEXT-decompressionMethod-11762 (and the indirect
// count twin): a GDeflate 1.0 region must decompress to at most 64 KiB.
inline constexpr uint64_t kGDeflateMaxDecompressedSize = 65536ull;

// Flat mirror of VkDecompressMemoryRegionEXT so the planner never aliases
// device memory.
struct DecompressionRegion {
    uint64_t src_address{0u};
    uint64_t dst_address{0u};
    uint64_t compressed_size{0u};
    uint64_t decompressed_size{0u};
};

enum class DecompressionValidationStatus : uint32_t {
    kOk = 0u,
    // The extension is unsupported, switched off, or forced off for an
    // imported logical device.
    kExtensionDisabled,
    // No region was supplied.
    kNoRegions,
    // VUID-...-decompressionMethod-07690: exactly one bit must be set.
    kMethodNotSingleBit,
    // VUID-...-decompressionMethod-11763/11810: the method must be reported by
    // VkPhysicalDeviceMemoryDecompressionPropertiesEXT::decompressionMethods.
    kMethodUnsupported,
    // VUID-...-srcAddress-07685 / -dstAddress-07687: 4-byte alignment.
    kSourceNotAligned,
    kDestinationNotAligned,
    // VUID-...-compressedSize-11795 / -decompressedSize-11796: non-zero size.
    kEmptySource,
    kEmptyDestination,
    // VUID-VkDecompressMemoryInfoEXT-decompressionMethod-11762.
    kGDeflateRegionTooLarge,
    // src and dst must not name the same range (undefined results).
    kSourceEqualsDestination,
    // VUID-...-srcAddress-07691: the two ranges must not overlap.
    kRangesOverlap,
    // A range end address overflows 64 bits.
    kAddressOverflow,
    // VUID-vkCmdDecompressMemoryIndirectCountEXT-stride-11767: the stride must
    // be a multiple of 4 and at least sizeof(VkDecompressMemoryRegionEXT).
    kIndirectStrideInvalid,
    // VUID-vkCmdDecompressMemoryIndirectCountEXT-maxDecompressionCount-11768.
    kIndirectCountTooLarge,
};

[[nodiscard]] constexpr const char *decompression_validation_status_name(
    DecompressionValidationStatus status) noexcept {
    switch (status) {
        case DecompressionValidationStatus::kOk: return "ok";
        case DecompressionValidationStatus::kExtensionDisabled: return "extension-disabled";
        case DecompressionValidationStatus::kNoRegions: return "no-regions";
        case DecompressionValidationStatus::kMethodNotSingleBit: return "method-not-single-bit";
        case DecompressionValidationStatus::kMethodUnsupported: return "method-unsupported";
        case DecompressionValidationStatus::kSourceNotAligned: return "source-not-4-byte-aligned";
        case DecompressionValidationStatus::kDestinationNotAligned: return "destination-not-4-byte-aligned";
        case DecompressionValidationStatus::kEmptySource: return "empty-source";
        case DecompressionValidationStatus::kEmptyDestination: return "empty-destination";
        case DecompressionValidationStatus::kGDeflateRegionTooLarge: return "gdeflate-region-too-large";
        case DecompressionValidationStatus::kSourceEqualsDestination: return "source-equals-destination";
        case DecompressionValidationStatus::kRangesOverlap: return "ranges-overlap";
        case DecompressionValidationStatus::kAddressOverflow: return "address-overflow";
        case DecompressionValidationStatus::kIndirectStrideInvalid: return "indirect-stride-invalid";
        case DecompressionValidationStatus::kIndirectCountTooLarge: return "indirect-count-too-large";
    }
    return "unknown";
}

struct DecompressionPlan {
    DecompressionValidationStatus status{DecompressionValidationStatus::kOk};
    // Index of the offending region, 0 when the failure is not region-local.
    uint32_t region_index{0u};
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == DecompressionValidationStatus::kOk;
    }
};

// Number of bits set in `method`.
[[nodiscard]] constexpr uint32_t decompression_method_bit_count(
    VkMemoryDecompressionMethodFlagsEXT method) noexcept {
    auto bits = static_cast<uint64_t>(method);
    auto count = 0u;
    while (bits != 0u) {
        count += static_cast<uint32_t>(bits & 1u);
        bits >>= 1u;
    }
    return count;
}

// Validates one decompression region against the spec's valid-usage rules.
[[nodiscard]] constexpr DecompressionValidationStatus
validate_decompression_region(
    DecompressionRegion region,
    VkMemoryDecompressionMethodFlagsEXT method) noexcept {
    if ((region.src_address & 3u) != 0u) {
        return DecompressionValidationStatus::kSourceNotAligned;
    }
    if ((region.dst_address & 3u) != 0u) {
        return DecompressionValidationStatus::kDestinationNotAligned;
    }
    if (region.compressed_size == 0u) {
        return DecompressionValidationStatus::kEmptySource;
    }
    if (region.decompressed_size == 0u) {
        return DecompressionValidationStatus::kEmptyDestination;
    }
    if ((method & VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_EXT) != 0u &&
        region.decompressed_size > kGDeflateMaxDecompressedSize) {
        return DecompressionValidationStatus::kGDeflateRegionTooLarge;
    }
    if (region.src_address == region.dst_address) {
        return DecompressionValidationStatus::kSourceEqualsDestination;
    }
    if (region.src_address > UINT64_MAX - region.compressed_size ||
        region.dst_address > UINT64_MAX - region.decompressed_size) {
        return DecompressionValidationStatus::kAddressOverflow;
    }
    auto src_end = region.src_address + region.compressed_size;
    auto dst_end = region.dst_address + region.decompressed_size;
    if (region.src_address < dst_end && region.dst_address < src_end) {
        return DecompressionValidationStatus::kRangesOverlap;
    }
    return DecompressionValidationStatus::kOk;
}

// Validates a whole direct (vkCmdDecompressMemoryEXT) request: negotiation,
// method bits, the supported-method mask, every region and every pair of
// regions (the spec leaves overlapping *regions* undefined, so reject them).
// `regions` may be null only when `region_count` is zero.
[[nodiscard]] constexpr DecompressionPlan plan_decompression(
    bool extension_enabled, VkMemoryDecompressionMethodFlagsEXT method,
    VkMemoryDecompressionMethodFlagsEXT supported_methods,
    const DecompressionRegion *regions, uint32_t region_count) noexcept {
    if (!extension_enabled) {
        return {DecompressionValidationStatus::kExtensionDisabled, 0u};
    }
    if (decompression_method_bit_count(method) != 1u) {
        return {DecompressionValidationStatus::kMethodNotSingleBit, 0u};
    }
    if ((static_cast<uint64_t>(method) & static_cast<uint64_t>(supported_methods)) == 0u) {
        return {DecompressionValidationStatus::kMethodUnsupported, 0u};
    }
    if (region_count == 0u || regions == nullptr) {
        return {DecompressionValidationStatus::kNoRegions, 0u};
    }
    for (auto i = 0u; i < region_count; i++) {
        auto status = validate_decompression_region(regions[i], method);
        if (status != DecompressionValidationStatus::kOk) {
            return {status, i};
        }
    }
    for (auto i = 0u; i < region_count; i++) {
        auto a_src_end = regions[i].src_address + regions[i].compressed_size;
        auto a_dst_end = regions[i].dst_address + regions[i].decompressed_size;
        for (auto j = i + 1u; j < region_count; j++) {
            auto b_src_end = regions[j].src_address + regions[j].compressed_size;
            auto b_dst_end = regions[j].dst_address + regions[j].decompressed_size;
            // Sources may alias (they are read only); destinations and
            // source/destination pairs of different regions must not overlap.
            if (regions[i].dst_address < b_dst_end && regions[j].dst_address < a_dst_end) {
                return {DecompressionValidationStatus::kRangesOverlap, j};
            }
            if (regions[i].src_address < b_dst_end && regions[j].dst_address < a_src_end) {
                return {DecompressionValidationStatus::kRangesOverlap, j};
            }
            if (regions[j].src_address < a_dst_end && regions[i].dst_address < b_src_end) {
                return {DecompressionValidationStatus::kRangesOverlap, i};
            }
        }
    }
    return {DecompressionValidationStatus::kOk, 0u};
}

// Byte size of one VkDecompressMemoryRegionEXT, the minimum legal
// `vkCmdDecompressMemoryIndirectCountEXT` stride
// (VUID-vkCmdDecompressMemoryIndirectCountEXT-stride-11767).
inline constexpr uint32_t kDecompressionIndirectStride = 4u * sizeof(uint64_t);

// Validates a vkCmdDecompressMemoryIndirectCountEXT request.
[[nodiscard]] constexpr DecompressionPlan plan_decompression_indirect(
    bool extension_enabled, VkMemoryDecompressionMethodFlagsEXT method,
    VkMemoryDecompressionMethodFlagsEXT supported_methods,
    uint64_t indirect_commands_address, uint64_t indirect_count_address,
    uint32_t max_decompression_count, uint64_t device_max_indirect_count,
    uint32_t stride) noexcept {
    if (!extension_enabled) {
        return {DecompressionValidationStatus::kExtensionDisabled, 0u};
    }
    if (decompression_method_bit_count(method) != 1u) {
        return {DecompressionValidationStatus::kMethodNotSingleBit, 0u};
    }
    if ((static_cast<uint64_t>(method) & static_cast<uint64_t>(supported_methods)) == 0u) {
        return {DecompressionValidationStatus::kMethodUnsupported, 0u};
    }
    if ((indirect_commands_address & 3u) != 0u) {
        return {DecompressionValidationStatus::kSourceNotAligned, 0u};
    }
    if ((indirect_count_address & 3u) != 0u) {
        return {DecompressionValidationStatus::kDestinationNotAligned, 0u};
    }
    if (stride < kDecompressionIndirectStride || (stride & 3u) != 0u) {
        return {DecompressionValidationStatus::kIndirectStrideInvalid, 0u};
    }
    if (static_cast<uint64_t>(max_decompression_count) > device_max_indirect_count) {
        return {DecompressionValidationStatus::kIndirectCountTooLarge, 0u};
    }
    if (max_decompression_count == 0u) {
        return {DecompressionValidationStatus::kNoRegions, 0u};
    }
    return {DecompressionValidationStatus::kOk, 0u};
}

// ---------------------------------------------------------------------------
// VK_EXT_image_tiling_control
// ---------------------------------------------------------------------------

// Backend-level tiling policy. `kDefault` is the pre-existing behaviour and the
// fallback for a device without the extension.
enum class ImageTilingPolicy : uint32_t {
    kDefault = 0u,
    kMinimizeMemory,
    kMaximizePerformance,
};

[[nodiscard]] constexpr const char *image_tiling_policy_name(
    ImageTilingPolicy policy) noexcept {
    switch (policy) {
        case ImageTilingPolicy::kDefault: return "default";
        case ImageTilingPolicy::kMinimizeMemory: return "minimize-memory";
        case ImageTilingPolicy::kMaximizePerformance: return "maximize-performance";
    }
    return "unknown";
}

struct ImageTilingControlPlan {
    // Whether VkImageTilingControlCreateInfoEXT belongs in the
    // VkImageCreateInfo pNext chain of this image.
    bool chain_create_info{false};
    VkImageTilingControlEXT tiling_control{VK_IMAGE_TILING_CONTROL_DEFAULT_EXT};
    // True when the requested policy was actually expressible.
    [[nodiscard]] constexpr bool applied() const noexcept {
        return chain_create_info;
    }
};

// Per the spec, VK_IMAGE_TILING_CONTROL_DEFAULT_EXT "is the same selection as
// with the VkImageTilingControlCreateInfoEXT structure omitted", so the
// structure is chained only for a non-default policy on an enabled device whose
// image is not linear (VUID-VkImageTilingControlCreateInfoEXT-tilingControl-12479).
[[nodiscard]] constexpr ImageTilingControlPlan plan_image_tiling_control(
    bool extension_enabled, ImageTilingPolicy policy,
    VkImageTiling tiling) noexcept {
    if (policy == ImageTilingPolicy::kDefault || !extension_enabled ||
        tiling == VK_IMAGE_TILING_LINEAR) {
        return {.chain_create_info = false,
                .tiling_control = VK_IMAGE_TILING_CONTROL_DEFAULT_EXT};
    }
    return {.chain_create_info = true,
            .tiling_control = policy == ImageTilingPolicy::kMinimizeMemory ?
                                  VK_IMAGE_TILING_CONTROL_MIN_SIZE_EXT :
                                  VK_IMAGE_TILING_CONTROL_MAX_PERFORMANCE_EXT};
}

// ---------------------------------------------------------------------------
// VK_EXT_texture_compression_astc_3d
// ---------------------------------------------------------------------------

// The 2D ASTC block footprints the backend's PixelFormat enumeration exposes.
enum class AstcBlockFootprint : uint32_t {
    kNone = 0u,
    k4x4,
    k5x5,
    k6x6,
    k8x8,
    k10x10,
    k12x12,
};

// Block footprint of a 2D ASTC PixelFormat, or kNone for every other format.
[[nodiscard]] constexpr AstcBlockFootprint astc_block_footprint(
    luisa::compute::PixelFormat format) noexcept {
    using PF = luisa::compute::PixelFormat;
    switch (format) {
        case PF::ASTC_4x4:
        case PF::ASTC_4x4_SRGB: return AstcBlockFootprint::k4x4;
        case PF::ASTC_5x5:
        case PF::ASTC_5x5_SRGB: return AstcBlockFootprint::k5x5;
        case PF::ASTC_6x6:
        case PF::ASTC_6x6_SRGB: return AstcBlockFootprint::k6x6;
        case PF::ASTC_8x8:
        case PF::ASTC_8x8_SRGB: return AstcBlockFootprint::k8x8;
        case PF::ASTC_10x10:
        case PF::ASTC_10x10_SRGB: return AstcBlockFootprint::k10x10;
        case PF::ASTC_12x12:
        case PF::ASTC_12x12_SRGB: return AstcBlockFootprint::k12x12;
        default: return AstcBlockFootprint::kNone;
    }
}

enum class Astc3DFormatStatus : uint32_t {
    // Not an ASTC block format; the caller uses the plain 2D mapping.
    kNotAstcBlockFormat = 0u,
    // VK_EXT_texture_compression_astc_3d is unsupported/switched off; the
    // caller keeps the pre-existing "unsupported format" failure.
    kExtensionDisabled,
    // No 3D compressed ASTC format covers this block footprint
    // (8x8, 10x10 and 12x12 have no 3D twin).
    kNoAstc3DCounterpart,
    // Mapped to a VK_FORMAT_ASTC_*_BLOCK_EXT 3D format.
    kAstc3DFormat,
};

[[nodiscard]] constexpr const char *astc_3d_format_status_name(
    Astc3DFormatStatus status) noexcept {
    switch (status) {
        case Astc3DFormatStatus::kNotAstcBlockFormat: return "not-astc-block-format";
        case Astc3DFormatStatus::kExtensionDisabled: return "extension-disabled";
        case Astc3DFormatStatus::kNoAstc3DCounterpart: return "no-astc-3d-counterpart";
        case Astc3DFormatStatus::kAstc3DFormat: return "astc-3d-format";
    }
    return "unknown";
}

struct Astc3DFormatPlan {
    Astc3DFormatStatus status{Astc3DFormatStatus::kNotAstcBlockFormat};
    VkFormat format{VK_FORMAT_UNDEFINED};
    [[nodiscard]] constexpr bool is_astc_block_format() const noexcept {
        return status != Astc3DFormatStatus::kNotAstcBlockFormat;
    }
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == Astc3DFormatStatus::kAstc3DFormat;
    }
};

// Maps a 3D image requested with a 2D ASTC block format onto the equally sized
// 3D-compressed ASTC format of VK_EXT_texture_compression_astc_3d.
//
// A 2D ASTC format only covers an NxNx1 block, so it can never be used for a
// VK_IMAGE_TYPE_3D image: without the extension such a request fails. With the
// extension, the NxNxN 3D block format is the compatible 3D arrangement. Only
// 4x4x4, 5x5x5 and 6x6x6 exist, so 8x8, 10x10 and 12x12 keep failing even when
// the extension is enabled.
[[nodiscard]] constexpr Astc3DFormatPlan plan_astc_3d_format(
    bool extension_enabled, AstcBlockFootprint footprint,
    bool srgb, uint32_t dimension) noexcept {
    if (footprint == AstcBlockFootprint::kNone || dimension != 3u) {
        return {Astc3DFormatStatus::kNotAstcBlockFormat, VK_FORMAT_UNDEFINED};
    }
    if (!extension_enabled) {
        return {Astc3DFormatStatus::kExtensionDisabled, VK_FORMAT_UNDEFINED};
    }
    switch (footprint) {
        case AstcBlockFootprint::k4x4:
            return {Astc3DFormatStatus::kAstc3DFormat,
                    srgb ? VK_FORMAT_ASTC_4x4x4_SRGB_BLOCK_EXT :
                           VK_FORMAT_ASTC_4x4x4_UNORM_BLOCK_EXT};
        case AstcBlockFootprint::k5x5:
            return {Astc3DFormatStatus::kAstc3DFormat,
                    srgb ? VK_FORMAT_ASTC_5x5x5_SRGB_BLOCK_EXT :
                           VK_FORMAT_ASTC_5x5x5_UNORM_BLOCK_EXT};
        case AstcBlockFootprint::k6x6:
            return {Astc3DFormatStatus::kAstc3DFormat,
                    srgb ? VK_FORMAT_ASTC_6x6x6_SRGB_BLOCK_EXT :
                           VK_FORMAT_ASTC_6x6x6_UNORM_BLOCK_EXT};
        case AstcBlockFootprint::k8x8:
        case AstcBlockFootprint::k10x10:
        case AstcBlockFootprint::k12x12:
        case AstcBlockFootprint::kNone:
            break;
    }
    return {Astc3DFormatStatus::kNoAstc3DCounterpart, VK_FORMAT_UNDEFINED};
}

// ---------------------------------------------------------------------------
// VK_KHR_extended_flags
// ---------------------------------------------------------------------------
//
// VK_KHR_extended_flags only widens flag *types* (VkFormatFeatureFlags4KHR,
// VkImageCreateFlags2KHR, VkImageUsageFlags2KHR) and defines no new bits. The
// backend's flag plumbing therefore stays 32-bit and the extension is
// negotiated, logged and reported but never marked active. The invariant the
// backend relies on is asserted in test_vk_ext_memory.cpp: every
// VkImageCreateFlagBits, VkImageUsageFlagBits and VkBufferUsageFlagBits the
// backend composes fits in 32 bits, and the single 64-bit buffer usage bit the
// memory-decompression path needs (VK_BUFFER_USAGE_2_MEMORY_DECOMPRESSION_BIT_EXT)
// is expressed through VkBufferUsageFlags2CreateInfoKHR instead of widening the
// existing VkBufferUsageFlags plumbing.

// Whether VkBufferUsageFlags2CreateInfoKHR may be chained on VkBufferCreateInfo
// to express usage bits above bit 31. It is defined by VK_KHR_maintenance5 and
// promoted to Vulkan 1.4, so a Vulkan 1.3 device only reaches it through the
// extension.
//
// `api_version` must be the effective API version of the logical device (the
// lower of the instance and physical-device versions), not the physical-device
// version alone: a 1.4 physical device behind a 1.3 instance cannot use core 1.4
// structures. `maintenance5_available` is the caller's attestation that the
// logical device offers the structure. Vulkan exposes no query for a logical
// device's enabled extension list, so the backend passes physical-device support
// and documents the residual risk (see vk_memory_decompression.cpp).
[[nodiscard]] constexpr bool plan_buffer_usage_flags2(
    uint32_t api_version, bool maintenance5_available) noexcept {
    return maintenance5_available || api_version >= VK_API_VERSION_1_4;
}

}// namespace lc::vk::detail
