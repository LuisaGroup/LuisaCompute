#pragma once
#include <cstdint>
#include <iterator>
#include <limits>
#include <vulkan/vulkan_core.h>

namespace lc::vk {
class Device;
}// namespace lc::vk

namespace lc::vk::detail {

// ===========================================================================
// VK_EXT_descriptor_heap: host-only planning
// ===========================================================================
//
// `VK_EXT_descriptor_heap` replaces descriptor sets with two *device address*
// ranges ("heaps") that a command buffer binds directly:
//
//   * the resource heap holds buffer, image (including texel buffer),
//     acceleration-structure and (with `VK_ARM_tensors`) tensor descriptors,
//   * the sampler heap holds sampler descriptors only.
//
// Descriptors are produced on the host (`vkWriteResourceDescriptorsEXT` /
// `vkWriteSamplerDescriptorsEXT` write the implementation's bit pattern into a
// host address range) and are read by the GPU out of a buffer created with
// `VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT`, so the application owns the staging
// upload and the memory.
//
// Everything in this namespace is a pure function of values the caller already
// negotiated (the `VkPhysicalDeviceDescriptorHeapPropertiesEXT` limits and the
// `descriptorHeap`/`descriptorHeapCaptureReplay` feature bits). No Vulkan entry
// point is called and no `Device` is touched, which keeps the layout of a heap
// unit-testable on a machine without the extension - the only evidence available
// for an extension this backend cannot yet route work through.
//
// The layout this planner produces is deliberately simple and fully explicit:
//
//   heap:  [ reserved range ][ region 0 ][ padding ][ region 1 ] ...
//          ^ offset 0        ^ reserved_range_size
//
// * the implementation reserved range (`minResourceHeapReservedRange` /
//   `minSamplerHeapReservedRange[-WithEmbedded]`) always starts at offset 0, so
//   `reservedRangeOffset` is 0 and therefore a multiple of every descriptor
//   alignment (VUID-vkCmdBindResourceHeapEXT-pBindInfo-11435/11436),
// * every region starts at a multiple of its own descriptor alignment and has a
//   stride of `align_up(descriptorSize, descriptorAlignment)`, so the
//   `heapOffset`/`heapArrayStride` values a `VkDescriptorSetAndBindingMappingEXT`
//   reports for that region satisfy VUID-VkDescriptorSetAndBindingMappingEXT-
//   source-11251/11252/11253/12406,
// * the heap size is `align_up(payload, heapAlignment)` and never exceeds
//   `max*HeapSize`, so `heapRange.address`/`heapRange.size` satisfy
//   VUID-vkCmdBindResourceHeapEXT-pBindInfo-11234/11235.
//
// The planner never produces a capture-replay structure. `descriptorHeapCaptureReplay`
// is left at VK_FALSE by the extension registry (it enables exactly one feature
// bit, `descriptorHeap`), so any request that would need replay must be refused
// instead of silently degrading.

// ---------------------------------------------------------------------------
// Descriptor classes and the heap that owns them
// ---------------------------------------------------------------------------

// The four descriptor families the resource/sampler heap split distinguishes.
// The class decides which heap a slot may live in *and* which advertised
// descriptor size/alignment applies to it.
enum class DescriptorClass : uint8_t {
    // VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER / STORAGE_BUFFER /
    // ACCELERATION_STRUCTURE_KHR (all three take `data.pAddressRange`).
    kBuffer = 0u,
    // VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE / STORAGE_IMAGE / INPUT_ATTACHMENT and
    // the texel-buffer types (SPIR-V declares a texel buffer as OpTypeImage, so
    // its mapping offset must respect imageDescriptorAlignment).
    kImage,
    // VK_DESCRIPTOR_TYPE_TENSOR_ARM (`data.pTensorARM`, VK_ARM_tensors).
    kTensor,
    // VK_DESCRIPTOR_TYPE_SAMPLER, sampler heap only.
    kSampler,
};

inline constexpr uint32_t kDescriptorClassCount = 4u;

[[nodiscard]] constexpr const char *descriptor_class_name(DescriptorClass klass) noexcept {
    switch (klass) {
        case DescriptorClass::kBuffer: return "buffer";
        case DescriptorClass::kImage: return "image";
        case DescriptorClass::kTensor: return "tensor";
        case DescriptorClass::kSampler: return "sampler";
    }
    return "unknown";
}

[[nodiscard]] constexpr bool descriptor_class_is_sampler(DescriptorClass klass) noexcept {
    return klass == DescriptorClass::kSampler;
}

// One heap holds at most this many regions. Four are needed by the bindless
// layout (buffer + 2D + 3D in the resource heap, samplers in the sampler heap);
// the bound is generous but fixed so the plan stays a plain value type.
inline constexpr uint32_t kMaxDescriptorHeapRegions = 8u;

// ---------------------------------------------------------------------------
// Device limits
// ---------------------------------------------------------------------------

// Plain mirror of the members of `VkPhysicalDeviceDescriptorHeapPropertiesEXT`
// (plus the tensor members of `VkPhysicalDeviceDescriptorHeapTensorPropertiesARM`)
// that decide a heap layout. Kept as a value type so the planner never depends on
// a queried Vulkan structure and the unit test can describe a device inline.
struct DescriptorHeapLimits {
    uint64_t sampler_heap_alignment{0u};
    uint64_t resource_heap_alignment{0u};
    uint64_t max_sampler_heap_size{0u};
    uint64_t max_resource_heap_size{0u};
    uint64_t min_sampler_heap_reserved_range{0u};
    uint64_t min_sampler_heap_reserved_range_with_embedded{0u};
    uint64_t min_resource_heap_reserved_range{0u};
    uint64_t sampler_descriptor_size{0u};
    uint64_t image_descriptor_size{0u};
    uint64_t buffer_descriptor_size{0u};
    // Zero unless `VK_ARM_tensors` is enabled *and* its descriptor-heap tensor
    // properties were queried; a tensor region then reports
    // kDescriptorSizeUnavailable instead of guessing a size.
    uint64_t tensor_descriptor_size{0u};
    uint64_t sampler_descriptor_alignment{0u};
    uint64_t image_descriptor_alignment{0u};
    uint64_t buffer_descriptor_alignment{0u};
    uint64_t tensor_descriptor_alignment{0u};
    uint64_t max_push_data_size{0u};
    uint32_t max_descriptor_heap_embedded_samplers{0u};
    bool sparse_descriptor_heaps{false};
    bool protected_descriptor_heaps{false};
};

[[nodiscard]] constexpr DescriptorHeapLimits descriptor_heap_limits(
    const VkPhysicalDeviceDescriptorHeapPropertiesEXT &properties) noexcept {
    return DescriptorHeapLimits{
        .sampler_heap_alignment = properties.samplerHeapAlignment,
        .resource_heap_alignment = properties.resourceHeapAlignment,
        .max_sampler_heap_size = properties.maxSamplerHeapSize,
        .max_resource_heap_size = properties.maxResourceHeapSize,
        .min_sampler_heap_reserved_range = properties.minSamplerHeapReservedRange,
        .min_sampler_heap_reserved_range_with_embedded = properties.minSamplerHeapReservedRangeWithEmbedded,
        .min_resource_heap_reserved_range = properties.minResourceHeapReservedRange,
        .sampler_descriptor_size = properties.samplerDescriptorSize,
        .image_descriptor_size = properties.imageDescriptorSize,
        .buffer_descriptor_size = properties.bufferDescriptorSize,
        .tensor_descriptor_size = 0u,
        .sampler_descriptor_alignment = properties.samplerDescriptorAlignment,
        .image_descriptor_alignment = properties.imageDescriptorAlignment,
        .buffer_descriptor_alignment = properties.bufferDescriptorAlignment,
        .tensor_descriptor_alignment = 0u,
        .max_push_data_size = properties.maxPushDataSize,
        .max_descriptor_heap_embedded_samplers = properties.maxDescriptorHeapEmbeddedSamplers,
        .sparse_descriptor_heaps = properties.sparseDescriptorHeaps == VK_TRUE,
        .protected_descriptor_heaps = properties.protectedDescriptorHeaps == VK_TRUE};
}

// `VK_ARM_tensors` publishes its descriptor sizes through its own property
// structure, chained next to `VkPhysicalDeviceDescriptorHeapPropertiesEXT`. The
// caller must only use this overload when both extensions are enabled; the
// planner itself treats a zero tensor size as "tensors unavailable".
[[nodiscard]] constexpr DescriptorHeapLimits descriptor_heap_limits(
    const VkPhysicalDeviceDescriptorHeapPropertiesEXT &properties,
    const VkPhysicalDeviceDescriptorHeapTensorPropertiesARM &tensor_properties) noexcept {
    auto limits = descriptor_heap_limits(properties);
    limits.tensor_descriptor_size = tensor_properties.tensorDescriptorSize;
    limits.tensor_descriptor_alignment = tensor_properties.tensorDescriptorAlignment;
    return limits;
}

// ---------------------------------------------------------------------------
// Arithmetic helpers
// ---------------------------------------------------------------------------

// Descriptor sizes returned by `vkGetPhysicalDeviceDescriptorSizeEXT` are
// explicitly allowed to be non power-of-two, so every rounding step in this
// header uses modulo arithmetic instead of bit masking.
[[nodiscard]] constexpr uint64_t align_up(uint64_t value, uint64_t alignment) noexcept {
    if (alignment <= 1u) { return value; }
    auto remainder = value % alignment;
    if (remainder == 0u) { return value; }
    auto padding = alignment - remainder;
    if (value > std::numeric_limits<uint64_t>::max() - padding) {
        // Saturate instead of wrapping: an overflowing request can then only
        // fail the size check below, never produce a small "legal" heap.
        return std::numeric_limits<uint64_t>::max();
    }
    return value + padding;
}

[[nodiscard]] constexpr uint64_t align_down(uint64_t value, uint64_t alignment) noexcept {
    if (alignment <= 1u) { return value; }
    return value - value % alignment;
}

[[nodiscard]] constexpr uint64_t descriptor_size(const DescriptorHeapLimits &limits,
                                                 DescriptorClass klass) noexcept {
    switch (klass) {
        case DescriptorClass::kBuffer: return limits.buffer_descriptor_size;
        case DescriptorClass::kImage: return limits.image_descriptor_size;
        case DescriptorClass::kTensor: return limits.tensor_descriptor_size;
        case DescriptorClass::kSampler: return limits.sampler_descriptor_size;
    }
    return 0u;
}

[[nodiscard]] constexpr uint64_t descriptor_alignment(const DescriptorHeapLimits &limits,
                                                      DescriptorClass klass) noexcept {
    switch (klass) {
        case DescriptorClass::kBuffer: return limits.buffer_descriptor_alignment;
        case DescriptorClass::kImage: return limits.image_descriptor_alignment;
        case DescriptorClass::kTensor: return limits.tensor_descriptor_alignment;
        case DescriptorClass::kSampler: return limits.sampler_descriptor_alignment;
    }
    return 0u;
}

// Distance between two neighbouring slots of one class. Rounding the advertised
// size up to the advertised alignment is what makes
// `heapOffset + slot * stride` legal for the mapping structures, and it wastes
// at most one alignment step per slot compared to the "packed" sizes reported by
// `vkGetPhysicalDeviceDescriptorSizeEXT`.
[[nodiscard]] constexpr uint64_t descriptor_stride(const DescriptorHeapLimits &limits,
                                                    DescriptorClass klass) noexcept {
    return align_up(descriptor_size(limits, klass), descriptor_alignment(limits, klass));
}

// Number of descriptors of `stride` bytes that fit in `available_bytes`.
[[nodiscard]] constexpr uint64_t descriptor_slot_capacity(uint64_t available_bytes,
                                                          uint64_t stride) noexcept {
    return stride == 0u ? 0u : available_bytes / stride;
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

// Every planner answers with one of these instead of asserting, so the unit test
// can drive the whole matrix without a device and the backend can log the exact
// reason it stays on the descriptor-set path.
enum class DescriptorHeapError : uint8_t {
    kNone = 0u,
    // `OptionalExtension::DESCRIPTOR_HEAP` is not enabled (unsupported, switched
    // off through VulkanDeviceConfigExt, or an imported logical device). The
    // pre-existing descriptor-set path must be used.
    kExtensionDisabled,
    // The caller asked for a capture-replay structure. This backend never sets
    // `VkPhysicalDeviceDescriptorHeapFeaturesEXT::descriptorHeapCaptureReplay`.
    kCaptureReplayUnsupported,
    // The device reported no heap alignment / no maximum heap size, so no layout
    // can be derived from its properties.
    kNoDeviceLimits,
    // The heap has no region, or a region requested zero slots.
    kZeroSlots,
    kTooManyRegions,
    // The class has no advertised descriptor size/alignment on this device (the
    // usual case for tensors without VK_ARM_tensors).
    kDescriptorSizeUnavailable,
    // Samplers may only live in the sampler heap, everything else only in the
    // resource heap.
    kSamplerInResourceHeap,
    kResourceInSamplerHeap,
    // Embedded samplers exist only in the sampler heap and only when the device
    // advertises maxDescriptorHeapEmbeddedSamplers.
    kEmbeddedSamplersRequireSamplerHeap,
    kEmbeddedSamplersUnsupported,
    kTooManyEmbeddedSamplers,
    // The implementation reserved range alone does not fit the effective maximum.
    kHeapSizeLimitTooSmall,
    // The requested payload (reserved range + regions) does not fit the effective
    // maximum. `slot_capacity`/`fitting_slots` report what would fit.
    kHeapTooSmall,
    // The slot index is outside its region (or the class has no region at all).
    kSlotOutOfRange,
    // vkCmdPushDataEXT: offset and size must be multiples of 4 and
    // offset + size must not exceed maxPushDataSize.
    kZeroPushData,
    kPushDataMisaligned,
    kPushDataOutOfRange,
};

[[nodiscard]] constexpr const char *descriptor_heap_error_name(DescriptorHeapError error) noexcept {
    switch (error) {
        case DescriptorHeapError::kNone: return "none";
        case DescriptorHeapError::kExtensionDisabled: return "VK_EXT_descriptor_heap not enabled";
        case DescriptorHeapError::kCaptureReplayUnsupported: return "descriptor heap capture replay is not enabled";
        case DescriptorHeapError::kNoDeviceLimits: return "device reported no descriptor heap limits";
        case DescriptorHeapError::kZeroSlots: return "no descriptor slots requested";
        case DescriptorHeapError::kTooManyRegions: return "too many descriptor heap regions";
        case DescriptorHeapError::kDescriptorSizeUnavailable: return "descriptor class has no advertised size";
        case DescriptorHeapError::kSamplerInResourceHeap: return "sampler descriptor in the resource heap";
        case DescriptorHeapError::kResourceInSamplerHeap: return "resource descriptor in the sampler heap";
        case DescriptorHeapError::kEmbeddedSamplersRequireSamplerHeap: return "embedded samplers outside the sampler heap";
        case DescriptorHeapError::kEmbeddedSamplersUnsupported: return "device supports no embedded samplers";
        case DescriptorHeapError::kTooManyEmbeddedSamplers: return "too many embedded samplers";
        case DescriptorHeapError::kHeapSizeLimitTooSmall: return "reserved range exceeds the heap size limit";
        case DescriptorHeapError::kHeapTooSmall: return "descriptor heap is too small for the requested slots";
        case DescriptorHeapError::kSlotOutOfRange: return "descriptor slot out of range";
        case DescriptorHeapError::kZeroPushData: return "empty push data";
        case DescriptorHeapError::kPushDataMisaligned: return "push data offset/size is not a multiple of 4";
        case DescriptorHeapError::kPushDataOutOfRange: return "push data exceeds maxPushDataSize";
    }
    return "unknown descriptor heap error";
}

// ---------------------------------------------------------------------------
// Heap request / plan
// ---------------------------------------------------------------------------

// One contiguous arena of equally sized descriptors inside a heap.
struct DescriptorHeapRegionRequest {
    DescriptorClass klass{DescriptorClass::kBuffer};
    uint64_t slots{0u};
};

struct DescriptorHeapRequest {
    // `Device::optional_extension_enabled(OptionalExtension::DESCRIPTOR_HEAP)`.
    // False selects the pre-existing descriptor-set path and makes every field
    // below irrelevant.
    bool extension_enabled{false};
    // True when the caller wants replayable descriptors. Refused (see
    // kCaptureReplayUnsupported) because the registry never enables
    // `descriptorHeapCaptureReplay`.
    bool capture_replay_requested{false};
    // Sampler heap when true, resource heap when false.
    bool sampler_heap{false};
    // Number of samplers embedded through
    // `VkDescriptorMappingSource*EXT::pEmbeddedSampler`. Non-zero selects
    // `minSamplerHeapReservedRangeWithEmbedded` as the reserved range and is only
    // legal for the sampler heap.
    uint32_t embedded_sampler_count{0u};
    // Hard cap on the heap size, or 0 to use the device maximum
    // (`maxResourceHeapSize` / `maxSamplerHeapSize`).
    uint64_t heap_size_limit{0u};
    uint32_t region_count{0u};
    DescriptorHeapRegionRequest regions[kMaxDescriptorHeapRegions]{};
};

struct DescriptorHeapRegionPlan {
    DescriptorClass klass{DescriptorClass::kBuffer};
    // Index of the region's first slot within the whole heap (the sum of the
    // slot counts of the preceding regions), not a byte offset.
    uint64_t first_slot{0u};
    uint64_t slots{0u};
    uint64_t descriptor_size{0u};
    uint64_t descriptor_alignment{0u};
    uint64_t descriptor_stride{0u};
    // Absolute byte offset of slot 0 inside the heap buffer, i.e. the value a
    // `VkDescriptorMappingSourceConstantOffsetEXT::heapOffset` must carry for
    // this region.
    uint64_t byte_offset{0u};
    // Slots of this region that fit the effective heap size; smaller than
    // `slots` exactly when the plan reports kHeapTooSmall.
    uint64_t fitting_slots{0u};
};

struct DescriptorHeapPlan {
    DescriptorHeapError error{DescriptorHeapError::kExtensionDisabled};
    // True only when `error == kNone`: the plan may be handed to the device.
    bool usable{false};
    bool sampler_heap{false};
    // Always false. The field exists so a caller cannot mistake "no replay
    // structures" for an oversight: this backend never enables the capture
    // replay feature bit and therefore never emits replay data.
    bool capture_replay_structs{false};
    uint64_t heap_alignment{0u};
    // Device maximum, reduced by the caller's `heap_size_limit`.
    uint64_t heap_size_limit{0u};
    // `align_down(heap_size_limit, heap_alignment)`: the largest payload that can
    // still be rounded up to a legal heap size. Every capacity below is derived
    // from this budget, not from the raw limit.
    uint64_t payload_budget{0u};
    uint64_t reserved_range_offset{0u};
    uint64_t reserved_range_size{0u};
    // End of the last region (reserved range included), before heap-alignment
    // rounding.
    uint64_t payload_bytes{0u};
    // Aligned heap size handed to `VkBindHeapInfoEXT::heapRange.size`.
    uint64_t heap_size{0u};
    // Total number of descriptors that fit (`fitting_slots` summed).
    uint64_t slot_capacity{0u};
    uint32_t region_count{0u};
    DescriptorHeapRegionPlan regions[kMaxDescriptorHeapRegions]{};
};

[[nodiscard]] constexpr DescriptorHeapPlan plan_descriptor_heap(
    const DescriptorHeapLimits &limits,
    const DescriptorHeapRequest &request) noexcept {
    DescriptorHeapPlan plan{};
    plan.sampler_heap = request.sampler_heap;
    plan.heap_alignment = request.sampler_heap ? limits.sampler_heap_alignment : limits.resource_heap_alignment;
    plan.heap_size_limit = request.sampler_heap ? limits.max_sampler_heap_size : limits.max_resource_heap_size;
    if (!request.extension_enabled) {
        plan.error = DescriptorHeapError::kExtensionDisabled;
        return plan;
    }
    if (request.capture_replay_requested) {
        // `descriptorHeapCaptureReplay` is never enabled (see the header comment),
        // so a capture-replay descriptor heap cannot be produced at all.
        plan.error = DescriptorHeapError::kCaptureReplayUnsupported;
        return plan;
    }
    if (plan.heap_alignment == 0u || plan.heap_size_limit == 0u) {
        plan.error = DescriptorHeapError::kNoDeviceLimits;
        return plan;
    }
    if (request.region_count == 0u && request.embedded_sampler_count == 0u) {
        // A heap with neither a region nor embedded samplers would only reserve
        // implementation storage and could never be accessed.
        plan.error = DescriptorHeapError::kZeroSlots;
        return plan;
    }
    if (request.region_count > kMaxDescriptorHeapRegions) {
        plan.error = DescriptorHeapError::kTooManyRegions;
        return plan;
    }
    // Validate every region before laying anything out, so a rejected plan is
    // never half-filled.
    for (uint32_t i = 0u; i < request.region_count; i++) {
        auto &region = request.regions[i];
        if (descriptor_class_is_sampler(region.klass) != request.sampler_heap) {
            plan.error = request.sampler_heap ?
                             DescriptorHeapError::kResourceInSamplerHeap :
                             DescriptorHeapError::kSamplerInResourceHeap;
            return plan;
        }
        if (region.slots == 0u) {
            plan.error = DescriptorHeapError::kZeroSlots;
            return plan;
        }
        if (descriptor_size(limits, region.klass) == 0u ||
            descriptor_alignment(limits, region.klass) == 0u) {
            plan.error = DescriptorHeapError::kDescriptorSizeUnavailable;
            return plan;
        }
    }
    if (request.embedded_sampler_count != 0u) {
        if (!request.sampler_heap) {
            plan.error = DescriptorHeapError::kEmbeddedSamplersRequireSamplerHeap;
            return plan;
        }
        if (limits.max_descriptor_heap_embedded_samplers == 0u) {
            plan.error = DescriptorHeapError::kEmbeddedSamplersUnsupported;
            return plan;
        }
        if (request.embedded_sampler_count > limits.max_descriptor_heap_embedded_samplers) {
            plan.error = DescriptorHeapError::kTooManyEmbeddedSamplers;
            return plan;
        }
    }
    auto reserved = request.sampler_heap ?
                        limits.min_sampler_heap_reserved_range :
                        limits.min_resource_heap_reserved_range;
    if (request.sampler_heap && request.embedded_sampler_count != 0u &&
        limits.min_sampler_heap_reserved_range_with_embedded > reserved) {
        // Using an embedded sampler forces the larger reservation so the
        // implementation can store the embedded descriptors next to the heap.
        reserved = limits.min_sampler_heap_reserved_range_with_embedded;
    }
    plan.reserved_range_offset = 0u;
    plan.reserved_range_size = reserved;
    if (request.heap_size_limit != 0u && request.heap_size_limit < plan.heap_size_limit) {
        plan.heap_size_limit = request.heap_size_limit;
    }
    // `heapSize = align_up(payload, heapAlignment)` must not exceed the limit, which
    // is the same as `payload <= align_down(limit, heapAlignment)`. Deriving the
    // capacity from the rounded budget keeps `fitting_slots` exact instead of one
    // alignment step optimistic.
    plan.payload_budget = align_down(plan.heap_size_limit, plan.heap_alignment);
    if (reserved >= plan.payload_budget) {
        plan.error = DescriptorHeapError::kHeapSizeLimitTooSmall;
        return plan;
    }
    plan.region_count = request.region_count;
    uint64_t offset = reserved;
    uint64_t first_slot = 0u;
    uint64_t capacity = 0u;
    bool overflow = false;
    for (uint32_t i = 0u; i < request.region_count; i++) {
        auto &region = request.regions[i];
        auto alignment = descriptor_alignment(limits, region.klass);
        auto size = descriptor_size(limits, region.klass);
        auto stride = align_up(size, alignment);
        offset = align_up(offset, alignment);
        auto fitting = offset < plan.payload_budget ?
                           descriptor_slot_capacity(plan.payload_budget - offset, stride) :
                           0u;
        if (fitting > region.slots) { fitting = region.slots; }
        plan.regions[i] = DescriptorHeapRegionPlan{
            .klass = region.klass,
            .first_slot = first_slot,
            .slots = region.slots,
            .descriptor_size = size,
            .descriptor_alignment = alignment,
            .descriptor_stride = stride,
            .byte_offset = offset,
            .fitting_slots = fitting};
        if (fitting < region.slots) { overflow = true; }
        first_slot = first_slot > std::numeric_limits<uint64_t>::max() - region.slots ?
                         std::numeric_limits<uint64_t>::max() :
                         first_slot + region.slots;
        capacity += fitting;
        // `region.slots * stride` cannot be allowed to wrap: a wrapped end offset
        // would make an oversized request look like a small legal heap.
        if (stride == 0u || region.slots > std::numeric_limits<uint64_t>::max() / stride) {
            offset = std::numeric_limits<uint64_t>::max();
            overflow = true;
            continue;
        }
        offset = offset > std::numeric_limits<uint64_t>::max() - region.slots * stride ?
                     std::numeric_limits<uint64_t>::max() :
                     offset + region.slots * stride;
    }
    plan.payload_bytes = offset;
    plan.slot_capacity = capacity;
    auto rounded = align_up(offset, plan.heap_alignment);
    if (overflow || rounded > plan.heap_size_limit) {
        plan.error = DescriptorHeapError::kHeapTooSmall;
        // Report the largest bindable size instead of the oversized request, so a
        // log line or an assertion message can state both.
        plan.heap_size = plan.payload_budget;
        plan.usable = false;
        return plan;
    }
    plan.heap_size = rounded;
    plan.error = DescriptorHeapError::kNone;
    plan.usable = true;
    return plan;
}

// Location of one slot inside a planned heap. `valid == false` means the plan is
// not usable or the class has no region with that many slots.
struct DescriptorSlot {
    bool valid{false};
    // Absolute byte offset inside the heap buffer.
    uint64_t byte_offset{0u};
    // Bytes the implementation writes for this descriptor, i.e. the minimum
    // `VkHostAddressRangeEXT::size` of its write command.
    uint64_t descriptor_size{0u};
    // Distance between neighbouring slots of the same region.
    uint64_t descriptor_stride{0u};
};

[[nodiscard]] constexpr DescriptorSlot descriptor_slot(const DescriptorHeapPlan &plan,
                                                       DescriptorClass klass,
                                                       uint64_t slot) noexcept {
    if (!plan.usable) { return DescriptorSlot{}; }
    for (uint32_t i = 0u; i < plan.region_count; i++) {
        auto &region = plan.regions[i];
        if (region.klass != klass) { continue; }
        if (slot >= region.slots) { return DescriptorSlot{}; }
        return DescriptorSlot{
            .valid = true,
            .byte_offset = region.byte_offset + slot * region.descriptor_stride,
            .descriptor_size = region.descriptor_size,
            .descriptor_stride = region.descriptor_stride};
    }
    return DescriptorSlot{};
}

// ---------------------------------------------------------------------------
// Slot allocation
// ---------------------------------------------------------------------------

// Returned by `DescriptorSlotPool::allocate()` when the pool is exhausted.
inline constexpr uint64_t kInvalidDescriptorSlot = ~uint64_t{0u};

// Upper bound on the number of disjoint free ranges one pool can represent.
// Releasing a slot that would need a new range while the table is full fails
// closed (the pool is left unchanged) instead of silently dropping a free slot.
inline constexpr uint32_t kMaxDescriptorSlotRanges = 64u;

// First-fit allocator over the slots of one planned heap region.
//
// The layout is the descriptor-heap analogue of the backend's `FirstFit`-based
// `Device::HeapAlloc` pools (`src/backends/vk/device.h`): slots are handed out
// from the lowest free range first, so a freshly reset pool behaves like a bump
// allocator, and a released slot is merged back into its neighbours so the table
// cannot grow without bound. It is a plain value type with no allocation, which
// makes it usable both as a member of `DescriptorHeap` and as a directly
// testable unit.
class DescriptorSlotPool {
public:
    constexpr DescriptorSlotPool() noexcept = default;
    constexpr DescriptorSlotPool(uint64_t first_slot, uint64_t slot_count) noexcept {
        reset(first_slot, slot_count);
    }
    // Re-initializes the pool so that every slot is free.
    constexpr void reset(uint64_t first_slot, uint64_t slot_count) noexcept {
        _first_slot = first_slot;
        _slot_count = slot_count;
        _allocated = 0u;
        _free_count = 0u;
        if (slot_count != 0u) {
            _free[0] = FreeRange{.begin = first_slot, .end = first_slot + slot_count};
            _free_count = 1u;
        }
    }
    // Hands out the lowest free slot, or `kInvalidDescriptorSlot`.
    [[nodiscard]] constexpr uint64_t allocate() noexcept {
        if (_free_count == 0u) { return kInvalidDescriptorSlot; }
        auto slot = _free[0u].begin;
        _free[0u].begin++;
        if (_free[0u].begin == _free[0u].end) {
            for (uint32_t i = 1u; i < _free_count; i++) { _free[i - 1u] = _free[i]; }
            _free_count--;
        }
        _allocated++;
        return slot;
    }
    // Returns a slot to the pool. False when the slot is outside the pool, already
    // free, or the free-range table cannot represent the result.
    [[nodiscard]] constexpr bool release(uint64_t slot) noexcept {
        if (!is_allocated(slot)) { return false; }
        // Free ranges stay sorted and never touch each other, so the insert
        // position is the first range that starts above the released slot.
        uint32_t position = 0u;
        while (position < _free_count && _free[position].begin < slot) { position++; }
        auto const merges_left = position != 0u && _free[position - 1u].end == slot;
        auto const merges_right = position < _free_count && _free[position].begin == slot + 1u;
        if (merges_left) {
            _free[position - 1u].end = merges_right ? _free[position].end : slot + 1u;
            if (merges_right) {
                for (uint32_t i = position + 1u; i < _free_count; i++) {
                    _free[i - 1u] = _free[i];
                }
                _free_count--;
            }
        } else if (merges_right) {
            _free[position].begin = slot;
        } else {
            if (_free_count == kMaxDescriptorSlotRanges) { return false; }
            for (uint32_t i = _free_count; i > position; i--) {
                _free[i] = _free[i - 1u];
            }
            _free[position] = FreeRange{.begin = slot, .end = slot + 1u};
            _free_count++;
        }
        _allocated--;
        return true;
    }
    // True when the slot belongs to this pool and is currently allocated.
    [[nodiscard]] constexpr bool is_allocated(uint64_t slot) const noexcept {
        if (slot < _first_slot || slot >= _first_slot + _slot_count) { return false; }
        for (uint32_t i = 0u; i < _free_count; i++) {
            if (slot >= _free[i].begin && slot < _free[i].end) { return false; }
        }
        return true;
    }
    [[nodiscard]] constexpr bool is_free(uint64_t slot) const noexcept {
        if (slot < _first_slot || slot >= _first_slot + _slot_count) { return false; }
        for (uint32_t i = 0u; i < _free_count; i++) {
            if (slot >= _free[i].begin && slot < _free[i].end) { return true; }
        }
        return false;
    }
    [[nodiscard]] constexpr uint64_t first_slot() const noexcept { return _first_slot; }
    [[nodiscard]] constexpr uint64_t capacity() const noexcept { return _slot_count; }
    [[nodiscard]] constexpr uint64_t allocated() const noexcept { return _allocated; }
    [[nodiscard]] constexpr uint32_t free_range_count() const noexcept { return _free_count; }

private:
    struct FreeRange {
        uint64_t begin{0u};
        // One past the last free slot of the range.
        uint64_t end{0u};
    };
    uint64_t _first_slot{0u};
    uint64_t _slot_count{0u};
    uint64_t _allocated{0u};
    uint32_t _free_count{0u};
    FreeRange _free[kMaxDescriptorSlotRanges]{};
};

// ---------------------------------------------------------------------------
// vkCmdPushDataEXT
// ---------------------------------------------------------------------------

struct PushDataPlan {
    DescriptorHeapError error{DescriptorHeapError::kExtensionDisabled};
    bool valid{false};
    // Byte offset inside the push-data block.
    uint32_t offset{0u};
    uint64_t size{0u};
};

// Push data replaces `vkCmdPushConstants` when heaps are in use (push constants
// rely on descriptor set layout state). VkPushDataInfoEXT-offset-11243/11418/11419
// require a non-empty range whose offset and size are multiples of 4 and whose end
// stays within `maxPushDataSize`.
[[nodiscard]] constexpr PushDataPlan plan_push_data(bool extension_enabled,
                                                     uint64_t max_push_data_size,
                                                     uint64_t offset,
                                                     uint64_t size) noexcept {
    PushDataPlan plan{};
    if (!extension_enabled) {
        plan.error = DescriptorHeapError::kExtensionDisabled;
        return plan;
    }
    if (size == 0u || max_push_data_size == 0u) {
        plan.error = DescriptorHeapError::kZeroPushData;
        return plan;
    }
    if (offset % 4u != 0u || size % 4u != 0u) {
        plan.error = DescriptorHeapError::kPushDataMisaligned;
        return plan;
    }
    if (offset > max_push_data_size || size > max_push_data_size - offset) {
        plan.error = DescriptorHeapError::kPushDataOutOfRange;
        return plan;
    }
    plan.error = DescriptorHeapError::kNone;
    plan.valid = true;
    plan.offset = static_cast<uint32_t>(offset);
    plan.size = size;
    return plan;
}

// ---------------------------------------------------------------------------
// Bindless-array migration
// ---------------------------------------------------------------------------

// What `src/backends/vk/bindless_array.{h,cpp}` would need in order to run on
// descriptor heaps instead of three update-after-bind descriptor sets.
//
// The current layout is:
//   * set 0 binding 0: `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` heap,
//     `Device::bindless_heap_capacity()` slots, indexed dynamically
//     (`BindlessArray::_typed_descriptor_indices` is a free list over it),
//   * set 0 binding 0 (arrays 2/3): `VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE` heaps for
//     2D and 3D textures, allocated from the same `FirstFit` pools,
//   * set 1 binding 0: `sampler_heap_size` (16) immutable samplers indexed from
//     the per-slot record's upper 4 bits.
//
// On heaps those three heaps become one BUFFER region and up to two IMAGE regions
// of the resource heap, and the immutable sampler set becomes either a SAMPLER
// region or embedded samplers. The shader-visible slot indices do not change -
// a `VkDescriptorSetAndBindingMappingEXT` with
// `VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT` maps them with
// `heapOffset = region.byte_offset` and `heapArrayStride = region.descriptor_stride`.
struct BindlessDescriptorHeapRequest {
    bool extension_enabled{false};
    // `BindlessArray::_slot_count` (bounded by `Device::bindless_heap_capacity()`).
    uint64_t bindless_capacity{0u};
    // `detail::sampler_heap_size`, i.e. the immutable sampler set's slot count.
    uint64_t sampler_count{0u};
    bool buffer_array{false};
    bool tex2d_array{false};
    bool tex3d_array{false};
    // True when the ray-tracing dispatch path also stores acceleration structures
    // in a descriptor heap. They are buffer descriptors, so they share the buffer
    // region's stride and only need their own region when the caller reserves
    // different capacities.
    bool acceleration_structures{false};
    // True when the samplers are embedded through `pEmbeddedSampler` instead of
    // being stored in the sampler heap's SAMPLER region.
    bool embed_samplers{false};
    uint64_t resource_heap_size_limit{0u};
    uint64_t sampler_heap_size_limit{0u};
};

struct BindlessArrayRegion {
    bool present{false};
    DescriptorClass klass{DescriptorClass::kBuffer};
    uint64_t slots{0u};
    uint64_t first_slot{0u};
    uint64_t byte_offset{0u};
    uint64_t descriptor_size{0u};
    uint64_t descriptor_stride{0u};
};

// The design artifact for the follow-up wiring: the two heaps the bindless array
// would own, the mapping offset/stride of every array, and the shader-side
// requirements that live in files outside this work package.
struct BindlessDescriptorHeapMigrationPlan {
    bool possible{false};
    DescriptorHeapError error{DescriptorHeapError::kExtensionDisabled};
    DescriptorHeapPlan resource_heap{};
    DescriptorHeapPlan sampler_heap{};
    BindlessArrayRegion buffer_array{};
    BindlessArrayRegion tex2d_array{};
    BindlessArrayRegion tex3d_array{};
    BindlessArrayRegion acceleration_structure_array{};
    BindlessArrayRegion sampler_array{};
    // `VkShaderDescriptorSetAndBindingMappingInfoEXT` must be chained on every
    // `VkPipelineShaderStageCreateInfo` (and `VkShaderCreateInfoEXT`) that uses a
    // heap; otherwise "Mappings must be declared for all variables with a
    // DescriptorSet and Binding in the shader resource interface".
    bool requires_shader_binding_mapping{true};
    // `VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT` is a 64-bit pipeline flag, so
    // `VkPipelineCreateFlags2CreateInfoKHR` must be chained as well.
    bool requires_pipeline_flags2{true};
    // `vkCmdBindResourceHeapEXT` / `vkCmdBindSamplerHeapEXT` replace
    // `vkCmdBindDescriptorSets` for the heap state.
    bool requires_heap_bind_commands{true};
    // Push-data bytes the mapping needs. Zero while every array uses a constant
    // offset; a `HEAP_WITH_PUSH_INDEX` mapping would need `maxPushDataSize` bytes
    // for the array-base indices.
    uint32_t required_push_data_bytes{0u};
};

// Exact places the follow-up wiring has to touch. Kept next to the planner so the
// design artifact cannot drift away from the code it describes.
inline constexpr const char *bindless_descriptor_heap_blockers[] = {
    "src/backends/vk/bindless_array.cpp:600-627 writes buffer descriptors with "
    "vkUpdateDescriptorSets into Device::bdls_buffer_set(); on heaps every write "
    "becomes vkWriteResourceDescriptorsEXT into the buffer region",
    "src/backends/vk/bindless_array.cpp:665/721 writes 2D/3D image descriptors with "
    "vkUpdateDescriptorSets into Device::bdls_tex2d_set()/bdls_tex3d_set(); on heaps "
    "they become vkWriteResourceDescriptorsEXT into the image regions",
    "src/backends/vk/bindless_array.cpp:1019-1038 and _defer_descriptor_recycling "
    "return descriptor indices to Device::{buffer,tex2d,tex3d}_heap_pool; a heap "
    "first-fit allocator over region slots replaces those pools",
    "src/backends/vk/stream.cpp:3627/3716 records vkCmdBindDescriptorSets for the "
    "bindless and sampler sets; heaps require vkCmdBindResourceHeapEXT / "
    "vkCmdBindSamplerHeapEXT instead",
    "src/backends/vk/compute_shader.cpp:87-100 builds the "
    "VkPipelineShaderStageCreateInfo without a "
    "VkShaderDescriptorSetAndBindingMappingInfoEXT and uses 32-bit "
    "VkComputePipelineCreateInfo::flags, which cannot carry "
    "VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT (0x1000000000)",
    "src/backends/vk/device.cpp:2698-2870 creates the update-after-bind descriptor "
    "pools/sets and the immutable sampler set that a heap implementation has to "
    "replace with heap buffers",
};

inline constexpr uint32_t bindless_descriptor_heap_blocker_count =
    static_cast<uint32_t>(std::size(bindless_descriptor_heap_blockers));

[[nodiscard]] constexpr BindlessArrayRegion bindless_array_region(
    const DescriptorHeapPlan &heap, uint32_t region_index) noexcept {
    if (region_index >= heap.region_count) { return BindlessArrayRegion{}; }
    auto &region = heap.regions[region_index];
    return BindlessArrayRegion{
        .present = true,
        .klass = region.klass,
        .slots = region.slots,
        .first_slot = region.first_slot,
        .byte_offset = region.byte_offset,
        .descriptor_size = region.descriptor_size,
        .descriptor_stride = region.descriptor_stride};
}

inline constexpr uint32_t bindless_region_absent = ~uint32_t{0u};

[[nodiscard]] constexpr BindlessDescriptorHeapMigrationPlan
plan_bindless_descriptor_heap_migration(const DescriptorHeapLimits &limits,
                                        const BindlessDescriptorHeapRequest &request) noexcept {
    BindlessDescriptorHeapMigrationPlan plan{};
    // Resource heap: the buffer array, then 2D and 3D image arrays in the order
    // the shader declares them. The region indices are remembered so each array's
    // mapping offset/stride can be reported without re-deriving them.
    DescriptorHeapRequest resource_request{};
    resource_request.extension_enabled = request.extension_enabled;
    resource_request.sampler_heap = false;
    resource_request.heap_size_limit = request.resource_heap_size_limit;
    uint32_t buffer_index = bindless_region_absent;
    uint32_t accel_index = bindless_region_absent;
    uint32_t tex2d_index = bindless_region_absent;
    uint32_t tex3d_index = bindless_region_absent;
    if (request.buffer_array) {
        buffer_index = resource_request.region_count;
        resource_request.regions[resource_request.region_count++] =
            DescriptorHeapRegionRequest{.klass = DescriptorClass::kBuffer,
                                        .slots = request.bindless_capacity};
    }
    if (request.acceleration_structures) {
        accel_index = resource_request.region_count;
        resource_request.regions[resource_request.region_count++] =
            DescriptorHeapRegionRequest{.klass = DescriptorClass::kBuffer,
                                        .slots = request.bindless_capacity};
    }
    if (request.tex2d_array) {
        tex2d_index = resource_request.region_count;
        resource_request.regions[resource_request.region_count++] =
            DescriptorHeapRegionRequest{.klass = DescriptorClass::kImage,
                                        .slots = request.bindless_capacity};
    }
    if (request.tex3d_array) {
        tex3d_index = resource_request.region_count;
        resource_request.regions[resource_request.region_count++] =
            DescriptorHeapRegionRequest{.klass = DescriptorClass::kImage,
                                        .slots = request.bindless_capacity};
    }
    if (resource_request.region_count == 0u) {
        // An array with no storage is not a migration candidate.
        plan.error = DescriptorHeapError::kZeroSlots;
        return plan;
    }
    plan.resource_heap = plan_descriptor_heap(limits, resource_request);
    if (!plan.resource_heap.usable) {
        plan.error = plan.resource_heap.error;
        return plan;
    }
    // Sampler heap: the immutable sampler set becomes a SAMPLER region, unless the
    // samplers are embedded (which needs no region but a bigger reservation).
    DescriptorHeapRequest sampler_request{};
    sampler_request.extension_enabled = request.extension_enabled;
    sampler_request.sampler_heap = true;
    sampler_request.heap_size_limit = request.sampler_heap_size_limit;
    if (request.embed_samplers) {
        sampler_request.embedded_sampler_count =
            static_cast<uint32_t>(request.sampler_count);
    } else if (request.sampler_count != 0u) {
        sampler_request.regions[sampler_request.region_count++] =
            DescriptorHeapRegionRequest{.klass = DescriptorClass::kSampler,
                                        .slots = request.sampler_count};
    }
    if (sampler_request.region_count != 0u || request.embed_samplers) {
        plan.sampler_heap = plan_descriptor_heap(limits, sampler_request);
        if (!plan.sampler_heap.usable) {
            plan.error = plan.sampler_heap.error;
            return plan;
        }
    }
    plan.buffer_array = bindless_array_region(plan.resource_heap, buffer_index);
    plan.acceleration_structure_array = bindless_array_region(plan.resource_heap, accel_index);
    plan.tex2d_array = bindless_array_region(plan.resource_heap, tex2d_index);
    plan.tex3d_array = bindless_array_region(plan.resource_heap, tex3d_index);
    plan.sampler_array = bindless_array_region(plan.sampler_heap, 0u);
    plan.error = DescriptorHeapError::kNone;
    plan.possible = true;
    return plan;
}

}// namespace lc::vk::detail

namespace lc::vk {

// ===========================================================================
// VK_EXT_descriptor_heap: opt-in heap owner
// ===========================================================================
//
// This class is a self-contained *foundation*, deliberately not wired into the
// default bindless path: switching `bindless_array.cpp` onto heaps also needs the
// shader-side `VkShaderDescriptorSetAndBindingMappingInfoEXT`, which lives in
// files this work package does not own (see
// `detail::bindless_descriptor_heap_blockers`). Until that wiring exists the class
// is only reachable from code that explicitly asks for a descriptor heap.
//
// Layout it owns:
//   * one device-visible resource heap buffer and one sampler heap buffer, both
//     created with VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT |
//     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
//   * one host-visible staging buffer per heap that `vkWriteResourceDescriptorsEXT` /
//     `vkWriteSamplerDescriptorsEXT` write into; `record_upload` copies the dirty
//     interval of each heap into it and makes it shader-readable,
//   * a `detail::DescriptorSlotPool` (first-fit, bump-ordered, bounded free-range
//     table) per planned region for handing out and recycling descriptor slots,
//   * `record_bind_*_heap` / `record_push_data` for the three heap state commands.
//
// Every entry point of the class fails closed: the constructor refuses a device
// whose `OptionalExtension::DESCRIPTOR_HEAP` is not enabled, and each recording
// helper asserts that the corresponding volk entry point was really loaded, so a
// disabled extension can never leak a heap struct into a command buffer.
class DescriptorHeap {
public:
    static constexpr uint32_t max_regions = detail::kMaxDescriptorHeapRegions;

    struct SlotRange {
        detail::DescriptorClass klass{detail::DescriptorClass::kBuffer};
        uint64_t slots{0u};
    };

    struct Config {
        // At least one resource region is required; a heap with no region at all
        // cannot hold a descriptor and is rejected by the planner.
        uint32_t resource_region_count{0u};
        SlotRange resource_regions[max_regions]{};
        // The sampler heap is optional (pass no region and no embedded samplers to
        // own a resource heap only).
        uint32_t sampler_region_count{0u};
        SlotRange sampler_regions[max_regions]{};
        // Samplers embedded through `pEmbeddedSampler` (sampler heap only). Selects
        // `minSamplerHeapReservedRangeWithEmbedded`.
        uint32_t embedded_sampler_count{0u};
        // 0 uses the device maximum for each heap.
        uint64_t resource_heap_size_limit{0u};
        uint64_t sampler_heap_size_limit{0u};
    };

    // `device` must outlive the heap. Fails closed with LUISA_ERROR unless
    // `OptionalExtension::DESCRIPTOR_HEAP` is enabled on it.
    DescriptorHeap(Device *device, const Config &config) noexcept;
    ~DescriptorHeap() noexcept;
    DescriptorHeap(const DescriptorHeap &) = delete;
    DescriptorHeap(DescriptorHeap &&) = delete;
    DescriptorHeap &operator=(const DescriptorHeap &) = delete;
    DescriptorHeap &operator=(DescriptorHeap &&) = delete;

    // The only gate a caller may use: true when the optional extension is enabled
    // *and* the device exposes buffer device addresses (a heap range is a device
    // address range).
    [[nodiscard]] static bool is_available(const Device *device) noexcept;

    // Queries `VkPhysicalDeviceDescriptorHeapPropertiesEXT` (and the ARM tensor
    // descriptor properties when VK_ARM_tensors is enabled). Only valid when
    // `is_available()`; returns a zeroed value otherwise.
    [[nodiscard]] static detail::DescriptorHeapLimits query_limits(Device *device) noexcept;

    // `vkGetPhysicalDeviceDescriptorSizeEXT` for one descriptor type, or 0 when the
    // extension is not enabled. The value is the exact number of bytes the
    // implementation writes, i.e. the minimum `VkHostAddressRangeEXT::size`; it is
    // never larger than the corresponding `*DescriptorSize` limit.
    [[nodiscard]] static VkDeviceSize descriptor_size_for_type(Device *device,
                                                               VkDescriptorType type) noexcept;

    [[nodiscard]] const detail::DescriptorHeapPlan &resource_heap_plan() const noexcept { return _resource_plan; }
    [[nodiscard]] const detail::DescriptorHeapPlan &sampler_heap_plan() const noexcept { return _sampler_plan; }

    [[nodiscard]] VkDeviceAddress resource_heap_address() const noexcept { return _resource.address; }
    [[nodiscard]] VkDeviceAddress sampler_heap_address() const noexcept { return _sampler.address; }
    [[nodiscard]] VkDeviceSize resource_heap_size() const noexcept { return _resource.size; }
    [[nodiscard]] VkDeviceSize sampler_heap_size() const noexcept { return _sampler.size; }
    [[nodiscard]] bool owns_sampler_heap() const noexcept { return _sampler.size != 0u; }

    // First-fit slot allocation over the regions of this heap. The returned index is
    // the plan-wide slot index (`plan.regions[i].first_slot + n`) and is exactly what
    // the writers below take; `kInvalidDescriptorSlot` means the pool is exhausted.
    // Allocation does not touch descriptor memory, only the free-range table.
    [[nodiscard]] uint64_t allocate_slot(detail::DescriptorClass klass) noexcept;
    // Returns a slot to its pool. False when the class has no region, the slot does
    // not belong to it, or it is already free.
    [[nodiscard]] bool release_slot(detail::DescriptorClass klass, uint64_t slot) noexcept;
    [[nodiscard]] uint64_t allocated_slots(detail::DescriptorClass klass) const noexcept;
    [[nodiscard]] uint64_t slot_capacity(detail::DescriptorClass klass) const noexcept;
    // Byte offset of a slot for the callers that own a raw device address range.
    [[nodiscard]] detail::DescriptorSlot slot_location(detail::DescriptorClass klass,
                                                       uint64_t slot) const noexcept;

    // Host-side descriptor writes. The slot index is the heap-wide index the plan
    // assigned to the descriptor's class (`first_slot + i`). Writes go to the
    // staging copy; `record_upload` moves them to device memory. Staged bytes must
    // not be overwritten before the command buffer that uploads them has completed.
    // `uniform_buffer` selects VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER (and therefore
    // minUniformBufferOffsetAlignment) instead of a storage buffer.
    void write_buffer_descriptor(uint64_t slot, VkDeviceAddress address, VkDeviceSize size,
                                 bool uniform_buffer) noexcept;
    void write_texel_buffer_descriptor(uint64_t slot, VkFormat format,
                                       VkDeviceAddress address, VkDeviceSize size,
                                       bool storage) noexcept;
    void write_image_descriptor(uint64_t slot, const VkImageViewCreateInfo &view,
                                VkImageLayout layout, bool storage) noexcept;
    void write_acceleration_structure_descriptor(uint64_t slot, VkDeviceAddress address) noexcept;
    // Requires `OptionalExtension::TENSOR_ARM`; fails closed otherwise.
    void write_tensor_descriptor(uint64_t slot, const VkTensorViewCreateInfoARM &view) noexcept;
    void write_sampler_descriptor(uint64_t slot, const VkSamplerCreateInfo &info) noexcept;

    // Records `vkCmdCopyBuffer` for the dirty interval of every heap and the
    // transfer-write -> shader-read barrier that makes the descriptors visible.
    // Returns false when nothing was staged since the last upload.
    bool record_upload(VkCommandBuffer command_buffer) noexcept;

    // Records `vkCmdBindResourceHeapEXT` / `vkCmdBindSamplerHeapEXT`.
    void record_bind_resource_heap(VkCommandBuffer command_buffer) const noexcept;
    void record_bind_sampler_heap(VkCommandBuffer command_buffer) const noexcept;

    // Records `vkCmdPushDataEXT`. Fails closed when `detail::plan_push_data`
    // rejects the range.
    void record_push_data(VkCommandBuffer command_buffer, uint32_t offset,
                          const void *data, VkDeviceSize size) const noexcept;

private:
    // A VkBuffer with its own VkDeviceMemory, as created by
    // `VkAllocator::allocate_native_buffer`.
    struct Storage {
        VkBuffer buffer{};
        VkDeviceMemory memory{};
        VkDeviceSize size{0u};
        VkDeviceAddress address{0u};
        void *mapped{nullptr};
    };
    // Dirty interval of one heap, in bytes, relative to the staging buffer.
    struct PendingRange {
        VkDeviceSize begin{0u};
        VkDeviceSize end{0u};
        void mark(VkDeviceSize offset, VkDeviceSize size) noexcept {
            if (size == 0u) { return; }
            if (begin == end) {
                begin = offset;
                end = offset + size;
            } else {
                if (offset < begin) { begin = offset; }
                if (offset + size > end) { end = offset + size; }
            }
        }
        [[nodiscard]] bool empty() const noexcept { return begin == end; }
        void clear() noexcept {
            begin = 0u;
            end = 0u;
        }
    };

private:
    Device *_device{nullptr};
    detail::DescriptorHeapLimits _limits{};
    detail::DescriptorHeapPlan _resource_plan{};
    detail::DescriptorHeapPlan _sampler_plan{};
    // Device-visible heap buffers (device local, TRANSFER_DST).
    Storage _resource{};
    Storage _sampler{};
    // Host-visible staging copies, one per heap: a descriptor's byte offset inside
    // its heap is also its offset inside that heap's staging buffer, and separate
    // buffers keep resource and sampler staging bytes from aliasing each other.
    Storage _resource_staging{};
    Storage _sampler_staging{};
    PendingRange _resource_pending{};
    PendingRange _sampler_pending{};
    // One first-fit pool per planned region, indexed like `plan.regions`.
    detail::DescriptorSlotPool _resource_pools[max_regions]{};
    detail::DescriptorSlotPool _sampler_pools[max_regions]{};

private:
    // Resolves the pool of the first region of `klass`, or nullptr.
    [[nodiscard]] detail::DescriptorSlotPool *_pool(const detail::DescriptorHeapPlan &plan,
                                                    detail::DescriptorSlotPool *pools,
                                                    detail::DescriptorClass klass) noexcept;
    [[nodiscard]] const detail::DescriptorSlotPool *_pool(const detail::DescriptorHeapPlan &plan,
                                                          const detail::DescriptorSlotPool *pools,
                                                          detail::DescriptorClass klass) const noexcept;
    // Copy the dirty interval into the device heap and publish it to shaders.
    void _record_copy(VkCommandBuffer command_buffer, const Storage &staging,
                      const Storage &destination, const PendingRange &range) const noexcept;
    static void _record_barrier(VkCommandBuffer command_buffer, VkBuffer buffer,
                                VkDeviceSize offset, VkDeviceSize size) noexcept;
    // Shared prologue of every writer: resolves the slot and returns a host address
    // range inside the staging buffer, or fails closed.
    [[nodiscard]] VkHostAddressRangeEXT _slot_range(const detail::DescriptorHeapPlan &plan,
                                                      detail::DescriptorClass klass,
                                                      uint64_t slot, const char *what) noexcept;
    void _write_resource_descriptor(const detail::DescriptorHeapPlan &plan,
                                    detail::DescriptorClass klass, uint64_t slot,
                                    VkResourceDescriptorInfoEXT &info,
                                    const char *what) noexcept;
    // `vkGetPhysicalDeviceDescriptorSizeEXT` for the descriptor types that are
    // legal on this device, or 0 when the query must be skipped (a type whose
    // extension is not enabled).
    [[nodiscard]] VkDeviceSize _exact_descriptor_size(VkDescriptorType type) const noexcept;
};

}// namespace lc::vk
