#pragma once

#include <cstddef>
#include <cstdint>

namespace luisa::compute {
class Type;
}// namespace luisa::compute

namespace lc::spirv {

enum class SpirvTypedBufferLayoutStatus : uint8_t {
    COMPATIBLE,
    UNSUPPORTED_TYPE,
    LOGICAL_BOOL,
    INVALID_ARRAY_STRIDE,
    INVALID_MATRIX_STRIDE,
    MISALIGNED_STRUCT_MEMBER,
    INVALID_STRUCT_STRIDE,
    INVALID_RUNTIME_ARRAY_STRIDE,
};

struct SpirvTypedBufferLayout {
    SpirvTypedBufferLayoutStatus status{
        SpirvTypedBufferLayoutStatus::UNSUPPORTED_TYPE};
    size_t base_alignment{0u};
    size_t byte_offset{0u};
    const luisa::compute::Type *offending_type{nullptr};

    [[nodiscard]] bool compatible() const noexcept {
        return status == SpirvTypedBufferLayoutStatus::COMPATIBLE;
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return compatible();
    }
};

// Plans the standard Vulkan StorageBuffer layout for one runtime-array
// element. Luisa host types cap vector alignment at 16 bytes, whereas a
// three- or four-component 64-bit vector has a 32-byte Vulkan base alignment.
// Aggregates that place such a value at a merely 16-byte-aligned offset must
// use the uint32 word-storage ABI instead of a typed SSBO declaration.
[[nodiscard]] SpirvTypedBufferLayout
plan_spirv_typed_buffer_layout(
    const luisa::compute::Type *element_type) noexcept;

[[nodiscard]] inline bool spirv_typed_buffer_layout_compatible(
    const luisa::compute::Type *element_type) noexcept {
    return plan_spirv_typed_buffer_layout(element_type).compatible();
}

}// namespace lc::spirv
