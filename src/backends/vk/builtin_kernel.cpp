#include <cstring>

#include "builtin_kernel.h"
#include "../common/indirect_dispatch_layout.h"
#include "device.h"
#include "vulkan_builtin_contract.h"
#include "vulkan_builtin_spirv_embedded.h"

namespace lc::vk {
namespace {

[[nodiscard]] vstd::vector<uint> load_embedded_spirv(
    luisa::string_view name,
    const unsigned char *data,
    size_t size) noexcept {
    LUISA_ASSERT(
        data != nullptr && size >= 5u * sizeof(uint) &&
            size % sizeof(uint) == 0u,
        "Invalid embedded SPIR-V for {}: {} bytes.",
        name,
        size);
    vstd::vector<uint> spirv;
    spirv.resize(size / sizeof(uint));
    std::memcpy(spirv.data(), data, size);
    LUISA_ASSERT(
        spirv.front() == 0x07230203u,
        "Embedded Vulkan built-in '{}' has invalid SPIR-V magic 0x{:08x}.",
        name,
        spirv.front());
    return spirv;
}

[[nodiscard]] vstd::vector<hlsl::Property>
builtin_buffer_properties() noexcept {
    vstd::vector<hlsl::Property> properties;
    properties.reserve(detail::vulkan_builtin_buffer_properties.size());
    for (auto property : detail::vulkan_builtin_buffer_properties) {
        properties.emplace_back(property);
    }
    return properties;
}

[[nodiscard]] ComputeShader *load_embedded_spirv(
    Device *device,
    luisa::string_view name,
    const unsigned char *data,
    size_t size,
    uint3 block_size,
    uint32_t push_constant_size) noexcept {
    auto spirv = load_embedded_spirv(name, data, size);
    auto properties = builtin_buffer_properties();
    return new ComputeShader(
        device,
        block_size,
        properties,
        {},
        spirv,
        {},
        luisa::span<std::byte const>{},
        false,
        false,
        false,
        {},
        luisa::span<std::byte const>{},
        0u,
        luisa::nullopt,
        push_constant_size,
        detail::ShaderCodegenDialect::VULKAN_BUILTIN);
}

}// namespace

ComputeShader *BuiltinKernel::load_indirect_prepare_kernel(Device *device) {
    constexpr auto contract = detail::vulkan_builtin_kernel_contract(
        detail::VulkanBuiltinKernel::INDIRECT_PREPARE);
    return load_embedded_spirv(
        device,
        "indirect_prepare.spv",
        luisa_compute_vk_builtin_indirect_prepare_spv,
        luisa_compute_vk_builtin_indirect_prepare_spv_size,
        uint3{contract.block_size_x, 1u, 1u},
        contract.push_constant_size);
}

ComputeShader *BuiltinKernel::load_accel_set_kernel(Device *device) {
    constexpr auto contract = detail::vulkan_builtin_kernel_contract(
        detail::VulkanBuiltinKernel::ACCEL_PROCESS);
    static_assert(sizeof(uint2) == contract.push_constant_size);
    return load_embedded_spirv(
        device,
        "accel_process.spv",
        luisa_compute_vk_builtin_accel_process_spv,
        luisa_compute_vk_builtin_accel_process_spv_size,
        uint3{contract.block_size_x, 1u, 1u},
        contract.push_constant_size);
}

ComputeShader *BuiltinKernel::load_bindless_set_kernel(Device *device) {
    constexpr auto contract = detail::vulkan_builtin_kernel_contract(
        detail::VulkanBuiltinKernel::BINDLESS_UPLOAD);
    static_assert(sizeof(uint) == contract.push_constant_size);
    return load_embedded_spirv(
        device,
        "bindless_upload.spv",
        luisa_compute_vk_builtin_bindless_upload_spv,
        luisa_compute_vk_builtin_bindless_upload_spv_size,
        uint3{contract.block_size_x, 1u, 1u},
        contract.push_constant_size);
}

}// namespace lc::vk
