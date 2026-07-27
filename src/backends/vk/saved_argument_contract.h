#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <luisa/ast/function.h>
#include <luisa/ast/type.h>
#include <luisa/ast/usage.h>
#include <luisa/core/stl/memory.h>

#include "../common/spirv/spirv_codegen/kernel_argument_role.h"

namespace lc::vk {

using namespace luisa::compute;

// Persisted description of one shader argument. A zero BUFFER struct_size is
// intentional and means byte-addressed storage (Type::buffer(nullptr)); typed
// buffers carry their nonzero logical element stride.
struct SavedArgument {
    static constexpr uint32_t invalid_buffer_metadata_index = ~0u;
    static constexpr uint32_t buffer_metadata_index_mask = 0x7fffffffu;
    static constexpr uint32_t native_buffer_role_flag = 0x80000000u;
    static constexpr uint32_t unspecified_native_resource_roles = ~0u;
    // MSVC 2022 (14.44) workaround: static constexpr initialized from
    // cross-namespace inline constexpr typedef aliases may fail.
    inline static constexpr uint32_t native_buffer_device_address =
        spirv::kernel_argument_role::buffer_device_address;
    inline static constexpr uint32_t native_accel_role_traversal =
        spirv::kernel_argument_role::accel_traversal;
    inline static constexpr uint32_t native_accel_role_instance =
        spirv::kernel_argument_role::accel_instance;
    inline static constexpr uint32_t native_accel_role_known_mask =
        spirv::kernel_argument_role::accel_known_mask;
    inline static constexpr uint32_t native_bindless_role_known_mask =
        spirv::kernel_argument_role::bindless_known_mask;
    Type::Tag tag{};
    Usage var_usage{};
    uint32_t struct_size{};
    // The argument tag is the discriminator for this stable 32-bit ABI word:
    // BUFFER stores its dense metadata index plus the native role flag;
    // native ACCEL and BINDLESS_ARRAY arguments store their exact role masks.
    // All other arguments and legacy resource artifacts keep the all-ones
    // unspecified sentinel. Keep one canonical scalar instead of a union so
    // hashing/serialization never reads an inactive member.
    uint32_t resource_aux{invalid_buffer_metadata_index};
    SavedArgument() = default;
    SavedArgument(Function kernel, Variable const &var)
        : SavedArgument(var.type()) {
        var_usage = kernel.variable_usage(var.uid());
    }
    SavedArgument(Usage usage, Variable const &var)
        : SavedArgument(var.type()) {
        var_usage = usage;
    }
    explicit SavedArgument(Type const *type);
    void set_buffer_metadata_index(uint32_t index) noexcept {
        if (index == invalid_buffer_metadata_index) {
            resource_aux = invalid_buffer_metadata_index;
        } else {
            auto role = resource_aux == invalid_buffer_metadata_index ?
                            0u :
                            resource_aux & native_buffer_role_flag;
            resource_aux = role | index;
        }
    }
    [[nodiscard]] uint32_t buffer_metadata_index() const noexcept {
        return resource_aux & buffer_metadata_index_mask;
    }
    void set_native_buffer_roles(uint32_t roles) noexcept {
        if (resource_aux == invalid_buffer_metadata_index) { return; }
        resource_aux =
            (resource_aux & buffer_metadata_index_mask) |
            ((roles & native_buffer_device_address) != 0u ?
                 native_buffer_role_flag :
                 0u);
    }
    [[nodiscard]] bool native_buffer_uses_device_address() const noexcept {
        return tag == Type::Tag::BUFFER && has_buffer_metadata() &&
               (resource_aux & native_buffer_role_flag) != 0u;
    }
    void set_native_accel_roles(uint32_t roles) noexcept {
        resource_aux = roles;
    }
    [[nodiscard]] uint32_t native_accel_roles() const noexcept {
        return resource_aux;
    }
    void set_native_bindless_roles(uint32_t roles) noexcept {
        resource_aux = roles;
    }
    [[nodiscard]] uint32_t native_bindless_roles() const noexcept {
        return resource_aux;
    }
    [[nodiscard]] bool has_buffer_metadata() const noexcept {
        return tag == Type::Tag::BUFFER &&
               resource_aux != invalid_buffer_metadata_index;
    }
    [[nodiscard]] bool has_explicit_native_accel_roles() const noexcept {
        return tag == Type::Tag::ACCEL &&
               resource_aux !=
                   unspecified_native_resource_roles;
    }
    [[nodiscard]] bool has_explicit_native_bindless_roles() const noexcept {
        return tag == Type::Tag::BINDLESS_ARRAY &&
               resource_aux != unspecified_native_resource_roles;
    }
    [[nodiscard]] bool native_bindless_uses_device_address() const noexcept {
        return has_explicit_native_bindless_roles() &&
               (resource_aux & native_buffer_device_address) != 0u;
    }
    [[nodiscard]] bool native_accel_uses_traversal() const noexcept {
        return has_explicit_native_accel_roles() &&
               (resource_aux &
                native_accel_role_traversal) != 0u;
    }
    [[nodiscard]] bool native_accel_uses_instance_buffer() const noexcept {
        return has_explicit_native_accel_roles() &&
               (resource_aux &
                native_accel_role_instance) != 0u;
    }
};
static_assert(std::is_trivially_copyable_v<SavedArgument> &&
              std::is_standard_layout_v<SavedArgument> &&
              sizeof(SavedArgument) == 4u * sizeof(uint32_t));

enum class SavedArgumentContractStatus : uint8_t {
    SUCCESS,
    INVALID_TAG,
    INVALID_USAGE,
    INVALID_RESOURCE_SIZE,
    INVALID_METADATA,
    INVALID_RESOURCE_ROLES,
    NON_DENSE_METADATA,
    INCOMPATIBLE_TRAILERS,
    VALIDATION_COUNT_MISMATCH
};

struct SavedArgumentContract {
    SavedArgumentContractStatus status{SavedArgumentContractStatus::SUCCESS};
    size_t metadata_count{};
    size_t validation_resource_count{};
    [[nodiscard]] explicit operator bool() const noexcept {
        return status == SavedArgumentContractStatus::SUCCESS;
    }
};

[[nodiscard]] constexpr const char *saved_argument_contract_status_name(
    SavedArgumentContractStatus status) noexcept {
    switch (status) {
        case SavedArgumentContractStatus::SUCCESS: return "success";
        case SavedArgumentContractStatus::INVALID_TAG: return "invalid type tag";
        case SavedArgumentContractStatus::INVALID_USAGE: return "invalid usage mask";
        case SavedArgumentContractStatus::INVALID_RESOURCE_SIZE: return "resource carries an invalid uniform size";
        case SavedArgumentContractStatus::INVALID_METADATA: return "invalid buffer-metadata assignment";
        case SavedArgumentContractStatus::INVALID_RESOURCE_ROLES: return "invalid native resource-role mask";
        case SavedArgumentContractStatus::NON_DENSE_METADATA: return "buffer-metadata slots are not dense";
        case SavedArgumentContractStatus::INCOMPATIBLE_TRAILERS: return "native buffer metadata and HLSL validation words are both enabled";
        case SavedArgumentContractStatus::VALIDATION_COUNT_MISMATCH: return "HLSL validation count does not match buffer and bindless arguments";
    }
    return "unknown";
}

// Validate the serialized argument/trailer ABI without consulting live AST
// objects. This is deliberately allocation-free so malformed cache records
// cannot turn validation into an attacker-controlled allocation.
[[nodiscard]] inline SavedArgumentContract plan_saved_argument_contract(
    luisa::span<const SavedArgument> arguments,
    uint32_t validation_count) noexcept {
    SavedArgumentContract contract{};
    for (size_t argument_index = 0u;
         argument_index < arguments.size(); ++argument_index) {
        auto argument = arguments[argument_index];
        if (luisa::to_underlying(argument.tag) >
            luisa::to_underlying(Type::Tag::CUSTOM)) {
            contract.status = SavedArgumentContractStatus::INVALID_TAG;
            return contract;
        }
        if ((luisa::to_underlying(argument.var_usage) &
             ~luisa::to_underlying(Usage::READ_WRITE)) != 0u) {
            contract.status = SavedArgumentContractStatus::INVALID_USAGE;
            return contract;
        }
        switch (argument.tag) {
            case Type::Tag::TEXTURE:
            case Type::Tag::BINDLESS_ARRAY:
            case Type::Tag::ACCEL:
            case Type::Tag::COOPERATIVE_VECTOR_REF:
            case Type::Tag::COOPERATIVE_MATRIX_REF:
            case Type::Tag::CUSTOM:
                if (argument.struct_size != 0u) {
                    contract.status =
                        SavedArgumentContractStatus::INVALID_RESOURCE_SIZE;
                    return contract;
                }
                break;
            default: break;
        }
        if (argument.tag == Type::Tag::ACCEL) {
            if (argument.has_explicit_native_accel_roles() &&
                (argument.native_accel_roles() &
                 ~SavedArgument::native_accel_role_known_mask) != 0u) {
                contract.status =
                    SavedArgumentContractStatus::INVALID_RESOURCE_ROLES;
                return contract;
            }
        } else if (argument.tag == Type::Tag::BINDLESS_ARRAY) {
            if (argument.has_explicit_native_bindless_roles() &&
                (argument.native_bindless_roles() &
                 ~SavedArgument::native_bindless_role_known_mask) != 0u) {
                contract.status =
                    SavedArgumentContractStatus::INVALID_RESOURCE_ROLES;
                return contract;
            }
        } else if (argument.tag != Type::Tag::BUFFER &&
                   argument.resource_aux !=
                       SavedArgument::unspecified_native_resource_roles) {
            contract.status = SavedArgumentContractStatus::INVALID_METADATA;
            return contract;
        }
        if (argument.tag == Type::Tag::BUFFER ||
            argument.tag == Type::Tag::BINDLESS_ARRAY) {
            contract.validation_resource_count++;
        }
        if (argument.has_buffer_metadata()) {
            if (argument.buffer_metadata_index() >= arguments.size()) {
                contract.status =
                    SavedArgumentContractStatus::INVALID_METADATA;
                return contract;
            }
            for (size_t previous = 0u;
                 previous < argument_index; ++previous) {
                if (arguments[previous].has_buffer_metadata() &&
                    arguments[previous].buffer_metadata_index() ==
                        argument.buffer_metadata_index()) {
                    contract.status =
                        SavedArgumentContractStatus::INVALID_METADATA;
                    return contract;
                }
            }
            contract.metadata_count++;
        }
    }
    // Unique metadata indices are dense exactly when each slot in
    // [0, metadata_count) appears. Scan the tiny persisted argument table
    // directly rather than allocating a bitmap.
    for (size_t slot = 0u; slot < contract.metadata_count; ++slot) {
        auto found = false;
        for (auto argument : arguments) {
            found |= argument.has_buffer_metadata() &&
                     argument.buffer_metadata_index() == slot;
        }
        if (!found) {
            contract.status =
                SavedArgumentContractStatus::NON_DENSE_METADATA;
            return contract;
        }
    }
    if (validation_count != 0u && contract.metadata_count != 0u) {
        contract.status =
            SavedArgumentContractStatus::INCOMPATIBLE_TRAILERS;
        return contract;
    }
    if (validation_count != 0u &&
        validation_count != contract.validation_resource_count) {
        contract.status =
            SavedArgumentContractStatus::VALIDATION_COUNT_MISMATCH;
    }
    return contract;
}

}// namespace lc::vk
