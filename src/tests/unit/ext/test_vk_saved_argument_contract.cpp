#include "ut/ut.hpp"

#include "saved_argument_contract.h"

#include <array>

using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] lc::vk::SavedArgument saved(
    luisa::compute::Type::Tag tag,
    uint32_t size = 0u,
    uint32_t metadata = lc::vk::SavedArgument::invalid_buffer_metadata_index) {
    lc::vk::SavedArgument argument;
    argument.tag = tag;
    argument.var_usage = luisa::compute::Usage::READ;
    argument.struct_size = size;
    argument.set_buffer_metadata_index(metadata);
    return argument;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "vk_saved_argument_contract_accepts_typed_and_byte_buffers"_test = [] {
        using namespace lc::vk;
        using luisa::compute::Type;
        std::array arguments{
            saved(Type::Tag::BUFFER, 16u, 0u),
            saved(Type::Tag::BUFFER, 0u, 1u),
            saved(Type::Tag::BINDLESS_ARRAY)};
        auto native = plan_saved_argument_contract(arguments, 0u);
        expect(static_cast<bool>(native));
        expect(eq(native.metadata_count, 2u));
        expect(eq(native.validation_resource_count, 3u));

        arguments[0].set_buffer_metadata_index(
            SavedArgument::invalid_buffer_metadata_index);
        arguments[1].set_buffer_metadata_index(
            SavedArgument::invalid_buffer_metadata_index);
        auto hlsl = plan_saved_argument_contract(arguments, 3u);
        expect(static_cast<bool>(hlsl));
        expect(eq(hlsl.metadata_count, 0u));
        expect(eq(hlsl.validation_resource_count, 3u));
    };

    "vk_saved_argument_contract_rejects_noncanonical_metadata"_test = [] {
        using namespace lc::vk;
        using luisa::compute::Type;
        std::array duplicate{
            saved(Type::Tag::BUFFER, 4u, 0u),
            saved(Type::Tag::BUFFER, 4u, 0u)};
        expect(plan_saved_argument_contract(duplicate, 0u).status ==
               SavedArgumentContractStatus::INVALID_METADATA);

        std::array gap{
            saved(Type::Tag::BUFFER, 4u, 1u),
            saved(Type::Tag::BUFFER, 4u)};
        expect(plan_saved_argument_contract(gap, 0u).status ==
               SavedArgumentContractStatus::NON_DENSE_METADATA);

        std::array wrong_kind{
            saved(Type::Tag::TEXTURE, 0u, 0u)};
        expect(plan_saved_argument_contract(wrong_kind, 0u).status ==
               SavedArgumentContractStatus::INVALID_METADATA);
    };

    "vk_saved_argument_contract_matches_hlsl_validation_resources"_test = [] {
        using namespace lc::vk;
        using luisa::compute::Type;
        std::array arguments{
            saved(Type::Tag::BUFFER, 4u),
            saved(Type::Tag::BINDLESS_ARRAY),
            saved(Type::Tag::CUSTOM)};
        auto exact = plan_saved_argument_contract(arguments, 2u);
        expect(static_cast<bool>(exact));
        expect(eq(exact.validation_resource_count, 2u))
            << "custom/indirect arguments have no HLSL validation field";
        expect(plan_saved_argument_contract(arguments, 3u).status ==
               SavedArgumentContractStatus::VALIDATION_COUNT_MISMATCH);

        arguments[0].set_buffer_metadata_index(0u);
        expect(plan_saved_argument_contract(arguments, 2u).status ==
               SavedArgumentContractStatus::INCOMPATIBLE_TRAILERS);
    };

    "vk_saved_argument_contract_rejects_resource_payload_bytes"_test = [] {
        using namespace lc::vk;
        using luisa::compute::Type;
        std::array argument{saved(Type::Tag::ACCEL, 4u)};
        expect(plan_saved_argument_contract(argument, 0u).status ==
               SavedArgumentContractStatus::INVALID_RESOURCE_SIZE);
        argument[0].tag = static_cast<Type::Tag>(0xffffffffu);
        argument[0].struct_size = 0u;
        expect(plan_saved_argument_contract(argument, 0u).status ==
               SavedArgumentContractStatus::INVALID_TAG);
    };

    "vk_saved_argument_contract_validates_native_accel_role_bits"_test = [] {
        using namespace lc::vk;
        using luisa::compute::Type;
        std::array<lc::vk::SavedArgument, 1u> argument{saved(Type::Tag::ACCEL)};
        argument[0].set_native_accel_roles(
            SavedArgument::native_accel_role_traversal |
            SavedArgument::native_accel_role_instance);
        expect(static_cast<bool>(
            plan_saved_argument_contract(argument, 0u)));

        argument[0].set_native_accel_roles(
            SavedArgument::native_accel_role_known_mask | (1u << 31u));
        expect(plan_saved_argument_contract(argument, 0u).status ==
               SavedArgumentContractStatus::INVALID_RESOURCE_ROLES);
    };

    "vk_saved_argument_contract_preserves_device_address_roles"_test = [] {
        using namespace lc::vk;
        using luisa::compute::Type;
        std::array arguments{
            saved(Type::Tag::BUFFER, 4u, 0u),
            saved(Type::Tag::BINDLESS_ARRAY)};
        arguments[0].set_native_buffer_roles(
            SavedArgument::native_buffer_device_address);
        arguments[1].set_native_bindless_roles(
            SavedArgument::native_buffer_device_address);

        expect(arguments[0].has_buffer_metadata());
        expect(eq(arguments[0].buffer_metadata_index(), 0u));
        expect(arguments[0].native_buffer_uses_device_address());
        arguments[0].set_native_buffer_roles(
            lc::spirv::kernel_argument_role::none);
        expect(arguments[0].has_buffer_metadata());
        expect(eq(arguments[0].buffer_metadata_index(), 0u));
        expect(!arguments[0].native_buffer_uses_device_address());
        arguments[0].set_native_buffer_roles(
            SavedArgument::native_buffer_device_address);
        expect(arguments[1].has_explicit_native_bindless_roles());
        expect(arguments[1].native_bindless_uses_device_address());
        expect(static_cast<bool>(
            plan_saved_argument_contract(arguments, 0u)));

        arguments[1].set_native_bindless_roles(1u << 31u);
        expect(plan_saved_argument_contract(arguments, 0u).status ==
               SavedArgumentContractStatus::INVALID_RESOURCE_ROLES);
    };
}
