// Test for Metal AIR library container generation.
// This test covers:
// - Host-version to metallib/AIR/MSL target mapping
// - Deterministic multi-function container generation and validation
// - Corruption, truncation, and invalid-input rejection

#include "ut/ut.hpp"

#include "metal_metallib.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

using namespace luisa;
using namespace luisa::compute::metal;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr std::array kDirectAir{
    std::byte{0x42u}, std::byte{0x43u}, std::byte{0xc0u}, std::byte{0xdeu},
    std::byte{0x01u}, std::byte{0x02u}, std::byte{0x03u}};
constexpr std::array kIndirectAir{
    std::byte{0x42u}, std::byte{0x43u}, std::byte{0xc0u}, std::byte{0xdeu},
    std::byte{0x11u}, std::byte{0x12u}, std::byte{0x13u}, std::byte{0x14u},
    std::byte{0x15u}};
constexpr std::array<string_view, 2u> kEntryPoints{
    "kernel_main", "kernel_main_indirect"};
constexpr std::array kProgramTypes{
    MetalLibProgramType::KERNEL,
    MetalLibProgramType::KERNEL};

[[nodiscard]] auto test_functions() noexcept {
    return std::array{
        MetalLibFunction{
            .name = kEntryPoints[0u],
            .air_module = kDirectAir,
            .type = MetalLibProgramType::KERNEL},
        MetalLibFunction{
            .name = kEntryPoints[1u],
            .air_module = kIndirectAir,
            .type = MetalLibProgramType::KERNEL}};
}

[[nodiscard]] auto make_test_library() noexcept {
    auto functions = test_functions();
    return make_metallib(metallib_target_for_macos(26u), functions);
}

[[nodiscard]] uint16_t read_u16(span<const std::byte> data, size_t offset) noexcept {
    auto x = static_cast<uint16_t>(std::to_integer<uint8_t>(data[offset]));
    return static_cast<uint16_t>(x | static_cast<uint16_t>(std::to_integer<uint8_t>(data[offset + 1u])) << 8u);
}

[[nodiscard]] uint32_t read_u32(span<const std::byte> data, size_t offset) noexcept {
    auto x = 0u;
    for (auto i = 0u; i < 4u; i++) {
        x |= static_cast<uint32_t>(std::to_integer<uint8_t>(data[offset + i])) << (i * 8u);
    }
    return x;
}

[[nodiscard]] uint64_t read_u64(span<const std::byte> data, size_t offset) noexcept {
    auto x = 0ull;
    for (auto i = 0u; i < 8u; i++) {
        x |= static_cast<uint64_t>(std::to_integer<uint8_t>(data[offset + i])) << (i * 8u);
    }
    return x;
}

[[nodiscard]] bool matches_tag(span<const std::byte> data, size_t offset, string_view tag) noexcept {
    if (offset > data.size() || tag.size() > data.size() - offset) { return false; }
    for (auto i = 0u; i < tag.size(); i++) {
        if (std::to_integer<uint8_t>(data[offset + i]) != static_cast<uint8_t>(tag[i])) { return false; }
    }
    return true;
}

[[nodiscard]] size_t find_tag_value(span<const std::byte> data, string_view wanted, size_t occurrence = 0u) noexcept {
    auto function_list_offset = static_cast<size_t>(read_u64(data, 24u));
    auto function_count = read_u32(data, function_list_offset);
    auto position = function_list_offset + 4u;
    for (auto function_index = 0u; function_index < function_count; function_index++) {
        auto group_end = position + read_u32(data, position);
        position += 4u;
        while (position + 4u <= group_end && !matches_tag(data, position, "ENDT")) {
            auto value_size = read_u16(data, position + 4u);
            if (matches_tag(data, position, wanted)) {
                if (occurrence == 0u) { return position + 6u; }
                occurrence--;
            }
            position += 6u + value_size;
        }
        position = group_end;
    }
    return std::numeric_limits<size_t>::max();
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    "metal_metallib_version_mapping"_test = [] {
        auto macos_13 = metallib_target_for_macos(13u, 6u, 1u);
        expect(static_cast<bool>(macos_13.file_format == MetalLibVersion{1u, 2u, 7u}));
        expect(static_cast<bool>(macos_13.platform == MetalLibVersion{13u, 6u, 1u}));
        expect(static_cast<bool>(macos_13.air == MetalLibVersion{2u, 5u, 0u}));
        expect(static_cast<bool>(macos_13.metal == MetalLibVersion{3u, 0u, 0u}));

        auto macos_14 = metallib_target_for_macos(14u);
        expect(static_cast<bool>(macos_14.file_format == MetalLibVersion{1u, 2u, 7u}));
        expect(static_cast<bool>(macos_14.air == MetalLibVersion{2u, 6u, 0u}));
        expect(static_cast<bool>(macos_14.metal == MetalLibVersion{3u, 1u, 0u}));

        auto macos_15 = metallib_target_for_macos(15u);
        expect(static_cast<bool>(macos_15.file_format == MetalLibVersion{1u, 2u, 8u}));
        expect(static_cast<bool>(macos_15.air == MetalLibVersion{2u, 7u, 0u}));
        expect(static_cast<bool>(macos_15.metal == MetalLibVersion{3u, 2u, 0u}));

        auto compatibility_16 = metallib_target_for_macos(16u);
        auto macos_26 = metallib_target_for_macos(26u);
        expect(static_cast<bool>(compatibility_16 == macos_26));
        expect(static_cast<bool>(macos_26.file_format == MetalLibVersion{1u, 2u, 9u}));
        expect(static_cast<bool>(macos_26.air == MetalLibVersion{2u, 8u, 0u}));
        expect(static_cast<bool>(macos_26.metal == MetalLibVersion{4u, 0u, 0u}));

        auto macos_27 = metallib_target_for_macos(27u);
        expect(static_cast<bool>(macos_27.file_format == MetalLibVersion{1u, 2u, 9u}));
        expect(static_cast<bool>(macos_27.air == MetalLibVersion{2u, 9u, 0u}));
        expect(static_cast<bool>(macos_27.metal == MetalLibVersion{4u, 1u, 0u}));

        auto ios_26 = metallib_target_for_ios(26u, 4u);
        expect(ios_26.operating_system == MetalLibPlatform::IOS);
        expect(static_cast<bool>(ios_26.file_format == MetalLibVersion{1u, 2u, 9u}));
        expect(static_cast<bool>(ios_26.platform == MetalLibVersion{26u, 4u, 0u}));
        expect(static_cast<bool>(ios_26.air == MetalLibVersion{2u, 8u, 0u}));
        expect(static_cast<bool>(ios_26.metal == MetalLibVersion{4u, 0u, 0u}));
    };

    "metal_metallib_ios_header_convention"_test = [] {
        auto functions = test_functions();
        auto library = make_metallib(
            metallib_target_for_ios(26u, 4u), functions);
        expect(!library.empty());
        expect(validate_metallib(library, kEntryPoints, kProgramTypes));
        expect(eq(read_u16(library, 4u), uint16_t{1u}));
        expect(eq(std::to_integer<uint8_t>(library[11u]), uint8_t{0x82u}));
        expect(eq(read_u16(library, 12u), uint16_t{26u}));
        expect(eq(std::to_integer<uint8_t>(library[14u]), uint8_t{4u}));

        auto mismatched_header = library;
        mismatched_header[5u] = std::byte{0x80u};
        expect(!validate_metallib(mismatched_header));
        mismatched_header = library;
        mismatched_header[11u] = std::byte{0x81u};
        expect(!validate_metallib(mismatched_header));
    };

    "metal_metallib_two_distinct_functions"_test = [] {
        auto library = make_test_library();
        expect(!library.empty());
        expect(matches_tag(library, 0u, "MTLB"));
        expect(validate_metallib(library, kEntryPoints, kProgramTypes));

        auto function_list_offset = static_cast<size_t>(read_u64(library, 24u));
        auto module_list_offset = static_cast<size_t>(read_u64(library, 72u));
        expect(eq(read_u32(library, function_list_offset), 2u));
        expect(eq(read_u64(library, 80u), kDirectAir.size() + kIndirectAir.size()));
        expect(std::equal(kDirectAir.begin(), kDirectAir.end(), library.begin() + module_list_offset));
        expect(std::equal(kIndirectAir.begin(), kIndirectAir.end(), library.begin() + module_list_offset + kDirectAir.size()));

        auto first_hash = find_tag_value(library, "HASH", 0u);
        auto second_hash = find_tag_value(library, "HASH", 1u);
        expect(first_hash != std::numeric_limits<size_t>::max());
        expect(second_hash != std::numeric_limits<size_t>::max());
        expect(!std::equal(library.begin() + first_hash,
                           library.begin() + first_hash + 32u,
                           library.begin() + second_hash));
    };

    "metal_metallib_is_deterministic"_test = [] {
        auto first = make_test_library();
        auto second = make_test_library();
        expect(static_cast<bool>(first == second));
    };

    "metal_metallib_expected_entries"_test = [] {
        auto library = make_test_library();
        constexpr std::array<string_view, 2u> reversed{
            "kernel_main_indirect", "kernel_main"};
        constexpr std::array<string_view, 2u> wrong{
            "kernel_main", "not_kernel_main_indirect"};
        constexpr std::array<string_view, 1u> wrong_count{"kernel_main"};
        constexpr std::array wrong_types{
            MetalLibProgramType::VERTEX,
            MetalLibProgramType::FRAGMENT};
        constexpr std::array wrong_type_count{
            MetalLibProgramType::KERNEL};
        expect(validate_metallib(library));
        expect(validate_metallib(library, kEntryPoints));
        expect(validate_metallib(library, kEntryPoints, kProgramTypes));
        expect(!validate_metallib(library, reversed));
        expect(!validate_metallib(library, wrong));
        expect(!validate_metallib(library, wrong_count));
        expect(!validate_metallib(library, kEntryPoints, wrong_types));
        expect(!validate_metallib(library, kEntryPoints, wrong_type_count));
    };

    "metal_metallib_rejects_truncation"_test = [] {
        auto library = make_test_library();
        auto all_prefixes_rejected = true;
        for (auto size = 0u; size < library.size(); size++) {
            if (validate_metallib(span<const std::byte>{library}.first(size), kEntryPoints)) {
                all_prefixes_rejected = false;
                break;
            }
        }
        expect(all_prefixes_rejected);
    };

    "metal_metallib_rejects_hash_corruption"_test = [] {
        auto module_corruption = make_test_library();
        auto module_list_offset = static_cast<size_t>(read_u64(module_corruption, 72u));
        module_corruption[module_list_offset] ^= std::byte{0xffu};
        expect(!validate_metallib(module_corruption, kEntryPoints));

        auto hash_corruption = make_test_library();
        auto hash_offset = find_tag_value(hash_corruption, "HASH");
        expect(hash_offset != std::numeric_limits<size_t>::max());
        hash_corruption[hash_offset] ^= std::byte{0xffu};
        expect(!validate_metallib(hash_corruption, kEntryPoints));
    };

    "metal_metallib_rejects_trailing_data"_test = [] {
        auto library = make_test_library();
        library.emplace_back(std::byte{0u});
        auto declared_size = static_cast<uint64_t>(library.size());
        for (auto i = 0u; i < 8u; i++) {
            library[16u + i] = static_cast<std::byte>(declared_size >> (i * 8u));
        }
        expect(!validate_metallib(library, kEntryPoints));
    };

    "metal_metallib_rejects_invalid_input"_test = [] {
        auto target = metallib_target_for_macos(26u);
        auto functions = test_functions();
        expect(make_metallib(target, span<const MetalLibFunction>{}).empty());

        auto empty_name = functions;
        empty_name[0u].name = {};
        expect(make_metallib(target, empty_name).empty());

        auto embedded_null = functions;
        embedded_null[0u].name = string_view{"bad\0name", 8u};
        expect(make_metallib(target, embedded_null).empty());

        string oversized_name(std::numeric_limits<uint16_t>::max(), 'x');
        auto oversized = functions;
        oversized[0u].name = oversized_name;
        expect(make_metallib(target, oversized).empty());

        auto empty_module = functions;
        empty_module[0u].air_module = {};
        expect(make_metallib(target, empty_module).empty());

        auto invalid_type = functions;
        invalid_type[0u].type = static_cast<MetalLibProgramType>(7u);
        expect(make_metallib(target, invalid_type).empty());

        auto duplicate_name = functions;
        duplicate_name[1u].name = duplicate_name[0u].name;
        expect(make_metallib(target, duplicate_name).empty());

        auto old_file_format = target;
        old_file_format.file_format.patch = 6u;
        expect(make_metallib(old_file_format, functions).empty());

        auto old_platform = target;
        old_platform.platform.major = 12u;
        expect(make_metallib(old_platform, functions).empty());

        auto wide_platform_component = target;
        wide_platform_component.platform.minor = 256u;
        expect(make_metallib(wide_platform_component, functions).empty());

        auto invalid_air = target;
        invalid_air.air.major = 0u;
        expect(make_metallib(invalid_air, functions).empty());

        auto invalid_metal = target;
        invalid_metal.metal.major = 0u;
        expect(make_metallib(invalid_metal, functions).empty());
    };

    return 0;
}
