#include "ut/ut.hpp"

#include <cstdint>
#include <limits>

#include <luisa/runtime/rhi/pixel.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "pixel_storage_size_widens_extent_products_before_multiplication"_test = [] {
        constexpr auto result = checked_pixel_storage_size(
            PixelStorage::BYTE1, uint3{65536u, 65536u, 1u});
        static_assert(
            (sizeof(size_t) > sizeof(uint32_t) &&
             static_cast<bool>(result) &&
             result.size == (uint64_t{1u} << 32u)) ||
            (sizeof(size_t) <= sizeof(uint32_t) &&
             result.status == PixelStorageSizeStatus::OVERFLOW));
        if constexpr (sizeof(size_t) > sizeof(uint32_t)) {
            expect(static_cast<bool>(result));
            expect(eq(result.size,
                      static_cast<size_t>(uint64_t{1u} << 32u)));
        } else {
            expect(result.status == PixelStorageSizeStatus::OVERFLOW);
        }
    };

    "pixel_storage_size_block_rounding_does_not_overflow_extent_addition"_test = [] {
        constexpr auto result = checked_pixel_storage_size(
            PixelStorage::BC1,
            uint3{std::numeric_limits<uint32_t>::max(), 1u, 1u});
        static_assert(
            (sizeof(size_t) > sizeof(uint32_t) &&
             static_cast<bool>(result) &&
             result.size == (uint64_t{1u} << 33u)) ||
            (sizeof(size_t) <= sizeof(uint32_t) &&
             result.status == PixelStorageSizeStatus::OVERFLOW));
        if constexpr (sizeof(size_t) > sizeof(uint32_t)) {
            constexpr auto expected = static_cast<size_t>(
                uint64_t{1u} << 33u);
            expect(static_cast<bool>(result));
            expect(eq(result.size, expected));
        } else {
            expect(result.status == PixelStorageSizeStatus::OVERFLOW);
        }
    };

    "pixel_storage_size_reports_host_overflow_and_invalid_storage"_test = [] {
        constexpr auto overflow = checked_pixel_storage_size(
            PixelStorage::FLOAT4,
            uint3{std::numeric_limits<uint32_t>::max(),
                  std::numeric_limits<uint32_t>::max(),
                  std::numeric_limits<uint32_t>::max()});
        static_assert(overflow.status ==
                      PixelStorageSizeStatus::OVERFLOW);
        expect(overflow.status == PixelStorageSizeStatus::OVERFLOW);

        constexpr auto invalid = checked_pixel_storage_size(
            static_cast<PixelStorage>(pixel_storage_count),
            uint3{1u, 1u, 1u});
        static_assert(invalid.status ==
                      PixelStorageSizeStatus::INVALID_STORAGE);
        expect(invalid.status == PixelStorageSizeStatus::INVALID_STORAGE);
    };

    "pixel_storage_size_retains_normal_and_zero_extent_semantics"_test = [] {
        constexpr auto rgba = checked_pixel_storage_size(
            PixelStorage::FLOAT4, uint3{3u, 2u, 1u});
        static_assert(rgba.size == 96u);
        expect(static_cast<bool>(rgba));
        expect(eq(rgba.size, size_t{96u}));

        constexpr auto bc = checked_pixel_storage_size(
            PixelStorage::BC1, uint3{5u, 7u, 0u});
        static_assert(bc.size == 32u);
        expect(static_cast<bool>(bc));
        expect(eq(bc.size, size_t{32u}));

        constexpr auto empty = checked_pixel_storage_size(
            PixelStorage::BYTE4, uint3{0u, 7u, 1u});
        static_assert(empty.size == 0u);
        expect(static_cast<bool>(empty));
        expect(eq(empty.size, size_t{0u}));
    };
}
