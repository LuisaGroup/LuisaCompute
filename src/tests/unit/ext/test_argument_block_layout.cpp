// Pure tests for checked host/device argument-block layout boundaries.
// No backend or device is required.

#include "ut/ut.hpp"

#include <cstdint>
#include <limits>

#include "argument_block_layout.h"

using namespace boost::ut;
using namespace boost::ut::literals;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "argument_block_accepts_exact_limit_and_rejects_plus_one"_test = [] {
        using namespace lc;

        ArgumentBlockLayout exact{64u};
        size_t offset = std::numeric_limits<size_t>::max();
        expect(exact.append(63u, 1u, offset));
        expect(eq(offset, 0u));
        expect(exact.finalize_words());
        expect(eq(exact.size(), 64u));

        ArgumentBlockLayout rounded_over_limit{63u};
        expect(rounded_over_limit.append(63u, 1u, offset));
        expect(!rounded_over_limit.finalize_words());
        expect(rounded_over_limit.status() ==
               ArgumentBlockLayoutStatus::LIMIT_EXCEEDED);
        expect(eq(rounded_over_limit.size(), 63u));

        constexpr auto uint32_limit = static_cast<size_t>(
            std::numeric_limits<uint32_t>::max());
        ArgumentBlockLayout uint32_exact{uint32_limit};
        expect(uint32_exact.append(uint32_limit, 1u, offset));
        expect(eq(uint32_exact.size(), uint32_limit));
        offset = 17u;
        expect(!uint32_exact.append(1u, 1u, offset));
        if constexpr (sizeof(size_t) > sizeof(uint32_t)) {
            expect(uint32_exact.status() ==
                   ArgumentBlockLayoutStatus::LIMIT_EXCEEDED);
        } else {
            expect(uint32_exact.status() ==
                   ArgumentBlockLayoutStatus::ADDITION_OVERFLOW);
        }
        expect(eq(uint32_exact.size(), uint32_limit));
        expect(eq(offset, 17u));
    };

    "argument_block_rejects_invalid_alignment_and_host_overflow"_test = [] {
        using namespace lc;
        constexpr auto max_size =
            std::numeric_limits<size_t>::max();
        size_t offset = 0u;

        ArgumentBlockLayout invalid_alignment;
        expect(!invalid_alignment.append(1u, 3u, offset));
        expect(invalid_alignment.status() ==
               ArgumentBlockLayoutStatus::INVALID_ALIGNMENT);

        ArgumentBlockLayout alignment_overflow;
        expect(alignment_overflow.append(max_size - 3u, 1u, offset));
        expect(!alignment_overflow.align_to(8u));
        expect(alignment_overflow.status() ==
               ArgumentBlockLayoutStatus::ALIGNMENT_OVERFLOW);
        expect(eq(alignment_overflow.size(), max_size - 3u));

        ArgumentBlockLayout addition_overflow;
        expect(addition_overflow.append(max_size - 3u, 1u, offset));
        expect(!addition_overflow.append(4u, 1u, offset));
        expect(addition_overflow.status() ==
               ArgumentBlockLayoutStatus::ADDITION_OVERFLOW);
        expect(eq(addition_overflow.size(), max_size - 3u));

        ArgumentBlockLayout multiplication_overflow;
        expect(!multiplication_overflow.append_array(
            max_size / 16u + 1u, 16u, 1u, offset));
        expect(multiplication_overflow.status() ==
               ArgumentBlockLayoutStatus::MULTIPLICATION_OVERFLOW);
        expect(eq(multiplication_overflow.size(), 0u));

        ArgumentBlockLayout zero_stride;
        offset = 19u;
        expect(!zero_stride.append_array(1u, 0u, 1u, offset));
        expect(zero_stride.status() ==
               ArgumentBlockLayoutStatus::INVALID_STRIDE);
        expect(eq(zero_stride.size(), 0u));
        expect(eq(offset, 19u));

        ArgumentBlockLayout empty_array;
        expect(empty_array.append_array(0u, 0u, 4u, offset));
        expect(eq(offset, 0u));
        expect(eq(empty_array.size(), 0u));
    };

    "argument_block_cumulative_padding_is_checked_and_transactional"_test = [] {
        using namespace lc;
        size_t offset = 0u;
        ArgumentBlockLayout cumulative{96u};
        expect(cumulative.append_padded(33u, 32u, offset));
        expect(eq(offset, 0u));
        expect(eq(cumulative.size(), 64u));
        expect(cumulative.append_padded(32u, 32u, offset));
        expect(eq(offset, 64u));
        expect(eq(cumulative.size(), 96u));

        offset = 11u;
        expect(!cumulative.append_padded(1u, 32u, offset));
        expect(cumulative.status() ==
               ArgumentBlockLayoutStatus::LIMIT_EXCEEDED);
        expect(eq(cumulative.size(), 96u));
        expect(eq(offset, 11u));
    };

    "argument_block_metadata_and_validation_trailers_are_exact"_test = [] {
        using namespace lc;
        size_t offset = 0u;

        ArgumentBlockLayout metadata{40u};
        expect(metadata.append(3u, 1u, offset));
        ArgumentBlockTrailerPlacement metadata_placement{};
        expect(metadata.append_trailers(
            ArgumentBlockTrailerLayout{
                .metadata_count = 2u,
                .metadata_stride = 16u,
                .metadata_alignment = 8u,
                .word_alignment = 4u},
            metadata_placement));
        expect(eq(metadata_placement.metadata_offset, 8u));
        expect(eq(metadata_placement.metadata_size, 32u));
        expect(eq(metadata_placement.validation_offset, 40u));
        expect(eq(metadata_placement.validation_size, 0u));
        expect(eq(metadata_placement.final_size, 40u));
        expect(eq(metadata.size(), 40u));

        ArgumentBlockLayout validation{16u};
        expect(validation.append(3u, 1u, offset));
        ArgumentBlockTrailerPlacement validation_placement{};
        expect(validation.append_trailers(
            ArgumentBlockTrailerLayout{
                .validation_count = 3u,
                .validation_stride = sizeof(uint32_t),
                .word_alignment = sizeof(uint32_t)},
            validation_placement));
        expect(eq(validation_placement.metadata_offset, 3u));
        expect(eq(validation_placement.metadata_size, 0u));
        expect(eq(validation_placement.validation_offset, 4u));
        expect(eq(validation_placement.validation_size, 12u));
        expect(eq(validation_placement.final_size, 16u));
        expect(eq(validation.size(), 16u));
    };

    "argument_block_validation_values_are_exact_and_representable"_test = [] {
        using namespace lc;
        uint32_t value = 0xdeadbeefu;

        expect(argument_block_validation_value(96u, 24u, value));
        expect(eq(value, 4u));
        expect(argument_block_validation_value(96u, 0u, value));
        expect(eq(value, 96u));

        value = 0xdeadbeefu;
        expect(!argument_block_validation_value(95u, 24u, value));
        expect(eq(value, 0xdeadbeefu));

        if constexpr (sizeof(size_t) > sizeof(uint32_t)) {
            constexpr auto too_large =
                static_cast<size_t>(std::numeric_limits<uint32_t>::max()) +
                1u;
            expect(!argument_block_validation_value(
                too_large, 0u, value));
            expect(eq(value, 0xdeadbeefu));
        }
    };

    "argument_block_rejects_mixed_trailers_without_partial_commit"_test = [] {
        using namespace lc;
        size_t offset = 0u;
        ArgumentBlockLayout layout{64u};
        expect(layout.append(3u, 1u, offset));
        ArgumentBlockTrailerPlacement placement{
            .metadata_offset = 7u,
            .metadata_size = 8u,
            .validation_offset = 9u,
            .validation_size = 10u,
            .final_size = 11u};
        expect(!layout.append_trailers(
            ArgumentBlockTrailerLayout{
                .metadata_count = 1u,
                .metadata_stride = 16u,
                .metadata_alignment = 8u,
                .validation_count = 1u,
                .validation_stride = sizeof(uint32_t)},
            placement));
        expect(layout.status() ==
               ArgumentBlockLayoutStatus::INCOMPATIBLE_TRAILERS);
        expect(eq(layout.size(), 3u));
        expect(eq(placement.metadata_offset, 7u));
        expect(eq(placement.metadata_size, 8u));
        expect(eq(placement.validation_offset, 9u));
        expect(eq(placement.validation_size, 10u));
        expect(eq(placement.final_size, 11u));
    };
}
