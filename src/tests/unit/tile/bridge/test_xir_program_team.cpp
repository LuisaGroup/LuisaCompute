// Test the internal XIR program-team value-layout foundation on the host.
// Covers per-axis padding, checked owner/slot round trips, empty/invalid
// shapes, and explicit read-transition facts. No kernel/GPU admission claim.

#include "ut/ut.hpp"
#include "program_team.h"
#include <array>
#include <limits>

using namespace luisa;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace tile = luisa::compute::tile;
namespace team_detail = tile::bridge::xir::detail;

namespace {
[[nodiscard]] tile::IndexSpace matrix_space(tile::Dim row, uint64_t rows, tile::Dim column, uint64_t columns) {
    std::array axes{tile::IndexAxis{row, tile::Extent::constant(rows)},
                    tile::IndexAxis{column, tile::Extent::constant(columns)}};
    return tile::IndexSpace{axes};
}

void check_round_trips(const team_detail::ValueLayout &layout, uint64_t rows, uint64_t columns) {
    auto width = layout.team().width();
    for (uint64_t row = 0u; row < rows; row++) {
        for (uint64_t column = 0u; column < columns; column++) {
            std::array coordinates{row, column};
            auto reader = static_cast<uint32_t>((row + column + 1u) % width);
            auto mapped = layout.map(coordinates, reader);
            expect(mapped.has_value());
            if (!mapped) { continue; }
            auto expected_owner = reader;
            auto expected_flat = row * columns + column;
            if (auto axis = layout.cyclic_axis_index()) {
                if (*axis == 0u) {
                    expected_owner = static_cast<uint32_t>(row % width);
                    expected_flat = (row / width) * columns + column;
                } else {
                    expected_owner = static_cast<uint32_t>(column % width);
                    expected_flat = row * (columns / width + static_cast<uint64_t>(columns % width != 0u)) + column / width;
                }
            }
            expect(eq(mapped->owner, expected_owner));
            expect(eq(mapped->local_flat, expected_flat));
            auto inverted = layout.unmap(mapped->owner, mapped->local_flat);
            expect(inverted.has_value());
            if (inverted) {
                expect(eq(inverted->size(), size_t{2u}));
                expect(eq((*inverted)[0u], row));
                expect(eq((*inverted)[1u], column));
            }
        }
    }
    uint64_t valid_slots = 0u;
    for (uint32_t lane = 0u; lane < width; lane++) {
        for (uint64_t slot = 0u; slot < layout.local_elements(); slot++) {
            auto coordinates = layout.unmap(lane, slot);
            if (!coordinates) { continue; }
            valid_slots++;
            auto mapped = layout.map(*coordinates, lane);
            expect(mapped.has_value());
            if (mapped) {
                expect(eq(mapped->owner, lane));
                expect(eq(mapped->local_flat, slot));
            }
        }
    }
    auto copies = layout.kind() == team_detail::ValueLayout::Kind::REPLICATED ? width : 1u;
    expect(eq(valid_slots, rows * columns * copies));
    expect(!layout.unmap(width, 0u));
    expect(!layout.unmap(0u, layout.local_elements()));
    expect(!layout.map(std::array{rows, uint64_t{0u}}, 0u));
    expect(!layout.map(std::array{uint64_t{0u}, columns}, 0u));
    expect(!layout.map(std::array{uint64_t{0u}, uint64_t{0u}}, width));
    expect(!layout.map(std::array{uint64_t{0u}}, 0u));
}
}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    "tile_xir_program_team_geometry_is_not_a_backend_limit"_test = [] {
        for (auto width : {1u, 2u, 8u, 16u, 32u, 64u, 128u, uint32_t{1u} << 31u}) {
            auto team = team_detail::ProgramTeamLayout::create(width);
            expect(team.has_value());
            if (team) { expect(eq(team->width(), width)); }
        }
        for (auto width : {0u, 3u, 6u, 31u, 63u, std::numeric_limits<uint32_t>::max()}) {
            expect(!team_detail::ProgramTeamLayout::create(width));
        }
    };

    "tile_xir_program_team_padding_is_per_axis"_test = [] {
        tile::DimensionContext dimensions;
        auto row = dimensions.create_dimension("row");
        auto key = dimensions.create_dimension("key");
        for (auto width : {32u, 64u}) {
            auto team = team_detail::ProgramTeamLayout::create(width);
            if (!team) {
                expect(false);
                continue;
            }
            for (auto columns : {16u, 33u, 65u}) {
                auto space = matrix_space(row, 4u, key, columns);
                auto replicated = team_detail::ValueLayout::replicated(*team, space);
                auto cyclic = team_detail::ValueLayout::cyclic(*team, space, key);
                expect(replicated.has_value() && cyclic.has_value());
                if (!replicated || !cyclic) { continue; }
                auto slots_per_row = columns / width + static_cast<uint32_t>(columns % width != 0u);
                expect(eq(replicated->local_elements(), uint64_t{4u} * columns));
                expect(eq(cyclic->local_elements(), uint64_t{4u} * slots_per_row));
                expect(eq(cyclic->local_extents()[0u], uint64_t{4u}));
                expect(eq(cyclic->local_extents()[1u], static_cast<uint64_t>(slots_per_row)));
                expect(eq(cyclic->logical_elements(), uint64_t{4u} * columns));
                expect(cyclic->cyclic_axis() == key);
                if (width == 32u && columns == 33u) { expect(eq(cyclic->local_elements(), uint64_t{8u})); }
                if (columns == 16u) { expect(eq(cyclic->local_elements(), uint64_t{4u})); }
            }
            std::array axes{tile::IndexAxis{row, tile::Extent::constant(4u)}};
            auto result = team_detail::ValueLayout::replicated(*team, tile::IndexSpace{axes});
            expect(result.has_value());
            if (result) { expect(eq(result->local_elements(), uint64_t{4u})); }
        }
    };

    "tile_xir_program_team_checked_mapping_round_trips"_test = [] {
        tile::DimensionContext dimensions;
        auto row = dimensions.create_dimension();
        auto column = dimensions.create_dimension();
        for (auto width : {1u, 2u, 8u, 32u, 64u}) {
            auto team = team_detail::ProgramTeamLayout::create(width);
            if (!team) {
                expect(false);
                continue;
            }
            for (auto columns : {0u, 1u, 7u, 16u, 31u, 32u, 33u, 65u}) {
                auto space = matrix_space(row, 4u, column, columns);
                auto replicated = team_detail::ValueLayout::replicated(*team, space);
                expect(replicated.has_value());
                if (replicated) { check_round_trips(*replicated, 4u, columns); }
                for (auto axis : {row, column}) {
                    auto cyclic = team_detail::ValueLayout::cyclic(*team, space, axis);
                    expect(cyclic.has_value());
                    if (cyclic) { check_round_trips(*cyclic, 4u, columns); }
                }
            }
        }
    };

    "tile_xir_program_team_layout_identity_is_not_shape_identity"_test = [] {
        tile::DimensionContext dimensions;
        auto row = dimensions.create_dimension("row");
        auto key = dimensions.create_dimension("key");
        auto extra = dimensions.create_dimension("extra");
        auto team = team_detail::ProgramTeamLayout::create(32u);
        if (!team) {
            expect(false);
            return;
        }
        auto source = matrix_space(row, 4u, key, 33u);
        auto row_layout = team_detail::ValueLayout::cyclic(*team, source, row);
        auto key_layout = team_detail::ValueLayout::cyclic(*team, source, key);
        expect(row_layout.has_value() && key_layout.has_value());
        if (!row_layout || !key_layout) { return; }
        expect(source.add(extra, 2u));
        expect(eq(row_layout->space().rank(), size_t{2u}));
        expect(eq(key_layout->space().rank(), size_t{2u}));
        auto a = row_layout->map(std::array{uint64_t{3u}, uint64_t{32u}}, 7u);
        auto b = key_layout->map(std::array{uint64_t{3u}, uint64_t{32u}}, 7u);
        expect(a.has_value() && b.has_value());
        if (a && b) {
            expect(eq(a->owner, 3u));
            expect(eq(a->local_flat, uint64_t{32u}));
            expect(eq(b->owner, 0u));
            expect(eq(b->local_flat, uint64_t{7u}));
        }
        expect(!key_layout->unmap(1u, 7u));// key=33 is padding, not a value.
    };

    "tile_xir_program_team_invalid_and_empty_shapes"_test = [] {
        tile::DimensionContext dimensions, other_dimensions;
        auto row = dimensions.create_dimension("row");
        auto key = dimensions.create_dimension("key");
        auto absent = dimensions.create_dimension("absent");
        auto foreign = other_dimensions.create_dimension("key");
        auto dynamic = dimensions.create_dynamic_extent();
        auto team = team_detail::ProgramTeamLayout::create(32u);
        if (!team) {
            expect(false);
            return;
        }
        auto valid = matrix_space(row, 4u, key, 33u);
        for (auto axis : {tile::Dim{}, absent, foreign}) { expect(!team_detail::ValueLayout::cyclic(*team, valid, axis)); }
        for (auto invalid : {
                 std::array{tile::IndexAxis{row, tile::Extent::constant(4u)}, tile::IndexAxis{row, tile::Extent::constant(33u)}},
                 std::array{tile::IndexAxis{row, tile::Extent::constant(4u)}, tile::IndexAxis{foreign, tile::Extent::constant(33u)}},
                 std::array{tile::IndexAxis{tile::Dim{}, tile::Extent::constant(4u)}, tile::IndexAxis{key, tile::Extent::constant(33u)}},
                 std::array{tile::IndexAxis{row, tile::Extent{}}, tile::IndexAxis{key, tile::Extent::constant(33u)}},
                 std::array{tile::IndexAxis{row, tile::Extent::dynamic(dynamic)}, tile::IndexAxis{key, tile::Extent::constant(0u)}},
                 std::array{tile::IndexAxis{row, tile::Extent::constant(std::numeric_limits<uint64_t>::max())}, tile::IndexAxis{key, tile::Extent::constant(2u)}}}) {
            auto space = tile::IndexSpace{invalid};
            expect(!team_detail::ValueLayout::replicated(*team, space));
            expect(!team_detail::ValueLayout::cyclic(*team, space, key));
        }
        auto scalar = team_detail::ValueLayout::replicated(*team, {});
        expect(scalar.has_value());
        if (scalar) {
            expect(eq(scalar->local_elements(), uint64_t{1u}));
            auto mapped = scalar->map({}, 9u);
            expect(mapped.has_value());
            if (mapped) {
                expect(eq(mapped->owner, 9u));
                expect(eq(mapped->local_flat, uint64_t{0u}));
            }
            auto coordinates = scalar->unmap(9u, 0u);
            expect(coordinates.has_value());
            if (coordinates) { expect(coordinates->empty()); }
        }
        expect(!team_detail::ValueLayout::cyclic(*team, {}, row));
        std::array zero_axes{tile::IndexAxis{row, tile::Extent::constant(std::numeric_limits<uint64_t>::max())},
                             tile::IndexAxis{key, tile::Extent::constant(2u)}, tile::IndexAxis{absent, tile::Extent::constant(0u)}};
        auto zero = team_detail::ValueLayout::cyclic(*team, tile::IndexSpace{zero_axes}, key);
        expect(zero.has_value());
        if (zero) {
            expect(eq(zero->local_elements(), uint64_t{0u}));
            expect(!zero->unmap(0u, 0u));
            expect(!zero->map(std::array{uint64_t{0u}, uint64_t{0u}, uint64_t{0u}}, 0u));
        }
    };

    "tile_xir_program_team_uint64_mapping_is_checked"_test = [] {
        tile::DimensionContext dimensions;
        auto index = dimensions.create_dimension();
        auto maximum = std::numeric_limits<uint64_t>::max();
        auto team = team_detail::ProgramTeamLayout::create(64u);
        if (!team) {
            expect(false);
            return;
        }
        std::array axes{tile::IndexAxis{index, tile::Extent::constant(maximum)}};
        auto layout = team_detail::ValueLayout::cyclic(*team, tile::IndexSpace{axes}, index);
        expect(layout.has_value());
        if (!layout) { return; }
        expect(eq(layout->local_elements(), maximum / 64u + 1u));
        auto mapped = layout->map(std::array{maximum - 1u}, 63u);
        expect(mapped.has_value());
        if (mapped) {
            expect(eq(mapped->owner, 62u));
            auto coordinates = layout->unmap(mapped->owner, mapped->local_flat);
            expect(coordinates.has_value());
            if (coordinates) { expect(eq((*coordinates)[0u], maximum - 1u)); }
        }
        expect(!layout->map(std::array{maximum}, 0u));
        expect(!layout->unmap(63u, layout->local_elements() - 1u));
    };

    "tile_xir_program_team_read_transitions_require_projection_facts"_test = [] {
        tile::DimensionContext dimensions;
        auto row = dimensions.create_dimension();
        auto key = dimensions.create_dimension();
        auto team = team_detail::ProgramTeamLayout::create(32u);
        if (!team) {
            expect(false);
            return;
        }
        auto space = matrix_space(row, 4u, key, 33u);
        auto replicated = team_detail::ValueLayout::replicated(*team, space);
        auto cyclic = team_detail::ValueLayout::cyclic(*team, space, key);
        expect(replicated.has_value() && cyclic.has_value());
        if (!replicated || !cyclic) { return; }
        using Projection = team_detail::ReadProjection;
        using Transition = team_detail::ReadTransition;
        for (auto projection : {Projection::UNKNOWN, Projection::OWNER_PRESERVING, Projection::TEAM_UNIFORM}) {
            expect(replicated->read_transition(projection) == Transition::REPLICATED_LOCAL);
        }
        expect(cyclic->read_transition(Projection::OWNER_PRESERVING) == Transition::OWNER_LOCAL);
        expect(cyclic->read_transition(Projection::TEAM_UNIFORM) == Transition::UNIFORM_BROADCAST);
        expect(cyclic->read_transition(Projection::UNKNOWN) == Transition::UNSUPPORTED);
        // Even a unit distributed axis does not prove the other coordinates
        // uniform; it cannot turn an unknown projection into a broadcast.
        auto unit_axis = team_detail::ValueLayout::cyclic(*team, matrix_space(row, 4u, key, 1u), key);
        expect(unit_axis.has_value());
        if (unit_axis) { expect(unit_axis->read_transition(Projection::UNKNOWN) == Transition::UNSUPPORTED); }
    };
    return 0;
}
