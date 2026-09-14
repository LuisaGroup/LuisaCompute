// Test the internal XIR program-team value-layout foundation on the host.
// Covers per-axis padding, checked owner/slot round trips, empty/invalid
// shapes, explicit read-transition facts, and executable per-phase candidate
// admission. Host planning checks do not claim GPU execution or performance.

#include "ut/ut.hpp"
#include "program_team.h"
#include "program_plan.h"
#include "tile_llm_test_utils.h"
#include <luisa/tile/verifier.h>
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

void collect_operations(tile::Block &block, vector<tile::Operation *> &operations) {
    for (auto operation : block.operations()) {
        operations.emplace_back(operation);
        for (auto &region : operation->regions()) {
            for (auto child : region->blocks()) { collect_operations(*child, operations); }
        }
    }
}
void check_same_layout(const team_detail::ValueLayout &a, const team_detail::ValueLayout &b) {
    expect(a.space() == b.space());
    expect(a.kind() == b.kind());
    expect(a.cyclic_axis() == b.cyclic_axis());
    expect(eq(a.local_elements(), b.local_elements()));
}
void check_carries(const tile::Operation &operation, const team_detail::ProgramTeamPlan &plan) {
    auto body = operation.region(0u)->block(0u);
    auto yield = body->operations().back();
    expect(plan.phase(&operation).kind() == team_detail::ValueLayout::Kind::REPLICATED);
    expect(eq(plan.phase(&operation).local_elements(), *operation.domain()->static_volume()));
    for (size_t i = 0u; i < operation.result_count(); i++) {
        if (!operation.result(i)->type().is_tile()) { continue; }
        auto &result = plan.layout(operation.result(i));
        check_same_layout(result, plan.layout(body->argument(operation.domain()->rank() + i)));
        for (auto value : {operation.operand(i), yield->operand(i)}) {
            auto producer = value->defining_operation();
            if (producer && producer->kind() == tile::OperationKind::CONSTANT) {
                expect(plan.layout(value).kind() == team_detail::ValueLayout::Kind::REPLICATED);
            } else {
                check_same_layout(result, plan.layout(value));
            }
        }
    }
}

void test_attention_phase_plan() {
    using Kind = tile::OperationKind;
    using Projection = team_detail::ReadProjection;
    using LayoutKind = team_detail::ValueLayout::Kind;
    for (auto width : {32u, 64u}) {
        for (auto query_rows : {1, 4}) {
            for (auto keys : {7, 33, 65}) {
                auto test_case = luisa::test::tile_llm::attention(1, 2, 1, 9, 97, 17, 19, query_rows, keys);
                expect(test_case.kernel.valid());
                auto &function = test_case.kernel.function();
                expect(!team_detail::packet_local_program(function, width));
                auto plan = team_detail::ProgramTeamPlan::create(function, width);
                expect(plan.has_value()) << "complete QK/online softmax/PV, width=" << width << ", bq=" << query_rows << ", bk=" << keys;
                if (!plan) { continue; }
                vector<tile::Operation *> operations, matrices;
                collect_operations(*function.body().block(0u), operations);
                size_t row_maps = 0u, pipelines = 0u;
                for (auto op : operations) {
                    if (op->kind() == Kind::MMA) { matrices.emplace_back(op); }
                    if (op->kind() == Kind::PIPELINE) {
                        pipelines++;
                        expect(eq(op->result_count(), size_t{3u}));
                        check_carries(*op, *plan);
                    }
                    if (op->kind() != Kind::TILE_MAP) { continue; }
                    auto body = op->region(0u)->block(0u);
                    for (auto child : body->operations()) {
                        if (child->kind() != Kind::REDUCE) { continue; }
                        row_maps++;
                        expect(plan->phase(op).kind() == LayoutKind::REPLICATED);
                        expect(plan->layout(op->result(0u)).kind() == LayoutKind::REPLICATED);
                        expect(eq(plan->layout(op->result(0u)).local_elements(), static_cast<uint64_t>(query_rows)));
                        expect(plan->phase(child).cyclic_axis() == child->domain()->axis(0u).dimension);
                        expect(eq(plan->phase(child).local_elements(), ceil_div(static_cast<uint64_t>(keys), static_cast<uint64_t>(width))));
                    }
                }
                expect(eq(pipelines, size_t{1u}));
                expect(eq(row_maps, size_t{2u}));
                expect(eq(matrices.size(), size_t{2u}));
                if (matrices.size() != 2u) { continue; }
                for (auto matrix : matrices) {
                    check_same_layout(plan->phase(matrix), plan->layout(matrix->result(0u)));
                    expect(plan->mma_operand_projection(matrix, 0u) == Projection::TEAM_UNIFORM);
                    expect(plan->mma_operand_projection(matrix, 1u) == Projection::OWNER_PRESERVING);
                }
                auto qk = matrices[0u], pv = matrices[1u];
                auto output_axis = qk->result(0u)->type().index_space()->axis(3u).dimension;
                auto &key = plan->layout(qk->operand(1u));
                expect(key.cyclic_axis() == output_axis);
                expect(key.cyclic_axis_index() == size_t{2u}) << "actual key axes are [b,h,n,d], not a transposed [b,h,d,n] view";
                expect(eq(key.local_elements(), uint64_t{17u} * ceil_div(static_cast<uint64_t>(keys), static_cast<uint64_t>(width))));
                expect(eq(plan->layout(qk->operand(0u)).local_elements(), static_cast<uint64_t>(query_rows)));
                expect(eq(plan->layout(pv->operand(0u)).local_elements(), static_cast<uint64_t>(query_rows) * ceil_div(static_cast<uint64_t>(keys), static_cast<uint64_t>(width))));
                expect(eq(plan->layout(pv->operand(1u)).local_elements(), static_cast<uint64_t>(keys)));
                if (query_rows == 4 && keys == 33 && width == 32u) {
                    expect(eq(plan->layout(pv->operand(0u)).local_elements(), uint64_t{8u}));
                }
            }
        }
    }
}

void test_temporal_phase_plan() {
    using namespace tile;
    for (auto pipelined : {false, true}) {
        for (auto swap : {false, true}) {
            auto definition = tile_kernel("temporal_team_layout", [=](TensorView<const float, 2> A, TensorView<float, 2> O) {
                auto m = axis("m", 4), n = axis("n", 7);
                for (auto &nest : parallel(shape(1))) {
                    auto a = A.tile(coord(0, 0), shape(m, n)).load();
                    auto b = a + 1.0f;
                    auto range = pipelined ? nest.pipeline(shape(3)) : nest.serial(shape(3));
                    for (auto &step : range) {
                        static_cast<void>(step);
                        if (swap) {
                            auto old = a;
                            a = b;
                            b = old;
                        } else {
                            a += 1.0f;
                        }
                    }
                    O(coord(0, 0), shape(m, n)).store(a + b);
                }
            });
            auto kernel = definition.capture(tensor_shape(4, 7), tensor_shape(4, 7));
            expect(kernel.valid());
            for (auto width : {32u, 64u}) {
                auto plan = team_detail::ProgramTeamPlan::create(kernel.function(), width);
                expect(plan.has_value());
                if (!plan) { continue; }
                vector<Operation *> operations;
                collect_operations(*kernel.function().body().block(0u), operations);
                auto found = 0u;
                for (auto operation : operations) {
                    if (operation->kind() != (pipelined ? OperationKind::PIPELINE : OperationKind::SERIAL)) { continue; }
                    found++;
                    expect(eq(operation->result_count(), swap ? size_t{2u} : size_t{1u}));
                    check_carries(*operation, *plan);
                    expect(eq(plan->phase(operation).local_elements(), uint64_t{3u}));
                    expect(eq(plan->layout(operation->result(0u)).local_elements(), uint64_t{4u}));
                }
                expect(eq(found, 1u));
            }
        }
    }
}

void test_phase_admission_boundaries() {
    using namespace tile;
    auto old = luisa::test::tile_llm::rows(luisa::test::tile_llm::RowOp::RMS_NORM, 3, 7);
    expect(team_detail::packet_local_program(old.kernel.function(), 32u));
    expect(!team_detail::ProgramTeamPlan::create(old.kernel.function(), 32u)) << "old packet path keeps priority";
    for (auto policy : {reduction::ordered_tree, reduction::fold_left, reduction::fold_right}) {
        auto test_case = luisa::test::tile_llm::attention(1, 1, 1, 5, 17, 7, 9, 4, 7);
        vector<Operation *> operations;
        collect_operations(*test_case.kernel.function().body().block(0u), operations);
        for (auto operation : operations) {
            if (operation->kind() == OperationKind::REDUCE) { operation->set_reduction_policy(policy); }
        }
        expect(test_case.kernel.valid());
        expect(!team_detail::ProgramTeamPlan::create(test_case.kernel.function(), 32u));
    }
    for (auto arithmetic_index : {false, true}) {
        auto definition = tile_kernel("unsupported_projection", [=](TensorView<const float, 2> A, TensorView<float, 1> O) {
            auto m = axis("m", 4), n = axis("n", 7);
            for (auto &root : parallel(shape(1))) {
                static_cast<void>(root);
                auto source = A.tile(coord(0, 0), shape(m, n)).load();
                auto rows = map<float>(shape(m), [&](const Nest &row) {
                    auto index = arithmetic_index ? row.index(m) + 0 : row.index(m);
                    return source.at(coord(index, 0));
                });
                // The same producer also needs n ownership for a row reduce.
                // Choosing only the source-owner coordinate as "uniform"
                // would incorrectly admit the varying [m,0] full projection.
                auto summed = reduce(source, n, add);
                O(coord(0), shape(m)).store(rows + summed);
            }
        });
        auto kernel = definition.capture(tensor_shape(4, 7), tensor_shape(4));
        expect(kernel.valid());
        expect(!team_detail::ProgramTeamPlan::create(kernel.function(), 32u));
    }
    auto nested = tile_kernel("nested_parallel_capability", [](TensorView<float, 2> O) {
                      for (auto &outer : parallel(shape(4))) {
                          for (auto &inner : outer.parallel(shape(7))) {
                              O(outer.index(), inner.index()).store(1.0f);
                          }
                      }
                  }).capture(tensor_shape(4, 7));
    expect(nested.valid());
    expect(!team_detail::ProgramTeamPlan::create(nested.function(), 32u));
    auto empty = tile_kernel("empty_team_shape", [](TensorView<float, 2> O) {
                     auto m = axis("m", 4), n = axis("n", 0);
                     for (auto &root : parallel(shape(1))) {
                         static_cast<void>(root);
                         O(coord(0, 0), shape(m, n)).store(zeros<float>(shape(m, n)));
                     }
                 }).capture(tensor_shape(4, 0));
    expect(empty.valid());
    expect(!team_detail::ProgramTeamPlan::create(empty.function(), 32u));
    auto nested_map = tile_kernel("nested_map_capability", [](TensorView<float, 1> O) {
                          auto m = axis("m", 4), n = axis("n", 7);
                          for (auto &root : parallel(shape(1))) {
                              static_cast<void>(root);
                              auto result = map<float>(shape(m), [&](const Nest &row) {
                                  auto nested = map<float>(shape(n), [&](const Nest &column) { return cast<float>(row.index() + column.index()); });
                                  return nested.at(coord(0));
                              });
                              O(coord(0), shape(m)).store(result);
                          }
                      }).capture(tensor_shape(4));
    expect(nested_map.valid());
    expect(!team_detail::ProgramTeamPlan::create(nested_map.function(), 32u));
}

void test_padded_integer_arithmetic() {
    using namespace tile;
    for (auto divisor : {0, -1, 2}) {
        auto definition = tile_kernel("team_integer_division", [=](TensorView<const int32_t, 2> A, TensorView<int32_t, 2> O) {
            auto m = axis("m", 4), n = axis("n", 7);
            for (auto &root : parallel(shape(1))) {
                static_cast<void>(root);
                auto input = A.tile(coord(0, 0), shape(m, n)).load();
                O(coord(0, 0), shape(m, n)).store(input / divisor);
            }
        });
        auto kernel = definition.capture(tensor_shape(4, 7), tensor_shape(4, 7));
        expect(kernel.valid());
        expect(team_detail::ProgramTeamPlan::create(kernel.function(), 32u).has_value() == (divisor == 2));
    }
    auto dynamic_divisor = tile_kernel("team_dynamic_integer_division", [](TensorView<const int32_t, 2> A, TensorView<int32_t, 2> O) {
                               auto m = axis("m", 4), n = axis("n", 7);
                               for (auto &root : parallel(shape(1))) {
                                   static_cast<void>(root);
                                   auto input = A.tile(coord(0, 0), shape(m, n)).load();
                                   O(coord(0, 0), shape(m, n)).store(input / input);
                               }
                           }).capture(tensor_shape(4, 7), tensor_shape(4, 7));
    expect(dynamic_divisor.valid());
    expect(!team_detail::ProgramTeamPlan::create(dynamic_divisor.function(), 32u));
}

void test_uniform_extract_from_matching_axis() {
    using namespace tile;
    auto kernel = tile_kernel("uniform_extract_same_owner_axis", [](TensorView<const float, 2> A, TensorView<float, 1> O) {
                      auto m = axis("m", 4), n = axis("n", 7);
                      for (auto &root : parallel(shape(1))) {
                          static_cast<void>(root);
                          auto input = A.tile(coord(0, 0), shape(m, n)).load();
                          auto result = map<float>(shape(n), [&](const Nest &element) {
                              return input.at(coord(0, 0)) + cast<float>(element.index());
                          });
                          O(coord(0), shape(n)).store(result);
                      }
                  }).capture(tensor_shape(4, 7), tensor_shape(7));
    expect(kernel.valid());
    for (auto width : {32u, 64u}) {
        auto plan = team_detail::ProgramTeamPlan::create(kernel.function(), width);
        expect(plan.has_value());
        if (!plan) { continue; }
        vector<Operation *> operations;
        collect_operations(*kernel.function().body().block(0u), operations);
        auto extracts = 0u;
        for (auto operation : operations) {
            if (operation->kind() != OperationKind::TILE_EXTRACT) { continue; }
            extracts++;
            auto parent = operation->parent_block()->parent_region()->parent_operation();
            expect(plan->layout(operation->operand(0u)).cyclic_axis() == plan->phase(parent).cyclic_axis());
            expect(plan->extract_projection(operation) == team_detail::ReadProjection::TEAM_UNIFORM);
            expect(plan->layout(operation->operand(0u)).read_transition(plan->extract_projection(operation)) == team_detail::ReadTransition::UNIFORM_BROADCAST)
                << "matching layout axes do not make a literal-zero extract owner-local";
        }
        expect(eq(extracts, 1u));
    }
}

void test_index_arithmetic_is_not_float_arithmetic() {
    using namespace tile;
    for (auto opcode : {ElementwiseOp::DIV, ElementwiseOp::MOD, ElementwiseOp::CAST}) {
        for (auto divisor : {int64_t{0}, int64_t{-1}, int64_t{2}}) {
            Module module;
            auto function = module.create_function("index_arithmetic_admission");
            auto root = function->body().append_block();
            auto root_dim = module.dimensions().create_dimension("program");
            auto step_dim = module.dimensions().create_dimension("time");
            std::array root_axes{IndexAxis{root_dim, Extent::constant(1u)}};
            std::array step_axes{IndexAxis{step_dim, Extent::constant(3u)}};
            IRBuilder builder{root};
            auto parallel = builder.create_structured(OperationKind::PARALLEL, IndexSpace{root_axes});
            builder.set_insertion_block(parallel->region(0u)->block(0u));
            auto pipeline = builder.create_structured(OperationKind::PIPELINE, IndexSpace{step_axes});
            auto body = pipeline->region(0u)->block(0u);
            builder.set_insertion_block(body);
            std::array literal_types{opcode == ElementwiseOp::CAST ? Type::scalar(ScalarType::FLOAT32) : Type::index()};
            auto literal = builder.create(OperationKind::CONSTANT, {}, literal_types);
            literal->set_attribute("value", opcode == ElementwiseOp::CAST ? Attribute{1.25} : Attribute{divisor});
            vector<Value *> operands;
            if (opcode != ElementwiseOp::CAST) { operands.emplace_back(body->argument(0u)); }
            operands.emplace_back(literal->result(0u));
            auto arithmetic = builder.create_elementwise(opcode, operands, Type::index());
            expect(arithmetic != nullptr);
            static_cast<void>(builder.create(OperationKind::YIELD));
            builder.set_insertion_block(parallel->region(0u)->block(0u));
            static_cast<void>(builder.create(OperationKind::YIELD));
            auto verification = verify(module);
            for (const auto &diagnostic : verification.diagnostics()) {
                expect(false) << diagnostic.message;
            }
            expect(verification.ok()) << "opcode=" << static_cast<uint32_t>(opcode) << " divisor=" << divisor;
            auto permitted = opcode != ElementwiseOp::CAST && divisor == 2;
            expect(team_detail::ProgramTeamPlan::create(*function, 32u).has_value() == permitted)
                << "Index uses signed integer safety despite ScalarType::INVALID";
        }
    }
}
}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    "tile_xir_program_team_attention_phase_admission"_test = test_attention_phase_plan;
    "tile_xir_program_team_temporal_carry_layouts"_test = test_temporal_phase_plan;
    "tile_xir_program_team_phase_admission_boundaries"_test = test_phase_admission_boundaries;
    "tile_xir_program_team_padded_integer_arithmetic"_test = test_padded_integer_arithmetic;
    "tile_xir_program_team_uniform_extract_matching_axis"_test = test_uniform_extract_from_matching_axis;
    "tile_xir_program_team_index_arithmetic_safety"_test = test_index_arithmetic_is_not_float_arithmetic;

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
