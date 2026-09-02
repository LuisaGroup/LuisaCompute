// Tests for the exact Tile layout correspondence -> native TVM TIRx bridge.

#include "ut/ut.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include <tvm/runtime/tensor.h>
#include <tvm/tirx/stmt_functor.h>

#include <luisa/tile/bridge/tirx/compiler.h>
#include <luisa/tile/bridge/tirx/layout.h>
#include <luisa/tile/bridge/tirx/lower.h>
#include <luisa/tile/dsl.h>

using namespace luisa;
using namespace luisa::compute::tile;
using namespace luisa::compute::tile::bridge::tirx;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct TiledReplicaFixture {
    DimensionContext dimensions;
    Dim row = dimensions.create_dimension("row");
    Dim column = dimensions.create_dimension("column");
    Dim replica = dimensions.create_dimension("replica");
    Dim lane = dimensions.create_dimension("physical lane");
    Dim warp = dimensions.create_dimension("physical warp");
    Dim slot = dimensions.create_dimension("physical slot");
    IndexSpace logical;
    LayoutSpec layout;

    TiledReplicaFixture() noexcept
        : layout{[&] {
              static_cast<void>(logical.add(row, 8u));
              static_cast<void>(logical.add(column, 16u));
              return logical;
          }()} {
        // Official TIRx documentation example:
        // S[(8,2,4,2):(4@laneid,1@warpid,1@laneid,1@m)]
        // + R[2:4@warpid] + 5@warpid
        static_cast<void>(layout.add_shard(8u, 4, lane));
        static_cast<void>(layout.add_shard(2u, 1, warp));
        static_cast<void>(layout.add_shard(4u, 1, lane));
        static_cast<void>(layout.add_shard(2u, 1, slot));
        static_cast<void>(layout.add_replica(replica, 2u, 4, warp));
        static_cast<void>(layout.add_offset(warp, 5));
    }

    [[nodiscard]] luisa::vector<AxisBinding> bindings() const noexcept {
        return {{lane, "laneid"}, {warp, "warpid"}, {slot, "m"}};
    }
};

[[nodiscard]] bool contains_point(
    luisa::span<const luisa::vector<int64_t>> points,
    std::initializer_list<int64_t> expected) noexcept {
    luisa::vector<int64_t> value{expected};
    for (auto &&point : points) {
        if (point == value) { return true; }
    }
    return false;
}

void test_correspondence() {
    TiledReplicaFixture fixture;
    expect(fixture.layout.verify());
    auto correspondence = fixture.layout.correspondence();
    expect(correspondence.has_value());
    expect(correspondence->verify());

    auto properties = correspondence->analyze_finite();
    expect(properties.enumerated);
    expect(properties.total);
    expect(properties.covers_logical_space);
    expect(eq(properties.fiber_points, 256u));
    expect(eq(properties.logical_points, 128u));
    expect(eq(properties.minimum_replication, 2u));
    expect(eq(properties.maximum_replication, 2u));

    int64_t first[]{0, 0};
    auto first_placements = correspondence->placements(first);
    expect(first_placements.has_value());
    expect(eq(first_placements->size(), 2u));
    expect(contains_point(*first_placements, {0, 5, 0}));
    expect(contains_point(*first_placements, {0, 9, 0}));

    int64_t last[]{7, 15};
    auto last_placements = correspondence->placements(last);
    expect(last_placements.has_value());
    expect(eq(last_placements->size(), 2u));
    expect(contains_point(*last_placements, {31, 6, 1}));
    expect(contains_point(*last_placements, {31, 10, 1}));
}

void test_coincident_replica_witnesses() {
    DimensionContext dimensions;
    auto logical_dimension = dimensions.create_dimension("logical");
    auto replica_dimension = dimensions.create_dimension("replica");
    auto physical_dimension = dimensions.create_dimension("physical");
    IndexSpace logical;
    expect(logical.add(logical_dimension, 4u));
    LayoutSpec layout{logical};
    expect(layout.add_shard(4u, 1, physical_dimension));
    expect(layout.add_replica(replica_dimension, 3u, 0, physical_dimension));
    auto correspondence = layout.correspondence();
    expect(correspondence.has_value());
    auto properties = correspondence->analyze_finite();
    expect(properties.covers_logical_space);
    expect(eq(properties.fiber_points, 12u));
    expect(eq(properties.minimum_replication, 1u));
    expect(eq(properties.maximum_replication, 1u));
}

[[gnu::noinline]] bool verify_native_layout(const tvm::tirx::Layout &layout) {
    return layout->VerifyWellFormed();
}

[[gnu::noinline]] tvm::ffi::Map<tvm::ffi::String, tvm::PrimExpr> apply_native_layout(
    const tvm::tirx::Layout &layout,
    tvm::ffi::Array<tvm::PrimExpr> coordinate,
    tvm::ffi::Array<tvm::PrimExpr> shape) {
    return layout->Apply(std::move(coordinate), std::move(shape));
}

[[nodiscard]] int64_t native_coordinate(
    const tvm::ffi::Map<tvm::ffi::String, tvm::PrimExpr> &placement,
    const char *axis) {
    auto value = placement.Get(tvm::ffi::String{axis});
    if (!value) { return std::numeric_limits<int64_t>::min(); }
    auto constant = value->as<tvm::IntImmNode>();
    return constant == nullptr ? std::numeric_limits<int64_t>::min() : constant->value;
}

void test_native_export() {
    TiledReplicaFixture fixture;
    auto bindings = fixture.bindings();
    auto exported = export_layout(fixture.layout, bindings);
    expect(exported.ok());
    tvm::tirx::Layout native = exported.value;
    expect(verify_native_layout(native));
    auto first = apply_native_layout(
        native,
        {tvm::IntImm::Int64(0), tvm::IntImm::Int64(0)},
        {tvm::IntImm::Int64(8), tvm::IntImm::Int64(16)});
    expect(eq(native_coordinate(first, "m"), int64_t{0}));
    expect(eq(native_coordinate(first, "warpid"), int64_t{5}));
    expect(eq(native_coordinate(first, "laneid"), int64_t{0}));
    auto last = apply_native_layout(
        native,
        {tvm::IntImm::Int64(7), tvm::IntImm::Int64(15)},
        {tvm::IntImm::Int64(8), tvm::IntImm::Int64(16)});
    expect(eq(native_coordinate(last, "m"), int64_t{1}));
    expect(eq(native_coordinate(last, "warpid"), int64_t{6}));
    expect(eq(native_coordinate(last, "laneid"), int64_t{31}));

    auto missing = bindings;
    missing.pop_back();
    expect(!export_layout(fixture.layout, missing).ok());
    auto duplicate = bindings;
    duplicate.emplace_back(AxisBinding{fixture.lane, "another_lane"});
    expect(!export_layout(fixture.layout, duplicate).ok());
    auto alias = bindings;
    alias[2].name = "laneid";
    expect(!export_layout(fixture.layout, alias).ok());
}

void test_native_compiler() {
    tvm::tirx::PrimVar lhs{"lhs", tvm::PrimType::Float(32)};
    tvm::tirx::PrimVar rhs{"rhs", tvm::PrimType::Float(32)};
    tvm::tirx::PrimFunc function{
        {lhs, rhs},
        tvm::tirx::Return{tvm::tirx::Add{lhs, rhs}},
        tvm::PrimType::Float(32)};
    auto compilation = compile(std::move(function), "tile_tirx_scalar_add");
    expect(compilation.ok()) << compilation.error();
    if (!compilation) { return; }
    auto entry = compilation.module().value()->GetFunction("tile_tirx_scalar_add", true);
    expect(entry.has_value());
    if (!entry) { return; }
    auto result = (*entry)(1.25f, 2.5f).cast<float>();
    expect(eq(result, 3.75f));
}

void test_native_index_expressions() {
    DimensionContext dimensions;
    auto axis = dimensions.create_dimension("logical index");
    auto output = dimensions.create_dimension("address");
    IndexSpace logical;
    IndexSpace storage;
    expect(logical.add(axis, 16u));
    expect(storage.add(output, 128u));
    auto i = IndexExpr::coordinate(axis);
    auto c = [](int64_t value) { return IndexExpr::constant(value); };
    IndexExpr expressions[]{
        i * c(3) + c(1),
        floor_div(i - c(11), c(3)) + c(4),
        modulo(i - c(11), c(3)),
        floor_div(i - c(11), c(-3)) + c(3),
        modulo(i + c(15), c(-3)) + c(2),
        bit_xor(i, shift_right(i, c(1))),
        bit_and(i, c(7)),
        shift_right(shift_left(i, c(60)), c(60)),
        shift_right(i - c(16), c(60)),
        shift_left(c(1), bit_and(i, c(3)))};
    auto serial = 0u;
    for (auto &&expression : expressions) {
        IndexExpr outputs[]{expression};
        IndexMap map{logical, storage, outputs};
        auto properties = map.analyze_finite();
        expect(properties.total && properties.in_bounds);
        // General index arithmetic need not be injective. Only use as a
        // writable Memory layout imposes the stronger storage proof.
        tvm::tirx::PrimVar parameter{"index", tvm::PrimType::Int(64)};
        auto native = lower_index_map(map, {parameter});
        expect(native.ok()) << native.error;
        if (!native || native.value.size() != 1u) { continue; }
        tvm::tirx::PrimFunc function{{parameter}, tvm::tirx::Return{native.value[0]}, tvm::PrimType::Int(64)};
        auto name = std::string{"tile_index_expression_"} + std::to_string(serial++);
        auto compilation = compile(std::move(function), name);
        expect(compilation.ok()) << compilation.error();
        if (!compilation) { continue; }
        auto entry = compilation.module().value()->GetFunction(tvm::ffi::String{name}, true);
        expect(entry.has_value());
        if (!entry) { continue; }
        auto correct = true;
        for (auto index = int64_t{0}; index < 16; index++) {
            int64_t point[]{index};
            auto expected = map.apply(point);
            auto actual = (*entry)(index).cast<int64_t>();
            correct &= expected.has_value() && actual == expected->front();
        }
        expect(correct) << name;
        expect(!lower_index_map(map, {}).ok());
        expect(!lower_index_map(map, {tvm::IntImm{tvm::PrimType::Int(32), 0}}).ok());
    }
}

void test_native_buffer_kernel() {
    constexpr int64_t n = 17;
    auto extent = tvm::IntImm::Int64(n);
    auto input = tvm::tirx::decl_buffer({extent}, tvm::PrimType::Float(32), "input");
    auto output = tvm::tirx::decl_buffer({extent}, tvm::PrimType::Float(32), "output");
    tvm::tirx::PrimVar i{"i", tvm::PrimType::Int(64)};
    auto value = tvm::tirx::BufferLoad{input, {i}} + tvm::FloatImm{tvm::PrimType::Float(32), 1.0};
    auto body = tvm::tirx::For{
        i,
        tvm::IntImm::Int64(0),
        extent,
        tvm::tirx::ForKind::kSerial,
        tvm::tirx::BufferStore{output, value, {i}}};
    tvm::tirx::PrimFunc function{{input.var(), output.var()}, std::move(body)};
    auto compilation = compile(std::move(function), "tile_tirx_add_one");
    expect(compilation.ok()) << compilation.error();
    if (!compilation) { return; }
    auto entry = compilation.module().value()->GetFunction("tile_tirx_add_one", true);
    expect(entry.has_value());
    if (!entry) { return; }

    tvm::Device cpu{kDLCPU, 0};
    auto input_tensor = tvm::runtime::Tensor::Empty(
        tvm::ffi::Shape{n}, DLDataType{kDLFloat, 32, 1}, cpu);
    auto output_tensor = tvm::runtime::Tensor::Empty(
        tvm::ffi::Shape{n}, DLDataType{kDLFloat, 32, 1}, cpu);
    luisa::vector<float> input_values(n);
    luisa::vector<float> output_values(n, 0.0f);
    for (auto index = 0u; index < input_values.size(); index++) {
        input_values[index] = static_cast<float>(index) * 0.25f - 1.0f;
    }
    input_tensor.CopyFromBytes(input_values.data(), input_values.size() * sizeof(float));
    (*entry)(input_tensor, output_tensor);
    output_tensor.CopyToBytes(output_values.data(), output_values.size() * sizeof(float));
    for (auto index = 0u; index < output_values.size(); index++) {
        expect(eq(output_values[index], input_values[index] + 1.0f));
    }

    // Aliasing is legal by default; noalias is an explicit CompileOptions
    // contract rather than an assumption made by the bridge.
    (*entry)(input_tensor, input_tensor);
    input_tensor.CopyToBytes(output_values.data(), output_values.size() * sizeof(float));
    for (auto index = 0u; index < output_values.size(); index++) {
        expect(eq(output_values[index], input_values[index] + 1.0f));
    }
}

void test_dsl_elementwise_end_to_end() {
    constexpr int64_t n = 17;
    auto definition = tile_kernel(
        "tile_dsl_axpy", [](TensorView<const float, 1> x,
                            TensorView<const float, 1> y,
                            TensorView<float, 1> result) {
            auto i = axis("i", result.extent<0>());
            for (auto &element : parallel(shape(i))) {
                auto index = element.index();
                result(index).store(x(index).load() + 2.0f * y(index).load());
            }
        });
    auto kernel = definition.capture(
        tensor_shape("x", n), tensor_shape("y", n), tensor_shape("result", n));
    expect(kernel.valid());
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    auto compilation = compile(std::move(native.value), "tile_dsl_axpy");
    expect(compilation.ok()) << compilation.error();
    if (!compilation) { return; }
    auto entry = compilation.module().value()->GetFunction("tile_dsl_axpy", true);
    expect(entry.has_value());
    if (!entry) { return; }

    tvm::Device cpu{kDLCPU, 0};
    auto x = tvm::runtime::Tensor::Empty(tvm::ffi::Shape{n}, DLDataType{kDLFloat, 32, 1}, cpu);
    auto y = tvm::runtime::Tensor::Empty(tvm::ffi::Shape{n}, DLDataType{kDLFloat, 32, 1}, cpu);
    auto result = tvm::runtime::Tensor::Empty(tvm::ffi::Shape{n}, DLDataType{kDLFloat, 32, 1}, cpu);
    luisa::vector<float> x_values(n);
    luisa::vector<float> y_values(n);
    luisa::vector<float> result_values(n, 0.0f);
    for (auto i = 0u; i < x_values.size(); i++) {
        x_values[i] = static_cast<float>(i);
        y_values[i] = static_cast<float>(3u * i + 1u);
    }
    x.CopyFromBytes(x_values.data(), x_values.size() * sizeof(float));
    y.CopyFromBytes(y_values.data(), y_values.size() * sizeof(float));
    (*entry)(x, y, result);
    result.CopyToBytes(result_values.data(), result_values.size() * sizeof(float));
    for (auto i = 0u; i < result_values.size(); i++) {
        expect(eq(result_values[i], x_values[i] + 2.0f * y_values[i]));
    }
}

void test_dsl_reduction_end_to_end() {
    constexpr int64_t rows = 5;
    constexpr int64_t columns = 7;
    auto definition = tile_kernel(
        "tile_dsl_row_sum", [](TensorView<const float, 2> source, TensorView<float, 1> result) {
            auto row = axis("row", source.extent<0>());
            auto column = axis("column", source.extent<1>());
            for (auto &row_nest : parallel(shape(row))) {
                auto sum = Scalar<float>{0.0f};
                for (auto &column_nest : row_nest.reduce(shape(column))) {
                    sum += source(row_nest[row], column_nest[column]).load();
                }
                result(row_nest[row]).store(sum);
            }
        });
    auto kernel = definition.capture(
        tensor_shape("input", rows, columns), tensor_shape("result", rows));
    expect(kernel.valid());
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    auto source_parameter = native.value->params[0u].as<tvm::tirx::BufferVar>();
    expect(source_parameter.has_value());
    tvm::tirx::PostOrderVisit(native.value->body, [&](const tvm::ffi::ObjectRef &node) {
        if (auto load = node.as<tvm::tirx::BufferLoad>()) {
            if (load.value()->buffer.name() == "input") {
                expect(source_parameter.has_value() && load.value()->buffer.same_as(source_parameter.value()));
            }
        }
    });
    auto compilation = compile(std::move(native.value), "tile_dsl_row_sum");
    expect(compilation.ok()) << compilation.error();
    if (!compilation) { return; }
    auto entry = compilation.module().value()->GetFunction("tile_dsl_row_sum", true);
    expect(entry.has_value());
    if (!entry) { return; }

    tvm::Device cpu{kDLCPU, 0};
    auto input = tvm::runtime::Tensor::Empty(
        tvm::ffi::Shape{rows, columns}, DLDataType{kDLFloat, 32, 1}, cpu);
    auto result = tvm::runtime::Tensor::Empty(
        tvm::ffi::Shape{rows}, DLDataType{kDLFloat, 32, 1}, cpu);
    luisa::vector<float> input_values(rows * columns);
    luisa::vector<float> result_values(rows, 0.0f);
    for (auto i = 0u; i < input_values.size(); i++) {
        input_values[i] = static_cast<float>((i % 9u) + 1u);
    }
    input.CopyFromBytes(input_values.data(), input_values.size() * sizeof(float));
    (*entry)(input, result);
    result.CopyToBytes(result_values.data(), result_values.size() * sizeof(float));
    for (auto row = 0u; row < static_cast<size_t>(rows); row++) {
        auto expected = 0.0f;
        for (auto column = 0u; column < static_cast<size_t>(columns); column++) {
            expected += input_values[row * columns + column];
        }
        expect(eq(result_values[row], expected));
    }
}

void test_dsl_masked_stencil_end_to_end() {
    constexpr int64_t n = 19;
    auto definition = tile_kernel(
        "tile_dsl_masked_stencil", [](TensorView<const float, 1> source, TensorView<float, 1> result) {
            auto i = axis("i", source.extent<0>());
            for (auto &element : parallel(shape(i))) {
                auto index = element[i];
                auto left_index = index - 1;
                auto right_index = index + 1;
                auto left_valid = (left_index >= 0) && (left_index < source.extent<0>());
                auto right_valid = (right_index >= 0) && (right_index < source.extent<0>());
                auto left = source(left_index).load(left_valid, 0.0f);
                auto center = source(index).load();
                auto right = source(right_index).load(right_valid, 0.0f);
                result(element[i]).store(left + 2.0f * center + right);
            }
        });
    auto kernel = definition.capture(tensor_shape("source", n), tensor_shape("result", n));
    expect(kernel.valid());
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    auto compilation = compile(std::move(native.value), "tile_dsl_masked_stencil");
    expect(compilation.ok()) << compilation.error();
    if (!compilation) { return; }
    auto entry = compilation.module().value()->GetFunction("tile_dsl_masked_stencil", true);
    expect(entry.has_value());
    if (!entry) { return; }

    tvm::Device cpu{kDLCPU, 0};
    auto source = tvm::runtime::Tensor::Empty(
        tvm::ffi::Shape{n}, DLDataType{kDLFloat, 32, 1}, cpu);
    auto result = tvm::runtime::Tensor::Empty(
        tvm::ffi::Shape{n}, DLDataType{kDLFloat, 32, 1}, cpu);
    luisa::vector<float> source_values(n);
    luisa::vector<float> result_values(n, 0.0f);
    for (auto i = 0u; i < source_values.size(); i++) {
        source_values[i] = static_cast<float>(i + 1u);
    }
    source.CopyFromBytes(source_values.data(), source_values.size() * sizeof(float));
    (*entry)(source, result);
    result.CopyToBytes(result_values.data(), result_values.size() * sizeof(float));
    for (auto i = 0u; i < result_values.size(); i++) {
        auto expected = 2.0f * source_values[i];
        if (i != 0u) { expected += source_values[i - 1u]; }
        if (i + 1u != source_values.size()) { expected += source_values[i + 1u]; }
        expect(std::abs(result_values[i] - expected) < 1e-6f);
    }
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_tirx_correspondence"_test = test_correspondence;
    "tile_tirx_coincident_replica_witnesses"_test = test_coincident_replica_witnesses;
    "tile_tirx_native_export"_test = test_native_export;
    "tile_tirx_native_compiler"_test = test_native_compiler;
    "tile_tirx_native_index_expressions"_test = test_native_index_expressions;
    "tile_tirx_native_buffer_kernel"_test = test_native_buffer_kernel;
    "tile_tirx_dsl_elementwise_end_to_end"_test = test_dsl_elementwise_end_to_end;
    "tile_tirx_dsl_reduction_end_to_end"_test = test_dsl_reduction_end_to_end;
    "tile_tirx_dsl_masked_stencil_end_to_end"_test = test_dsl_masked_stencil_end_to_end;
}
