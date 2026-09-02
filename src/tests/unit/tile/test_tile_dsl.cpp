// Tests for the execution-structure-first C++ Tile DSL capture surface.

#include "ut/ut.hpp"

#include <luisa/tile.h>

#include <concepts>

using namespace luisa;
using namespace luisa::compute::tile;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] Operation *only_root_operation(Kernel &kernel) noexcept {
    auto root = kernel.function().body().block(0u);
    return root->operation_count() == 1u ? root->operations().front() : nullptr;
}

void test_elementwise_capture() {
    auto kernel = define("vector_add", [] {
        auto i = axis("i", 17u);
        auto a = input<float>("a", shape(i));
        auto b = input<float>("b", shape(i));
        auto c = output<float>("c", shape(i));

        for (auto &element : parallel(shape(i))) {
            c[element] = a[element].load() + b[element].load();
        }
    });

    expect(kernel.valid());
    auto root = kernel.function().body().block(0u);
    expect(eq(root->argument_count(), 3u));
    expect(root->argument(0u)->name() == "a");
    expect(root->argument(1u)->name() == "b");
    expect(root->argument(2u)->name() == "c");

    auto loop = only_root_operation(kernel);
    expect(loop != nullptr);
    if (loop == nullptr) { return; }
    expect(loop->kind() == OperationKind::PARALLEL);
    expect(loop->domain().has_value());
    expect(eq(loop->domain()->rank(), 1u));
    expect(eq(loop->operand_count(), 0u));
    expect(eq(loop->result_count(), 0u));
    auto body = loop->region(0u)->block(0u);
    expect(eq(body->argument_count(), 1u));
    expect(eq(body->operation_count(), 5u));
    expect(body->operation(0u)->kind() == OperationKind::VIEW_LOAD);
    expect(body->operation(1u)->kind() == OperationKind::VIEW_LOAD);
    expect(body->operation(2u)->kind() == OperationKind::ELEMENTWISE);
    expect(body->operation(2u)->elementwise_op() == ElementwiseOp::ADD);
    expect(body->operation(3u)->kind() == OperationKind::VIEW_STORE);
    expect(body->operation(4u)->kind() == OperationKind::YIELD);
}

void test_reduction_capture_and_implicit_carry() {
    auto kernel = define("row_sum", [] {
        auto row = axis("row", 5u);
        auto column = axis("column", 7u);
        auto a = input<float>("a", shape(row, column));
        auto result = output<float>("result", shape(row));

        for (auto &row_nest : parallel(shape(row))) {
            auto sum = Scalar<float>{0.0f};
            for (auto &column_nest : row_nest.reduce(shape(column))) {
                sum += a(row_nest[row], column_nest[column]).load();
            }
            result(row_nest[row]) = sum;
        }
    });

    expect(kernel.valid());
    auto outer = only_root_operation(kernel);
    expect(outer != nullptr);
    if (outer == nullptr) { return; }
    auto outer_body = outer->region(0u)->block(0u);
    expect(eq(outer_body->operation_count(), 4u));
    expect(outer_body->operation(0u)->kind() == OperationKind::CONSTANT);
    auto reduction = outer_body->operation(1u);
    expect(reduction->kind() == OperationKind::REDUCE);
    expect(eq(reduction->operand_count(), 1u));
    expect(eq(reduction->result_count(), 1u));
    expect(reduction->operand(0u) == outer_body->operation(0u)->result(0u));

    auto reduction_body = reduction->region(0u)->block(0u);
    expect(eq(reduction_body->argument_count(), 2u));
    expect(reduction_body->argument(0u)->type().kind() == TypeKind::INDEX);
    expect(reduction_body->argument(1u)->type() == Type::scalar(ScalarType::FLOAT32));
    expect(eq(reduction_body->operation_count(), 3u));
    auto add = reduction_body->operation(1u);
    expect(add->kind() == OperationKind::ELEMENTWISE);
    expect(add->elementwise_op() == ElementwiseOp::ADD);
    expect(add->operand(0u) == reduction_body->argument(1u));
    auto yield = reduction_body->operation(2u);
    expect(yield->kind() == OperationKind::YIELD);
    expect(yield->operand(0u) == add->result(0u));

    auto store = outer_body->operation(2u);
    expect(store->kind() == OperationKind::VIEW_STORE);
    expect(store->operand(store->operand_count() - 1u) == reduction->result(0u));
    expect(outer_body->operation(3u)->kind() == OperationKind::YIELD);
}

void test_parallel_cannot_capture_scalar_carry() {
    auto kernel = define("invalid_parallel_carry", [] {
        auto i = axis("i", 4u);
        auto value = Scalar<int32_t>{0};
        for (auto &element : parallel(shape(i))) {
            value += cast<int32_t>(element[i]);
        }
    });
    expect(!kernel.valid());
    expect(!kernel.diagnostics().empty());
}

void test_logical_and_masked_view_capture() {
    static_assert(!std::convertible_to<Scalar<bool>, bool>);
    auto kernel = define("masked_stencil", [] {
        auto i = axis("i", 8u);
        auto source = input<float>("source", shape(i));
        auto result = output<float>("result", shape(i));
        for (auto &element : parallel(shape(i))) {
            auto index = element[i];
            auto source_index = index - 1;
            auto in_bounds = (source_index >= 0) && (source_index < 8);
            auto use_fallback = !in_bounds || (index < 0);
            result[element] = source(source_index).load(!use_fallback, 0.0f);
        }
    });
    expect(kernel.valid());
    auto loop = only_root_operation(kernel);
    expect(loop != nullptr);
    if (loop == nullptr) { return; }
    auto body = loop->region(0u)->block(0u);
    Operation *load = nullptr;
    size_t logical_count = 0u;
    for (auto operation : body->operations()) {
        if (operation->kind() == OperationKind::VIEW_LOAD) { load = operation; }
        if (operation->kind() == OperationKind::ELEMENTWISE &&
            (operation->elementwise_op() == ElementwiseOp::LOGICAL_AND ||
             operation->elementwise_op() == ElementwiseOp::LOGICAL_OR ||
             operation->elementwise_op() == ElementwiseOp::LOGICAL_NOT)) {
            logical_count++;
        }
    }
    expect(load != nullptr);
    if (load != nullptr) {
        expect(eq(load->operand_count(), 4u));
        expect(load->operand(2u)->type() == Type::scalar(ScalarType::BOOL));
        expect(load->operand(3u)->type() == Type::scalar(ScalarType::FLOAT32));
    }
    expect(eq(logical_count, 4u));
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_dsl_elementwise_capture"_test = test_elementwise_capture;
    "tile_dsl_reduction_capture"_test = test_reduction_capture_and_implicit_carry;
    "tile_dsl_rejects_parallel_scalar_carry"_test = test_parallel_cannot_capture_scalar_carry;
    "tile_dsl_logical_and_masked_view_capture"_test = test_logical_and_masked_view_capture;
}
