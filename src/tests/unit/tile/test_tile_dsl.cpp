// Tests for the execution-structure-first C++ Tile DSL capture surface.

#include "ut/ut.hpp"

#include <luisa/tile.h>

#include <concepts>
#include <type_traits>
#include <utility>

using namespace luisa;
using namespace luisa::compute::tile;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] Operation *only_root_operation(Kernel &kernel) noexcept {
    auto root = kernel.function().body().block(0u);
    return root->operation_count() == 1u ? root->operations().front() : nullptr;
}

template<typename T>
concept writable_element = requires(T element, Scalar<float> value) {
    element.store(value);
};

void test_lambda_signature_capture() {
    static_assert(!writable_element<ElementRef<const float>>);
    static_assert(writable_element<ElementRef<float>>);
    static_assert(std::same_as<decltype(std::declval<ElementRef<const float>>().load()), Scalar<float>>);
    static_assert(!std::is_convertible_v<ElementRef<float>, Scalar<float>>);
    static_assert(!std::is_convertible_v<ElementRef<const float>, Scalar<float>>);
    static_assert(!std::is_assignable_v<ElementRef<float> &, Scalar<float>>);
    static_assert(!std::is_assignable_v<ElementRef<float> &&, Scalar<float>>);
    static_assert(!std::is_assignable_v<ElementRef<float> &, float>);
    static_assert(!std::is_copy_assignable_v<ElementRef<float>>);
    static_assert(!std::is_move_assignable_v<ElementRef<float>>);
    static_assert(!std::is_assignable_v<ElementRef<float> &, ElementRef<const float>>);
    static_assert(std::is_copy_constructible_v<ElementRef<float>>);
    static_assert(std::is_move_constructible_v<ElementRef<float>>);

    auto captures = 0u;
    auto definition = tile_kernel(
        "signature_vector_add",
        [&](TensorView<const float, 1> a,
            TensorView<const float, 1> b,
            TensorView<float, 1> result) noexcept {
            captures++;
            auto i = axis("i", result.extent<0>());
            for (auto &element : parallel(shape(i))) {
                auto index = element.index();
                result(index).store(a(index).load() + b(index).load());
            }
        });
    expect(eq(captures, 0u));

    auto first = definition.capture(
        tensor_shape("a", 17u), tensor_shape("b", 17u), tensor_shape("result", 17u));
    auto second = definition.capture(tensor_shape(31u), tensor_shape(31u), tensor_shape(31u));
    expect(eq(captures, 2u));
    expect(first.valid());
    expect(second.valid());
    auto first_root = first.function().body().block(0u);
    auto second_root = second.function().body().block(0u);
    expect(eq(first_root->argument_count(), 3u));
    expect(eq(second_root->argument_count(), 3u));
    expect(first_root->argument(0u)->name() == "a");
    expect(first_root->argument(1u)->name() == "b");
    expect(first_root->argument(2u)->name() == "result");
    expect(second_root->argument(0u)->name() == "arg0");
    expect(second_root->argument(1u)->name() == "arg1");
    expect(second_root->argument(2u)->name() == "arg2");
    expect(eq(first_root->argument(0u)->type().index_space()->axis(0u).extent.constant_value(), 17u));
    expect(eq(second_root->argument(0u)->type().index_space()->axis(0u).extent.constant_value(), 31u));

    auto first_loop = only_root_operation(first);
    auto second_loop = only_root_operation(second);
    expect(first_loop != nullptr);
    expect(second_loop != nullptr);
    if (first_loop == nullptr || second_loop == nullptr) { return; }
    expect(eq(first_loop->domain()->axis(0u).extent.constant_value(), 17u));
    expect(eq(second_loop->domain()->axis(0u).extent.constant_value(), 31u));
    auto body = first_loop->region(0u)->block(0u);
    auto add = body->operation(2u);
    expect(add->operand(0u)->defining_operation()->operand(0u) == first_root->argument(0u));
    expect(add->operand(1u)->defining_operation()->operand(0u) == first_root->argument(1u));
    expect(body->operation(3u)->operand(0u) == first_root->argument(2u));
}

void test_elementwise_capture() {
    auto definition = tile_kernel(
        "vector_add", [](TensorView<const float, 1> a,
                         TensorView<const float, 1> b,
                         TensorView<float, 1> c) {
            auto i = axis("i", c.extent<0>());

            for (auto &element : parallel(shape(i))) {
                auto index = element.index();
                c(index).store(a(index).load() + b(index).load());
            }
        });
    auto kernel = definition.capture(
        tensor_shape("a", 17u), tensor_shape("b", 17u), tensor_shape("c", 17u));

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
    auto definition = tile_kernel(
        "row_sum", [](TensorView<const float, 2> a, TensorView<float, 1> result) {
            auto row = axis("row", a.extent<0>());
            auto column = axis("column", a.extent<1>());

            for (auto &row_nest : parallel(shape(row))) {
                auto sum = Scalar<float>{0.0f};
                for (auto &column_nest : row_nest.reduce(shape(column))) {
                    sum += a(row_nest[row], column_nest[column]).load();
                }
                result(row_nest[row]).store(sum);
            }
        });
    auto kernel = definition.capture(tensor_shape("a", 5u, 7u), tensor_shape("result", 5u));

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
    auto definition = tile_kernel("invalid_parallel_carry", [] {
        auto i = axis("i", 4u);
        auto value = Scalar<int32_t>{0};
        for (auto &element : parallel(shape(i))) {
            value += cast<int32_t>(element[i]);
        }
    });
    auto kernel = definition.capture();
    expect(!kernel.valid());
    expect(!kernel.diagnostics().empty());
}

void test_logical_and_masked_view_capture() {
    static_assert(!std::convertible_to<Scalar<bool>, bool>);
    auto definition = tile_kernel(
        "masked_stencil", [](TensorView<const float, 1> source, TensorView<float, 1> result) {
            auto i = axis("i", source.extent<0>());
            for (auto &element : parallel(shape(i))) {
                auto index = element[i];
                auto source_index = index - 1;
                auto in_bounds = (source_index >= 0) && (source_index < source.extent<0>());
                auto use_fallback = !in_bounds || (index < 0);
                result(index).store(source(source_index).load(!use_fallback, 0.0f));
            }
        });
    auto kernel = definition.capture(tensor_shape("source", 8u), tensor_shape("result", 8u));
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

void test_pipeline_policy() {
    auto capture = [](PipelinePolicy policy) {
        return tile_kernel("pipeline_policy", [=] {
                   for (auto &step : pipeline(shape(3), policy)) {
                       step.stage("producer");
                       step.stage("consumer");
                   }
               })
            .capture();
    };
    expect(capture({0u, 1u}).valid());
    expect(capture({1u, 3u}).valid());
    expect(!capture({2u, 0u}).valid());
    auto kernel = capture({2u, 1u});
    expect(kernel.valid());
    auto pipeline = only_root_operation(kernel);
    expect(pipeline != nullptr);
    if (pipeline != nullptr) {
        pipeline->set_attribute("stages", Attribute{int64_t{-1}});
        expect(!verify(kernel.module()).ok());
        pipeline->set_attribute("stages", Attribute{uint64_t{2u}});
        pipeline->set_attribute("initiation_interval", Attribute{uint64_t{1u} << 32u});
        expect(!verify(kernel.module()).ok());
    }
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_dsl_lambda_signature_and_fresh_specializations"_test = test_lambda_signature_capture;
    "tile_dsl_elementwise_capture"_test = test_elementwise_capture;
    "tile_dsl_reduction_capture"_test = test_reduction_capture_and_implicit_carry;
    "tile_dsl_rejects_parallel_scalar_carry"_test = test_parallel_cannot_capture_scalar_carry;
    "tile_dsl_logical_and_masked_view_capture"_test = test_logical_and_masked_view_capture;
    "tile_dsl_pipeline_policy_validation"_test = test_pipeline_policy;
}
