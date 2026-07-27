// Test for AST variable-usage propagation from bindless buffer writes.
// This covers generic, typed, uniform, typed-uniform, and cooperative-vector
// store operations.

#include "ut/ut.hpp"

#include <luisa/ast/function_builder.h>
#include <luisa/ast/op.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] Usage bindless_usage_for(CallOp op) noexcept {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto bindless_uid = uint32_t{};
    auto builder = FuncBuilder::define_kernel([&] {
        auto &current = *FuncBuilder::current();
        auto bindless = current.bindless_array();
        bindless_uid = bindless->variable().uid();
        auto zero = current.literal(Type::of<uint>(), 0u);
        auto value = current.literal(Type::of<uint>(), 42u);
        current.call(op, {bindless, zero, zero, value});
    });
    return Function{builder.get()}.variable_usage(bindless_uid);
}

[[nodiscard]] Usage bindless_cooperative_store_usage_for(
    CallOp op) noexcept {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto bindless_uid = uint32_t{};
    auto builder = FuncBuilder::define_kernel([&] {
        auto &current = *FuncBuilder::current();
        auto bindless = current.bindless_array();
        bindless_uid = bindless->variable().uid();
        auto buffer_handle = current.literal(Type::of<uint>(), 0u);
        auto offset = current.local(
            Type::cooperative_vector_ref(
                CoopRefVecType::FLOAT32, 8u));
        auto value = current.local(
            Type::cooperative_vector(Type::of<float>(), 8u));
        current.call(
            op, {bindless, buffer_handle, offset, value});
    });
    return Function{builder.get()}.variable_usage(bindless_uid);
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "generic_bindless_buffer_write_marks_write_usage"_test = [] {
        expect(bindless_usage_for(
                   CallOp::BINDLESS_BUFFER_WRITE) == Usage::WRITE);
    };
    "uniform_bindless_buffer_write_marks_write_usage"_test = [] {
        expect(bindless_usage_for(
                   CallOp::UNIFORM_BINDLESS_BUFFER_WRITE) == Usage::WRITE);
    };
    "typed_bindless_buffer_write_marks_write_usage"_test = [] {
        expect(bindless_usage_for(
                   CallOp::TYPED_BINDLESS_BUFFER_WRITE) == Usage::WRITE);
    };
    "typed_uniform_bindless_buffer_write_marks_write_usage"_test = [] {
        expect(bindless_usage_for(
                   CallOp::TYPED_UNIFORM_BINDLESS_BUFFER_WRITE) == Usage::WRITE);
    };
    "bindless_cooperative_vector_store_marks_write_usage"_test = [] {
        expect(bindless_cooperative_store_usage_for(
                   CallOp::BINDLESS_COOPERATIVE_VECTOR_STORE) == Usage::WRITE);
    };
    "typed_bindless_cooperative_vector_store_marks_write_usage"_test = [] {
        expect(bindless_cooperative_store_usage_for(
                   CallOp::TYPED_BINDLESS_COOPERATIVE_VECTOR_STORE) == Usage::WRITE);
    };
}
