#include "ut/ut.hpp"

#include <luisa/ast/type.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/integer_alignment.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

template<typename T>
[[nodiscard]] Constant *make_constant(Module &module, T value) noexcept {
    return module.create_constant(Type::of<T>(), &value);
}

}// namespace

int main() {

    "integer_alignment_proves_runtime_soa_linear_form"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *capacity = kernel->create_value_argument(Type::of<uint>());
        auto *frame = kernel->create_value_argument(Type::of<uint>());
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *capacity_term = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_MUL,
            {capacity, make_constant(module, 12u)});
        auto *frame_term = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_MUL,
            {frame, make_constant(module, 4u)});
        auto *offset = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {capacity_term, frame_term});
        builder.return_void();

        expect(integer_value_guaranteed_alignment(offset, 16u) == 4u)
            << "capacity * 12 + frame * 4 is divisible by four for all inputs";
        expect(integer_value_guaranteed_alignment(capacity, 16u) == 1u)
            << "an unconstrained value argument has no fabricated alignment";
    };

    "integer_alignment_obeys_modular_transfer_rules"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *value = kernel->create_value_argument(Type::of<uint>());
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *times_six = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_MUL,
            {value, make_constant(module, 6u)});
        auto *plus_eight = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {times_six, make_constant(module, 8u)});
        auto *masked = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_BIT_AND,
            {value, make_constant(module, ~uint{7u})});
        auto *selected = builder.call(
            Type::of<uint>(), ArithmeticOp::SELECT,
            {masked, make_constant(module, 16u), condition});
        auto *shifted = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_SHIFT_LEFT,
            {value, make_constant(module, 3u)});
        builder.return_void();

        expect(integer_value_guaranteed_alignment(times_six, 16u) == 2u);
        expect(integer_value_guaranteed_alignment(plus_eight, 16u) == 2u);
        expect(integer_value_guaranteed_alignment(masked, 16u) == 8u);
        expect(integer_value_guaranteed_alignment(selected, 16u) == 8u);
        expect(integer_value_guaranteed_alignment(shifted, 16u) == 8u);
    };

    "integer_alignment_handles_zero_negative_casts_and_caps"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *zero = make_constant(module, uint64_t{0u});
        auto *negative_eight = make_constant(module, int32_t{-8});
        auto *cast = builder.cast_(
            Type::of<uint64_t>(), CastOp::STATIC_CAST,
            negative_eight);
        builder.return_void();

        expect(integer_value_guaranteed_alignment(zero, 12u) == 8u)
            << "a non-power-of-two cap is rounded down before analysis";
        expect(integer_value_guaranteed_alignment(negative_eight, 16u) == 8u);
        expect(integer_value_guaranteed_alignment(cast, 16u) == 8u)
            << "integer extension preserves low zero bits";
        expect(integer_value_guaranteed_alignment(cast, 0u) == 1u);
    };

    "integer_alignment_does_not_self_prove_cyclic_phi"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *entry = kernel->create_body_block();
        auto *loop = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        builder.br(loop);
        builder.set_insertion_point(loop);
        auto *phi = builder.phi(
            Type::of<uint>(), {{make_constant(module, 0u), entry}});
        auto *next = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {phi, make_constant(module, 4u)});
        phi->add_incoming(next, loop);
        builder.br(loop);

        expect(integer_value_guaranteed_alignment(phi, 16u) == 1u)
            << "recursive facts require a dedicated fixed-point proof";
    };

    return 0;
}
