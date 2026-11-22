// Host-only regression for SoA boolean element groups. An SoA group of
// sub-word elements shares one 32-bit word, and boolean lanes occupy one byte
// each (0x00/0x01) in ascending byte order. That packing must be expressed with
// integer widening and shifts: a bitwise cast that involves a boolean value is
// not valid XIR because no backend can reinterpret a register-level boolean
// pattern, so a boolean group must never lower to one.
#include "ut/ut.hpp"

#include <luisa/luisa-compute.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

struct SoABoolPackMatrix {
    float3x3 m;
    float2x2 n;
};

LUISA_STRUCT(SoABoolPackMatrix, m, n) {};

struct SoABoolPackFixture {
    float3 a;
    bool2 b;
    bool c;
    SoABoolPackMatrix d;
    std::array<std::array<int4, 1>, 1> e;
};

LUISA_STRUCT(SoABoolPackFixture, a, b, c, d, e) {};

namespace {

struct SoALowering {
    size_t shift_left_count{0u};
    size_t shift_right_count{0u};
    size_t bool_byte_shift_count{0u};
    size_t boolean_bitwise_cast_count{0u};
};

[[nodiscard]] SoALowering inspect_soa_lowering(const Module &module) noexcept {
    SoALowering lowering;
    auto is_boolean = [](const Type *type) noexcept {
        return type != nullptr && type->is_bool_or_bool_vector();
    };
    for (auto function : module.function_list()) {
        auto definition = function->definition();
        if (definition == nullptr) { continue; }
        definition->traverse_instructions([&](const Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>()) {
                auto arithmetic = static_cast<const ArithmeticInst *>(inst);
                auto counter = arithmetic->op() == ArithmeticOp::BINARY_SHIFT_LEFT ?
                                   &lowering.shift_left_count :
                               arithmetic->op() == ArithmeticOp::BINARY_SHIFT_RIGHT ?
                                   &lowering.shift_right_count :
                                   nullptr;
                if (counter == nullptr) { return; }
                ++*counter;
                // A whole-byte shift is how one boolean lane reaches its own
                // byte inside the shared 32-bit word.
                auto amount = uint64_t{0u};
                if (arithmetic->operand_count() == 2u &&
                    try_decode_constant_nonnegative_integer(arithmetic->operand(1u), amount) &&
                    amount != 0u && amount % 8u == 0u) {
                    lowering.bool_byte_shift_count++;
                }
            } else if (inst->isa<CastInst>()) {
                auto cast = static_cast<const CastInst *>(inst);
                if (cast->op() == xir::CastOp::BITWISE_CAST &&
                    (is_boolean(cast->type()) ||
                     (cast->value() != nullptr && is_boolean(cast->value()->type())))) {
                    lowering.boolean_bitwise_cast_count++;
                }
            }
        });
    }
    return lowering;
}

[[nodiscard]] SoALowering expect_valid_soa_module(const compute::Function &kernel) noexcept {
    SoALowering lowering;
    auto module = ast_to_xir_translate(kernel, {});
    if (module == nullptr) {
        expect(false) << "AST-to-XIR translation returned a null module.";
        return lowering;
    }
    auto verification = xir_verify_module(module.get());
    if (!verification.succeeded()) {
        expect(false) << verification.errors.front().message;
        return lowering;
    }
    expect(verification.succeeded());
    lowering = inspect_soa_lowering(*module);
    expect(lowering.boolean_bitwise_cast_count == 0u)
        << "a boolean SoA group must not lower to a bitwise cast";
    return lowering;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "soa_struct_write_packs_boolean_lanes_into_bytes"_test = [] {
        Kernel1D kernel = [](SOAVar<SoABoolPackFixture> soa,
                             BufferVar<SoABoolPackFixture> source) noexcept {
            auto i = dispatch_x();
            soa.write(i, source.read(i));
        };
        compute::Function ast_function{kernel.function().get()};
        auto lowering = expect_valid_soa_module(ast_function);
        // `bool2 b` contributes one byte shift for its second lane; the scalar
        // `bool c` stays in byte zero.
        expect(lowering.shift_left_count >= 1u);
        expect(lowering.bool_byte_shift_count >= 1u);
    };

    "soa_struct_read_unpacks_boolean_lanes_from_bytes"_test = [] {
        Kernel1D kernel = [](SOAVar<SoABoolPackFixture> soa,
                             BufferVar<SoABoolPackFixture> target) noexcept {
            auto i = dispatch_x();
            target.write(i, soa.read(i));
        };
        compute::Function ast_function{kernel.function().get()};
        auto lowering = expect_valid_soa_module(ast_function);
        // Both boolean members are recovered from their own byte.
        expect(lowering.shift_right_count >= 1u);
        expect(lowering.bool_byte_shift_count >= 1u);
    };

    return 0;
}
