// Test for integer shift and rotate lowering at the native SPIR-V boundary.
// This test covers:
// - mixed-width scalar shift operands permitted by SPIR-V
// - signed versus logical right-shift selection
// - modulo-normalized left/right rotation with wider shift counts

#include "ut/ut.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <spirv-tools/libspirv.hpp>

#include <luisa/dsl/sugar.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>

#include "spirv_codegen/dialect.h"
#include "spirv_codegen/entry.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void set_environment_variable(const char *name,
                              const char *value) noexcept {
#ifdef _WIN32
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
#endif
}

class ScopedEnvironmentVariable {
private:
    const char *_name;
    std::optional<std::string> _previous;

public:
    ScopedEnvironmentVariable(const char *name,
                              const char *value) noexcept
        : _name{name} {
        if (auto previous = std::getenv(name)) {
            _previous.emplace(previous);
        }
        set_environment_variable(name, value);
    }
    ~ScopedEnvironmentVariable() noexcept {
        set_environment_variable(
            _name, _previous ? _previous->c_str() : nullptr);
    }
    ScopedEnvironmentVariable(
        const ScopedEnvironmentVariable &) = delete;
    ScopedEnvironmentVariable &operator=(
        const ScopedEnvironmentVariable &) = delete;
};

template<typename T>
[[nodiscard]] luisa::compute::xir::Constant *make_constant(
    Module &module, T value) noexcept {
    return module.create_constant(Type::of<T>(), &value);
}

struct IntegerType {
    uint32_t width{0u};
    bool is_signed{false};

    [[nodiscard]] explicit operator bool() const noexcept {
        return width != 0u;
    }
};

struct ShiftSignature {
    spv::Op opcode{};
    uint32_t result_id{0u};
    uint32_t base_id{0u};
    uint32_t shift_id{0u};
    IntegerType result;
    IntegerType base;
    IntegerType shift;
};

struct IntegerConstant {
    uint32_t id{0u};
    IntegerType type;
    uint64_t bits{0u};
};

struct IntegerOperation {
    spv::Op opcode{};
    uint32_t result_id{0u};
    uint32_t lhs_id{0u};
    uint32_t rhs_id{0u};
};

struct IntegerBitcast {
    uint32_t result_id{0u};
    uint32_t operand_id{0u};
};

struct IntegerModuleFacts {
    std::vector<ShiftSignature> shifts;
    std::vector<IntegerConstant> constants;
    std::vector<IntegerOperation> operations;
    std::vector<IntegerBitcast> bitcasts;
    size_t untyped_shift_count{0u};
    size_t unsigned_64_mod_count{0u};
    size_t unsigned_64_sub_count{0u};
    size_t unsigned_32_or_count{0u};
};

[[nodiscard]] IntegerModuleFacts inspect_integer_module(
    const std::vector<uint32_t> &words) noexcept {
    std::unordered_map<uint32_t, IntegerType> integer_types;
    std::unordered_map<uint32_t, uint32_t> value_types;
    for (auto offset = size_t{5u}; offset < words.size();) {
        auto word_count = static_cast<size_t>(words[offset] >> 16u);
        if (word_count == 0u || word_count > words.size() - offset) {
            return {};
        }
        auto opcode = static_cast<spv::Op>(words[offset] & 0xffffu);
        if (opcode == spv::Op::OpTypeInt && word_count == 4u) {
            integer_types.emplace(
                words[offset + 1u],
                IntegerType{.width = words[offset + 2u],
                            .is_signed = words[offset + 3u] != 0u});
        }
        bool has_result = false;
        bool has_result_type = false;
        spv::HasResultAndType(
            opcode, &has_result, &has_result_type);
        if (has_result && has_result_type && word_count >= 3u) {
            value_types.emplace(words[offset + 2u],
                                words[offset + 1u]);
        }
        offset += word_count;
    }

    auto type_of_value = [&](uint32_t value) noexcept {
        auto value_iter = value_types.find(value);
        if (value_iter == value_types.end()) { return IntegerType{}; }
        auto type_iter = integer_types.find(value_iter->second);
        return type_iter == integer_types.end() ?
                   IntegerType{} :
                   type_iter->second;
    };
    auto type_of_id = [&](uint32_t type) noexcept {
        auto iter = integer_types.find(type);
        return iter == integer_types.end() ?
                   IntegerType{} :
                   iter->second;
    };

    IntegerModuleFacts facts;
    for (auto offset = size_t{5u}; offset < words.size();) {
        auto word_count = static_cast<size_t>(words[offset] >> 16u);
        if (word_count == 0u || word_count > words.size() - offset) {
            return {};
        }
        auto opcode = static_cast<spv::Op>(words[offset] & 0xffffu);
        if (opcode == spv::Op::OpConstant &&
            (word_count == 4u || word_count == 5u)) {
            auto type = type_of_id(words[offset + 1u]);
            if (type) {
                auto bits = static_cast<uint64_t>(words[offset + 3u]);
                if (word_count == 5u) {
                    bits |= static_cast<uint64_t>(
                                words[offset + 4u])
                            << 32u;
                }
                facts.constants.emplace_back(IntegerConstant{
                    .id = words[offset + 2u],
                    .type = type,
                    .bits = bits});
            }
        } else if ((opcode == spv::Op::OpShiftLeftLogical ||
                    opcode == spv::Op::OpShiftRightLogical ||
                    opcode == spv::Op::OpShiftRightArithmetic) &&
                   word_count == 5u) {
            auto signature = ShiftSignature{
                .opcode = opcode,
                .result_id = words[offset + 2u],
                .base_id = words[offset + 3u],
                .shift_id = words[offset + 4u],
                .result = type_of_id(words[offset + 1u]),
                .base = type_of_value(words[offset + 3u]),
                .shift = type_of_value(words[offset + 4u])};
            if (!signature.result || !signature.base ||
                !signature.shift) {
                facts.untyped_shift_count++;
            }
            facts.shifts.emplace_back(signature);
            facts.operations.emplace_back(IntegerOperation{
                .opcode = opcode,
                .result_id = words[offset + 2u],
                .lhs_id = words[offset + 3u],
                .rhs_id = words[offset + 4u]});
        } else if ((opcode == spv::Op::OpUMod ||
                    opcode == spv::Op::OpISub ||
                    opcode == spv::Op::OpBitwiseOr) &&
                   word_count == 5u) {
            facts.operations.emplace_back(IntegerOperation{
                .opcode = opcode,
                .result_id = words[offset + 2u],
                .lhs_id = words[offset + 3u],
                .rhs_id = words[offset + 4u]});
            auto result = type_of_id(words[offset + 1u]);
            if (opcode == spv::Op::OpUMod && result.width == 64u &&
                !result.is_signed) {
                facts.unsigned_64_mod_count++;
            } else if (opcode == spv::Op::OpISub &&
                       result.width == 64u && !result.is_signed) {
                facts.unsigned_64_sub_count++;
            } else if (opcode == spv::Op::OpBitwiseOr &&
                       result.width == 32u && !result.is_signed) {
                facts.unsigned_32_or_count++;
            }
        } else if (opcode == spv::Op::OpBitcast &&
                   word_count == 4u) {
            facts.bitcasts.emplace_back(IntegerBitcast{
                .result_id = words[offset + 2u],
                .operand_id = words[offset + 3u]});
        }
        offset += word_count;
    }
    return facts;
}

[[nodiscard]] uint32_t find_constant(
    const IntegerModuleFacts &facts, IntegerType type,
    uint64_t bits) noexcept {
    auto found = uint32_t{0u};
    for (auto &&constant : facts.constants) {
        if (constant.type.width == type.width &&
            constant.type.is_signed == type.is_signed &&
            constant.bits == bits) {
            if (found != 0u) { return 0u; }
            found = constant.id;
        }
    }
    return found;
}

[[nodiscard]] const IntegerOperation *find_operation(
    const IntegerModuleFacts &facts, uint32_t result_id) noexcept {
    for (auto &&operation : facts.operations) {
        if (operation.result_id == result_id) { return &operation; }
    }
    return nullptr;
}

[[nodiscard]] uint32_t strip_one_bitcast(
    const IntegerModuleFacts &facts, uint32_t value) noexcept {
    for (auto &&bitcast : facts.bitcasts) {
        if (bitcast.result_id == value) { return bitcast.operand_id; }
    }
    return value;
}

[[nodiscard]] bool shift_uses_direct_rotate_count(
    const IntegerModuleFacts &facts, const ShiftSignature &shift,
    uint32_t count_id, uint32_t width_id) noexcept {
    auto *mod = find_operation(facts, shift.shift_id);
    return mod != nullptr && mod->opcode == spv::Op::OpUMod &&
           strip_one_bitcast(facts, mod->lhs_id) == count_id &&
           mod->rhs_id == width_id;
}

[[nodiscard]] bool shift_uses_reverse_rotate_count(
    const IntegerModuleFacts &facts, const ShiftSignature &shift,
    uint32_t count_id, uint32_t width_id) noexcept {
    auto *reverse_mod = find_operation(facts, shift.shift_id);
    if (reverse_mod == nullptr ||
        reverse_mod->opcode != spv::Op::OpUMod) {
        return false;
    }
    auto *sub = find_operation(facts, reverse_mod->lhs_id);
    if (sub == nullptr || sub->opcode != spv::Op::OpISub) {
        return false;
    }
    auto *direct_mod = find_operation(facts, sub->rhs_id);
    return direct_mod != nullptr &&
           direct_mod->opcode == spv::Op::OpUMod &&
           strip_one_bitcast(facts, direct_mod->lhs_id) == count_id &&
           direct_mod->rhs_id == width_id &&
           sub->lhs_id == width_id && reverse_mod->rhs_id == width_id;
}

[[nodiscard]] bool has_exact_rotate_dataflow(
    const IntegerModuleFacts &facts, uint32_t value_id,
    uint32_t count_id, uint32_t width_id, spv::Op direct_opcode,
    spv::Op reverse_opcode) noexcept {
    const ShiftSignature *direct = nullptr;
    const ShiftSignature *reverse = nullptr;
    auto shift_count = size_t{0u};
    for (auto &&shift : facts.shifts) {
        if (shift.base_id != value_id) { continue; }
        shift_count++;
        if (shift.opcode == direct_opcode &&
            shift_uses_direct_rotate_count(
                facts, shift, count_id, width_id)) {
            if (direct != nullptr) { return false; }
            direct = &shift;
        }
        if (shift.opcode == reverse_opcode &&
            shift_uses_reverse_rotate_count(
                facts, shift, count_id, width_id)) {
            if (reverse != nullptr) { return false; }
            reverse = &shift;
        }
    }
    if (shift_count != 2u || direct == nullptr || reverse == nullptr) {
        return false;
    }
    auto or_count = size_t{0u};
    for (auto &&operation : facts.operations) {
        if (operation.opcode != spv::Op::OpBitwiseOr) { continue; }
        auto has_both_shifts =
            (operation.lhs_id == direct->result_id &&
             operation.rhs_id == reverse->result_id) ||
            (operation.lhs_id == reverse->result_id &&
             operation.rhs_id == direct->result_id);
        or_count += has_both_shifts ? 1u : 0u;
    }
    return or_count == 1u;
}

[[nodiscard]] size_t count_shift(
    const IntegerModuleFacts &facts, spv::Op opcode,
    IntegerType result, IntegerType base,
    IntegerType shift) noexcept {
    auto count = size_t{0u};
    for (auto &&candidate : facts.shifts) {
        count += candidate.opcode == opcode &&
                         candidate.result.width == result.width &&
                         candidate.result.is_signed == result.is_signed &&
                         candidate.base.width == base.width &&
                         candidate.base.is_signed == base.is_signed &&
                         candidate.shift.width == shift.width &&
                         candidate.shift.is_signed == shift.is_signed ?
                     1u :
                     0u;
    }
    return count;
}

[[nodiscard]] size_t count_shift_values(
    const IntegerModuleFacts &facts, spv::Op opcode,
    uint32_t base_id, uint32_t shift_id) noexcept {
    auto count = size_t{0u};
    for (auto &&candidate : facts.shifts) {
        count += candidate.opcode == opcode &&
                         candidate.base_id == base_id &&
                         candidate.shift_id == shift_id ?
                     1u :
                     0u;
    }
    return count;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "spirv_mixed_width_shifts_and_rotates_are_typed_exactly"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *unsigned_output = kernel->create_resource_argument(
            Type::buffer(Type::of<luisa::ulong>()));
        auto *signed_output = kernel->create_resource_argument(
            Type::buffer(Type::of<luisa::slong>()));
        auto *body = kernel->create_body_block();

        auto *wide_value = make_constant(
            module, luisa::ulong{0x0123456789abcdefull});
        auto *left_count = make_constant(module, uint32_t{5u});
        auto *right_count = make_constant(module, uint32_t{11u});
        auto *signed_value = make_constant(
            module, luisa::slong{-0x0123456789abcde});
        auto *arithmetic_count = make_constant(module, uint32_t{9u});
        auto *rotate_left_value = make_constant(
            module, uint32_t{0x12345678u});
        auto *rotate_left_count = make_constant(
            module, luisa::ulong{69u});
        auto *rotate_right_value = make_constant(
            module, uint32_t{0x89abcdefu});
        auto *rotate_right_count = make_constant(
            module, luisa::slong{77});

        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto binary = [&](const Type *type, ArithmeticOp op,
                          Value *lhs, Value *rhs) noexcept {
            std::array<Value *, 2u> operands{lhs, rhs};
            return builder.call(
                type, op,
                luisa::span<Value *const>{operands.data(), operands.size()});
        };
        auto *shift_left = binary(
            Type::of<luisa::ulong>(), ArithmeticOp::BINARY_SHIFT_LEFT,
            wide_value, left_count);
        auto *shift_right = binary(
            Type::of<luisa::ulong>(), ArithmeticOp::BINARY_SHIFT_RIGHT,
            wide_value, right_count);
        auto *arithmetic_right = binary(
            Type::of<luisa::slong>(), ArithmeticOp::BINARY_SHIFT_RIGHT,
            signed_value, arithmetic_count);
        auto *rotate_left = binary(
            Type::of<uint32_t>(), ArithmeticOp::BINARY_ROTATE_LEFT,
            rotate_left_value, rotate_left_count);
        auto *rotate_right = binary(
            Type::of<uint32_t>(), ArithmeticOp::BINARY_ROTATE_RIGHT,
            rotate_right_value, rotate_right_count);

        auto write = [&](Value *output, uint32_t index,
                         Value *value) noexcept {
            auto *index_value = make_constant(module, index);
            std::array<Value *, 3u> operands{output, index_value, value};
            builder.call(
                ResourceWriteOp::BUFFER_WRITE,
                luisa::span<Value *const>{operands.data(), operands.size()});
        };
        write(unsigned_output, 0u, shift_left);
        write(unsigned_output, 1u, shift_right);
        write(unsigned_output, 2u,
              builder.static_cast_(Type::of<luisa::ulong>(), rotate_left));
        write(unsigned_output, 3u,
              builder.static_cast_(Type::of<luisa::ulong>(), rotate_right));
        write(signed_output, 0u, arithmetic_right);
        builder.return_void();

        auto dialect =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(dialect.succeeded())
            << "mixed-width integer operations must be valid XIR at the "
               "native SPIR-V handoff";
        if (!dialect.succeeded()) { return; }

        Kernel1D ast_kernel = [](BufferULong, BufferSLong) noexcept {};
        kernel->set_block_size(
            ast_kernel.function()->function().block_size());
        ScopedEnvironmentVariable optimization_level{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        constexpr auto target_features =
            lc::spirv::SpirvTargetFeatures::from_enabled_mask(
                lc::spirv::target_feature::shader_int64);
        auto compiled =
            lc::spirv::SpirvCodegenEntry::compile_spirv_xir(
                ast_kernel.function()->function(), &module,
                ShaderOption{.enable_cache = false}, target_features);

        spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
        std::string diagnostics;
        tools.SetMessageConsumer(
            [&diagnostics](spv_message_level_t, const char *,
                           const spv_position_t &,
                           const char *message) {
                if (!diagnostics.empty()) { diagnostics.push_back('\n'); }
                diagnostics.append(message);
            });
        expect(tools.Validate(compiled.spv_bin.data(),
                              compiled.spv_bin.size()))
            << "mixed-width shift/rotate SPIR-V failed Vulkan validation: "
            << diagnostics;
        expect(eq(compiled.required_target_features,
                  lc::spirv::target_feature::shader_int64));

        auto facts = inspect_integer_module(compiled.spv_bin);
        expect(eq(facts.untyped_shift_count, 0u));
        constexpr IntegerType u32{.width = 32u, .is_signed = false};
        constexpr IntegerType u64{.width = 64u, .is_signed = false};
        constexpr IntegerType i64{.width = 64u, .is_signed = true};
        auto wide_value_id = find_constant(
            facts, u64, 0x0123456789abcdefull);
        auto left_count_id = find_constant(facts, u32, 5u);
        auto right_count_id = find_constant(facts, u32, 11u);
        auto signed_value_id = find_constant(
            facts, i64,
            static_cast<uint64_t>(luisa::slong{-0x0123456789abcde}));
        auto arithmetic_count_id = find_constant(facts, u32, 9u);
        auto rotate_left_value_id = find_constant(
            facts, u32, 0x12345678u);
        auto rotate_left_count_id = find_constant(facts, u64, 69u);
        auto rotate_right_value_id = find_constant(
            facts, u32, 0x89abcdefu);
        auto rotate_right_count_id = find_constant(facts, i64, 77u);
        auto rotate_width_id = find_constant(facts, u64, 32u);
        expect(static_cast<bool>(
            wide_value_id != 0u && left_count_id != 0u &&
            right_count_id != 0u && signed_value_id != 0u &&
            arithmetic_count_id != 0u &&
            rotate_left_value_id != 0u &&
            rotate_left_count_id != 0u &&
            rotate_right_value_id != 0u &&
            rotate_right_count_id != 0u && rotate_width_id != 0u))
            << "all distinct boundary literals must survive in the opt0 module";
        expect(eq(count_shift(
                      facts, spv::Op::OpShiftLeftLogical,
                      u64, u64, u32),
                  1u))
            << "direct uint64 << uint32 must preserve the uint32 shift type";
        expect(eq(count_shift(
                      facts, spv::Op::OpShiftRightLogical,
                      u64, u64, u32),
                  1u))
            << "direct uint64 >> uint32 must remain a logical shift";
        expect(eq(count_shift(
                      facts, spv::Op::OpShiftRightArithmetic,
                      i64, i64, u32),
                  1u))
            << "signed int64 >> uint32 must select arithmetic shift";
        expect(eq(count_shift_values(
                      facts, spv::Op::OpShiftLeftLogical,
                      wide_value_id, left_count_id),
                  1u));
        expect(eq(count_shift_values(
                      facts, spv::Op::OpShiftRightLogical,
                      wide_value_id, right_count_id),
                  1u));
        expect(eq(count_shift_values(
                      facts, spv::Op::OpShiftRightArithmetic,
                      signed_value_id, arithmetic_count_id),
                  1u));
        expect(eq(count_shift(
                      facts, spv::Op::OpShiftLeftLogical,
                      u32, u32, u64),
                  2u));
        expect(eq(count_shift(
                      facts, spv::Op::OpShiftRightLogical,
                      u32, u32, u64),
                  2u));
        expect(eq(facts.unsigned_64_mod_count, 4u))
            << "each rotate must reduce its forward and reverse shift modulo 32";
        expect(eq(facts.unsigned_64_sub_count, 2u));
        expect(eq(facts.unsigned_32_or_count, 2u));
        expect(has_exact_rotate_dataflow(
            facts, rotate_left_value_id, rotate_left_count_id,
            rotate_width_id,
            spv::Op::OpShiftLeftLogical,
            spv::Op::OpShiftRightLogical))
            << "rotate-left must use the reduced count for the left shift and "
               "the reverse count for the right shift";
        expect(has_exact_rotate_dataflow(
            facts, rotate_right_value_id, rotate_right_count_id,
            rotate_width_id,
            spv::Op::OpShiftRightLogical,
            spv::Op::OpShiftLeftLogical))
            << "rotate-right must use the reduced count for the right shift and "
               "the reverse count for the left shift";
        expect(eq(facts.shifts.size(), 7u))
            << "no extra or missing shift may hide a changed rotate lowering";
    };

    return 0;
}
