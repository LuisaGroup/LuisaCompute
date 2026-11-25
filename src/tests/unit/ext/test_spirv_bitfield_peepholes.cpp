// Regression tests for the native SPIR-V integer bitfield decode peepholes.
//
// The `q8_0`/`w4` weight decoders sign-extend a packed code with the exact
// two's-complement idiom `cast<int>(((word >> k) & mask) ^ half) - half`, and
// the `bf16` decoder writes `(word & 0xFFFFu) << 16u`. Both lower to chains of
// SPIR-V bitwise instructions unless the code generator recognizes them:
//
// * `((x >> k) & mask) ^ half) - half` -> `OpBitFieldSExtract(bitcast(x), k, w)`
// * `(x & mask) << k` -> `x << k` when `mask << k == ~0u << k`
//
// The tests below pin the exact dataflow of every fold, plus the negative cases
// that must keep the generic emission path: a mask that does not cover the bits
// the shift keeps, a sign bit that does not match the mask width, a field that
// does not fit in 32 bits, and a compensation constant that is off by one.
// Every compiled module is validated with SPIRV-Tools.
#include "ut/ut.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <spirv-tools/libspirv.hpp>
#include <luisa/dsl/sugar.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/verifier.h>
#include <luisa/core/stl/optional.h>
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
    luisa::optional<std::string> _previous;

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

[[nodiscard]] bool validates(const std::vector<uint32_t> &words) noexcept {
    spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
    std::string diagnostics;
    tools.SetMessageConsumer(
        [&diagnostics](spv_message_level_t, const char *, const spv_position_t &,
                       const char *message) {
            if (!diagnostics.empty()) { diagnostics.push_back('\n'); }
            diagnostics.append(message == nullptr ? "" : message);
        });
    if (tools.Validate(words.data(), words.size())) { return true; }
    std::string text;
    tools.Disassemble(words.data(), words.size(), &text);
    // Diagnostics go to stderr because this helper runs outside the assertion
    // context; the caller still fails the test with `expect(validates(...))`.
    std::fprintf(stderr, "SPIR-V validation failed:\n%s\n%s\n",
                 diagnostics.c_str(), text.c_str());
    return false;
}

[[nodiscard]] std::string disassemble(const std::vector<uint32_t> &words) noexcept {
    spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
    std::string text;
    tools.Disassemble(words.data(), words.size(), &text);
    return text;
}

struct BinaryRecord {
    spv::Op opcode{spv::Op::OpNop};
    uint32_t result_id{0u};
    uint32_t lhs_id{0u};
    uint32_t rhs_id{0u};
};

struct BitfieldExtractRecord {
    uint32_t result_id{0u};
    uint32_t result_type{0u};
    uint32_t base_id{0u};
    uint32_t offset{0u};
    uint32_t count{0u};
};

struct ModuleFacts {
    std::vector<BitfieldExtractRecord> extracts;
    std::vector<BinaryRecord> binaries;
    // id -> raw 32-bit constant bits
    std::unordered_map<uint32_t, uint32_t> constants;
    // OpBitcast result id -> operand id
    std::unordered_map<uint32_t, uint32_t> bitcast_operand;
    // result id -> result type id
    std::unordered_map<uint32_t, uint32_t> result_types;
    // result type id -> (component count, signed, bit width)
    struct IntType {
        uint32_t components{0u};
        bool is_signed{false};
        uint32_t width{0u};
    };
    std::unordered_map<uint32_t, IntType> int_types;
    size_t bitwise_and_count{0u};
    size_t shift_left_count{0u};
    // OpStore object ids: the values the program actually writes.
    std::vector<uint32_t> stored_values;
};

[[nodiscard]] ModuleFacts inspect(const std::vector<uint32_t> &words) noexcept {
    ModuleFacts facts;
    for (auto offset = size_t{5u}; offset < words.size();) {
        auto word_count = static_cast<size_t>(words[offset] >> 16u);
        if (word_count == 0u || word_count > words.size() - offset) {
            return {};
        }
        auto opcode = static_cast<spv::Op>(words[offset] & 0xffffu);
        auto has_result = false;
        auto has_result_type = false;
        spv::HasResultAndType(opcode, &has_result, &has_result_type);
        if (has_result && has_result_type && word_count >= 3u) {
            facts.result_types.emplace(words[offset + 2u], words[offset + 1u]);
        }
        if (opcode == spv::Op::OpTypeInt && word_count == 4u) {
            facts.int_types.emplace(
                words[offset + 1u],
                ModuleFacts::IntType{
                    .components = 1u,
                    .is_signed = words[offset + 3u] != 0u,
                    .width = words[offset + 2u]});
        } else if (opcode == spv::Op::OpTypeVector && word_count == 4u) {
            auto component = facts.int_types.find(words[offset + 2u]);
            if (component != facts.int_types.end()) {
                facts.int_types.emplace(
                    words[offset + 1u],
                    ModuleFacts::IntType{
                        .components = words[offset + 3u],
                        .is_signed = component->second.is_signed,
                        .width = component->second.width});
            }
        } else if (opcode == spv::Op::OpConstant && word_count == 4u) {
            facts.constants.emplace(words[offset + 2u], words[offset + 3u]);
        } else if (opcode == spv::Op::OpBitcast && word_count == 4u) {
            facts.bitcast_operand.emplace(words[offset + 2u], words[offset + 3u]);
        } else if (opcode == spv::Op::OpBitFieldSExtract && word_count == 6u) {
            facts.extracts.emplace_back(BitfieldExtractRecord{
                .result_id = words[offset + 2u],
                .result_type = words[offset + 1u],
                .base_id = words[offset + 3u],
                .offset = [&] {
                    auto iter = facts.constants.find(words[offset + 4u]);
                    return iter == facts.constants.end() ? ~0u : iter->second;
                }(),
                .count = [&] {
                    auto iter = facts.constants.find(words[offset + 5u]);
                    return iter == facts.constants.end() ? ~0u : iter->second;
                }()});
        } else if ((opcode == spv::Op::OpShiftLeftLogical ||
                    opcode == spv::Op::OpShiftRightLogical ||
                    opcode == spv::Op::OpBitwiseAnd ||
                    opcode == spv::Op::OpBitwiseXor ||
                    opcode == spv::Op::OpIAdd ||
                    opcode == spv::Op::OpISub) &&
                   word_count == 5u) {
            facts.binaries.emplace_back(BinaryRecord{
                .opcode = opcode,
                .result_id = words[offset + 2u],
                .lhs_id = words[offset + 3u],
                .rhs_id = words[offset + 4u]});
            facts.bitwise_and_count += opcode == spv::Op::OpBitwiseAnd ? 1u : 0u;
            facts.shift_left_count += opcode == spv::Op::OpShiftLeftLogical ? 1u : 0u;
        } else if (opcode == spv::Op::OpStore && word_count >= 3u) {
            facts.stored_values.emplace_back(words[offset + 2u]);
        }
        offset += word_count;
    }
    return facts;
}

[[nodiscard]] size_t count_opcode(const ModuleFacts &facts,
                                  spv::Op opcode) noexcept {
    auto count = size_t{0u};
    for (auto &&binary : facts.binaries) {
        count += binary.opcode == opcode ? 1u : 0u;
    }
    return count;
}

// `OpShiftLeftLogical` by `amount` whose base is an `OpBitwiseAnd` result, i.e.
// a `(x & mask) << amount` that the redundant-mask fold did NOT remove.
[[nodiscard]] size_t count_and_fed_shift_left(const ModuleFacts &facts,
                                              uint32_t amount) noexcept {
    auto count = size_t{0u};
    for (auto &&binary : facts.binaries) {
        if (binary.opcode != spv::Op::OpShiftLeftLogical) { continue; }
        auto amount_iter = facts.constants.find(binary.rhs_id);
        if (amount_iter == facts.constants.end() ||
            amount_iter->second != amount) {
            continue;
        }
        for (auto &&producer : facts.binaries) {
            if (producer.opcode == spv::Op::OpBitwiseAnd &&
                producer.result_id == binary.lhs_id) {
                count++;
                break;
            }
        }
    }
    return count;
}

[[nodiscard]] const ModuleFacts::IntType *int_type_of(
    const ModuleFacts &facts, uint32_t value_id) noexcept {
    auto type_iter = facts.result_types.find(value_id);
    if (type_iter == facts.result_types.end()) { return nullptr; }
    auto iter = facts.int_types.find(type_iter->second);
    return iter == facts.int_types.end() ? nullptr : &iter->second;
}

// `OpShiftLeftLogical` whose result is a `<components x i32>`/`<components x
// u32>` vector, i.e. a vector shift that the redundant-mask fold kept.
[[nodiscard]] size_t count_vector_shift_left(const ModuleFacts &facts,
                                             uint32_t components) noexcept {
    auto count = size_t{0u};
    for (auto &&binary : facts.binaries) {
        if (binary.opcode != spv::Op::OpShiftLeftLogical) { continue; }
        auto type_iter = facts.result_types.find(binary.result_id);
        if (type_iter == facts.result_types.end()) { continue; }
        auto int_iter = facts.int_types.find(type_iter->second);
        if (int_iter == facts.int_types.end()) { continue; }
        count += int_iter->second.components == components &&
                         int_iter->second.width == 32u ?
                     1u : 0u;
    }
    return count;
}

// Builds one straight-line kernel covering every signed-decode sub-pattern plus
// the negative cases. Input `packed` is a uint32 buffer; outputs are a uint32
// buffer for the shift fold and an int32 buffer for the sign-extension folds.
[[nodiscard]] std::vector<uint32_t> compile_scalar_patterns(uint32_t opt_level) {
    Module module;
    auto *uint_type = Type::of<uint32_t>();
    auto *int_type = Type::of<int32_t>();
    auto *kernel = module.create_kernel();
    auto *packed = kernel->create_resource_argument(Type::buffer(uint_type));
    auto *shifted = kernel->create_resource_argument(Type::buffer(uint_type));
    auto *extracted = kernel->create_resource_argument(Type::buffer(int_type));
    auto *body = kernel->create_body_block();
    XIRBuilder builder;
    builder.set_insertion_point(body);
    auto read = [&](uint32_t index) noexcept {
        return builder.call(
            uint_type, ResourceReadOp::BUFFER_READ,
            {packed, make_constant(module, index)});
    };
    auto write_uint = [&](uint32_t index, Value *value) noexcept {
        builder.call(
            ResourceWriteOp::BUFFER_WRITE,
            {shifted, make_constant(module, index), value});
    };
    auto write_int = [&](uint32_t index, Value *value) noexcept {
        builder.call(
            ResourceWriteOp::BUFFER_WRITE,
            {extracted, make_constant(module, index), value});
    };
    auto binary = [&](const Type *type, ArithmeticOp op,
                      Value *lhs, Value *rhs) noexcept {
        std::array<Value *, 2u> operands{lhs, rhs};
        return builder.call(
            type, op, luisa::span<Value *const>{operands.data(), operands.size()});
    };
    // The exact `qwen_q8_0_to_f32` / `qwen_w4_to_f32` decode expression:
    // `cast<int>(((word >> k) & mask) ^ half) - half`.
    // 1. `(x & 0xFFFFu) << 16u` (bf16 low half) must lose the mask.
    write_uint(
        0u, binary(uint_type, ArithmeticOp::BINARY_SHIFT_LEFT,
                   binary(uint_type, ArithmeticOp::BINARY_BIT_AND, read(0u),
                          make_constant(module, 0xFFFFu)),
                   make_constant(module, 16u)));
    // 2. `((x & 0xFFu) ^ 0x80u) - 128` -> OpBitFieldSExtract(x, 0, 8).
    write_int(
        1u, binary(int_type, ArithmeticOp::BINARY_SUB,
                   builder.static_cast_(
                       int_type,
                       binary(uint_type, ArithmeticOp::BINARY_BIT_XOR,
                              binary(uint_type, ArithmeticOp::BINARY_BIT_AND,
                                     read(1u), make_constant(module, 0xFFu)),
                              make_constant(module, 0x80u))),
                   make_constant(module, int32_t{128})));
    // 3. `(((x >> 8u) & 0xFFu) ^ 0x80u) - 128` -> extract(x, 8, 8).
    write_int(
        2u, binary(int_type, ArithmeticOp::BINARY_SUB,
                   builder.static_cast_(
                       int_type,
                       binary(uint_type, ArithmeticOp::BINARY_BIT_XOR,
                              binary(uint_type, ArithmeticOp::BINARY_BIT_AND,
                                     binary(uint_type, ArithmeticOp::BINARY_SHIFT_RIGHT,
                                            read(2u), make_constant(module, 8u)),
                                     make_constant(module, 0xFFu)),
                              make_constant(module, 0x80u))),
                   make_constant(module, int32_t{128})));
    // 4. The same field in the `+ (-128)` form that XIR normalization emits.
    write_int(
        3u, binary(int_type, ArithmeticOp::BINARY_ADD,
                   builder.static_cast_(
                       int_type,
                       binary(uint_type, ArithmeticOp::BINARY_BIT_XOR,
                              binary(uint_type, ArithmeticOp::BINARY_BIT_AND,
                                     binary(uint_type, ArithmeticOp::BINARY_SHIFT_RIGHT,
                                            read(3u), make_constant(module, 8u)),
                                     make_constant(module, 0xFFu)),
                              make_constant(module, 0x80u))),
                   make_constant(module, int32_t{-128})));
    // 5. `((x >> 24u) ^ 0x80u) - 128` (top byte without a mask) -> extract(24, 8).
    write_int(
        4u, binary(int_type, ArithmeticOp::BINARY_SUB,
                   builder.static_cast_(
                       int_type,
                       binary(uint_type, ArithmeticOp::BINARY_BIT_XOR,
                              binary(uint_type, ArithmeticOp::BINARY_SHIFT_RIGHT,
                                     read(4u), make_constant(module, 24u)),
                              make_constant(module, 0x80u))),
                   make_constant(module, int32_t{128})));
    // 6. A 4-bit signed field `((x >> 4u) & 0xFu ^ 0x8u) - 8` -> extract(4, 4).
    write_int(
        5u, binary(int_type, ArithmeticOp::BINARY_SUB,
                   builder.static_cast_(
                       int_type,
                       binary(uint_type, ArithmeticOp::BINARY_BIT_XOR,
                              binary(uint_type, ArithmeticOp::BINARY_BIT_AND,
                                     binary(uint_type, ArithmeticOp::BINARY_SHIFT_RIGHT,
                                            read(5u), make_constant(module, 4u)),
                                     make_constant(module, 0xFu)),
                              make_constant(module, 0x8u))),
                   make_constant(module, int32_t{8})));
    // 7. A 20-bit field ending exactly at bit 31 -> extract(12, 20).
    write_int(
        6u, binary(int_type, ArithmeticOp::BINARY_SUB,
                   builder.static_cast_(
                       int_type,
                       binary(uint_type, ArithmeticOp::BINARY_BIT_XOR,
                              binary(uint_type, ArithmeticOp::BINARY_BIT_AND,
                                     binary(uint_type, ArithmeticOp::BINARY_SHIFT_RIGHT,
                                            read(6u), make_constant(module, 12u)),
                                     make_constant(module, 0xFFFFFu)),
                              make_constant(module, 0x80000u))),
                   make_constant(module, int32_t{0x80000})));
    // 8. NEGATIVE: `(x & 0xFFu) << 16u` keeps bits the shift keeps only in
    // part, so the mask is not redundant and must stay.
    write_uint(
        7u, binary(uint_type, ArithmeticOp::BINARY_SHIFT_LEFT,
                   binary(uint_type, ArithmeticOp::BINARY_BIT_AND, read(7u),
                          make_constant(module, 0xFFu)),
                   make_constant(module, 16u)));
    // 9. NEGATIVE: sign bit 0x40 with an 0xFF mask is not a signed field.
    write_int(
        8u, binary(int_type, ArithmeticOp::BINARY_SUB,
                   builder.static_cast_(
                       int_type,
                       binary(uint_type, ArithmeticOp::BINARY_BIT_XOR,
                              binary(uint_type, ArithmeticOp::BINARY_BIT_AND,
                                     read(8u), make_constant(module, 0xFFu)),
                              make_constant(module, 0x40u))),
                   make_constant(module, int32_t{64})));
    // 10. NEGATIVE: off-by-one compensation constant.
    write_int(
        9u, binary(int_type, ArithmeticOp::BINARY_SUB,
                   builder.static_cast_(
                       int_type,
                       binary(uint_type, ArithmeticOp::BINARY_BIT_XOR,
                              binary(uint_type, ArithmeticOp::BINARY_BIT_AND,
                                     read(9u), make_constant(module, 0xFFu)),
                              make_constant(module, 0x80u))),
                   make_constant(module, int32_t{129})));
    // 11. The duplicated mask the Qwen decoders actually emit:
    // `(((x & 0xFFu) & 0xFFu) ^ 0x80u) - 128` -> extract(x, 0, 8) and both
    // masks must disappear.
    write_int(
        11u, binary(int_type, ArithmeticOp::BINARY_SUB,
                    builder.static_cast_(
                        int_type,
                        binary(uint_type, ArithmeticOp::BINARY_BIT_XOR,
                               binary(uint_type, ArithmeticOp::BINARY_BIT_AND,
                                      binary(uint_type, ArithmeticOp::BINARY_BIT_AND,
                                             read(11u), make_constant(module, 0xFFu)),
                                      make_constant(module, 0xFFu)),
                               make_constant(module, 0x80u))),
                    make_constant(module, int32_t{128})));
    // 12. NEGATIVE: offset 25 + count 8 does not fit in 32 bits.
    write_int(
        10u, binary(int_type, ArithmeticOp::BINARY_SUB,
                    builder.static_cast_(
                        int_type,
                        binary(uint_type, ArithmeticOp::BINARY_BIT_XOR,
                               binary(uint_type, ArithmeticOp::BINARY_BIT_AND,
                                      binary(uint_type, ArithmeticOp::BINARY_SHIFT_RIGHT,
                                             read(10u), make_constant(module, 25u)),
                                      make_constant(module, 0xFFu)),
                               make_constant(module, 0x80u))),
                    make_constant(module, int32_t{128})));
    builder.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto dialect = lc::spirv::validate_spirv_xir_codegen_dialect(&module);
    expect(dialect.succeeded())
        << (dialect.diagnostics.empty() ?
                "SPIR-V handoff dialect validation failed" :
                dialect.diagnostics.front().message.c_str());
    Kernel1D ast_kernel = [](BufferUInt, BufferUInt, BufferInt) noexcept {};
    kernel->set_block_size(ast_kernel.function()->function().block_size());
    auto level = std::to_string(opt_level);
    ScopedEnvironmentVariable optimization_level{"LUISA_SPIRV_OPT_LEVEL", level.c_str()};
    ScopedEnvironmentVariable pass_override{"LUISA_SPIRV_OPT_PASSES", nullptr};
    auto compiled = lc::spirv::SpirvCodegenEntry::compile_spirv_xir(
        ast_kernel.function()->function(), &module,
        ShaderOption{.enable_cache = false},
        lc::spirv::SpirvTargetFeatures{});
    return compiled.spv_bin;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "spirv_signed_decode_folds_into_bitfield_extract"_test = [] {
        // Optimization is disabled so the assertions observe exactly the
        // emission peepholes instead of any SPIRV-Tools rewrite.
        auto words = compile_scalar_patterns(0u);
        expect(validates(words));
        auto facts = inspect(words);
        // All seven signed decode patterns of the fixture are positive: (0,8)
        // single mask, (0,8) duplicated mask, (8,8) SUB, (8,8) ADD, (24,8),
        // (4,4) and (12,20).
        expect(eq(facts.extracts.size(), size_t{7u}))
            << "every signed decode field must fold into one "
               "OpBitFieldSExtract:\n"
            << disassemble(words);
        auto expect_extract = [&](uint32_t offset, uint32_t count,
                                  size_t expected_matches) noexcept {
            auto matches = size_t{0u};
            for (auto &&extract : facts.extracts) {
                matches += extract.offset == offset && extract.count == count ? 1u : 0u;
            }
            expect(eq(matches, expected_matches))
                << "expected " << expected_matches << " OpBitFieldSExtract("
                << offset << ", " << count << "):\n"
                << disassemble(words);
        };
        expect_extract(0u, 8u, 2u);  // single mask and duplicated mask
        // The `- half` and the `+ (-half)` spellings decode the same field.
        expect_extract(8u, 8u, 2u);
        expect_extract(24u, 8u, 1u);
        expect_extract(4u, 4u, 1u);
        expect_extract(12u, 20u, 1u);
        // Every extract must have a signed 32-bit result type and a bit-cast
        // base of the same shape, and its Offset/Count must be 32-bit scalar
        // constants (already guaranteed by `inspect`, which resolves them).
        for (auto &&extract : facts.extracts) {
            auto type_iter = facts.int_types.find(extract.result_type);
            expect(type_iter != facts.int_types.end())
                << "OpBitFieldSExtract result type must be a 32-bit integer:\n"
                << disassemble(words);
            if (type_iter == facts.int_types.end()) { continue; }
            expect(type_iter->second.width == 32u && type_iter->second.is_signed)
                << "OpBitFieldSExtract must produce a signed 32-bit value:\n"
                << disassemble(words);
            expect(facts.bitcast_operand.contains(extract.base_id))
                << "the extract base must be the uint32 source bit-cast to "
                   "int32:\n"
                << disassemble(words);
            auto operand_iter = facts.bitcast_operand.find(extract.base_id);
            auto base_type = operand_iter == facts.bitcast_operand.end() ?
                                 nullptr :
                                 int_type_of(facts, operand_iter->second);
            expect(base_type != nullptr &&
                   base_type->width == 32u && !base_type->is_signed)
                << "the extract must consume the original unsigned word:\n"
                << disassemble(words);
        }
        // The folded patterns must not leave their XOR, AND or subtract behind;
        // the three rejected patterns keep exactly one OpBitwiseXor each.
        expect(eq(count_opcode(facts, spv::Op::OpBitwiseXor), size_t{3u}))
            << "only the rejected patterns may keep OpBitwiseXor:\n"
            << disassemble(words);
        // Three of the four negative patterns keep a `- half` subtraction.
        expect(eq(count_opcode(facts, spv::Op::OpISub), size_t{3u}))
            << "only the rejected patterns may keep OpISub:\n"
            << disassemble(words);
        // The four rejected patterns (non-redundant mask, wrong sign bit,
        // off-by-one compensation, field past bit 31) keep one OpBitwiseAnd
        // each; the duplicated mask of the real Qwen decode folds completely
        // and pattern 8 is the only `(x & mask) << 16` that may survive.
        expect(eq(facts.bitwise_and_count, size_t{4u}))
            << "only the rejected patterns may keep OpBitwiseAnd:\n"
            << disassemble(words);
        expect(eq(facts.shift_left_count, size_t{2u}))
            << "both << 16 forms must still be emitted exactly once:\n"
            << disassemble(words);
        expect(eq(count_and_fed_shift_left(facts, 16u), size_t{1u}))
            << "the bf16 `(x & 0xFFFF) << 16` mask must be gone while the "
               "non-redundant `(x & 0xFF) << 16` mask stays:\n"
            << disassemble(words);
    };

    "spirv_bitfield_peepholes_survive_the_default_compute_preset"_test = [] {
        auto words = compile_scalar_patterns(2u);
        expect(validates(words));
        auto facts = inspect(words);
        expect(eq(facts.extracts.size(), size_t{7u}))
            << "the compute preset must not undo the bitfield folds:\n"
            << disassemble(words);
    };

    "spirv_signed_decode_folds_vector_fields_with_scalar_offset_and_count"_test = [] {
        Module module;
        auto *uint_type = Type::of<uint32_t>();
        auto *int_type = Type::of<int32_t>();
        auto *uint4_type = Type::vector(uint_type, 4u);
        auto *int4_type = Type::vector(int_type, 4u);
        auto make_splat_uint4 = [&](uint32_t value) noexcept {
            std::array<uint32_t, 4u> data{value, value, value, value};
            return module.create_constant(uint4_type, data.data());
        };
        auto make_splat_int4 = [&](int32_t value) noexcept {
            std::array<int32_t, 4u> data{value, value, value, value};
            return module.create_constant(int4_type, data.data());
        };
        auto *kernel = module.create_kernel();
        auto *packed = kernel->create_resource_argument(Type::buffer(uint_type));
        auto *output = kernel->create_resource_argument(Type::buffer(int_type));
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        std::array<Value *, 4u> lanes{
            builder.call(uint_type, ResourceReadOp::BUFFER_READ,
                         {packed, make_constant(module, 0u)}),
            builder.call(uint_type, ResourceReadOp::BUFFER_READ,
                         {packed, make_constant(module, 1u)}),
            builder.call(uint_type, ResourceReadOp::BUFFER_READ,
                         {packed, make_constant(module, 2u)}),
            builder.call(uint_type, ResourceReadOp::BUFFER_READ,
                         {packed, make_constant(module, 3u)})};
        auto *vector = builder.call(
            uint4_type, ArithmeticOp::AGGREGATE,
            luisa::span<Value *const>{lanes.data(), lanes.size()});
        auto *shifted = builder.call(
            uint4_type, ArithmeticOp::BINARY_SHIFT_RIGHT,
            {vector, make_splat_uint4(16u)});
        auto *masked = builder.call(
            uint4_type, ArithmeticOp::BINARY_BIT_AND,
            {shifted, make_splat_uint4(0xFFFFu)});
        auto *xored = builder.call(
            uint4_type, ArithmeticOp::BINARY_BIT_XOR,
            {masked, make_splat_uint4(0x8000u)});
        auto *signed_vector = builder.cast_(int4_type, xir::CastOp::STATIC_CAST, xored);
        auto *decoded = builder.call(
            int4_type, ArithmeticOp::BINARY_ADD,
            {signed_vector, make_splat_int4(-0x8000)});
        auto *lane = builder.call(
            int_type, ArithmeticOp::EXTRACT,
            {decoded, make_constant(module, 0u)});
        builder.call(
            ResourceWriteOp::BUFFER_WRITE,
            {output, make_constant(module, 0u), lane});
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto dialect = lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(dialect.succeeded())
        << (dialect.diagnostics.empty() ?
                "SPIR-V handoff dialect validation failed" :
                dialect.diagnostics.front().message.c_str());
        Kernel1D ast_kernel = [](BufferUInt, BufferInt) noexcept {};
        kernel->set_block_size(ast_kernel.function()->function().block_size());
        ScopedEnvironmentVariable optimization_level{"LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable pass_override{"LUISA_SPIRV_OPT_PASSES", nullptr};
        auto compiled = lc::spirv::SpirvCodegenEntry::compile_spirv_xir(
            ast_kernel.function()->function(), &module,
            ShaderOption{.enable_cache = false},
            lc::spirv::SpirvTargetFeatures{});
        expect(validates(compiled.spv_bin));
        auto facts = inspect(compiled.spv_bin);
        expect(eq(facts.extracts.size(), size_t{1u}))
            << "a 4-component signed decode must fold into one vector "
               "OpBitFieldSExtract:\n"
            << disassemble(compiled.spv_bin);
        if (facts.extracts.size() == 1u) {
            expect(eq(facts.extracts.front().offset, 16u));
            expect(eq(facts.extracts.front().count, 16u));
            auto type_iter = facts.int_types.find(facts.extracts.front().result_type);
            expect(type_iter != facts.int_types.end() &&
                   type_iter->second.components == 4u &&
                   type_iter->second.width == 32u &&
                   type_iter->second.is_signed)
                << "a vector fold must keep the <4 x i32> result type:\n"
                << disassemble(compiled.spv_bin);
        }
        expect(eq(facts.bitwise_and_count, size_t{0u}))
            << "the vector mask must be folded away:\n"
            << disassemble(compiled.spv_bin);
        expect(eq(count_opcode(facts, spv::Op::OpBitwiseXor), size_t{0u}))
            << "the vector xor must be folded away:\n"
            << disassemble(compiled.spv_bin);
    };
    "spirv_bitfield_peepholes_edge_cases_preserve_shared_nodes_and_fold_deep_chains"_test = [] {
        Module module;
        auto *uint_type = Type::of<uint32_t>();
        auto *int_type = Type::of<int32_t>();
        auto *uint4_type = Type::vector(uint_type, 4u);
        auto *kernel = module.create_kernel();
        auto *packed = kernel->create_resource_argument(Type::buffer(uint_type));
        auto *shifted = kernel->create_resource_argument(Type::buffer(uint_type));
        auto *extracted = kernel->create_resource_argument(Type::buffer(int_type));
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto read = [&](uint32_t index) noexcept {
            return builder.call(
                uint_type, ResourceReadOp::BUFFER_READ,
                {packed, make_constant(module, index)});
        };
        auto write_uint = [&](uint32_t index, Value *value) noexcept {
            builder.call(ResourceWriteOp::BUFFER_WRITE,
                         {shifted, make_constant(module, index), value});
        };
        auto write_int = [&](uint32_t index, Value *value) noexcept {
            builder.call(ResourceWriteOp::BUFFER_WRITE,
                         {extracted, make_constant(module, index), value});
        };
        auto binary = [&](const Type *type, ArithmeticOp op,
                          Value *lhs, Value *rhs) noexcept {
            std::array<Value *, 2u> operands{lhs, rhs};
            return builder.call(
                type, op, luisa::span<Value *const>{operands.data(), operands.size()});
        };
        auto masked = [&](Value *value, uint32_t bits) noexcept {
            return binary(uint_type, ArithmeticOp::BINARY_BIT_AND, value,
                          make_constant(module, bits));
        };
        auto decoded = [&](Value *value) noexcept {
            return builder.static_cast_(
                int_type,
                binary(uint_type, ArithmeticOp::BINARY_BIT_XOR, value,
                       make_constant(module, 0x80u)));
        };
        auto sign_extend = [&](Value *value, int32_t half) noexcept {
            return binary(int_type, ArithmeticOp::BINARY_SUB, value,
                          make_constant(module, half));
        };
        // (a) `s = read(0u) >> 8u` has two readers: the signed decode below and
        // a direct store. The decode still folds onto the source word, but `s`
        // itself must stay emitted or the store would name a dropped id.
        auto *shared_shift = binary(uint_type, ArithmeticOp::BINARY_SHIFT_RIGHT,
                                    read(0u), make_constant(module, 8u));
        write_uint(0u, shared_shift);
        write_int(0u, sign_extend(decoded(masked(shared_shift, 0xFFu)),
                                  int32_t{128}));
        // (b) The mask duplication the Qwen decoders emit, five hops away from
        // the `- half` consumer.
        auto *deep_shift = binary(uint_type, ArithmeticOp::BINARY_SHIFT_RIGHT,
                                  read(1u), make_constant(module, 8u));
        write_int(1u, sign_extend(decoded(masked(masked(deep_shift, 0xFFu), 0xFFu)),
                                  int32_t{128}));
        // (c) The commuted `(-128) + cast<int>(...)` spelling: the compensation
        // constant sits on the left of the add.
        write_int(2u, binary(int_type, ArithmeticOp::BINARY_ADD,
                             make_constant(module, int32_t{-128}),
                             decoded(masked(read(2u), 0xFFu))));
        // (d) The vector bf16 decode `(v & 0xFFFFu) << 16u`: the mask is
        // redundant, the shift is not.
        std::array<Value *, 4u> lanes{read(3u), read(4u), read(5u), read(6u)};
        std::array<uint32_t, 4u> low_half{0xFFFFu, 0xFFFFu, 0xFFFFu, 0xFFFFu};
        std::array<uint32_t, 4u> high_shift{16u, 16u, 16u, 16u};
        auto *lanes_value = builder.call(
            uint4_type, ArithmeticOp::AGGREGATE,
            luisa::span<Value *const>{lanes.data(), lanes.size()});
        auto *vector_masked = binary(
            uint4_type, ArithmeticOp::BINARY_BIT_AND, lanes_value,
            module.create_constant(uint4_type, low_half.data()));
        auto *vector_shifted = binary(
            uint4_type, ArithmeticOp::BINARY_SHIFT_LEFT, vector_masked,
            module.create_constant(uint4_type, high_shift.data()));
        write_uint(1u, builder.call(uint_type, ArithmeticOp::EXTRACT,
                                    {vector_shifted, make_constant(module, 0u)}));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto dialect = lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(dialect.succeeded())
            << (dialect.diagnostics.empty() ?
                    "SPIR-V handoff dialect validation failed" :
                    dialect.diagnostics.front().message.c_str());
        Kernel1D ast_kernel = [](BufferUInt, BufferUInt, BufferInt) noexcept {};
        kernel->set_block_size(ast_kernel.function()->function().block_size());
        ScopedEnvironmentVariable optimization_level{"LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable pass_override{"LUISA_SPIRV_OPT_PASSES", nullptr};
        auto compiled = lc::spirv::SpirvCodegenEntry::compile_spirv_xir(
            ast_kernel.function()->function(), &module,
            ShaderOption{.enable_cache = false},
            lc::spirv::SpirvTargetFeatures{});
        auto words = compiled.spv_bin;
        expect(validates(words));
        auto facts = inspect(words);
        // (a), (b) and (c) each fold into exactly one signed extract.
        expect(eq(facts.extracts.size(), size_t{3u}))
            << "the shared, deep and commuted decodes must all fold:\n"
            << disassemble(words);
        auto count_extract = [&](uint32_t offset, uint32_t count) noexcept {
            auto matches = size_t{0u};
            for (auto &&extract : facts.extracts) {
                matches += extract.offset == offset && extract.count == count ? 1u : 0u;
            }
            return matches;
        };
        expect(eq(count_extract(8u, 8u), size_t{2u}))
            << "(a) and (b) both decode the byte at bit 8:\n"
            << disassemble(words);
        expect(eq(count_extract(0u, 8u), size_t{1u}))
            << "(c) decodes the low byte with the constant on the left:\n"
            << disassemble(words);
        // Every fold must consume its own source word: `read(0)` for (a),
        // `read(1)` for (b) and `read(2)` for (c).
        std::vector<uint32_t> sources;
        for (auto &&extract : facts.extracts) {
            auto operand_iter = facts.bitcast_operand.find(extract.base_id);
            expect(operand_iter != facts.bitcast_operand.end())
                << "every folded extract must bit-cast its unsigned source:\n"
                << disassemble(words);
            if (operand_iter != facts.bitcast_operand.end()) {
                sources.emplace_back(operand_iter->second);
            }
        }
        auto distinct_sources = sources.size() == size_t{3u};
        for (auto i = 0u; i < sources.size(); ++i) {
            for (auto j = i + 1u; j < sources.size(); ++j) {
                distinct_sources = distinct_sources && sources[i] != sources[j];
            }
        }
        expect(distinct_sources)
            << "each fold must read its own source word:\n"
            << disassemble(words);
        // (a) `s` keeps exactly one logical right shift because a second reader
        // needs it, and that id must be the value of the direct store: a
        // declared-but-unused id would validate and still be a codegen bug.
        auto surviving_shifts = size_t{0u};
        auto surviving_shift = uint32_t{0u};
        for (auto &&record : facts.binaries) {
            if (record.opcode != spv::Op::OpShiftRightLogical) { continue; }
            auto amount_iter = facts.constants.find(record.rhs_id);
            if (amount_iter == facts.constants.end() ||
                amount_iter->second != 8u) {
                continue;
            }
            surviving_shift = record.result_id;
            surviving_shifts++;
        }
        expect(eq(surviving_shifts, size_t{1u}))
            << "only the multi-use intermediate `>> 8` may survive:\n"
            << disassemble(words);
        auto directly_stored = false;
        for (auto value_id : facts.stored_values) {
            directly_stored = directly_stored || value_id == surviving_shift;
        }
        expect(directly_stored)
            << "the surviving shared shift must feed the direct store:\n"
            << disassemble(words);
        // (b) A folded decode leaves no mask, xor or subtraction behind, so the
        // whole module must be free of all three.
        expect(eq(facts.bitwise_and_count, size_t{0u}))
            << "every mask must be folded away:\n"
            << disassemble(words);
        expect(eq(count_opcode(facts, spv::Op::OpBitwiseXor), size_t{0u}))
            << "every decode xor must be folded away:\n"
            << disassemble(words);
        expect(eq(count_opcode(facts, spv::Op::OpISub), size_t{0u}))
            << "every decode subtraction must be folded away:\n"
            << disassemble(words);
        // (d) The redundant vector mask is gone while the shift survives.
        expect(eq(count_vector_shift_left(facts, 4u), size_t{1u}))
            << "the vector `<< 16` must survive its folded mask:\n"
            << disassemble(words);
    };
}