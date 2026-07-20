// Exact floating-point constant coverage for the native SPIR-V builder.
// The disassembly round trip is intentional: SPIRV-Tools must preserve every
// payload, including FP8 encodings, NaN payloads, and signed zero.

#include "ut/ut.hpp"

#include <luisa/dsl/sugar.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <SPIRV/SpvBuilder.h>
#include <spirv-tools/libspirv.hpp>

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

struct ExpectedConstant {
    spv::Id type;
    spv::Id id;
    spv::Id repeated_id;
    spv::Id high_bits_id;
    uint64_t bits;
    uint32_t width;
};

struct RawFloatModule {
    std::vector<uint32_t> words;
    std::vector<ExpectedConstant> constants;
    std::array<spv::Id, 256u> e4m3_ids{};
    std::array<spv::Id, 256u> e5m2_ids{};
};

struct ParsedConstant {
    spv::Id type;
    std::vector<uint32_t> literal_words;
};

using ParsedConstants = std::unordered_map<spv::Id, ParsedConstant>;

struct FloatTypeSignature {
    uint32_t width{0u};
    std::optional<uint32_t> encoding;

    [[nodiscard]] bool operator==(
        const FloatTypeSignature &) const noexcept = default;
};

struct ParsedFloatConstant {
    FloatTypeSignature type;
    std::vector<uint32_t> literal_words;
};

[[nodiscard]] RawFloatModule make_raw_float_module() {
    spv::Builder builder{spv::Spv_1_5, 0u, nullptr};
    builder.setMemoryModel(spv::AddressingModel::Logical,
                           spv::MemoryModel::GLSL450);
    builder.addCapability(spv::Capability::Shader);
    builder.addCapability(spv::Capability::Float16);

    RawFloatModule module;
    module.constants.reserve(256u * 2u + 12u);

    auto add_constant = [&](spv::Id type, uint32_t width,
                            uint64_t bits) {
        auto high_bits = width == 64u ?
                             bits :
                             bits | (~0ull << width);
        // Feed the non-canonical form first so the emitted operand itself,
        // rather than only a later cache lookup, exercises width masking.
        auto id = builder.makeFpConstantFromBits(type, high_bits);
        auto repeated_id = builder.makeFpConstantFromBits(type, bits);
        auto high_bits_id = builder.makeFpConstantFromBits(type, high_bits);
        module.constants.emplace_back(ExpectedConstant{
            .type = type,
            .id = id,
            .repeated_id = repeated_id,
            .high_bits_id = high_bits_id,
            .bits = bits,
            .width = width});
        return id;
    };

    auto e4m3_type = builder.makeFloatE4M3Type();
    auto e5m2_type = builder.makeFloatE5M2Type();
    for (auto bits = 0u; bits < 256u; ++bits) {
        module.e4m3_ids[bits] = add_constant(e4m3_type, 8u, bits);
        module.e5m2_ids[bits] = add_constant(e5m2_type, 8u, bits);
    }

    auto float16_type = builder.makeFloatType(16u);
    for (auto bits : {0x0000ull, 0x8000ull, 0x7e55ull, 0x7d01ull}) {
        static_cast<void>(add_constant(float16_type, 16u, bits));
    }

    auto float32_type = builder.makeFloatType(32u);
    for (auto bits : {0x00000000ull, 0x80000000ull,
                      0x7fc12345ull, 0xffa54321ull}) {
        static_cast<void>(add_constant(float32_type, 32u, bits));
    }

    auto float64_type = builder.makeFloatType(64u);
    for (auto bits : {0x0000000000000000ull, 0x8000000000000000ull,
                      0x7ff8123456789abcull, 0xfff123456789abcdull}) {
        static_cast<void>(add_constant(float64_type, 64u, bits));
    }

    auto void_type = builder.makeVoidType();
    spv::Block *entry = nullptr;
    auto function = builder.makeFunctionEntry(
        spv::NoPrecision, void_type, "main", spv::LinkageType::Max,
        {}, {}, &entry);
    builder.addEntryPoint(spv::ExecutionModel::GLCompute, function, "main");
    builder.addExecutionMode(function, spv::ExecutionMode::LocalSize,
                             1, 1, 1);
    builder.enterFunction(function);
    builder.setBuildPoint(entry);
    builder.makeReturn(false);
    builder.leaveFunction();
    builder.dump(module.words);
    return module;
}

[[nodiscard]] ParsedConstants parse_constants(
    const std::vector<uint32_t> &words) {
    ParsedConstants constants;
    if (words.size() < 5u) { return constants; }
    for (std::size_t offset = 5u; offset < words.size();) {
        auto word_count = static_cast<std::size_t>(words[offset] >> 16u);
        if (word_count == 0u || offset + word_count > words.size()) {
            return {};
        }
        auto op = static_cast<spv::Op>(words[offset] & 0xffffu);
        if (op == spv::Op::OpConstant && word_count >= 4u) {
            auto type = words[offset + 1u];
            auto id = words[offset + 2u];
            std::vector<uint32_t> literal_words;
            literal_words.reserve(word_count - 3u);
            for (auto i = 3u; i < word_count; ++i) {
                literal_words.emplace_back(words[offset + i]);
            }
            constants.emplace(
                id, ParsedConstant{type, std::move(literal_words)});
        }
        offset += word_count;
    }
    return constants;
}

[[nodiscard]] std::vector<ParsedFloatConstant> parse_float_constants(
    const std::vector<uint32_t> &words) {
    std::unordered_map<spv::Id, FloatTypeSignature> float_types;
    if (words.size() < 5u) { return {}; }
    for (std::size_t offset = 5u; offset < words.size();) {
        auto word_count = static_cast<std::size_t>(words[offset] >> 16u);
        if (word_count == 0u || offset + word_count > words.size()) {
            return {};
        }
        auto op = static_cast<spv::Op>(words[offset] & 0xffffu);
        if (op == spv::Op::OpTypeFloat &&
            (word_count == 3u || word_count == 4u)) {
            FloatTypeSignature signature{.width = words[offset + 2u]};
            if (word_count == 4u) {
                signature.encoding = words[offset + 3u];
            }
            float_types.emplace(words[offset + 1u], signature);
        }
        offset += word_count;
    }

    std::vector<ParsedFloatConstant> constants;
    for (std::size_t offset = 5u; offset < words.size();) {
        auto word_count = static_cast<std::size_t>(words[offset] >> 16u);
        if (word_count == 0u || offset + word_count > words.size()) {
            return {};
        }
        auto op = static_cast<spv::Op>(words[offset] & 0xffffu);
        if (op == spv::Op::OpConstant && word_count >= 4u) {
            if (auto type = float_types.find(words[offset + 1u]);
                type != float_types.end()) {
                std::vector<uint32_t> literal_words;
                literal_words.reserve(word_count - 3u);
                for (auto i = 3u; i < word_count; ++i) {
                    literal_words.emplace_back(words[offset + i]);
                }
                constants.emplace_back(ParsedFloatConstant{
                    .type = type->second,
                    .literal_words = std::move(literal_words)});
            }
        }
        offset += word_count;
    }
    return constants;
}

[[nodiscard]] std::vector<uint32_t> expected_literal_words(
    uint64_t bits, uint32_t width) {
    auto low = static_cast<uint32_t>(bits & 0xffffffffull);
    if (width == 64u) {
        return {low, static_cast<uint32_t>(bits >> 32u)};
    }
    return {low};
}

[[nodiscard]] std::vector<uint32_t> expected_literal_words(
    const ExpectedConstant &constant) {
    return expected_literal_words(constant.bits, constant.width);
}

[[nodiscard]] bool has_exact_payload(
    const ParsedConstants &parsed,
    const ExpectedConstant &expected) {
    auto iter = parsed.find(expected.id);
    return iter != parsed.end() &&
           iter->second.type == expected.type &&
           iter->second.literal_words == expected_literal_words(expected);
}

[[nodiscard]] size_t count_exact_float_constant(
    const std::vector<ParsedFloatConstant> &constants,
    FloatTypeSignature type, uint64_t bits) {
    auto expected = expected_literal_words(bits, type.width);
    auto count = size_t{0u};
    for (auto &&constant : constants) {
        count += constant.type == type &&
                         constant.literal_words == expected ?
                     1u :
                     0u;
    }
    return count;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "spirv_raw_float_constants_preserve_exact_payloads"_test = [] {
        auto module = make_raw_float_module();

        std::unordered_set<spv::Id> unique_ids;
        unique_ids.reserve(module.constants.size());
        for (auto &&constant : module.constants) {
            expect(constant.id == constant.repeated_id)
                << "equal raw constants must be interned";
            expect(constant.id == constant.high_bits_id)
                << "bits above the declared width must be ignored";
            unique_ids.emplace(constant.id);
        }
        expect(unique_ids.size() == module.constants.size())
            << "distinct type/payload pairs must remain distinct";
        for (auto bits = 0u; bits < 256u; ++bits) {
            expect(module.e4m3_ids[bits] != module.e5m2_ids[bits])
                << "the two FP8 encodings must not share constants";
        }

        spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
        std::string diagnostics;
        tools.SetMessageConsumer(
            [&diagnostics](spv_message_level_t, const char *,
                           const spv_position_t &, const char *message) {
                if (!diagnostics.empty()) { diagnostics.push_back('\n'); }
                diagnostics.append(message);
            });
        expect(tools.Validate(module.words))
            << "raw-constant module failed Vulkan validation: "
            << diagnostics;

        auto initial_constants = parse_constants(module.words);
        expect(initial_constants.size() == module.constants.size())
            << "unexpected number of OpConstant instructions";
        for (auto &&constant : module.constants) {
            expect(has_exact_payload(initial_constants, constant))
                << "builder changed raw payload for result id "
                << constant.id;
        }

        std::string text;
        constexpr auto disassembly_options =
            SPV_BINARY_TO_TEXT_OPTION_NO_HEADER;
        expect(tools.Disassemble(module.words, &text,
                                 disassembly_options))
            << "failed to disassemble raw floating-point constants";
        expect(text.find("Float8E4M3EXT") != std::string::npos);
        expect(text.find("Float8E5M2EXT") != std::string::npos);

        std::vector<uint32_t> round_trip;
        expect(tools.Assemble(
            text, &round_trip,
            SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS))
            << "failed to reassemble raw floating-point constants";
        diagnostics.clear();
        expect(tools.Validate(round_trip))
            << "round-tripped module failed Vulkan validation: "
            << diagnostics;

        // Numeric IDs are preserved above, so exact per-ID comparison proves
        // that text disassembly/reassembly retained all 512 FP8 values and the
        // representative IEEE NaN/signed-zero payloads.
        auto round_trip_constants = parse_constants(round_trip);
        expect(round_trip_constants.size() == module.constants.size())
            << "text round trip changed the number of constants";
        for (auto &&constant : module.constants) {
            expect(has_exact_payload(round_trip_constants, constant))
                << "text round trip changed raw payload for result id "
                << constant.id;
        }
    };

    "spirv_xir_float_constants_preserve_exact_payloads"_test = [] {
        constexpr auto e4m3_encoding = static_cast<uint32_t>(
            spv::FPEncoding::Float8E4M3EXT);
        constexpr auto e5m2_encoding = static_cast<uint32_t>(
            spv::FPEncoding::Float8E5M2EXT);
        const auto *e4m3_type = Type::from("float8e4m3");
        const auto *e5m2_type = Type::from("float8e5m2");
        expect(e4m3_type != nullptr);
        expect(e5m2_type != nullptr);
        if (e4m3_type == nullptr || e5m2_type == nullptr) { return; }

        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);

        struct ExpectedXIRConstant {
            FloatTypeSignature type;
            uint64_t bits;
        };
        std::vector<ExpectedXIRConstant> expected;
        auto keep_constant = [&](const Type *type,
                                 FloatTypeSignature signature,
                                 uint64_t bits) noexcept {
            luisa::compute::xir::Constant *constant = nullptr;
            switch (type->size()) {
                case 1u: {
                    auto raw = static_cast<uint8_t>(bits);
                    constant = module.create_constant(type, &raw);
                    break;
                }
                case 2u: {
                    auto raw = static_cast<uint16_t>(bits);
                    constant = module.create_constant(type, &raw);
                    break;
                }
                case 4u: {
                    auto raw = static_cast<uint32_t>(bits);
                    constant = module.create_constant(type, &raw);
                    break;
                }
                case 8u: {
                    auto raw = bits;
                    constant = module.create_constant(type, &raw);
                    break;
                }
                default: break;
            }
            expect(constant != nullptr);
            if (constant == nullptr) { return; }
            auto *slot = builder.alloca_local(type);
            builder.store(slot, constant);
            expected.emplace_back(ExpectedXIRConstant{
                .type = signature,
                .bits = bits});
        };

        const FloatTypeSignature e4m3{
            .width = 8u, .encoding = e4m3_encoding};
        const FloatTypeSignature e5m2{
            .width = 8u, .encoding = e5m2_encoding};
        const FloatTypeSignature f16{.width = 16u};
        const FloatTypeSignature f32{.width = 32u};
        const FloatTypeSignature f64{.width = 64u};

        // Each format carries both signs of zero plus representative NaN
        // encodings; the IEEE formats also exercise distinct payloads. Raw
        // integer storage avoids host FP canonicalization before the XIR
        // constant is created.
        for (auto bits : {0x00ull, 0x80ull, 0x7full, 0xffull}) {
            keep_constant(e4m3_type, e4m3, bits);
        }
        for (auto bits : {0x00ull, 0x80ull, 0x7dull, 0xfeull}) {
            keep_constant(e5m2_type, e5m2, bits);
        }
        for (auto bits : {0x0000ull, 0x8000ull,
                          0x7e55ull, 0x7d01ull}) {
            keep_constant(Type::of<half>(), f16, bits);
        }
        for (auto bits : {0x00000000ull, 0x80000000ull,
                          0x7fc12345ull, 0xffa54321ull}) {
            keep_constant(Type::of<float>(), f32, bits);
        }
        for (auto bits : {0x0000000000000000ull,
                          0x8000000000000000ull,
                          0x7ff8123456789abcull,
                          0xfff123456789abcdull}) {
            keep_constant(Type::of<double>(), f64, bits);
        }
        builder.return_void();

        Kernel1D ast_kernel = []() noexcept {};
        kernel->set_block_size(
            ast_kernel.function()->function().block_size());
        constexpr auto required_features =
            lc::spirv::target_feature::shader_float8 |
            lc::spirv::target_feature::shader_float16 |
            lc::spirv::target_feature::shader_float64;
        constexpr auto target_features =
            lc::spirv::SpirvTargetFeatures::from_enabled_mask(
                required_features);
        ScopedEnvironmentVariable optimization_level{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        auto compiled =
            lc::spirv::SpirvCodegenEntry::compile_spirv_xir(
                ast_kernel.function()->function(), &module,
                ShaderOption{.enable_cache = false,
                             .enable_fast_math = false},
                target_features);

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
            << "exact-XIR floating constant module failed Vulkan 1.2 "
               "validation: "
            << diagnostics;
        expect(eq(compiled.required_target_features,
                  required_features))
            << "exact-XIR constants must record Float8, Float16, and "
               "Float64 requirements";

        auto constants = parse_float_constants(compiled.spv_bin);
        expect(eq(constants.size(), expected.size()))
            << "opt0 exact-XIR module must contain exactly the requested "
               "floating constants";
        for (auto &&constant : expected) {
            expect(eq(count_exact_float_constant(
                          constants, constant.type, constant.bits),
                      1u))
                << luisa::format(
                       "expected one {}-bit floating constant with raw bits "
                       "0x{:016x}",
                       constant.type.width, constant.bits);
        }
    };

    return 0;
}
