// Test for volatile direct-buffer accesses in native XIR-to-SPIR-V.
// This test covers:
// - fixed-point propagation of exact per-argument coherence requirements
// - Coherent decoration only on the volatile buffer descriptor
// - Volatile load/store operands, matching device fences, and Vulkan validation

#include "ut/ut.hpp"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <spirv-tools/libspirv.hpp>

#include <luisa/dsl/sugar.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/verifier.h>

#include "spirv_codegen/argument_usage.h"
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

[[nodiscard]] std::string decode_literal_string(
    const std::vector<uint32_t> &words,
    size_t begin, size_t end) {
    std::string text;
    for (auto index = begin; index < end; ++index) {
        auto word = words[index];
        for (auto byte = 0u; byte < 4u; ++byte) {
            auto c = static_cast<char>(
                (word >> (byte * 8u)) & 0xffu);
            if (c == '\0') { return text; }
            text.push_back(c);
        }
    }
    return text;
}

struct SpirvVolatileFacts {
    std::unordered_map<std::string, uint32_t> named_ids;
    std::unordered_set<uint32_t> coherent_ids;
    size_t volatile_loads{0u};
    size_t volatile_stores{0u};
    size_t memory_barriers{0u};
    bool structurally_valid{true};
};

[[nodiscard]] SpirvVolatileFacts inspect_volatile_facts(
    const std::vector<uint32_t> &words) {
    SpirvVolatileFacts facts;
    constexpr auto volatile_mask = static_cast<uint32_t>(
        spv::MemoryAccessMask::Volatile);
    for (auto offset = size_t{5u}; offset < words.size();) {
        auto word_count = static_cast<size_t>(words[offset] >> 16u);
        if (word_count == 0u || word_count > words.size() - offset) {
            facts.structurally_valid = false;
            break;
        }
        auto op = static_cast<spv::Op>(words[offset] & 0xffffu);
        if (op == spv::Op::OpName && word_count >= 3u) {
            facts.named_ids.emplace(
                decode_literal_string(words, offset + 2u,
                                      offset + word_count),
                words[offset + 1u]);
        } else if (op == spv::Op::OpDecorate && word_count >= 3u &&
                   words[offset + 2u] == static_cast<uint32_t>(
                                              spv::Decoration::Coherent)) {
            facts.coherent_ids.emplace(words[offset + 1u]);
        } else if (op == spv::Op::OpLoad && word_count >= 5u &&
                   (words[offset + 4u] & volatile_mask) != 0u) {
            facts.volatile_loads++;
        } else if (op == spv::Op::OpStore && word_count >= 4u &&
                   (words[offset + 3u] & volatile_mask) != 0u) {
            facts.volatile_stores++;
        } else if (op == spv::Op::OpMemoryBarrier) {
            facts.memory_barriers++;
        }
        offset += word_count;
    }
    return facts;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "spirv_volatile_buffer_coherence_propagates_per_argument"_test = [] {
        Module module;
        auto *callable = module.create_callable(Type::of<void>());
        auto *callable_buffer = callable->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        auto *zero = module.create_constant_zero(Type::of<uint32_t>());
        auto *one = module.create_constant_one(Type::of<uint32_t>());
        XIRBuilder builder;
        builder.set_insertion_point(callable->create_body_block());
        static_cast<void>(builder.call(
            Type::of<uint32_t>(),
            ResourceReadOp::BUFFER_VOLATILE_READ,
            {callable_buffer, zero}));
        builder.return_void();

        auto *kernel = module.create_kernel();
        auto *transitive_buffer = kernel->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        auto *direct_buffer = kernel->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        auto *ordinary_buffer = kernel->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        builder.set_insertion_point(kernel->create_body_block());
        static_cast<void>(builder.call(
            Type::of<void>(), callable, {transitive_buffer}));
        builder.call(
            ResourceWriteOp::BUFFER_VOLATILE_WRITE,
            {direct_buffer, zero, one});
        static_cast<void>(builder.call(
            Type::of<uint32_t>(), ResourceReadOp::BUFFER_READ,
            {ordinary_buffer, zero}));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto analysis =
            lc::spirv::analyze_spirv_function_argument_usage(&module);
        expect(lc::spirv::
                   spirv_function_argument_requires_buffer_coherence(
                       analysis, callable, callable_buffer));
        expect(lc::spirv::
                   spirv_function_argument_requires_buffer_coherence(
                       analysis, kernel, transitive_buffer));
        expect(lc::spirv::
                   spirv_function_argument_requires_buffer_coherence(
                       analysis, kernel, direct_buffer));
        expect(!lc::spirv::
                    spirv_function_argument_requires_buffer_coherence(
                        analysis, kernel, ordinary_buffer));
    };

    "spirv_volatile_buffer_emission_is_coherent_and_valid"_test = [] {
        Kernel1D ast_kernel = [](BufferUInt, BufferUInt) noexcept {};
        auto ast_function = ast_kernel.function()->function();
        auto ast_arguments = ast_function.arguments();
        expect(eq(ast_arguments.size(), 2u));

        Module module;
        auto *kernel = module.create_kernel();
        kernel->set_block_size(ast_function.block_size());
        auto *volatile_buffer = kernel->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        auto *ordinary_buffer = kernel->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        auto *zero = module.create_constant_zero(Type::of<uint32_t>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *volatile_value = builder.call(
            Type::of<uint32_t>(),
            ResourceReadOp::BUFFER_VOLATILE_READ,
            {volatile_buffer, zero});
        builder.call(
            ResourceWriteOp::BUFFER_VOLATILE_WRITE,
            {volatile_buffer, zero, volatile_value});
        auto *ordinary_value = builder.call(
            Type::of<uint32_t>(), ResourceReadOp::BUFFER_READ,
            {ordinary_buffer, zero});
        builder.call(
            ResourceWriteOp::BUFFER_WRITE,
            {ordinary_buffer, zero, ordinary_value});
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        auto result = lc::spirv::SpirvCodegenEntry::compile_spirv_xir(
            ast_function, &module,
            ShaderOption{.enable_cache = false}, {});

        spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
        expect(tools.Validate(
            result.spv_bin.data(), result.spv_bin.size()));
        auto facts = inspect_volatile_facts(result.spv_bin);
        expect(facts.structurally_valid);
        expect(eq(facts.volatile_loads, 1u));
        expect(eq(facts.volatile_stores, 1u));
        expect(eq(facts.memory_barriers, 2u));
        if (ast_arguments.size() == 2u) {
            auto volatile_name =
                std::string{"_buf_"} +
                std::to_string(ast_arguments[0].uid());
            auto ordinary_name =
                std::string{"_buf_"} +
                std::to_string(ast_arguments[1].uid());
            auto volatile_id = facts.named_ids.find(volatile_name);
            auto ordinary_id = facts.named_ids.find(ordinary_name);
            expect(volatile_id != facts.named_ids.end());
            expect(ordinary_id != facts.named_ids.end());
            if (volatile_id != facts.named_ids.end()) {
                expect(facts.coherent_ids.contains(
                    volatile_id->second));
            }
            if (ordinary_id != facts.named_ids.end()) {
                expect(!facts.coherent_ids.contains(
                    ordinary_id->second));
            }
        }
    };

    return 0;
}
