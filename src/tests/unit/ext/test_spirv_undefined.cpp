// Validator-backed test for preserving XIR undefined values in native SPIR-V.

#include "ut/ut.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <spirv-tools/libspirv.hpp>

#include <luisa/dsl/sugar.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/verifier.h>

#include "spirv_codegen/entry.h"

using namespace boost::ut;
using namespace boost::ut::literals;
using namespace luisa;
using namespace luisa::compute;

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

[[nodiscard]] bool contains_array_undefined(
    const std::vector<uint32_t> &words) noexcept {
    if (words.size() < 5u) { return false; }
    std::unordered_set<uint32_t> array_types;
    for (auto offset = 5u; offset < words.size();) {
        auto word_count = words[offset] >> 16u;
        auto op = static_cast<spv::Op>(words[offset] & 0xffffu);
        if (word_count == 0u || offset + word_count > words.size()) {
            return false;
        }
        if (op == spv::Op::OpTypeArray && word_count == 4u) {
            array_types.emplace(words[offset + 1u]);
        }
        offset += word_count;
    }
    for (auto offset = 5u; offset < words.size();) {
        auto word_count = words[offset] >> 16u;
        auto op = static_cast<spv::Op>(words[offset] & 0xffffu);
        if (word_count == 0u || offset + word_count > words.size()) {
            return false;
        }
        if (op == spv::Op::OpUndef && word_count == 3u &&
            array_types.contains(words[offset + 1u])) {
            return true;
        }
        offset += word_count;
    }
    return false;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "native_spirv_preserves_undefined_aggregate"_test = [] {
        using Bank = std::array<float4, 3u>;
        Kernel1D kernel = [](BufferVar<Bank> output) {
            output.write(dispatch_id().x, undefined<Bank>());
        };
        auto module = xir::ast_to_xir_translate(
            kernel.function()->function(), {});
        expect(module != nullptr);
        expect(xir::xir_verify_module(module.get()).succeeded());

        ScopedEnvironmentVariable disable_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        auto compiled = lc::spirv::SpirvCodegenEntry::compile_spirv_xir(
            kernel.function()->function(), module.get(),
            ShaderOption{.enable_cache = false});

        spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
        expect(tools.Validate(compiled.spv_bin.data(),
                              compiled.spv_bin.size()))
            << "SPIR-V containing aggregate OpUndef must validate";
        std::string text;
        expect(tools.Disassemble(compiled.spv_bin.data(),
                                 compiled.spv_bin.size(), &text));
        expect(contains_array_undefined(compiled.spv_bin))
            << "native XIR codegen must not refine aggregate undefined to zero";
    };
}
