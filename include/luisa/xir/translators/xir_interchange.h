#pragma once

#include <cstddef>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

struct XIRInterchangeDiagnostic {
    size_t offset{0u};
    size_t line{0u};
    size_t column{0u};
    luisa::string message;
};

struct XIRInterchangeTextWriteResult {
    luisa::string text;
    luisa::vector<XIRInterchangeDiagnostic> diagnostics;
    [[nodiscard]] bool succeeded() const noexcept { return diagnostics.empty(); }
    [[nodiscard]] explicit operator bool() const noexcept { return succeeded(); }
};

struct XIRInterchangeBitcodeWriteResult {
    luisa::vector<std::byte> bitcode;
    luisa::vector<XIRInterchangeDiagnostic> diagnostics;
    [[nodiscard]] bool succeeded() const noexcept { return diagnostics.empty(); }
    [[nodiscard]] explicit operator bool() const noexcept { return succeeded(); }
};

struct XIRInterchangeParseResult {
    luisa::unique_ptr<Module> module;
    luisa::vector<XIRInterchangeDiagnostic> diagnostics;
    [[nodiscard]] bool succeeded() const noexcept { return module != nullptr && diagnostics.empty(); }
    [[nodiscard]] explicit operator bool() const noexcept { return succeeded(); }
};

[[nodiscard]] LUISA_XIR_API XIRInterchangeTextWriteResult
xir_to_interchange_text(const Module *module) noexcept;

[[nodiscard]] LUISA_XIR_API XIRInterchangeParseResult
xir_from_interchange_text(luisa::string_view text) noexcept;

[[nodiscard]] LUISA_XIR_API XIRInterchangeBitcodeWriteResult
xir_to_bitcode(const Module *module) noexcept;

[[nodiscard]] LUISA_XIR_API XIRInterchangeParseResult
xir_from_bitcode(luisa::span<const std::byte> bitcode) noexcept;

}// namespace luisa::compute::xir
