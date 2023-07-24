//
// Created by Mike Smith on 2023/7/24.
//

#include <luisa/osl/type.h>
#include <luisa/osl/symbol.h>
#include <luisa/osl/instruction.h>
#include <luisa/osl/shader.h>

namespace luisa::compute::osl {

Shader::Shader(luisa::string osl_spec,
               uint version_major, uint version_minor,
               Shader::Tag tag, luisa::string identifier,
               luisa::vector<luisa::string> hints,
               vector<Shader::CodeMarker> code_markers,
               luisa::vector<luisa::unique_ptr<Type>> types,
               luisa::vector<luisa::unique_ptr<Symbol>> symbols,
               luisa::vector<luisa::unique_ptr<Instruction>> instructions) noexcept
    : _osl_spec{std::move(osl_spec)},
      _osl_version_major{static_cast<uint16_t>(version_major)},
      _osl_version_minor{static_cast<uint16_t>(version_minor)},
      _tag{tag},
      _identifier{std::move(identifier)},
      _hints{std::move(hints)},
      _code_markers{std::move(code_markers)},
      _types{std::move(types)},
      _symbols{std::move(symbols)},
      _instructions{std::move(instructions)} {}

Shader::~Shader() noexcept = default;
Shader::Shader(Shader &&) noexcept = default;

luisa::span<const luisa::unique_ptr<Type>>
Shader::types() const noexcept { return _types; }

luisa::span<const luisa::unique_ptr<Symbol>>
Shader::symbols() const noexcept { return _symbols; }

luisa::span<const luisa::unique_ptr<Instruction>>
Shader::instructions() const noexcept { return _instructions; }

}// namespace luisa::compute::osl
