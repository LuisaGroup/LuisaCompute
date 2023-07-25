//
// Created by Mike Smith on 2023/7/24.
//

#include <luisa/core/logging.h>

#include <luisa/osl/type.h>
#include <luisa/osl/symbol.h>
#include <luisa/osl/instruction.h>
#include <luisa/osl/shader.h>

namespace luisa::compute::osl {

Shader::Shader(luisa::string osl_spec,
               uint version_major, uint version_minor,
               Shader::Tag tag, luisa::string identifier,
               luisa::vector<Hint> hints,
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

luisa::span<const Hint>
Shader::hints() const noexcept { return _hints; }

luisa::span<const luisa::unique_ptr<Type>>
Shader::types() const noexcept { return _types; }

luisa::span<const luisa::unique_ptr<Symbol>>
Shader::symbols() const noexcept { return _symbols; }

luisa::span<const luisa::unique_ptr<Instruction>>
Shader::instructions() const noexcept { return _instructions; }

luisa::string Shader::dump() const noexcept {
    auto s = luisa::format("{} {}.{}\n", _osl_spec,
                           _osl_version_major,
                           _osl_version_minor);
    switch (_tag) {
        case Tag::GENERIC: s.append("shader "); break;
        case Tag::SURFACE: s.append("surface "); break;
        case Tag::DISPLACEMENT: s.append("displacement "); break;
        case Tag::LIGHT: s.append("light "); break;
        case Tag::VOLUME: s.append("volume "); break;
    }
    s.append(_identifier);
    if (!_hints.empty()) {
        s.append("\t");
        for (auto &&h : _hints) {
            s.append(h.dump()).append(" ");
        }
        s.pop_back();
    }
    for (auto &&symbol : _symbols) {
        s.append("\n").append(symbol->dump());
    }
    auto m = 0u;// code marker index
    for (auto i = 0u; i < _instructions.size(); i++) {
        while (m < _code_markers.size() &&
               _code_markers[m].instruction == i) {
            auto &&marker = _code_markers[m];
            s.append(luisa::format("\ncode {}", marker.identifier));
            m++;
        }
        s.append("\n\t").append(_instructions[i]->dump());
    }
    s.append("\n\tend\n");
    return s;
}

}// namespace luisa::compute::osl
