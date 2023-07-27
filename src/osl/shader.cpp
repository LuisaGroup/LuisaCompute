#include <algorithm>

#include <luisa/core/logging.h>

#include <luisa/osl/type.h>
#include <luisa/osl/symbol.h>
#include <luisa/osl/instruction.h>
#include <luisa/osl/shader.h>

namespace luisa::compute::osl {

Shader::Shader(luisa::string osl_spec,
               uint32_t version_major, uint32_t version_minor,
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
      _instructions{std::move(instructions)} {
    std::stable_sort(
        _symbols.begin(), _symbols.end(),
        [](auto &&lhs, auto &&rhs) noexcept {
            auto score_tag = [](Symbol::Tag tag) noexcept {
                switch (tag) {
                    case Symbol::Tag::SYM_CONST: return 0u;
                    case Symbol::Tag::SYM_GLOBAL: return 1u;
                    case Symbol::Tag::SYM_PARAM: return 2u;
                    case Symbol::Tag::SYM_OUTPUT_PARAM: return 3u;
                    case Symbol::Tag::SYM_LOCAL: return 4u;
                    case Symbol::Tag::SYM_TEMP: return 5u;
                    default: break;
                }
                return std::numeric_limits<uint32_t>::max();
            };
            return score_tag(lhs->tag()) < score_tag(rhs->tag());
        });
}

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
    auto s = luisa::format("{} {}.{:02}\n", _osl_spec,
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
    auto print_code_markers = [m = 0u, &markers = _code_markers, &s](auto i) mutable noexcept {
        while (m < markers.size() && markers[m].instruction == i) {
            auto &&marker = markers[m];
            s.append(luisa::format("\ncode {}", marker.identifier));
            m++;
        }
    };
    for (auto i = 0u; i < _instructions.size(); i++) {
        print_code_markers(i);
        s.append(luisa::format("\n{:>6}:\t{}", i, _instructions[i]->dump()));
    }
    print_code_markers(_instructions.size());
    s.append("\n\tend\n");
    return s;
}

}// namespace luisa::compute::osl
