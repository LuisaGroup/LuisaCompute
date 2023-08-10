#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::osl {

class Type;
class Symbol;
class Instruction;
class Hint;

class LC_OSL_API Shader {

public:
    enum struct Tag {
        GENERIC,
        SURFACE,
        DISPLACEMENT,
        LIGHT,
        VOLUME,
    };

    struct CodeMarker {
        luisa::string identifier;
        uint32_t instruction;
    };

private:
    luisa::string _osl_spec;
    uint16_t _osl_version_major;
    uint16_t _osl_version_minor;

private:
    Tag _tag;
    luisa::string _identifier;
    luisa::vector<Hint> _hints;
    luisa::vector<CodeMarker> _code_markers;
    luisa::vector<luisa::unique_ptr<Type>> _types;
    luisa::vector<luisa::unique_ptr<Symbol>> _symbols;
    luisa::vector<luisa::unique_ptr<Instruction>> _instructions;

public:
    Shader(luisa::string osl_spec,
           uint32_t version_major, uint32_t version_minor,
           Tag tag, luisa::string identifier,
           luisa::vector<Hint> hints,
           luisa::vector<CodeMarker> code_markers,
           luisa::vector<luisa::unique_ptr<Type>> types,
           luisa::vector<luisa::unique_ptr<Symbol>> symbols,
           luisa::vector<luisa::unique_ptr<Instruction>> instructions) noexcept;
    ~Shader() noexcept;
    Shader(Shader &&) noexcept;
    Shader(const Shader &) noexcept = delete;
    Shader &operator=(Shader &&) noexcept = delete;
    Shader &operator=(const Shader &) noexcept = delete;
    [[nodiscard]] auto osl_spec() const noexcept { return luisa::string_view{_osl_spec}; }
    [[nodiscard]] auto osl_version_major() const noexcept { return static_cast<uint32_t>(_osl_version_major); }
    [[nodiscard]] auto osl_version_minor() const noexcept { return static_cast<uint32_t>(_osl_version_minor); }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto identifier() const noexcept { return luisa::string_view{_identifier}; }
    [[nodiscard]] auto code_markers() const noexcept { return luisa::span{_code_markers}; }
    [[nodiscard]] luisa::span<const Hint> hints() const noexcept;
    [[nodiscard]] luisa::span<const luisa::unique_ptr<Type>> types() const noexcept;
    [[nodiscard]] luisa::span<const luisa::unique_ptr<Symbol>> symbols() const noexcept;
    [[nodiscard]] luisa::span<const luisa::unique_ptr<Instruction>> instructions() const noexcept;

    // for debugging
    [[nodiscard]] luisa::string dump() const noexcept;
};

}// namespace luisa::compute::osl
