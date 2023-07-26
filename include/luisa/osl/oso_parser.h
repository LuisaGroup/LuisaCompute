#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/unordered_map.h>

#include <luisa/osl/shader.h>

namespace luisa::compute::osl {

class Shader;
class Symbol;
class Type;
class Instruction;
class Literal;
class Hint;

class LC_OSL_API OSOParser {

private:
    struct ShaderDecl;
    struct State {
        luisa::string_view source;
        uint32_t line;
        uint32_t column;
    };

private:
    State _state;
    luisa::string_view _path;
    luisa::unique_ptr<ShaderDecl> _shader;
    luisa::vector<Shader::CodeMarker> _code_markers;
    luisa::vector<luisa::unique_ptr<Type>> _types;
    luisa::vector<luisa::unique_ptr<Symbol>> _symbols;
    luisa::vector<luisa::unique_ptr<Instruction>> _instructions;
    luisa::unordered_map<luisa::string, Symbol *> _id_to_symbol;
    luisa::unordered_map<luisa::string, Type *> _id_to_type;

private:
    explicit OSOParser(luisa::string_view source,
                       luisa::string_view path = {}) noexcept;
    ~OSOParser() noexcept;

public:
    OSOParser(const OSOParser &) noexcept = delete;
    OSOParser(OSOParser &&) noexcept = delete;
    OSOParser &operator=(const OSOParser &) noexcept = delete;
    OSOParser &operator=(OSOParser &&) noexcept = delete;

private:
    [[nodiscard]] luisa::string _location() const noexcept;
    [[nodiscard]] luisa::unique_ptr<Shader> _parse() noexcept;

    void _parse_shader_decl() noexcept;
    void _parse_symbols() noexcept;
    void _parse_instructions() noexcept;
    void _materialize_structs() noexcept;

private:
    [[nodiscard]] auto _backup() const noexcept { return _state; }
    void _restore(State state) noexcept { _state = state; }

private:
    [[nodiscard]] bool _eof() const noexcept;
    [[nodiscard]] bool _eol() const noexcept;
    [[nodiscard]] char _peek() const noexcept;
    [[nodiscard]] char _read() noexcept;
    void _match(char expected) noexcept;
    void _match(luisa::string_view expected) noexcept;
    void _match_eol() noexcept;
    void _skip_whitespaces() noexcept;
    void _skip_empty_lines() noexcept;
    [[nodiscard]] bool _is_number() const noexcept;
    [[nodiscard]] bool _is_string() const noexcept;
    [[nodiscard]] bool _is_identifier() const noexcept;
    [[nodiscard]] bool _is_hint() const noexcept;
    [[nodiscard]] luisa::string _parse_identifier() noexcept;
    [[nodiscard]] double _parse_number() noexcept;
    [[nodiscard]] luisa::string _parse_string(bool keep_quotes = false) noexcept;
    [[nodiscard]] Hint _parse_hint() noexcept;
    [[nodiscard]] luisa::vector<Hint> _parse_hints() noexcept;
    [[nodiscard]] const Type *_parse_type() noexcept;
    [[nodiscard]] luisa::vector<Literal> _parse_initial_values() noexcept;
    [[nodiscard]] luisa::vector<const Symbol *> _parse_arguments() noexcept;
    [[nodiscard]] luisa::vector<int> _parse_jump_targets() noexcept;
    [[nodiscard]] luisa::unique_ptr<Symbol> _parse_symbol() noexcept;

public:
    [[nodiscard]] static luisa::unique_ptr<Shader> parse(luisa::string_view source) noexcept;
    [[nodiscard]] static luisa::unique_ptr<Shader> parse_file(luisa::string_view path) noexcept;
};

}// namespace luisa::compute::osl
