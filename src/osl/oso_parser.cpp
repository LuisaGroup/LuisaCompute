#include <cmath>
#include <fstream>

#include <luisa/core/logging.h>
#include <luisa/core/stl/queue.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/unordered_map.h>

#include <luisa/osl/type.h>
#include <luisa/osl/literal.h>
#include <luisa/osl/symbol.h>
#include <luisa/osl/instruction.h>
#include <luisa/osl/oso_parser.h>

namespace luisa::compute::osl {

struct OSOParser::ShaderDecl {

    luisa::string osl_spec;
    uint16_t osl_version_major;
    uint16_t osl_version_minor;
    Shader::Tag tag;
    luisa::string identifier;
    luisa::vector<Hint> hints;

    ShaderDecl(luisa::string osl_spec,
               uint32_t version_major, uint32_t version_minor,
               Shader::Tag tag, luisa::string identifier,
               luisa::vector<Hint> hints) noexcept
        : osl_spec{std::move(osl_spec)},
          osl_version_major{static_cast<uint16_t>(version_major)},
          osl_version_minor{static_cast<uint16_t>(version_minor)},
          tag{tag}, identifier{std::move(identifier)},
          hints{std::move(hints)} {}
};

namespace detail {

[[nodiscard]] inline auto is_identifier_head(char c) noexcept {
    return isalpha(c) || c == '_' || c == '$';
}

[[nodiscard]] inline auto is_identifier_body(char c) noexcept {
    return isalnum(c) || c == '_' || c == '$' || c == '.';
}

[[nodiscard]] inline auto is_number_head(char c) noexcept {
    return isdigit(c) || c == '.' || c == '-' || c == '+';
}

[[nodiscard]] inline auto is_number_body(char c) noexcept {
    return isdigit(c) || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-';
}

}// namespace detail

OSOParser::OSOParser(luisa::string_view source,
                     luisa::string_view path) noexcept
    : _state{source, 0u, 0u}, _path{path} {}

OSOParser::~OSOParser() noexcept = default;

luisa::unique_ptr<Shader> OSOParser::parse(luisa::string_view source) noexcept {
    OSOParser parser{source};
    return parser._parse();
}

luisa::unique_ptr<Shader> OSOParser::parse_file(luisa::string_view path) noexcept {
    std::ifstream file{luisa::filesystem::path{path}};
    LUISA_ASSERT(file.is_open(), "Failed to open file '{}'.", path);
    luisa::string source{std::istreambuf_iterator<char>{file},
                         std::istreambuf_iterator<char>{}};
    OSOParser parser{source, path};
    return parser._parse();
}

void OSOParser::_parse_shader_decl() noexcept {
    _skip_empty_lines();
    // version : IDENTIFIER FLOAT_LITERAL ENDOFLINE
    _skip_whitespaces();
    auto osl_spec = _parse_identifier();
    _skip_whitespaces();
    auto osl_version = _parse_number();
    _skip_whitespaces();
    _match_eol();
    auto osl_version_major = static_cast<uint32_t>(osl_version);
    auto osl_version_minor = static_cast<uint32_t>(std::round((osl_version - osl_version_major) * 100.));
    LUISA_VERBOSE_WITH_LOCATION("Shader version: {} {}.{:02}",
                                osl_spec, osl_version_major, osl_version_minor);
    _skip_empty_lines();
    // shader_declaration : shader_type IDENTIFIER hints_opt ENDOFLINE
    _skip_whitespaces();
    auto tag = [](luisa::string_view s) noexcept {
        using namespace std::string_view_literals;
        if (s == "surface"sv) { return Shader::Tag::SURFACE; }
        if (s == "displacement"sv) { return Shader::Tag::DISPLACEMENT; }
        if (s == "volume"sv) { return Shader::Tag::VOLUME; }
        if (s == "light"sv) { return Shader::Tag::LIGHT; }
        if (s == "shader"sv) { return Shader::Tag::GENERIC; }
        LUISA_WARNING_WITH_LOCATION("Unknown shader tag '{}'. "
                                    "Treating the shader as generic.",
                                    s);
        return Shader::Tag::GENERIC;
    }(_parse_identifier());
    _skip_whitespaces();
    auto identifier = _parse_identifier();
    _skip_whitespaces();
    auto hints = _parse_hints();
    _skip_whitespaces();
    _match_eol();

    // create shader data
    _shader = luisa::make_unique<ShaderDecl>(
        std::move(osl_spec), osl_version_major, osl_version_minor,
        tag, std::move(identifier), std::move(hints));
}

luisa::unique_ptr<Symbol> OSOParser::_parse_symbol() noexcept {

    // symbol : SYMTYPE typespec arraylen_opt IDENTIFIER initial_values_opt hints_opt ENDOFLINE
    // arraylen_opt : '[' INT_LITERAL ']' | '[' ']'
    auto backup = _backup();
    auto op = _parse_identifier();
    using namespace std::string_view_literals;
    auto tag = [&op]() noexcept -> luisa::optional<Symbol::Tag> {
        if (op == "param"sv) { return Symbol::Tag::SYM_PARAM; }
        if (op == "oparam"sv) { return Symbol::Tag::SYM_OUTPUT_PARAM; }
        if (op == "local"sv) { return Symbol::Tag::SYM_LOCAL; }
        if (op == "temp"sv) { return Symbol::Tag::SYM_TEMP; }
        if (op == "global"sv) { return Symbol::Tag::SYM_GLOBAL; }
        if (op == "const"sv) { return Symbol::Tag::SYM_CONST; }
        return luisa::nullopt;
    }();
    if (!tag) {
        LUISA_ASSERT(op == "code"sv,
                     "Unknown symbol tag '{}' at {}. "
                     "Expected 'code'.",
                     op, _location());
        _restore(backup);
        return nullptr;
    }
    _skip_whitespaces();
    auto type = _parse_type();
    _skip_whitespaces();
    auto array_len = 0;
    if (_peek() == '[') {
        static_cast<void>(_read());
        _skip_whitespaces();
        if (_is_number()) {
            auto len = _parse_number();
            LUISA_ASSERT(len > 0 && static_cast<int>(len) == len,
                         "Invalid array length '{}' at {}. "
                         "Expected a positive integer.",
                         len, _location());
            array_len = static_cast<int>(len);
        } else {
            array_len = -1;// unbounded array
        }
        _skip_whitespaces();
        _match(']');
        _skip_whitespaces();
    }
    auto identifier = _parse_identifier();
    _skip_whitespaces();
    auto initial_values = _parse_initial_values();
    _skip_whitespaces();
    auto hints = _parse_hints();
    _skip_whitespaces();
    _match_eol();

    // find its parent
    Symbol *parent = nullptr;
    // find the last '.' in the identifier
    if (auto dot = identifier.find_last_of('.');
        dot != luisa::string ::npos) {
        auto parent_ident = luisa::string_view{identifier}.substr(0u, dot);
        auto iter = _id_to_symbol.find(parent_ident);
        LUISA_ASSERT(iter != _id_to_symbol.end(),
                     "Unknown parent symbol '{}' of '{}' at {}.",
                     parent_ident, identifier, _location());
        parent = iter->second;
    }
    // create the symbol
    auto symbol = luisa::make_unique<Symbol>(
        *tag, type, array_len, parent, std::move(identifier),
        std::move(initial_values), std::move(hints));
    if (parent) { parent->add_child(symbol.get()); }
    return symbol;
}

void OSOParser::_parse_symbols() noexcept {
    while (!_eof()) {
        _skip_empty_lines();
        _skip_whitespaces();
        auto symbol = _parse_symbol();
        if (!symbol) { break; }
        _id_to_symbol.emplace(symbol->identifier(), symbol.get());
        _symbols.emplace_back(std::move(symbol));
    }
}

void OSOParser::_parse_instructions() noexcept {
    // instruction : label opcode arguments_opt jumptargets_opt hints_opt ENDOFLINE
    // codemarker : CODE IDENTIFIER ENDOFLINE
    while (!_eof()) {
        _skip_empty_lines();
        _skip_whitespaces();
        // skip label if any
        if (_is_number()) {
            auto label = _parse_number();
            LUISA_WARNING_WITH_LOCATION(
                "Labels are not supported. "
                "Skipping {} at {}.",
                label, _location());
            _skip_whitespaces();
            _match(':');
            _skip_whitespaces();
        }
        auto opcode = _parse_identifier();
        using namespace std::string_view_literals;
        if (opcode == "end"sv) { break; }
        if (opcode == "code"sv) {// code markers
            _skip_whitespaces();
            auto marker = _parse_identifier();
            auto offset = static_cast<uint32_t>(_instructions.size());
            _code_markers.emplace_back(Shader::CodeMarker{
                .identifier = std::move(marker),
                .instruction = offset});
        } else {// normal instructions
            _skip_whitespaces();
            auto args = _parse_arguments();
            _skip_whitespaces();
            auto targets = _parse_jump_targets();
            _skip_whitespaces();
            auto hints = _parse_hints();
            auto instr = luisa::make_unique<Instruction>(
                std::move(opcode), std::move(args),
                std::move(targets), std::move(hints));
            _instructions.emplace_back(std::move(instr));
        }
        _skip_whitespaces();
        _match_eol();
    }
}

luisa::vector<const Symbol *> OSOParser::_parse_arguments() noexcept {
    luisa::vector<const Symbol *> args;
    while (!_eol()) {
        if (!_is_identifier()) { break; }
        auto ident = _parse_identifier();
        auto iter = _id_to_symbol.find(ident);
        LUISA_ASSERT(iter != _id_to_symbol.end(),
                     "Unknown symbol '{}' at {}.",
                     ident, _location());
        args.emplace_back(iter->second);
        _skip_whitespaces();
    }
    return args;
}

luisa::vector<int> OSOParser::_parse_jump_targets() noexcept {
    luisa::vector<int> targets;
    while (!_eol()) {
        if (!_is_number()) { break; }
        auto target = _parse_number();
        LUISA_ASSERT(target >= 0 && static_cast<int>(target) == target,
                     "Invalid jump target '{}' at {}. "
                     "Expected a positive integer.",
                     target, _location());
        targets.emplace_back(static_cast<int>(target));
        _skip_whitespaces();
    }
    return targets;
}

luisa::string OSOParser::_parse_identifier() noexcept {
    auto head = _read();
    LUISA_ASSERT(detail::is_identifier_head(head),
                 "Invalid identifier head '{}' at {}. "
                 "Expected [a-zA-Z_$].",
                 head, _location());
    luisa::string result;
    result.push_back(head);
    while (!_eol()) {
        auto c = _peek();
        if (!detail::is_identifier_body(c)) { break; }
        result.push_back(_read());
    }
    return result;
}

double OSOParser::_parse_number() noexcept {
    auto p_begin = _state.source.data();
    auto head = _read();
    LUISA_ASSERT(detail::is_number_head(head),
                 "Invalid number head '{}' at {}. "
                 "Expected [0-9.].",
                 head, _location());
    while (!_eol()) {
        auto c = _peek();
        if (!detail::is_number_body(c)) { break; }
        static_cast<void>(_read());
    }
    if (*p_begin == '+') { ++p_begin; }
    auto p_end = _state.source.data();
    auto p_ret = p_end;
    auto x = std::strtod(p_begin, const_cast<char **>(&p_ret));
    LUISA_ASSERT(p_ret == p_end,
                 "Invalid number '{}' at {}. "
                 "Expected [0-9.].",
                 luisa::string_view{p_begin, static_cast<size_t>(p_end - p_begin)}, _location());
    return x;
}

luisa::string OSOParser::_parse_string(bool keep_quotes) noexcept {
    // parse the string without escape sequences
    if (keep_quotes) {
        auto p_begin = _state.source.data();
        _match('"');
        while (!_eol()) {
            auto c = _peek();
            if (c == '"') { break; }
            static_cast<void>(_read());
            if (c == '\\') { static_cast<void>(_read()); }
        }
        _match('"');
        auto p_end = _state.source.data();
        return luisa::string{p_begin, static_cast<size_t>(p_end - p_begin)};
    }
    // handle escape sequences
    luisa::string result;
    _match('"');
    while (!_eol()) {
        auto c = _peek();
        if (c == '"') { break; }
        static_cast<void>(_read());
        if (c == '\\') {
            switch (auto escape = _read()) {
                case 'n': result.push_back('\n'); break;
                case 't': result.push_back('\t'); break;
                case 'r': result.push_back('\r'); break;
                case 'f': result.push_back('\f'); break;
                case 'v': result.push_back('\v'); break;
                case 'b': result.push_back('\b'); break;
                case 'a': result.push_back('\a'); break;
                case '\\': result.push_back('\\'); break;
                case '"': result.push_back('"'); break;
                case '\'': result.push_back('\''); break;
                case '?': result.push_back('?'); break;
                default:
                    LUISA_WARNING_WITH_LOCATION(
                        "Unknown escape sequence '\\{}' at {}. "
                        "Returning the original character '{}'.",
                        escape, _location(), escape);
                    result.push_back(escape);
                    break;
            }
        } else {
            result.push_back(c);
        }
    }
    _match('"');
    return result;
}

Hint OSOParser::_parse_hint() noexcept {
    // IDORLITERAL: {FLT}|{STR}|{INTEGER}|({IDENT}(\[{INTEGER}?\])?)
    // HINTPATTERN: \%{IDENT}(\{({IDORLITERAL}(\,{IDORLITERAL})*)?\})?
    _match('%');
    auto name = _parse_identifier();
    luisa::vector<luisa::string> args;
    if (_peek() == '{') {
        static_cast<void>(_read());
        while (!_eol()) {
            if (_peek() == '}') { break; }
            if (_is_number()) {
                args.emplace_back(luisa::format("{}", _parse_number()));
            } else if (_is_string()) {
                args.emplace_back(_parse_string(true));
            } else {
                auto ident = _parse_identifier();
                if (_peek() == '[') {
                    _match('[');
                    if (_is_number()) {
                        args.emplace_back(luisa::format("{}[{}]", ident, _parse_number()));
                    } else {
                        args.emplace_back(luisa::format("{}[]", ident));
                    }
                    _skip_whitespaces();
                    _match(']');
                } else {
                    args.emplace_back(ident);
                }
            }
            if (_peek() == '}') { break; }
            _match(',');
        }
        _match('}');
    }
    return {std::move(name), std::move(args)};
}

const Type *OSOParser::_parse_type() noexcept {
    // typespec : simple_typename | CLOSURE simple_typename | STRUCT IDENTIFIER
    // simple_typename : COLORTYPE | FLOATTYPE | INTTYPE | MATRIXTYPE | NORMALTYPE | POINTTYPE | STRINGTYPE | VECTORTYPE | VOIDTYPE
    auto ident = _parse_identifier();
    using namespace std::string_view_literals;
    luisa::unique_ptr<Type> type;
    if (ident == "closure") {
        _skip_whitespaces();
        auto gentype = _parse_type();
        LUISA_ASSERT(gentype->identifier() == "color"sv,
                     "Invalid closure type '{}' at {}. "
                     "Expected 'color'.",
                     gentype->identifier(), _location());
        if (auto iter = _id_to_type.find("closure color"sv);
            iter != _id_to_type.end()) {
            return iter->second;
        }
        type = luisa::make_unique<ClosureType>(gentype);
    } else if (ident == "struct") {
        _skip_whitespaces();
        ident = _parse_identifier();
        if (auto iter = _id_to_type.find(ident);
            iter != _id_to_type.end()) {
            return iter->second;
        }
        type = luisa::make_unique<StructType>(std::move(ident));
    } else {
        if (auto iter = _id_to_type.find(ident);
            iter != _id_to_type.end()) {
            return iter->second;
        }
        if (ident == "int"sv) {
            type = luisa::make_unique<SimpleType>(SimpleType::Primitive::INT);
        } else if (ident == "float"sv) {
            type = luisa::make_unique<SimpleType>(SimpleType::Primitive::FLOAT);
        } else if (ident == "point"sv) {
            type = luisa::make_unique<SimpleType>(SimpleType::Primitive::POINT);
        } else if (ident == "normal"sv) {
            type = luisa::make_unique<SimpleType>(SimpleType::Primitive::NORMAL);
        } else if (ident == "vector"sv) {
            type = luisa::make_unique<SimpleType>(SimpleType::Primitive::VECTOR);
        } else if (ident == "color"sv) {
            type = luisa::make_unique<SimpleType>(SimpleType::Primitive::COLOR);
        } else if (ident == "matrix"sv) {
            type = luisa::make_unique<SimpleType>(SimpleType::Primitive::MATRIX);
        } else if (ident == "string"sv) {
            type = luisa::make_unique<SimpleType>(SimpleType::Primitive::STRING);
        } else {
            LUISA_ERROR_WITH_LOCATION(
                "Unknown type '{}' at {}.",
                ident, _location());
        }
    }
    auto p_type = type.get();
    _id_to_type.emplace(type->identifier(), p_type);
    _types.emplace_back(std::move(type));
    return p_type;
}

luisa::vector<Literal> OSOParser::_parse_initial_values() noexcept {
    luisa::vector<Literal> values;
    while (!_eol()) {
        if (_is_number()) {
            auto x = _parse_number();
            values.emplace_back(x);
        } else if (_is_string()) {
            auto x = _parse_string();
            values.emplace_back(x);
        } else {
            break;
        }
        _skip_whitespaces();
    }
    return values;
}

luisa::vector<Hint> OSOParser::_parse_hints() noexcept {
    luisa::vector<Hint> hints;
    while (_is_hint()) {
        hints.emplace_back(_parse_hint());
        _skip_whitespaces();
    }
    return hints;
}

void OSOParser::_materialize_structs() noexcept {

    // probe all root structs
    luisa::queue<const Symbol *> queue;
    for (auto &&symbol : _symbols) {
        if (symbol->type()->tag() == Type::Tag::STRUCT &&
            symbol->is_root()) {
            queue.push(symbol.get());
        }
    }
    // process
    luisa::unordered_set<const Type *> processed;
    while (!queue.empty()) {
        auto symbol = queue.front();
        queue.pop();
        // skip if already processed
        if (!processed.emplace(symbol->type()).second) {
            continue;
        }
        // process children
        luisa::span<const luisa::string> field_names;
        for (auto &&hint : symbol->hints()) {
            using namespace std::string_view_literals;
            if (hint.identifier() == "structfields"sv) {
                field_names = hint.args();
                break;
            }
        }
        if (!field_names.empty()) {
            LUISA_ASSERT(field_names.size() == symbol->children().size(),
                         "Invalid struct fields hint for struct '{}'. "
                         "Expected {} field names, but got {}.",
                         symbol->identifier(), symbol->children().size(),
                         field_names.size());
        }
        luisa::vector<StructType::Field> fields;
        fields.reserve(symbol->children().size());
        for (auto i = 0u; i < symbol->children().size(); i++) {
            auto child = symbol->children()[i];
            if (child->type()->tag() == Type::Tag::STRUCT) {
                queue.emplace(child);
            }
            // process field
            LUISA_ASSERT(child->identifier().starts_with(symbol->identifier()),
                         "Invalid child '{}' of '{}'. Expected '{}.*'.",
                         child->identifier(), symbol->identifier(),
                         symbol->identifier());
            auto field_name = child->identifier().substr(symbol->identifier().size());
            LUISA_ASSERT(field_name.starts_with('.'),
                         "Invalid child '{}' of '{}'. Expected '{}.*'.",
                         child->identifier(), symbol->identifier(),
                         symbol->identifier());
            field_name = field_name.substr(1u);
            if (!field_names.empty()) {
                LUISA_ASSERT(field_name == field_names[i],
                             "Invalid child '{}' of '{}'. Expected '{}'.",
                             child->identifier(), symbol->identifier(),
                             field_names[i]);
            }
            auto array_len = 0u;
            if (symbol->is_array()) {
                LUISA_ASSERT(child->array_length() == symbol->array_length(),
                             "Invalid array length {} for symbol '{}' in struct '{}'. Expected {}.",
                             child->array_length(), child->identifier(),
                             symbol->type()->identifier(), symbol->array_length());
            } else {
                LUISA_ASSERT(!child->is_unbounded(),
                             "Invalid unbounded array length "
                             "for symbol '{}' in struct '{}'.",
                             child->identifier(),
                             symbol->type()->identifier());
                array_len = child->array_length();
            }
            fields.emplace_back(StructType::Field{
                .name = luisa::string{field_name},
                .type = child->type(),
                .array_length = array_len});
        }
        auto mutable_type_iter = _id_to_type.find(symbol->type()->identifier());
        LUISA_ASSERT(mutable_type_iter != _id_to_type.end() &&
                         mutable_type_iter->second->tag() == Type::Tag::STRUCT,
                     "Unknown struct type '{}' at {}.",
                     symbol->type()->identifier(), _location());
        static_cast<StructType *>(mutable_type_iter->second)
            ->set_fields(std::move(fields));
    }
}

luisa::unique_ptr<Shader> OSOParser::_parse() noexcept {
    // oso_file : version shader_declaration symbols_opt codemarker instructions
    _parse_shader_decl();
    _parse_symbols();
    _parse_instructions();
    _materialize_structs();
    return luisa::make_unique<Shader>(std::move(_shader->osl_spec),
                                      _shader->osl_version_major,
                                      _shader->osl_version_minor,
                                      _shader->tag,
                                      _shader->identifier,
                                      std::move(_shader->hints),
                                      std::move(_code_markers),
                                      std::move(_types),
                                      std::move(_symbols),
                                      std::move(_instructions));
}

char OSOParser::_peek() const noexcept {
    LUISA_ASSERT(!_eof(), "Unexpected EOF at {}.", _location());
    return _state.source.front();
}

char OSOParser::_read() noexcept {
    LUISA_ASSERT(!_eof(), "Unexpected EOF at {}.", _location());
    auto c = _state.source.front();
    _state.source.remove_prefix(1);
    if (c == '\n') {
        _state.line++;
        _state.column = 0u;
    } else {
        _state.column++;
    }
    return c;
}

void OSOParser::_match(char expected) noexcept {
    auto c = _read();
    if (c != expected) {
        LUISA_ERROR_WITH_LOCATION(
            "Unexpected character '{}' at {}. Expected '{}'.",
            c, _location(), expected);
    }
}

void OSOParser::_match(luisa::string_view expected) noexcept {
    for (auto c : expected) { _match(c); }
}

void OSOParser::_match_eol() noexcept {
    if (_eof()) { return; }
    _match('\n');
}

luisa::string OSOParser::_location() const noexcept {
    return _path.empty() ?
               luisa::format("({}:{})",
                             _state.line + 1u,
                             _state.column + 1u) :
               luisa::format("({}:{}:{})",
                             _path,
                             _state.line + 1u,
                             _state.column + 1u);
}

bool OSOParser::_eof() const noexcept {
    return _state.source.empty();
}

bool OSOParser::_eol() const noexcept {
    return _eof() || _peek() == '\n';
}

bool OSOParser::_is_number() const noexcept {
    return !_eol() && detail::is_number_head(_peek());
}

bool OSOParser::_is_string() const noexcept {
    return !_eol() && _peek() == '"';
}

bool OSOParser::_is_identifier() const noexcept {
    return !_eol() && detail::is_identifier_head(_peek());
}

bool OSOParser::_is_hint() const noexcept {
    return !_eol() && _peek() == '%';
}

void OSOParser::_skip_whitespaces() noexcept {
    auto is_whitespace = [](char c) noexcept {
        return c == ' ' || c == '\t' || c == '\f' || c == '\v' || c == '\r';
    };
    while (!_eof() && is_whitespace(_peek())) {
        static_cast<void>(_read());
    }
    if (!_eof() && _peek() == '#') {
        while (!_eof() && !_eol()) {
            static_cast<void>(_read());
        }
    }
}

void OSOParser::_skip_empty_lines() noexcept {
    while (!_eof()) {
        _skip_whitespaces();
        if (_eol()) {
            _match_eol();
        } else {
            break;
        }
    }
}

}// namespace luisa::compute::osl
