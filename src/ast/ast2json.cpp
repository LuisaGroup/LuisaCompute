//
// Created by Mike on 8/29/2023.
//

#include <utility>

#include <luisa/core/platform.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/logging.h>
#include <luisa/core/magic_enum.h>
#include <luisa/ast/function_builder.h>
#include <luisa/ast/ast2json.h>

namespace luisa::compute {

class JSON {

public:
    enum struct Tag : uint {
        VALUE_NULL,
        VALUE_STRING,
        VALUE_NUMBER,
        VALUE_OBJECT,
        VALUE_ARRAY,
        VALUE_BOOL
    };

    using String = luisa::string;
    using Array = luisa::vector<JSON>;
    using Object = luisa::unordered_map<luisa::string, JSON>;

private:
    Tag _tag;
    union {
        String *string;
        Array *array;
        Object *object;
        double number;
        bool boolean;
    } _value;

private:
    void _destroy() noexcept {
        switch (_tag) {
            case Tag::VALUE_NULL: {
                // do nothing
                break;
            }
            case Tag::VALUE_STRING: {
                if (auto s = std::exchange(_value.string, nullptr)) {
                    luisa::delete_with_allocator(s);
                }
                break;
            }
            case Tag::VALUE_NUMBER: {
                _value.number = 0.;
                break;
            }
            case Tag::VALUE_OBJECT: {
                if (auto o = std::exchange(_value.object, nullptr)) {
                    luisa::delete_with_allocator(o);
                }
                break;
            }
            case Tag::VALUE_ARRAY: {
                if (auto a = std::exchange(_value.array, nullptr)) {
                    luisa::delete_with_allocator(a);
                }
                break;
            }
            case Tag::VALUE_BOOL: {
                _value.boolean = false;
                break;
            }
        }
        _tag = Tag::VALUE_NULL;
    }

    void _copy(const JSON &other) noexcept {
        if (this == &other) { return; }
        switch (other._tag) {
            case Tag::VALUE_NULL:
                if (_tag != Tag::VALUE_NULL) {
                    _destroy();
                    _tag = Tag::VALUE_NULL;
                }
                break;
            case Tag::VALUE_STRING:
                if (_tag != Tag::VALUE_STRING) {
                    _destroy();
                    _tag = Tag::VALUE_STRING;
                    _value.string = luisa::new_with_allocator<String>(*other._value.string);
                } else {
                    *_value.string = *other._value.string;
                }
                break;
            case Tag::VALUE_NUMBER:
                if (_tag != Tag::VALUE_NUMBER) {
                    _destroy();
                    _tag = Tag::VALUE_NUMBER;
                }
                _value.number = other._value.number;
                break;
            case Tag::VALUE_OBJECT:
                if (_tag != Tag::VALUE_OBJECT) {
                    _destroy();
                    _tag = Tag::VALUE_OBJECT;
                    _value.object = luisa::new_with_allocator<Object>(*other._value.object);
                } else {
                    *_value.object = *other._value.object;
                }
                break;
            case Tag::VALUE_ARRAY:
                if (_tag != Tag::VALUE_ARRAY) {
                    _destroy();
                    _tag = Tag::VALUE_ARRAY;
                    _value.array = luisa::new_with_allocator<Array>(*other._value.array);
                } else {
                    *_value.array = *other._value.array;
                }
                break;
            case Tag::VALUE_BOOL:
                if (_tag != Tag::VALUE_BOOL) {
                    _destroy();
                    _tag = Tag::VALUE_BOOL;
                }
                _value.boolean = other._value.boolean;
                break;
        }
    }

public:
    JSON() noexcept : _tag{Tag::VALUE_NULL}, _value{} {}
    ~JSON() noexcept { _destroy(); }

    [[nodiscard]] static auto make(Tag tag) noexcept {
        auto v = JSON{};
        v._tag = tag;
        switch (tag) {
            case Tag::VALUE_NULL: break;
            case Tag::VALUE_STRING: v._value.string = luisa::new_with_allocator<String>(); break;
            case Tag::VALUE_NUMBER: v._value.number = 0.; break;
            case Tag::VALUE_OBJECT: v._value.object = luisa::new_with_allocator<Object>(); break;
            case Tag::VALUE_ARRAY: v._value.array = luisa::new_with_allocator<Array>(); break;
            case Tag::VALUE_BOOL: v._value.boolean = false; break;
        }
        return v;
    }
    [[nodiscard]] static auto make_null() noexcept {
        return JSON{};
    }
    [[nodiscard]] static auto make_string(String s = {}) noexcept {
        auto v = JSON{};
        v._tag = Tag::VALUE_STRING;
        v._value.string = luisa::new_with_allocator<String>(std::move(s));
        return v;
    }
    [[nodiscard]] static auto make_object(Object o = {}) noexcept {
        auto v = JSON{};
        v._tag = Tag::VALUE_OBJECT;
        v._value.object = luisa::new_with_allocator<Object>(std::move(o));
        return v;
    }
    [[nodiscard]] static auto make_array(Array a = {}) noexcept {
        auto v = JSON{};
        v._tag = Tag::VALUE_ARRAY;
        v._value.array = luisa::new_with_allocator<Array>(std::move(a));
        return v;
    }
    [[nodiscard]] static auto make_number(double n = 0.) noexcept {
        auto v = JSON{};
        v._tag = Tag::VALUE_NUMBER;
        v._value.number = n;
        return v;
    }
    [[nodiscard]] static auto make_number(float n) noexcept {
        return make_number(static_cast<double>(n));
    }
    [[nodiscard]] static auto make_number(half n) noexcept {
        return make_number(static_cast<double>(n));
    }
    [[nodiscard]] static auto make_number(int16_t n) noexcept {
        return make_number(static_cast<double>(n));
    }
    [[nodiscard]] static auto make_number(uint16_t n) noexcept {
        return make_number(static_cast<double>(n));
    }
    [[nodiscard]] static auto make_number(int32_t n) noexcept {
        return make_number(static_cast<double>(n));
    }
    [[nodiscard]] static auto make_number(uint n) noexcept {
        return make_number(static_cast<double>(n));
    }
    [[nodiscard]] static auto make_number(long long n) noexcept {
        LUISA_ASSERT(static_cast<int64_t>(static_cast<double>(n)) == n,
                     "JSON(int64_t) cannot represent {}.", n);
        return make_number(static_cast<double>(n));
    }
    [[nodiscard]] static auto make_number(unsigned long long n) noexcept {
        LUISA_ASSERT(static_cast<uint64_t>(static_cast<double>(n)) == n,
                     "JSON(uint64_t) cannot represent {}.", n);
        return make_number(static_cast<double>(n));
    }
    [[nodiscard]] static auto make_number(long n) noexcept {
        return sizeof(n) == 4u ?
                   make_number(static_cast<int32_t>(n)) :
                   make_number(static_cast<long long>(n));
    }
    [[nodiscard]] static auto make_number(unsigned long n) noexcept {
        return sizeof(n) == 4u ?
                   make_number(static_cast<uint32_t>(n)) :
                   make_number(static_cast<unsigned long long>(n));
    }
    [[nodiscard]] static auto make_bool(bool b = false) noexcept {
        auto v = JSON{};
        v._tag = Tag::VALUE_BOOL;
        v._value.boolean = b;
        return v;
    }

public:
    JSON(const JSON &other) noexcept
        : _tag{Tag::VALUE_NULL}, _value{} { _copy(other); }

    JSON(JSON &&other) noexcept
        : _tag{other._tag}, _value{other._value} {
        other._tag = Tag::VALUE_NULL;
        other._value = {};
    }

    explicit JSON(std::nullptr_t) noexcept : JSON{} {}
    explicit JSON(bool b) noexcept : JSON{make_bool(b)} {}
    JSON(String s) noexcept : JSON{make_string(std::move(s))} {}
    JSON(luisa::string_view s) noexcept : JSON{make_string(String{s})} {}
    JSON(const char *s) noexcept : JSON{make_string(String{s})} {}
    JSON(Object o) noexcept : JSON{make_object(std::move(o))} {}
    JSON(Array a) noexcept : JSON{make_array(std::move(a))} {}
    JSON(luisa::span<const JSON> a) noexcept : JSON{make_array(Array{a.cbegin(), a.cend()})} {}
    JSON(double n) noexcept : JSON{make_number(n)} {}
    JSON(float n) noexcept : JSON{make_number(n)} {}
    JSON(half n) noexcept : JSON{make_number(n)} {}
    JSON(int16_t n) noexcept : JSON{make_number(n)} {}
    JSON(uint16_t n) noexcept : JSON{make_number(n)} {}
    JSON(int32_t n) noexcept : JSON{make_number(n)} {}
    JSON(uint32_t n) noexcept : JSON{make_number(n)} {}
    JSON(long n) noexcept : JSON{make_number(n)} {}
    JSON(unsigned long n) noexcept : JSON{make_number(n)} {}
    JSON(long long n) noexcept : JSON{make_number(n)} {}
    JSON(unsigned long long n) noexcept : JSON{make_number(n)} {}

    JSON &operator=(const JSON &rhs) noexcept {
        _copy(rhs);
        return *this;
    }

    JSON &operator=(JSON &&rhs) noexcept {
        if (this != &rhs) {
            this->~JSON();
            new (this) JSON{std::move(rhs)};
        }
        return *this;
    }

    JSON &operator=(std::nullptr_t) noexcept {
        _copy(make_null());
        return *this;
    }
    JSON &operator=(String s) noexcept {
        _copy(make_string(std::move(s)));
        return *this;
    }
    JSON &operator=(luisa::string_view s) noexcept {
        _copy(make_string(String{s}));
        return *this;
    }
    JSON &operator=(const char *s) noexcept {
        _copy(make_string(String{s}));
        return *this;
    }
    JSON &operator=(Object o) noexcept {
        _copy(make_object(std::move(o)));
        return *this;
    }
    JSON &operator=(Array a) noexcept {
        _copy(make_array(std::move(a)));
        return *this;
    }
    JSON &operator=(luisa::span<const JSON> a) noexcept {
        _copy(make_array(Array{a.cbegin(), a.cend()}));
        return *this;
    }
    JSON &operator=(double n) noexcept {
        _copy(make_number(n));
        return *this;
    }
    JSON &operator=(float n) noexcept {
        _copy(make_number(n));
        return *this;
    }
    JSON &operator=(half n) noexcept {
        _copy(make_number(n));
        return *this;
    }
    JSON &operator=(int32_t n) noexcept {
        _copy(make_number(n));
        return *this;
    }
    JSON &operator=(uint32_t n) noexcept {
        _copy(make_number(n));
        return *this;
    }
    JSON &operator=(int16_t n) noexcept {
        _copy(make_number(n));
        return *this;
    }
    JSON &operator=(uint16_t n) noexcept {
        _copy(make_number(n));
        return *this;
    }
    JSON &operator=(long n) noexcept {
        _copy(make_number(static_cast<double>(n)));
        return *this;
    }
    JSON &operator=(unsigned long n) noexcept {
        _copy(make_number(static_cast<double>(n)));
        return *this;
    }
    JSON &operator=(long long n) noexcept {
        _copy(make_number(static_cast<double>(n)));
        return *this;
    }
    JSON &operator=(unsigned long long n) noexcept {
        _copy(make_number(static_cast<double>(n)));
        return *this;
    }
    JSON &operator=(bool b) noexcept {
        _copy(make_bool(b));
        return *this;
    }

public:
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] bool is_null() const noexcept { return _tag == Tag::VALUE_NULL; }
    [[nodiscard]] bool is_string() const noexcept { return _tag == Tag::VALUE_STRING; }
    [[nodiscard]] bool is_object() const noexcept { return _tag == Tag::VALUE_OBJECT; }
    [[nodiscard]] bool is_number() const noexcept { return _tag == Tag::VALUE_NUMBER; }
    [[nodiscard]] bool is_array() const noexcept { return _tag == Tag::VALUE_ARRAY; }
    [[nodiscard]] bool is_bool() const noexcept { return _tag == Tag::VALUE_BOOL; }

public:
    [[nodiscard]] auto &as_string() noexcept {
        LUISA_ASSERT(is_string(),
                     "JSON value (tag = {}) is not a string.",
                     luisa::to_string(_tag));
        return *_value.string;
    }
    [[nodiscard]] auto &as_object() noexcept {
        LUISA_ASSERT(is_object(),
                     "JSON value (tag = {}) is not an object.",
                     luisa::to_string(_tag));
        return *_value.object;
    }
    [[nodiscard]] auto &as_array() noexcept {
        LUISA_ASSERT(is_null() || is_array(),
                     "JSON value (tag = {}) is not an array.",
                     luisa::to_string(_tag));
        return *_value.array;
    }
    [[nodiscard]] auto &as_number() noexcept {
        LUISA_ASSERT(is_number(),
                     "JSON value (tag = {}) is not a number.",
                     luisa::to_string(_tag));
        return _value.number;
    }
    [[nodiscard]] auto &as_bool() noexcept {
        LUISA_ASSERT(is_bool(),
                     "JSON value (tag = {}) is not a boolean.",
                     luisa::to_string(_tag));
        return _value.boolean;
    }

    [[nodiscard]] auto as_string() const noexcept {
        return luisa::string_view{const_cast<JSON *>(this)->as_string()};
    }
    [[nodiscard]] const auto &as_object() const noexcept {
        return const_cast<JSON *>(this)->as_object();
    }
    [[nodiscard]] auto as_array() const noexcept {
        const auto &a = const_cast<JSON *>(this)->as_array();
        return luisa::span{a};
    }
    [[nodiscard]] auto as_number() const noexcept {
        return const_cast<JSON *>(this)->as_number();
    }
    [[nodiscard]] auto as_bool() const noexcept {
        return const_cast<JSON *>(this)->as_bool();
    }

public:
    [[nodiscard]] decltype(auto) operator[](luisa::string_view key) noexcept {
        if (is_null()) { *this = make_object(); }
        return as_object()[key];
    }
    [[nodiscard]] decltype(auto) operator[](size_t index) noexcept {
        if (is_null()) { *this = make_array(); }
        LUISA_ASSERT(index < as_array().size(),
                     "JSON array index out of range: {} >= {}.",
                     index, as_array().size());
        return as_array()[index];
    }
    [[nodiscard]] decltype(auto) operator[](luisa::string_view key) const noexcept {
        auto iter = as_object().find(key);
        LUISA_ASSERT(iter != as_object().end(),
                     "JSON object does not contain key: {}.",
                     key);
        return iter->second;
    }
    [[nodiscard]] decltype(auto) operator[](size_t index) const noexcept {
        LUISA_ASSERT(index < as_array().size(),
                     "JSON array index out of range: {} >= {}.",
                     index, as_array().size());
        return as_array()[index];
    }
    auto &emplace_back(JSON v) noexcept {
        if (is_null()) { *this = make_array(); }
        return as_array().emplace_back(std::move(v));
    }
    auto &emplace(luisa::string key, JSON v) noexcept {
        if (is_null()) { *this = make_object(); }
        return as_object()[key] = std::move(v);
    }
    auto &emplace(size_t index, JSON v) noexcept {
        if (is_null()) { *this = make_array(); }
        as_array().resize(std::max(as_array().size(), index + 1u));
        return as_array()[index] = std::move(v);
    }

public:
    static void _dump_string_escaped(luisa::string &ss, luisa::string_view s) noexcept {
        ss.push_back('"');
        for (auto c : s) {
            switch (c) {
                case '"': ss.append("\\\""); break;
                case '\\': ss.append("\\\\"); break;
                case '\b': ss.append("\\b"); break;
                case '\f': ss.append("\\f"); break;
                case '\n': ss.append("\\n"); break;
                case '\r': ss.append("\\r"); break;
                case '\t': ss.append("\\t"); break;
                default: ss.push_back(c); break;
            }
        }
        ss.push_back('"');
    }
    void _dump_to(luisa::string &ss, uint level, uint indent) const noexcept {
        switch (_tag) {
            case Tag::VALUE_NULL: {
                ss.append("null");
                break;
            }
            case Tag::VALUE_STRING: {
                _dump_string_escaped(ss, *_value.string);
                break;
            }
            case Tag::VALUE_NUMBER: {
                ss.append(luisa::format("{}", _value.number));
                break;
            }
            case Tag::VALUE_OBJECT: {
                ss.push_back('{');
                if (!_value.object->empty()) {
                    auto first = true;
                    for (auto &&[k, v] : *_value.object) {
                        if (!first) { ss.push_back(','); }
                        if (indent != 0u) {
                            ss.push_back('\n');
                            ss.append(level * indent + indent, ' ');
                        } else if (!first) {
                            ss.push_back(' ');
                        }
                        _dump_string_escaped(ss, k);
                        ss.append(": ");
                        v._dump_to(ss, level + 1, indent);
                        first = false;
                    }
                    if (indent != 0u) {
                        ss.push_back('\n');
                        ss.append(level * indent, ' ');
                    }
                }
                ss.push_back('}');
                break;
            }
            case Tag::VALUE_ARRAY: {
                ss.push_back('[');
                if (!_value.array->empty()) {
                    auto first = true;
                    for (auto &&v : *_value.array) {
                        if (!first) { ss.push_back(','); }
                        if (indent != 0u) {
                            ss.push_back('\n');
                            ss.append(level * indent + indent, ' ');
                        } else if (!first) {
                            ss.push_back(' ');
                        }
                        v._dump_to(ss, level + 1u, indent);
                        first = false;
                    }
                    if (indent != 0u) {
                        ss.push_back('\n');
                        ss.append(level * indent, ' ');
                    }
                }
                ss.push_back(']');
                break;
            }
            case Tag::VALUE_BOOL: {
                ss.append(_value.boolean ? "true" : "false");
                break;
            }
        }
    }

public:
    void dump_to(luisa::string &ss, uint indent = 2u) const noexcept {
        _dump_to(ss, 0u, indent);
    }

    [[nodiscard]] auto dump(uint indent = 2u) const noexcept {
        auto ss = luisa::string{};
        ss.reserve(std::min<size_t>(luisa::pagesize(), 4_k) - 1u);
        dump_to(ss, indent);
        if (ss.size() * 2u < ss.capacity()) { ss.shrink_to_fit(); }
        return ss;
    }
};

class AST2JSON {

private:
    struct FunctionContext {
        Function f;
        JSON j;
        luisa::unordered_map<uint, uint> variable_to_index;
    };

private:
    JSON _root;
    luisa::unordered_map<const Type *, uint> _type_to_index;
    luisa::unordered_map<const std::byte *, uint> _constant_to_index;
    luisa::unordered_map<uint64_t, uint> _function_to_index;
    luisa::unordered_map<const ExternalFunction *, uint> _external_function_to_index;
    FunctionContext *_func_ctx{nullptr};

private:
    [[nodiscard]] luisa::string _encode_base64(const void *data, size_t size) noexcept {
        constexpr auto encode_table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        luisa::string ss;
        ss.reserve((size + 2u) / 3u * 4u);
        auto p = reinterpret_cast<const uint8_t *>(data);
        auto i = 0u;
        while (i + 3u <= size) {
            auto x = static_cast<uint>(p[i++]) << 16u;
            x |= static_cast<uint>(p[i++]) << 8u;
            x |= static_cast<uint>(p[i++]);
            ss.push_back(encode_table[(x >> 18u) & 0x3fu]);
            ss.push_back(encode_table[(x >> 12u) & 0x3fu]);
            ss.push_back(encode_table[(x >> 6u) & 0x3fu]);
            ss.push_back(encode_table[x & 0x3fu]);
        }
        if (i + 2u == size) {
            auto x = static_cast<uint>(p[i++]) << 16u;
            x |= static_cast<uint>(p[i++]) << 8u;
            ss.push_back(encode_table[(x >> 18u) & 0x3fu]);
            ss.push_back(encode_table[(x >> 12u) & 0x3fu]);
            ss.push_back(encode_table[(x >> 6u) & 0x3fu]);
            ss.push_back('=');
        } else if (i + 1u == size) {
            auto x = static_cast<uint>(p[i++]) << 16u;
            ss.push_back(encode_table[(x >> 18u) & 0x3fu]);
            ss.push_back(encode_table[(x >> 12u) & 0x3fu]);
            ss.push_back('=');
            ss.push_back('=');
        }
        return ss;
    }

private:
    [[nodiscard]] uint _type_index(const Type *type) noexcept {
        if (auto iter = _type_to_index.find(type);
            iter != _type_to_index.end()) {
            return iter->second;
        }
        if (_type_to_index.empty()) {
            _type_to_index.reserve(Type::count() + 1u);
            _root["types"] = [] {
                JSON::Array a;
                a.reserve(Type::count() + 1u);
                return a;
            }();
        }
        auto converted_type = [type, this] {
            if (type == nullptr) { return JSON::make_null(); }
            JSON t;
            t["tag"] = luisa::to_string(type->tag());
            switch (type->tag()) {
                case Type::Tag::VECTOR:
                case Type::Tag::MATRIX:
                case Type::Tag::ARRAY:
                case Type::Tag::TEXTURE: {
                    t["element"] = _type_index(type->element());
                    t["dimension"] = type->dimension();
                    break;
                }
                case Type::Tag::STRUCTURE: {
                    t["alignment"] = type->alignment();
                    t["members"] = [&] {
                        JSON::Array members;
                        members.reserve(type->members().size());
                        for (auto &&m : type->members()) {
                            members.emplace_back(_type_index(m));
                        }
                        return members;
                    }();
                    break;
                }
                case Type::Tag::BUFFER: {
                    t["element"] = _type_index(type->element());
                    break;
                }
                case Type::Tag::CUSTOM: {
                    t["id"] = type->description();
                    break;
                }
                default: break;
            }
            return t;
        }();
        auto &types = _root["types"].as_array();
        auto index = static_cast<uint>(types.size());
        _type_to_index[type] = static_cast<uint>(index);
        types.emplace_back(std::move(converted_type));
        return index;
    }
    [[nodiscard]] uint _variable_index(Variable v, bool is_argument) noexcept {
        if (auto iter = _func_ctx->variable_to_index.find(v.uid());
            iter != _func_ctx->variable_to_index.end()) {
            return iter->second;
        }
        auto &vars = _func_ctx->j["variables"].as_array();
        auto index = static_cast<uint>(vars.size());
        _func_ctx->variable_to_index[v.uid()] = index;
        // For callables, built-in variables are already lowered
        // to local variables by the FunctionBuilder.
        auto tag = (v.is_local() || v.is_builtin()) && is_argument ?
                       "ARGUMENT" :
                       luisa::to_string(v.tag());
        vars.emplace_back(JSON::Object{
            {"tag", tag},
            {"type", _type_index(v.type())},
        });
        return index;
    }
    [[nodiscard]] uint _constant_index(ConstantData c) noexcept {
        if (auto iter = _constant_to_index.find(c.raw());
            iter != _constant_to_index.end()) {
            return iter->second;
        }
        if (_constant_to_index.empty()) {
            _root["constants"] = JSON::make_array();
        }
        auto converted_constant = [c, this] {
            JSON j;
            j["type"] = _type_index(c.type());
            j["raw"] = _encode_base64(c.raw(), c.type()->size());
            return j;
        }();
        auto &constants = _root["constants"].as_array();
        auto index = static_cast<uint>(constants.size());
        _constant_to_index[c.raw()] = index;
        constants.emplace_back(std::move(converted_constant));
        return index;
    }
    [[nodiscard]] uint _external_function_index(const ExternalFunction *f) noexcept {
        if (auto iter = _external_function_to_index.find(f);
            iter != _external_function_to_index.end()) {
            return iter->second;
        }
        if (_external_function_to_index.empty()) {
            _root["external_functions"] = JSON::make_array();
        }
        auto converted_ext_func = [f, this] {
            JSON j;
            j["name"] = f->name();
            j["return_type"] = _type_index(f->return_type());
            j["argument_types"] = [f, this] {
                JSON::Array args;
                args.reserve(f->argument_types().size());
                for (auto &&t : f->argument_types()) {
                    args.emplace_back(_type_index(t));
                }
                return args;
            }();
            j["argument_usages"] = [f] {
                JSON::Array usages;
                usages.reserve(f->argument_usages().size());
                for (auto &&u : f->argument_usages()) {
                    usages.emplace_back(luisa::to_string(u));
                }
                return usages;
            }();
            return j;
        }();
        auto &ext_funcs = _root["external_functions"].as_array();
        auto index = static_cast<uint>(ext_funcs.size());
        _external_function_to_index[f] = index;
        ext_funcs.emplace_back(std::move(converted_ext_func));
        return index;
    }
    [[nodiscard]] uint _function_index(Function f) noexcept {
        LUISA_ASSERT(f.tag() != Function::Tag::RASTER_STAGE,
                     "Raster stage functions are not supported.");
        if (auto iter = _function_to_index.find(f.hash());
            iter != _function_to_index.end()) {
            return iter->second;
        }
        if (_function_to_index.empty()) {
            _root["functions"] = JSON::make_array();
        }
        FunctionContext ctx{.f = f};
        auto var_count = f.arguments().size() +
                         f.builtin_variables().size() +
                         f.local_variables().size() +
                         f.shared_variables().size();
        ctx.j["variables"] = [var_count] {
            JSON::Array a;
            a.reserve(var_count);
            return a;
        }();
        ctx.variable_to_index.reserve(var_count);
        // push the context
        auto old_ctx = std::exchange(_func_ctx, &ctx);
        // convert
        ctx.j["tag"] = luisa::to_string(f.tag());
        ctx.j["arguments"] = [&] {
            JSON::Array a;
            a.reserve(f.arguments().size());
            for (auto &&arg : f.arguments()) {
                a.emplace_back(_variable_index(arg, true));
            }
            return a;
        }();
        if (f.tag() == Function::Tag::KERNEL) {
            ctx.j["bound_arguments"] = [&] {
                JSON::Array a;
                a.reserve(f.bound_arguments().size());
                for (auto b : f.bound_arguments()) {
                    a.emplace_back(luisa::visit(
                        [&a]<typename T>(T b) noexcept -> JSON {
                            if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                                return JSON::Object{
                                    {"tag", "BUFFER"},
                                    {"handle", luisa::format("{}", b.handle)},
                                    {"offset", luisa::format("{}", b.offset)},
                                    {"size", luisa::format("{}", b.size)},
                                };
                            } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                                return JSON::Object{
                                    {"tag", "TEXTURE"},
                                    {"handle", luisa::format("{}", b.handle)},
                                    {"level", luisa::format("{}", b.level)},
                                };
                            } else if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                                return JSON::Object{
                                    {"tag", "BINDLESS_ARRAY"},
                                    {"handle", luisa::format("{}", b.handle)},
                                };
                            } else if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                                return JSON::Object{
                                    {"tag", "ACCEL"},
                                    {"handle", luisa::format("{}", b.handle)},
                                };
                            } else {
                                LUISA_ERROR_WITH_LOCATION("Invalid bound argument type.");
                            }
                        },
                        b));
                }
                return a;
            }();
            ctx.j["block_size"] = JSON::Array{
                f.block_size().x,
                f.block_size().y,
                f.block_size().z};
        } else {
            ctx.j["return_type"] = _type_index(f.return_type());
        }
        ctx.j["body"] = _convert_stmt(f.body());
        // pop the context and check the stack
        auto popped_ctx = std::exchange(_func_ctx, old_ctx);
        LUISA_ASSERT(popped_ctx == &ctx, "Function context stack corrupted.");
        // insert into the root table
        auto &funcs = _root["functions"].as_array();
        auto index = static_cast<uint>(funcs.size());
        _function_to_index.emplace(f.hash(), index);
        funcs.emplace_back(std::move(ctx.j));
        return index;
    }
    [[nodiscard]] JSON _convert_expr(const Expression *expr) noexcept {
        if (expr == nullptr) { return JSON::make_null(); }
        JSON j;
        j["tag"] = luisa::to_string(expr->tag());
        j["type"] = _type_index(expr->type());
        switch (expr->tag()) {
            case Expression::Tag::UNARY: _convert_unary_expr(j, static_cast<const UnaryExpr *>(expr)); break;
            case Expression::Tag::BINARY: _convert_binary_expr(j, static_cast<const BinaryExpr *>(expr)); break;
            case Expression::Tag::MEMBER: _convert_member_expr(j, static_cast<const MemberExpr *>(expr)); break;
            case Expression::Tag::ACCESS: _convert_access_expr(j, static_cast<const AccessExpr *>(expr)); break;
            case Expression::Tag::LITERAL: _convert_literal_expr(j, static_cast<const LiteralExpr *>(expr)); break;
            case Expression::Tag::REF: _convert_ref_expr(j, static_cast<const RefExpr *>(expr)); break;
            case Expression::Tag::CONSTANT: _convert_constant_expr(j, static_cast<const ConstantExpr *>(expr)); break;
            case Expression::Tag::CALL: _convert_call_expr(j, static_cast<const CallExpr *>(expr)); break;
            case Expression::Tag::CAST: _convert_cast_expr(j, static_cast<const CastExpr *>(expr)); break;
            case Expression::Tag::TYPE_ID: _convert_type_id_expr(j, static_cast<const TypeIDExpr *>(expr)); break;
            case Expression::Tag::STRING_ID: _convert_string_id_expr(j, static_cast<const StringIDExpr *>(expr)); break;
            case Expression::Tag::CPUCUSTOM: LUISA_NOT_IMPLEMENTED();
            case Expression::Tag::GPUCUSTOM: LUISA_NOT_IMPLEMENTED();
        }
        return j;
    }
    void _convert_unary_expr(JSON &j, const UnaryExpr *expr) noexcept {
        j["operand"] = _convert_expr(expr->operand());
        j["op"] = luisa::to_string(expr->op());
    }
    void _convert_binary_expr(JSON &j, const BinaryExpr *expr) noexcept {
        j["lhs"] = _convert_expr(expr->lhs());
        j["rhs"] = _convert_expr(expr->rhs());
        j["op"] = luisa::to_string(expr->op());
    }
    void _convert_member_expr(JSON &j, const MemberExpr *expr) noexcept {
        j["self"] = _convert_expr(expr->self());
        if (expr->is_swizzle()) {
            luisa::string swizzle;
            for (auto i = 0u; i < expr->swizzle_size(); i++) {
                swizzle.push_back("xyzw"[expr->swizzle_index(i)]);
            }
            j["swizzle"] = std::move(swizzle);
        } else {
            j["member"] = expr->member_index();
        }
    }
    void _convert_access_expr(JSON &j, const AccessExpr *expr) noexcept {
        j["range"] = _convert_expr(expr->range());
        j["index"] = _convert_expr(expr->index());
    }
    void _convert_literal_expr(JSON &j, const LiteralExpr *expr) noexcept {
        j["value"] = luisa::visit(
            [this](auto v) noexcept {
                return _encode_base64(&v, sizeof(v));
            },
            expr->value());
    }
    void _convert_ref_expr(JSON &j, const RefExpr *expr) noexcept {
        j["variable"] = _variable_index(expr->variable(), false);
    }
    void _convert_constant_expr(JSON &j, const ConstantExpr *expr) noexcept {
        j["data"] = _constant_index(expr->data());
    }
    void _convert_call_expr(JSON &j, const CallExpr *expr) noexcept {
        j["op"] = luisa::to_string(expr->op());
        if (expr->is_custom()) {
            j["custom"] = _function_index(expr->custom());
        } else if (expr->is_external()) {
            j["external"] = _external_function_index(expr->external());
        }
        j["arguments"] = [&] {
            JSON::Array a;
            a.reserve(expr->arguments().size());
            for (auto &&arg : expr->arguments()) {
                a.emplace_back(_convert_expr(arg));
            }
            return a;
        }();
    }
    void _convert_cast_expr(JSON &j, const CastExpr *expr) noexcept {
        j["op"] = luisa::to_string(expr->op());
        j["expression"] = _convert_expr(expr->expression());
    }
    void _convert_type_id_expr(JSON &j, const TypeIDExpr *expr) noexcept {
        j["data_type"] = _type_index(expr->data_type());
    }
    void _convert_string_id_expr(JSON &j, const StringIDExpr *expr) noexcept {
        j["data"] = expr->data();
    }
    [[nodiscard]] JSON _convert_stmt(const Statement *stmt) noexcept {
        JSON j;
        j["tag"] = luisa::to_string(stmt->tag());
        switch (stmt->tag()) {
            case Statement::Tag::BREAK: /* do nothing */ break;
            case Statement::Tag::CONTINUE: /* do nothing */ break;
            case Statement::Tag::RETURN: _convert_return_stmt(j, static_cast<const ReturnStmt *>(stmt)); break;
            case Statement::Tag::SCOPE: _convert_scope_stmt(j, static_cast<const ScopeStmt *>(stmt)); break;
            case Statement::Tag::IF: _convert_if_stmt(j, static_cast<const IfStmt *>(stmt)); break;
            case Statement::Tag::LOOP: _convert_loop_stmt(j, static_cast<const LoopStmt *>(stmt)); break;
            case Statement::Tag::EXPR: _convert_expr_stmt(j, static_cast<const ExprStmt *>(stmt)); break;
            case Statement::Tag::SWITCH: _convert_switch_stmt(j, static_cast<const SwitchStmt *>(stmt)); break;
            case Statement::Tag::SWITCH_CASE: _convert_switch_case_stmt(j, static_cast<const SwitchCaseStmt *>(stmt)); break;
            case Statement::Tag::SWITCH_DEFAULT: _convert_switch_default_stmt(j, static_cast<const SwitchDefaultStmt *>(stmt)); break;
            case Statement::Tag::ASSIGN: _convert_assign_stmt(j, static_cast<const AssignStmt *>(stmt)); break;
            case Statement::Tag::FOR: _convert_for_stmt(j, static_cast<const ForStmt *>(stmt)); break;
            case Statement::Tag::COMMENT: _convert_comment_stmt(j, static_cast<const CommentStmt *>(stmt)); break;
            case Statement::Tag::RAY_QUERY: _convert_ray_query_stmt(j, static_cast<const RayQueryStmt *>(stmt)); break;
            case Statement::Tag::AUTO_DIFF: _convert_autodiff_stmt(j, static_cast<const AutoDiffStmt *>(stmt)); break;
        }
        return j;
    }
    void _convert_return_stmt(JSON &j, const ReturnStmt *stmt) noexcept {
        j["expression"] = _convert_expr(stmt->expression());
    }
    void _convert_scope_stmt(JSON &j, const ScopeStmt *stmt) noexcept {
        j["statements"] = [&] {
            JSON::Array a;
            a.reserve(stmt->statements().size());
            for (auto &&s : stmt->statements()) {
                a.emplace_back(_convert_stmt(s));
            }
            return a;
        }();
    }
    void _convert_if_stmt(JSON &j, const IfStmt *stmt) noexcept {
        j["condition"] = _convert_expr(stmt->condition());
        j["true_branch"] = _convert_stmt(stmt->true_branch());
        j["false_branch"] = _convert_stmt(stmt->false_branch());
    }
    void _convert_loop_stmt(JSON &j, const LoopStmt *stmt) noexcept {
        j["body"] = _convert_stmt(stmt->body());
    }
    void _convert_expr_stmt(JSON &j, const ExprStmt *stmt) noexcept {
        j["expression"] = _convert_expr(stmt->expression());
    }
    void _convert_switch_stmt(JSON &j, const SwitchStmt *stmt) noexcept {
        j["expression"] = _convert_expr(stmt->expression());
        j["body"] = _convert_stmt(stmt->body());
    }
    void _convert_switch_case_stmt(JSON &j, const SwitchCaseStmt *stmt) noexcept {
        LUISA_ASSERT(stmt->expression()->tag() == Expression::Tag::LITERAL,
                     "Switch case expression must be a literal.");
        auto literal = static_cast<const LiteralExpr *>(stmt->expression());
        j["value"] = luisa::visit(
            []<typename T>(T v) noexcept -> int32_t {
                if constexpr (std::is_integral_v<T>) {
                    auto vv = static_cast<int32_t>(v);
                    LUISA_ASSERT(static_cast<T>(vv) == v,
                                 "Switch case expression must "
                                 "be an int32 literal (got {}).",
                                 Type::of<T>()->description());
                    return vv;
                } else {
                    LUISA_ERROR_WITH_LOCATION(
                        "Switch case expression must be an integer literal.");
                }
            },
            literal->value());
        j["body"] = _convert_stmt(stmt->body());
    }
    void _convert_switch_default_stmt(JSON &j, const SwitchDefaultStmt *stmt) noexcept {
        j["body"] = _convert_stmt(stmt->body());
    }
    void _convert_assign_stmt(JSON &j, const AssignStmt *stmt) noexcept {
        j["lhs"] = _convert_expr(stmt->lhs());
        j["rhs"] = _convert_expr(stmt->rhs());
    }
    void _convert_for_stmt(JSON &j, const ForStmt *stmt) noexcept {
        j["variable"] = _convert_expr(stmt->variable());
        j["condition"] = _convert_expr(stmt->condition());
        j["step"] = _convert_expr(stmt->step());
        j["body"] = _convert_stmt(stmt->body());
    }
    void _convert_comment_stmt(JSON &j, const CommentStmt *stmt) noexcept {
        j["comment"] = stmt->comment();
    }
    void _convert_ray_query_stmt(JSON &j, const RayQueryStmt *stmt) noexcept {
        j["query"] = _convert_expr(stmt->query());
        j["on_triangle_candidate"] = _convert_stmt(stmt->on_triangle_candidate());
        j["on_procedural_candidate"] = _convert_stmt(stmt->on_procedural_candidate());
    }
    void _convert_autodiff_stmt(JSON &j, const AutoDiffStmt *stmt) noexcept {
        j["body"] = _convert_stmt(stmt->body());
    }

public:
    [[nodiscard]] static JSON convert(Function f) noexcept {
        AST2JSON converter;
        auto entry = converter._function_index(f);
        LUISA_ASSERT(converter._func_ctx == nullptr,
                     "Function context stack corrupted.");
        auto j = std::move(converter._root);
        j["entry"] = entry;
        return j;
    }
};

[[nodiscard]] LC_AST_API luisa::string to_json(Function f) noexcept {
    return AST2JSON::convert(f).dump();
}

}// namespace luisa::compute
