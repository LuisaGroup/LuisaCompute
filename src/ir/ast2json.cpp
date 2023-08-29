//
// Created by Mike on 8/29/2023.
//

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/logging.h>
#include <luisa/core/magic_enum.h>
#include <luisa/ir/ast2json.h>

#include <utility>

namespace luisa::compute {

class JSON {

public:
    enum struct Tag : uint32_t {
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
    [[nodiscard]] static auto make_number(uint32_t n) noexcept {
        return make_number(static_cast<double>(n));
    }
    [[nodiscard]] static auto make_number(int64_t n) noexcept {
        LUISA_ASSERT(static_cast<int64_t>(static_cast<double>(n)) == n,
                     "JSON(int64_t) cannot represent {}.", n);
        return make_number(static_cast<double>(n));
    }
    [[nodiscard]] static auto make_number(uint64_t n) noexcept {
        LUISA_ASSERT(static_cast<uint64_t>(static_cast<double>(n)) == n,
                     "JSON(uint64_t) cannot represent {}.", n);
        return make_number(static_cast<double>(n));
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
    JSON(int64_t n) noexcept : JSON{make_number(n)} {}
    JSON(uint64_t n) noexcept : JSON{make_number(n)} {}

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
    JSON &operator=(int64_t n) noexcept {
        _copy(make_number(static_cast<double>(n)));
        return *this;
    }
    JSON &operator=(uint64_t n) noexcept {
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
    [[nodiscard]] auto as_bool() noexcept {
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
    [[nodiscard]] auto &emplace_back(JSON v) noexcept {
        if (is_null()) { *this = make_array(); }
        return as_array().emplace_back(std::move(v));
    }
    [[nodiscard]] auto &emplace(luisa::string key, JSON v) noexcept {
        if (is_null()) { *this = make_object(); }
        return as_object()[key] = std::move(v);
    }
    [[nodiscard]] auto &emplace(size_t index, JSON v) noexcept {
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
    void _dump_to(luisa::string &ss, uint32_t level, uint32_t indent) const noexcept {
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
    void dump_to(luisa::string &ss, uint32_t indent = 2u) const noexcept {
        _dump_to(ss, 0u, indent);
    }

    [[nodiscard]] auto dump(uint32_t indent = 2u) const noexcept {
        auto ss = luisa::string{};
        ss.reserve(4096u);
        dump_to(ss, indent);
        ss.shrink_to_fit();
        return ss;
    }
};

class AST2JSON {

private:
};

[[nodiscard]] luisa::string to_json(Function) noexcept {
    JSON json;
    json["hello"] = {{"world", 1.0},
                     {"hello\n\taaaa", 2.0},
                     {"world2", nullptr},
                     {"hello", true},
                     {"help", "string"},
                     {"nested", JSON::Array{1., 2., 3., 4.}},
                     {"yay",
                      JSON::Object{{"hello", "world"},
                                   {"test", ""}}}};
    return json.dump(0u);
}

}// namespace luisa::compute
