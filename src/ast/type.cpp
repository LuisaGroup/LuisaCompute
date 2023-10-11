#include <bit>
#include <charconv>
#include <utility>
#include <algorithm>

#include <luisa/core/pool.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/hash.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/logging.h>
#include <luisa/ast/type_registry.h>
#include <luisa/ast/type.h>

namespace luisa::compute {

namespace detail {

LC_AST_API luisa::string make_array_description(luisa::string_view elem, size_t dim) noexcept {
    return luisa::format("array<{},{}>", elem, dim);
}

LC_AST_API luisa::string make_struct_description(size_t alignment, std::initializer_list<luisa::string_view> members) noexcept {
    auto desc = luisa::format("struct<{}", alignment);
    for (auto m : members) { desc.append(",").append(m); }
    desc.append(">");
    return desc;
}

LC_AST_API luisa::string make_buffer_description(luisa::string_view elem) noexcept {
    return luisa::format("buffer<{}>", elem);
}

struct TypeImpl final : public Type {
    uint64_t hash{};
    Tag tag{};
    uint size{};
    uint16_t alignment{};
    uint16_t dimension{};
    uint index{};
    luisa::string description;
    luisa::vector<const Type *> members;
};

/// Type registry class
class LC_AST_API TypeRegistry {

public:
    struct TypeDescAndHash {
        string_view desc;
        uint64_t hash;
        [[nodiscard]] auto operator==(TypeDescAndHash rhs) const noexcept {
            return hash == rhs.hash && desc == rhs.desc;
        }
        [[nodiscard]] auto operator==(const TypeImpl *rhs) const noexcept {
            return *this == TypeDescAndHash{rhs->description, rhs->hash};
        }
    };

    struct TypeHash {
        using is_avalanching = void;
        using is_transparent = void;
        [[nodiscard]] uint64_t operator()(const Type *type) const noexcept { return type->hash(); }
        [[nodiscard]] uint64_t operator()(TypeDescAndHash const &desc) const noexcept { return desc.hash; }
    };

private:
    luisa::Pool<TypeImpl, false, false> _type_pool;
    luisa::vector<TypeImpl *> _types;
    luisa::unordered_set<const TypeImpl *, TypeHash> _type_set;
    mutable std::recursive_mutex _mutex;

private:
    [[nodiscard]] const TypeImpl *_decode(luisa::string_view desc) noexcept;
    [[nodiscard]] static auto _compute_hash(luisa::string_view desc) noexcept {
        using namespace std::string_view_literals;
        static auto seed = hash_value("__hash_type"sv);
        return hash_value(desc, seed);
    };
    [[nodiscard]] auto _register(TypeImpl *type) noexcept {
        type->index = static_cast<uint32_t>(_types.size());
        auto ret = _type_set.emplace(type);
        if (ret.second) [[likely]] {
            _types.emplace_back(type);
        } else {
            _type_pool.destroy(type);
        }
        return *ret.first;
    }

public:
    ~TypeRegistry() noexcept {
        for (auto t : _types) {
            std::destroy_at(t);
        }
    }
    /// Get registry instance
    [[nodiscard]] static TypeRegistry &instance() noexcept {
        static TypeRegistry registry;
        return registry;
    }
    /// Construct Type object from description
    [[nodiscard]] const Type *decode_type(luisa::string_view desc) noexcept;
    /// Construct custom type
    [[nodiscard]] const Type *custom_type(luisa::string_view desc) noexcept;
    /// Return type count
    [[nodiscard]] size_t type_count() const noexcept;
    /// Traverse all types using visitor
    void traverse(TypeVisitor &visitor) const noexcept;
};

const Type *TypeRegistry::decode_type(luisa::string_view desc) noexcept {
    using namespace std::literals;
    if (desc == "void"sv) { return nullptr; }
    std::lock_guard lock{_mutex};
    return _decode(desc);
}

const Type *TypeRegistry::custom_type(luisa::string_view name) noexcept {
    // validate name
    LUISA_ASSERT(!name.empty() &&
                     name != "void" &&
                     name != "int" &&
                     name != "uint" &&
                     name != "short" &&
                     name != "byte" &&
                     name != "ubyte" &&
                     name != "ushort" &&
                     name != "long" &&
                     name != "ulong" &&
                     name != "float" &&
                     name != "half" &&
                     name != "double" &&
                     name != "bool" &&
                     !name.starts_with("vector<") &&
                     !name.starts_with("matrix<") &&
                     !name.starts_with("array<") &&
                     !name.starts_with("struct<") &&
                     !name.starts_with("buffer<") &&
                     !name.starts_with("texture<") &&
                     name != "accel" &&
                     name != "bindless_array" &&
                     !isdigit(name.front() /* already checked not empty */),
                 "Invalid custom type name: {}", name);
    LUISA_ASSERT(std::all_of(name.cbegin(), name.cend(),
                             [](char c) { return isalnum(c) || c == '_'; }),
                 "Invalid custom type name: {}", name);
    std::lock_guard lock{_mutex};
    auto h = _compute_hash(name);
    if (auto iter = _type_set.find(TypeDescAndHash{name, h});
        iter != _type_set.end()) { return *iter; }

    auto t = _type_pool.create();
    t->hash = h;
    t->tag = Type::Tag::CUSTOM;
    t->size = Type::custom_struct_size;
    t->alignment = Type::custom_struct_alignment;
    t->dimension = 1u;
    t->description = name;
    return _register(t);
}

size_t TypeRegistry::type_count() const noexcept {
    std::lock_guard lock{_mutex};
    return _types.size();
}

void TypeRegistry::traverse(TypeVisitor &visitor) const noexcept {
    std::unique_lock lock{_mutex};
    for (auto &&t : _types) {
        visitor.visit(t);
    }
}

const TypeImpl *TypeRegistry::_decode(luisa::string_view desc) noexcept {
    if (desc == "void") [[unlikely]] {
        return nullptr;
    }
    auto hash = _compute_hash(desc);
    if (auto iter = _type_set.find(TypeDescAndHash{desc, hash});
        iter != _type_set.cend()) { return *iter; }

    using namespace std::string_view_literals;
    auto read_identifier = [&desc]() noexcept {
        auto i = 0u;
        for (; i < desc.size() && (isalpha(desc[i]) || isdigit(desc[i]) || desc[i] == '_'); i++) {}
        auto t = desc.substr(0u, i);
        desc = desc.substr(i);
        return t;
    };

    auto read_number = [&desc]() noexcept {
        size_t number;
        auto result = std::from_chars(desc.data(), desc.data() + desc.size(), number);
        if (result.ec != std::errc{}) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Failed to parse number from type description: '{}'.",
                desc);
        }
        desc = desc.substr(result.ptr - desc.data());
        return number;
    };

    auto match = [&desc](char c) noexcept {
        if (!desc.starts_with(c)) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Expected '{}' from type description: '{}'.",
                c, desc);
        }
        desc = desc.substr(1);
    };

    auto split = [&desc]() noexcept {
        auto balance = 0u;
        auto i = 0u;
        for (; i < desc.size(); i++) {
            if (auto c = desc[i]; c == '<') {
                balance++;
            } else if (c == '>') {
                if (balance == 0u) { break; }
                if (--balance == 0u) {
                    i++;
                    break;
                }
            } else if (c == ',' && balance == 0u) {
                break;
            }
        }
        if (balance != 0u) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Unbalanced '<' and '>' in "
                "type description: {}.",
                desc);
        }
        auto t = desc.substr(0u, i);
        desc = desc.substr(i);
        return t;
    };

    auto info = _type_pool.create();
    info->description = desc;
    info->hash = hash;

    auto type_identifier = read_identifier();
#define TRY_PARSE_SCALAR_TYPE(T, TAG, s) \
    if (type_identifier == #T##sv) {     \
        info->tag = Type::Tag::TAG;      \
        info->size = s;                  \
        info->alignment = s;             \
        info->dimension = 1u;            \
    } else
    TRY_PARSE_SCALAR_TYPE(bool, BOOL, 1u)
    TRY_PARSE_SCALAR_TYPE(byte, INT8, 1u)
    TRY_PARSE_SCALAR_TYPE(ubyte, UINT8, 1u)
    TRY_PARSE_SCALAR_TYPE(short, INT16, 2u)
    TRY_PARSE_SCALAR_TYPE(ushort, UINT16, 2u)
    TRY_PARSE_SCALAR_TYPE(int, INT32, 4u)
    TRY_PARSE_SCALAR_TYPE(uint, UINT32, 4u)
    TRY_PARSE_SCALAR_TYPE(long, INT64, 8u)
    TRY_PARSE_SCALAR_TYPE(ulong, UINT64, 8u)
    TRY_PARSE_SCALAR_TYPE(half, FLOAT16, 2u)
    TRY_PARSE_SCALAR_TYPE(float, FLOAT32, 4u)
    TRY_PARSE_SCALAR_TYPE(double, FLOAT64, 8u)
#undef TRY_PARSE_SCALAR_TYPE
    if (type_identifier == "vector"sv) {
        info->tag = Type::Tag::VECTOR;
        match('<');
        info->members.emplace_back(_decode(split()));
        match(',');
        info->dimension = read_number();
        match('>');
        auto elem = info->members.front();
        if (!elem->is_scalar()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid vector element: {}.",
                elem->description());
        }
        if (info->dimension != 2 &&
            info->dimension != 3 &&
            info->dimension != 4) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid vector dimension: {}.",
                info->dimension);
        }
        info->alignment = std::min(
            elem->size() * (info->dimension == 3 ? 4 : info->dimension),
            static_cast<size_t>(16u));
        info->size = luisa::align(elem->size() * info->dimension, info->alignment);
    } else if (type_identifier == "matrix"sv) {
        info->tag = Type::Tag::MATRIX;
        match('<');
        info->dimension = read_number();
        match('>');
        info->members.emplace_back(_decode("float"sv));
        if (info->dimension == 2) {
            info->size = sizeof(float2x2);
            info->alignment = alignof(float2x2);
        } else if (info->dimension == 3) {
            info->size = sizeof(float3x3);
            info->alignment = alignof(float3x3);
        } else if (info->dimension == 4) {
            info->size = sizeof(float4x4);
            info->alignment = alignof(float4x4);
        } else [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid matrix dimension: {}.",
                info->dimension);
        }
    } else if (type_identifier == "array"sv) {
        info->tag = Type::Tag::ARRAY;
        match('<');
        info->members.emplace_back(_decode(split()));
        match(',');
        info->dimension = read_number();
        match('>');
        if (info->members.back()->is_buffer() ||
            info->members.back()->is_texture()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Arrays are not allowed to "
                "hold buffers or images.");
        }
        info->alignment = info->members.front()->alignment();
        info->size = info->members.front()->size() * info->dimension;
    } else if (type_identifier == "struct"sv) {
        info->tag = Type::Tag::STRUCTURE;
        match('<');
        info->alignment = read_number();
        while (desc.starts_with(',')) {
            desc = desc.substr(1);
            info->members.emplace_back(_decode(split()));
        }
        match('>');
        info->size = 0u;
        auto max_member_alignment = static_cast<size_t>(0u);
        for (auto member : info->members) {
            if (member->is_buffer() || member->is_texture()) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION(
                    "Structures are not allowed to have buffers or images as members.");
            }
            auto ma = member->alignment();
            max_member_alignment = std::max(ma, max_member_alignment);
            info->size = (info->size + ma - 1u) / ma * ma + member->size();
        }
        if (auto a = info->alignment; a > 16u || std::bit_floor(a) != a) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Invalid structure alignment {}.", a);
        } else if (a < max_member_alignment && a != 0u) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Struct alignment {} is smaller than the largest member alignment {}.",
                info->alignment, max_member_alignment);
        }
        info->size = (info->size + info->alignment - 1u) / info->alignment * info->alignment;
    } else if (type_identifier == "buffer"sv) {
        info->tag = Type::Tag::BUFFER;
        match('<');
        auto m = info->members.emplace_back(_decode(split()));
        match('>');
        if (m && (m->is_buffer() || m->is_texture())) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Buffers are not allowed to "
                "hold buffers or images.");
        }
        info->alignment = 8u;
        info->size = 8u;
    } else if (type_identifier == "texture"sv) {
        info->tag = Type::Tag::TEXTURE;
        match('<');
        info->dimension = read_number();
        match(',');
        auto m = info->members.emplace_back(_decode(split()));
        match('>');
        if (auto t = m->tag();
            t != Type::Tag::INT32 &&
            t != Type::Tag::UINT32 &&
            t != Type::Tag::FLOAT32) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Images can only hold int32, uint32, or float32.");
        }
        info->size = 8u;
        info->alignment = 8u;
    } else if (type_identifier == "bindless_array"sv) {
        info->tag = Type::Tag::BINDLESS_ARRAY;
        info->size = 8u;
        info->alignment = 8u;
    } else if (type_identifier == "accel"sv) {
        info->tag = Type::Tag::ACCEL;
        info->size = 8u;
        info->alignment = 8u;
    } else [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Unknown type identifier: {}.",
            type_identifier);
    }
    if (!desc.empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Found junk after type description: {}.",
            desc);
    }
    return _register(info);
}

}// namespace detail

luisa::span<const Type *const> Type::members() const noexcept {
    LUISA_ASSERT(is_structure(),
                 "Calling members() on a non-structure type {}.",
                 description());
    return static_cast<const detail::TypeImpl *>(this)->members;
}

const Type *Type::element() const noexcept {
    if (is_scalar()) { return this; }
    LUISA_ASSERT(is_array() || is_vector() || is_matrix() || is_buffer() || is_texture(),
                 "Calling element() on a non-array/vector/matrix/buffer/image type {}.",
                 description());
    return static_cast<const detail::TypeImpl *>(this)->members.front();
}

const Type *Type::from(std::string_view description) noexcept {
    return detail::TypeRegistry::instance().decode_type(description);
}

size_t Type::count() noexcept {
    return detail::TypeRegistry::instance().type_count();
}

void Type::traverse(TypeVisitor &visitor) noexcept {
    detail::TypeRegistry::instance().traverse(visitor);
}

class TypeVisitorAdapter final : public TypeVisitor {

private:
    luisa::function<void(const Type *)> _visitor;

public:
    explicit TypeVisitorAdapter(luisa::function<void(const Type *)> visitor) noexcept
        : _visitor(std::move(visitor)) {}
    void visit(const Type *type) noexcept override { _visitor(type); }
};

void Type::traverse(const function<void(const Type *)> &visitor) noexcept {
    TypeVisitorAdapter adapter{visitor};
    traverse(adapter);
}

bool Type::operator==(const Type &rhs) const noexcept {
    return hash() == rhs.hash() /* short path */ &&
           description() == rhs.description();
}

bool Type::operator<(const Type &rhs) const noexcept {
    return index() < rhs.index();
}

uint Type::index() const noexcept {
    return static_cast<const detail::TypeImpl *>(this)->index;
}

uint64_t Type::hash() const noexcept {
    return static_cast<const detail::TypeImpl *>(this)->hash;
}

size_t Type::size() const noexcept {
    return static_cast<const detail::TypeImpl *>(this)->size;
}

size_t Type::alignment() const noexcept {
    return static_cast<const detail::TypeImpl *>(this)->alignment;
}

Type::Tag Type::tag() const noexcept {
    return static_cast<const detail::TypeImpl *>(this)->tag;
}

luisa::string_view Type::description() const noexcept {
    return static_cast<const detail::TypeImpl *>(this)->description;
}

uint Type::dimension() const noexcept {
    LUISA_ASSERT(is_scalar() || is_array() || is_vector() || is_matrix() || is_texture(),
                 "Calling dimension() on a non-array, non-vector, "
                 "non-matrix, or non-image type {}.",
                 description());
    return static_cast<const detail::TypeImpl *>(this)->dimension;
}

bool Type::is_scalar() const noexcept {
    switch (tag()) {
        case Tag::BOOL:
        case Tag::FLOAT32:
        case Tag::INT32:
        case Tag::UINT32:
        case Tag::INT64:
        case Tag::UINT64:
        case Tag::FLOAT16:
        case Tag::INT16:
        case Tag::UINT16:
        case Tag::FLOAT64:
            return true;
        default:
            return false;
    }
}

bool Type::is_arithmetic() const noexcept {
    switch (tag()) {
        case Tag::FLOAT32:
        case Tag::INT32:
        case Tag::UINT32:
        case Tag::INT64:
        case Tag::UINT64:
        case Tag::FLOAT16:
        case Tag::INT16:
        case Tag::UINT16:
            return true;
        default:
            return false;
    }
}

bool Type::is_basic() const noexcept {
    return is_scalar() || is_vector() || is_matrix();
}

bool Type::is_array() const noexcept { return tag() == Tag::ARRAY; }
bool Type::is_vector() const noexcept { return tag() == Tag::VECTOR; }
bool Type::is_matrix() const noexcept { return tag() == Tag::MATRIX; }
bool Type::is_structure() const noexcept { return tag() == Tag::STRUCTURE; }
bool Type::is_buffer() const noexcept { return tag() == Tag::BUFFER; }
bool Type::is_texture() const noexcept { return tag() == Tag::TEXTURE; }
bool Type::is_bindless_array() const noexcept { return tag() == Tag::BINDLESS_ARRAY; }
bool Type::is_accel() const noexcept { return tag() == Tag::ACCEL; }
bool Type::is_custom() const noexcept { return tag() == Tag::CUSTOM; }

const Type *Type::array(const Type *elem, size_t n) noexcept {
    return from(luisa::format("array<{},{}>", elem->description(), n));
}

const Type *Type::vector(const Type *elem, size_t n) noexcept {
    LUISA_ASSERT(n >= 2 && n <= 4, "Invalid vector dimension.");
    LUISA_ASSERT(elem->is_scalar(), "Vector element must be a scalar.");
    return from(luisa::format("vector<{},{}>", elem->description(), n));
}

const Type *Type::matrix(size_t n) noexcept {
    LUISA_ASSERT(n >= 2 && n <= 4, "Invalid matrix dimension.");
    return from(luisa::format("matrix<{}>", n));
}

const Type *Type::buffer(const Type *elem) noexcept {
    LUISA_ASSERT(!elem->is_buffer() && !elem->is_texture(), "Buffer cannot hold buffers or images.");
    return from(luisa::format("buffer<{}>", elem->description()));
}

const Type *Type::texture(const Type *elem, size_t dimension) noexcept {
    if (elem->is_vector()) { elem = elem->element(); }
    LUISA_ASSERT(elem->is_arithmetic(),
                 "Texture element must be an arithmetic, but got {}.",
                 elem->description());
    LUISA_ASSERT(dimension == 2u || dimension == 3u, "Texture dimension must be 2 or 3");
    return from(luisa::format("texture<{},{}>", dimension, elem->description()));
}

const Type *Type::structure(size_t alignment, luisa::span<const Type *> members) noexcept {
    LUISA_ASSERT(alignment == 4u || alignment == 8u || alignment == 16u,
                 "Invalid structure alignment {} (must be 4, 8, or 16).",
                 alignment);
    auto desc = luisa::format("struct<{}", alignment);
    for (auto member : members) {
        desc.append(",").append(member->description());
    }
    desc.append(">");
    return from(desc);
}

const Type *Type::structure(luisa::span<const Type *> members) noexcept {
    auto alignment = 4u;
    for (auto m : members) { alignment = std::max<size_t>(m->alignment(), alignment); }
    return structure(alignment, members);
}

const Type *Type::structure(size_t alignment, std::initializer_list<const Type *> members) noexcept {
    LUISA_ASSERT(alignment == 4u || alignment == 8u || alignment == 16u,
                 "Invalid structure alignment {} (must be 4, 8, or 16).",
                 alignment);
    auto desc = luisa::format("struct<{}", alignment);
    for (auto member : members) {
        desc.append(",").append(member->description());
    }
    desc.append(">");
    return from(desc);
}

const Type *Type::structure(std::initializer_list<const Type *> members) noexcept {
    auto alignment = 4u;
    for (auto m : members) { alignment = std::max<size_t>(m->alignment(), alignment); }
    return structure(alignment, members);
}

const Type *Type::custom(luisa::string_view name) noexcept {
    return detail::TypeRegistry::instance().custom_type(name);
}

bool Type::is_bool() const noexcept { return tag() == Tag::BOOL; }
bool Type::is_int32() const noexcept { return tag() == Tag::INT32; }
bool Type::is_uint32() const noexcept { return tag() == Tag::UINT32; }
bool Type::is_float32() const noexcept { return tag() == Tag::FLOAT32; }
bool Type::is_int16() const noexcept { return tag() == Tag::INT16; }
bool Type::is_uint16() const noexcept { return tag() == Tag::UINT16; }
bool Type::is_int64() const noexcept { return tag() == Tag::INT64; }
bool Type::is_uint64() const noexcept { return tag() == Tag::UINT64; }
bool Type::is_float16() const noexcept { return tag() == Tag::FLOAT16; }

bool Type::is_bool_vector() const noexcept { return is_vector() && element()->is_bool(); }
bool Type::is_int32_vector() const noexcept { return is_vector() && element()->is_int32(); }
bool Type::is_uint32_vector() const noexcept { return is_vector() && element()->is_uint32(); }
bool Type::is_float32_vector() const noexcept { return is_vector() && element()->is_float32(); }
bool Type::is_int16_vector() const noexcept { return is_vector() && element()->is_int16(); }
bool Type::is_uint16_vector() const noexcept { return is_vector() && element()->is_uint16(); }
bool Type::is_float16_vector() const noexcept { return is_vector() && element()->is_float16(); }
bool Type::is_int64_vector() const noexcept { return is_vector() && element()->is_int64(); }
bool Type::is_uint64_vector() const noexcept { return is_vector() && element()->is_uint64(); }

bool Type::is_resource() const noexcept {
    return is_buffer() || is_texture() || is_bindless_array() || is_accel();
}

}// namespace luisa::compute
