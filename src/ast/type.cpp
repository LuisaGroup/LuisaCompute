#include <cctype>
#include <charconv>
#include <limits>
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

LUISA_AST_API luisa::string make_array_description(luisa::string_view elem, size_t dim) noexcept {
    return luisa::format("array<{},{}>", elem, dim);
}

LUISA_AST_API luisa::string make_struct_description(size_t alignment, std::initializer_list<luisa::string_view> members) noexcept {
    return make_struct_description(alignment, members, {});
}

LUISA_AST_API luisa::string make_struct_description(
    size_t alignment,
    std::initializer_list<luisa::string_view> members,
    luisa::span<const Attribute> attributes) noexcept {
    LUISA_ASSERT(attributes.empty() || attributes.size() == members.size(),
                 "Invalid structure member attribute count {} (expected {}).",
                 attributes.size(), members.size());
    auto desc = luisa::format("struct<{}", alignment);
    auto index = size_t{0u};
    for (auto member : members) {
        desc.push_back(',');
        if (!attributes.empty()) {
            auto &attribute = attributes[index];
            LUISA_ASSERT(attribute.value.empty() || !attribute.key.empty(),
                         "Structure member attribute value requires a key.");
            if (!attribute.key.empty()) {
                desc.push_back('[');
                desc.append(attribute.key);
                if (!attribute.value.empty()) {
                    desc.push_back('(');
                    desc.append(attribute.value);
                    desc.push_back(')');
                }
                desc.push_back(']');
            }
        }
        desc.append(member);
        ++index;
    }
    desc.append(">");
    return desc;
}

LUISA_AST_API luisa::string make_buffer_description(luisa::string_view elem) noexcept {
    return luisa::format("buffer<{}>", elem);
}

struct TypeImpl final : public Type {
    // Type::_tag occupies the first four bytes. Pair it with size before the
    // eight-byte hash so the non-empty base does not grow TypeImpl by padding.
    uint size{};
    uint64_t hash{};
    uint16_t alignment{};
    uint dimension{};
    uint index{};
    luisa::string description{};
    luisa::fixed_vector<const Type *, 1> members{};
    luisa::vector<Attribute> member_attributes{};
};

/// Type registry class
class LUISA_AST_API TypeRegistry {

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
        static auto type_seed = hash_value("__hash_type"sv);
        return hash_value(desc, type_seed);
    };
    [[nodiscard]] auto _register(TypeImpl *type) noexcept {
        LUISA_ASSERT(_types.size() <= std::numeric_limits<uint>::max(),
                     "Too many types registered (maximum is {}).",
                     std::numeric_limits<uint>::max());
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
    [[nodiscard]] const Type *custom_type(luisa::string_view name) noexcept;
    /// Return type count
    [[nodiscard]] size_t type_count() const noexcept;
    /// Traverse all types using visitor
    void traverse(TypeVisitor &visitor) const noexcept;

    static void reset() noexcept {
        for (auto t : instance()._types) {
            std::destroy_at(t);
        }
        instance()._type_pool = luisa::Pool<TypeImpl, false, false>{};
        instance()._type_set = luisa::unordered_set<const TypeImpl *, TypeHash>{};
        instance()._types = luisa::vector<TypeImpl *>{};
    }
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
                     name != "float8e4m3" &&
                     name != "float8e5m2" &&
                     name != "int4" &&
                     name != "fp4e2m1" &&
                     name != "bool" &&
                     !name.starts_with("vector<") &&
                     !name.starts_with("coopvec<") &&
                     !name.starts_with("matrix<") &&
                     !name.starts_with("array<") &&
                     !name.starts_with("struct<") &&
                     !name.starts_with("buffer<") &&
                     !name.starts_with("texture<") &&
                     !name.starts_with("coopvec_ref<") &&
                     !name.starts_with("coopmat_ref<") &&
                     name != "accel" &&
                     name != "bindless_array" &&
                     !std::isdigit(static_cast<unsigned char>(name.front() /* already checked not empty */)),
                 "Invalid custom type name: {}", name);
    LUISA_ASSERT(std::all_of(name.cbegin(), name.cend(),
                             [](char c) {
                                 return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
                             }),
                 "Invalid custom type name: {}", name);
    std::lock_guard lock{_mutex};
    auto h = _compute_hash(name);
    if (auto iter = _type_set.find(TypeDescAndHash{name, h});
        iter != _type_set.end()) { return *iter; }

    auto t = _type_pool.create();
    t->hash = h;
    t->_tag = Type::Tag::CUSTOM;
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
        for (; i < desc.size(); i++) {
            auto c = static_cast<unsigned char>(desc[i]);
            if (!std::isalpha(c) && !std::isdigit(c) && c != '_') { break; }
        }
        auto t = desc.substr(0u, i);
        desc = desc.substr(i);
        return t;
    };

    auto read_number = [&desc]() noexcept {
        size_t number{};
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
    auto try_match = [&desc](char c) noexcept {
        if (!desc.starts_with(c)) {
            return false;
        }
        desc = desc.substr(1);
        return true;
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

    auto checked_dimension = [&info](size_t value, luisa::string_view kind) noexcept {
        if (value > std::numeric_limits<uint>::max()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "{} {} is not representable in type '{}'.",
                kind, value, info->description);
        }
        return static_cast<uint>(value);
    };

    auto checked_layout_product = [&info](size_t lhs, size_t rhs) noexcept {
        constexpr auto max_type_size = static_cast<size_t>(std::numeric_limits<uint>::max());
        if (lhs != 0u && rhs > max_type_size / lhs) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Type layout size overflow in '{}'.",
                info->description);
        }
        return static_cast<uint>(lhs * rhs);
    };

    auto checked_layout_append = [&info](uint offset, size_t alignment, size_t size) noexcept {
        constexpr auto max_type_size = static_cast<size_t>(std::numeric_limits<uint>::max());
        if (alignment == 0u || alignment > max_type_size) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid member alignment {} in type '{}'.",
                alignment, info->description);
        }
        auto wide_offset = static_cast<size_t>(offset);
        auto remainder = wide_offset % alignment;
        auto padding = remainder == 0u ? 0u : alignment - remainder;
        if (padding > max_type_size - wide_offset) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Type layout alignment overflow in '{}'.",
                info->description);
        }
        auto aligned_offset = wide_offset + padding;
        if (size > max_type_size - aligned_offset) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Type layout size overflow in '{}'.",
                info->description);
        }
        return static_cast<uint>(aligned_offset + size);
    };

    auto type_identifier = read_identifier();
#define TRY_PARSE_SCALAR_TYPE(T, TAG, s) \
    if (type_identifier == #T##sv) {     \
        info->_tag = Type::Tag::TAG;     \
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
    TRY_PARSE_SCALAR_TYPE(float8e4m3, FLOAT8_E4M3, 1u)
    TRY_PARSE_SCALAR_TYPE(float8e5m2, FLOAT8_E5M2, 1u)
    // 4-bit sub-byte quantized types: stored as 1 byte per element (the lower
    // nibble holds the value; the upper nibble is zero/unused), matching the
    // host-side tensor_element_type_size_bytes packing (2 elements per byte).
    TRY_PARSE_SCALAR_TYPE(int4, INT4, 1u)
    TRY_PARSE_SCALAR_TYPE(fp4e2m1, FP4_E2M1, 1u)
#undef TRY_PARSE_SCALAR_TYPE
    if (type_identifier == "vector"sv) {
        info->_tag = Type::Tag::VECTOR;
        match('<');
        info->members.emplace_back(_decode(split()));
        match(',');
        auto dimension = read_number();
        match('>');
        auto elem = info->members.front();
        if (elem == nullptr || !elem->is_scalar()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid vector element in type '{}'.",
                info->description);
        }
        if (dimension != 2u && dimension != 3u && dimension != 4u) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid vector dimension: {}.",
                dimension);
        }
        info->dimension = static_cast<uint>(dimension);
        info->alignment = std::min(
            elem->size() * (info->dimension == 3 ? 4 : info->dimension),
            static_cast<size_t>(16u));
        info->size = luisa::align(elem->size() * info->dimension, info->alignment);
    } else if (type_identifier == "matrix"sv) {
        info->_tag = Type::Tag::MATRIX;
        match('<');
        auto dimension = read_number();
        match('>');
        info->members.emplace_back(_decode("float"sv));
        if (dimension == 2u) {
            info->dimension = 2u;
            info->size = sizeof(float2x2);
            info->alignment = alignof(float2x2);
        } else if (dimension == 3u) {
            info->dimension = 3u;
            info->size = sizeof(float3x3);
            info->alignment = alignof(float3x3);
        } else if (dimension == 4u) {
            info->dimension = 4u;
            info->size = sizeof(float4x4);
            info->alignment = alignof(float4x4);
        } else [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid matrix dimension: {}.",
                dimension);
        }
    } else if (type_identifier == "array"sv) {
        info->_tag = Type::Tag::ARRAY;
        match('<');
        info->members.emplace_back(_decode(split()));
        match(',');
        auto dimension = read_number();
        match('>');
        auto m = info->members.back();
        if (m == nullptr || m->is_resource() || m->is_custom() ||
            m->is_cooperative_vector_ref() || m->is_cooperative_matrix_ref()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid array element in type '{}'.",
                info->description);
        }
        info->dimension = checked_dimension(dimension, "Array dimension");
        info->alignment = m->alignment();
        info->size = checked_layout_product(m->size(), dimension);
    } else if (type_identifier == "coopvec"sv) {
        info->_tag = Type::Tag::COOPERATIVE_VECTOR;
        match('<');
        info->members.emplace_back(_decode(split()));
        match(',');
        auto dimension = read_number();
        match('>');
        auto m = info->members.back();
        if (m == nullptr || !m->is_scalar()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid cooperative vector element in type '{}'.",
                info->description);
        }
        info->dimension = checked_dimension(dimension, "Cooperative vector dimension");
        info->alignment = m->alignment();
        info->size = checked_layout_product(m->size(), dimension);
    } else if (type_identifier == "coopvec_ref"sv) {
        info->_tag = Type::Tag::COOPERATIVE_VECTOR_REF;
        match('<');
        auto dimension = read_number();
        match(',');
        auto element_type = read_number();
        if (element_type >= Type::coop_ref_type_size) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Cooperative vector type enum {} is out of range.", element_type);
        }
        match('>');
        info->dimension = checked_dimension(dimension, "Cooperative vector reference dimension");
        info->alignment = static_cast<uint16_t>(element_type);
    } else if (type_identifier == "coopmat_ref"sv) {
        // coopmat_ref<N, M, type>
        info->_tag = Type::Tag::COOPERATIVE_MATRIX_REF;
        match('<');
        auto n = read_number();
        match(',');
        auto m = read_number();
        match(',');
        auto element_type = read_number();
        if (element_type >= Type::coop_ref_type_size) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Cooperative matrix type enum {} is out of range.", element_type);
        }
        match('>');
        info->dimension = checked_dimension(n, "Cooperative matrix row count");
        info->size = checked_dimension(m, "Cooperative matrix column count");
        info->alignment = static_cast<uint16_t>(element_type);
    } else if (type_identifier == "struct"sv) {
        info->_tag = Type::Tag::STRUCTURE;
        match('<');
        auto alignment = read_number();
        if (alignment != 1u && alignment != 4u &&
            alignment != 8u && alignment != 16u) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Invalid structure alignment {}.", alignment);
        }
        info->alignment = static_cast<uint16_t>(alignment);
        while (desc.starts_with(',')) {
            desc = desc.substr(1);
            if (try_match('[')) {
                auto attr_key = read_identifier();
                luisa::string_view attr_value;
                if (try_match('(')) {
                    attr_value = read_identifier();
                    match(')');
                }
                match(']');
                info->member_attributes.resize(info->members.size());
                info->member_attributes.emplace_back(luisa::string{attr_key}, luisa::string{attr_value});
            }
            // TODO: match attribute
            info->members.emplace_back(_decode(split()));
        }
        if (!info->member_attributes.empty()) {
            info->member_attributes.resize(info->members.size());
        }
        match('>');
        auto layout_size = 0u;
        auto max_member_alignment = static_cast<size_t>(0u);
        for (auto member : info->members) {
            if (member == nullptr || member->is_resource() || member->is_custom() ||
                member->is_cooperative_vector_ref() || member->is_cooperative_matrix_ref()) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION(
                    "Invalid structure member in type '{}'.",
                    info->description);
            }
            auto ma = member->alignment();
            max_member_alignment = std::max(ma, max_member_alignment);
            layout_size = checked_layout_append(layout_size, ma, member->size());
        }
        if (alignment < max_member_alignment) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Struct alignment {} is smaller than the largest member alignment {}.",
                alignment, max_member_alignment);
        }
        info->size = checked_layout_append(layout_size, alignment, 0u);
    } else if (type_identifier == "buffer"sv) {
        info->_tag = Type::Tag::BUFFER;
        while (try_match('[')) {
            auto attr_key = read_identifier();
            luisa::string_view attr_value;
            if (try_match('(')) {
                attr_value = read_identifier();
                match(')');
            }
            match(']');
            info->member_attributes.emplace_back(luisa::string{attr_key}, luisa::string{attr_value});
        }
        match('<');
        auto m = info->members.emplace_back(_decode(split()));
        match('>');
        if (m) {
            if (m->is_resource() || m->is_cooperative_vector() ||
                m->is_cooperative_vector_ref() || m->is_cooperative_matrix_ref()) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION(
                    "Invalid buffer element in type '{}'.",
                    info->description);
            }
            if (m->is_structure() && !m->member_attributes().empty()) {
                LUISA_ERROR_WITH_LOCATION(
                    "Buffers are not allowed to "
                    "hold structure with attributes.");
            }
        }
        info->alignment = 8u;
        info->size = 8u;
    } else if (type_identifier == "texture"sv) {
        info->_tag = Type::Tag::TEXTURE;
        while (try_match('[')) {
            auto attr_key = read_identifier();
            luisa::string_view attr_value;
            if (try_match('(')) {
                attr_value = read_identifier();
                match(')');
            }
            match(']');
            info->member_attributes.emplace_back(luisa::string{attr_key}, luisa::string{attr_value});
        }
        match('<');
        auto dimension = read_number();
        match(',');
        auto m = info->members.emplace_back(_decode(split()));
        match('>');
        if (dimension != 2u && dimension != 3u) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Invalid texture dimension: {}.", dimension);
        }
        if (m == nullptr) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Invalid texture element in type '{}'.", info->description);
        }
        if (auto t = m->tag();
            t != Type::Tag::INT32 &&
            t != Type::Tag::UINT32 &&
            t != Type::Tag::FLOAT32) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Images can only hold int32, uint32, or float32.");
        }
        info->dimension = static_cast<uint>(dimension);
        info->size = 8u;
        info->alignment = 8u;
    } else if (type_identifier == "bindless_array"sv) {
        info->_tag = Type::Tag::BINDLESS_ARRAY;
        info->size = 8u;
        info->alignment = 8u;
    } else if (type_identifier == "accel"sv) {
        info->_tag = Type::Tag::ACCEL;
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

luisa::span<Type const *const> Type::members() const noexcept {
    LUISA_ASSERT(is_structure(),
                 "Calling members() on a non-structure type {}.",
                 description());
    return static_cast<const detail::TypeImpl *>(this)->members;
}

luisa::span<const Attribute> Type::member_attributes() const noexcept {
    LUISA_ASSERT(is_structure() || is_buffer() || is_texture(),
                 "Calling member_attributes() on a non-structure, buffer or texture type {}.",
                 description());
    return static_cast<const detail::TypeImpl *>(this)->member_attributes;
}

const Type *Type::element() const noexcept {
    if (is_scalar()) { return this; }
    LUISA_ASSERT(is_array() || is_cooperative_vector() || is_vector() || is_matrix() || is_buffer() || is_texture(),
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
    luisa::function<void(const Type *)> _visitor{};

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
    LUISA_ASSERT(!is_resource() && !is_custom() && !is_cooperative_vector_ref() && !is_cooperative_matrix_ref(),
                 "Trying to take size of backend-specific type.");
    return static_cast<const detail::TypeImpl *>(this)->size;
}

size_t Type::alignment() const noexcept {
    LUISA_ASSERT(!is_resource() && !is_custom() && !is_cooperative_vector_ref() && !is_cooperative_matrix_ref(),
                 "Trying to take alignment of backend-specific type.");
    return static_cast<const detail::TypeImpl *>(this)->alignment;
}

luisa::string_view Type::description() const noexcept {
    return static_cast<const detail::TypeImpl *>(this)->description;
}

uint Type::dimension() const noexcept {
    LUISA_ASSERT(is_scalar() || is_array() || is_cooperative_vector() || is_cooperative_vector_ref() || is_vector() || is_matrix() || is_texture(),
                 "Calling dimension() on a non-array, non-vector, "
                 "non-matrix, or non-image type {}.",
                 description());
    return static_cast<const detail::TypeImpl *>(this)->dimension;
}

uint2 Type::coop_matrix_dimension() const noexcept {
    LUISA_ASSERT(is_cooperative_matrix_ref(), "Calling coop_matrix_dimension() on a non-cooperative-matrix {}", description());
    auto impl = static_cast<const detail::TypeImpl *>(this);
    return {impl->dimension, impl->size};
}

auto Type::coop_vec_ref_type() const noexcept -> CoopRefVecType {
    LUISA_ASSERT(is_cooperative_vector_ref() || is_cooperative_matrix_ref(), "Calling coop_vec_ref_type() on a non-cooperative vector ref");
    return static_cast<CoopRefVecType>(static_cast<const detail::TypeImpl *>(this)->alignment);
}

const Type *Type::array(const Type *elem, size_t n) noexcept {
    LUISA_ASSERT(elem != nullptr && !elem->is_resource() && !elem->is_custom() &&
                     !elem->is_cooperative_vector_ref() && !elem->is_cooperative_matrix_ref(),
                 "Array element must be a data type.");
    LUISA_ASSERT(n <= std::numeric_limits<uint>::max(),
                 "Array dimension {} is too large.", n);
    return from(luisa::format("array<{},{}>", elem->description(), n));
}

const Type *Type::cooperative_vector(const Type *elem, size_t n) noexcept {
    LUISA_ASSERT(elem != nullptr && elem->is_scalar(),
                 "Cooperative vector element must be a scalar type.");
    LUISA_ASSERT(n <= std::numeric_limits<uint>::max(),
                 "Cooperative vector dimension {} is too large.", n);
    return from(luisa::format("coopvec<{},{}>", elem->description(), n));
}
const Type *Type::cooperative_vector_ref(CoopRefVecType type, size_t n) noexcept {
    LUISA_ASSERT(luisa::to_underlying(type) < Type::coop_ref_type_size,
                 "Cooperative vector type enum out of range.");
    LUISA_ASSERT(n <= std::numeric_limits<uint>::max(),
                 "Cooperative vector reference dimension {} is too large.", n);
    return from(luisa::format("coopvec_ref<{},{}>", n, luisa::to_underlying(type)));
}
const Type *Type::cooperative_matrix_ref(CoopRefVecType type, size_t n, size_t m) noexcept {
    LUISA_ASSERT(luisa::to_underlying(type) < Type::coop_ref_type_size,
                 "Cooperative matrix type enum out of range.");
    LUISA_ASSERT(n <= std::numeric_limits<uint>::max() &&
                     m <= std::numeric_limits<uint>::max(),
                 "Cooperative matrix dimensions {} x {} are too large.", n, m);
    return from(luisa::format("coopmat_ref<{},{},{}>", n, m, luisa::to_underlying(type)));
}

const Type *Type::vector(const Type *elem, size_t n) noexcept {
    LUISA_ASSERT(n >= 2 && n <= 4, "Invalid vector dimension.");
    LUISA_ASSERT(elem != nullptr && elem->is_scalar(), "Vector element must be a scalar.");
    return from(luisa::format("vector<{},{}>", elem->description(), n));
}

const Type *Type::matrix(size_t n) noexcept {
    LUISA_ASSERT(n >= 2 && n <= 4, "Invalid matrix dimension.");
    return from(luisa::format("matrix<{}>", n));
}

const Type *Type::buffer(const Type *elem, luisa::span<const Attribute> attributes) noexcept {
    LUISA_ASSERT(elem == nullptr || !elem->is_resource(),
                 "Buffer element cannot be a resource type.");
    LUISA_ASSERT(elem == nullptr ||
                     (!elem->is_cooperative_vector() &&
                      !elem->is_cooperative_vector_ref() &&
                      !elem->is_cooperative_matrix_ref()),
                 "Buffer cannot hold cooperative data-structure.");
    LUISA_ASSERT(elem == nullptr || !elem->is_structure() || elem->member_attributes().empty(),
                 "Buffer cannot hold structure with custom attributes.");
    auto element_description = elem == nullptr ? luisa::string_view{"void"} : elem->description();
    if (!attributes.empty()) [[unlikely]] /*usually would not use attribute*/ {
        luisa::string r{"buffer"};
        for (auto &attr : attributes) {
            if (!attr) continue;
            if (attr.value.empty()) {
                r.append(luisa::format("[{}]", attr.key));
            } else {
                r.append(luisa::format("[{}({})]", attr.key, attr.value));
            }
        }
        r.append(luisa::format("<{}>", element_description));
        return from(r);
    } else {
        return from(luisa::format("buffer<{}>", element_description));
    }
}

const Type *Type::texture(const Type *elem, size_t dimension, luisa::span<const Attribute> attributes) noexcept {
    LUISA_ASSERT(elem != nullptr, "Texture element must not be void.");
    if (elem->is_vector()) { elem = elem->element(); }
    LUISA_ASSERT(elem->is_int32() || elem->is_uint32() || elem->is_float32(),
                 "Texture element must be int32, uint32, or float32, but got {}.",
                 elem->description());
    LUISA_ASSERT(dimension == 2u || dimension == 3u, "Texture dimension must be 2 or 3");
    if (!attributes.empty()) [[unlikely]] /*usually would not use attribute*/ {
        luisa::string r{"texture"};
        for (auto &attr : attributes) {
            if (!attr) continue;
            if (attr.value.empty()) {
                r.append(luisa::format("[{}]", attr.key));
            } else {
                r.append(luisa::format("[{}({})]", attr.key, attr.value));
            }
        }
        r.append(luisa::format("<{},{}>", dimension, elem->description()));
        return from(r);
    } else {
        return from(luisa::format("texture<{},{}>", dimension, elem->description()));
    }
}

const Type *Type::structure(size_t alignment, luisa::span<Type const *const> members, luisa::span<const Attribute> attributes) noexcept {
    LUISA_ASSERT(alignment == 1 || alignment == 4u || alignment == 8u || alignment == 16u,
                 "Invalid structure alignment {} (must be 1, 4, 8 or 16).",
                 alignment);
    LUISA_ASSERT(attributes.empty() || attributes.size() == members.size(),
                 "Invalid attribute size (must be empty or same as members' size");
    auto desc = luisa::format("struct<{}", alignment);
    for (auto member : members) {
        LUISA_ASSERT(member != nullptr && !member->is_resource() && !member->is_custom() &&
                         !member->is_cooperative_vector_ref() && !member->is_cooperative_matrix_ref(),
                     "Structure members must be data types.");
    }
    if (!attributes.empty()) [[unlikely]] /*usually would not use attribute*/ {
        for (size_t i = 0; i < members.size(); ++i) {
            desc.append(",");
            auto &a = attributes[i];
            if (!a.key.empty()) {
                desc.append("[").append(a.key);
                if (!a.value.empty()) {
                    desc.append("(").append(a.value).append(")");
                }
                desc.append("]");
            }
            desc.append(members[i]->description());
        }

    } else {
        for (auto member : members) {
            desc.append(",").append(member->description());
        }
    }
    desc.append(">");
    return from(desc);
}

const Type *Type::structure(luisa::span<Type const *const> members, luisa::span<const Attribute> attributes) noexcept {
    auto alignment = 4u;
    for (auto m : members) {
        LUISA_ASSERT(m != nullptr && !m->is_resource() && !m->is_custom() &&
                         !m->is_cooperative_vector_ref() && !m->is_cooperative_matrix_ref(),
                     "Structure members must be data types.");
        alignment = std::max<size_t>(m->alignment(), alignment);
    }
    return structure(alignment, members, attributes);
}

const Type *Type::structure(size_t alignment, std::initializer_list<const Type *> members, luisa::span<const Attribute> attributes) noexcept {
    return structure(alignment, luisa::span{members.begin(), members.size()}, attributes);
}

const Type *Type::structure(std::initializer_list<const Type *> members, luisa::span<const Attribute> attributes) noexcept {
    return structure(luisa::span{members.begin(), members.size()}, attributes);
}

const Type *Type::custom(luisa::string_view name) noexcept {
    return detail::TypeRegistry::instance().custom_type(name);
}

bool Type::is_bool_or_bool_vector() const noexcept { return is_bool() || is_bool_vector(); }
bool Type::is_int_or_int_vector() const noexcept { return is_int() || is_int_vector(); }
bool Type::is_uint_or_uint_vector() const noexcept { return is_uint() || is_uint_vector(); }
bool Type::is_float_or_float_vector() const noexcept { return is_float() || is_float_vector(); }

bool Type::is_int_vector() const noexcept { return is_vector() && element()->is_int(); }
bool Type::is_uint_vector() const noexcept { return is_vector() && element()->is_uint(); }
bool Type::is_float_vector() const noexcept { return is_vector() && element()->is_float(); }

bool Type::is_bool_vector() const noexcept { return is_vector() && element()->is_bool(); }
bool Type::is_int32_vector() const noexcept { return is_vector() && element()->is_int32(); }
bool Type::is_uint32_vector() const noexcept { return is_vector() && element()->is_uint32(); }
bool Type::is_float16_vector() const noexcept { return is_vector() && element()->is_float16(); }
bool Type::is_float32_vector() const noexcept { return is_vector() && element()->is_float32(); }
bool Type::is_float64_vector() const noexcept { return is_vector() && element()->is_float64(); }
bool Type::is_int8_vector() const noexcept { return is_vector() && element()->is_int8(); }
bool Type::is_uint8_vector() const noexcept { return is_vector() && element()->is_uint8(); }
bool Type::is_int16_vector() const noexcept { return is_vector() && element()->is_int16(); }
bool Type::is_uint16_vector() const noexcept { return is_vector() && element()->is_uint16(); }
bool Type::is_int64_vector() const noexcept { return is_vector() && element()->is_int64(); }
bool Type::is_uint64_vector() const noexcept { return is_vector() && element()->is_uint64(); }

void Type::reset_type_registry() noexcept {
    ::luisa::compute::detail::TypeRegistry::reset();
}

}// namespace luisa::compute
