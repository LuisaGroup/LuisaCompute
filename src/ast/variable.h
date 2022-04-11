//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <ast/type.h>
#include <ast/usage.h>
#include <serialize/key_value_pair.h>

namespace luisa::compute {

namespace detail {
class FunctionBuilder;
}
class AstSerializer;

/// Variable class
class Variable {
    friend class AstSerializer;

public:
    /// Variable tags
    enum struct Tag : uint32_t {

        // data
        LOCAL,
        SHARED,

        // reference
        REFERENCE,

        // resources
        BUFFER,
        TEXTURE,
        BINDLESS_ARRAY,
        ACCEL,

        // builtins
        THREAD_ID,
        BLOCK_ID,
        DISPATCH_ID,
        DISPATCH_SIZE
    };

private:
    const Type *_type;
    uint32_t _uid;
    Tag _tag;

private:
    friend class detail::FunctionBuilder;
    constexpr Variable(const Type *type, Tag tag, uint32_t uid) noexcept
        : _type{type}, _uid{uid}, _tag{tag} {}

public:
    Variable() noexcept = default;
    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] auto uid() const noexcept { return _uid; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto hash() const noexcept {
        auto u0 = static_cast<uint64_t>(_uid);
        auto u1 = static_cast<uint64_t>(_tag);
        using namespace std::string_view_literals;
        return hash64(u0 | (u1 << 32u), hash64(_type->hash(), hash64("__hash_variable"sv)));
    }
    [[nodiscard]] auto operator==(Variable rhs) const noexcept { return _uid == rhs._uid; }

    template<typename S>
    void save(S& s) {
        luisa::string description(_type->description());
        s.serialize(
            MAKE_NAME_PAIR(_uid), 
            MAKE_NAME_PAIR(_tag), 
            KeyValuePair{"_type", description}
        );
    }

    template<typename S>
    void load(S& s) {
        luisa::string description;
        s.serialize(
            MAKE_NAME_PAIR(_uid), 
            MAKE_NAME_PAIR(_tag), 
            KeyValuePair{"_type", description}
        );
        _type = Type::from(description);
    }
};

}// namespace luisa::compute
