//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <ast/type.h>
#include <ast/usage.h>

namespace luisa {

namespace compute {
namespace detail {
class FunctionBuilder;
class SSABuilder;
}// namespace detail

/// Variable class
class Variable {

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
    friend class detail::SSABuilder;
    constexpr Variable(const Type *type, Tag tag, uint32_t uid) noexcept
        : _type{type}, _uid{uid}, _tag{tag} {}

public:
    Variable() noexcept = default;
    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] auto uid() const noexcept { return _uid; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] uint64_t hash() const noexcept;
    [[nodiscard]] auto operator==(Variable rhs) const noexcept { return _uid == rhs._uid; }
};

}// namespace compute

template<>
struct hash<compute::Variable> {
    using is_avalanching = void;
    [[nodiscard]] auto operator()(compute::Variable v) const noexcept { return v.hash(); }
};

}// namespace luisa