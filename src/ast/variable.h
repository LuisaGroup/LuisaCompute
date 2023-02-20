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
class LC_AST_API Variable {

public:
    /// Variable tags
    enum struct Tag : uint8_t {

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
        DISPATCH_SIZE,
        KERNEL_ID,
        OBJECT_ID
    };

private:
    const Type *_type;
    uint32_t _uid;
    Tag _tag;
    bool _is_argument;

private:
    friend class detail::FunctionBuilder;
    friend class detail::SSABuilder;
    Variable(const Type *type, Tag tag, uint32_t uid, bool is_arg = false) noexcept
        : _type{type}, _uid{uid}, _tag{tag}, _is_argument{is_arg} {}

public:
    Variable() noexcept = default;
    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] auto uid() const noexcept { return _uid; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] uint64_t hash() const noexcept;
    [[nodiscard]] auto operator==(Variable rhs) const noexcept { return _uid == rhs._uid; }
    [[nodiscard]] auto is_local() const noexcept { return _tag == Tag::LOCAL; }
    [[nodiscard]] auto is_shared() const noexcept { return _tag == Tag::SHARED; }
    [[nodiscard]] auto is_reference() const noexcept { return _tag == Tag::REFERENCE; }
    [[nodiscard]] auto is_argument() const noexcept { return _is_argument; }
    [[nodiscard]] auto is_resource() const noexcept {
        return _tag == Tag::BUFFER ||
               _tag == Tag::TEXTURE ||
               _tag == Tag::BINDLESS_ARRAY ||
               _tag == Tag::ACCEL;
    }
    [[nodiscard]] auto is_builtin() const noexcept {
        return _tag == Tag::THREAD_ID ||
               _tag == Tag::BLOCK_ID ||
               _tag == Tag::DISPATCH_ID ||
               _tag == Tag::DISPATCH_SIZE ||
               _tag == Tag::KERNEL_ID ||
               _tag == Tag::OBJECT_ID;
    }
    [[nodiscard]] auto externally_initialized() const noexcept {
        return is_reference() ||
               is_resource() ||
               is_builtin() ||
               _is_argument;
    }
};

}// namespace compute

template<>
struct hash<compute::Variable> {
    using is_avalanching = void;
    [[nodiscard]] auto operator()(compute::Variable v) const noexcept { return v.hash(); }
};

}// namespace luisa