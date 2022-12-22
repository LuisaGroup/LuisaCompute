//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <ast/type.h>
#include <ast/usage.h>

namespace luisa::compute {

namespace detail {
class FunctionBuilder;
}

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
        DISPATCH_SIZE,
        KERNEL_ID,
        OBJECT_ID
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
    [[nodiscard]] uint64_t hash() const noexcept;
    [[nodiscard]] auto operator==(Variable rhs) const noexcept { return _uid == rhs._uid; }
    [[nodiscard]] auto is_local() const noexcept { return _tag == Tag::LOCAL; }
    [[nodiscard]] auto is_shared() const noexcept { return _tag == Tag::SHARED; }
    [[nodiscard]] auto is_reference() const noexcept { return _tag == Tag::REFERENCE; }
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
};

}// namespace luisa::compute
