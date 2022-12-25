//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <ast/type.h>
#include <ast/usage.h>
#include <core/stl/hash.h>

namespace luisa::compute {

class AstSerializer;

namespace detail {
class FunctionBuilder;
}

/// Variable class
class Variable {
    friend class AstSerializer;

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
        // raster builtins
        OBJECT_ID
    };

private:
    const Type *_type;
    uint32_t _uid;
    Tag _tag;
    bool _is_arg;

private:
    friend class detail::FunctionBuilder;
    constexpr Variable(const Type *type, Tag tag, uint32_t uid, bool is_arg) noexcept
        : _type{type}, _uid{uid}, _tag{tag}, _is_arg{is_arg} {}

public:
    Variable() noexcept = default;
    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] auto uid() const noexcept { return _uid; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto is_arg() const noexcept { return _is_arg; }
    [[nodiscard]] auto hash() const noexcept {
        auto u0 = static_cast<uint64_t>(_uid);
        auto u1 = static_cast<uint64_t>(_tag);
        using namespace std::string_view_literals;
        return hash64(u0 | (u1 << 32u), hash64(_type->hash(), hash64("__hash_variable"sv)));
    }
    [[nodiscard]] auto operator==(Variable rhs) const noexcept { return _uid == rhs._uid; }
};

}// namespace luisa::compute
