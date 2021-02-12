//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <ast/type.h>
#include <core/union.h>

namespace luisa::compute {

class Variable {

public:
    enum struct Tag : uint32_t {

        // data
        LOCAL,
        SHARED,
        UNIFORM,
        CONSTANT,

        // resources
        BUFFER,
        TEXTURE,
        TEXTURE_HEAP,

        // builtins
        THREAD_ID,
        BLOCK_ID,
        DISPATCH_ID
    };

private:
    const Type *_type;
    Tag _tag;
    uint32_t _uid;

public:
    constexpr Variable(const Type *type, Tag tag, uint32_t uid) noexcept
        : _type{type}, _tag{tag}, _uid{uid} {}
    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] auto uid() const noexcept { return _uid; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
};

}// namespace luisa::compute
