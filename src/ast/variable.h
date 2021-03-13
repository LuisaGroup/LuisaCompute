//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <ast/type.h>

namespace luisa::compute {

class Variable {

public:
    enum struct Tag : uint32_t {

        // data
        LOCAL,
        SHARED,
        UNIFORM,

        // resources
        BUFFER,
        TEXTURE,
        // TODO: Bindless Texture
        // TODO: Writable Texture, Writable Buffer
        
        // builtins
        THREAD_ID,
        BLOCK_ID,
        DISPATCH_ID
    };

private:
    const Type *_type;
    uint32_t _uid;
    Tag _tag;

private:
    friend class FunctionBuilder;
    constexpr Variable(const Type *type, Tag tag, uint32_t uid) noexcept
        : _type{type}, _uid{uid}, _tag{tag} {}

public:
    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] auto uid() const noexcept { return _uid; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
};

}// namespace luisa::compute
