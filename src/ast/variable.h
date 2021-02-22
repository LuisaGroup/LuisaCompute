//
// Created by Mike Smith on 2020/12/2.
//

#pragma once
#ifndef LC_BACKEND
#include <ast/type.h>
#endif

namespace luisa::compute {
class IType;
class Variable {
    friend class Function;

public:
    enum struct Tag : uint32_t {

        // data
        LOCAL,
        SHARED,
        CONSTANT,

        UNIFORM,

        // resources
        BUFFER,
        TEXTURE,
        // TODO: Bindless Texture

        // builtins
        THREAD_ID,
        BLOCK_ID,
        DISPATCH_ID
    };

private:
    const IType *_type;
    Tag _tag;
    uint32_t _uid;
    constexpr Variable(const IType *type, Tag tag, uint32_t uid) noexcept
        : _type{type}, _tag{tag}, _uid{uid} {}

public:
#ifndef LC_BACKEND
    [[nodiscard]] auto type() const noexcept { return static_cast<Type const *>(_type); }
#endif
    [[nodiscard]] auto itype() const noexcept { return _type; }
    [[nodiscard]] auto uid() const noexcept { return _uid; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
};

}// namespace luisa::compute
