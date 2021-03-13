//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <ast/type.h>

namespace luisa::compute {

class Variable {

public:
    enum struct Tag : uint16_t {

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
    
    enum Usage : uint16_t {
        USAGE_NONE = 0,
        USAGE_READ = 0x01,
        USAGE_WRITE = 0x02,
        USAGE_READ_WRITE = USAGE_READ | USAGE_WRITE
    };

private:
    const Type *_type;
    uint32_t _uid;
    Tag _tag;
    Usage _usage;

private:
    friend class FunctionBuilder;
    constexpr Variable(const Type *type, Tag tag, uint32_t uid, Usage usage = USAGE_NONE) noexcept
        : _type{type}, _uid{uid}, _tag{tag}, _usage{usage} {}

public:
    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] auto uid() const noexcept { return _uid; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto usage() const noexcept { return _usage; }
};

}// namespace luisa::compute
