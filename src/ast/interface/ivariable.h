#pragma once

#include "interface_common.h"

namespace luisa::compute {

class IVariable {
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
};

}// namespace luisa::compute
