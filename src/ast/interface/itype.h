#pragma once
#include "interface_common.h"
namespace luisa::compute {
class IType {
public:
    enum struct Tag : uint16_t {

        BOOL,

        FLOAT,
        INT8,
        UINT8,
        INT16,
        UINT16,
        INT32,
        UINT32,

        VECTOR,
        MATRIX,

        ARRAY,

        ATOMIC,
        STRUCTURE,

        BUFFER,
        // TODO: TEXTURE
    };
};
}// namespace luisa::compute