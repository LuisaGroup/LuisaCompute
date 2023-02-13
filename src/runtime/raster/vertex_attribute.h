#pragma once

#include <ast/type.h>
#include <runtime/pixel.h>
#include <core/stl/vector.h>

namespace luisa::compute {

enum class VertexAttributeType : uint8_t {
    Position,
    Normal,
    Tangent,
    Color,
    UV0,
    UV1,
    UV2,
    UV3
};

enum class VertexElementFormat : uint8_t{
    XYZW8UNorm,
    
    XY16UNorm,
    XYZW16UNorm,

    XY16Float,
    XYZW16Float,

    X32Float,
    XY32Float,
    XYZ32Float,
    XYZW32Float,

};

constexpr size_t VertexElementFormatStride(VertexElementFormat format)noexcept{
    switch(format){
        case VertexElementFormat::X32Float: return 4;
        case VertexElementFormat::XY32Float: return 8;
        case VertexElementFormat::XYZ32Float: return 12;
        case VertexElementFormat::XYZW32Float: return 16;
        case VertexElementFormat::XYZW8UNorm: return 4;
        case VertexElementFormat::XY16Float: return 4;
        case VertexElementFormat::XYZW16Float: return 8;
        case VertexElementFormat::XY16UNorm: return 4;
        case VertexElementFormat::XYZW16UNorm: return 8;
        default: return 0;
    }
}

constexpr size_t kVertexAttributeCount = static_cast<size_t>(VertexAttributeType::UV3) + 1;

struct VertexAttribute {
    VertexAttributeType type;
    VertexElementFormat format;
};

}// namespace luisa::compute
