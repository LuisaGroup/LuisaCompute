#pragma once

#include <luisa/ast/type.h>
#include <luisa/runtime/rhi/pixel.h>
#include <luisa/core/stl/vector.h>

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

constexpr size_t kVertexAttributeCount = static_cast<size_t>(VertexAttributeType::UV3) + 1;

struct VertexAttribute {
    VertexAttributeType type;
    PixelFormat format;
};

}// namespace luisa::compute

