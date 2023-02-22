#pragma once

#include <core/stl/optional.h>
#include <runtime/depth_format.h>
#include <runtime/raster/vertex_attribute.h>
#include <runtime/raster/viewport.h>

namespace luisa::compute {

enum class Comparison : uint8_t {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always
};

class MeshFormat {

private:
    luisa::vector<luisa::vector<VertexAttribute>> _streams;

public:
    MeshFormat() noexcept {}
    MeshFormat(MeshFormat &&) noexcept = default;
    MeshFormat(MeshFormat const &) noexcept = default;
    MeshFormat &operator=(MeshFormat &&) noexcept = default;
    MeshFormat &operator=(MeshFormat const &) noexcept = default;
    size_t vertex_attribute_count() const noexcept {
        size_t count = 0;
        for (auto &&i : _streams) {
            count += i.size();
        }
        return count;
    }
    size_t vertex_stream_count() const noexcept { return _streams.size(); }
    void emplace_vertex_stream(luisa::span<VertexAttribute const> attributes) noexcept {
        _streams.emplace_back(attributes.begin(), attributes.end());
    }
    luisa::span<VertexAttribute const> attributes(size_t stream_index) const noexcept { return _streams[stream_index]; }
    decltype(auto) begin() const noexcept { return _streams.begin(); }
    decltype(auto) end() const noexcept { return _streams.end(); }
};

enum class FillMode : uint8_t {
    WireFrame,
    Solid
};

enum class CullMode : uint8_t {
    None,
    Front,
    Back
};

enum class TopologyType : uint8_t {
    Point,
    Line,
    Triangle
};

enum class BlendWeight : uint8_t {
    Zero,
    One,
    PrimColor,
    ImgColor,
    PrimAlpha,
    ImgAlpha,
    OneMinusPrimColor,
    OneMinusImgColor,
    OneMinusPrimAlpha,
    OneMinusImgAlpha
};

enum class BlendOp : uint8_t {
    Add,
    Subtract,
    Min,
    Max
};

enum class StencilOp : uint8_t {
    Keep,
    Zero,
    Replace
};

struct BlendState {
    bool enableBlend{};
    BlendOp op{BlendOp::Add};
    BlendWeight prim_op{BlendWeight::Zero};
    BlendWeight img_op{BlendWeight::One};
};

struct DepthState {
    bool enableDepth{};
    Comparison comparison{};
    bool write{};
};

struct StencilFaceOp {
    StencilOp stencil_fail_op{};
    StencilOp depth_fail_op{};
    StencilOp pass_op{};
    Comparison comparison{};
};

struct StencilState {
    bool enableStencil{};
    StencilFaceOp front_face_op{};
    StencilFaceOp back_face_op{};
    uint8_t read_mask{};
    uint8_t write_mask{};
};

struct RasterState {
    FillMode fill_mode{FillMode::Solid};
    CullMode cull_mode{CullMode::Back};
    BlendState blend_state{};
    DepthState depth_state{};
    StencilState stencil_state{};
    TopologyType topology{TopologyType::Triangle};
    bool front_counter_clockwise{false};
    bool depth_clip{true};
    bool conservative{false};
};

// Make raster state hash-able
static_assert(alignof(RasterState) == 1);

}// namespace luisa::compute
