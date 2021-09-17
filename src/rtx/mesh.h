//
// Created by Mike Smith on 2021/7/22.
//

#pragma once

#include <runtime/buffer.h>
#include <dsl/syntax.h>

namespace luisa::compute {

struct Triangle {
    uint i0;
    uint i1;
    uint i2;
};

class Mesh : public Resource {

private:
    bool _built{false};

private:
    friend class Device;
    explicit Mesh(Device::Interface *device) noexcept
        : Resource{device, Tag::MESH, device->create_mesh()} {}

public:
    Mesh() noexcept = default;
    using Resource::operator bool;

    template<typename Vertex>
    [[nodiscard]] Command *build(AccelBuildHint mode, BufferView<Vertex> vertices, BufferView<Triangle> triangles) noexcept {
        _built = true;
        return MeshBuildCommand::create(
            handle(), mode,
            vertices.handle(), vertices.offset_bytes(), sizeof(Vertex), vertices.size(),
            triangles.handle(), triangles.offset_bytes(), triangles.size());
    }

    template<typename Vertex>
    [[nodiscard]] Command *build(AccelBuildHint mode, const Buffer<Vertex> &vertices, BufferView<Triangle> triangles) noexcept {
        return build(mode, vertices.view(), triangles);
    }

    [[nodiscard]] Command *update() noexcept;
};

}

LUISA_STRUCT(luisa::compute::Triangle, i0, i1, i2) {};
