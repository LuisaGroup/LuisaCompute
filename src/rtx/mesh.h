//
// Created by Mike Smith on 2021/7/22.
//

#pragma once

#include <runtime/device.h>
#include <runtime/buffer.h>
#include <dsl/syntax.h>

namespace luisa::compute {

struct Triangle {
    uint i[3];
};

class Mesh {

private:
    Device::Handle _device;
    uint64_t _handle{};

private:
    friend class Device;
    explicit Mesh(Device::Handle device) noexcept
        : _device{std::move(device)},
          _handle{_device->create_mesh()} {}
    void _destroy() noexcept;

public:
    Mesh() noexcept = default;
    ~Mesh() noexcept { _destroy(); }
    Mesh(Mesh &&) noexcept = default;
    Mesh &operator=(Mesh &&rhs) noexcept;

    template<typename Vertex>
    [[nodiscard]] Command *build(AccelBuildHint mode, BufferView<Vertex> vertices, BufferView<Triangle> triangles) const noexcept {
        return MeshBuildCommand::create(
            _handle, mode,
            vertices.handle(), vertices.offset_bytes(), sizeof(Vertex), vertices.size(),
            triangles.handle(), triangles.offset_bytes(), triangles.size());
    }
    [[nodiscard]] Command *update() const noexcept;
    [[nodiscard]] uint64_t handle() const noexcept { return _handle; }
    [[nodiscard]] explicit operator bool() const noexcept { return _device != nullptr; }
};

}

LUISA_STRUCT(luisa::compute::Triangle, i)
