#pragma once

#include <runtime/device.h>
#include <runtime/buffer.h>

namespace luisa::compute {

class LC_RUNTIME_API ProceduralPrimitive final : public Resource {

    friend class Device;

private:
    uint64_t _aabb_buffer{};
    size_t _aabb_offset{};
    size_t _aabb_count{};
    ProceduralPrimitive(DeviceInterface *device, BufferView<AABB> aabb, const AccelOption &option) noexcept;

public:
    ProceduralPrimitive() noexcept = default;
    using Resource::operator bool;
    [[nodiscard]] luisa::unique_ptr<Command> build(AccelBuildRequest request = AccelBuildRequest::PREFER_UPDATE) noexcept;
    [[nodiscard]] auto aabb_offset() const noexcept { return _aabb_offset; }
    [[nodiscard]] auto aabb_count() const noexcept { return _aabb_count; }
};
}// namespace luisa::compute