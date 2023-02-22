#pragma once

#include <runtime/device.h>
#include <runtime/buffer.h>
#include <runtime/rtx/aabb.h>

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
    ProceduralPrimitive(ProceduralPrimitive &&) noexcept = default;
    ProceduralPrimitive(ProceduralPrimitive const &) noexcept = delete;
    ProceduralPrimitive &operator=(ProceduralPrimitive &&) noexcept = default;
    ProceduralPrimitive &operator=(ProceduralPrimitive const &) noexcept = delete;

    using Resource::operator bool;
    [[nodiscard]] luisa::unique_ptr<Command> build(AccelBuildRequest request = AccelBuildRequest::PREFER_UPDATE) noexcept;
    [[nodiscard]] auto aabb_offset() const noexcept { return _aabb_offset; }
    [[nodiscard]] auto aabb_count() const noexcept { return _aabb_count; }
};

template<typename AABBBuffer>
ProceduralPrimitive Device::create_procedural_primitive(
    AABBBuffer &&aabb_buffer, const AccelOption &option) noexcept {
    return this->_create<ProceduralPrimitive>(std::forward<AABBBuffer>(aabb_buffer), option);
}

}// namespace luisa::compute
