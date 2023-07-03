#pragma once

#include <luisa/runtime/device.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/rtx/aabb.h>

namespace luisa::compute {

// ProceduralPrimitive is buttom-level acceleration structure(BLAS) for ray-tracing, it present AABB for custom intersection.
// Remember, the AABB intersection is conservative and may-be inaccurate, so never use it as "box intersection"
class LC_RUNTIME_API ProceduralPrimitive final : public Resource {

    friend class Device;

private:
    uint64_t _aabb_buffer{};
    size_t _aabb_buffer_offset{};
    size_t _aabb_buffer_size{};
    ProceduralPrimitive(DeviceInterface *device,
                        BufferView<AABB> aabb,
                        const AccelOption &option) noexcept;

public:
    ProceduralPrimitive() noexcept = default;
    ~ProceduralPrimitive() noexcept override;
    ProceduralPrimitive(ProceduralPrimitive &&) noexcept = default;
    ProceduralPrimitive(ProceduralPrimitive const &) noexcept = delete;
    ProceduralPrimitive &operator=(ProceduralPrimitive &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    ProceduralPrimitive &operator=(ProceduralPrimitive const &) noexcept = delete;

    using Resource::operator bool;
    // build procedural primitives' based bottom-level acceleration structure
    [[nodiscard]] luisa::unique_ptr<Command> build(
        AccelBuildRequest request = AccelBuildRequest::PREFER_UPDATE) noexcept;
    [[nodiscard]] auto aabb_buffer_offset() const noexcept {
        _check_is_valid();
        return _aabb_buffer_offset;
    }
    [[nodiscard]] auto aabb_buffer_size() const noexcept {
        _check_is_valid();
        return _aabb_buffer_size;
    }
};

template<typename AABBBuffer>
ProceduralPrimitive Device::create_procedural_primitive(
    AABBBuffer &&aabb_buffer, const AccelOption &option) noexcept {
    return this->_create<ProceduralPrimitive>(std::forward<AABBBuffer>(aabb_buffer), option);
}

}// namespace luisa::compute
