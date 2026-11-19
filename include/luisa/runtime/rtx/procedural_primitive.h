#pragma once

#include <luisa/runtime/device.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/rtx/aabb.h>

namespace luisa::compute {

// ProceduralPrimitive is bottom-level acceleration structure(BLAS) for ray-tracing, it present AABB for custom intersection.
// Remember, the AABB intersection is conservative and may-be inaccurate, so never use it as "box intersection"
class LUISA_RUNTIME_API ProceduralPrimitive final : public Resource {

    friend class Device;

private:
    void *_aabb_buffer_native_handle{};
    uint64_t _aabb_buffer{};
    size_t _aabb_buffer_offset_bytes{};
    size_t _aabb_buffer_size_bytes{};
    size_t _aabb_buffer_total_size_bytes{};
    uint _motion_keyframe_count{};

    ProceduralPrimitive(DeviceInterface *device,
                        BufferView<AABB> aabb,
                        const AccelOption &option) noexcept;

public:
    [[nodiscard]] BufferView<AABB> aabb_buffer() const noexcept {
        return {_aabb_buffer_native_handle, _aabb_buffer, sizeof(AABB),
                _aabb_buffer_offset_bytes, _aabb_buffer_size_bytes / sizeof(AABB),
                _aabb_buffer_total_size_bytes / sizeof(AABB)};
    }
    ProceduralPrimitive() noexcept = default;
    ~ProceduralPrimitive() noexcept override { _release(); }
    ProceduralPrimitive(ProceduralPrimitive &&) noexcept = default;
    ProceduralPrimitive(ProceduralPrimitive const &) noexcept = default;
    ProceduralPrimitive &operator=(ProceduralPrimitive &&rhs) noexcept = default;
    ProceduralPrimitive &operator=(ProceduralPrimitive const &rhs) noexcept = default;

    using Resource::operator bool;
    using Resource::release;
    // build procedural primitives' based bottom-level acceleration structure
    [[nodiscard]] luisa::unique_ptr<Command> build(
        AccelBuildRequest request = AccelBuildRequest::PREFER_UPDATE) noexcept;
    [[nodiscard]] auto aabb_buffer_offset() const noexcept {
        _check_is_valid();
        return _aabb_buffer_offset_bytes;
    }
    [[nodiscard]] auto aabb_buffer_size() const noexcept {
        _check_is_valid();
        return _aabb_buffer_size_bytes;
    }
    [[nodiscard]] auto motion_keyframe_count() const noexcept {
        _check_is_valid();
        return std::max<uint>(1u, _motion_keyframe_count);
    }
    [[nodiscard]] auto aabb_count_per_motion_keyframe() const noexcept {
        auto n = this->motion_keyframe_count();
        return _aabb_buffer_size_bytes / sizeof(AABB) / n;
    }

protected:
    void destroy() noexcept override;
};

template<typename AABBBuffer>
ProceduralPrimitive Device::create_procedural_primitive(
    AABBBuffer &&aabb_buffer, const AccelOption &option) noexcept {
    return this->_create<ProceduralPrimitive>(std::forward<AABBBuffer>(aabb_buffer), option);
}

}// namespace luisa::compute
