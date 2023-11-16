#pragma once

#include <luisa/runtime/device.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/rhi/curve_basis.h>

namespace luisa::compute {

class LC_RUNTIME_API Curve final : public Resource {

public:
    using BuildRequest = AccelBuildRequest;

private:
    CurveBasis _basis{};
    size_t _cp_count{};
    size_t _seg_count{};
    // control point buffer
    uint64_t _cp_buffer{};
    size_t _cp_buffer_offset{};
    size_t _cp_stride{};
    // segment buffer
    uint64_t _seg_buffer{};
    size_t _seg_buffer_offset{};

private:
    friend class Device;

    template<typename CPBuffer, typename SegBuffer>
        requires is_buffer_or_view_v<CPBuffer> &&
                 is_buffer_or_view_v<SegBuffer> &&
                 (sizeof(buffer_element_t<CPBuffer>) >= sizeof(float4)) &&
                 std::same_as<buffer_element_t<SegBuffer>, uint>
    [[nodiscard]] static ResourceCreationInfo _create_resource(
        DeviceInterface *device, const AccelOption &option,
        const CPBuffer &control_point_buffer [[maybe_unused]],
        const SegBuffer &segment_buffer [[maybe_unused]]) noexcept {
        return device->create_curve(option);
    }

private:
    template<typename CPBuffer, typename SegBuffer>
    Curve(DeviceInterface *device,
          CurveBasis basis,
          const CPBuffer &control_point_buffer,
          const SegBuffer &segment_buffer,
          const AccelOption &option) noexcept
        : Resource{device, Resource::Tag::CURVE,
                   _create_resource(device, option,
                                    control_point_buffer,
                                    segment_buffer)},
          _basis{basis},
          _cp_count{control_point_buffer.size()},
          _seg_count{segment_buffer.size()},
          _cp_buffer{BufferView{control_point_buffer}.handle()},
          _cp_buffer_offset{BufferView{control_point_buffer}.offset_bytes()},
          _cp_stride{control_point_buffer.stride()},
          _seg_buffer{BufferView{segment_buffer}.handle()},
          _seg_buffer_offset{BufferView{segment_buffer}.offset_bytes()} {}

public:
    Curve() noexcept = default;
    ~Curve() noexcept override;
    Curve(Curve &&) noexcept = default;
    Curve(const Curve &) noexcept = delete;
    Curve &operator=(Curve &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    Curve &operator=(const Curve &) noexcept = delete;
    using Resource::operator bool;

public:
    [[nodiscard]] CurveBasis basis() const noexcept;
    [[nodiscard]] size_t control_point_count() const noexcept;
    [[nodiscard]] size_t segment_count() const noexcept;

public:
    [[nodiscard]] luisa::unique_ptr<Command> build(BuildRequest request = BuildRequest::PREFER_UPDATE) noexcept;
};

template<typename CPBuffer, typename SegmentBuffer>
Curve Device::create_curve(CurveBasis basis,
                           CPBuffer &&control_points,
                           SegmentBuffer &&segments,
                           const AccelOption &option) noexcept {
    return this->_create<Curve>(basis,
                                std::forward<CPBuffer>(control_points),
                                std::forward<SegmentBuffer>(segments),
                                option);
}

}// namespace luisa::compute
