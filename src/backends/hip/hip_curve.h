//
// HIPRT curve geometry implemented as custom AABB primitives.
//

#pragma once

#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>

#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/rtx/curve.h>

#include "hip_geometry.h"

namespace luisa::compute::hip {

class HIPCommandEncoder;

/// Device-visible curve data consumed by hiprt_device_wrapper.hip.
/// Keep this layout in sync with LuisaCurveGeometryData in that file.
struct alignas(16) HIPCurveDeviceData {
    uint64_t control_points;
    uint64_t segments;
    uint32_t control_point_stride;
    uint32_t basis;
};

static_assert(sizeof(HIPCurveDeviceData) == 32u);
static_assert(alignof(HIPCurveDeviceData) == 16u);

class HIPCurve final : public HIPGeometry {

private:
    AccelOption _option;
    hiprtContext _hiprt_ctx{nullptr};
    hiprtGeometry _geometry{nullptr};
    hipDeviceptr_t _aabb_buffer{};
    size_t _aabb_buffer_size{};
    hipDeviceptr_t _device_data{};
    hipDeviceptr_t _source_control_points{};
    hipDeviceptr_t _source_segments{};
    hipDeviceptr_t _control_points{};
    size_t _control_points_size{};
    hipDeviceptr_t _segments{};
    size_t _segments_size{};
    size_t _control_point_count{};
    size_t _segment_count{};
    size_t _control_point_stride{};
    CurveBasis _basis{};
    mutable spin_mutex _mutex;

public:
    explicit HIPCurve(hiprtContext ctx, const AccelOption &option) noexcept;
    ~HIPCurve() noexcept override;
    void build(HIPCommandEncoder &encoder, CurveBuildCommand *command) noexcept;
    [[nodiscard]] hiprtGeometry handle() const noexcept override {
        std::scoped_lock lock{_mutex};
        return _geometry;
    }
    [[nodiscard]] Kind kind() const noexcept override { return Kind::CURVE; }
    [[nodiscard]] uint64_t codegen_handle() const noexcept override {
        std::scoped_lock lock{_mutex};
        return reinterpret_cast<uint64_t>(_device_data);
    }
    [[nodiscard]] auto option() const noexcept { return _option; }
};

}// namespace luisa::compute::hip
