#include "cuda_error.h"
#include "cuda_buffer.h"
#include "cuda_command_encoder.h"
#include "cuda_stream.h"
#include "cuda_device.h"
#include "cuda_curve.h"

namespace luisa::compute::cuda {

CUDACurve::CUDACurve(const AccelOption &option) noexcept
    : CUDAPrimitive{Tag::CURVE, option} {}

optix::BuildInput CUDACurve::_make_build_input() const noexcept {
    optix::BuildInput build_input{};
    build_input.type = optix::BUILD_INPUT_TYPE_CURVES;
    build_input.curveArray.curveType = _basis;
    build_input.curveArray.numPrimitives = _seg_count;
    build_input.curveArray.vertexBuffers = _motion_buffer_pointers(_cp_buffer, _cp_count * _cp_stride);
    build_input.curveArray.numVertices = _cp_count / motion_keyframe_count();
    build_input.curveArray.vertexStrideInBytes = _cp_stride;
    static thread_local CUdeviceptr width_buffers[max_motion_keyframe_count] = {};
    for (auto i = 0u; i < motion_keyframe_count(); i++) {
        width_buffers[i] = build_input.curveArray.vertexBuffers[i] + sizeof(float) * 3u;
    }
    build_input.curveArray.widthBuffers = width_buffers;
    build_input.curveArray.widthStrideInBytes = _cp_stride;
    build_input.curveArray.normalBuffers = nullptr;
    build_input.curveArray.normalStrideInBytes = 0u;
    build_input.curveArray.indexBuffer = _seg_buffer;
    build_input.curveArray.indexStrideInBytes = sizeof(uint32_t);
    build_input.curveArray.flag = optix::GEOMETRY_FLAG_DISABLE_ANYHIT;
    build_input.curveArray.primitiveIndexOffset = 0u;
    build_input.curveArray.endcapFlags = optix::CURVE_ENDCAP_DEFAULT;
    return build_input;
}

void CUDACurve::build(CUDACommandEncoder &encoder, CurveBuildCommand *command) noexcept {

    auto cp_count = command->cp_count();
    auto cp_stride = command->cp_stride();
    LUISA_ASSERT(cp_stride >= sizeof(float4),
                 "Invalid control point buffer stride {} (must be at least {}).",
                 cp_stride, sizeof(float4));
    auto seg_count = command->seg_count();
    auto basis = [b = command->basis()] {
        switch (b) {
            case CurveBasis::PIECEWISE_LINEAR: return optix::PRIMITIVE_TYPE_ROUND_LINEAR;
            case CurveBasis::CUBIC_BSPLINE: return optix::PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
            case CurveBasis::CATMULL_ROM: return optix::PRIMITIVE_TYPE_ROUND_CATMULLROM;
            case CurveBasis::BEZIER: return optix::PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER;
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid curve basis 0x{:x}.", luisa::to_underlying(b));
    }();
    auto cp_buffer = reinterpret_cast<const CUDABuffer *>(command->cp_buffer());
    auto seg_buffer = reinterpret_cast<const CUDABuffer *>(command->seg_buffer());
    LUISA_ASSERT(command->cp_buffer_offset() + cp_count * cp_stride <= cp_buffer->size_bytes(),
                 "Control point buffer out of range.");
    LUISA_ASSERT(command->seg_buffer_offset() + seg_count * sizeof(uint32_t) <= seg_buffer->size_bytes(),
                 "Segment buffer out of range.");

    std::scoped_lock lock{_mutex};

    auto requires_build =
        // not built yet
        _handle == 0u ||
        // not allowed to update
        !option().allow_update ||
        // user wants to force build
        command->request() == AccelBuildRequest::FORCE_BUILD ||
        // curve basis changed
        basis != _basis ||
        // buffers changed
        cp_buffer->device_address() + command->cp_buffer_offset() != _cp_buffer ||
        command->cp_count() != _cp_count ||
        command->cp_stride() != _cp_stride ||
        seg_buffer->device_address() + command->seg_buffer_offset() != _seg_buffer ||
        command->seg_count() != _seg_count;

    // update parameters
    _basis = basis;
    _cp_count = cp_count;
    _seg_count = seg_count;
    _cp_buffer = cp_buffer->device_address() + command->cp_buffer_offset();
    _cp_stride = cp_stride;
    _seg_buffer = seg_buffer->device_address() + command->seg_buffer_offset();

    // build or update
    if (requires_build) {
        _build(encoder);
    } else {
        _update(encoder);
    }
}

}// namespace luisa::compute::cuda
