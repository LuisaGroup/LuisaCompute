#include "cuda_error.h"
#include "cuda_buffer.h"
#include "cuda_stream.h"
#include "cuda_device.h"
#include "cuda_command_encoder.h"
#include "cuda_procedural_primitive.h"

namespace luisa::compute::cuda {

CUDAProceduralPrimitive::CUDAProceduralPrimitive(const AccelOption &option) noexcept
    : CUDAPrimitive{Tag::PROCEDURAL, option} {}

optix::BuildInput CUDAProceduralPrimitive::_make_build_input() const noexcept {
    optix::BuildInput build_input{};
    static const auto geometry_flag = static_cast<uint32_t>(optix::GEOMETRY_FLAG_DISABLE_ANYHIT);
    build_input.type = optix::BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    build_input.customPrimitiveArray.aabbBuffers = _motion_buffer_pointers(_aabb_buffer, _aabb_buffer_size);
    build_input.customPrimitiveArray.numPrimitives = _aabb_buffer_size / sizeof(optix::Aabb) / motion_keyframe_count();
    build_input.customPrimitiveArray.strideInBytes = sizeof(optix::Aabb);
    build_input.customPrimitiveArray.flags = &geometry_flag;
    build_input.customPrimitiveArray.numSbtRecords = 1u;
    return build_input;
}

static_assert(sizeof(optix::Aabb) == 24ull, "Invalid Aabb size.");
void CUDAProceduralPrimitive::build(CUDACommandEncoder &encoder,
                                    ProceduralPrimitiveBuildCommand *command) noexcept {
    auto aabb_buffer = reinterpret_cast<const CUDABuffer *>(command->aabb_buffer());
    LUISA_ASSERT(command->aabb_buffer_offset() + command->aabb_buffer_size() <= aabb_buffer->size_bytes(),
                 "AABB buffer out of range.");

    std::scoped_lock lock{_mutex};

    auto requires_build =
        // not built yet
        _handle == 0u ||
        // not allowed to update
        !option().allow_update ||
        // user enforced rebuild
        command->request() == AccelBuildRequest::FORCE_BUILD ||
        // buffer changed
        _aabb_buffer != aabb_buffer->device_address() + command->aabb_buffer_offset() ||
        _aabb_buffer_size != command->aabb_buffer_size();

    // update the buffer
    _aabb_buffer = aabb_buffer->device_address() + command->aabb_buffer_offset();
    _aabb_buffer_size = command->aabb_buffer_size();

    // build or update
    if (requires_build) {
        _build(encoder);
    } else {
        _update(encoder);
    }
}

}// namespace luisa::compute::cuda
