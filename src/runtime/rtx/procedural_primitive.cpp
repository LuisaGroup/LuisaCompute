#include <luisa/runtime/rtx/procedural_primitive.h>

namespace luisa::compute {

ProceduralPrimitive::ProceduralPrimitive(DeviceInterface *device,
                                         BufferView<AABB> aabb,
                                         const AccelOption &option) noexcept
    : Resource(device, Resource::Tag::PROCEDURAL_PRIMITIVE,
               device->create_procedural_primitive(option)),
      _aabb_buffer_native_handle{aabb.native_handle()},
      _aabb_buffer{aabb.handle()},
      _aabb_buffer_offset_bytes{aabb.offset_bytes()},
      _aabb_buffer_size_bytes{aabb.size_bytes()},
      _aabb_buffer_total_size_bytes{aabb.total_size_bytes()},
      _motion_keyframe_count{option.motion.keyframe_count} {
    LUISA_ASSERT(_motion_keyframe_count <= 1u || aabb.size() % _motion_keyframe_count == 0u,
                 "AABB count must be multiple of motion keyframe count.");
}

luisa::unique_ptr<Command> ProceduralPrimitive::build(AccelBuildRequest request) noexcept {
    _check_is_valid();
    return luisa::make_unique<ProceduralPrimitiveBuildCommand>(
        handle(), request, _aabb_buffer, _aabb_buffer_offset_bytes, _aabb_buffer_size_bytes);
}

ProceduralPrimitive::~ProceduralPrimitive() noexcept {
    if (*this) { device()->destroy_procedural_primitive(handle()); }
}

}// namespace luisa::compute
