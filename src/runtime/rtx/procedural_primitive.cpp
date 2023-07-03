#include <luisa/runtime/rtx/procedural_primitive.h>

namespace luisa::compute {

ProceduralPrimitive::ProceduralPrimitive(DeviceInterface *device,
                                         BufferView<AABB> aabb,
                                         const AccelOption &option) noexcept
    : Resource(device, Resource::Tag::PROCEDURAL_PRIMITIVE,
               device->create_procedural_primitive(option)),
      _aabb_buffer(aabb.handle()),
      _aabb_buffer_offset(aabb.offset_bytes()),
      _aabb_buffer_size(aabb.size_bytes()) {
}

luisa::unique_ptr<Command> ProceduralPrimitive::build(AccelBuildRequest request) noexcept {
    _check_is_valid();
    return luisa::make_unique<ProceduralPrimitiveBuildCommand>(
        handle(), request, _aabb_buffer, _aabb_buffer_offset, _aabb_buffer_size);
}

ProceduralPrimitive::~ProceduralPrimitive() noexcept {
    if (*this) { device()->destroy_procedural_primitive(handle()); }
}

}// namespace luisa::compute
