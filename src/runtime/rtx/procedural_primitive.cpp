#include <runtime/rtx/procedural_primitive.h>

namespace luisa::compute {

ProceduralPrimitive::ProceduralPrimitive(DeviceInterface *device, BufferView<AABB> aabb, const AccelOption &option) noexcept
    : _aabb_buffer(aabb.handle()),
      _aabb_size(aabb.size_bytes()),
      _aabb_offset(aabb.offset_bytes()),
      Resource(device, Resource::Tag::PROCEDURAL_PRIMITIVE,
               device->create_procedural_primitive(option)) {
}

luisa::unique_ptr<Command> ProceduralPrimitive::build(AccelBuildRequest request) noexcept {
    return ProceduralPrimitiveBuildCommand::create(handle(), request, _aabb_buffer, _aabb_offset, _aabb_size);
}

ProceduralPrimitive::~ProceduralPrimitive() noexcept {
    if (*this) { device()->destroy_procedural_primitive(handle()); }
}

}// namespace luisa::compute
