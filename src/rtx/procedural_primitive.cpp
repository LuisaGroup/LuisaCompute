#include <rtx/procedural_primitive.h>

namespace luisa::compute {

ProceduralPrimitive::ProceduralPrimitive(DeviceInterface *device, const Buffer<AABB> &buffer, size_t aabb_offset, size_t aabb_count, const MeshBuildOption &option) noexcept
    : _aabb_buffer(buffer.handle()),
      _aabb_count(aabb_count),
      _aabb_offset(aabb_offset),
      Resource(device, Resource::Tag::PROCEDURAL_PRIMITIVE,
               device->create_mesh(option.hint, DeviceInterface::MeshType::ProceduralPrimitive, option.allow_compact, option.allow_update)) {
}

luisa::unique_ptr<Command> ProceduralPrimitive::build(AccelBuildRequest request) noexcept {
    return PrimBuildCommand::create(handle(), request, _aabb_buffer, _aabb_offset, _aabb_count);
}

ProceduralPrimitive Device::create_primitive(
    const Buffer<AABB> &aabb_buffer,
    size_t aabb_offset,
    size_t aabb_count,
    MeshBuildOption option) {
    return this->_create<ProceduralPrimitive>(aabb_buffer, aabb_offset, aabb_count, option);
}

}// namespace luisa::compute
