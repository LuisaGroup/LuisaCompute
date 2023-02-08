#include <rtx/procedural_primitive.h>

namespace luisa::compute {

ProceduralPrimitive::ProceduralPrimitive(DeviceInterface *device,
                                         const AccelCreateOption &option,
                                         BufferView<AABB> buffer) noexcept
    : Resource(device,
               Resource::Tag::PROCEDURAL_PRIMITIVE,
               device->create_procedural_primitive(option,
                                                   buffer.handle(),
                                                   buffer.offset_bytes(),
                                                   buffer.size())) {
}

luisa::unique_ptr<Command> ProceduralPrimitive::build(AccelBuildRequest request) noexcept {
    return ProceduralPrimitiveBuildCommand::create(handle(), request);
}

ProceduralPrimitive Device::create_procedural_primitive(
    const AccelCreateOption &option, BufferView<AABB> aabb_buffer) noexcept {
    return this->_create<ProceduralPrimitive>(option, aabb_buffer);
}

}// namespace luisa::compute
