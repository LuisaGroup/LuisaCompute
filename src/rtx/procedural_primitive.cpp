#include <runtime/device_interface.h>
#include <rtx/procedural_primitive.h>

namespace luisa::compute {

ProceduralPrimitive::ProceduralPrimitive(DeviceInterface *device,
                                         const AccelOption &option,
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
    BufferView<AABB> aabb_buffer, const AccelOption &option) noexcept {
    return this->_create<ProceduralPrimitive>(option, aabb_buffer);
}

}// namespace luisa::compute
