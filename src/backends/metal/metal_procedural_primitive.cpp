#include <luisa/core/logging.h>
#include <luisa/runtime/rtx/aabb.h>
#include "metal_buffer.h"
#include "metal_procedural_primitive.h"

namespace luisa::compute::metal {

MetalProceduralPrimitive::MetalProceduralPrimitive(MTL::Device *device,
                                                   const AccelOption &option) noexcept
    : MetalPrimitive{device, option} {}

MetalProceduralPrimitive::~MetalProceduralPrimitive() noexcept {
    if (_descriptor) { _descriptor->release(); }
}

void MetalProceduralPrimitive::_do_add_resources(luisa::vector<MTL::Resource *> &resources) const noexcept {
    auto prim_desc = _descriptor->geometryDescriptors()
                         ->object<MTL::AccelerationStructureBoundingBoxGeometryDescriptor>(0u);
    resources.emplace_back(prim_desc->boundingBoxBuffer());
}

void MetalProceduralPrimitive::build(MetalCommandEncoder &encoder,
                                     ProceduralPrimitiveBuildCommand *command) noexcept {

    std::scoped_lock lock{mutex()};

    auto aabb_buffer = reinterpret_cast<MetalBuffer *>(command->aabb_buffer());
    auto aabb_buffer_handle = aabb_buffer->handle();
    auto aabb_buffer_offset = command->aabb_buffer_offset();
    auto aabb_buffer_size = command->aabb_buffer_size();
    constexpr auto aabb_stride = sizeof(AABB);
    LUISA_ASSERT(aabb_buffer_size % aabb_stride == 0u, "Invalid AABB buffer size.");

    auto aabb_buffer_changed = [&](auto desc) noexcept {
        return desc->boundingBoxBuffer() != aabb_buffer_handle ||
               desc->boundingBoxBufferOffset() != aabb_buffer_offset ||
               desc->boundingBoxCount() * aabb_stride != aabb_buffer_size;
    };

    using GeometryDescriptor = MTL::AccelerationStructureBoundingBoxGeometryDescriptor;
    auto requires_build = handle() == nullptr ||
                          !option().allow_update ||
                          command->request() == AccelBuildRequest::FORCE_BUILD ||
                          _descriptor == nullptr ||
                          aabb_buffer_changed(_descriptor->geometryDescriptors()
                                                  ->object<GeometryDescriptor>(0u));

    if (requires_build) {
        if (_descriptor) { _descriptor->release(); }
        auto geom_desc = GeometryDescriptor::alloc()->init();
        geom_desc->setBoundingBoxBuffer(aabb_buffer_handle);
        geom_desc->setBoundingBoxBufferOffset(aabb_buffer_offset);
        geom_desc->setBoundingBoxCount(aabb_buffer_size / aabb_stride);
        geom_desc->setBoundingBoxStride(aabb_stride);
        geom_desc->setOpaque(false);
        geom_desc->setAllowDuplicateIntersectionFunctionInvocation(true);
        geom_desc->setIntersectionFunctionTableOffset(0u);
        auto geom_desc_object = static_cast<NS::Object *>(geom_desc);
        auto geom_desc_array = NS::Array::array(&geom_desc_object, 1u);
        _descriptor = MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
        _descriptor->setGeometryDescriptors(geom_desc_array);
        _descriptor->setUsage(usage());
        _do_build(encoder, _descriptor);
    } else {
        _do_update(encoder, _descriptor);
    }
}

}// namespace luisa::compute::metal

