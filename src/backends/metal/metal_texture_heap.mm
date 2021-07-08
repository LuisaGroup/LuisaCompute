//
// Created by Mike Smith on 2021/7/1.
//

#include <backends/metal/metal_device.h>
#include <backends/metal/metal_texture_heap.h>

namespace luisa::compute::metal {

MTLHeapDescriptor *MetalTextureHeap::_heap_descriptor(size_t size) noexcept {
    auto desc = [[MTLHeapDescriptor alloc] init];
    desc.size = size;
    desc.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    desc.type = MTLHeapTypeAutomatic;
    desc.storageMode = MTLStorageModePrivate;
    return desc;
}

MetalTextureHeap::MetalTextureHeap(MetalDevice *device, size_t size) noexcept
    : _device{device},
      _handle{[device->handle() newHeapWithDescriptor:_heap_descriptor(size)]} {

    static constexpr auto src = @"#include <metal_stdlib>\n"
                                 "struct Texture {\n"
                                 "  metal::texture2d<float> handle2d;\n"
                                 "  metal::texture3d<float> handle3d;\n"
                                 "  metal::sampler sampler;\n"
                                 "};\n"
                                 "[[kernel]] void k(device const Texture *heap) {}\n";
    auto library = [_device->handle() newLibraryWithSource:src options:nullptr error:nullptr];
    auto function = [library newFunctionWithName:@"k"];
    _encoder = [function newArgumentEncoderWithBufferIndex:0];
    if (auto enc_size = _encoder.encodedLength; enc_size != 24u) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid heap texture encoded size: {} (expected 16).",
            enc_size);
    }
    _buffer = [_device->handle() newBufferWithLength:_encoder.encodedLength * TextureHeap::max_slot_count
                                             options:MTLResourceOptionCPUCacheModeWriteCombined
                                                     | MTLResourceStorageModeShared];
}

id<MTLTexture> MetalTextureHeap::allocate_texture(MTLTextureDescriptor *desc, uint32_t index, TextureSampler sampler) noexcept {
    auto texture = [_handle newTextureWithDescriptor:desc];
    auto sampler_state = _device->texture_sampler(sampler);
    [_encoder setArgumentBuffer:_buffer offset:16u /* checked */ * index];
    [_encoder setTexture:texture atIndex:desc.textureType == MTLTextureType2D ? 0u : 1u];
    [_encoder setSamplerState:sampler_state atIndex:2u];
    return texture;
}

}
