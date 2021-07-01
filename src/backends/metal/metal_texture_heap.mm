//
// Created by Mike Smith on 2021/7/1.
//

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

MetalTextureHeap::MetalTextureHeap(id<MTLDevice> device, size_t size) noexcept
    : _handle{[device newHeapWithDescriptor:_heap_descriptor(size)]} {

    static constexpr auto src = @"#include <metal_stdlib>\n"
                                 "struct Heap { metal::texture2d<float> t[65536]; };\n"
                                 "[[kernel]] void k(device const Heap &heap) {}\n";
    auto library = [device newLibraryWithSource:src options:nullptr error:nullptr];
    auto function = [library newFunctionWithName:@"k"];
    _encoder = [function newArgumentEncoderWithBufferIndex:0];
    _buffer = [device newBufferWithLength:_encoder.encodedLength
                                  options:MTLResourceOptionCPUCacheModeWriteCombined
                                          | MTLResourceStorageModeShared];
}

id<MTLTexture> MetalTextureHeap::allocate_texture(MTLTextureDescriptor *desc, uint32_t index) noexcept {
    auto texture = [_handle newTextureWithDescriptor:desc];
    [_encoder setArgumentBuffer:_buffer offset:0u];
    [_encoder setTexture:texture atIndex:index];
    return texture;
}

}
