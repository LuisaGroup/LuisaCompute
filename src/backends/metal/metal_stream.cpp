//
// Created by Mike Smith on 2023/4/15.
//

#include <backends/metal/metal_event.h>
#include <backends/metal/metal_texture.h>
#include <backends/metal/metal_swapchain.h>
#include <backends/metal/metal_stream.h>

namespace luisa::compute::metal {

MetalStream::MetalStream(MTL::Device *device, StreamTag tag [[maybe_unused]]) noexcept
    : _queue{device->newCommandQueue()} {
    // TODO
}

MetalStream::~MetalStream() noexcept {
    _queue->release();
}

void MetalStream::signal(MetalEvent *event) noexcept {
    event->signal(_queue);
}

void MetalStream::wait(MetalEvent *event) noexcept {
    event->wait(_queue);
}

void MetalStream::synchronize() noexcept {
    auto autorelease_pool = NS::AutoreleasePool::alloc()->init();
    auto command_buffer = _queue->commandBufferWithUnretainedReferences();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    autorelease_pool->release();
}

void MetalStream::set_name(luisa::string_view name) noexcept {
    if (name.empty()) {
        _queue->setLabel(nullptr);
    } else {
        luisa::string mtl_name{name};
        auto autorelease_pool = NS::AutoreleasePool::alloc()->init();
        _queue->setLabel(NS::String::string(mtl_name.c_str(), NS::UTF8StringEncoding));
        autorelease_pool->release();
    }
}

void MetalStream::dispatch(CommandList &&list) noexcept {
    // TODO
}

void MetalStream::present(MetalSwapchain *swapchain, MetalTexture *image) noexcept {
    swapchain->present(_queue, image->level(0u));
}

}// namespace luisa::compute::metal
