//
// Created by Mike Smith on 2023/4/15.
//

#include <core/logging.h>
#include <backends/metal/metal_event.h>
#include <backends/metal/metal_texture.h>
#include <backends/metal/metal_swapchain.h>
#include <backends/metal/metal_command_encoder.h>
#include <backends/metal/metal_stream.h>

namespace luisa::compute::metal {

MetalStream::MetalStream(MTL::Device *device,
                         StreamTag tag [[maybe_unused]],
                         size_t max_commands) noexcept
    : _queue{max_commands == 0u ?
                 device->newCommandQueue() :
                 device->newCommandQueue(max_commands)},
      _upload_pool{device, 64_M, true},
      _download_pool{device, 32_M, false} {
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
        auto mtl_name = NS::String::alloc()->init(
            const_cast<char *>(name.data()), name.size(),
            NS::UTF8StringEncoding, false);
        _queue->setLabel(mtl_name);
        mtl_name->release();
    }
}

void MetalStream::dispatch(CommandList &&list) noexcept {
    if (list.empty()) {
        LUISA_WARNING_WITH_LOCATION(
            "MetalStream::dispatch: Command list is empty.");
    } else {
        MetalCommandEncoder encoder{this};
        auto commands = list.steal_commands();
        auto callbacks = list.steal_callbacks();
        auto autorelease_pool = NS::AutoreleasePool::alloc()->init();
        for (auto &command : commands) { command->accept(encoder); }
        encoder.submit(std::move(callbacks));
        autorelease_pool->release();
    }
}

void MetalStream::present(MetalSwapchain *swapchain, MetalTexture *image) noexcept {
    swapchain->present(_queue, image->level(0u));
}

void MetalStream::submit(MTL::CommandBuffer *command_buffer,
                         MetalStream::CallbackContainer &&callbacks) noexcept {
    if (!callbacks.empty()) {
        {
            std::scoped_lock lock{_callback_mutex};
            _callback_lists.emplace(std::move(callbacks));
        }
        command_buffer->addCompletedHandler(^(MTL::CommandBuffer *) noexcept {
          auto callbakcs = [self = this] {
              std::scoped_lock lock{self->_callback_mutex};
              if (self->_callback_lists.empty()) {
                  LUISA_WARNING_WITH_LOCATION(
                      "MetalStream::submit: Callback list is empty.");
                  return CallbackContainer{};
              }
              auto callbacks = std::move(self->_callback_lists.front());
              self->_callback_lists.pop();
              return callbacks;
          }();
          for (auto callback : callbakcs) { callback->recycle(); }
        });
    }
    command_buffer->commit();
}

}// namespace luisa::compute::metal
