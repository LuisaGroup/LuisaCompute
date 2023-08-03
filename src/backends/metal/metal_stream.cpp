#include <luisa/core/logging.h>
#include "metal_event.h"
#include "metal_texture.h"
#include "metal_swapchain.h"
#include "metal_command_encoder.h"
#include "metal_stream.h"

namespace luisa::compute::metal {

MetalStream::MetalStream(MTL::Device *device,
                         size_t max_commands) noexcept
    : _queue{max_commands == 0u ?
                 device->newCommandQueue() :
                 device->newCommandQueue(max_commands)} {}

MetalStream::~MetalStream() noexcept {
    _queue->release();
}

MetalStageBufferPool *MetalStream::upload_pool() noexcept {
    {
        std::scoped_lock lock{_upload_pool_creation_mutex};
        if (_upload_pool == nullptr) {
            _upload_pool = luisa::make_unique<MetalStageBufferPool>(
                _queue->device(), 64_M, true);
        }
    }
    return _upload_pool.get();
}

MetalStageBufferPool *MetalStream::download_pool() noexcept {
    {
        std::scoped_lock lock{_download_pool_creation_mutex};
        if (_download_pool == nullptr) {
            _download_pool = luisa::make_unique<MetalStageBufferPool>(
                _queue->device(), 32_M, false);
        }
    }
    return _download_pool.get();
}

void MetalStream::signal(MetalEvent *event, uint64_t value) noexcept {
    auto command_buffer = _queue->commandBufferWithUnretainedReferences();
    event->signal(command_buffer, value);
    command_buffer->commit();
}

void MetalStream::wait(MetalEvent *event, uint64_t value) noexcept {
    auto command_buffer = _queue->commandBufferWithUnretainedReferences();
    event->wait(command_buffer, value);
    command_buffer->commit();
}

void MetalStream::synchronize() noexcept {
    auto command_buffer = _queue->commandBufferWithUnretainedReferences();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
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

void MetalStream::_encode(MetalCommandEncoder &encoder,
                          Command *command) noexcept {
    command->accept(encoder);
}

void MetalStream::_do_dispatch(MetalCommandEncoder &encoder,
                               CommandList &&list) noexcept {
    if (list.empty()) {
        LUISA_WARNING_WITH_LOCATION(
            "MetalStream::dispatch: Command list is empty.");
    } else {
        auto commands = list.steal_commands();
        auto callbacks = list.steal_callbacks();
        {
            std::scoped_lock lock{_dispatch_mutex};
            for (auto &command : commands) { _encode(encoder, command.get()); }
            encoder.submit(std::move(callbacks));
        }
    }
}

void MetalStream::dispatch(CommandList &&list) noexcept {
    MetalCommandEncoder encoder{this};
    _do_dispatch(encoder, std::move(list));
}

void MetalStream::present(MetalSwapchain *swapchain, MetalTexture *image) noexcept {
    swapchain->present(_queue, image->handle());
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
#ifndef NDEBUG
    command_buffer->addCompletedHandler(^(MTL::CommandBuffer *cb) noexcept {
        if (auto error = cb->error()) {
            LUISA_WARNING("CommandBuffer execution error: {}.",
                          error->localizedDescription()->utf8String());
        }
        if (auto logs = cb->logs()) {
            luisa_compute_metal_stream_print_function_logs(logs);
        }
    });
#endif
    command_buffer->commit();
}

}// namespace luisa::compute::metal
