//
// Created by Mike Smith on 2023/4/15.
//

#pragma once

#include <core/stl/string.h>
#include <runtime/rhi/stream_tag.h>
#include <runtime/command_list.h>
#include <backends/metal/metal_api.h>

namespace luisa::compute::metal {

class MetalEvent;
class MetalTexture;
class MetalSwapchain;

class MetalStream {

private:
    MTL::CommandQueue *_queue;

public:
    MetalStream(MTL::Device *device, StreamTag tag) noexcept;
    ~MetalStream() noexcept;
    void signal(MetalEvent *event) noexcept;
    void wait(MetalEvent *event) noexcept;
    void synchronize() noexcept;
    void dispatch(CommandList &&list) noexcept;
    void present(MetalSwapchain *swapchain, MetalTexture *image) noexcept;
    void set_name(luisa::string_view name) noexcept;
};

}// namespace luisa::compute::metal
