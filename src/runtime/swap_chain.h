#pragma once
#include <utility>

#include <core/spin_mutex.h>
#include <runtime/resource.h>
#include <runtime/event.h>
#include <runtime/command_list.h>
#include <runtime/command_buffer.h>
#include <runtime/command_reorder_visitor.h>
#include <runtime/image.h>
namespace luisa::compute {
class LC_RUNTIME_API SwapChain final : public Resource {
    friend class Device;
    SwapChain(
        Device::Interface* device,
        uint64_t window_handle,
        uint64_t stream_handle,
        uint width,
        uint height,
        bool allow_hdr,
        uint back_buffer_size);

public:
    PixelStorage backend_storage() const;
};
}// namespace luisa::compute