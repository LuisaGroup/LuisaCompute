#if LUISA_BACKEND_ENABLE_OIDN

#include <luisa/runtime/stream.h>

#include "metal_stream.h"
#include "metal_denoiser.h"

namespace luisa::compute::metal {

class MetalOidnDenoiser final : public OidnDenoiser {
public:
    using OidnDenoiser::OidnDenoiser;
    void execute(bool async) noexcept override {
        auto lock = luisa::make_unique<std::shared_lock<std::shared_mutex>>(_mutex);
        exec_filters();
        if (!async) {
            _oidn_device.sync();
        } else {
            auto cmd_list = CommandList{};
            cmd_list.add_callback([lock_ = std::move(lock), this]() mutable {
                LUISA_ASSERT(lock_, "Callback called twice.");
                lock_.reset();
            });
            _device->dispatch(_stream, std::move(cmd_list));
        }
    }
};

luisa::shared_ptr<DenoiserExt::Denoiser> MetalDenoiserExt::create(uint64_t stream) noexcept {
    auto metal_stream = reinterpret_cast<MetalStream *>(stream);
    auto oidn_device = oidn::newMetalDevice(metal_stream->queue());
    return luisa::make_shared<MetalOidnDenoiser>(_device, std::move(oidn_device), stream);
}

luisa::shared_ptr<DenoiserExt::Denoiser> MetalDenoiserExt::create(Stream &stream) noexcept {
    return this->create(stream.handle());
}

}// namespace luisa::compute::metal

#endif
