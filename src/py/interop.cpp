#include "interop.h"
#include <luisa/backends/ext/cuda/cuda_external_ext.h>

namespace luisa::compute {

struct CUDADeviceConfigExtImpl : public CUDADeviceConfigExt {
    ExternalVkDevice external_device;
    explicit CUDADeviceConfigExtImpl(ExternalVkDevice external_device) : external_device(external_device) {}
    [[nodiscard]] ExternalVkDevice get_external_vk_device() const noexcept override {
        return external_device;
    }
};

PyInterop::PyInterop(DeviceInterface *device) : _render_device_idx(~0u) {
    if (auto dx = device->extension("DxCudaInterop"))
        _ext = static_cast<DxCudaInterop *>(dx);
    else if (auto vk = device->extension("VkCudaInterop"))
        _ext = static_cast<VkCudaInterop *>(vk);

    if (!_ext.valid()) {
        return;
    }
    _render_device_idx = _ext.visit_or(~0u, [&](auto &&a) {
        return (uint)a->cuda_device_index();
    });
    if (_render_device_idx == ~0u) return;
    DeviceConfig config{.device_index = _render_device_idx};
    auto vk = _ext.try_get<VkCudaInterop *>();
    if (vk) {
        config.extension = luisa::make_unique<CUDADeviceConfigExtImpl>(
            (*vk)->get_external_vk_device());
    }
    config.headless = true;
    compute_device = device->context().create_device("cuda", &config);
    _cu_ext = compute_device.extension<CUDAExternalExt>();
}

void PyInterop::_init_event() {
    if (_event.valid()) return;
    if (_render_device_idx == ~0u) [[unlikely]] {
        LUISA_ERROR("Cuda device invalid.");
    }
    _ext.visit([&]<typename T>(T const &t) {
        if constexpr (std::is_same_v<T, DxCudaInterop *>) {
            _event = t->create_timeline_event();
        } else if constexpr (std::is_same_v<T, VkCudaInterop *>) {
            _event = compute_device.create_timeline_event();
        } else {
            static_assert(luisa::always_false_v<T>, "Unsupported interop.");
        }
    });
}

void PyInterop::compute_to_render_fence(
    void *signalled_cu_stream_ptr,
    Stream &wait_render_stream) {
    _init_event();
    auto idx = ++_event_fence;
    _event.visit([&]<typename T>(T &t) {
        if constexpr (std::is_same_v<T, DxCudaTimelineEvent>) {
            auto dx = _ext.try_get<DxCudaInterop *>();
            if (!dx) [[unlikely]] {
                LUISA_ERROR("Unsupported backend.");
            }
            t.cuda_signal_external(*dx, signalled_cu_stream_ptr, idx);
        } else {
            compute_device.extension<CUDAExternalExt>()->cuda_stream_signal(signalled_cu_stream_ptr, t.handle(), idx);
        }
    });
    _event.visit([&]<typename T>(T &t) {
        if constexpr (std::is_same_v<T, DxCudaTimelineEvent>) {
            wait_render_stream << t.dx_wait(idx);
        } else {
            auto vk = _ext.try_get<VkCudaInterop *>();
            if (!vk) [[unlikely]] {
                LUISA_ERROR("Unsupported backend.");
            }
            wait_render_stream << (*vk)->vk_wait(t, idx);
        }
    });
}

void PyInterop::render_to_compute_fence(
    Stream &signalled_render_stream,
    void *wait_cu_stream_ptr) {
    _init_event();
    auto idx = ++_event_fence;
    _event.visit([&]<typename T>(T &t) {
        if constexpr (std::is_same_v<T, DxCudaTimelineEvent>) {
            signalled_render_stream << t.dx_signal(idx);
        } else {
            auto vk = _ext.try_get<VkCudaInterop *>();
            if (!vk) [[unlikely]] {
                LUISA_ERROR("Unsupported backend.");
            }
            signalled_render_stream << (*vk)->vk_signal(t, idx);
        }
    });
    _event.visit([&]<typename T>(T &t) {
        if constexpr (std::is_same_v<T, DxCudaTimelineEvent>) {
            auto dx = _ext.try_get<DxCudaInterop *>();
            if (!dx) [[unlikely]] {
                LUISA_ERROR("Unsupported backend.");
            }
            t.cuda_wait_external(*dx, wait_cu_stream_ptr, idx);
        } else {
            compute_device.extension<CUDAExternalExt>()->cuda_stream_wait(wait_cu_stream_ptr, t.handle(), idx);
        }
    });
}

PyInterop::~PyInterop() = default;

BufferCreationInfo PyInterop::create_interop_buffer(const Type *element, size_t elem_count) {
    if (_render_device_idx == ~0u) [[unlikely]] {
        LUISA_ERROR("Cuda device invalid.");
    }
    BufferCreationInfo r{};
    r.invalidate();
    _ext.visit([&](auto &t) {
        r = t->create_interop_buffer(element, elem_count);
    });
    return r;
}

ResourceCreationInfo PyInterop::create_interop_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels, bool simultaneous_access, bool allow_raster_target) {
    if (_render_device_idx == ~0u) [[unlikely]] {
        LUISA_ERROR("Cuda device invalid.");
    }
    ResourceCreationInfo r{};
    r.invalidate();
    _ext.visit([&](auto &t) {
        r = t->create_interop_texture(
            format, dimension,
            width, height, depth,
            mipmap_levels, simultaneous_access, allow_raster_target);
    });
    return r;
}

void PyInterop::cuda_buffer(uint64_t buffer_handle, uint64_t *cuda_ptr, uint64_t *cuda_handle /*CUexternalMemory* */) {
    if (_render_device_idx == ~0u) [[unlikely]] {
        LUISA_ERROR("Cuda device invalid.");
    }
    _ext.visit([&](auto &t) {
        t->cuda_buffer(buffer_handle, cuda_ptr, cuda_handle);
    });
}

void PyInterop::unmap(void *cuda_ptr, void *cuda_handle) {
    if (_render_device_idx == ~0u) [[unlikely]] {
        LUISA_ERROR("Cuda device invalid.");
    }
    _ext.visit([&](auto &t) {
        t->unmap(cuda_ptr, cuda_handle);
    });
}

}// namespace luisa::compute