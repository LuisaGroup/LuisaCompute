#pragma once
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/backends/ext/native_resource_ext.hpp>
namespace luisa::compute {
class DxCudaInterop : public DeviceExtension {
protected:
    ~DxCudaInterop() noexcept = default;

public:
    static constexpr luisa::string_view name = "DxCudaInterop";

    template<typename T>
    Buffer<T> create_shared_buffer(unsigned size, void** cuda_ptr) 
    {
        auto buffer = ResourceGenerator::create_native_buffer<T>(GetBufferInfo(sizeof(T)*size), GetDevice());
        *cuda_ptr = reinterpret_cast<void*>(cuda_buffer(buffer.handle()));
        return buffer;
    }
    template<typename T>
    Image<T> create_shared_image(PixelStorage storage, unsigned width, unsigned height, void** cuda_ptr)
    {
        auto buffer = ResourceGenerator::create_native_image<T>(GetImageInfo(sizeof(T) * size), GetDevice(), storage, {width,height}, 1);
        *cuda_ptr = reinterpret_cast<void*>(cuda_texture(buffer.handle()));
        return buffer;
    }

protected:
    virtual uint64_t cuda_buffer(uint64_t dx_buffer_handle) noexcept = 0;
    virtual uint64_t cuda_texture(uint64_t dx_texture_handle) noexcept = 0;
    virtual uint64_t cuda_event(uint64_t dx_event_handle) noexcept = 0;

    //virtual BufferCreationInfo GetImageInfo(unsigned sizeInBytes, unsigned int, unsigned height) = 0;
    virtual BufferCreationInfo GetBufferInfo(unsigned sizeInBytes) = 0;
    virtual DeviceInterface* GetDevice() = 0;
};
}// namespace luisa::compute
