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
    struct BufferDeleter 
    {
        void* cuda_ptr = nullptr;
        DxCudaInterop* ext = nullptr;
        uint64_t memory_handle = 0;
        BufferDeleter(DxCudaInterop* ext, void* ptr, uint64_t handle) : ext(ext), cuda_ptr(ptr), memory_handle(handle) {}

        void operator()(Buffer<T>* buffer) const 
        {
            ext->unmap(cuda_ptr, reinterpret_cast<void*>(memory_handle));
            delete buffer;
        }
    };


    template<typename T>
    std::shared_ptr<Buffer<T>> create_shared_buffer(unsigned size, void** cuda_ptr) 
    {
        LUISA_INFO("size:{}, stride:{}",sizeof(T) * size, sizeof(T));
        auto buffer = ResourceGenerator::create_native_buffer<T>(GetBufferInfo(sizeof(T)*size), GetDevice());
        uint64_t memory_handle = 0;
        cuda_buffer(buffer.handle(), reinterpret_cast<uint64_t*>(cuda_ptr), reinterpret_cast<uint64_t*>(&memory_handle));

        Buffer<T>* heapBuffer = new Buffer<T>(std::move(buffer));  // Transfer the state to a heap-allocated Buffer
        BufferDeleter<T> deleter(this, *cuda_ptr, memory_handle);

        return std::shared_ptr<Buffer<T>>(heapBuffer, deleter);
    }
    virtual void unmap(void* cuda_ptr, void* cuda_handle) = 0;

protected:
    virtual void cuda_buffer(uint64_t dx_buffer_handle, uint64_t* cuda_ptr, uint64_t* cuda_handle) noexcept = 0;
    virtual uint64_t cuda_texture(uint64_t dx_texture_handle) noexcept = 0;
    virtual uint64_t cuda_event(uint64_t dx_event_handle) noexcept = 0;

    //virtual BufferCreationInfo GetImageInfo(unsigned sizeInBytes, unsigned int, unsigned height) = 0;
    virtual BufferCreationInfo GetBufferInfo(unsigned sizeInBytes) = 0;
    virtual DeviceInterface* GetDevice() = 0;
};
}// namespace luisa::compute
