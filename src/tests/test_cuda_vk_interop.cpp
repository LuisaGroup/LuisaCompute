#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/backends/ext/vk_cuda_interop.h>

using namespace luisa;
using namespace luisa::compute;
struct CudaDeviceConfigExtImpl : public CudaDeviceConfigExt {
    ExternalVkDevice external_device;
    [[nodiscard]] ExternalVkDevice get_external_vk_device() const noexcept override {
        return external_device;
    }
};
int main(int argc, char *argv[]) {
    Context context{argv[0]};
    Device vk_device = context.create_device("vk");
    auto interop_ext = vk_device.extension<VkCudaInterop>();
    auto ext_device = luisa::make_unique<CudaDeviceConfigExtImpl>();
    ext_device->external_device = interop_ext->get_external_vk_device();
    DeviceConfig cuda_settings{
        .extension = std::move(ext_device)};
    Device cuda_device = context.create_device("cuda", &cuda_settings);
    Stream cuda_stream = cuda_device.create_stream();
    Stream vk_stream = vk_device.create_stream();

    auto interop_event = cuda_device.create_event();

    auto interop_buffer = interop_ext->create_buffer<uint>(1);
    uint64_t cuda_ptr;
    uint64_t cuda_handle;
    interop_ext->cuda_buffer(interop_buffer.handle(), &cuda_ptr, &cuda_handle);
    auto cuda_buffer = cuda_device.import_external_buffer<uint>(reinterpret_cast<void *>(cuda_ptr), 1);
    uint input = 114514;
    uint output{};
    vk_stream << interop_buffer.copy_from(&input) << interop_ext->vk_signal(interop_event);
    cuda_stream << interop_event.wait() << cuda_buffer.copy_to(&output) << synchronize();
    LUISA_INFO("Result: {}", output);
    interop_ext->unmap(reinterpret_cast<void *>(cuda_ptr), reinterpret_cast<void *>(cuda_handle));
    
    // Always remember to synchronize event it-self!
    interop_event.synchronize();
}