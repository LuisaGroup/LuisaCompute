#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/backends/ext/dx_cuda_interop.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context context{argv[0]};
    Device cuda_device = context.create_device("cuda");
    Device dx_device = context.create_device("dx");
    Stream cuda_stream = cuda_device.create_stream();
    Stream dx_stream = dx_device.create_stream();

    auto interop_ext = dx_device.extension<DxCudaInterop>();
    TimelineEvent dx_event = interop_ext->create_timeline_event();
    auto cuda_event = interop_ext->cuda_event(dx_event.handle());
    // auto cuda_event = cuda_device.impl()->create_event();
    
    auto interop_buffer = interop_ext->create_buffer<uint>(1);
    uint64_t cuda_ptr;
    uint64_t cuda_handle;
    interop_ext->cuda_buffer(interop_buffer.handle(), &cuda_ptr, &cuda_handle);
    auto cuda_buffer = cuda_device.import_external_buffer<uint>(reinterpret_cast<void *>(cuda_ptr), 1);
    uint input = 114514;
    uint output{};

    // cuda_stream <<  cuda_buffer.copy_from(&input) << DxCudaInterop::Signal{interop_ext, (uint64_t)cuda_event.native_handle, 1};
    // cuda_stream << DxCudaInterop::Wait{interop_ext, (uint64_t)cuda_event.native_handle, 1} << cuda_buffer.copy_to(&output) << synchronize();
    dx_stream << interop_buffer.copy_from(&input) << dx_event.signal(1);
    cuda_stream << DxCudaInterop::Wait{interop_ext, (uint64_t)cuda_event, 1} << cuda_buffer.copy_to(&output) << synchronize();
    LUISA_INFO("Result: {}", output);
    interop_ext->unmap(reinterpret_cast<void *>(cuda_ptr), reinterpret_cast<void *>(cuda_handle));
    // cuda_device.impl()->destroy_event(cuda_event.handle);
    interop_ext->destroy_cuda_event(cuda_event);
}