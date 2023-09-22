#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <stb/stb_image_resize.h>

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/dsl/syntax.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
    BindlessArray heap = device.create_bindless_array(64);
    Stream stream = device.create_stream();
    Buffer<int> buffer0 = device.create_buffer<int>(1);
    Buffer<int> buffer1 = device.create_buffer<int>(1);
    Buffer<int> out_buffer = device.create_buffer<int>(2);
    heap.emplace_on_update(5, buffer0);
    heap.emplace_on_update(6, buffer1);
    Kernel1D kernel = [&] {
        out_buffer->write(dispatch_id().x, heap->buffer<int>(dispatch_id().x + 5).read(0));
    };
    Shader1D<> shader = device.compile(kernel);
    int v0 = 555;
    int v1 = 666;
    int result[2];
    stream << heap.update() << synchronize();
    stream << buffer0.copy_from(&v0) << buffer1.copy_from(&v1) << shader().dispatch(2) << out_buffer.copy_to(result) << synchronize();
    LUISA_INFO("Value: {}, {}", result[0], result[1]);
}

