//
// Created by Mike Smith on 2021/4/6.
//

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <stb/stb_image_resize.h>

#include <core/logging.h>
#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/event.h>
#include <dsl/syntax.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);
    auto heap = device.create_bindless_array(4);
    auto stream = device.create_stream();
    auto buffer0 = device.create_buffer<int>(1);
    auto buffer1 = device.create_buffer<int>(1);
    auto out_buffer = device.create_buffer<int>(2);
    heap.emplace_on_update(0, buffer0);
    heap.emplace_on_update(1, buffer1);
    Kernel1D kernel = [&] {
        out_buffer->write(dispatch_id().x, heap->buffer<int>(dispatch_id().x).read(0));
    };
    auto shader = device.compile(kernel);
    int v0 = 555;
    int v1 = 666;
    int result[2];
    stream << heap.update() << synchronize();
    stream << buffer0.copy_from(&v0) << buffer1.copy_from(&v1) << shader().dispatch(2) << out_buffer.copy_to(result) << synchronize();
    LUISA_INFO("Value: {}, {}", result[0], result[1]);
}
