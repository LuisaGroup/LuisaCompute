// Validates non-uniform bindless buffer slot selection and reads.

#include "ut/ut.hpp"
#include "test_device.h"

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <stb/stb_image_resize2.h>

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/dsl/syntax.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_bindless(Device &device) {

    log_level_verbose();

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
    auto shader = device.compile(kernel);
    int v0 = 555;
    int v1 = 666;
    int result[2];
    stream << heap.update() << synchronize();
    stream << buffer0.copy_from(luisa::span{&v0, 1}) << buffer1.copy_from(luisa::span{&v1, 1}) << shader().dispatch(2) << out_buffer.copy_to(luisa::span{result, 2}) << synchronize();
    LUISA_INFO("Value: {}, {}", result[0], result[1]);
    expect(result[0] == v0) << "bindless slot 5 read mismatch";
    expect(result[1] == v1) << "bindless slot 6 read mismatch";
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    test_bindless(device);
}
