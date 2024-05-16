/**
 * @file test_bindless.cpp
 * @brief The Bindless Test Suite
 * @author sailing-innocent
 * @date 2024-05-16
 */

#include "common/config.h"
#include "luisa/core/logging.h"
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/dsl/syntax.h>
using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {

int test_bindless_buffer(Device &device) {
    log_level_verbose();
    BindlessArray heap = device.create_bindless_array(64);
    Stream stream = device.create_stream();
    Buffer<int> buffer0 = device.create_buffer<int>(1);
    Buffer<int> buffer1 = device.create_buffer<int>(1);
    Buffer<int> out_buffer = device.create_buffer<int>(2);
    constexpr int offset = 5;
    heap.emplace_on_update(offset + 0, buffer0);
    heap.emplace_on_update(offset + 1, buffer1);
    Kernel1D kernel = [&] {
        out_buffer->write(dispatch_id().x, heap->buffer<int>(dispatch_id().x + offset).read(0));
    };
    auto shader = device.compile(kernel);
    int v0 = 555;
    int v1 = 666;
    int result[2];
    stream << heap.update() << synchronize();
    stream << buffer0.copy_from(&v0) << buffer1.copy_from(&v1) << shader().dispatch(2) << out_buffer.copy_to(result) << synchronize();
    CHECK(result[0] == v0);
    CHECK(result[1] == v1);
    return 0;
}

}// namespace luisa::test

TEST_SUITE("runtime") {
    using namespace luisa::test;
    LUISA_TEST_CASE_WITH_DEVICE("bindless_buffer", test_bindless_buffer(device) == 0);
}