//
// Created by Mike on 4/6/2023.
//

#include <luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

void test_buffer_io(Device &device) noexcept {

    auto printer = Printer{device};
    auto stream = device.create_stream();
    auto buffer0 = device.create_buffer<float>(4);
    auto buffer1 = device.create_buffer<float>(4);
    auto buffer2 = device.create_buffer<float3>(4);
    auto buffer2view = buffer2.view(2, 2);
    auto buffer2_element = buffer2.view().as<float>();

    auto filler = device.compile<1>([&] {
        auto id = thread_id().x;
        buffer0->write(id, 0.0f);
        buffer1->write(id, 0.0f);
        buffer2->write(id, float3{0.0f});
    });

    auto iteration = device.compile<1>([&] {
        auto id = dispatch_id().x;
        auto res0 = buffer0->read(id);
        auto res1 = buffer1->read(id);
        for (int i = 0; i < 3; i++) {
            buffer2_element->atomic(id * 4 + i).fetch_add(0.0f);
        }

        //auto res2 = buffer2->read(id);
        printer.info("{} : res0 = {}, res1 = {}", id, res0, res1);
        buffer0->write(id, res0 + 1.0f);
        buffer1->write(id, res1 + 1.0f);
        //buffer2->write(id, res2 + 1.0f);
    });

    stream << filler().dispatch(4);

    auto used = device.compile<1>([&]() {
    });
    for (size_t frame = 0; frame < 3; frame++) {
        auto cmdlist = CommandList::create();
        for (size_t i = 0; i < 2; i++) {
            cmdlist << iteration().dispatch(4);
        }

        stream << printer.reset() << cmdlist.commit() << used().dispatch(1) << printer.retrieve() << synchronize();
    }
}

#include <tests/common/config.h>

TEST_CASE("buffer_io") {
    auto argv = luisa::test::argv();
    Context context{argv[0]};
    SUBCASE("cuda") {
        auto device = context.create_device("cuda");
        test_buffer_io(device);
    }
    SUBCASE("dx") {
        auto device = context.create_device("dx");
        test_buffer_io(device);
    }
}