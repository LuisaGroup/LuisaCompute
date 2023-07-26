#include "common/config.h"
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

void test_buffer_io(Device &device) noexcept {

    auto printer = Printer{device};
    Stream stream = device.create_stream();
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

    auto iteration = device.compile<1>([&](UInt iter, BufferFloat2 result) {
        auto id = dispatch_id().x;
        auto res0 = buffer0->read(id);
        auto res1 = buffer1->read(id);
        for (int i = 0; i < 3; i++) {
            buffer2_element->atomic(id * 4 + i).fetch_add(0.0f);
        }

        //auto res2 = buffer2->read(id);
        printer.info("{} : res0 = {}, res1 = {}", id, res0, res1);
        result.write(iter * dispatch_size_x() + id, make_float2(res0, res1));

        buffer0->write(id, res0 + 1.0f);
        buffer1->write(id, res1 + 1.0f);
        //buffer2->write(id, res2 + 1.0f);
    });

    stream << filler().dispatch(4);

    auto used = device.compile<1>([&]() {
    });

    auto result_buffer = device.create_buffer<float2>(4 * 2);
    auto result_readback = luisa::vector<float2>(4 * 2);
    for (size_t frame = 0; frame < 3; frame++) {
        auto cmdlist = CommandList::create();
        for (size_t i = 0; i < 2; i++) {
            cmdlist << iteration(i, result_buffer).dispatch(4);
        }

        stream << printer.reset()
               << cmdlist.commit()
               << used().dispatch(1)
               << result_buffer.copy_to(result_readback.data())
               << printer.retrieve()
               << synchronize();

        for (auto i = 0u; i < 2u; i++) {
            for (auto j = 0u; j < 4u; j++) {
                auto res = result_readback[i * 4 + j];
                REQUIRE_EQ(res.x, frame * 2 + i);
                REQUIRE_EQ(res.y, frame * 2 + i);
            }
        }
    }
}

TEST_CASE("buffer_io") {
    auto argv = luisa::test::argv();
    Context context{argv[0]};
    for (auto &&backend : context.installed_backends()) {
        SUBCASE(backend.c_str()) {
            Device device = context.create_device(backend);
            test_buffer_io(device);
        }
    }
}

