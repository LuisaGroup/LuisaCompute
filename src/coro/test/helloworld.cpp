#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::compute::coroutine {

}// namespace luisa::compute::coroutine

int main(int argc, char *argv[]) {

    Context context{argv[0]};
    if (argc <= 1) { exit(1); }
    Device device = context.create_device(argv[1]);
    Stream stream = device.create_stream();
    constexpr uint2 resolution = make_uint2(1024, 1024);
    Image<float> image{device.create_image<float>(PixelStorage::BYTE4, resolution)};
    luisa::vector<std::byte> host_image(image.view().size_bytes());

    Kernel1D test = [] {
        coroutine::Generator<uint(uint)> range = [](UInt n) {
            auto x = def(0u);
            $while (x < n) {
                $yield(x);
                x += 1u;
            };
        };
        auto iter = range(10u);
        $while (!iter.update().is_terminated()) {
            auto x = iter.value();
            device_log("x = {}", x);
        };
        for (auto x : range(10u)) {
            device_log("x = {}", x);
        }
    };
    auto shader = device.compile(test);
    stream << shader().dispatch(1) << synchronize();

    coroutine::Coroutine nested2 = [](UInt n) {
        $for (i, n) {
            device_log("nested2: {} / {}", i, n);
            $suspend();
        };
    };

    coroutine::Coroutine nested1 = [&](UInt n) {
        $for (i, n) {
            $await nested2(i);
            device_log("nested1: {} / {}", i, n);
        };
    };

    coroutine::Coroutine top_level = [&]() {
        $await nested1(10u);
    };

    coroutine::StateMachineCoroScheduler sched{device, top_level};
    stream << sched().dispatch(1u) << synchronize();
}
