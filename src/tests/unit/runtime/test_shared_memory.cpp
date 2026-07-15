#include "ut/ut.hpp"
#include "test_device.h"
#include <random>
#include <numeric>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

template<typename T>
class AtomicQueue {

private:
    Buffer<T> _buffer;
    Buffer<uint> _counter;
    Shader1D<> _reset;

public:
    AtomicQueue(Device &device, size_t capacity) noexcept
        : _buffer{device.create_buffer<T>(capacity)},
          _counter{device.create_buffer<uint>(1u)} {
        _reset = device.compile<1>([this] { _counter->write(0u, 0u); });
    }

    [[nodiscard]] auto counter() const noexcept { return _counter.view(); }
    [[nodiscard]] auto buffer() const noexcept { return _buffer.view(); }

    void push(Expr<T> value) noexcept {
        Shared<uint> index{1u};
        $if (thread_x() == 0u) { index.write(0u, 0u); };
        sync_block();
        auto local_index = index.atomic(0u).fetch_add(1u);
        sync_block();
        $if (thread_x() == 0u) {
            auto local_count = index.read(0u);
            auto global_offset = _counter->atomic(0u).fetch_add(local_count);
            index.write(0u, global_offset);
        };
        sync_block();
        auto global_index = index.read(0u) + local_index;
        _buffer->write(global_index, value);
    }

    [[nodiscard]] auto reset() noexcept { return _reset().dispatch(1u); }
};

void test_shared_memory(Device &device) {

    log_level_verbose();

    static constexpr auto queue_size = 8_M;
    AtomicQueue<float> q{device, queue_size};

    Callable lcg = [](UInt &state) noexcept {
        constexpr uint lcg_a = 1664525u;
        constexpr uint lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) *
               (1.0f / static_cast<float>(0x01000000u));
    };

    auto test = device.compile<1>([&](BufferUInt seed_buffer) noexcept {
        auto x = dispatch_x();
        auto seed = seed_buffer.read(x);
        auto r = lcg(seed);
        seed_buffer.write(x, seed);
        q.push(r);
    });

    auto stream = device.create_stream();
    auto sampler_state_buffer = device.create_buffer<uint>(queue_size);

    luisa::vector<uint> sampler_seeds(queue_size);
    std::generate(sampler_seeds.begin(), sampler_seeds.end(),
                  std::mt19937{std::random_device{}()});

    auto n = 0u;
    luisa::vector<float> values(queue_size);

    CommandList cmd_list;
    cmd_list << sampler_state_buffer.copy_from(luisa::span{sampler_seeds})
             << q.reset()
             << test(sampler_state_buffer).dispatch(queue_size)
             << q.buffer().copy_to(luisa::span{values})
             << q.counter().copy_to(luisa::span{&n, 1});
    stream << cmd_list.commit() << synchronize();
    expect(true) << "shared memory test completed";

    auto mean = std::reduce(values.cbegin(), values.cend(), 0.0) / queue_size;
    LUISA_INFO("count = {} (expected {}), mean = {} (expected ~0.5)",
               n, queue_size, mean);
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_shared_memory(device);
}
