// Deterministic shared-memory atomic queue test.
//
// This checks that per-block shared allocation, shared atomics, block barriers,
// and the global reservation atomic together produce exactly one queue entry
// per dispatched thread. Queue ordering is intentionally unspecified, so the
// result is compared as a multiset against an independent host LCG oracle.

#include "ut/ut.hpp"
#include "test_device.h"

#include <algorithm>
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

    static constexpr auto block_size = 256u;
    static constexpr auto queue_size = 4096u;
    static constexpr auto lcg_a = 1664525u;
    static constexpr auto lcg_c = 1013904223u;
    AtomicQueue<float> q{device, queue_size};

    Callable lcg = [](UInt &state) noexcept {
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) *
               (1.0f / static_cast<float>(0x01000000u));
    };

    auto test = device.compile<1>([&](BufferUInt seed_buffer) noexcept {
        set_block_size(block_size, 1u, 1u);
        auto x = dispatch_x();
        auto seed = seed_buffer.read(x);
        auto r = lcg(seed);
        seed_buffer.write(x, seed);
        q.push(r);
    });

    auto stream = device.create_stream();
    auto sampler_state_buffer = device.create_buffer<uint>(queue_size);

    luisa::vector<uint> sampler_seeds(queue_size);
    luisa::vector<uint> updated_seeds(queue_size);
    luisa::vector<float> expected_values(queue_size);
    for (auto i = 0u; i < queue_size; i++) {
        // The odd multiplier makes the initial states distinct modulo 2^32.
        auto seed = 0x31415926u + i * 0x9e3779b9u;
        sampler_seeds[i] = seed;
        seed = lcg_a * seed + lcg_c;
        updated_seeds[i] = seed;
        expected_values[i] = static_cast<float>(seed & 0x00ffffffu) *
                             (1.0f / static_cast<float>(0x01000000u));
    }

    auto n = 0u;
    luisa::vector<float> values(queue_size);
    luisa::vector<uint> device_updated_seeds(queue_size);

    CommandList cmd_list;
    cmd_list << sampler_state_buffer.copy_from(luisa::span{sampler_seeds})
             << q.reset()
             << test(sampler_state_buffer).dispatch(queue_size)
             << q.buffer().copy_to(luisa::span{values})
             << q.counter().copy_to(luisa::span{&n, 1})
             << sampler_state_buffer.copy_to(luisa::span{device_updated_seeds});
    stream << cmd_list.commit() << synchronize();

    expect(eq(n, queue_size))
        << "the shared-memory queue must reserve exactly one entry per thread";
    expect(static_cast<bool>(device_updated_seeds == updated_seeds))
        << "every thread must update its LCG state exactly once";

    // Blocks may reserve global queue ranges in any order. Sorting preserves a
    // strict value-by-value oracle without imposing an invalid scheduling order.
    std::sort(values.begin(), values.end());
    std::sort(expected_values.begin(), expected_values.end());
    expect(static_cast<bool>(values == expected_values))
        << "the queue must contain exactly the host-computed LCG value multiset";
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
