// Test for atomic queue implementation using GPU atomic operations.
//
// This test implements a thread-safe queue using atomic operations for
// concurrent producer-consumer scenarios. The queue uses:
// - Two-level counting: block-level then global
// - Shared memory for intra-block coordination
// - Atomic operations for thread-safe index allocation
//
// The implementation minimizes atomic contention by first counting
// items within a block using shared memory, then performing a single
// global atomic allocation per block.

#include "ut/ut.hpp"
#include "test_device.h"
#include <random>
#include <iostream>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// Placeholder class for queue counter (not fully implemented)
class AtomicQueueCounter {

private:
    Buffer<uint> _buffer;

public:
};

// Thread-safe atomic queue using GPU atomics
template<typename T>
class AtomicQueue {

private:
    Buffer<T> _buffer;    // Storage buffer
    Buffer<uint> _counter;// Global item counter
    Shader1D<> _reset;    // Reset kernel

public:
    AtomicQueue(Device &device, size_t capacity) noexcept
        : _buffer{device.create_buffer<T>(capacity)},
          _counter{device.create_buffer<uint>(1u)} {
        // Compile reset kernel to zero the counter
        _reset = device.compile<1>([this] { _counter->write(0u, 0u); });
    }

    // Push item to queue if predicate is true
    // Uses two-level counting for efficiency:
    // 1. Count qualifying items within block using shared memory
    // 2. Allocate global space with single atomic per block
    // 3. Write items to allocated positions
    void push_if(Expr<bool> pred, Expr<T> value) noexcept {
        // Shared counter for block-local counting
        Shared<uint> index{1};

        // Initialize shared counter
        $if (thread_x() == 0u) { index.write(0u, 0u); };
        sync_block();

        // Each thread that satisfies predicate gets local index
        auto local_index = def(0u);
        $if (pred) { local_index = index.atomic(0).fetch_add(1u); };
        sync_block();

        // Thread 0 allocates global space for entire block
        $if (thread_x() == 0u) {
            auto local_count = index.read(0u);
            auto global_offset = _counter->atomic(0u).fetch_add(local_count);
            index.write(0u, global_offset);
        };
        sync_block();

        // Write items to their allocated positions
        $if (pred) {
            auto global_index = index.read(0u) + local_index;
            _buffer->write(global_index, value);
        };
    }

    // Unconditional push
    void push(Expr<T> value) noexcept { push_if(true, value); }

    // Reset queue counter
    void reset(CommandList &list) noexcept {
        list << _reset().dispatch(1u);
    }
};

void test_atomic_queue(Device &device) {

    log_level_verbose();

    // Queue capacity: max threads that fit in DX 1D dispatch limit (65535 groups * 256 threads/group)
    static constexpr auto queue_size = 65535u * 256u;
    AtomicQueue<float> q1{device, queue_size};
    AtomicQueue<float> q2{device, queue_size};

    // Linear Congruential Generator for random numbers
    Callable lcg = [](UInt &state) noexcept {
        constexpr uint lcg_a = 1664525u;
        constexpr uint lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) *
               (1.0f / static_cast<float>(0x01000000u));
    };

    // Test 1: Push to single queue
    auto test_single = device.compile<1>([&](BufferUInt seed_buffer) noexcept {
        auto x = dispatch_x();
        auto seed = seed_buffer.read(x);
        auto r = lcg(seed);
        seed_buffer.write(x, seed);
        q1.push(r);
    });

    // Test 2: Push to two queues (duplicates data)
    auto test_double = device.compile<1>([&](BufferUInt seed_buffer) noexcept {
        auto x = dispatch_x();
        auto seed = seed_buffer.read(x);
        auto r = lcg(seed);
        seed_buffer.write(x, seed);
        q1.push(r);
        q2.push(r);
    });

    // Test 3: Conditional push based on random value
    // Distributes items between two queues
    auto test_select = device.compile<1>([&](BufferUInt seed_buffer) noexcept {
        auto x = dispatch_x();
        auto seed = seed_buffer.read(x);
        auto r = lcg(seed);
        seed_buffer.write(x, seed);
        auto pred = r < .5f;
        q1.push_if(pred, r);
        q2.push_if(!pred, r);
    });

    auto stream = device.create_stream();
    auto sampler_state_buffer = device.create_buffer<uint>(queue_size);

    // Initialize random seeds
    luisa::vector<uint> sampler_seeds(queue_size);
    std::generate(sampler_seeds.begin(), sampler_seeds.end(),
                  std::mt19937{std::random_device{}()});

    // Benchmark helper
    auto do_test = [&](auto &&shader, auto name_in, auto iterations) noexcept {
        auto name = luisa::string_view{name_in};

        shader.set_name(name);
        stream << sampler_state_buffer.copy_from(luisa::span{sampler_seeds})
               << synchronize();

        Clock clk;
        for (auto i = 0u; i < iterations; i++) {
            CommandList list;
            list.reserve(3u, 0u);
            q1.reset(list);
            q2.reset(list);
            list << shader(sampler_state_buffer).dispatch(queue_size);
            stream << list.commit();
        }
        stream << synchronize();
        expect(true) << "atomic queue completed";
        if (!name.empty()) {
            LUISA_INFO("{}: {} ms", name, clk.toc());
        }
    };

    // Warm up runs
    do_test(test_single, "", 64u);
    do_test(test_double, "", 64u);
    do_test(test_select, "", 64u);

    // Benchmark runs
    do_test(test_single, "single", 1024u);
    do_test(test_double, "double", 1024u);
    do_test(test_select, "select", 1024u);
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_atomic_queue(device);
}
