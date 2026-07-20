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
#include <algorithm>
#include <array>
#include <numeric>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

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

    [[nodiscard]] auto &storage() noexcept { return _buffer; }
    [[nodiscard]] auto &counter() noexcept { return _counter; }
};

void test_atomic_queue(Device &device) {

    // Several blocks are enough to exercise both the shared and global atomic
    // allocation paths without turning a correctness test into a multi-hour
    // benchmark or flooding a backend's command queue.
    static constexpr auto queue_size = 4096u;
    AtomicQueue<uint> q1{device, queue_size};
    AtomicQueue<uint> q2{device, queue_size};

    // Test 1: Push to single queue
    auto test_single = device.compile<1>([&]() noexcept {
        auto x = dispatch_x();
        q1.push(x);
    });

    // Test 2: Push to two queues (duplicates data)
    auto test_double = device.compile<1>([&]() noexcept {
        auto x = dispatch_x();
        q1.push(x);
        q2.push(x);
    });

    // Test 3: Conditional push based on random value
    // Distributes items between two queues
    auto test_select = device.compile<1>([&]() noexcept {
        auto x = dispatch_x();
        auto pred = (x & 1u) == 0u;
        q1.push_if(pred, x);
        q2.push_if(!pred, x);
    });

    auto stream = device.create_stream();
    luisa::vector<uint> values_1(queue_size);
    luisa::vector<uint> values_2(queue_size);
    std::array<uint, 1u> count_1{};
    std::array<uint, 1u> count_2{};

    auto verify_values = [](luisa::span<const uint> values, uint count,
                            luisa::vector<uint> expected, luisa::string_view label) noexcept {
        auto count_valid = count == expected.size() && count <= values.size();
        expect(count_valid) << luisa::format("{} count: expected {}, got {}", label, expected.size(), count);
        if (!count_valid) { return; }
        luisa::vector<uint> actual{values.begin(), values.begin() + count};
        std::sort(actual.begin(), actual.end());
        std::sort(expected.begin(), expected.end());
        expect(actual == expected) << luisa::format("{} contents", label);
    };

    auto run_case = [&](auto &&shader, luisa::vector<uint> expected_1,
                        luisa::vector<uint> expected_2, luisa::string_view name) noexcept {
        CommandList list;
        q1.reset(list);
        q2.reset(list);
        list << shader().dispatch(queue_size);
        stream << list.commit()
               << q1.counter().copy_to(luisa::span{count_1})
               << q2.counter().copy_to(luisa::span{count_2})
               << q1.storage().copy_to(luisa::span{values_1})
               << q2.storage().copy_to(luisa::span{values_2})
               << synchronize();
        verify_values(values_1, count_1[0], std::move(expected_1), luisa::format("{} queue 1", name));
        verify_values(values_2, count_2[0], std::move(expected_2), luisa::format("{} queue 2", name));
    };

    luisa::vector<uint> all(queue_size);
    std::iota(all.begin(), all.end(), 0u);
    luisa::vector<uint> even;
    luisa::vector<uint> odd;
    even.reserve(queue_size / 2u);
    odd.reserve(queue_size / 2u);
    for (auto i = 0u; i < queue_size; i++) {
        (i % 2u == 0u ? even : odd).emplace_back(i);
    }
    run_case(test_single, all, {}, "single");
    run_case(test_double, all, all, "double");
    run_case(test_select, even, odd, "partition");
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
