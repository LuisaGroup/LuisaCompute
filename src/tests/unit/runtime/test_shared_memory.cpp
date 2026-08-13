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

void test_shared_histogram_merge(Device &device) {
    constexpr uint block_size = 128u;
    constexpr uint block_count = 96u;
    constexpr uint dispatch_size = block_size * block_count;
    constexpr uint bucket_count = 2u;

    auto histogram = device.create_buffer<uint>(bucket_count);
    auto block_histogram = device.create_buffer<uint>(block_count);
    auto keys = device.create_buffer<uint>(dispatch_size);
    Kernel1D count = [=](BufferUInt global_histogram,
                         BufferUInt block_counts, BufferUInt input_keys,
                         UInt item_count, UInt n) noexcept {
        set_block_size(block_size, 1u, 1u);
        Shared<uint> local_histogram{bucket_count};
        $if (thread_x() < bucket_count) {
            local_histogram[thread_x()] = 0u;
        };
        sync_block();
        $for (item, 0u, item_count) {
            auto index = thread_x() + item * block_size +
                         item_count * block_size * block_x();
            $if (index < n) {
                auto bucket = min(input_keys.read(index), bucket_count - 1u);
                local_histogram.atomic(bucket).fetch_add(1u);
            };
        };
        sync_block();
        $if (thread_x() == 0u) {
            block_counts.write(block_x(), local_histogram[0u]);
        };
        $if (thread_x() < bucket_count) {
            global_histogram.atomic(thread_x()).fetch_add(
                local_histogram[thread_x()]);
        };
    };
    Kernel1D prefix = [=](BufferUInt global_histogram) noexcept {
        set_block_size(32u, 1u, 1u);
        $if (thread_x() == 0u) {
            auto running = def(0u);
            for (auto bucket = 0u; bucket < bucket_count; bucket++) {
                auto count = global_histogram.read(bucket);
                global_histogram.write(bucket, running);
                running += count;
            }
        };
    };
    auto count_shader = device.compile(count);
    auto prefix_shader = device.compile(prefix);
    uint initial[bucket_count]{};
    luisa::vector<uint> input_keys(dispatch_size, 0u);
    uint raw[bucket_count]{};
    luisa::vector<uint> raw_blocks(block_count);
    uint offsets[bucket_count]{};
    auto stream = device.create_stream();
    stream << histogram.copy_from(luisa::span{initial, bucket_count})
           << keys.copy_from(luisa::span{input_keys})
           << count_shader(histogram, block_histogram, keys,
                           1u, dispatch_size)
                  .dispatch(dispatch_size)
           << histogram.copy_to(luisa::span{raw, bucket_count})
           << block_histogram.copy_to(luisa::span{raw_blocks})
           << synchronize();
    auto block_counts_correct = true;
    for (auto block = 0u; block < block_count; block++) {
        if (raw_blocks[block] != block_size) {
            LUISA_WARNING("shared histogram block {} counted {}, expected {}",
                          block, raw_blocks[block], block_size);
            block_counts_correct = false;
            break;
        }
    }
    if (raw[0u] != dispatch_size || raw[1u] != 0u) {
        LUISA_WARNING("shared histogram raw counts: [{}, {}], expected [{}, 0]",
                      raw[0u], raw[1u], dispatch_size);
    }
    expect(block_counts_correct)
        << "discarded LDS atomic results must complete before a block barrier";
    expect(raw[0u] == dispatch_size && raw[1u] == 0u)
        << "a shared block histogram must merge every item globally";
    stream << prefix_shader(histogram).dispatch(32u)
           << histogram.copy_to(luisa::span{offsets, bucket_count})
           << synchronize();
    expect(offsets[0u] == 0u && offsets[1u] == dispatch_size)
        << "a following kernel must observe the complete histogram";
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_shared_memory(device);
    test_shared_histogram_merge(device);
}
