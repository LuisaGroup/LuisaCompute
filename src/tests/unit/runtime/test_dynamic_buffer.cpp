// Test for device-side DynamicBuffer suballocation.
// This covers concurrent allocation, overflow reporting, and byte storage.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/dynamic_buffer.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/dynamic_buffer.h>

#include <array>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_dynamic_buffer(Device &device) {
    constexpr uint capacity = 256u;
    constexpr uint request_count = 96u;
    auto arena = device.create_dynamic_buffer(capacity);
    auto offsets = device.create_buffer<uint>(request_count);
    auto values = device.create_buffer<uint>(request_count);
    std::array<uint, 1u> counter_host{};
    std::array<uint, 1u> overflow_host{};
    std::array<uint, request_count> output_offsets{};
    std::array<uint, request_count> output_values{};

    auto shader = device.compile<1>([](ByteBufferVar storage,
                                       BufferUInt counter,
                                       BufferUInt overflow,
                                       BufferUInt offsets,
                                       BufferUInt values) noexcept {
        auto id = dispatch_id().x;
        auto offset = dynamic_buffer_allocate(counter, overflow, sizeof(uint), 256u);
        offsets.write(id, offset);
        $if (offset != dynamic_buffer_invalid_offset) {
            storage.write(offset, id);
            values.write(id, storage.read<uint>(offset));
        } $else {
            values.write(id, dynamic_buffer_invalid_offset);
        };
    });

    auto stream = device.create_stream();
    stream << arena.reset_counter()
           << arena.reset_overflow()
           << shader(arena.storage(), arena.counter(), arena.overflow(), offsets, values)
                  .dispatch(request_count)
           << arena.counter().copy_to(counter_host.data())
           << arena.overflow().copy_to(overflow_host.data())
           << offsets.copy_to(output_offsets.data())
           << values.copy_to(output_values.data())
           << synchronize();

    expect(counter_host[0] == capacity)
        << "counter must stop at the last successful allocation";
    expect(overflow_host[0] == 1u)
        << "overflow must be reported instead of truncating allocations";
    std::array<bool, capacity / sizeof(uint)> seen{};
    for (auto i = 0u; i < request_count; i++) {
        auto offset = output_offsets[i];
        if (offset != dynamic_buffer_invalid_offset && offset < capacity) {
            seen[offset / sizeof(uint)] = true;
            expect(output_values[i] == i) << "successful allocations must round-trip storage writes";
        } else {
            expect(output_values[i] == dynamic_buffer_invalid_offset)
                << "overflowed allocations must return an invalid offset";
        }
    }
    auto all_ranges_unique = true;
    for (auto value : seen) { all_ranges_unique &= value; }
    expect(all_ranges_unique) << "in-range allocations must address disjoint storage";
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    test_dynamic_buffer(dc->device);
}
