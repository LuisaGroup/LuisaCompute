#include "ut/ut.hpp"

#include <array>
#include <cstdint>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    Context context{argc > 0 ? argv[0] : ""};
    auto device = context.create_device("simd");

    constexpr auto thread_count = 73u;
    constexpr auto elements_per_slot = 40u;
    constexpr auto second_buffer_offset = 4u;
    constexpr auto output_stride = 5u;
    auto first = device.create_buffer<uint>(elements_per_slot);
    auto second = device.create_buffer<uint>(
        elements_per_slot + 2u * second_buffer_offset);
    auto output = device.create_buffer<uint>(
        thread_count * output_stride);
    auto addresses = device.create_buffer<uint64_t>(thread_count);
    auto bindless = device.create_bindless_array(2u);
    bindless.emplace_on_update(0u, first);
    bindless.emplace_on_update(
        1u, second.view(
                second_buffer_offset, elements_per_slot));

    Kernel1D kernel = [](BindlessVar array, BufferUInt result,
                         BufferVar<uint64_t> address_result) noexcept {
        set_block_size(32u, 1u, 1u);
        auto gid = dispatch_x();
        auto slot = gid & 1u;
        auto element = gid >> 1u;
        auto typed = array.buffer<uint>(slot, true, false);
        auto bytes = array.byte_buffer(slot, true, false);
        auto typed_value = typed.read(element);
        auto byte_value = bytes.read<uint>(
            element * static_cast<uint>(sizeof(uint)));
        auto typed_size = typed.size();
        auto byte_size = bytes.size();
        auto uniform_size =
            array.buffer<uint>(0u, true, true).size();
        result.write(gid * 5u + 0u, typed_value);
        result.write(gid * 5u + 1u, byte_value);
        result.write(gid * 5u + 2u, typed_size);
        result.write(gid * 5u + 3u, byte_size);
        result.write(gid * 5u + 4u, uniform_size);
        address_result.write(gid, typed.device_address());
        typed.write(element, typed_value + 1000u);
    };

    std::array<uint, elements_per_slot> first_host{};
    std::array<uint,
               elements_per_slot + 2u * second_buffer_offset>
        second_host{};
    for (auto i = 0u; i < first_host.size(); i++) {
        first_host[i] = 100u + i;
    }
    second_host.fill(0xdeadbeefu);
    for (auto i = 0u; i < elements_per_slot; i++) {
        second_host[second_buffer_offset + i] = 200u + i;
    }

    auto shader = device.compile(kernel);
    auto stream = device.create_stream();
    luisa::vector<uint> output_host(
        thread_count * output_stride, 0u);
    luisa::vector<uint64_t> address_host(thread_count, 0u);
    stream << first.copy_from(luisa::span{first_host})
           << second.copy_from(luisa::span{second_host})
           << bindless.update()
           << shader(bindless, output, addresses).dispatch(thread_count)
           << output.copy_to(luisa::span{output_host})
           << addresses.copy_to(luisa::span{address_host})
           << first.copy_to(luisa::span{first_host})
           << second.copy_to(luisa::span{second_host})
           << synchronize();

    std::array<uint64_t, 2u> slot_addresses{};
    for (auto gid = 0u; gid < thread_count; gid++) {
        auto slot = gid & 1u;
        auto element = gid >> 1u;
        auto expected = (slot == 0u ? 100u : 200u) + element;
        expect(output_host[gid * output_stride + 0u] == expected)
            << "typed bindless read mismatch";
        expect(output_host[gid * output_stride + 1u] == expected)
            << "byte-addressed bindless read mismatch";
        expect(output_host[gid * output_stride + 2u] ==
               elements_per_slot)
            << "typed bindless size mismatch";
        expect(output_host[gid * output_stride + 3u] ==
               elements_per_slot * sizeof(uint))
            << "byte-addressed bindless size mismatch";
        expect(output_host[gid * output_stride + 4u] ==
               elements_per_slot)
            << "uniform-slot bindless size mismatch";
        expect(address_host[gid] != 0u)
            << "bindless buffer address must be nonzero";
        if (slot_addresses[slot] == 0u) {
            slot_addresses[slot] = address_host[gid];
        } else {
            expect(address_host[gid] == slot_addresses[slot])
                << "each bindless slot must expose a stable address";
        }
    }

    for (auto i = 0u; i < elements_per_slot; i++) {
        auto expected_first = 100u + i + (i <= 36u ? 1000u : 0u);
        auto expected_second = 200u + i + (i <= 35u ? 1000u : 0u);
        expect(first_host[i] == expected_first)
            << "bindless write to the first slot mismatch";
        expect(second_host[second_buffer_offset + i] == expected_second)
            << "bindless write through an offset view mismatch";
    }
    for (auto i = 0u; i < second_buffer_offset; i++) {
        expect(second_host[i] == 0xdeadbeefu)
            << "bindless view write modified its prefix";
        expect(second_host[second_buffer_offset +
                           elements_per_slot + i] == 0xdeadbeefu)
            << "bindless view write modified its suffix";
    }
}
