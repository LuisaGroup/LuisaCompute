// Strict Metal AIR runtime test for typed bindless AST normalization.
// This test covers:
// - Typed and typed-uniform scalar/aggregate buffer reads and writes.
// - Typed and typed-uniform byte-address reads.
// - Typed and typed-uniform 2D texture reads and size queries.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/struct.h>

#include <array>
#include <cstddef>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

struct MetalAIRBindlessBool4 {
    bool x;
    bool y;
    bool z;
    bool w;
};

static_assert(sizeof(MetalAIRBindlessBool4) == 4u);
static_assert(offsetof(MetalAIRBindlessBool4, x) == 0u);
static_assert(offsetof(MetalAIRBindlessBool4, y) == 1u);
static_assert(offsetof(MetalAIRBindlessBool4, z) == 2u);
static_assert(offsetof(MetalAIRBindlessBool4, w) == 3u);
static_assert(sizeof(byte4) == 4u);

LUISA_STRUCT(MetalAIRBindlessBool4, x, y, z, w) {};

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    constexpr auto bool_slot = 0u;
    constexpr auto slot = 1u;
    constexpr auto byte_slot = 2u;
    std::array<uint, 4u> buffer_values{11u, 22u, 33u, 44u};
    std::array<MetalAIRBindlessBool4, 2u> bool_values{
        MetalAIRBindlessBool4{true, false, true, false},
        MetalAIRBindlessBool4{false, true, true, true}};
    std::array<byte4, 2u> byte_values{
        byte4{1, 2, 3, 4},
        byte4{5, 6, 7, 8}};
    std::array<float4, 4u> texture_values{
        make_float4(1.0f, 2.0f, 3.0f, 4.0f),
        make_float4(5.0f, 6.0f, 7.0f, 8.0f),
        make_float4(9.0f, 10.0f, 11.0f, 12.0f),
        make_float4(13.0f, 14.0f, 15.0f, 16.0f)};
    std::array<uint, 12u> uint_results{};
    std::array<float4, 2u> texture_results{};

    auto source_buffer = dc->device.create_buffer<uint>(buffer_values.size());
    auto bool_buffer = dc->device.create_buffer<MetalAIRBindlessBool4>(bool_values.size());
    auto byte_buffer = dc->device.create_buffer<byte4>(byte_values.size());
    auto source_texture = dc->device.create_image<float>(
        PixelStorage::FLOAT4, make_uint2(2u));
    auto heap = dc->device.create_bindless_array(3u);
    auto uint_output = dc->device.create_buffer<uint>(uint_results.size());
    auto texture_output = dc->device.create_buffer<float4>(texture_results.size());

    heap.emplace_on_update(bool_slot, bool_buffer);
    heap.emplace_on_update(slot, source_buffer);
    heap.emplace_on_update(byte_slot, byte_buffer);
    heap.emplace_on_update(slot, source_texture, Sampler::point_edge());

    Kernel1D check_typed_bindless = [](BindlessVar bindless,
                                       BufferUInt uint_output_buffer,
                                       BufferFloat4 texture_output_buffer,
                                       UInt dynamic_slot,
                                       UInt dynamic_bool_slot,
                                       UInt dynamic_byte_slot) noexcept {
        set_block_size(64u);

        auto typed_buffer = bindless.buffer<uint>(dynamic_slot, true, false);
        auto typed_uniform_buffer = bindless.buffer<uint>(dynamic_slot, true, true);
        auto typed_bytes = bindless.byte_buffer(dynamic_slot, true, false);
        auto typed_uniform_bytes = bindless.byte_buffer(dynamic_slot, true, true);

        uint_output_buffer.write(0u, typed_buffer.read(0u));
        uint_output_buffer.write(1u, typed_uniform_buffer.read(3u));
        uint_output_buffer.write(2u, typed_bytes.read<uint>(sizeof(uint)));
        uint_output_buffer.write(3u, typed_uniform_bytes.read<uint>(2u * sizeof(uint)));
        typed_buffer.write(1u, typed_buffer.read(1u) + 1000u);
        typed_uniform_buffer.write(2u, typed_uniform_buffer.read(2u) + 2000u);

        auto typed_bool_buffer = bindless.buffer<MetalAIRBindlessBool4>(
            dynamic_bool_slot, true, false);
        auto typed_uniform_bool_buffer = bindless.buffer<MetalAIRBindlessBool4>(
            dynamic_bool_slot, true, true);
        auto bool_value = typed_bool_buffer.read(0u);
        auto uniform_bool_value = typed_uniform_bool_buffer.read(1u);
        uint_output_buffer.write(
            8u, ite(bool_value.x, 1u, 0u) |
                    ite(bool_value.y, 2u, 0u) |
                    ite(bool_value.z, 4u, 0u) |
                    ite(bool_value.w, 8u, 0u));
        uint_output_buffer.write(
            9u, ite(uniform_bool_value.x, 1u, 0u) |
                    ite(uniform_bool_value.y, 2u, 0u) |
                    ite(uniform_bool_value.z, 4u, 0u) |
                    ite(uniform_bool_value.w, 8u, 0u));
        Var<MetalAIRBindlessBool4> reversed_bool_value;
        reversed_bool_value.x = bool_value.w;
        reversed_bool_value.y = bool_value.z;
        reversed_bool_value.z = bool_value.y;
        reversed_bool_value.w = bool_value.x;
        typed_bool_buffer.write(0u, reversed_bool_value);

        auto typed_byte_buffer = bindless.buffer<byte4>(
            dynamic_byte_slot, true, false);
        auto typed_uniform_byte_buffer = bindless.buffer<byte4>(
            dynamic_byte_slot, true, true);
        auto byte_value = typed_byte_buffer.read(0u);
        auto uniform_byte_value = typed_uniform_byte_buffer.read(1u);
        uint_output_buffer.write(
            10u, cast<uint>(byte_value.x) + cast<uint>(byte_value.y) +
                     cast<uint>(byte_value.z) + cast<uint>(byte_value.w));
        uint_output_buffer.write(
            11u, cast<uint>(uniform_byte_value.x) + cast<uint>(uniform_byte_value.y) +
                     cast<uint>(uniform_byte_value.z) + cast<uint>(uniform_byte_value.w));
        typed_byte_buffer.write(0u, byte_value.wzyx());

        auto typed_texture = bindless.tex2d(dynamic_slot, true, false);
        auto typed_uniform_texture = bindless.tex2d(dynamic_slot, true, true);
        auto typed_size = typed_texture.size();
        auto typed_uniform_size = typed_uniform_texture.size();
        uint_output_buffer.write(4u, typed_size.x);
        uint_output_buffer.write(5u, typed_size.y);
        uint_output_buffer.write(6u, typed_uniform_size.x);
        uint_output_buffer.write(7u, typed_uniform_size.y);
        texture_output_buffer.write(
            0u, typed_texture.read(make_uint2(1u, 0u)));
        texture_output_buffer.write(
            1u, typed_uniform_texture.read(make_uint2(0u, 1u)));
    };

    auto shader = dc->device.compile(check_typed_bindless);
    auto stream = dc->device.create_stream();
    stream << source_buffer.copy_from(luisa::span{buffer_values})
           << bool_buffer.copy_from(luisa::span{bool_values})
           << byte_buffer.copy_from(luisa::span{byte_values})
           << source_texture.copy_from(luisa::span{texture_values})
           << heap.update()
           << shader(heap, uint_output, texture_output, slot, bool_slot, byte_slot).dispatch(1u)
           << source_buffer.copy_to(luisa::span{buffer_values})
           << bool_buffer.copy_to(luisa::span{bool_values})
           << byte_buffer.copy_to(luisa::span{byte_values})
           << uint_output.copy_to(luisa::span{uint_results})
           << texture_output.copy_to(luisa::span{texture_results})
           << synchronize();

    expect(uint_results[0] == 11u);
    expect(uint_results[1] == 44u);
    expect(uint_results[2] == 22u);
    expect(uint_results[3] == 33u);
    expect(uint_results[4] == 2u);
    expect(uint_results[5] == 2u);
    expect(uint_results[6] == 2u);
    expect(uint_results[7] == 2u);
    expect(uint_results[8] == 5u);
    expect(uint_results[9] == 14u);
    expect(uint_results[10] == 10u);
    expect(uint_results[11] == 26u);
    expect(buffer_values[1] == 1022u);
    expect(buffer_values[2] == 2033u);
    expect(bool_values[0].x == false);
    expect(bool_values[0].y == true);
    expect(bool_values[0].z == false);
    expect(bool_values[0].w == true);
    expect(byte_values[0].x == 4);
    expect(byte_values[0].y == 3);
    expect(byte_values[0].z == 2);
    expect(byte_values[0].w == 1);
    expect(static_cast<bool>(all(texture_results[0] == texture_values[1])));
    expect(static_cast<bool>(all(texture_results[1] == texture_values[2])));

    return 0;
}
