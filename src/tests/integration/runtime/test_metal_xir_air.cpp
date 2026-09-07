#include "ut/ut.hpp"
#include "test_device.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <condition_variable>
#include <limits>
#include <mutex>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/dispatch_indirect.h>
#include <luisa/dsl/struct.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

struct MetalAIRBool4 {
    bool x;
    bool y;
    bool z;
    bool w;
};

struct MetalAIRUniformInner {
    MetalAIRBool4 flags;
    byte4 bytes;
    float scale;
};

struct MetalAIRUniformOuter {
    MetalAIRUniformInner inner;
    uint values[3];
    uint2 extent;
};

static_assert(sizeof(MetalAIRBool4) == 4u);
static_assert(offsetof(MetalAIRBool4, x) == 0u);
static_assert(offsetof(MetalAIRBool4, y) == 1u);
static_assert(offsetof(MetalAIRBool4, z) == 2u);
static_assert(offsetof(MetalAIRBool4, w) == 3u);
static_assert(sizeof(bool4) == 4u);
static_assert(sizeof(byte4) == 4u);

LUISA_STRUCT(MetalAIRBool4, x, y, z, w) {};
LUISA_STRUCT(MetalAIRUniformInner, flags, bytes, scale) {};
LUISA_STRUCT(MetalAIRUniformOuter, inner, values, extent) {};

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    constexpr auto element_count = 64u;
    std::array<uint, element_count> input{};
    std::array<uint, element_count> output{};
    for (auto i = 0u; i < element_count; i++) { input[i] = i * 5u; }

    auto source = dc->device.create_buffer<uint>(element_count);
    auto destination = dc->device.create_buffer<uint>(element_count);
    Kernel1D transform = [](BufferUInt source_buffer,
                            BufferUInt destination_buffer,
                            UInt bias) noexcept {
        set_block_size(32u);
        auto index = dispatch_id().x;
        destination_buffer.write(
            index, source_buffer.read(index) * 3u + bias);
    };
    auto shader = dc->device.compile(transform);
    auto stream = dc->device.create_stream();
    stream << source.copy_from(luisa::span{input})
           << shader(source, destination, 7u).dispatch(element_count)
           << destination.copy_to(luisa::span{output})
           << synchronize();

    for (auto i = 0u; i < element_count; i++) {
        expect(output[i] == input[i] * 3u + 7u);
    }

    auto byte_source = dc->device.create_byte_buffer(sizeof(input));
    auto byte_destination = dc->device.create_byte_buffer(sizeof(output));
    Kernel1D transform_bytes = [](ByteBufferVar source_buffer,
                                  ByteBufferVar destination_buffer) noexcept {
        set_block_size(32u);
        auto index = dispatch_id().x;
        auto byte_offset = index * static_cast<uint>(sizeof(uint));
        destination_buffer.write(
            byte_offset, source_buffer.read<uint>(byte_offset) + 11u);
    };
    auto byte_shader = dc->device.compile(transform_bytes);
    stream << byte_source.copy_from(input.data())
           << byte_shader(byte_source, byte_destination).dispatch(element_count)
           << byte_destination.copy_to(output.data())
           << synchronize();

    for (auto i = 0u; i < element_count; i++) {
        expect(output[i] == input[i] + 11u);
    }

    // A bool in AIR memory occupies a full byte even though its SSA value is
    // i1. Reading offset zero must therefore not observe the true byte next
    // to it at offset one.
    constexpr auto adjacent_bool_bytes = 0x00000100u;
    std::array<uint, 1u> adjacent_bool_mask{};
    auto adjacent_bool_source = dc->device.create_byte_buffer(sizeof(adjacent_bool_bytes));
    auto adjacent_bool_destination = dc->device.create_buffer<uint>(adjacent_bool_mask.size());
    Kernel1D check_adjacent_byte_bools = [](ByteBufferVar source_buffer,
                                            BufferUInt destination_buffer) noexcept {
        auto b0 = source_buffer.read<bool>(0u);
        auto b1 = source_buffer.read<bool>(1u);
        destination_buffer.write(0u, select(0u, 1u, b0) | select(0u, 2u, b1));
    };
    auto adjacent_bool_shader = dc->device.compile(check_adjacent_byte_bools);
    stream << adjacent_bool_source.copy_from(&adjacent_bool_bytes)
           << adjacent_bool_shader(adjacent_bool_source, adjacent_bool_destination).dispatch(1u)
           << adjacent_bool_destination.copy_to(luisa::span{adjacent_bool_mask})
           << synchronize();
    expect(adjacent_bool_mask[0] == 2u);

    constexpr auto block_size = 32u;
    std::array<uint, block_size> reduction_barrier_output{};
    auto reduction_barrier_destination = dc->device.create_buffer<uint>(block_size);
    Kernel1D reduce_and_synchronize = [](BufferUInt destination_buffer) noexcept {
        set_block_size(block_size);
        Shared<uint> shared_values{block_size};
        auto lane = thread_x();
        shared_values.write(lane, lane + 1u);
        sync_block();
        auto values = def(make_uint4(2u, 3u, 4u, 5u));
        auto sum = reduce_sum(values);
        auto product = reduce_prod(values);
        destination_buffer.write(
            dispatch_x(),
            shared_values.read(block_size - 1u - lane) + sum + product);
    };
    auto reduction_barrier_shader = dc->device.compile(reduce_and_synchronize);
    stream << reduction_barrier_shader(reduction_barrier_destination).dispatch(block_size)
           << reduction_barrier_destination.copy_to(luisa::span{reduction_barrier_output})
           << synchronize();

    for (auto i = 0u; i < block_size; i++) {
        expect(reduction_barrier_output[i] == 166u - i);
    }

    // Keep newly supported AIR intrinsics in this strict-mode test so that
    // regressions cannot be hidden by the MSL fallback path.
    constexpr auto bit_pattern = 0x01010100u;
    std::array<uint, 1u> bit_input{bit_pattern};
    std::array<uint, 1u> debug_break_guard{0u};
    std::array<uint, 1u> device_counter{0u};
    std::array<float, 1u> device_cas{0.f};
    std::array<uint, 5u> uint_output{};
    std::array<float, 3u> float_output{};
    auto bit_input_buffer = dc->device.create_buffer<uint>(bit_input.size());
    auto debug_break_guard_buffer = dc->device.create_buffer<uint>(debug_break_guard.size());
    auto device_counter_buffer = dc->device.create_buffer<uint>(device_counter.size());
    auto device_cas_buffer = dc->device.create_buffer<float>(device_cas.size());
    auto uint_output_buffer = dc->device.create_buffer<uint>(uint_output.size());
    auto float_output_buffer = dc->device.create_buffer<float>(float_output.size());
    Kernel1D check_air_intrinsics = [](BufferUInt bits,
                                       BufferUInt debug_break_condition,
                                       BufferUInt device_count,
                                       BufferFloat device_float,
                                       BufferUInt uint_results,
                                       BufferFloat float_results) noexcept {
        set_block_size(block_size);
        Shared<uint> shared_count{1u};
        Shared<float> shared_float{1u};
        auto lane = thread_x();
        // The guard is a runtime buffer read, so LLVM cannot fold the branch
        // or the debug trap away. The zero host value keeps the trap untaken
        // while pipeline creation and dispatch still compile it.
        $if (debug_break_condition.read(0u) != 0u) {
            $debug_break(lane);
        };
        $if (lane == 0u) {
            shared_count.write(0u, 0u);
            shared_float.write(0u, 0.f);
        };
        sync_block();
        device_count.atomic(0u).fetch_add(1u);
        shared_count.atomic(0u).fetch_add(1u);
        $if (lane == 0u) {
            auto pattern = bits.read(0u);
            uint_results.write(0u, clz(pattern));
            uint_results.write(1u, ctz(pattern));
            uint_results.write(2u, popcount(pattern));
            uint_results.write(3u, reverse(pattern));
            float_results.write(0u, device_float.atomic(0u).compare_exchange(0.f, 2.5f));
            float_results.write(1u, shared_float.atomic(0u).compare_exchange(0.f, 3.5f));
        };
        sync_block();
        $if (lane == 0u) {
            uint_results.write(4u, shared_count.read(0u));
            float_results.write(2u, shared_float.read(0u));
        };
    };
    auto air_intrinsics_shader = dc->device.compile(check_air_intrinsics);
    stream << bit_input_buffer.copy_from(luisa::span{bit_input})
           << debug_break_guard_buffer.copy_from(luisa::span{debug_break_guard})
           << device_counter_buffer.copy_from(luisa::span{device_counter})
           << device_cas_buffer.copy_from(luisa::span{device_cas})
           << air_intrinsics_shader(bit_input_buffer, debug_break_guard_buffer,
                                    device_counter_buffer, device_cas_buffer,
                                    uint_output_buffer, float_output_buffer)
                  .dispatch(block_size)
           << device_counter_buffer.copy_to(luisa::span{device_counter})
           << device_cas_buffer.copy_to(luisa::span{device_cas})
           << uint_output_buffer.copy_to(luisa::span{uint_output})
           << float_output_buffer.copy_to(luisa::span{float_output})
           << synchronize();
    expect(device_counter[0] == block_size);
    expect(uint_output[0] == std::countl_zero(bit_pattern));
    expect(uint_output[1] == std::countr_zero(bit_pattern));
    expect(uint_output[2] == std::popcount(bit_pattern));
    expect(uint_output[3] == 0x00808080u);
    expect(uint_output[4] == block_size);
    expect(device_cas[0] == 2.5f);
    expect(float_output[0] == 0.f);
    expect(float_output[1] == 0.f);
    expect(float_output[2] == 3.5f);

    constexpr auto air_warp_width = 32u;
    constexpr auto air_warp_uint_stride = 18u;
    std::array<uint, air_warp_width * air_warp_uint_stride> air_warp_uint_output{};
    std::array<float, air_warp_width> air_warp_half_output{};
    auto air_warp_uint_buffer = dc->device.create_buffer<uint>(air_warp_uint_output.size());
    auto air_warp_half_buffer = dc->device.create_buffer<float>(air_warp_half_output.size());
    Kernel1D check_air_warp = [](BufferUInt uint_results,
                                 BufferFloat half_results) noexcept {
        set_block_size(32u);
        set_warp_size(32u);
        auto lane = warp_lane_id();
        auto value = lane + 1u;
        auto even = (lane & 1u) == 0u;
        auto base = lane * 18u;
        uint_results.write(base + 0u, warp_prefix_sum(value));
        uint_results.write(base + 1u, warp_prefix_count_bits(even));
        uint_results.write(base + 2u, warp_read_lane(value, 31u));
        uint_results.write(base + 3u, warp_read_first_active_lane(value));
        uint_results.write(base + 4u, warp_active_sum(value));
        uint_results.write(base + 5u, warp_active_product(ite(lane == 0u, 2u, 1u)));
        uint_results.write(base + 6u, warp_active_min(value));
        uint_results.write(base + 7u, warp_active_max(value));
        uint_results.write(base + 8u, warp_active_bit_and(0xf0f0f0f0u));
        uint_results.write(base + 9u, warp_active_bit_or(1u << lane));
        uint_results.write(base + 10u, warp_active_bit_xor(1u << lane));
        uint_results.write(base + 11u, warp_active_count_bits(even));
        uint_results.write(base + 12u, ite(warp_active_all(lane < 32u), 1u, 0u));
        uint_results.write(base + 13u, ite(warp_active_any(lane == 7u), 1u, 0u));
        uint_results.write(base + 14u, warp_active_bit_mask(even).x);
        uint_results.write(base + 15u, ite(warp_is_first_active_lane(), 1u, 0u));
        uint_results.write(base + 16u, warp_first_active_lane());
        auto equal = warp_active_all_equal(make_uint4(7u, 7u, 7u, 7u));
        auto equal_mask = ite(equal.x, 1u, 0u) |
                          ite(equal.y, 2u, 0u) |
                          ite(equal.z, 4u, 0u) |
                          ite(equal.w, 8u, 0u);
        uint_results.write(base + 17u, equal_mask);
        auto half_prefix = warp_prefix_sum(make_half4(.5_h));
        half_results.write(lane, cast<float>(half_prefix.x));
        reorder_shader_execution(lane, 5u);
    };
    auto air_warp_shader = dc->device.compile(check_air_warp);
    stream << air_warp_shader(air_warp_uint_buffer, air_warp_half_buffer).dispatch(air_warp_width)
           << air_warp_uint_buffer.copy_to(luisa::span{air_warp_uint_output})
           << air_warp_half_buffer.copy_to(luisa::span{air_warp_half_output})
           << synchronize();
    for (auto lane = 0u; lane < air_warp_width; lane++) {
        auto base = lane * air_warp_uint_stride;
        expect(air_warp_uint_output[base + 0u] == lane * (lane + 1u) / 2u);
        expect(air_warp_uint_output[base + 1u] == (lane + 1u) / 2u);
        expect(air_warp_uint_output[base + 2u] == air_warp_width);
        expect(air_warp_uint_output[base + 3u] == 1u);
        expect(air_warp_uint_output[base + 4u] == air_warp_width * (air_warp_width + 1u) / 2u);
        expect(air_warp_uint_output[base + 5u] == 2u);
        expect(air_warp_uint_output[base + 6u] == 1u);
        expect(air_warp_uint_output[base + 7u] == air_warp_width);
        expect(air_warp_uint_output[base + 8u] == 0xf0f0f0f0u);
        expect(air_warp_uint_output[base + 9u] == 0xffffffffu);
        expect(air_warp_uint_output[base + 10u] == 0xffffffffu);
        expect(air_warp_uint_output[base + 11u] == air_warp_width / 2u);
        expect(air_warp_uint_output[base + 12u] == 1u);
        expect(air_warp_uint_output[base + 13u] == 1u);
        expect(air_warp_uint_output[base + 14u] == 0x55555555u);
        expect(air_warp_uint_output[base + 15u] == (lane == 0u ? 1u : 0u));
        expect(air_warp_uint_output[base + 16u] == 0u);
        expect(air_warp_uint_output[base + 17u] == 15u);
        expect(air_warp_half_output[lane] == static_cast<float>(lane) * .5f);
    }

    constexpr uint2 air_image_size{2u, 3u};
    constexpr uint3 air_volume_size{2u, 3u, 4u};
    constexpr size_t air_image_texels = 6u;
    constexpr size_t air_volume_texels = 24u;
    std::array<float4, air_image_texels> image_float_input{}, image_float_output{};
    std::array<int4, air_image_texels> image_int_input{}, image_int_output{};
    std::array<uint4, air_image_texels> image_uint_input{}, image_uint_output{};
    std::array<float4, air_volume_texels> volume_float_input{}, volume_float_output{};
    std::array<int4, air_volume_texels> volume_int_input{}, volume_int_output{};
    std::array<uint4, air_volume_texels> volume_uint_input{}, volume_uint_output{};
    for (size_t i = 0u; i < air_image_texels; i++) {
        auto s = static_cast<int>(i);
        auto u = static_cast<uint>(i);
        auto f = static_cast<float>(i);
        image_float_input[i] = make_float4(f + .25f, f + .5f, -f, 1.f);
        image_int_input[i] = make_int4(s, -s - 1, s * 3, -7);
        image_uint_input[i] = make_uint4(u, u + 1u, u * 2u, 9u);
    }
    for (size_t i = 0u; i < air_volume_texels; i++) {
        auto s = static_cast<int>(i);
        auto u = static_cast<uint>(i);
        auto f = static_cast<float>(i);
        volume_float_input[i] = make_float4(f + .25f, f + .5f, -f, 2.f);
        volume_int_input[i] = make_int4(s + 10, -s - 2, s * 2, -9);
        volume_uint_input[i] = make_uint4(u + 10u, u + 2u, u * 3u, 11u);
    }
    auto image_float_source = dc->device.create_image<float>(PixelStorage::FLOAT4, air_image_size);
    auto image_float_destination = dc->device.create_image<float>(PixelStorage::FLOAT4, air_image_size);
    auto image_int_source = dc->device.create_image<int>(PixelStorage::INT4, air_image_size);
    auto image_int_destination = dc->device.create_image<int>(PixelStorage::INT4, air_image_size);
    auto image_uint_source = dc->device.create_image<uint>(PixelStorage::INT4, air_image_size);
    auto image_uint_destination = dc->device.create_image<uint>(PixelStorage::INT4, air_image_size);
    auto volume_float_source = dc->device.create_volume<float>(PixelStorage::FLOAT4, air_volume_size);
    auto volume_float_destination = dc->device.create_volume<float>(PixelStorage::FLOAT4, air_volume_size);
    auto volume_int_source = dc->device.create_volume<int>(PixelStorage::INT4, air_volume_size);
    auto volume_int_destination = dc->device.create_volume<int>(PixelStorage::INT4, air_volume_size);
    auto volume_uint_source = dc->device.create_volume<uint>(PixelStorage::INT4, air_volume_size);
    auto volume_uint_destination = dc->device.create_volume<uint>(PixelStorage::INT4, air_volume_size);
    Kernel3D check_air_textures = [](ImageFloat image_f_src, ImageFloat image_f_dst,
                                     ImageInt image_i_src, ImageInt image_i_dst,
                                     ImageUInt image_u_src, ImageUInt image_u_dst,
                                     VolumeFloat volume_f_src, VolumeFloat volume_f_dst,
                                     VolumeInt volume_i_src, VolumeInt volume_i_dst,
                                     VolumeUInt volume_u_src, VolumeUInt volume_u_dst) noexcept {
        auto p = dispatch_id();
        auto volume_extent = volume_f_src.size();
        volume_f_dst.write(
            p, volume_f_src.read(p) +
                   make_float4(cast<float>(volume_extent.x), cast<float>(volume_extent.y),
                               cast<float>(volume_extent.z), -1.f));
        volume_i_dst.write(
            p, volume_i_src.read(p) +
                   make_int4(cast<int>(volume_extent.x), cast<int>(volume_extent.y),
                             cast<int>(volume_extent.z), -1));
        volume_u_dst.write(
            p, volume_u_src.read(p) +
                   make_uint4(volume_extent.x, volume_extent.y, volume_extent.z, 1u));
        $if (p.z == 0u) {
            auto q = p.xy();
            auto image_extent = image_f_src.size();
            image_f_dst.write(
                q, image_f_src.read(q) +
                       make_float4(cast<float>(image_extent.x), cast<float>(image_extent.y), 1.f, -1.f));
            image_i_dst.write(
                q, image_i_src.read(q) +
                       make_int4(cast<int>(image_extent.x), cast<int>(image_extent.y), 1, -1));
            image_u_dst.write(
                q, image_u_src.read(q) +
                       make_uint4(image_extent.x, image_extent.y, 1u, 1u));
        };
    };
    auto air_texture_shader = dc->device.compile(check_air_textures);
    stream << image_float_source.copy_from(luisa::span{image_float_input})
           << image_int_source.copy_from(luisa::span{image_int_input})
           << image_uint_source.copy_from(luisa::span{image_uint_input})
           << volume_float_source.copy_from(luisa::span{volume_float_input})
           << volume_int_source.copy_from(luisa::span{volume_int_input})
           << volume_uint_source.copy_from(luisa::span{volume_uint_input})
           << air_texture_shader(image_float_source, image_float_destination,
                                 image_int_source, image_int_destination,
                                 image_uint_source, image_uint_destination,
                                 volume_float_source, volume_float_destination,
                                 volume_int_source, volume_int_destination,
                                 volume_uint_source, volume_uint_destination)
                  .dispatch(air_volume_size)
           << image_float_destination.copy_to(luisa::span{image_float_output})
           << image_int_destination.copy_to(luisa::span{image_int_output})
           << image_uint_destination.copy_to(luisa::span{image_uint_output})
           << volume_float_destination.copy_to(luisa::span{volume_float_output})
           << volume_int_destination.copy_to(luisa::span{volume_int_output})
           << volume_uint_destination.copy_to(luisa::span{volume_uint_output})
           << synchronize();
    auto image_float_delta = make_float4(2.f, 3.f, 1.f, -1.f);
    auto image_int_delta = make_int4(2, 3, 1, -1);
    auto image_uint_delta = make_uint4(2u, 3u, 1u, 1u);
    for (size_t i = 0u; i < air_image_texels; i++) {
        expect(static_cast<bool>(all(image_float_output[i] == image_float_input[i] + image_float_delta)));
        expect(static_cast<bool>(all(image_int_output[i] == image_int_input[i] + image_int_delta)));
        expect(static_cast<bool>(all(image_uint_output[i] == image_uint_input[i] + image_uint_delta)));
    }
    auto volume_float_delta = make_float4(2.f, 3.f, 4.f, -1.f);
    auto volume_int_delta = make_int4(2, 3, 4, -1);
    auto volume_uint_delta = make_uint4(2u, 3u, 4u, 1u);
    for (size_t i = 0u; i < air_volume_texels; i++) {
        expect(static_cast<bool>(all(volume_float_output[i] == volume_float_input[i] + volume_float_delta)));
        expect(static_cast<bool>(all(volume_int_output[i] == volume_int_input[i] + volume_int_delta)));
        expect(static_cast<bool>(all(volume_uint_output[i] == volume_uint_input[i] + volume_uint_delta)));
    }

    // Direct image/volume variables currently expose read/write/size through
    // the public DSL but not sampling, so construct the corresponding AST
    // calls explicitly. Keep these resources sample-only in this kernel: AIR
    // requires a sampled texture argument to use `air.sample` access rather
    // than the `air.read`/`air.write` access used above.
    constexpr auto sample_color = make_float4(.25f, .5f, .75f, 1.f);
    std::array<float4, 4u> sample_image_input{};
    std::array<float4, 8u> sample_volume_input{};
    std::array<float4, 8u> sample_output{};
    sample_image_input.fill(sample_color);
    sample_volume_input.fill(sample_color);
    auto sample_image = dc->device.create_image<float>(
        PixelStorage::FLOAT4, make_uint2(2u));
    auto sample_volume = dc->device.create_volume<float>(
        PixelStorage::FLOAT4, make_uint3(2u));
    auto sample_destination = dc->device.create_buffer<float4>(sample_output.size());
    Kernel1D check_air_direct_sampling = [](ImageFloat image,
                                            VolumeFloat volume,
                                            BufferFloat4 destination_buffer) noexcept {
        auto f = luisa::compute::detail::FunctionBuilder::current();
        auto point = f->literal(Type::of<uint>(), static_cast<uint>(SamplerFilter::POINT));
        auto linear_point = f->literal(Type::of<uint>(), static_cast<uint>(SamplerFilter::LINEAR_POINT));
        auto linear_linear = f->literal(Type::of<uint>(), static_cast<uint>(SamplerFilter::LINEAR_LINEAR));
        auto anisotropic = f->literal(Type::of<uint>(), static_cast<uint>(SamplerFilter::ANISOTROPIC));
        auto edge = f->literal(Type::of<uint>(), static_cast<uint>(SamplerAddress::EDGE));
        auto repeat = f->literal(Type::of<uint>(), static_cast<uint>(SamplerAddress::REPEAT));
        auto mirror = f->literal(Type::of<uint>(), static_cast<uint>(SamplerAddress::MIRROR));
        auto zero = f->literal(Type::of<uint>(), static_cast<uint>(SamplerAddress::ZERO));
        auto uv = def(make_float2(.5f));
        auto uvw = def(make_float3(.5f));
        auto ddx2 = def(make_float2(.25f, 0.f));
        auto ddy2 = def(make_float2(0.f, .25f));
        auto ddx3 = def(make_float3(.25f, 0.f, 0.f));
        auto ddy3 = def(make_float3(0.f, .25f, 0.f));
        auto level = def(0.f);
        auto min_level = def(0.f);
        auto sample2d = [&](CallOp op,
                            std::initializer_list<const Expression *> args) noexcept {
            return def<float4>(f->call(Type::of<float4>(), op, args));
        };
        auto sample3d = [&](CallOp op,
                            std::initializer_list<const Expression *> args) noexcept {
            return def<float4>(f->call(Type::of<float4>(), op, args));
        };
        destination_buffer.write(
            0u, sample2d(CallOp::TEXTURE2D_SAMPLE,
                         {image.expression(), uv.expression(), point, edge}));
        destination_buffer.write(
            1u, sample2d(CallOp::TEXTURE2D_SAMPLE_LEVEL,
                         {image.expression(), uv.expression(), level.expression(), linear_point, repeat}));
        destination_buffer.write(
            2u, sample2d(CallOp::TEXTURE2D_SAMPLE_GRAD,
                         {image.expression(), uv.expression(), ddx2.expression(), ddy2.expression(),
                          linear_linear, mirror}));
        destination_buffer.write(
            3u, sample2d(CallOp::TEXTURE2D_SAMPLE_GRAD_LEVEL,
                         {image.expression(), uv.expression(), ddx2.expression(), ddy2.expression(),
                          min_level.expression(), anisotropic, zero}));
        destination_buffer.write(
            4u, sample3d(CallOp::TEXTURE3D_SAMPLE,
                         {volume.expression(), uvw.expression(), point, zero}));
        destination_buffer.write(
            5u, sample3d(CallOp::TEXTURE3D_SAMPLE_LEVEL,
                         {volume.expression(), uvw.expression(), level.expression(), linear_point, mirror}));
        destination_buffer.write(
            6u, sample3d(CallOp::TEXTURE3D_SAMPLE_GRAD,
                         {volume.expression(), uvw.expression(), ddx3.expression(), ddy3.expression(),
                          linear_linear, repeat}));
        destination_buffer.write(
            7u, sample3d(CallOp::TEXTURE3D_SAMPLE_GRAD_LEVEL,
                         {volume.expression(), uvw.expression(), ddx3.expression(), ddy3.expression(),
                          min_level.expression(), anisotropic, edge}));
    };
    auto air_direct_sampling_shader = dc->device.compile(check_air_direct_sampling);
    stream << sample_image.copy_from(luisa::span{sample_image_input})
           << sample_volume.copy_from(luisa::span{sample_volume_input})
           << air_direct_sampling_shader(sample_image, sample_volume, sample_destination).dispatch(1u)
           << sample_destination.copy_to(luisa::span{sample_output})
           << synchronize();
    for (auto value : sample_output) {
        expect(static_cast<bool>(all(value == sample_color)));
    }

    // Sampling and writing the same direct texture requires two physical AIR
    // argument-buffer fields: read_write and sample. The runtime binds the
    // same MTLTexture twice, while LLVM codegen routes each intrinsic through
    // the matching handle and metadata entry.
    std::array<float4, 1u> split_texture_input{sample_color};
    std::array<float4, 1u> split_texture_output{};
    auto split_texture = dc->device.create_image<float>(
        PixelStorage::FLOAT4, make_uint2(1u));
    Kernel1D check_air_split_texture = [](ImageFloat image) noexcept {
        auto f = luisa::compute::detail::FunctionBuilder::current();
        auto filter = f->literal(
            Type::of<uint>(), static_cast<uint>(SamplerFilter::POINT));
        auto address = f->literal(
            Type::of<uint>(), static_cast<uint>(SamplerAddress::EDGE));
        auto uv = def(make_float2(.5f));
        auto sampled = def<float4>(f->call(
            Type::of<float4>(), CallOp::TEXTURE2D_SAMPLE,
            {image.expression(), uv.expression(), filter, address}));
        image.write(make_uint2(0u), sampled + make_float4(1.f, 2.f, 3.f, 4.f));
    };
    auto split_texture_shader = dc->device.compile(
        check_air_split_texture);
    stream << split_texture.copy_from(luisa::span{split_texture_input})
           << split_texture_shader(split_texture).dispatch(1u)
           << split_texture.copy_to(luisa::span{split_texture_output})
           << synchronize();
    expect(static_cast<bool>(all(
        split_texture_output[0] ==
        sample_color + make_float4(1.f, 2.f, 3.f, 4.f))));

    // Exercise AIR's flattened indirect-argument locations across a texture,
    // a nested structure (including bool fields, byte4, and an array), a
    // top-level array, a buffer, and a trailing scalar. This ordering catches
    // reflection metadata that confuses logical argument ordinals with the
    // flattened AIR location index.
    MetalAIRUniformOuter aggregate_uniform{
        .inner = {
            .flags = {true, false, true, true},
            .bytes = byte4{1, 2, 3, 4},
            .scale = 2.f},
        .values = {11u, 13u, 17u},
        .extent = make_uint2(19u, 23u)};
    std::array<uint, 3u> array_uniform{29u, 31u, 37u};
    std::array<uint, 1u> aggregate_output{};
    std::array<uint4, 1u> aggregate_image_output{};
    auto aggregate_image = dc->device.create_image<uint>(PixelStorage::INT4, make_uint2(1u));
    auto aggregate_destination = dc->device.create_buffer<uint>(aggregate_output.size());
    Kernel1D check_aggregate_arguments = [](ImageUInt image,
                                            Var<MetalAIRUniformOuter> settings,
                                            ArrayUInt<3u> factors,
                                            BufferUInt destination_buffer,
                                            UInt tail) noexcept {
        auto flags = settings.inner.flags;
        auto flag_mask = ite(flags.x, 1u, 0u) |
                         ite(flags.y, 2u, 0u) |
                         ite(flags.z, 4u, 0u) |
                         ite(flags.w, 8u, 0u);
        auto bytes = settings.inner.bytes;
        auto byte_sum = cast<uint>(bytes.x) + cast<uint>(bytes.y) +
                        cast<uint>(bytes.z) + cast<uint>(bytes.w);
        auto value_sum = settings.values[0u] + settings.values[1u] + settings.values[2u];
        auto factor_sum = factors[0u] + factors[1u] + factors[2u];
        auto extent_sum = settings.extent.x + settings.extent.y;
        auto result = flag_mask + byte_sum + cast<uint>(settings.inner.scale) +
                      value_sum + factor_sum + extent_sum + tail;
        destination_buffer.write(0u, result);
        image.write(make_uint2(0u), make_uint4(result, flag_mask, byte_sum, factors[2u]));
    };
    auto aggregate_shader = dc->device.compile(check_aggregate_arguments);
    stream << aggregate_shader(aggregate_image, aggregate_uniform, array_uniform,
                               aggregate_destination, 41u)
                  .dispatch(1u)
           << aggregate_destination.copy_to(luisa::span{aggregate_output})
           << aggregate_image.copy_to(luisa::span{aggregate_image_output})
           << synchronize();
    constexpr auto expected_aggregate = 13u + 10u + 2u +
                                        (11u + 13u + 17u) +
                                        (29u + 31u + 37u) +
                                        (19u + 23u) + 41u;
    expect(aggregate_output[0] == expected_aggregate);
    expect(static_cast<bool>(all(
        aggregate_image_output[0] == make_uint4(expected_aggregate, 13u, 10u, 37u))));

    // A 4 KiB uniform array plus the destination binding exceeds Metal's
    // setBytes limit and must use the staged MTLBuffer root path.
    std::array<uint, 1024u> large_root_uniform{};
    large_root_uniform.front() = 0x12345678u;
    large_root_uniform.back() = 0x9abcdef0u;
    std::array<uint, 1u> large_root_output{};
    auto large_root_destination =
        dc->device.create_buffer<uint>(large_root_output.size());
    Kernel1D check_large_root = [](
                                    ArrayUInt<1024u> values,
                                    BufferUInt destination) noexcept {
        set_block_size(32u);
        destination.write(0u, values[0u] ^ values[1023u]);
    };
    auto large_root_shader = dc->device.compile(check_large_root);
    stream << large_root_shader(large_root_uniform, large_root_destination)
                  .dispatch(1u)
           << large_root_destination.copy_to(luisa::span{large_root_output})
           << synchronize();
    expect(large_root_output[0] ==
           (large_root_uniform.front() ^ large_root_uniform.back()));

    // The GPU-generated ICB keeps buffer(0) as an indirect reference to the
    // staged root block. Exercise the same >4 KiB payload through that path so
    // its upload allocation and residency declaration remain covered.
    Kernel1D prepare_large_root_indirect = [](
                                               Var<IndirectDispatchBuffer> dispatch_buffer) noexcept {
        dispatch_buffer.set_dispatch_count(1u);
        dispatch_buffer.set_kernel(
            0u, make_uint3(32u, 1u, 1u), make_uint3(1u), 0u);
    };
    auto prepare_large_root_indirect_shader =
        dc->device.compile(prepare_large_root_indirect);
    auto large_root_indirect_buffer =
        dc->device.create_indirect_dispatch_buffer(1u);
    large_root_uniform.front() = 0x0f1e2d3cu;
    large_root_uniform.back() = 0x4b5a6978u;
    constexpr std::array<uint, 1u> zero_large_root_output{};
    stream << large_root_destination.copy_from(
                  luisa::span{zero_large_root_output})
           << prepare_large_root_indirect_shader(large_root_indirect_buffer)
                  .dispatch(1u)
           << large_root_shader(large_root_uniform, large_root_destination)
                  .dispatch(large_root_indirect_buffer)
           << large_root_destination.copy_to(luisa::span{large_root_output})
           << synchronize();
    expect(large_root_output[0] ==
           (large_root_uniform.front() ^ large_root_uniform.back()));

    std::array<float, 2u> comparison_input{
        std::numeric_limits<float>::quiet_NaN(), 1.f};
    std::array<uint, comparison_input.size()> comparison_output{};
    auto comparison_source = dc->device.create_buffer<float>(comparison_input.size());
    auto comparison_destination = dc->device.create_buffer<uint>(comparison_output.size());
    Kernel1D compare_nan = [](BufferFloat source_buffer,
                              BufferUInt destination_buffer) noexcept {
        auto index = dispatch_x();
        auto value = source_buffer.read(index);
        destination_buffer.write(index, ite(value != value, 1u, 0u));
    };
    auto comparison_shader = dc->device.compile(
        compare_nan, {.enable_fast_math = false});
    stream << comparison_source.copy_from(luisa::span{comparison_input})
           << comparison_shader(comparison_source, comparison_destination).dispatch(comparison_input.size())
           << comparison_destination.copy_to(luisa::span{comparison_output})
           << synchronize();
    expect(comparison_output[0] == 1u);
    expect(comparison_output[1] == 0u);

    std::array<float, 10u> math_semantic_output{};
    auto math_semantic_destination = dc->device.create_buffer<float>(math_semantic_output.size());
    Kernel1D check_nan_math = [](BufferFloat source_buffer,
                                 BufferFloat destination_buffer) noexcept {
        auto nan = source_buffer.read(0u);
        destination_buffer.write(0u, min(1.f, nan));
        destination_buffer.write(1u, min(nan, 1.f));
        destination_buffer.write(2u, max(1.f, nan));
        destination_buffer.write(3u, max(nan, 1.f));
        destination_buffer.write(4u, clamp(nan, 0.f, 1.f));
        destination_buffer.write(5u, saturate(nan));
        destination_buffer.write(6u, step(0.f, nan));
        auto values = def(make_float4(nan, 2.f, 3.f, 4.f));
        destination_buffer.write(7u, reduce_min(values));
        destination_buffer.write(8u, reduce_max(values));
        destination_buffer.write(9u, smoothstep(0.f, 1.f, nan));
    };
    auto math_semantic_shader = dc->device.compile(
        check_nan_math, {.enable_fast_math = false});
    stream << math_semantic_shader(comparison_source, math_semantic_destination).dispatch(1u)
           << math_semantic_destination.copy_to(luisa::span{math_semantic_output})
           << synchronize();
    constexpr std::array expected_nan_math{1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 2.f, 4.f, 0.f};
    for (auto i = 0u; i < expected_nan_math.size(); i++) {
        expect(math_semantic_output[i] == expected_nan_math[i]);
    }

    std::array<MetalAIRBool4, 4u> bool_struct_input{
        MetalAIRBool4{false, false, false, false},
        MetalAIRBool4{true, false, true, false},
        MetalAIRBool4{false, true, false, true},
        MetalAIRBool4{true, true, true, true}};
    std::array<MetalAIRBool4, bool_struct_input.size()> bool_struct_output{};
    std::array<uint, bool_struct_input.size()> bool_struct_masks{};
    auto bool_struct_source = dc->device.create_buffer<MetalAIRBool4>(bool_struct_input.size());
    auto bool_struct_destination = dc->device.create_buffer<MetalAIRBool4>(bool_struct_output.size());
    auto bool_struct_mask_destination = dc->device.create_buffer<uint>(bool_struct_masks.size());
    Kernel1D check_bool_struct_layout = [](BufferVar<MetalAIRBool4> source_buffer,
                                           BufferVar<MetalAIRBool4> destination_buffer,
                                           BufferUInt mask_buffer) noexcept {
        auto index = dispatch_x();
        Var value = source_buffer.read(index);
        Var<MetalAIRBool4> reversed;
        reversed.x = value.w;
        reversed.y = value.z;
        reversed.z = value.y;
        reversed.w = value.x;
        destination_buffer.write(index, reversed);
        auto mask = ite(value.x, 1u, 0u) |
                    ite(value.y, 2u, 0u) |
                    ite(value.z, 4u, 0u) |
                    ite(value.w, 8u, 0u);
        mask_buffer.write(index, mask);
    };
    auto bool_struct_shader = dc->device.compile(check_bool_struct_layout);
    stream << bool_struct_source.copy_from(luisa::span{bool_struct_input})
           << bool_struct_shader(bool_struct_source,
                                 bool_struct_destination,
                                 bool_struct_mask_destination)
                  .dispatch(bool_struct_input.size())
           << bool_struct_destination.copy_to(luisa::span{bool_struct_output})
           << bool_struct_mask_destination.copy_to(luisa::span{bool_struct_masks})
           << synchronize();
    for (auto i = 0u; i < bool_struct_input.size(); i++) {
        auto expected_mask = static_cast<uint>(bool_struct_input[i].x) |
                             static_cast<uint>(bool_struct_input[i].y) << 1u |
                             static_cast<uint>(bool_struct_input[i].z) << 2u |
                             static_cast<uint>(bool_struct_input[i].w) << 3u;
        expect(bool_struct_masks[i] == expected_mask);
        expect(bool_struct_output[i].x == bool_struct_input[i].w);
        expect(bool_struct_output[i].y == bool_struct_input[i].z);
        expect(bool_struct_output[i].z == bool_struct_input[i].y);
        expect(bool_struct_output[i].w == bool_struct_input[i].x);
    }

    std::array<bool4, 4u> bool_vector_input{
        make_bool4(false, false, false, false),
        make_bool4(true, false, true, false),
        make_bool4(false, true, false, true),
        make_bool4(true, true, true, true)};
    std::array<bool4, bool_vector_input.size()> bool_vector_output{};
    std::array<uint, bool_vector_input.size()> bool_vector_masks{};
    auto bool_vector_source = dc->device.create_buffer<bool4>(bool_vector_input.size());
    auto bool_vector_destination = dc->device.create_buffer<bool4>(bool_vector_output.size());
    auto bool_vector_mask_destination = dc->device.create_buffer<uint>(bool_vector_masks.size());
    Kernel1D check_bool_vector_layout = [](BufferVar<bool4> source_buffer,
                                           BufferVar<bool4> destination_buffer,
                                           BufferUInt mask_buffer) noexcept {
        auto index = dispatch_x();
        Var value = source_buffer.read(index);
        destination_buffer.write(
            index, make_bool4(value.w, value.z, value.y, value.x));
        auto mask = ite(value.x, 1u, 0u) |
                    ite(value.y, 2u, 0u) |
                    ite(value.z, 4u, 0u) |
                    ite(value.w, 8u, 0u);
        mask_buffer.write(index, mask);
    };
    auto bool_vector_shader = dc->device.compile(check_bool_vector_layout);
    stream << bool_vector_source.copy_from(luisa::span{bool_vector_input})
           << bool_vector_shader(bool_vector_source,
                                 bool_vector_destination,
                                 bool_vector_mask_destination)
                  .dispatch(bool_vector_input.size())
           << bool_vector_destination.copy_to(luisa::span{bool_vector_output})
           << bool_vector_mask_destination.copy_to(luisa::span{bool_vector_masks})
           << synchronize();
    for (auto i = 0u; i < bool_vector_input.size(); i++) {
        auto expected_mask = static_cast<uint>(bool_vector_input[i].x) |
                             static_cast<uint>(bool_vector_input[i].y) << 1u |
                             static_cast<uint>(bool_vector_input[i].z) << 2u |
                             static_cast<uint>(bool_vector_input[i].w) << 3u;
        expect(bool_vector_masks[i] == expected_mask);
        expect(bool_vector_output[i].x == bool_vector_input[i].w);
        expect(bool_vector_output[i].y == bool_vector_input[i].z);
        expect(bool_vector_output[i].z == bool_vector_input[i].y);
        expect(bool_vector_output[i].w == bool_vector_input[i].x);
    }

    std::array<byte4, 4u> byte_vector_input{
        byte4{1, 2, 3, 4},
        byte4{5, 6, 7, 8},
        byte4{9, 10, 11, 12},
        byte4{13, 14, 15, 16}};
    std::array<byte4, byte_vector_input.size()> byte_vector_output{};
    auto byte_vector_source = dc->device.create_buffer<byte4>(byte_vector_input.size());
    auto byte_vector_destination = dc->device.create_buffer<byte4>(byte_vector_output.size());
    Kernel1D check_byte_vector_layout = [](BufferVar<byte4> source_buffer,
                                           BufferVar<byte4> destination_buffer) noexcept {
        auto index = dispatch_x();
        auto value = source_buffer.read(index);
        destination_buffer.write(index, value.wzyx());
    };
    auto byte_vector_shader = dc->device.compile(check_byte_vector_layout);
    stream << byte_vector_source.copy_from(luisa::span{byte_vector_input})
           << byte_vector_shader(byte_vector_source, byte_vector_destination)
                  .dispatch(byte_vector_input.size())
           << byte_vector_destination.copy_to(luisa::span{byte_vector_output})
           << synchronize();
    for (auto i = 0u; i < byte_vector_input.size(); i++) {
        expect(byte_vector_output[i].x == byte_vector_input[i].w);
        expect(byte_vector_output[i].y == byte_vector_input[i].z);
        expect(byte_vector_output[i].z == byte_vector_input[i].y);
        expect(byte_vector_output[i].w == byte_vector_input[i].x);
    }

    // PACK/UNPACK uses a storage wrapper aligned to at least four bytes. The
    // nested bool structure must therefore occupy one uint (four byte-sized
    // bools), while float3 occupies four uints because its ABI size/alignment
    // is 16 bytes. Run packing and unpacking as separate dispatches so the
    // round trip crosses device memory rather than folding locally.
    constexpr MetalAIRBool4 packed_flags{true, false, true, true};
    constexpr auto packed_vector = make_float3(1.25f, -2.5f, 9.75f);
    constexpr byte4 packed_bytes{1, 2, 3, 4};
    constexpr auto packed_bool = true;
    constexpr auto packed_byte = static_cast<byte>(0x5au);
    constexpr auto packed_short = static_cast<short>(0x1234u);
    std::array<uint, 9u> packed_words{};
    std::array<MetalAIRBool4, 1u> unpacked_flags{};
    std::array<float3, 1u> unpacked_vectors{};
    std::array<byte4, 1u> unpacked_bytes{};
    std::array<bool, 1u> unpacked_scalar_bools{};
    std::array<byte, 1u> unpacked_scalar_bytes{};
    std::array<short, 1u> unpacked_scalar_shorts{};
    auto packed_word_buffer = dc->device.create_buffer<uint>(packed_words.size());
    auto unpacked_flag_buffer = dc->device.create_buffer<MetalAIRBool4>(unpacked_flags.size());
    auto unpacked_vector_buffer = dc->device.create_buffer<float3>(unpacked_vectors.size());
    auto unpacked_byte_buffer = dc->device.create_buffer<byte4>(unpacked_bytes.size());
    auto unpacked_scalar_bool_buffer = dc->device.create_buffer<bool>(unpacked_scalar_bools.size());
    auto unpacked_scalar_byte_buffer = dc->device.create_buffer<byte>(unpacked_scalar_bytes.size());
    auto unpacked_scalar_short_buffer = dc->device.create_buffer<short>(unpacked_scalar_shorts.size());
    Kernel1D pack_air_values = [](BufferUInt words,
                                  Var<MetalAIRBool4> flags,
                                  Float3 vector,
                                  Var<byte4> bytes,
                                  Bool scalar_bool,
                                  Var<byte> scalar_byte,
                                  Short scalar_short) noexcept {
        pack_to(flags, words, 0u);
        pack_to(vector, words, 1u);
        pack_to(bytes, words, 5u);
        pack_to(scalar_bool, words, 6u);
        pack_to(scalar_byte, words, 7u);
        pack_to(scalar_short, words, 8u);
    };
    Kernel1D unpack_air_values = [](BufferUInt words,
                                    BufferVar<MetalAIRBool4> flags,
                                    BufferFloat3 vectors,
                                    BufferVar<byte4> bytes,
                                    BufferBool scalar_bools,
                                    BufferVar<byte> scalar_bytes,
                                    BufferShort scalar_shorts) noexcept {
        flags.write(0u, unpack_from<MetalAIRBool4>(words, 0u));
        vectors.write(0u, unpack_from<float3>(words, 1u));
        bytes.write(0u, unpack_from<byte4>(words, 5u));
        scalar_bools.write(0u, unpack_from<bool>(words, 6u));
        scalar_bytes.write(0u, unpack_from<byte>(words, 7u));
        scalar_shorts.write(0u, unpack_from<short>(words, 8u));
    };
    auto pack_air_shader = dc->device.compile(pack_air_values);
    auto unpack_air_shader = dc->device.compile(unpack_air_values);
    stream << pack_air_shader(packed_word_buffer, packed_flags,
                              packed_vector, packed_bytes,
                              packed_bool, packed_byte, packed_short)
                  .dispatch(1u)
           << unpack_air_shader(packed_word_buffer, unpacked_flag_buffer,
                                unpacked_vector_buffer, unpacked_byte_buffer,
                                unpacked_scalar_bool_buffer,
                                unpacked_scalar_byte_buffer,
                                unpacked_scalar_short_buffer)
                  .dispatch(1u)
           << packed_word_buffer.copy_to(luisa::span{packed_words})
           << unpacked_flag_buffer.copy_to(luisa::span{unpacked_flags})
           << unpacked_vector_buffer.copy_to(luisa::span{unpacked_vectors})
           << unpacked_byte_buffer.copy_to(luisa::span{unpacked_bytes})
           << unpacked_scalar_bool_buffer.copy_to(luisa::span{unpacked_scalar_bools})
           << unpacked_scalar_byte_buffer.copy_to(luisa::span{unpacked_scalar_bytes})
           << unpacked_scalar_short_buffer.copy_to(luisa::span{unpacked_scalar_shorts})
           << synchronize();
    expect(packed_words[0] == 0x01010001u);
    expect(packed_words[1] == std::bit_cast<uint>(packed_vector.x));
    expect(packed_words[2] == std::bit_cast<uint>(packed_vector.y));
    expect(packed_words[3] == std::bit_cast<uint>(packed_vector.z));
    expect(packed_words[5] == 0x04030201u);
    expect(packed_words[6] == 0x00000001u);
    expect(packed_words[7] == 0x0000005au);
    expect(packed_words[8] == 0x00001234u);
    expect(unpacked_flags[0].x == packed_flags.x);
    expect(unpacked_flags[0].y == packed_flags.y);
    expect(unpacked_flags[0].z == packed_flags.z);
    expect(unpacked_flags[0].w == packed_flags.w);
    expect(static_cast<bool>(all(unpacked_vectors[0] == packed_vector)));
    expect(unpacked_bytes[0].x == packed_bytes.x);
    expect(unpacked_bytes[0].y == packed_bytes.y);
    expect(unpacked_bytes[0].z == packed_bytes.z);
    expect(unpacked_bytes[0].w == packed_bytes.w);
    expect(unpacked_scalar_bools[0] == packed_bool);
    expect(unpacked_scalar_bytes[0] == packed_byte);
    expect(unpacked_scalar_shorts[0] == packed_short);

    // Volatile reads use a device fence before the volatile LLVM load;
    // volatile writes use a volatile store followed by the same AIR fence.
    Kernel1D transform_volatile = [](BufferUInt source_buffer,
                                     BufferUInt destination_buffer,
                                     ByteBufferVar byte_source_buffer,
                                     ByteBufferVar byte_destination_buffer) noexcept {
        auto index = dispatch_id().x;
        auto byte_offset = index * static_cast<uint>(sizeof(uint));
        destination_buffer.volatile_write(
            index, source_buffer.volatile_read(index) + 13u);
        byte_destination_buffer.volatile_write(
            byte_offset, byte_source_buffer.volatile_read<uint>(byte_offset) + 17u);
    };
    auto volatile_shader = dc->device.compile(transform_volatile);
    std::array<uint, element_count> volatile_typed_output{};
    std::array<uint, element_count> volatile_byte_output{};
    stream << volatile_shader(source, destination, byte_source, byte_destination).dispatch(element_count)
           << destination.copy_to(luisa::span{volatile_typed_output})
           << byte_destination.copy_to(volatile_byte_output.data())
           << synchronize();

    for (auto i = 0u; i < element_count; i++) {
        expect(volatile_typed_output[i] == input[i] + 13u);
        expect(volatile_byte_output[i] == input[i] + 17u);
    }

    // Metal's source backend currently defines lc_assert as a no-op. AIR must
    // accept the same instruction until a shared device assertion ABI exists.
    std::array<uint, 1u> assert_output{};
    auto assert_output_buffer = dc->device.create_buffer<uint>(assert_output.size());
    Kernel1D assert_is_noop = [](BufferUInt output_buffer) noexcept {
        device_assert(false, "Metal AIR assertion parity");
        output_buffer.write(0u, 0x51a7u);
    };
    auto assert_shader = dc->device.compile(assert_is_noop);
    stream << assert_shader(assert_output_buffer).dispatch(1u)
           << assert_output_buffer.copy_to(luisa::span{assert_output})
           << synchronize();
    expect(assert_output[0] == 0x51a7u);

    // The indirect entry carries the same hidden printer binding and format
    // table as the direct entry. Keep the dispatch tiny and sort the callback
    // records because GPU reservation order is intentionally unspecified.
    constexpr auto indirect_print_count = 8u;
    constexpr auto indirect_print_kernel_id = 37u;
    constexpr auto indirect_print_block_size = make_uint3(32u, 1u, 1u);
    Kernel1D prepare_indirect_print = [=](Var<IndirectDispatchBuffer> dispatch_buffer) noexcept {
        dispatch_buffer.set_dispatch_count(1u);
        dispatch_buffer.set_kernel(
            0u, indirect_print_block_size,
            make_uint3(indirect_print_count, 1u, 1u),
            indirect_print_kernel_id);
    };
    Kernel1D indirect_print = [=]() noexcept {
        set_block_size(indirect_print_block_size.x);
        device_log("indirect-air-print kernel={} lane={}",
                   kernel_id(), dispatch_x());
    };
    auto prepare_indirect_print_shader = dc->device.compile(prepare_indirect_print);
    auto indirect_print_shader = dc->device.compile(indirect_print);
    auto indirect_print_buffer = dc->device.create_indirect_dispatch_buffer(1u);
    std::mutex indirect_print_mutex;
    std::condition_variable indirect_print_cv;
    luisa::vector<luisa::string> indirect_print_messages;
    auto indirect_print_stream = dc->device.create_stream();
    indirect_print_stream.set_log_callback(
        [&](luisa::string_view message) noexcept {
            if (!message.starts_with("indirect-air-print ")) { return; }
            {
                std::scoped_lock lock{indirect_print_mutex};
                indirect_print_messages.emplace_back(message);
            }
            indirect_print_cv.notify_one();
        });
    indirect_print_stream
        << prepare_indirect_print_shader(indirect_print_buffer).dispatch(1u)
        << indirect_print_shader().dispatch(indirect_print_buffer)
        << synchronize();
    auto indirect_print_completed = [&] {
        std::unique_lock lock{indirect_print_mutex};
        return indirect_print_cv.wait_for(
            lock, std::chrono::seconds{5}, [&]() noexcept {
                return indirect_print_messages.size() == indirect_print_count;
            });
    }();
    expect(indirect_print_completed);
    if (indirect_print_completed) {
        std::sort(indirect_print_messages.begin(), indirect_print_messages.end());
        for (auto lane = 0u; lane < indirect_print_count; lane++) {
            expect(indirect_print_messages[lane] ==
                   luisa::format("indirect-air-print kernel={} lane={}",
                                 indirect_print_kernel_id, lane));
        }
    }
}
