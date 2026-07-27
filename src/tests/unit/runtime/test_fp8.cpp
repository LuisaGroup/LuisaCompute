#include <cstdint>
#include <random>
#include <vector>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

#include "fp8.h"

using namespace luisa;
using namespace luisa::compute;

namespace {

Kernel2D fp8_square_kernel = [](BufferVar<uint> input,
                                BufferVar<uint> output) noexcept {
    set_block_size(16u, 16u, 1u);
    auto packed_index = dispatch_id().x + dispatch_id().y * dispatch_size().x;
    auto packed = input.read(packed_index);
    UInt result = 0u;
    for (auto i : dynamic_range(4u)) {
        auto bits = (packed >> (i * 8u)) & 0xffu;
        auto value = fp8e4m3_to_float()(bits);
        auto squared_bits = fp8e4m3_from_float()(value * value);
        result |= squared_bits << (i * 8u);
    }
    output.write(packed_index, result);
};

}// namespace

int main(int argc, char **argv) {
    if (argc <= 1) {
        LUISA_WARNING("Usage: {} <backend>", argv[0]);
        return 1;
    }

    constexpr size_t side = 512u;
    constexpr size_t value_count = side * side;
    constexpr size_t packed_count = value_count / 4u;
    static_assert(value_count % 4u == 0u);

    std::mt19937 rng{42u};
    std::uniform_real_distribution<float> dist{-100.0f, 100.0f};
    std::vector<uint> packed_input(packed_count);
    std::vector<uint> packed_reference(packed_count);
    for (auto packed_index = 0u; packed_index < packed_count; packed_index++) {
        auto input_word = 0u;
        auto reference_word = 0u;
        for (auto lane = 0u; lane < 4u; lane++) {
            auto input_bits = static_cast<uint>(FP8E4M3::from_float(dist(rng)));
            auto decoded = FP8E4M3::to_float(static_cast<uint8_t>(input_bits));
            auto reference_bits = static_cast<uint>(FP8E4M3::from_float(decoded * decoded));
            input_word |= input_bits << (lane * 8u);
            reference_word |= reference_bits << (lane * 8u);
        }
        packed_input[packed_index] = input_word;
        packed_reference[packed_index] = reference_word;
    }

    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto input = device.create_buffer<uint>(packed_count);
    auto output = device.create_buffer<uint>(packed_count);
    auto shader = device.compile(fp8_square_kernel);
    stream << input.copy_from(luisa::span{packed_input}) << synchronize();

    for (auto i = 0u; i < 2u; i++) {
        stream << shader(input, output).dispatch(side, side / 4u) << synchronize();
    }
    Clock clock;
    auto commands = CommandList::create();
    for (auto i = 0u; i < 8u; i++) {
        commands << shader(input, output).dispatch(side, side / 4u);
    }
    stream << commands.commit() << synchronize();
    LUISA_INFO("FP8 E4M3 square: {:.4f} ms/dispatch", clock.toc() / 8.0);

    std::vector<uint> device_output(packed_count);
    stream << output.copy_to(luisa::span{device_output}) << synchronize();
    auto mismatch_count = 0u;
    for (auto i = 0u; i < packed_count; i++) {
        if (device_output[i] != packed_reference[i]) {
            if (mismatch_count < 16u) {
                LUISA_WARNING("FP8 packed mismatch at {}: expected 0x{:08x}, got 0x{:08x}",
                              i, packed_reference[i], device_output[i]);
            }
            mismatch_count++;
        }
    }
    if (mismatch_count != 0u) {
        LUISA_WARNING("FP8 device conversion had {} mismatched packed words out of {}.",
                      mismatch_count, packed_count);
        return 1;
    }
    LUISA_INFO("FP8 device conversion matches the CPU oracle for {} values.", value_count);
    return 0;
}
