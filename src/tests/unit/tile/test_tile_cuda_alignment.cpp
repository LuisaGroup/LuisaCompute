// Opt-in dual-entry selection is tested separately from the default native
// suite so inherited environment cannot change its source-shape assertions.
#include "ut/ut.hpp"
#include "test_device.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>
#include <luisa/core/platform.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/stream.h>
#include <luisa/backends/ext/cuda/cuda_graph_ext.h>
#include <luisa/tile/dsl.h>
#include <luisa/tile/runtime.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {
constexpr auto kEnvironment = "LUISA_CUDA_TILE_IR_ALIGNED16";

// This test process compiles serially; restore the inherited value even when
// a source-comparison scope ends early. Other test processes are unaffected.
struct Environment {
    luisa::optional<luisa::string> previous{get_environment_variable(kEnvironment)};
    static void set(const char *value) {
#ifdef _WIN32
        auto status = _putenv_s(kEnvironment, value == nullptr ? "" : value);
#else
        auto status = value == nullptr ? unsetenv(kEnvironment) : setenv(kEnvironment, value, 1);
#endif
        LUISA_ASSERT(status == 0, "Could not set test alignment environment.");
    }
    explicit Environment(const char *value) { set(value); }
    ~Environment() { set(previous ? previous->c_str() : nullptr); }
};

[[nodiscard]] tile::Shader compile(Device &device, const tile::Kernel &kernel, const char *setting) {
    Environment environment{setting};
    return tile::compile(device, kernel, {.lowering = tile::Lowering::NATIVE}, {.enable_fast_math = false});
}

void check_source(const tile::Shader &original, const tile::Shader &candidate, uint32_t mask) {
    expect(static_cast<bool>(original)) << original.metadata().error;
    expect(static_cast<bool>(candidate)) << candidate.metadata().error;
    if (!original || !candidate) { return; }
    auto &&plain = original.metadata().source;
    auto &&source = candidate.metadata().source;
    expect(plain.find("ct::assume_aligned") == string::npos);
    expect(original.metadata().realization.find("aligned16-requested") == string::npos);
    expect(candidate.metadata().realization.find(luisa::format("aligned16-buffer-mask={};", mask)) != string::npos);
    if (mask == 0u) {
        expect(source == plain) << "ineligible opt-in changed the source";
        expect(candidate.metadata().realization.find("aligned16-ineligible") != string::npos);
        return;
    }
    expect(source.starts_with(plain + '\n')) << "original entry changed";
    auto begin = source.find("extern \"C\" __tile_global__ void luisa_tile_aligned16(");
    expect(begin != string::npos);
    if (begin == string::npos) { return; }
    auto aligned = source.substr(begin);
    for (auto i = 0u; i < candidate.metadata().arguments.size(); i++) {
        auto assumption = luisa::format("    buffer{} = ct::assume_aligned<16>(buffer{});\n", i, i);
        auto found = aligned.find(assumption);
        expect((found != string::npos) == ((mask & (1u << i)) != 0u));
        if (found != string::npos) { aligned.erase(found, assumption.size()); }
    }
    auto entry = aligned.find("luisa_tile_aligned16");
    aligned.replace(entry, string_view{"luisa_tile_aligned16"}.size(), "luisa_tile_main");
    expect(aligned == plain.substr(plain.find("extern \"C\""))) << "aligned body differs beyond scalar assumptions";
}

template<typename T>
[[nodiscard]] uint16_t bits(T value) { return std::bit_cast<uint16_t>(value); }

template<typename T>
[[nodiscard]] T value(size_t index, size_t salt) {
    // Non-NaN original storage bits: both zero signs, infinities, normal and
    // subnormal values. Copy/selection must not change any of these bits.
    constexpr std::array<uint16_t, 8u> half_words{0u, 0x8000u, 0x7c00u, 0xfc00u, 0x3c00u, 0xbc00u, 1u, 0x8001u};
    constexpr std::array<uint16_t, 8u> bfloat_words{0u, 0x8000u, 0x7f80u, 0xff80u, 0x3f80u, 0xbf80u, 1u, 0x8001u};
    if constexpr (std::is_same_v<T, half>) { return std::bit_cast<T>(half_words[(index + salt) % 8u]); }
    else { return std::bit_cast<T>(bfloat_words[(index + salt) % 8u]); }
}

template<typename T>
void launch_and_update(Device &device) {
    using namespace tile;
    constexpr size_t rows = 4u, columns = 64u, count = rows * columns, total = count + 129u;
    auto kernel = tile_kernel("independent_buffers_copy", [](TensorView<const T, 2> a,
                                                            TensorView<const T, 2> b,
                                                            TensorView<const T, 2> unused,
                                                            TensorView<T, 2> output) {
        static_cast<void>(unused);
        auto gr = axis("gr", 2), gc = axis("gc", 2), r = axis("r", 2), c = axis("c", 32);
        auto domain = shape(r, c);
        for (auto &block : parallel(shape(gr, gc))) {
            auto origin = coord(block.index(gr) * 2, block.index(gc) * 32);
            auto av = a.tile(origin, domain).load(), bv = b.tile(origin, domain).load();
            auto pick_a = broadcast_to(iota(c), domain) < 16;
            output.tile(origin, domain).store(ite(pick_a, av, bv));
        }
    }).capture(tensor_shape(rows, columns), tensor_shape(rows, columns), tensor_shape(rows, columns), tensor_shape(rows, columns));
    auto original = compile(device, kernel, nullptr);
    auto shader = compile(device, kernel, "1");
    check_source(original, shader, 0b1011u);
    if (!original || !shader) { return; }
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto *extension = device.extension<CudaGraphExt>();
    LUISA_ASSERT(extension != nullptr, "CUDA graph extension is required.");
    std::array<Buffer<T>, 4u> buffers;
    std::array<vector<T>, 4u> host, actual;
    auto guard = T{-719.5f};
    for (auto i = 0u; i < 4u; i++) {
        buffers[i] = device.create_buffer<T>(total);
        host[i].resize(total);
        actual[i].resize(total);
        expect((reinterpret_cast<uintptr_t>(buffers[i].native_handle()) & 15u) == 0u);
    }
    using Offsets = std::array<size_t, 4u>;
    auto upload = [&](Offsets offsets, size_t generation) {
        for (auto i = 0u; i < 4u; i++) {
            std::fill(host[i].begin(), host[i].end(), guard);
            if (i != 3u) {
                for (auto j = size_t{0u}; j < count; j++) { host[i][offsets[i] + j] = value<T>(j, i * 3u + generation); }
            }
            stream << buffers[i].copy_from(span{host[i]});
        }
        stream << synchronize();
    };
    auto command = [&](Offsets offsets) {
        return shader(buffers[0].view(offsets[0], count), buffers[1].view(offsets[1], count),
                      buffers[2].view(offsets[2], count), buffers[3].view(offsets[3], count)).dispatch();
    };
    auto commands = [&](Offsets offsets) {
        auto list = CommandList::create();
        // Repeated complete calls share real output hazards, no fake dependency.
        for (auto repeat = 0u; repeat < 3u; repeat++) { list << command(offsets); }
        return std::move(list.commit()).command_list();
    };
    auto check = [&](Offsets offsets) {
        for (auto i = 0u; i < 4u; i++) { stream << buffers[i].copy_to(span{actual[i]}); }
        stream << synchronize();
        for (auto i = 0u; i < 4u; i++) {
            for (auto j = size_t{0u}; j < total; j++) {
                auto expected = host[i][j];
                if (i == 3u && j >= offsets[i] && j < offsets[i] + count) {
                    auto element = j - offsets[i];
                    auto source = element % 32u < 16u ? 0u : 1u;
                    expected = host[source][offsets[source] + element];
                }
                expect(bits(actual[i][j]) == bits(expected)) << "buffer=" << i << ", word=" << j;
            }
        }
    };
    constexpr std::array layouts{Offsets{64, 64, 64, 64}, Offsets{65, 65, 65, 65},
                                 Offsets{65, 64, 64, 64}, Offsets{64, 65, 64, 64},
                                 Offsets{64, 64, 64, 65}, Offsets{64, 64, 65, 64}};
    for (auto offsets : layouts) {
        upload(offsets, 0u);
        stream << command(offsets) << synchronize();
        check(offsets);
        auto graph = extension->create_graph(commands(offsets));
        LUISA_ASSERT(graph.handle().valid(), "Aligned Tile graph creation failed.");
        auto exec = extension->instantiate(graph.handle().handle);
        LUISA_ASSERT(exec.handle().valid(), "Aligned Tile graph instantiation failed.");
        upload(offsets, 1u);
        extension->launch(exec.handle().handle, stream.handle());
        check(offsets);
    }
    // Update across both alignment directions and verify values/guards. CUDA
    // provides no query of an executable graph's updated kernel parameters;
    // selection here is inferred from the reviewed shared selector, not
    // directly observed. A second executable retains its original bindings.
    auto graph = extension->create_graph(commands(layouts[0]));
    auto exec = extension->instantiate(graph.handle().handle);
    auto unchanged = extension->instantiate(graph.handle().handle);
    LUISA_ASSERT(exec.handle().valid() && unchanged.handle().valid(), "Graph instantiation failed.");
    for (auto index : {1u, 0u, 3u, 5u, 0u}) {
        auto offsets = layouts[index];
        auto updated = extension->update(exec.handle().handle, commands(offsets));
        expect(updated) << "whole graph update=" << index;
        if (!updated) { return; }
        upload(offsets, index + 2u);
        extension->launch(exec.handle().handle, stream.handle());
        check(offsets);
    }
    upload(layouts[0], 9u);
    extension->launch(unchanged.handle().handle, stream.handle());
    check(layouts[0]);
    expect(!extension->update_kernel_node(exec.handle().handle, 0u, make_uint3(1u), {}));
}

template<typename T, size_t Rank>
void rank_copy(Device &device, int64_t width, uint32_t expected_mask) {
    using namespace tile;
    auto kernel = tile_kernel("rank_copy", [=](TensorView<const T, Rank> input, TensorView<T, Rank> output) {
        auto c = axis("c", 32);
        if constexpr (Rank == 1u) {
            for (auto &block : parallel(shape((width + 31) / 32))) {
                auto position = coord(block.index() * 32);
                output.tile(position, shape(c)).store(input.tile(position, shape(c)).load());
            }
        } else {
            auto gb = axis("gb", 2), gr = axis("gr", 2), gc = axis("gc", (width + 31) / 32);
            auto b = axis("b", 1), r = axis("r", 1);
            for (auto &block : parallel(shape(gb, gr, gc))) {
                auto position = coord(block.index(gb), block.index(gr), block.index(gc) * 32);
                output.tile(position, shape(b, r, c)).store(input.tile(position, shape(b, r, c)).load());
            }
        }
    });
    auto captured = [&] {
        if constexpr (Rank == 1u) { return kernel.capture(tensor_shape(width), tensor_shape(width)); }
        else { return kernel.capture(tensor_shape(2, 2, width), tensor_shape(2, 2, width)); }
    }();
    auto original = compile(device, captured, nullptr);
    auto shader = compile(device, captured, "1");
    check_source(original, shader, expected_mask);
    if (!original || !shader) { return; }
    auto count = static_cast<size_t>(width) * (Rank == 1u ? 1u : 4u);
    auto total = count + 130u;
    auto input = device.create_buffer<T>(total), output = device.create_buffer<T>(total);
    vector<T> before(total, T{-719.5f}), blank(total, T{-719.5f}), actual(total), readback(total);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    for (auto offset : {64u, 65u}) {
        std::fill(before.begin(), before.end(), T{-719.5f});
        for (auto i = size_t{0u}; i < count; i++) { before[offset + i] = value<T>(i, 0u); }
        stream << input.copy_from(span{before}) << output.copy_from(span{blank})
               << shader(input.view(offset, count), output.view(offset, count)).dispatch()
               << output.copy_to(span{actual}) << input.copy_to(span{readback}) << synchronize();
        for (auto i = size_t{0u}; i < total; i++) {
            expect(bits(actual[i]) == bits(before[i]));
            expect(bits(readback[i]) == bits(before[i]));
        }
    }
}

void per_root_proof(Device &device) {
    using namespace tile;
    for (auto origin : {int64_t{32}, int64_t{1}, int64_t{-1}}) {
        auto width = origin == 32 ? int64_t{33} : int64_t{64};
        auto kernel = tile_kernel("mixed_proven_unproved_accesses", [=](TensorView<const half, 2> input,
                                                                        TensorView<half, 2> output) {
            auto r = axis("r", 1), c = axis("c", 32);
            for (auto &block : parallel(shape(1))) {
                static_cast<void>(block);
                auto first = input.tile(coord(0, 0), shape(r, c)).load();
                auto second = input.tile(coord(0, origin), shape(r, c)).load();
                output.tile(coord(0, 0), shape(r, c)).store(first);
                output.tile(coord(0, 32), shape(r, c)).store(second);
            }
        }).capture(tensor_shape(1, width), tensor_shape(1, 64));
        auto original = compile(device, kernel, nullptr);
        auto shader = compile(device, kernel, "1");
        // The first read is proved, but the second is respectively a partial
        // tail, an unaligned origin or a negative origin. The input root must
        // never retain an assumption from only its first access. Output is
        // independently eligible, including when the input pointer is odd.
        check_source(original, shader, 2u);
        if (!original || !shader) { return; }
        auto input = device.create_buffer<half>(static_cast<size_t>(width) + 130u);
        auto output = device.create_buffer<half>(194u);
        vector<half> before(input.size(), half{-719.5f}), actual(output.size()), readback(input.size());
        vector<half> blank(output.size(), half{-719.5f});
        for (auto i = int64_t{0}; i < width; i++) { before[65u + i] = value<half>(i, 0u); }
        auto stream = device.create_stream(StreamTag::COMPUTE);
        for (auto offset : {64u, 65u}) {
            stream << input.copy_from(span{before}) << output.copy_from(span{blank})
                   << shader(input.view(65u, static_cast<size_t>(width)), output.view(offset, 64u)).dispatch()
                   << input.copy_to(span{readback}) << output.copy_to(span{actual}) << synchronize();
            for (auto i = size_t{0u}; i < readback.size(); i++) { expect(bits(readback[i]) == bits(before[i])); }
            for (auto i = size_t{0u}; i < actual.size(); i++) {
                auto expected = half{-719.5f};
                if (i >= offset && i < offset + 64u) {
                    auto local = static_cast<int64_t>(i - offset);
                    auto source = local < 32 ? local : origin + local - 32;
                    expected = source >= 0 && source < width ? before[65u + source] : half{0.0f};
                }
                expect(bits(actual[i]) == bits(expected)) << "origin=" << origin << ", word=" << i;
            }
        }
    }
    auto float_kernel = tile_kernel("float_stays_unspecialized", [](TensorView<const float, 1> input,
                                                                   TensorView<float, 1> output) {
        for (auto &block : parallel(shape(1))) {
            static_cast<void>(block);
            output.tile(coord(0), shape(32)).store(input.tile(coord(0), shape(32)).load());
        }
    }).capture(tensor_shape(32), tensor_shape(32));
    auto original = compile(device, float_kernel, nullptr);
    auto requested = compile(device, float_kernel, "1");
    check_source(original, requested, 0u);
    auto other_value = compile(device, float_kernel, "true");
    expect(other_value.metadata().realization.find("aligned16-requested") == string::npos);
    expect(other_value.metadata().source == original.metadata().source);
}
}// namespace

int main(int argc, char *argv[]) {
    if (argc < 2) { return 2; }
    if (argv == nullptr) { return 2; }
    if (argv[0] == nullptr) { return 2; }
    if (argv[1] == nullptr) { return 2; }
    if (std::strcmp(argv[1], "cuda") != 0) { return 2; }
    for (auto name : {"LUISA_CUDA_TILE_IR", kEnvironment}) {
        auto value = get_environment_variable(name);
        if (!value) { return 2; }
        if (std::strcmp(value->c_str(), "1") != 0) { return 2; }
    }
    std::vector<const char *> ut_arguments{argv[0]};
    for (auto i = 2; i < argc; i++) {
        if (argv[i] == nullptr) { return 2; }
        ut_arguments.emplace_back(argv[i]);
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(static_cast<int>(ut_arguments.size()), ut_arguments.data());
    Context context{argv[0]};
    auto device = context.create_device("cuda");
    "tile_cuda_alignment_half_direct_graph_update"_test = [&] { launch_and_update<half>(device); };
    "tile_cuda_alignment_bfloat16_direct_graph_update"_test = [&] { launch_and_update<tile::bfloat16>(device); };
    "tile_cuda_alignment_rank_and_tail_proof"_test = [&] {
        rank_copy<half, 1u>(device, 64, 3u);
        rank_copy<half, 1u>(device, 33, 0u);
        rank_copy<tile::bfloat16, 3u>(device, 64, 3u);
        rank_copy<tile::bfloat16, 3u>(device, 33, 0u);
    };
    "tile_cuda_alignment_per_root_proof"_test = [&] { per_root_proof(device); };
    return 0;
}
