// Opt-in dual-entry selection is tested separately from the default native
// suite so inherited environment cannot change its source-shape assertions.
#include "ut/ut.hpp"
#include "test_device.h"
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <regex>
#include <string_view>
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

// Replace only the three exact generated lines, after checking their root,
// static extents, Tile shape, origins and binding against the original entry.
// The existing final full-body equality remains the final check.
[[nodiscard]] bool restore_partition_loads(luisa::string &aligned,
                                           luisa::string_view plain, uint32_t mask) {
    auto numbers = [](const std::string &text, std::vector<uint64_t> &out) {
        auto first = text.data(), last = first + text.size();
        while (first != last) {
            uint64_t value{};
            auto parsed = std::from_chars(first, last, value);
            if (parsed.ec != std::errc{} || parsed.ptr == first || value == 0u) { return false; }
            out.emplace_back(value);
            first = parsed.ptr;
            if (first == last) { break; }
            if (last - first < 2 || first[0] != ',' || first[1] != ' ') { return false; }
            first += 2;
            if (first == last) { return false; }
        }
        return !out.empty();
    };
    const std::regex triple{
        R"(([ ]*)auto (mem[0-9]+)_span = ct::tensor_span\{buffer([0-9]+), ct::extents<long long, ([0-9, ]+)>\{\}\};\n\1auto \2_partition = ct::partition_view\{\2_span, ct::shape<([0-9, ]+)>\{\}\};\n\1auto (v[0-9]+) = \2_partition\.(load|load_masked)\(([^\n]+)\);\n)"};
    auto cursor = size_t{0u};
    std::match_results<luisa::string::const_iterator> match;
    while (std::regex_search(aligned.cbegin() + cursor, aligned.cend(), match, triple)) {
        auto indent = match[1].str(), prefix = match[2].str(), root_text = match[3].str();
        auto view_text = match[4].str(), tile_text = match[5].str();
        auto result = match[6].str(), method = match[7].str(), chunks = match[8].str();
        auto masked = method == "load_masked";
        if (masked) {
            constexpr auto padding = std::string_view{"ct::view_padding_zero_t{}, "};
            if (!chunks.starts_with(padding)) { return false; }
            chunks.erase(0u, padding.size());
        }
        uint32_t root{};
        auto parsed = std::from_chars(root_text.data(), root_text.data() + root_text.size(), root);
        if (parsed.ec != std::errc{} || parsed.ptr != root_text.data() + root_text.size() || root >= 32u) { return false; }
        if ((mask & (uint32_t{1u} << root)) == 0u) { return false; }
        std::vector<uint64_t> view, tile;
        if (!numbers(view_text, view) || !numbers(tile_text, tile)) { return false; }
        if (view.size() != tile.size()) { return false; }
        auto binding = luisa::format("{}auto {} = ct::load({}_ptr);\n", indent, result, prefix);
        if (masked) {
            // Tie the exact +0 fallback type to this root's original ABI.
            auto signature_end = plain.find(") {\n");
            if (signature_end == luisa::string_view::npos) { return false; }
            auto signature = plain.substr(0u, signature_end + 1u);
            luisa::string scalar;
            for (auto type : {"__half", "__nv_bfloat16"}) {
                auto parameter = luisa::format("{} *buffer{}", type, root);
                auto found = signature.find(parameter);
                if (found != luisa::string_view::npos && found + parameter.size() < signature.size() &&
                    (signature[found + parameter.size()] == ',' || signature[found + parameter.size()] == ')')) {
                    if (!scalar.empty()) { return false; }
                    scalar = type;
                }
            }
            if (scalar.empty()) { return false; }
            binding = luisa::format("{}auto {} = ct::load_masked({}_ptr, {}_mask, ct::element_cast<{}>(0));\n",
                                    indent, result, prefix, prefix, scalar);
        }
        auto original = plain.find(binding);
        if (original == luisa::string_view::npos) { return false; }
        if (plain.find(binding, original + binding.size()) != luisa::string_view::npos) { return false; }
        auto preceding = [&](luisa::string_view line) {
            auto found = plain.find(line);
            if (found == luisa::string_view::npos || found >= original) { return false; }
            return plain.find(line, found + line.size()) == luisa::string_view::npos;
        };
        if (!preceding(luisa::format("{}auto {}_lane = ct::iota<ct::tile<long long, ct::shape<{}>>>();\n", indent, prefix, tile_text))) { return false; }
        if (masked && !preceding(luisa::format("{}auto {}_zero = ct::full<ct::tile<long long, ct::shape<{}>>>(0ll);\n", indent, prefix, tile_text))) { return false; }
        uint64_t volume{1u};
        for (auto extent : tile) {
            if (extent > UINT64_MAX / volume) { return false; }
            volume *= extent;
        }
        luisa::string pointer{"0ll"}, predicate{"true"};
        auto chunk_begin = size_t{0u};
        for (auto i = size_t{0u}; i < tile.size(); i++) {
            if (!masked && tile[i] > view[i]) { return false; }
            volume /= tile[i];
            auto suffix = luisa::format(") / {}ll", tile[i]);
            if (chunk_begin >= chunks.size() || chunks[chunk_begin] != '(') { return false; }
            auto close = chunks.find(suffix.c_str(), chunk_begin + 1u);
            if (close == std::string::npos) { return false; }
            auto origin = chunks.substr(chunk_begin + 1u, close - chunk_begin - 1u);
            if (origin.empty()) { return false; }
            if (!preceding(luisa::format("{}auto {}_c{} = {} + ({}_lane / {}ll) % {}ll;\n",
                                         indent, prefix, i, origin, prefix, volume, tile[i]))) { return false; }
            chunk_begin = close + suffix.size();
            if (i + 1u != tile.size()) {
                if (chunks.compare(chunk_begin, 2u, ", ") != 0) { return false; }
                chunk_begin += 2u;
            }
            auto coordinate = luisa::format("{}_c{}", prefix, i);
            predicate = luisa::format("({} && ({} >= 0ll) && ({} < {}ll))", predicate, coordinate, coordinate, view[i]);
            auto safe_coordinate = masked ? luisa::format("ct::select(({} >= 0ll) && ({} < {}ll), {}, {}_zero)",
                                                          coordinate, coordinate, view[i], coordinate, prefix) : coordinate;
            pointer = luisa::format("(({}) * {}ll + {})", pointer, view[i], safe_coordinate);
        }
        if (chunk_begin != chunks.size()) { return false; }
        if (masked) {
            if (!preceding(luisa::format("{}auto {}_mask = {};\n", indent, prefix, predicate))) { return false; }
            pointer = luisa::format("ct::select({}_mask, {}, {}_zero)", prefix, pointer, prefix);
        }
        if (!preceding(luisa::format("{}auto {}_ptr = buffer{} + {};\n", indent, prefix, root, pointer))) { return false; }
        auto position = cursor + static_cast<size_t>(match.position());
        aligned.replace(position, static_cast<size_t>(match.length()), binding);
        cursor = position + binding.size();
    }
    // Malformed or extra replacement statements cannot be silently ignored.
    if (aligned.find("ct::tensor_span") != luisa::string::npos) { return false; }
    if (aligned.find("ct::partition_view") != luisa::string::npos) { return false; }
    return true;
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
    auto restored = restore_partition_loads(aligned, plain, mask);
    expect(restored) << "aligned view replacement does not match original addressing";
    if (!restored) { return; }
    auto entry = aligned.find("luisa_tile_aligned16");
    aligned.replace(entry, string_view{"luisa_tile_aligned16"}.size(), "luisa_tile_main");
    expect(aligned == plain.substr(plain.find("extern \"C\""))) << "aligned body differs beyond scalar assumptions and verified load representation";
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

// One program avoids inter-program races. The first immutable load must keep
// the pre-store bits, and the second must observe the intervening store.
template<typename T>
void snapshot_order(Device &device) {
    using namespace tile;
    constexpr auto count = size_t{64u};
    auto kernel = tile_kernel("immutable_snapshot_order", [](TensorView<T, 1> inout,
                                                              TensorView<const T, 1> replacement,
                                                              TensorView<T, 1> output) {
        for (auto &program : parallel(shape(1))) {
            static_cast<void>(program);
            auto tile_shape = shape(axis("element", 64));
            auto before = inout.tile(coord(0), tile_shape).load();
            auto next = replacement.tile(coord(0), tile_shape).load();
            inout.tile(coord(0), tile_shape).store(next);
            auto after = inout.tile(coord(0), tile_shape).load();
            output.tile(coord(0), tile_shape).store(before);
            output.tile(coord(64), tile_shape).store(after);
        }
    }).capture(tensor_shape(count), tensor_shape(count), tensor_shape(count * 2u));
    auto original = compile(device, kernel, nullptr);
    auto shader = compile(device, kernel, "1");
    check_source(original, shader, 7u);
    if (!original || !shader) { return; }
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto *extension = device.extension<CudaGraphExt>();
    LUISA_ASSERT(extension != nullptr, "CUDA graph extension is required.");
    auto inout = device.create_buffer<T>(count + 130u);
    auto replacement = device.create_buffer<T>(count + 130u);
    auto output = device.create_buffer<T>(2u * count + 130u);
    vector<T> before(inout.size()), next(replacement.size()), blank(output.size(), T{-719.5f});
    vector<T> after(inout.size()), readonly(replacement.size()), actual(output.size());
    for (auto offset : {64u, 65u}) {
        auto command = [&] {
            return shader(inout.view(offset, count), replacement.view(offset, count),
                          output.view(offset, 2u * count)).dispatch();
        };
        auto list = CommandList::create();
        list << command();
        auto graph = extension->create_graph(std::move(list.commit()).command_list());
        LUISA_ASSERT(graph.handle().valid(), "Snapshot graph creation failed.");
        auto exec = extension->instantiate(graph.handle().handle);
        LUISA_ASSERT(exec.handle().valid(), "Snapshot graph instantiation failed.");
        for (auto generation = 0u; generation < 2u; generation++) {
            std::fill(before.begin(), before.end(), T{-719.5f});
            std::fill(next.begin(), next.end(), T{-719.5f});
            for (auto i = size_t{0u}; i < count; i++) {
                before[offset + i] = value<T>(i, generation);
                next[offset + i] = value<T>(i, generation + 3u);
            }
            stream << inout.copy_from(span{before}) << replacement.copy_from(span{next})
                   << output.copy_from(span{blank}) << synchronize();
            if (generation == 0u) { stream << command(); }
            else { extension->launch(exec.handle().handle, stream.handle()); }
            stream << inout.copy_to(span{after}) << replacement.copy_to(span{readonly})
                   << output.copy_to(span{actual}) << synchronize();
            for (auto i = size_t{0u}; i < after.size(); i++) {
                auto expected = before[i];
                if (i >= offset && i < offset + count) { expected = next[i]; }
                expect(bits(after[i]) == bits(expected));
                expect(bits(readonly[i]) == bits(next[i]));
            }
            for (auto i = size_t{0u}; i < actual.size(); i++) {
                auto expected = blank[i];
                if (i >= offset && i < offset + count) { expected = before[i]; }
                else if (i >= offset + count && i < offset + 2u * count) { expected = next[i - count]; }
                expect(bits(actual[i]) == bits(expected)) << "snapshot word=" << i;
            }
        }
    }
}

template<typename T>
[[nodiscard]] T shifted_alias_value(size_t index, size_t generation) {
    // Do not reuse value(): its period is eight, which would hide an incorrect
    // streaming copy into a view shifted by exactly eight storage elements.
    constexpr std::array<uint16_t, 8u> half_words{0u, 0x8000u, 0x7c00u, 0xfc00u, 1u, 0x8001u, 0x7bffu, 0xfbffu};
    constexpr std::array<uint16_t, 8u> bfloat_words{0u, 0x8000u, 0x7f80u, 0xff80u, 1u, 0x8001u, 0x7f7fu, 0xff7fu};
    if (index < 8u) {
        if constexpr (std::is_same_v<T, half>) { return std::bit_cast<T>(half_words[(index + generation) % 8u]); }
        else { return std::bit_cast<T>(bfloat_words[(index + generation) % 8u]); }
    }
    auto word = static_cast<uint16_t>((std::is_same_v<T, half> ? 0x2000u : 0x3e00u) +
                                     (index * 37u + generation * 101u) % 0x0200u);
    if ((index + generation) % 2u != 0u) { word |= 0x8000u; }
    return std::bit_cast<T>(word);
}

template<typename T>
void shifted_alias_snapshot(Device &device) {
    using namespace tile;
    constexpr auto count = size_t{64u}, total = size_t{208u};
    auto kernel = tile_kernel("shifted_alias_snapshot", [](TensorView<const T, 1> input,
                                                          TensorView<T, 1> output) {
        // Exactly one program: there is no cross-program memory race. The
        // immutable load is a complete snapshot before the overlapping store.
        for (auto &program : parallel(shape(1))) {
            static_cast<void>(program);
            auto domain = shape(axis("element", 64));
            auto snapshot = input.tile(coord(0), domain).load();
            output.tile(coord(0), domain).store(snapshot);
        }
    }).capture(tensor_shape(count), tensor_shape(count));
    auto original = compile(device, kernel, nullptr);
    auto candidate = compile(device, kernel, "1");
    check_source(original, candidate, 3u);
    if (!original || !candidate) { return; }
    expect(!original.metadata().disjoint_writes && !candidate.metadata().disjoint_writes)
        << "alignment does not grant disjoint input/output arguments";
    if (original.metadata().disjoint_writes || candidate.metadata().disjoint_writes) { return; }
    auto first_view = candidate.metadata().source.find("ct::partition_view");
    expect(first_view != string::npos) << "candidate did not realize a structured load";
    if (first_view == string::npos) { return; }
    expect(candidate.metadata().source.find("ct::partition_view", first_view + 1u) == string::npos);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto *extension = device.extension<CudaGraphExt>();
    LUISA_ASSERT(extension != nullptr, "CUDA graph extension is required.");
    auto buffer = device.create_buffer<T>(total);
    auto base = reinterpret_cast<uintptr_t>(buffer.native_handle());
    expect((base & 15u) == 0u);
    using Offsets = std::array<size_t, 2u>;
    // First two layouts select the aligned entry with overlapping ranges in
    // both directions. Last two exercise the unaligned original-entry fallback.
    constexpr std::array layouts{Offsets{64u, 72u}, Offsets{72u, 64u},
                                 Offsets{65u, 73u}, Offsets{73u, 65u}};
    vector<T> before(total), expected(total), actual(total), original_result(total);
    for (auto offsets : layouts) {
        for (auto generation = 0u; generation < 2u; generation++) {
            std::fill(before.begin(), before.end(), T{-719.5f});
            for (auto i = size_t{0u}; i < count; i++) { before[offsets[0] + i] = shifted_alias_value<T>(i, generation); }
            expected = before;
            for (auto i = size_t{0u}; i < count; i++) { expected[offsets[1] + i] = before[offsets[0] + i]; }
            auto aligned = offsets[0] % 8u == 0u;
            expect(((base + offsets[0] * sizeof(T)) & 15u) == (aligned ? 0u : 2u));
            expect(((base + offsets[1] * sizeof(T)) & 15u) == (aligned ? 0u : 2u));
            for (auto shader : std::array{&original, &candidate}) {
                auto command = [&] {
                    return (*shader)(buffer.view(offsets[0], count), buffer.view(offsets[1], count)).dispatch();
                };
                stream << buffer.copy_from(span{before}) << command()
                       << buffer.copy_to(span{actual}) << synchronize();
                for (auto i = size_t{0u}; i < total; i++) {
                    expect(bits(actual[i]) == bits(expected[i])) << "alias direct word=" << i;
                }
                if (shader == &original) { original_result = actual; }
                else {
                    for (auto i = size_t{0u}; i < total; i++) { expect(bits(actual[i]) == bits(original_result[i])); }
                }
                auto list = CommandList::create();
                list << command();
                auto graph = extension->create_graph(std::move(list.commit()).command_list());
                LUISA_ASSERT(graph.handle().valid(), "Shifted alias graph creation failed.");
                auto exec = extension->instantiate(graph.handle().handle);
                LUISA_ASSERT(exec.handle().valid(), "Shifted alias graph instantiation failed.");
                stream << buffer.copy_from(span{before}) << synchronize();
                extension->launch(exec.handle().handle, stream.handle());
                stream << buffer.copy_to(span{actual}) << synchronize();
                for (auto i = size_t{0u}; i < total; i++) {
                    expect(bits(actual[i]) == bits(expected[i])) << "alias graph word=" << i;
                    expect(bits(actual[i]) == bits(original_result[i]));
                }
            }
        }
    }
}


template<typename T>
void masked_partition_copy(Device &device, int64_t rows, int64_t columns) {
    using namespace tile;
    auto row_chunks = (rows + 7) / 8, column_chunks = (columns + 127) / 128;
    auto padded_rows = row_chunks * 8, padded_columns = column_chunks * 128;
    auto input_count = static_cast<size_t>(rows * columns);
    auto output_count = static_cast<size_t>(padded_rows * padded_columns);
    auto kernel = tile_kernel("masked_partition_storage", [=](TensorView<const T, 2> input,
                                                               TensorView<T, 2> output) {
        auto pr = axis("program_row", row_chunks), pc = axis("program_column", column_chunks);
        auto r = axis("row", 8), c = axis("column", 128);
        for (auto &program : parallel(shape(pr, pc))) {
            auto origin = coord(program.index(pr) * int64_t{8}, program.index(pc) * int64_t{128});
            auto snapshot = input.tile(origin, shape(r, c)).load();
            output.tile(origin, shape(r, c)).store(snapshot);
        }
    }).capture(tensor_shape(rows, columns), tensor_shape(padded_rows, padded_columns));
    expect(kernel.valid());
    if (!kernel.valid()) { return; }
    auto original = compile(device, kernel, nullptr);
    auto candidate = compile(device, kernel, "1");
    check_source(original, candidate, 3u);
    if (!original || !candidate) { return; }
    expect(candidate.metadata().source.find("_partition.load_masked(ct::view_padding_zero_t{}, ") != string::npos);
    // Wrong padding and a wrong chunk divisor must not disappear in source
    // normalization, even when all other source text is a genuine candidate.
    for (auto mutation : {0u, 1u}) {
        auto invalid = candidate.metadata().source.substr(original.metadata().source.size());
        auto token = mutation == 0u ? string_view{"view_padding_zero_t"} : string_view{") / 128ll"};
        auto location = invalid.find(token);
        expect(location != string::npos);
        if (location == string::npos) { return; }
        invalid.replace(location, token.size(), mutation == 0u ? "view_padding_negative_zero_t" : ") / 64ll");
        expect(!restore_partition_loads(invalid, original.metadata().source, 3u));
    }
    auto input = device.create_buffer<T>(input_count + 130u);
    auto output = device.create_buffer<T>(output_count + 130u);
    vector<T> before(input.size()), readonly(input.size()), blank(output.size(), T{-719.5f}), actual(output.size());
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto *extension = device.extension<CudaGraphExt>();
    LUISA_ASSERT(extension != nullptr, "CUDA graph extension is required.");
    auto input_base = reinterpret_cast<uintptr_t>(input.native_handle());
    auto output_base = reinterpret_cast<uintptr_t>(output.native_handle());
    for (auto offset : {size_t{64u}, size_t{65u}}) {
        expect(((input_base + offset * sizeof(T)) & 15u) == (offset == 64u ? 0u : 2u));
        expect(((output_base + offset * sizeof(T)) & 15u) == (offset == 64u ? 0u : 2u));
        for (auto generation = 0u; generation < 2u; generation++) {
            std::fill(before.begin(), before.end(), T{-719.5f});
            for (auto i = size_t{0u}; i < input_count; i++) { before[offset + i] = shifted_alias_value<T>(i, generation); }
            for (auto shader : std::array{&original, &candidate}) {
                auto command = [&] { return (*shader)(input.view(offset, input_count), output.view(offset, output_count)).dispatch(); };
                auto list = CommandList::create();
                list << command();
                auto graph = extension->create_graph(std::move(list.commit()).command_list());
                LUISA_ASSERT(graph.handle().valid(), "Masked copy graph creation failed.");
                auto exec = extension->instantiate(graph.handle().handle);
                LUISA_ASSERT(exec.handle().valid(), "Masked copy graph instantiation failed.");
                for (auto use_graph : {false, true}) {
                    stream << input.copy_from(span{before}) << output.copy_from(span{blank}) << synchronize();
                    if (use_graph) { extension->launch(exec.handle().handle, stream.handle()); }
                    else { stream << command(); }
                    stream << input.copy_to(span{readonly}) << output.copy_to(span{actual}) << synchronize();
                    for (auto i = size_t{0u}; i < before.size(); i++) { expect(bits(readonly[i]) == bits(before[i])); }
                    for (auto i = size_t{0u}; i < actual.size(); i++) {
                        auto expected = blank[i];
                        if (i >= offset && i < offset + output_count) {
                            auto logical = static_cast<int64_t>(i - offset);
                            auto row = logical / padded_columns, column = logical % padded_columns;
                            expected = row < rows && column < columns ? before[offset + static_cast<size_t>(row * columns + column)] : T{0.0f};
                        }
                        expect(bits(actual[i]) == bits(expected)) << "partial copy word=" << i;
                    }
                }
            }
        }
    }
}

template<typename T>
void masked_partition_alias_snapshots(Device &device) {
    using namespace tile;
    // One program reads a partial 56-of-64 snapshot, stores through a distinct
    // alias shifted by eight elements, and reads the same partial snapshot
    // again. Both roots are aligned; alignment establishes no no-alias fact.
    auto kernel = tile_kernel("masked_alias_two_snapshots", [](TensorView<const T, 1> input,
                                                               TensorView<T, 1> writer,
                                                               TensorView<const T, 1> replacement,
                                                               TensorView<T, 1> output) {
        for (auto &program : parallel(shape(1))) {
            static_cast<void>(program);
            auto domain = shape(axis("element", 64));
            auto before = input.tile(coord(0), domain).load();
            auto next = replacement.tile(coord(0), domain).load();
            writer.tile(coord(0), domain).store(next);
            auto after = input.tile(coord(0), domain).load();
            output.tile(coord(0), domain).store(before);
            output.tile(coord(64), domain).store(after);
        }
    }).capture(tensor_shape(56), tensor_shape(64), tensor_shape(64), tensor_shape(128));
    expect(kernel.valid());
    if (!kernel.valid()) { return; }
    auto original = compile(device, kernel, nullptr);
    auto candidate = compile(device, kernel, "1");
    check_source(original, candidate, 15u);
    if (!original || !candidate) { return; }
    expect(!original.metadata().disjoint_writes && !candidate.metadata().disjoint_writes);
    auto shared = device.create_buffer<T>(208u);
    auto replacement = device.create_buffer<T>(194u);
    auto output = device.create_buffer<T>(258u);
    vector<T> before(shared.size()), expected(shared.size()), after(shared.size());
    vector<T> next(replacement.size()), readonly(replacement.size());
    vector<T> blank(output.size(), T{-719.5f}), actual(output.size());
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto *extension = device.extension<CudaGraphExt>();
    LUISA_ASSERT(extension != nullptr, "CUDA graph extension is required.");
    using Offsets = std::array<size_t, 2u>;
    constexpr std::array layouts{Offsets{64u, 72u}, Offsets{72u, 64u}, Offsets{65u, 73u}, Offsets{73u, 65u}};
    for (auto offsets : layouts) {
        auto offset = offsets[0] % 8u == 0u ? size_t{64u} : size_t{65u};
        auto shared_base = reinterpret_cast<uintptr_t>(shared.native_handle());
        for (auto origin : offsets) { expect(((shared_base + origin * sizeof(T)) & 15u) == (offset == 64u ? 0u : 2u)); }
        for (auto generation = 0u; generation < 2u; generation++) {
            std::fill(before.begin(), before.end(), T{-719.5f});
            std::fill(next.begin(), next.end(), T{-719.5f});
            for (auto i = size_t{0u}; i < 56u; i++) { before[offsets[0] + i] = shifted_alias_value<T>(i, generation); }
            for (auto i = size_t{0u}; i < 64u; i++) { next[offset + i] = shifted_alias_value<T>(i, generation + 3u); }
            expected = before;
            for (auto i = size_t{0u}; i < 64u; i++) { expected[offsets[1] + i] = next[offset + i]; }
            for (auto shader : std::array{&original, &candidate}) {
                auto command = [&] {
                    return (*shader)(shared.view(offsets[0], 56u), shared.view(offsets[1], 64u),
                                     replacement.view(offset, 64u), output.view(offset, 128u)).dispatch();
                };
                auto list = CommandList::create();
                list << command();
                auto graph = extension->create_graph(std::move(list.commit()).command_list());
                LUISA_ASSERT(graph.handle().valid(), "Masked alias graph creation failed.");
                auto exec = extension->instantiate(graph.handle().handle);
                LUISA_ASSERT(exec.handle().valid(), "Masked alias graph instantiation failed.");
                for (auto use_graph : {false, true}) {
                    stream << shared.copy_from(span{before}) << replacement.copy_from(span{next})
                           << output.copy_from(span{blank}) << synchronize();
                    if (use_graph) { extension->launch(exec.handle().handle, stream.handle()); }
                    else { stream << command(); }
                    stream << shared.copy_to(span{after}) << replacement.copy_to(span{readonly})
                           << output.copy_to(span{actual}) << synchronize();
                    for (auto i = size_t{0u}; i < after.size(); i++) { expect(bits(after[i]) == bits(expected[i])); }
                    for (auto i = size_t{0u}; i < next.size(); i++) { expect(bits(readonly[i]) == bits(next[i])); }
                    for (auto i = size_t{0u}; i < actual.size(); i++) {
                        auto wanted = blank[i];
                        if (i >= offset && i < offset + 128u) {
                            auto local = i - offset;
                            if (local % 64u >= 56u) { wanted = T{0.0f}; }
                            else { wanted = local < 64u ? before[offsets[0] + local] : expected[offsets[0] + local - 64u]; }
                        }
                        expect(bits(actual[i]) == bits(wanted)) << "masked alias snapshot word=" << i;
                    }
                }
            }
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
        masked_partition_copy<half>(device, 257, 128);
        masked_partition_copy<tile::bfloat16>(device, 257, 128);
        masked_partition_copy<half>(device, 9, 136);
        masked_partition_copy<tile::bfloat16>(device, 9, 136);
        rank_copy<half, 1u>(device, 64, 3u);
        rank_copy<half, 1u>(device, 33, 0u);
        rank_copy<tile::bfloat16, 3u>(device, 64, 3u);
        rank_copy<tile::bfloat16, 3u>(device, 33, 0u);
    };
    "tile_cuda_alignment_per_root_proof"_test = [&] {
        per_root_proof(device);
        snapshot_order<half>(device);
        snapshot_order<tile::bfloat16>(device);
        shifted_alias_snapshot<half>(device);
        shifted_alias_snapshot<tile::bfloat16>(device);
        masked_partition_alias_snapshots<half>(device);
        masked_partition_alias_snapshots<tile::bfloat16>(device);
    };
    return 0;
}
