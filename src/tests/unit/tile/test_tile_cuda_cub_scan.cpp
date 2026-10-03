// Manual positive native test. Availability never substitutes for actual entry
// observation or full numerical/alias verification.
#include "ut/ut.hpp"
#include "test_device.h"
#include <cuda.h>
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>
#include <luisa/core/platform.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/stream.h>
#include <luisa/backends/ext/cuda/cuda_graph_ext.h>
#include <luisa/tile/algorithms.h>
#include <luisa/tile/runtime.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {
struct Environment {
    const char *name;
    luisa::optional<luisa::string> previous;
    static void set(const char *name, const char *value) {
#ifdef _WIN32
        auto status = _putenv_s(name, value == nullptr ? "" : value);
#else
        auto status = value == nullptr ? unsetenv(name) : setenv(name, value, 1);
#endif
        LUISA_ASSERT(status == 0, "Could not set CUB scan test environment.");
    }
    Environment(const char *n, const char *v) : name{n}, previous{get_environment_variable(n)} { set(n, v); }
    ~Environment() { set(name, previous ? previous->c_str() : nullptr); }
};

[[nodiscard]] bool cost_mode() {
    auto value = get_environment_variable("LUISA_CUDA_TILE_CUB_SCAN_COST");
    return value && *value == "1";
}

[[nodiscard]] uint32_t selected_threads(const tile::Shader &shader) {
    auto text = string_view{shader.metadata().realization};
    constexpr string_view marker = "; cub-scan-threads=";
    auto position = text.find(marker);
    LUISA_ASSERT(position != string_view::npos, "Missing CUB recipe metadata.");
    auto value = text.substr(position + marker.size());
    value = value.substr(0u, value.find(';'));
    uint32_t threads{};
    auto parsed = std::from_chars(value.data(), value.data() + value.size(), threads);
    LUISA_ASSERT(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() &&
                     (threads == 0u || threads == 128u || threads == 256u || threads == 512u || threads == 1024u),
                 "Invalid CUB recipe metadata.");
    return threads;
}

template<typename T>
[[nodiscard]] auto bits(T value) {
    using Word = std::conditional_t<sizeof(T) == 4u, uint32_t, uint16_t>;
    return std::bit_cast<Word>(value);
}

// This reads the real graph template CUfunction. CUDA has no API to inspect an
// executable graph's updated parameters; update tests therefore also use the
// snapshot alias witness below and keep the old executable valid on rejection.
void check_graph_entry(Device &device, const CudaGraphInstance &graph, bool candidate, uint32_t programs, uint32_t threads) {
    LUISA_ASSERT(graph.handle().valid(), "CUB scan graph creation failed.");
    auto context = static_cast<CUcontext>(device.native_handle());
    auto pushed = cuCtxPushCurrent(context);
    expect(pushed == CUDA_SUCCESS);
    if (pushed != CUDA_SUCCESS) { return; }
    struct Pop {
        ~Pop() {
            CUcontext previous{};
            expect(cuCtxPopCurrent(&previous) == CUDA_SUCCESS);
        }
    } pop;
    auto native = static_cast<CUgraph>(graph.handle().native_handle);
    size_t count{};
    auto result = cuGraphGetNodes(native, nullptr, &count);
    expect(result == CUDA_SUCCESS);
    if (result != CUDA_SUCCESS) { return; }
    std::vector<CUgraphNode> nodes(count);
    result = cuGraphGetNodes(native, nodes.data(), &count);
    expect(result == CUDA_SUCCESS);
    if (result != CUDA_SUCCESS) { return; }
    auto kernels = 0u;
    for (auto node : nodes) {
        CUgraphNodeType type{};
        auto typed = cuGraphNodeGetType(node, &type);
        expect(typed == CUDA_SUCCESS);
        if (typed != CUDA_SUCCESS) { continue; }
        if (type != CU_GRAPH_NODE_TYPE_KERNEL) { continue; }
        kernels++;
        CUDA_KERNEL_NODE_PARAMS params{};
        auto inspected = cuGraphKernelNodeGetParams(node, &params);
        expect(inspected == CUDA_SUCCESS);
        if (inspected != CUDA_SUCCESS) { continue; }
        expect(params.blockDimX == (candidate ? threads : 1u) && params.blockDimY == 1u && params.blockDimZ == 1u);
        expect(params.gridDimX == programs && params.gridDimY == 1u && params.gridDimZ == 1u);
        const char *name{};
        auto named = cuFuncGetName(&name, params.func);
        expect(named == CUDA_SUCCESS);
        if (named != CUDA_SUCCESS || name == nullptr) { continue; }
        auto expected = candidate ? "luisa_tile_cub_scan" : "luisa_tile_main";
        expect(std::strcmp(name, expected) == 0) << "actual graph entry=" << name;
    }
    expect(kernels == 1u);
}

template<typename T>
[[nodiscard]] tile::Kernel capture(int64_t rows, int64_t columns, int64_t br) {
    using namespace tile;
    auto padded = static_cast<int64_t>(std::bit_ceil(static_cast<uint64_t>(columns)));
    return tile_kernel("closed_prefix_value", [=](TensorView<const T, 2> input, TensorView<T, 2> output) {
               auto r = axis("r", br), c = axis("c", padded);
               for (auto &p : parallel(shape((rows + br - 1) / br))) {
                   auto row = p.index() * br;
                   auto loaded = cast<float>(input.tile(coord(row, 0), shape(r, c)).load());
                   output(coord(row, 0), shape(r, c)).store(cast<T>(inclusive_sum(loaded, c, reduction::unordered_tree)));
               }
           })
        .capture(tensor_shape(rows, columns), tensor_shape(rows, columns));
}

[[nodiscard]] tile::Shader compile(Device &device, const tile::Kernel &kernel, uint32_t threads, bool available = true) {
    auto original = [&] {
        Environment off{"LUISA_CUDA_TILE_CUB_SCAN", nullptr};
        Environment cost_off{"LUISA_CUDA_TILE_CUB_SCAN_COST", nullptr};
        return tile::compile(device, kernel, {.lowering = tile::Lowering::NATIVE}, {.enable_fast_math = false});
    }();
    auto shader = tile::compile(device, kernel, {.lowering = tile::Lowering::NATIVE}, {.enable_fast_math = false});
    LUISA_ASSERT(original && shader, "CUB scan compile failed: {}", shader.metadata().error);
    auto &&metadata = shader.metadata();
    if (cost_mode()) {
        expect(metadata.realization.find("cub-scan-cost-requested=1") != string::npos);
        expect(metadata.realization.find("cub-scan-cost-fit=67127e819f80a395aeecae55cd999c8e3f8b1bcf894fdf0672c58611cca87f66") != string::npos);
        expect(metadata.realization.find(luisa::format("cub-scan-cost-selected-threads={}", selected_threads(shader))) != string::npos);
        expect((selected_threads(shader) != 0u) == available);
    } else {
        expect(selected_threads(shader) == threads);
    }
    LUISA_ASSERT((metadata.realization.find("cub-scan-available;") != string::npos) == available,
                 "Unexpected CUB scan availability: {}", metadata.realization);
    expect(metadata.source == original.metadata().source);
    expect(original.metadata().realization.find("cub-scan-requested") == string::npos);
    expect(original.metadata().realization.find("cub-scan-cost-requested") == string::npos);
    expect(metadata.source.find("void luisa_tile_main(") != string::npos);
    expect(metadata.source.find("luisa_tile_cub_scan") == string::npos);
    return shader;
}

template<typename T>
void disjoint_values(Device &device, uint32_t threads, size_t input_offset, size_t output_offset, bool tail = false) {
    constexpr int64_t rows = 3;
    auto br = tail ? int64_t{2} : int64_t{1};
    // Legality fallbacks are independent of the CUB thread recipe. Keep them
    // in the original compiler's supported envelope; a 16384-wide FP32 scan
    // currently fails in tileiras before an optional candidate is attempted.
    auto columns = (tail || std::is_same_v<T, float>) ? int64_t{4096} : static_cast<int64_t>(threads) * 16;
    columns -= tail ? 1 : 0;
    auto n = static_cast<size_t>(rows * columns), total = n + 130u;
    auto available = !tail && !std::is_same_v<T, float>;
    auto shader = compile(device, capture<T>(rows, columns, br), threads, available);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto *ext = device.extension<CudaGraphExt>();
    LUISA_ASSERT(ext != nullptr, "CUDA graph extension required.");
    auto input = device.create_buffer<T>(total), output = device.create_buffer<T>(total);
    auto sentinel = T{-719.5f};
    vector<T> before(total, sentinel), blank(total, sentinel), actual(total), readonly(total);
    auto command = [&] { return shader(input.view(input_offset, n), output.view(output_offset, n)).dispatch(); };
    auto commands = [&] { CommandList list; list << command(); return list; };
    auto graph = ext->create_graph(commands());
    check_graph_entry(device, graph, available && input_offset % 8u == 0u && output_offset % 8u == 0u,
                      static_cast<uint32_t>((rows + br - 1) / br), selected_threads(shader));
    auto executable = ext->instantiate(graph.handle().handle);
    LUISA_ASSERT(executable.handle().valid(), "CUB scan graph instantiation failed.");
    for (auto generation = 0u; generation < 3u; generation++) {
        for (auto i = size_t{0u}; i < n; i++) {
            auto x = static_cast<float>(static_cast<int>((i * 17u + generation * 7u) % 31u) - 15) * .0625f;
            if (generation == 1u) { x = (i % 2u == 0u ? 1.0f : -1.0f) * .125f; }
            if (generation == 2u) { x = (i % 4u == 0u ? -0.0f : static_cast<float>(std::numeric_limits<T>::denorm_min())); }
            before[input_offset + i] = T{x};
        }
        auto upload = [&] { stream << input.copy_from(span{before}) << output.copy_from(span{blank}) << synchronize(); };
        auto check = [&] {
            stream << input.copy_to(span{readonly}) << output.copy_to(span{actual}) << synchronize();
            for (auto i = size_t{0u}; i < total; i++) {
                expect(bits(readonly[i]) == bits(before[i]));
                if (i < output_offset || i >= output_offset + n) { expect(bits(actual[i]) == bits(sentinel)); }
            }
            for (auto row = size_t{0u}; row < static_cast<size_t>(rows); row++) {
                double reference = 0.0, absolute = 0.0;
                for (auto col = size_t{0u}; col < static_cast<size_t>(columns); col++) {
                    auto i = row * columns + col;
                    auto x = static_cast<double>(static_cast<float>(before[input_offset + i]));
                    reference += x;
                    absolute += std::abs(x);
                    // Exact existing workload full-prefix bound; no new margin.
                    auto count = static_cast<double>(col + 1u), q = 2.0 * count + 2.0, u = 0x1p-24;
                    auto bound = (q * u / (1.0 - q * u) + 8.0 * count * 0x1p-53) * absolute + q * 0x1p-149;
                    if constexpr (!std::is_same_v<T, float>) {
                        auto unit = static_cast<double>(static_cast<float>(std::numeric_limits<T>::epsilon())) * .5;
                        auto eta = static_cast<double>(static_cast<float>(std::numeric_limits<T>::denorm_min())) * .5;
                        bound += unit * (std::abs(reference) + bound) + eta;
                    }
                    auto value = static_cast<double>(static_cast<float>(actual[output_offset + i]));
                    expect(std::isfinite(value));
                    expect(std::abs(value - reference) <= bound) << "row=" << row << ", col=" << col;
                    if (generation == 2u) {
                        // These small positive subnormal prefixes are exactly
                        // accumulated in FP32 before the one storage rounding.
                        expect(bits(actual[output_offset + i]) == bits(T{static_cast<float>(reference)}));
                    }
                }
            }
        };
        upload();
        stream << command() << synchronize();
        check();
        upload();
        ext->launch(executable.handle().handle, stream.handle());
        check();
    }
}

template<typename T>
void snapshot_alias_and_update(Device &device, uint32_t threads) {
    auto n = static_cast<size_t>(threads) * 16u;
    auto total = 2u * n + 256u;
    auto shader = compile(device, capture<T>(1, static_cast<int64_t>(n), 1), threads);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto *ext = device.extension<CudaGraphExt>();
    LUISA_ASSERT(ext != nullptr, "CUDA graph extension required.");
    auto buffer = device.create_buffer<T>(total);
    vector<T> before(total, T{-719.5f}), expected(total), actual(total);
    struct Layout {
        size_t input;
        size_t output;
        bool candidate;
    };
    const std::array layouts{Layout{64u, n + 128u, true}, Layout{65u, n + 129u, false},
                             Layout{64u, 128u, false}, Layout{128u, 64u, false}, Layout{64u, 64u, false}};
    auto command = [&](size_t index) {
        auto l = layouts[index];
        return shader(buffer.view(l.input, n), buffer.view(l.output, n)).dispatch();
    };
    auto commands = [&](size_t index) { CommandList list; list << command(index); return list; };
    auto upload = [&](size_t index, size_t generation) {
        auto l = layouts[index];
        std::fill(before.begin(), before.end(), T{-719.5f});
        for (auto i = size_t{0u}; i < n; i++) {
            before[l.input + i] = T{static_cast<float>(static_cast<int>((i * 17u + generation) % 31u) - 15) * .0625f};
        }
        expected = before;
        float sum = 0.0f;
        for (auto i = size_t{0u}; i < n; i++) {
            sum += static_cast<float>(before[l.input + i]);
            expected[l.output + i] = T{sum};
        }
        stream << buffer.copy_from(span{before}) << synchronize();
    };
    auto check = [&] {
        stream << buffer.copy_to(span{actual}) << synchronize();
        for (auto i = size_t{0u}; i < total; i++) { expect(bits(actual[i]) == bits(expected[i])) << "snapshot word=" << i; }
    };
    for (auto index = size_t{0u}; index < layouts.size(); index++) {
        upload(index, 0u);
        stream << command(index) << synchronize();
        check();
        auto graph = ext->create_graph(commands(index));
        check_graph_entry(device, graph, layouts[index].candidate, 1u, selected_threads(shader));
        auto executable = ext->instantiate(graph.handle().handle);
        LUISA_ASSERT(executable.handle().valid(), "Alias graph instantiation failed.");
        upload(index, 1u);
        ext->launch(executable.handle().handle, stream.handle());
        check();
    }
    auto graph = ext->create_graph(commands(0u));
    check_graph_entry(device, graph, true, 1u, selected_threads(shader));
    auto executable = ext->instantiate(graph.handle().handle);
    LUISA_ASSERT(executable.handle().valid(), "Update graph instantiation failed.");
    auto current = size_t{0u};
    auto accepted_updates = 0u, rejected_updates = 0u;
    for (auto index : {1u, 0u, 2u, 3u, 4u, 0u}) {
        auto requested = ext->create_graph(commands(index));
        check_graph_entry(device, requested, layouts[index].candidate, 1u, selected_threads(shader));
        if (ext->update(executable.handle().handle, commands(index))) {
            current = index;
            accepted_updates++;
        } else {
            rejected_updates++;
        }
        // Both modules stay shader-owned and alive. A rejected Driver update
        // must leave the old function, geometry and parameter bindings usable.
        upload(current, 3u);
        ext->launch(executable.handle().handle, stream.handle());
        check();
    }
    // A different kernel-node count cannot update the existing topology.
    // Verify the rejected update leaves the old executable and its bindings
    // usable, even when normal original/candidate function switches succeeded.
    CommandList changed_topology;
    changed_topology << command(0u) << command(0u);
    auto changed = ext->update(executable.handle().handle, std::move(changed_topology));
    expect(!changed);
    if (!changed) {
        rejected_updates++;
        upload(current, 4u);
        ext->launch(executable.handle().handle, stream.handle());
        check();
    }
    LUISA_INFO("CUB graph update attempts: accepted={}, rejected={} (old exec verified after each rejection).",
               accepted_updates, rejected_updates);
}
// Frozen-policy geometry witnesses. These expected recipes come from the
// frozen independent profile evaluation, not a call into the tested scorer.
// Nonuniform exact dyadic inputs expose stale width/grid/cache descriptors.
template<typename T>
void cost_geometry(Device &device, int64_t rows, int64_t columns, uint32_t expected_threads) {
    auto shader = compile(device, capture<T>(rows, columns, 1), expected_threads);
    expect(selected_threads(shader) == expected_threads);
    constexpr size_t pad = 64u;
    auto n = static_cast<size_t>(rows * columns), total = n + 2u * pad;
    auto sentinel = T{-719.5f};
    vector<T> before(total, sentinel), blank(total, sentinel), expected(total, sentinel), actual(total), readonly(total);
    for (auto row = int64_t{0}; row < rows; row++) {
        auto prefix = int64_t{0};
        for (auto column = int64_t{0}; column < columns; column++) {
            auto units = (row * 13 + column * 17) % 31 - 15;
            auto offset = pad + static_cast<size_t>(row * columns + column);
            before[offset] = T{static_cast<float>(units) * 0.0625f};
            prefix += units;
            expected[offset] = T{static_cast<float>(prefix) * 0.0625f};
        }
    }
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto input = device.create_buffer<T>(total), output = device.create_buffer<T>(total);
    auto command = [&] { return shader(input.view(pad, n), output.view(pad, n)).dispatch(); };
    auto upload = [&] { stream << input.copy_from(span{before}) << output.copy_from(span{blank}) << synchronize(); };
    auto check = [&] {
        stream << input.copy_to(span{readonly}) << output.copy_to(span{actual}) << synchronize();
        for (auto i = size_t{0u}; i < total; i++) {
            expect(bits(readonly[i]) == bits(before[i]));
            expect(bits(actual[i]) == bits(expected[i])) << "cost geometry word=" << i;
        }
    };
    upload();
    stream << command() << synchronize();
    check();
    auto *ext = device.extension<CudaGraphExt>();
    LUISA_ASSERT(ext != nullptr, "CUDA graph extension required.");
    CommandList commands;
    commands << command();
    auto graph = ext->create_graph(std::move(commands));
    check_graph_entry(device, graph, true, static_cast<uint32_t>(rows), expected_threads);
    auto executable = ext->instantiate(graph.handle().handle);
    LUISA_ASSERT(executable.handle().valid(), "Cost geometry graph instantiation failed.");
    upload();
    ext->launch(executable.handle().handle, stream.handle());
    check();
}
}// namespace

int main(int argc, char *argv[]) {
    if (argc < 2) { return 2; }
    if (argv == nullptr) { return 2; }
    if (argv[0] == nullptr) { return 2; }
    if (argv[1] == nullptr) { return 2; }
    if (std::strcmp(argv[1], "cuda") != 0) { return 2; }
    auto opt_in = get_environment_variable("LUISA_CUDA_TILE_IR");
    if (!opt_in) { return 2; }
    if (std::strcmp(opt_in->c_str(), "1") != 0) { return 2; }
    auto option = get_environment_variable("LUISA_CUDA_TILE_CUB_SCAN");
    uint32_t threads{};
    if (cost_mode()) {
        if (option && *option != "0") { return 2; }
        // Existing small value/alias witnesses use width 4096. The real graph
        // block comes from the installed recipe, not this fixture size seed.
        threads = 256u;
    } else {
        if (!option) { return 2; }
        for (auto candidate : {128u, 256u, 512u, 1024u}) {
            if (*option == luisa::format("{}", candidate)) { threads = candidate; }
        }
    }
    if (threads == 0u) { return 2; }
    Environment streaming{"LUISA_CUDA_TILE_STREAMING_SCAN", nullptr};
    Environment cost{"LUISA_CUDA_TILE_COLLECTIVE_COST", "0"};
    Environment partition_cost{"LUISA_CUDA_TILE_PARTITION_COST", "0"};
    Environment worker{"LUISA_CUDA_TILE_WORKER_WARPS", "0"};
    Environment aligned{"LUISA_CUDA_TILE_IR_ALIGNED16", "0"};
    Environment pure{"LUISA_CUDA_TILE_SCAN_CHUNK", "0"};
    Environment independent{"LUISA_CUDA_TILE_INDEPENDENT_AXIS", "0"};
    Environment partition{"LUISA_CUDA_TILE_PROGRAM_ROWS", "0"};
    std::vector<const char *> args{argv[0]};
    for (auto i = 2; i < argc; i++) {
        if (argv[i] == nullptr) { return 2; }
        args.emplace_back(argv[i]);
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(static_cast<int>(args.size()), args.data());
    Context context{argv[0]};
    auto device = context.create_device("cuda");
    "tile_cuda_cub_half_direct_graph"_test = [&] {
        for (auto input : {64u, 65u}) {
            for (auto output : {64u, 65u}) { disjoint_values<half>(device, threads, input, output); }
        }
    };
    "tile_cuda_cub_bfloat_direct_graph"_test = [&] {
        for (auto input : {64u, 65u}) {
            for (auto output : {64u, 65u}) { disjoint_values<tile::bfloat16>(device, threads, input, output); }
        }
    };
    "tile_cuda_cub_snapshot_alias_update"_test = [&] {
        snapshot_alias_and_update<half>(device, threads);
        snapshot_alias_and_update<tile::bfloat16>(device, threads);
    };
    if (cost_mode()) {
        "tile_cuda_cub_cost_geometry"_test = [&] {
            for (auto geometry : {std::array<int64_t, 3u>{3, 2048, 256}, {65, 4096, 512},
                                  {129, 8192, 256}, {256, 16384, 128}}) {
                cost_geometry<half>(device, geometry[0u], geometry[1u], static_cast<uint32_t>(geometry[2u]));
                cost_geometry<tile::bfloat16>(device, geometry[0u], geometry[1u], static_cast<uint32_t>(geometry[2u]));
            }
        };
    }
    "tile_cuda_cub_unsupported_original"_test = [&] {
        disjoint_values<float>(device, threads, 64u, 64u);
        disjoint_values<half>(device, threads, 64u, 64u, true);
        disjoint_values<tile::bfloat16>(device, threads, 64u, 64u, true);
    };
}
