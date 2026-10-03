// Manual positive native test. Availability never substitutes for actual entry
// observation or full numerical/alias verification.
#include "ut/ut.hpp"
#include "test_device.h"
#include <cuda.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>
#include <luisa/core/platform.h>
#include <luisa/core/logging.h>
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
        LUISA_ASSERT(status == 0, "Could not set partition test environment.");
    }
    Environment(const char *n, const char *v) : name{n}, previous{get_environment_variable(n)} { set(n, v); }
    ~Environment() { set(name, previous ? previous->c_str() : nullptr); }
};

template<typename T>
[[nodiscard]] auto bits(T value) {
    using Word = std::conditional_t<sizeof(T) == 4u, uint32_t, uint16_t>;
    return std::bit_cast<Word>(value);
}

// This reads the real graph template CUfunction. CUDA has no API to inspect an
// executable graph's updated parameters; update tests therefore also use the
// snapshot alias witness below and keep the old executable valid on rejection.
void check_graph_entry(Device &device, const CudaGraphInstance &graph, bool partition, uint32_t programs) {
    LUISA_ASSERT(graph.handle().valid(), "Program partition graph creation failed.");
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
        expect(params.blockDimX == 1u && params.blockDimY == 1u && params.blockDimZ == 1u);
        expect(params.gridDimX == programs && params.gridDimY == 1u && params.gridDimZ == 1u);
        const char *name{};
        auto named = cuFuncGetName(&name, params.func);
        expect(named == CUDA_SUCCESS);
        if (named != CUDA_SUCCESS || name == nullptr) { continue; }
        auto expected = partition ? "luisa_tile_partition" : "luisa_tile_main";
        expect(std::strcmp(name, expected) == 0) << "actual graph entry=" << name;
    }
    expect(kernels == 1u);
}

template<typename T>
[[nodiscard]] tile::Kernel capture(int64_t rows, int64_t columns, bool maximum) {
    using namespace tile;
    constexpr int64_t br = 4;
    auto padded = static_cast<int64_t>(std::bit_ceil(static_cast<uint64_t>(columns)));
    return tile_kernel("unrelated_independent_collective", [=](TensorView<const T, 2> input, TensorView<T, 2> output) {
               auto r = axis("r", br), c = axis("c", padded), o = axis("o", 1);
               for (auto &p : parallel(shape((rows + br - 1) / br))) {
                   auto row = p.index() * br;
                   auto loaded = cast<float>(input.tile(coord(row, 0), shape(r, c)).load());
                   auto reduced = maximum ?
                       reduce(ite(iota(c) < columns, loaded, -std::numeric_limits<float>::infinity()), c, tile::maximum) :
                       reduce(loaded, c, add);
                   output(coord(row, 0), shape(r, o)).store(cast<T>(reduced));
               }
           })
        .capture(tensor_shape(rows, columns), tensor_shape(rows, 1));
}

[[nodiscard]] tile::Shader compile(Device &device, const tile::Kernel &kernel, uint32_t target, uint32_t rows) {
    auto shader = tile::compile(device, kernel, {.lowering = tile::Lowering::NATIVE}, {.enable_fast_math = false});
    LUISA_ASSERT(static_cast<bool>(shader), "Program partition compile failed: {}", shader.metadata().error);
    auto &&metadata = shader.metadata();
    expect(metadata.realization.find(luisa::format("program-partition-rows={}", target)) != string::npos);
    LUISA_ASSERT(metadata.realization.find("program-partition-available;") != string::npos,
                 "Required partition entry unavailable: {}", metadata.realization);
    expect(metadata.realization.find("program-partition-original-rows=4") != string::npos);
    expect(metadata.realization.find(luisa::format("program-partition-grid-x={}", (rows + target - 1u) / target)) != string::npos);
    expect(metadata.source.find("void luisa_tile_main(") != string::npos);
    expect(metadata.source.find("void luisa_tile_partition(") != string::npos);
    expect(metadata.dispatch_size.x == (rows + 3u) / 4u &&
           metadata.dispatch_size.y == 1u && metadata.dispatch_size.z == 1u);
    return shader;
}

template<typename T>
void disjoint_values(Device &device, uint32_t target) {
    constexpr size_t columns = 2051u, pad = 13u;
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto *ext = device.extension<CudaGraphExt>();
    LUISA_ASSERT(ext != nullptr, "CUDA graph extension required.");
    for (auto rows : {3u, 7u}) {
        auto n = static_cast<size_t>(rows) * columns;
        auto input_count = n + 2u * pad, output_count = rows + 2u * pad;
        for (auto maximum : {false, true}) {
            auto shader = compile(device, capture<T>(rows, columns, maximum), target, rows);
            auto input = device.create_buffer<T>(input_count), output = device.create_buffer<T>(output_count);
            auto sentinel = T{-719.5f};
            vector<T> before(input_count, sentinel), blank(output_count, sentinel), actual(output_count), readonly(input_count);
            auto command = [&] { return shader(input.view(pad, n), output.view(pad, rows)).dispatch(); };
            auto commands = [&] { CommandList list; list << command(); return list; };
            auto graph = ext->create_graph(commands());
            check_graph_entry(device, graph, true, (rows + target - 1u) / target);
            auto executable = ext->instantiate(graph.handle().handle);
            LUISA_ASSERT(executable.handle().valid(), "Partition graph instantiation failed.");
            for (auto generation = 0u; generation < 3u; generation++) {
                for (auto i = size_t{0u}; i < n; i++) {
                    // All-negative MAX input detects accidental zero column padding.
                    auto x = -static_cast<float>((i * 17u + generation * 7u) % 31u + 1u) * .0625f;
                    if (generation == 1u) { x = (i % 2u == 0u ? 1.0f : -1.0f) * .125f; }
                    if (generation == 2u) { x = (i % 4u == 0u ? -0.0f : static_cast<float>(std::numeric_limits<T>::denorm_min())); }
                    before[pad + i] = T{x};
                }
                auto upload = [&] { stream << input.copy_from(span{before}) << output.copy_from(span{blank}) << synchronize(); };
                auto check = [&] {
                    stream << input.copy_to(span{readonly}) << output.copy_to(span{actual}) << synchronize();
                    for (auto i = size_t{0u}; i < input_count; i++) { expect(bits(readonly[i]) == bits(before[i])); }
                    for (auto i = size_t{0u}; i < output_count; i++) {
                        if (i < pad || i >= pad + rows) { expect(bits(actual[i]) == bits(sentinel)); }
                    }
                    for (auto row = size_t{0u}; row < rows; row++) {
                        double sum = 0.0, absolute = 0.0, peak = -std::numeric_limits<double>::infinity();
                        for (auto col = size_t{0u}; col < columns; col++) {
                            auto x = static_cast<double>(static_cast<float>(before[pad + row * columns + col]));
                            sum += x;
                            absolute += std::abs(x);
                            peak = std::max(peak, x);
                        }
                        auto reference = maximum ? peak : sum;
                        auto count = static_cast<double>(columns), q = 2.0 * count + 2.0, u = 0x1p-24;
                        // The same existing full-contribution bound; no extra tolerance.
                        auto bound = maximum ? 0.0 : (q * u / (1.0 - q * u) + 8.0 * count * 0x1p-53) * absolute + q * 0x1p-149;
                        if constexpr (!std::is_same_v<T, float>) {
                            if (!maximum) {
                                auto unit = static_cast<double>(static_cast<float>(std::numeric_limits<T>::epsilon())) * .5;
                                auto eta = static_cast<double>(static_cast<float>(std::numeric_limits<T>::denorm_min())) * .5;
                                bound += unit * (std::abs(reference) + bound) + eta;
                            }
                        }
                        auto value = static_cast<double>(static_cast<float>(actual[pad + row]));
                        expect(std::isfinite(value));
                        expect(std::abs(value - reference) <= bound) << "row=" << row << ", max=" << maximum;
                        if (maximum || generation == 2u) {
                            expect(bits(actual[pad + row]) == bits(T{static_cast<float>(reference)}));
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
    }
}

template<typename T>
void snapshot_alias_and_update(Device &device, uint32_t target) {
    constexpr size_t rows = 4u, columns = 17u, n = rows * columns, pad = 13u, total = n + rows + 4u * pad;
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto *ext = device.extension<CudaGraphExt>();
    LUISA_ASSERT(ext != nullptr, "CUDA graph extension required.");
    for (auto maximum : {false, true}) {
        // One original program snapshots every input row before storing four
        // results. Output overlaps the last input row, so splitting into other
        // programs without the disjoint-view guard would introduce a race.
        auto shader = compile(device, capture<T>(rows, columns, maximum), target, rows);
        auto buffer = device.create_buffer<T>(total);
        vector<T> before(total, T{-719.5f}), expected(total), actual(total);
        auto output_offset = [](bool alias) { return alias ? pad + (rows - 1u) * columns : pad + n + pad; };
        auto command = [&](bool alias) { return shader(buffer.view(pad, n), buffer.view(output_offset(alias), rows)).dispatch(); };
        auto commands = [&](bool alias) { CommandList list; list << command(alias); return list; };
        auto upload = [&](bool alias, size_t generation) {
            std::fill(before.begin(), before.end(), T{-719.5f});
            for (auto i = size_t{0u}; i < n; i++) {
                auto row = i / columns;
                auto value = -static_cast<float>((row + 1u) * 8u + (i + generation) % 5u) * .0625f;
                before[pad + i] = T{value};
            }
            expected = before;
            for (auto row = size_t{0u}; row < rows; row++) {
                // All inputs are small exact binary fractions; all FP32 sums
                // are exact, followed by a single host storage RNE conversion.
                float sum = 0.0f, peak = -std::numeric_limits<float>::infinity();
                for (auto col = size_t{0u}; col < columns; col++) {
                    auto value = static_cast<float>(before[pad + row * columns + col]);
                    sum += value;
                    peak = std::max(peak, value);
                }
                expected[output_offset(alias) + row] = T{maximum ? peak : sum};
            }
            stream << buffer.copy_from(span{before}) << synchronize();
        };
        auto check = [&] {
            stream << buffer.copy_to(span{actual}) << synchronize();
            for (auto i = size_t{0u}; i < total; i++) { expect(bits(actual[i]) == bits(expected[i])) << "snapshot word=" << i; }
        };
        for (auto alias : {false, true}) {
            upload(alias, 0u);
            stream << command(alias) << synchronize();
            check();
            auto graph = ext->create_graph(commands(alias));
            check_graph_entry(device, graph, !alias, alias ? 1u : static_cast<uint32_t>((rows + target - 1u) / target));
            auto executable = ext->instantiate(graph.handle().handle);
            LUISA_ASSERT(executable.handle().valid(), "Alias partition graph instantiation failed.");
            upload(alias, 1u);
            ext->launch(executable.handle().handle, stream.handle());
            check();
        }
        auto graph = ext->create_graph(commands(false));
        check_graph_entry(device, graph, true, static_cast<uint32_t>((rows + target - 1u) / target));
        auto executable = ext->instantiate(graph.handle().handle);
        LUISA_ASSERT(executable.handle().valid(), "Update partition graph instantiation failed.");
        auto current_alias = false;
        auto accepted_updates = 0u, rejected_updates = 0u;
        for (auto alias : {true, false, true, false}) {
            auto requested = ext->create_graph(commands(alias));
            check_graph_entry(device, requested, !alias, alias ? 1u : static_cast<uint32_t>((rows + target - 1u) / target));
            auto updated = ext->update(executable.handle().handle, commands(alias));
            if (updated) {
                current_alias = alias;
                accepted_updates++;
            } else {
                rejected_updates++;
            }
            upload(current_alias, 3u);
            ext->launch(executable.handle().handle, stream.handle());
            check();
        }
        LUISA_INFO("Partition graph update attempts: accepted={}, rejected={}, maximum={} (old exec verified after each rejection).",
                   accepted_updates, rejected_updates, maximum);
    }
}
}// namespace

int main(int argc, char *argv[]) {
    if (argc < 2 || argv == nullptr || argv[0] == nullptr || argv[1] == nullptr) { return 2; }
    if (std::strcmp(argv[1], "cuda") != 0) { return 2; }
    auto opt_in = get_environment_variable("LUISA_CUDA_TILE_IR");
    if (!opt_in || std::strcmp(opt_in->c_str(), "1") != 0) { return 2; }
    auto option = get_environment_variable("LUISA_CUDA_TILE_PROGRAM_ROWS");
    if (!option) { return 2; }
    uint32_t target{};
    if (std::strcmp(option->c_str(), "1") == 0) {
        target = 1u;
    } else if (std::strcmp(option->c_str(), "2") == 0) {
        target = 2u;
    } else {
        return 2;
    }
    Environment cost{"LUISA_CUDA_TILE_COLLECTIVE_COST", "0"};
    Environment partition_cost{"LUISA_CUDA_TILE_PARTITION_COST", "0"};
    Environment worker{"LUISA_CUDA_TILE_WORKER_WARPS", "0"};
    Environment aligned{"LUISA_CUDA_TILE_IR_ALIGNED16", "0"};
    Environment pure{"LUISA_CUDA_TILE_SCAN_CHUNK", "0"};
    Environment independent{"LUISA_CUDA_TILE_INDEPENDENT_AXIS", "0"};
    Environment streaming{"LUISA_CUDA_TILE_STREAMING_SCAN", "0"};
    std::vector<const char *> args{argv[0]};
    for (auto i = 2; i < argc; i++) {
        if (argv[i] == nullptr) { return 2; }
        args.emplace_back(argv[i]);
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(static_cast<int>(args.size()), args.data());
    Context context{argv[0]};
    auto device = context.create_device("cuda");
    "tile_cuda_partition_float_direct_graph"_test = [&] { disjoint_values<float>(device, target); };
    "tile_cuda_partition_half_direct_graph"_test = [&] { disjoint_values<half>(device, target); };
    "tile_cuda_partition_bfloat_direct_graph"_test = [&] { disjoint_values<tile::bfloat16>(device, target); };
    "tile_cuda_partition_float_alias_update"_test = [&] { snapshot_alias_and_update<float>(device, target); };
    "tile_cuda_partition_half_alias_update"_test = [&] { snapshot_alias_and_update<half>(device, target); };
    "tile_cuda_partition_bfloat_alias_update"_test = [&] { snapshot_alias_and_update<tile::bfloat16>(device, target); };
}
