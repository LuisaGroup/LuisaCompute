// Direct-pointer Tile kernels must retain their ABI, views, and dependencies
// when replayed through the public CUDA graph extension.
#include "ut/ut.hpp"
#include "test_device.h"

#include <array>
#include <algorithm>
#include <luisa/core/logging.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/command_list.h>
#include <luisa/backends/ext/cuda/cuda_graph_ext.h>
#include <luisa/tile/dsl.h>
#include <luisa/tile/runtime.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {
constexpr auto count = size_t{131u};
constexpr auto pad = size_t{7u};
constexpr auto sentinel = -8765.25f;

void check(luisa::span<const float> values, float offset, size_t base) {
    for (auto i = size_t{0u}; i < values.size(); i++) {
        auto expected = i >= base && i < base + count ? static_cast<float>(i - base) * .25f + offset : sentinel;
        expect(values[i] == expected) << "index=" << i << ", actual=" << values[i] << ", expected=" << expected;
    }
}
}// namespace

int main(int argc, char *argv[]) {
    if (argc != 3 || string_view{argv[1]} != "cuda" ||
        (string_view{argv[2]} != "native" && string_view{argv[2]} != "tirx")) {
        LUISA_WARNING("Usage: test_tile_cuda_graph cuda native|tirx (native requires LUISA_CUDA_TILE_IR=1)");
        return 1;
    }
    Context context{argv[0]};
    auto device = context.create_device("cuda");
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto *extension = device.extension<CudaGraphExt>();
    LUISA_ASSERT(extension != nullptr, "CUDA graph extension is required.");
    auto native = string_view{argv[2]} == "native";
    using namespace tile;
    auto kernel = tile_kernel("graph_pointer_permutation", [](TensorView<float, 1> output,
                                                              TensorView<const float, 1> unused,
                                                              TensorView<const float, 1> input) {
        static_cast<void>(unused);
        auto x = axis("x", 64);
        for (auto &block : parallel(shape(3))) {
            auto origin = block.index() * 64;
            auto value = input.tile(coord(origin), shape(x)).load();
            output.tile(coord(origin), shape(x)).store(value + 1.0f);
        }
    }).capture(tensor_shape(count), tensor_shape(count), tensor_shape(count));
    auto shader = tile::compile(device, kernel,
                                {.lowering = native ? Lowering::NATIVE : Lowering::TIRX},
                                {.enable_fast_math = false});
    LUISA_ASSERT(shader, "Requested Tile route is unavailable: {}", shader.metadata().error);
    LUISA_INFO("Graph shader realization: {}", shader.metadata().realization);
    if (native) { LUISA_ASSERT(shader.metadata().realization.find("NVRTC Tile IR") != string::npos, "Wrong native route."); }
    auto total = count + 4u * pad;
    auto source = device.create_buffer<float>(total);
    auto middle = device.create_buffer<float>(total);
    auto output = device.create_buffer<float>(total);
    auto unused = device.create_buffer<float>(count);
    vector<float> input(total, sentinel), blank(total, sentinel), actual(total), middle_actual(total);
    for (auto i = size_t{0u}; i < count; i++) { input[pad + i] = static_cast<float>(i) * .25f; }
    stream << source.copy_from(luisa::span{input}) << middle.copy_from(luisa::span{blank})
           << output.copy_from(luisa::span{blank}) << synchronize();

    auto commands = [&](size_t offset) {
        auto list = CommandList::create();
        list << shader(middle.view(offset, count), unused, source.view(pad, count)).dispatch();
        list << shader(output.view(offset, count), unused, middle.view(offset, count)).dispatch();
        list << shader(middle.view(offset, count), unused, output.view(offset, count)).dispatch();
        return std::move(list.commit()).command_list();
    };
    auto graph = extension->create_graph(commands(pad));
    LUISA_ASSERT(graph.handle().handle != CudaGraphExt::invalid_handle, "Tile graph creation failed.");
    auto exec = extension->instantiate(graph.handle().handle);
    auto second = extension->instantiate(graph.handle().handle);
    LUISA_ASSERT(exec.handle().handle != CudaGraphExt::invalid_handle && second.handle().handle != CudaGraphExt::invalid_handle,
                 "Tile graph instantiation failed.");

    "direct_buffer_tile_graph_replay"_test = [&] {
        for (auto repeat = 0u; repeat < 4u; repeat++) {
            extension->launch(exec.handle().handle, stream.handle());
            stream << output.copy_to(luisa::span{actual}) << middle.copy_to(luisa::span{middle_actual}) << synchronize();
            check(actual, 2.0f, pad);
            check(middle_actual, 3.0f, pad);
        }
        // The legacy raw node-update API has no Tile argument mapping. It
        // must refuse rather than install the ordinary DSL parameter layout.
        expect(!extension->update_kernel_node(exec.handle().handle, 0u, make_uint3(3u, 1u, 1u), {}));
    };
    "direct_buffer_tile_graph_whole_update"_test = [&] {
        auto updated = extension->update(exec.handle().handle, commands(2u * pad));
        expect(updated);
        if (!updated) { return; }
        stream << output.copy_from(luisa::span{blank}) << middle.copy_from(luisa::span{blank}) << synchronize();
        extension->launch(exec.handle().handle, stream.handle());
        stream << output.copy_to(luisa::span{actual}) << middle.copy_to(luisa::span{middle_actual}) << synchronize();
        check(actual, 2.0f, 2u * pad);
        check(middle_actual, 3.0f, 2u * pad);
        expect(!extension->update_kernel_node(exec.handle().handle, 1u, make_uint3(3u, 1u, 1u), {}));
        // Updating one executable must not mutate the other executable's
        // pointer list or the graph template.
        stream << output.copy_from(luisa::span{blank}) << middle.copy_from(luisa::span{blank}) << synchronize();
        extension->launch(second.handle().handle, stream.handle());
        stream << output.copy_to(luisa::span{actual}) << middle.copy_to(luisa::span{middle_actual}) << synchronize();
        check(actual, 2.0f, pad);
        check(middle_actual, 3.0f, pad);
    };
    "direct_buffer_tile_graph_rejects_bad_view"_test = [&] {
        ComputeDispatchCmdEncoder encoder{shader.handle(), 3u, 0u};
        encoder.encode_buffer(output.handle(), output.size_bytes() - sizeof(float), count * sizeof(float));
        encoder.encode_buffer(unused.handle(), 0u, unused.size_bytes());
        encoder.encode_buffer(source.handle(), pad * sizeof(float), count * sizeof(float));
        encoder.set_dispatch_size(shader.metadata().dispatch_size);
        auto list = CommandList::create();
        list << std::move(encoder).build();
        auto rejected = extension->create_graph(std::move(list.commit()).command_list());
        expect(rejected.handle().handle == CudaGraphExt::invalid_handle);
    };
    stream << source.copy_to(luisa::span{actual}) << synchronize();
    expect(std::equal(actual.begin(), actual.end(), input.begin())) << "read-only source changed";
    return 0;
}
