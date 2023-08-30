#include <numeric>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/dynamic_module.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include <luisa/gui/window.h>

#include <luisa/runtime/graph/graph.h>
#include <luisa/runtime/graph/graph_def.h>
#include <luisa/runtime/graph/graph_builder.h>
#include <luisa/runtime/graph/graph_var.h>
#include <luisa/runtime/graph/graph_dsl.h>

#include <luisa/backends/ext/graph_ext.h>
#include <luisa/runtime/graph/kernel_node_cmd_encoder.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    luisa::log_level_verbose();
    Context context{argv[0]};
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
    using namespace graph;

    std::cout << std::endl
              << "graph build info>>>" << std::endl;

    auto b0 = device.create_buffer<float>(8);
    auto b1 = device.create_buffer<int>(8);
    luisa::vector<float> h_b0(8);

    auto shader0 = device.compile<1>(
        [](BufferFloat b0, BufferInt b1) {
            b0.write(0, 1.0f);
            b0.write(1, 2.0f);
            auto b = b1.read(0);
        });
    auto shader1 = device.compile<1>(
        [](BufferFloat b0) {
            auto b = b0.read(0);
        });
    auto shader2 = device.compile<1>(
        [](BufferFloat b0) {
            auto b = b0.read(0);
        });

    GraphDef gd = [&](GraphBuffer<float> b0, GraphBuffer<int> b1, GraphVar<uint32_t> dispatch_size) {
        shader0(b0, b1).dispatch(dispatch_size);
        shader1(b0).dispatch(dispatch_size);
        shader2(b0).dispatch(dispatch_size);
    };

    auto graph_ext = device.extension<GraphExt>();
    auto g = graph_ext->create_graph(gd);
    stream << g(b0, b1, 1).dispatch();
    auto b0_new = device.create_buffer<float>(8);
    stream << g(b0_new, b1, 1).dispatch();
    stream << b0.copy_to(h_b0.data()) << synchronize();
    LUISA_INFO("b0[0] = {}, b0[1] = {}", h_b0[0], h_b0[1]);
    stream << b0_new.copy_to(h_b0.data()) << synchronize();
    LUISA_INFO("b0[0] = {}, b0[1] = {}", h_b0[0], h_b0[1]);

    auto &vars = gd._builder->_vars;
    for (auto &v : vars) {
        std::cout << "graph var " << v->arg_id() << ":[tag = " << (int)v->tag() << "]\n";
    }
    auto &kernels = gd._builder->_kernel_nodes;

    for (int kid = 0; auto &k : kernels) {
        auto args = k->kernel_args();
        std::cout << "kernel" << kid << "'s args:[";
        for (auto &&[arg_id, usage] : args) std::cout << "id=" << arg_id << " usage="
                                                      << (int)usage << "; ";
        std::cout << "] ";
        auto dispatch_args = k->dispatch_args();
        std::cout << "dispatch args:[";
        for (auto &&[arg_id, usage] : dispatch_args) std::cout << "id=" << arg_id << " usage="
                                                               << (int)usage << "; ";
        std::cout << "]\n";
        ++kid;
    }
    std::cout << "deps:" << std::endl;
    auto &deps = gd._builder->_deps;
    for (auto dep : deps) {
        std::cout << "[" << dep.src << "]->[" << dep.dst << "]\n";
    };

    std::cout << std::endl
              << "graph build info<<<" << std::endl;
}

void test_map() {
    static_assert(std::is_same_v<
                  luisa::compute::detail::prototype_to_creation_t<int>,
                  Int>);

    static_assert(std::is_same_v<
                  luisa::compute::detail::prototype_to_creation_t<int &>,
                  Int &>);

    static_assert(std::is_same_v<
                  luisa::compute::detail::prototype_to_creation_t<Buffer<int>>,
                  BufferInt>);

    static_assert(std::is_same_v<
                  luisa::compute::detail::definition_to_prototype_t<Int>,
                  int>);

    static_assert(std::is_same_v<
                  luisa::compute::detail::definition_to_prototype_t<Int &>,
                  int &>);

    static_assert(std::is_same_v<
                  luisa::compute::detail::definition_to_prototype_t<BufferInt>,
                  Buffer<int>>);

    static_assert(std::is_same_v<luisa::compute::detail::prototype_to_creation_tag_t<int>,
                                 luisa::compute::detail::ArgumentCreation>);
    static_assert(std::is_same_v<luisa::compute::detail::prototype_to_creation_tag_t<int &>,
                                 luisa::compute::detail::ReferenceArgumentCreation>);
}