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
using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    luisa::log_level_verbose();
    Context context{argv[0]};
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
    auto b0 = device.create_buffer<float>(8);
    auto b1 = device.create_buffer<float>(8);
    auto shader0 = device.compile<1>(
        [&](BufferFloat b0, BufferInt b1) {
            auto r = b0.read(0);
        });

    Kernel1D k = [&](BufferFloat b0, BufferFloat b1) {
        auto r = b0.read(0);
    };

    auto shader1 = device.compile<1>(
        [&](BufferFloat b1) {
            b1.write(0, 0.0f);
        });
    using namespace graph;

    std::cout << std::endl << "graph build info>>>" << std::endl;
    KernelNode *node0;
    GraphDef gd = [&](GraphBuffer<float> b0, GraphBuffer<int> b1) {
        node0 = shader0(b0, b1).dispatch(1);
        shader1(b0).dispatch(1);
    };
    auto &vars = gd._builder->_vars;
    for (auto &v : vars) {
        std::cout << "graph var " << v->arg_id() << ":[tag = " << (int)v->tag() << "]\n";
    }
    auto& kernels = gd._builder->_kernel_nodes;

    for (int kid = 0; auto &k : kernels) {
        auto &args = k->arg_set();
        std::cout << "kernel" << kid << "'s args:[";
        for (auto argid : args) std::cout << argid << " ";
        std::cout << "]\n";
        ++kid;
    }
    std::cout << std::endl << "graph build info<<<" << std::endl;
    //GraphNode *node0, *node1, *node2;

    //auto graph = [&](BufferView<float> b0, BufferView<float> b1) {
    //    shader0.operator()(b0, b1);
    //    node0 = Graph::add_node(shader0(b0, b1), 1);
    //    node1 = Graph::add_node(b0.copy_from(b1));
    //    node2 = Graph::add_node(shader1(b1), 1);
    //};

    //graph(b0, b1);
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