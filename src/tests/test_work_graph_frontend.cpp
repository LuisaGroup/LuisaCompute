

#include <luisa/luisa-compute.h>
#include <luisa/dsl/work_graph/work_graph.h>
#include <luisa/dsl/work_graph/work_graph_kernel.h>

int main() {
    using namespace luisa::compute;


    WorkGraphBuilder work_graph;

    auto* producer = work_graph.add_node<WorkGraphEmptyRecord>("producer");
    WorkGraphNodeKernel producer_kernel = []() {

    };

    producer->define(producer_kernel);


    return 0;
}