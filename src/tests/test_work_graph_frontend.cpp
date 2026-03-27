#include <luisa/luisa-compute.h>
#include <luisa/dsl/work_graph/work_graph_node.h>

int main() {
    using namespace luisa::compute;


    struct EmptyRecord {};
    auto broadcasting_kernel = []() {

    };

    auto broadcasting_node = luisa::compute::make_work_graph_node<EmptyRecord>(WorkGraphLaunchType::BROADCASTING, broadcasting_kernel);

    return 0;
}