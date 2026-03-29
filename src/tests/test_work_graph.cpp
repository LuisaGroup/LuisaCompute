#include <luisa/luisa-compute.h>
#include <luisa/dsl/work_graph/work_graph.h>
#include <luisa/dsl/work_graph/work_graph_kernel.h>
#include <luisa/backends/ext/work_graph_ext.h>
#include <luisa/runtime/work_graph/work_graph_program.h>
#include "Windows.h"
// see note in DX backend `Device.cpp`
extern "C" __declspec(dllexport) const uint32_t D3D12SDKVersion = 616;
extern "C" __declspec(dllexport) LPCSTR D3D12SDKPath = ".\\D3D12\\";

using namespace luisa::compute;

struct ConsumerRecord {
    uint2 datum;
};

LUISA_STRUCT(ConsumerRecord, datum) {};

WorkGraph describe_work_graph() {
    WorkGraphBuilder work_graph;

    auto producer = work_graph.add_node<WorkGraphEmptyRecord>("producer");
    auto producer_output = producer.output<ConsumerRecord>(16);
    WorkGraphNodeKernel producer_kernel = [&]() {
        Var<ConsumerRecord> out;
        producer_output.write(out, true);
    };
    producer.define(producer_kernel);


    auto consumer = work_graph.add_node<ConsumerRecord>("consumer");
    WorkGraphNodeKernel consumer_kernel = [&](Var<ConsumerRecord> input) {
        // do work
    };
    consumer.define(consumer_kernel);

    consumer << producer_output;

    return work_graph.build();
}

int main(int argc, char **argv) {
    Context ctx { argv[0] };
    Device device = ctx.create_device("dx");

    WorkGraph wg = describe_work_graph();

    WorkGraphProgram wg_program = device.compile(wg);

    return 0;
}