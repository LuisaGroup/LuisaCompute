

#include <luisa/luisa-compute.h>
#include <luisa/dsl/work_graph/work_graph.h>
#include <luisa/dsl/work_graph/work_graph_kernel.h>

using namespace luisa::compute;

struct ConsumerRecord {
    uint2 datum;
};

LUISA_STRUCT(ConsumerRecord, datum) {};

int main() {

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

    WorkGraph wg = work_graph.build();

    return 0;
}