#include <luisa/luisa-compute.h>
#include <luisa/dsl/work_graph/work_graph.h>
#include <luisa/dsl/work_graph/work_graph_kernel.h>
#include <luisa/backends/ext/work_graph_ext.h>
#include <luisa/runtime/work_graph/work_graph_program.h>
#include "Windows.h"
// see note in DX backend `Device.cpp`
extern "C" __declspec(dllexport) const uint32_t D3D12SDKVersion = 619;
extern "C" __declspec(dllexport) LPCSTR D3D12SDKPath = ".\\D3D12\\";

using namespace luisa::compute;

struct ConsumerRecord {
    uint index;
    uint data;
};

LUISA_STRUCT(ConsumerRecord, index, data) {};

WorkGraph basic_work_graph(const Buffer<uint>& out) {
    WorkGraphBuilder work_graph { "basic-work-graph" };

    auto producer = work_graph.add_node<WorkGraphLaunchType::BROADCASTING, WorkGraphEmptyRecord>("producer");
    producer.set_threadgroup_size({64, 1, 1});
    producer.set_dispatch_size({2, 1, 1}); // 2 groups * 64 threads = 128 total

    auto producer_output = producer.output<ConsumerRecord>(1);
    WorkGraphNodeKernel producer_kernel = [&]() {
        Var<ConsumerRecord> out;
        UInt index = dispatch_x();
        out.index = index;
        out.data = index;
        producer_output.write(out, true);
    };
    producer.define(producer_kernel);

    auto consumer = work_graph.add_node<WorkGraphLaunchType::THREAD, ConsumerRecord>("consumer");
    WorkGraphNodeKernel consumer_kernel = [&](Var<ConsumerRecord> input) {
        // do work
        out->write(input.index, input.data);
    };
    consumer.define(consumer_kernel);

    consumer << producer_output;

    return work_graph.build();
}

void basic_work_graph_test(Device& device, Stream& stream) {
    auto d_buffer = device.create_buffer<uint>(128);
    luisa::vector<uint> h_buffer;
    h_buffer.resize(128, 0);

    WorkGraph basic_wg = basic_work_graph(d_buffer);
    WorkGraphProgram basic_wg_program = device.compile(basic_wg);

    stream << basic_wg_program().dispatch(1, 0, nullptr) << synchronize();
    stream << d_buffer.copy_to(h_buffer.data()) << synchronize();

    for (size_t i = 0; i < h_buffer.size(); i += 1) {
        LUISA_ASSERT(h_buffer[i] == i, "expected {}, got {}", i, h_buffer[i]);
    }
}

struct ProducerRecord : DispatchGridRecord {};
LUISA_STRUCT(ProducerRecord, size) {};

WorkGraph dynamic_dispatch_grid(const Buffer<uint>& out) {
    WorkGraphBuilder work_graph { "basic-work-graph" };

    auto producer = work_graph.add_node<WorkGraphLaunchType::BROADCASTING, ProducerRecord>("producer");
    producer.set_threadgroup_size({64, 1, 1});
    producer.set_max_dispatch_size({128, 1, 1});

    auto producer_output = producer.output<ConsumerRecord>(1);
    WorkGraphNodeKernel producer_kernel = [&](Var<ProducerRecord>) {
        Var<ConsumerRecord> out;
        UInt index = dispatch_x();
        out.index = index;
        out.data = index;
        producer_output.write(out, true);
    };
    producer.define(producer_kernel);

    auto consumer = work_graph.add_node<WorkGraphLaunchType::THREAD, ConsumerRecord>("consumer");
    WorkGraphNodeKernel consumer_kernel = [&](Var<ConsumerRecord> input) {
        // do work
        out->write(input.index, input.data);
    };
    consumer.define(consumer_kernel);

    consumer << producer_output;

    return work_graph.build();
}

void dynamic_dispatch_grid_test(Device& device, Stream& stream) {
    // max_dispatch_size is {128, 1, 1} with 64 threads/group → 8192 max elements
    constexpr uint max_groups = 128u;
    constexpr uint threads_per_group = 64u;
    constexpr uint buffer_size = max_groups * threads_per_group;

    auto d_buffer = device.create_buffer<uint>(buffer_size);
    luisa::vector<uint> zeros(buffer_size, 0u);
    luisa::vector<uint> h_buffer(buffer_size);

    WorkGraph wg = dynamic_dispatch_grid(d_buffer);
    WorkGraphProgram program = device.compile(wg);

    auto run = [&](uint groups) {
        stream << d_buffer.copy_from(zeros.data()) << synchronize();
        ProducerRecord record{};
        record.size = uint3(groups, 1u, 1u);
        stream << program().dispatch(1, sizeof(ProducerRecord), &record) << synchronize();
        stream << d_buffer.copy_to(h_buffer.data()) << synchronize();
        const uint count = groups * threads_per_group;
        for (uint i = 0; i < count; i++)
            LUISA_ASSERT(h_buffer[i] == i, "groups={}: expected {} at [{}], got {}", groups, i, i, h_buffer[i]);
        for (uint i = count; i < buffer_size; i++)
            LUISA_ASSERT(h_buffer[i] == 0u, "groups={}: unexpected write at [{}] = {}", groups, i, h_buffer[i]);
        LUISA_INFO("dynamic_dispatch_grid: {} groups ({} threads) passed", groups, count);
    };

    run(1u);
    run(4u);
    run(16u);
    run(max_groups);
}

WorkGraph node_array(const Buffer<uint>& out) {
    LUISA_ASSERT(false, "unimplemented");
}

void node_array_test(Device& device, Stream& stream) {
    LUISA_ASSERT(false, "unimplemented");
}

int main(int argc, char **argv) {
    Context ctx { argv[0] };
    Device device = ctx.create_device("dx", nullptr, true);
    Stream stream = device.create_stream(StreamTag::COMPUTE);

    basic_work_graph_test(device, stream);
    dynamic_dispatch_grid_test(device, stream);

    return 0;
}