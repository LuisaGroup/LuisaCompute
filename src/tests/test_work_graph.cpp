#include <luisa/luisa-compute.h>
#include <luisa/dsl/work_graph/work_graph.h>
#include <luisa/dsl/work_graph/work_graph_kernel.h>
#include <luisa/backends/ext/work_graph_ext.h>
#include <luisa/runtime/work_graph/work_graph_program.h>
#include <luisa/runtime/rtx/accel.h>
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
    WorkGraphBuilder work_graph { "dynamic-dispatch-work-graph" };

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
    constexpr uint array_size = 4u;
    WorkGraphBuilder work_graph { "node-array-work-graph" };

    // Producer: 1 group × 4 threads, each routes dispatch_x() to consumer[dispatch_x()]
    auto producer = work_graph.add_node<WorkGraphLaunchType::BROADCASTING, WorkGraphEmptyRecord>("producer");
    producer.set_threadgroup_size({array_size, 1, 1});
    producer.set_dispatch_size({1, 1, 1});

    auto consumer_output = producer.array_output<ConsumerRecord>(1);

    WorkGraphNodeKernel producer_kernel = [&]() {
        UInt i = dispatch_x();
        Var<ConsumerRecord> record;
        record.index = i;
        record.data = i;
        consumer_output.write(record, i, true);
    };
    producer.define(producer_kernel);

    // Consumer array: 4 independent THREAD nodes, each writes out[input.index] = input.data
    auto consumers = work_graph.add_node_array<WorkGraphLaunchType::THREAD, ConsumerRecord>("consumer", array_size);
    WorkGraphNodeKernel consumer_kernel = [&](Var<ConsumerRecord> input) {
        out->write(input.index, input.data);
    };
    for (uint i = 0u; i < array_size; i++) {
        consumers[i].define(consumer_kernel);
    }

    consumers << consumer_output;

    return work_graph.build();
}

void node_array_test(Device& device, Stream& stream) {
    constexpr uint array_size = 4u;
    auto d_buffer = device.create_buffer<uint>(array_size);
    luisa::vector<uint> zeros(array_size, 0u);
    luisa::vector<uint> h_buffer(array_size);

    WorkGraph wg = node_array(d_buffer);
    WorkGraphProgram program = device.compile(wg);

    stream << d_buffer.copy_from(zeros.data()) << synchronize();
    stream << program().dispatch(1, 0, nullptr) << synchronize();
    stream << d_buffer.copy_to(h_buffer.data()) << synchronize();

    for (uint i = 0u; i < array_size; i++)
        LUISA_ASSERT(h_buffer[i] == i, "consumer[{}]: expected {}, got {}", i, i, h_buffer[i]);
    LUISA_INFO("node_array: passed");
}

// -----------------------------------------------------------------------
// Test 4: Bindless array as a bound resource in a work graph node
// -----------------------------------------------------------------------

struct BindlessRecord {
    uint slot;
};
LUISA_STRUCT(BindlessRecord, slot) {};

WorkGraph bindless_array_work_graph(const Buffer<uint> &out, const BindlessArray &heap) {
    WorkGraphBuilder work_graph{"bindless-work-graph"};

    // Producer: 1 group × 4 threads – each thread emits one record carrying its lane index
    auto producer = work_graph.add_node<WorkGraphLaunchType::BROADCASTING, WorkGraphEmptyRecord>("producer");
    producer.set_threadgroup_size({4, 1, 1});
    producer.set_dispatch_size({1, 1, 1});

    auto producer_output = producer.output<BindlessRecord>(1);
    WorkGraphNodeKernel producer_kernel = [&]() {
        Var<BindlessRecord> rec;
        rec.slot = dispatch_x();
        producer_output.write(rec, true);
    };
    producer.define(producer_kernel);

    // Consumer: reads the value stored in heap[slot] and writes it to the output buffer
    auto consumer = work_graph.add_node<WorkGraphLaunchType::THREAD, BindlessRecord>("consumer");
    WorkGraphNodeKernel consumer_kernel = [&](Var<BindlessRecord> input) {
        UInt slot = input.slot;
        UInt value = heap->buffer<uint>(slot).read(0u);
        out->write(slot, value);
    };
    consumer.define(consumer_kernel);

    consumer << producer_output;
    return work_graph.build();
}

void bindless_array_work_graph_test(Device &device, Stream &stream) {
    constexpr uint N = 4u;
    const uint values[N] = {111u, 222u, 333u, 444u};

    auto heap = device.create_bindless_array(64u);
    luisa::vector<Buffer<uint>> backing;
    backing.reserve(N);
    for (uint i = 0u; i < N; i++) {
        backing.emplace_back(device.create_buffer<uint>(1u));
        heap.emplace_on_update(i, backing[i]);
    }

    auto d_out = device.create_buffer<uint>(N);
    luisa::vector<uint> h_out(N, 0u);

    // Upload heap and backing buffer data
    stream << heap.update();
    for (uint i = 0u; i < N; i++)
        stream << backing[i].copy_from(&values[i]);
    stream << synchronize();

    WorkGraph wg = bindless_array_work_graph(d_out, heap);
    WorkGraphProgram program = device.compile(wg);

    stream << program().dispatch(1, 0, nullptr) << synchronize();
    stream << d_out.copy_to(h_out.data()) << synchronize();

    for (uint i = 0u; i < N; i++)
        LUISA_ASSERT(h_out[i] == values[i],
                     "slot {}: expected {}, got {}", i, values[i], h_out[i]);
    LUISA_INFO("bindless_array_work_graph: passed");
}

// -----------------------------------------------------------------------
// Test 5: Acceleration structure as a bound resource in a work graph node
// -----------------------------------------------------------------------

struct RayRecord {
    uint index;
};
LUISA_STRUCT(RayRecord, index) {};

WorkGraph accel_work_graph(const Buffer<uint> &out, const Accel &accel) {
    WorkGraphBuilder work_graph{"accel-work-graph"};

    // Producer: 1 group × 2 threads – each thread emits one RayRecord
    auto producer = work_graph.add_node<WorkGraphLaunchType::BROADCASTING, WorkGraphEmptyRecord>("producer");
    producer.set_threadgroup_size({2, 1, 1});
    producer.set_dispatch_size({1, 1, 1});

    auto producer_output = producer.output<RayRecord>(1);
    WorkGraphNodeKernel producer_kernel = [&]() {
        Var<RayRecord> rec;
        rec.index = dispatch_x();
        producer_output.write(rec, true);
    };
    producer.define(producer_kernel);

    // Consumer: traces a ray and writes 1 (hit) or 0 (miss) to the output buffer
    //   index 0 → ray hits the triangle
    //   index 1 → ray misses the triangle
    auto consumer = work_graph.add_node<WorkGraphLaunchType::THREAD, RayRecord>("consumer");
    WorkGraphNodeKernel consumer_kernel = [&](Var<RayRecord> input) {
        // Default origin is far from the triangle (miss)
        Float3 origin = def(make_float3(5.0f, 5.0f, 1.0f));
        $if (input.index == 0u) {
            // This ray should hit the triangle at (0,0,0)-(2,0,0)-(0,2,0)
            origin = make_float3(0.5f, 0.5f, 1.0f);
        };
        Var<Ray> ray = make_ray(origin, make_float3(0.0f, 0.0f, -1.0f));
        Bool hit = accel->intersect_any(ray, {});
        out->write(input.index, cast<uint>(hit));
    };
    consumer.define(consumer_kernel);

    consumer << producer_output;
    return work_graph.build();
}

void accel_work_graph_test(Device &device, Stream &stream) {
    // Triangle with vertices (0,0,0), (2,0,0), (0,2,0) in the z=0 plane
    const float3 vertices[3] = {
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(2.0f, 0.0f, 0.0f),
        make_float3(0.0f, 2.0f, 0.0f),
    };
    const Triangle triangles[1] = {Triangle{0u, 1u, 2u}};

    auto vertex_buf   = device.create_buffer<float3>(3u);
    auto triangle_buf = device.create_buffer<Triangle>(1u);
    stream << vertex_buf.copy_from(vertices)
           << triangle_buf.copy_from(triangles)
           << synchronize();

    auto mesh = device.create_mesh(vertex_buf, triangle_buf);
    auto accel = device.create_accel();
    accel.emplace_back(mesh, make_float4x4(1.f));
    stream << mesh.build() << accel.build() << synchronize();

    auto d_out = device.create_buffer<uint>(2u);
    luisa::vector<uint> h_out(2u, 0u);

    WorkGraph wg = accel_work_graph(d_out, accel);
    WorkGraphProgram program = device.compile(wg);

    stream << program().dispatch(1, 0, nullptr) << synchronize();
    stream << d_out.copy_to(h_out.data()) << synchronize();

    LUISA_ASSERT(h_out[0] == 1u, "ray 0 should hit, got {}", h_out[0]);
    LUISA_ASSERT(h_out[1] == 0u, "ray 1 should miss, got {}", h_out[1]);
    LUISA_INFO("accel_work_graph: passed");
}

int main(int argc, char **argv) {
    Context ctx { argv[0] };
    Device device = ctx.create_device("dx", nullptr, true);
    Stream stream = device.create_stream(StreamTag::COMPUTE);

    basic_work_graph_test(device, stream);
    dynamic_dispatch_grid_test(device, stream);
    node_array_test(device, stream);
    bindless_array_work_graph_test(device, stream);
    accel_work_graph_test(device, stream);

    return 0;
}