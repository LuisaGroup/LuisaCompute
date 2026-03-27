# Work Graphs
For now, we will only support `BROADCASTING` and `THREAD` launch modes, since they are much easier to reason about how the input records are accessed, and it is good enough for my own use case (rendering)

## User-facing API
```cpp
// 1. Record types are regular LUISA_STRUCT types
struct ProducerRecord { float3 position; uint data; };
LUISA_STRUCT(ProducerRecord, position, data) {};

struct ConsumerRecord { float value; };
LUISA_STRUCT(ConsumerRecord, value) {};

// 2. Start constructing work graph. this first step defines the "shape" of a given node (what its input and output records look like), and the assignment fills in what its implementation should look like. 
WorkGraphBuilder builder;

auto producer_node = builder.add_node<ProducerRecord>("Producer");
auto producer_output = producer_node.output<ConsumerRecord>(/*max_records=*/4);
WorkGraphKernelNode producer_kernel = [&](Var<ProducerRecord> input, BufferVar<float> buf) {
    Var<ConsumerRecord> r = def();
    producer_output.write(r, true);
};
producer_node.define(producer_kernel);

auto consumer_node = builder.add_node<ConsumerRecord>("Consumer");
WorkGraphKernelNode consumer_kernel = [&](Var<ConsumerRecord> input, BufferVar<float> buf) {
    // do work
}; 
consumer_node.define(consumer_kernel);

// 3. Create edges between work graph nodes
consumer_node << producer_output;

// 4. Finalize the work graph
auto desc = builder.build()

// 5. Compile & dispatch
auto wg = device.compile(desc, shader_option);
stream << wg(buf).dispatch(input_records) << synchronize();
```