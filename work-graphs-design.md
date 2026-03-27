# Work Graphs
For now, we will only support `BROADCASTING` and `THREAD` launch modes, since they are much easier to reason about how the input records are accessed, and it is good enough for my own use case (rendering)

## User-facing API
```cpp
// 1. Record types are regular LUISA_STRUCT types
struct MyRecord { float3 position; uint data; };
LUISA_STRUCT(MyRecord, position, data) {};

struct LeafRecord { float value; };
LUISA_STRUCT(LeafRecord, value) {};

// 2. Define node kernels (new WorkGraphNodeKernel template)
// *requires* that input record is either empty, or that it is the first argument to lambda
WorkGraphNodeKernel<MyRecord> producer{
    [&](Var<MyRecord> input, BufferVar<float> buf) {
        // node body -- buf is a "shared" bound resource
    }
};

WorkGraphNodeKernel<LeafRecord> consumer{
    [&](Var<LeafRecord> input, BufferVar<float> result) {
        // leaf node body
    }
};

// 3. Build graph with explicit edges
WorkGraphBuilder builder;
auto n0 = builder.add_node("Producer", producer);
auto n1 = builder.add_node("Consumer", consumer);
builder.add_edge<LeafRecord>(n0, n1, {.max_records = 256});
builder.set_entrypoint(n0);
auto desc = builder.build(); // validates types

// 4. Compile & dispatch
auto wg = device.compile(desc, shader_option);
stream << wg(buf, result).dispatch(input_records) << synchronize();
```