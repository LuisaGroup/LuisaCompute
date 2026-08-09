struct Element {
    uint buffer;
    uint texture2d;
    uint texture3d;
};

struct Source {
    uint index;
    Element element;
};

RWStructuredBuffer<Element> destination : register(u1);
StructuredBuffer<Source> source : register(t0);

struct PushConstants {
    uint count;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(256, 1, 1)]
void main(uint id : SV_DispatchThreadID) {
    if (id >= pc.count) { return; }
    Source value = source[id];
    destination[value.index] = value.element;
}
