//
// Created by Mike on 2022/4/6.
//

struct alignas(16) Property {
    unsigned int instance_id;
    unsigned int sbt_offset;
    unsigned int mask;
    unsigned int flags;
    uint2 traversable;
    uint2 pad;
};

struct alignas(16) Instance {
    float4 affine[3];
    Property property;
};

struct alignas(16) Request {
    unsigned int index;
    unsigned int flags;
    unsigned int visible;
    unsigned int padding;
    float4 affine[3];
};

static_assert(sizeof(Instance) == 80, "");
static_assert(sizeof(Request) == 64, "");

extern "C"
__global__ void initialize_instances(
    Instance *__restrict__ instances,
    const uint2 *__restrict__ gas,
    const Request *__restrict__ requests,
    unsigned int n) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) [[likely]] {
        constexpr auto update_flag_transform = 0x01u;
        constexpr auto update_flag_visibility = 0x02u;
        auto r = requests[tid];
        auto mask = instances[r.index].property.mask;
        __builtin_assume(mask <= 0xffu);
        if (r.flags & update_flag_visibility) { mask = r.visible ? 0xffu : 0x00u; }
        Property p{r.index, 0u, mask, 5u, gas[r.index]};
        instances[r.index].property = p;
        if (r.flags & update_flag_transform) {
            auto p = instances[r.index].affine;
            p[0] = r.affine[0];
            p[1] = r.affine[1];
            p[2] = r.affine[2];
        }
    }
}

extern "C"
__global__ void update_instances(
    Instance *__restrict__ instances,
    const Request *__restrict__ requests,
    unsigned int n) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) [[likely]] {
        constexpr auto update_flag_transform = 0x01u;
        constexpr auto update_flag_visibility = 0x02u;
        auto r = requests[tid];
        if (r.flags & update_flag_visibility) {
            instances[r.index].property.mask = r.visible ? 0xffu : 0x00u;
        }
        if (r.flags & update_flag_transform) {
            auto p = instances[r.index].affine;
            p[0] = r.affine[0];
            p[1] = r.affine[1];
            p[2] = r.affine[2];
        }
    }
}
