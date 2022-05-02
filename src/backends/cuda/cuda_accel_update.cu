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

struct alignas(16) Modification {
    unsigned int index;
    unsigned int flags;
    uint2 mesh;
    float4 affine[3];
};

static_assert(sizeof(Instance) == 80, "");
static_assert(sizeof(Modification) == 64, "");

extern "C"
__global__ void update_instances(
    Instance *__restrict__ instances,
    const Modification *__restrict__ mods,
    unsigned int n) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) [[likely]] {
        constexpr auto update_flag_mesh = 1u << 0u;
        constexpr auto update_flag_transform = 1u << 1u;
        constexpr auto update_flag_visibility_on = 1u << 2u;
        constexpr auto update_flag_visibility_off = 1u << 3u;
        constexpr auto update_flag_visibility =
            update_flag_visibility_on | update_flag_visibility_off;
        auto m = mods[tid];
        auto p = instances[m.index].property;
        p.instance_id = m.index;
        p.sbt_offset = 0u;
        p.flags = 0x5u;
        if (m.flags & update_flag_mesh) { p.traversable = m.mesh; }
        if (m.flags & update_flag_visibility) { p.mask = (m.flags & update_flag_visibility_on) ? 0xffu : 0x00u; }
        instances[m.index].property = p;
        if (m.flags & update_flag_transform) {
            auto t = instances[m.index].affine;
            t[0] = m.affine[0];
            t[1] = m.affine[1];
            t[2] = m.affine[2];
        }
    }
}

extern "C"
__global__ void __launch_bounds__(32u)
wait_value(const unsigned int *values, unsigned int expected, unsigned int worker_masks) {
    auto i = threadIdx.x;
    if (((worker_masks >> i) & 1u)) {
        while (__ldcv(values + i) != expected) {
            // idle
        }
    }
}
