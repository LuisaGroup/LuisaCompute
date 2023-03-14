//
// Created by Mike on 2022/4/6.
//

struct alignas(16) Property {
    unsigned int instance_id;
    unsigned int sbt_offset;
    unsigned int mask;
    unsigned int flags;
    unsigned long long traversable;
    unsigned long long pad;
};

struct alignas(16) Instance {
    float4 affine[3];
    Property property;
};

struct alignas(16) Modification {
    unsigned int index;
    unsigned int flags;
    unsigned long long primitive;
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

        constexpr auto update_flag_primitive = 1u << 0u;
        constexpr auto update_flag_transform = 1u << 1u;
        constexpr auto update_flag_opaque_on = 1u << 2u;
        constexpr auto update_flag_opaque_off = 1u << 3u;
        constexpr auto update_flag_visibility = 1u << 4u;
        constexpr auto update_flag_opaque = update_flag_opaque_on | update_flag_opaque_off;
        constexpr auto update_flag_vis_mask_offset = 24u;

        auto m = mods[tid];
        auto p = instances[m.index].property;
        p.instance_id = m.index;
        p.sbt_offset = 0u;
        p.flags = 0x5u;
        if (m.flags & update_flag_primitive) { p.traversable = m.primitive; }
        if (m.flags & update_flag_visibility) { p.mask = m.flags >> update_flag_vis_mask_offset; }
        instances[m.index].property = p;
        if (m.flags & update_flag_transform) {
            auto t = instances[m.index].affine;
            t[0] = m.affine[0];
            t[1] = m.affine[1];
            t[2] = m.affine[2];
        }
    }
}
