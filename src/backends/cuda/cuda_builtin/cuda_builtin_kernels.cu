// built-in update kernel for Accel

struct alignas(16) InstanceProperty {
    unsigned int user_id;
    unsigned int sbt_offset;
    unsigned int mask;
    unsigned int flags;
    unsigned long long traversable;
    unsigned long long pad;
};

struct alignas(16) Instance {
    float4 affine[3];
    InstanceProperty property;
};

struct alignas(16) InstanceModification {
    unsigned int index;
    unsigned int user_id;
    unsigned int flags;
    unsigned int vis_mask;
    float4 affine[3];
    unsigned long long primitive;
};

struct alignas(16) InstanceHandleMidification {
    unsigned long long index;
    unsigned long long handle;
};

enum InstanceFlags : unsigned int {
    INSTANCE_FLAG_NONE = 0u,
    INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING = 1u << 0u,
    INSTANCE_FLAG_FLIP_TRIANGLE_FACING = 1u << 1u,
    INSTANCE_FLAG_DISABLE_ANYHIT = 1u << 2u,
    INSTANCE_FLAG_ENFORCE_ANYHIT = 1u << 3u,
};

static_assert(sizeof(Instance) == 80, "");
static_assert(sizeof(InstanceModification) == 80, "");

extern "C" __global__ void update_accel_instance_handles(Instance *__restrict__ instances,
                                                         const InstanceHandleMidification *__restrict__ mods,
                                                         unsigned int n) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) [[likely]] {
        auto m = mods[tid];
        instances[m.index].property.traversable = m.handle;
    }
}

extern "C" __global__ void update_accel(Instance *__restrict__ instances,
                                        const InstanceModification *__restrict__ mods,
                                        unsigned int n) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) [[likely]] {

        constexpr auto update_flag_primitive = 1u << 0u;
        constexpr auto update_flag_transform = 1u << 1u;
        constexpr auto update_flag_opaque_on = 1u << 2u;
        constexpr auto update_flag_opaque_off = 1u << 3u;
        constexpr auto update_flag_visibility = 1u << 4u;
        constexpr auto update_flag_user_id = 1u << 5u;
        constexpr auto update_flag_procedural = 1u << 8u;
        constexpr auto update_flag_opaque = update_flag_opaque_on | update_flag_opaque_off;

        auto m = mods[tid];
        auto p = instances[m.index].property;
        p.sbt_offset = 0u;
        if (m.flags & update_flag_primitive) {
            p.traversable = m.primitive;
            p.flags = (m.flags & update_flag_procedural) ?
                          INSTANCE_FLAG_ENFORCE_ANYHIT :
                          INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
        }
        if (m.flags & update_flag_visibility) { p.mask = m.vis_mask; }
        if (m.flags & update_flag_opaque) {
            if (p.flags & INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING) {
                p.flags &= ~(INSTANCE_FLAG_DISABLE_ANYHIT |
                             INSTANCE_FLAG_ENFORCE_ANYHIT);
                p.flags |= (m.flags & update_flag_opaque_on) ?
                               INSTANCE_FLAG_DISABLE_ANYHIT :
                               INSTANCE_FLAG_ENFORCE_ANYHIT;
            }
        }
        if (m.flags & update_flag_user_id) {
            p.user_id = m.user_id;
        }
        instances[m.index].property = p;
        if (m.flags & update_flag_transform) {
            auto t = instances[m.index].affine;
            t[0] = m.affine[0];
            t[1] = m.affine[1];
            t[2] = m.affine[2];
        }
    }
}

// built-in update kernel for BindlessArray
struct alignas(16u) BindlessSlot {
    unsigned long long buffer;
    unsigned long long buffer_size;
    unsigned long long tex2d;
    unsigned long long tex3d;
};

static_assert(sizeof(BindlessSlot) == 32u, "");

struct alignas(16) SlotModification {
    struct Buffer {
        unsigned long long handle;
        unsigned long long size;
        unsigned int op;
    };
    struct Texture {
        unsigned long long handle;
        unsigned int sampler;// not used; processed on host
        unsigned int op;
    };
    unsigned long long slot;
    Buffer buffer;
    Texture tex2d;
    Texture tex3d;
};

static_assert(sizeof(SlotModification) == 64u, "");

extern "C" __global__ void update_bindless_array(BindlessSlot *__restrict__ array,
                                                 const SlotModification *__restrict__ mods,
                                                 unsigned int n) {
    constexpr auto op_none = 0u;
    constexpr auto op_update = 1u;
    constexpr auto op_remove = 2u;
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) [[likely]] {
        auto m = mods[tid];
        auto slot_id = m.slot;
        auto slot = array[slot_id];
        if (m.buffer.op == op_update) {
            slot.buffer = m.buffer.handle;
            slot.buffer_size = m.buffer.size;
        } else if (m.buffer.op == op_remove) {
            slot.buffer = 0u;
            slot.buffer_size = 0u;
        }
        if (m.tex2d.op == op_update) {
            slot.tex2d = m.tex2d.handle;
        } else if (m.tex2d.op == op_remove) {
            slot.tex2d = 0u;
        }
        if (m.tex3d.op == op_update) {
            slot.tex3d = m.tex3d.handle;
        } else if (m.tex3d.op == op_remove) {
            slot.tex3d = 0u;
        }
        array[slot_id] = slot;
    }
}
