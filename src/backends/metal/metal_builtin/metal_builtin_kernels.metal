#include <metal_stdlib>

using namespace metal;

struct alignas(16) AccelInstance {
    float transform[12];// column-major packed
    uint options;
    uint mask;
    uint intersection_function_offset;
    uint mesh_index;
};

struct alignas(16) AccelInstanceModification {
    uint index;
    uint user_id;
    uint flags;
    uint vis_mask;
    float4 affine[3];// row-major
    ulong primitive;
};

static_assert(sizeof(AccelInstance) == 64u, "");
static_assert(sizeof(AccelInstanceModification) == 80u, "");

[[kernel]] void update_accel_instances(device AccelInstance *__restrict__ instances,
                                       device const AccelInstanceModification *__restrict__ mods,
                                       constant uint &n,
                                       uint tid [[thread_position_in_grid]]) {
    if (tid < n) {
        constexpr auto update_flag_primitive = 1u << 0u;
        constexpr auto update_flag_transform = 1u << 1u;
        constexpr auto update_flag_opaque_on = 1u << 2u;
        constexpr auto update_flag_opaque_off = 1u << 3u;
        constexpr auto update_flag_visibility = 1u << 4u;
        constexpr auto update_flag_user_id = 1u << 5u;
        constexpr auto update_flag_opaque = update_flag_opaque_on | update_flag_opaque_off;

        constexpr auto instance_option_disable_culling = 1u;
        constexpr auto instance_option_opaque = 4u;
        constexpr auto instance_option_non_opaque = 8u;

        auto m = mods[tid];
        auto instance = instances[m.index];
        if (m.flags & update_flag_primitive) {
            instance.mesh_index = m.index;
        }
        if (m.flags & update_flag_visibility) {
            instance.mask = m.vis_mask;
        }
        if (m.flags & update_flag_opaque) {
            instance.options = (m.flags & update_flag_opaque_on) ?
                                   instance_option_disable_culling | instance_option_opaque :
                                   instance_option_disable_culling | instance_option_non_opaque;
        }
        if (m.flags & update_flag_transform) {
            instance.transform[0] = m.affine[0].x;
            instance.transform[1] = m.affine[1].x;
            instance.transform[2] = m.affine[2].x;
            instance.transform[3] = m.affine[0].y;
            instance.transform[4] = m.affine[1].y;
            instance.transform[5] = m.affine[2].y;
            instance.transform[6] = m.affine[0].z;
            instance.transform[7] = m.affine[1].z;
            instance.transform[8] = m.affine[2].z;
            instance.transform[9] = m.affine[0].w;
            instance.transform[10] = m.affine[1].w;
            instance.transform[11] = m.affine[2].w;
        }
        if (m.flags & update_flag_user_id) {
            instance.intersection_function_offset = m.user_id;
        }
        instances[m.index] = instance;
    }
}

struct alignas(16) BindlessSlot {
    device const void *buffer;
    ulong buffer_size : 48;
    uint sampler2d : 8;
    uint sampler3d : 8;
    metal::texture2d<float> tex2d;
    metal::texture3d<float> tex3d;
};

struct Sampler {
    uchar filter;
    uchar address;
};

[[nodiscard]] inline auto sampler_code(Sampler s) { return (s.filter << 2u) | s.address; }

struct alignas(16) BindlessSlotModification {
    struct Buffer {
        device const void *handle;
        ulong size;
        uint op;
    };
    struct Texture2D {
        metal::texture2d<float> handle;
        Sampler sampler;
        uint op;
    };
    struct Texture3D {
        metal::texture3d<float> handle;
        Sampler sampler;
        uint op;
    };
    ulong slot;
    Buffer buffer;
    Texture2D tex2d;
    Texture3D tex3d;
};

static_assert(sizeof(BindlessSlot) == 32u, "");
static_assert(sizeof(BindlessSlotModification) == 64u, "");

[[kernel]] void update_bindless_array(device BindlessSlot *__restrict__ slots,
                                      device const BindlessSlotModification *__restrict__ mods,
                                      constant const uint &n,
                                      uint tid [[thread_position_in_grid]]) {
    if (tid < n) {
        [[maybe_unused]] constexpr auto op_none = 0u;
        constexpr auto op_update = 1u;
        constexpr auto op_remove = 2u;
        auto m = mods[tid];
        auto slot = slots[m.slot];
        if (m.buffer.op == op_update) {
            slot.buffer = m.buffer.handle;
            slot.buffer_size = m.buffer.size;
        } else if (m.buffer.op == op_remove) {
            slot.buffer = nullptr;
            slot.buffer_size = 0u;
        }
        if (m.tex2d.op == op_update) {
            slot.tex2d = m.tex2d.handle;
            slot.sampler2d = sampler_code(m.tex2d.sampler);
        } else if (m.tex2d.op == op_remove) {
            slot.tex2d = {};
            slot.sampler2d = 0u;
        }
        if (m.tex3d.op == op_update) {
            slot.tex3d = m.tex3d.handle;
            slot.sampler3d = sampler_code(m.tex3d.sampler);
        } else if (m.tex3d.op == op_remove) {
            slot.tex3d = {};
            slot.sampler3d = 0u;
        }
        slots[m.slot] = slot;
    }
}

struct RasterData {
    float4 p [[position]];
    float2 uv;
};

[[vertex]] RasterData swapchain_vertex_shader(
    constant float2 *in [[buffer(0)]],
    uint vid [[vertex_id]]) {
    auto p = in[vid];
    return RasterData{float4(p, 0.f, 1.f),
                      saturate(p * float2(.5f, -.5f) + .5f)};
}

[[fragment]] float4 swapchain_fragment_shader(
    RasterData in [[stage_in]],
    texture2d<float, access::sample> image [[texture(0)]]) {
    return float4(image.sample(sampler(filter::linear), in.uv).xyz, 1.f);
}

struct alignas(16) ICBHeader {
    uint count;
};

struct alignas(16) ICBSlot {
    alignas(16) uint3 block_size;
    alignas(16) uint4 dispatch_size_and_kernel_id;
};

struct ICB {
    device const void *__restrict__ buffer;
    uint offset;
    uint capacity;
    command_buffer command_buffer;
    compute_pipeline_state pipeline_state;
};

static_assert(sizeof(ICB) == 32u);

[[kernel]] void prepare_indirect_dispatches(constant ICB &icb,
                                            constant void *__restrict__ kernel_args,
                                            uint tid [[thread_position_in_grid]]) {
    if (auto index = icb.offset + tid; index < icb.capacity) {
        compute_command cmd{icb.command_buffer, index};
        cmd.reset();
        auto header = static_cast<device const ICBHeader *>(icb.buffer);
        if (tid < header->count) {
            auto slots = reinterpret_cast<device const ICBSlot *>(header + 1u);
            auto slot = slots[index];
            auto block_size = slot.block_size;
            auto dispatch_size = slot.dispatch_size_and_kernel_id.xyz;
            if (all(dispatch_size > 0u)) {
                cmd.set_compute_pipeline_state(icb.pipeline_state);
                cmd.set_kernel_buffer(kernel_args, 0u);
                cmd.set_kernel_buffer(&(slots[index].dispatch_size_and_kernel_id), 1u);
                auto block_count = (dispatch_size + block_size - 1u) / block_size;
                cmd.concurrent_dispatch_threadgroups(block_count, block_size);
            }
        }
    }
}
