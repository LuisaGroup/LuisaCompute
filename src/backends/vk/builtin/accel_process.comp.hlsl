// Keep packed field masks and update flags synchronized with the host ABI.
#define LC_VULKAN_ACCEL_UPDATE_LAYOUT(shader_name, cpp_name, value) \
    static const uint LC_VULKAN_ACCEL_##shader_name = value;
#include "vulkan_accel_update_layout.def"
#undef LC_VULKAN_ACCEL_UPDATE_LAYOUT
static const uint LC_VULKAN_ACCEL_FLAG_OPAQUE =
    LC_VULKAN_ACCEL_FLAG_OPAQUE_ON |
    LC_VULKAN_ACCEL_FLAG_OPAQUE_OFF;

struct MeshInst {
    float4 p0;
    float4 p1;
    float4 p2;
    uint InstanceIDMask;
    uint ContributionOffsetFlags;
    uint2 accelStructPtr;
};

struct InputInst {
    float4 p0;
    float4 p1;
    float4 p2;
    uint2 mesh;
    uint IndexVisibility;
    uint UserIdFlags;
};

RWStructuredBuffer<MeshInst> _InstBuffer : register(u1);
StructuredBuffer<InputInst> _SetBuffer : register(t0);

struct PushConstants {
    uint dispatch_count;
    uint instance_count;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(LC_VULKAN_ACCEL_BLOCK_SIZE, 1, 1)]
void main(uint id : SV_DispatchThreadID) {
    if (id >= pc.dispatch_count) { return; }
    InputInst value = _SetBuffer[id];
    uint index = value.IndexVisibility & LC_VULKAN_ACCEL_INDEX_MASK;
    if (index >= pc.instance_count) { return; }
    uint visibility =
        (value.IndexVisibility >> LC_VULKAN_ACCEL_HIGH_BYTE_SHIFT) &
        LC_VULKAN_ACCEL_BYTE_MASK;
    uint user_id = value.UserIdFlags & LC_VULKAN_ACCEL_INDEX_MASK;
    uint flags =
        (value.UserIdFlags >> LC_VULKAN_ACCEL_HIGH_BYTE_SHIFT) &
        LC_VULKAN_ACCEL_BYTE_MASK;
    if ((flags & LC_VULKAN_ACCEL_FLAG_TRANSFORM) != 0u) {
        _InstBuffer[index].p0 = value.p0;
        _InstBuffer[index].p1 = value.p1;
        _InstBuffer[index].p2 = value.p2;
    }
    if ((flags & LC_VULKAN_ACCEL_FLAG_VISIBILITY) != 0u) {
        uint packed = _InstBuffer[index].InstanceIDMask;
        _InstBuffer[index].InstanceIDMask =
            (packed & LC_VULKAN_ACCEL_INDEX_MASK) |
            (visibility << LC_VULKAN_ACCEL_HIGH_BYTE_SHIFT);
    }
    if ((flags & LC_VULKAN_ACCEL_FLAG_USER_ID) != 0u) {
        uint packed = _InstBuffer[index].InstanceIDMask;
        _InstBuffer[index].InstanceIDMask =
            (packed & LC_VULKAN_ACCEL_HIGH_BYTE_MASK) | user_id;
    }
    uint contribution_flags =
        _InstBuffer[index].ContributionOffsetFlags &
        LC_VULKAN_ACCEL_HIGH_BYTE_MASK;
    if ((flags & LC_VULKAN_ACCEL_FLAG_OPAQUE) != 0u) {
        contribution_flags =
            ((flags & LC_VULKAN_ACCEL_FLAG_OPAQUE_ON) != 0u ?
                 LC_VULKAN_ACCEL_INSTANCE_FORCE_OPAQUE :
                 LC_VULKAN_ACCEL_INSTANCE_FORCE_NO_OPAQUE) <<
            LC_VULKAN_ACCEL_HIGH_BYTE_SHIFT;
    }
    _InstBuffer[index].ContributionOffsetFlags = contribution_flags;
    if ((flags & LC_VULKAN_ACCEL_FLAG_MESH) != 0u) {
        _InstBuffer[index].accelStructPtr = value.mesh;
    }
}
