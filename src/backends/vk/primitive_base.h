#pragma once
#include "resource.h"
namespace lc::vk {
// Base class for all primitive types that can be placed in a TLAS.
// This allows the TLAS to handle Blas (mesh/procedural), ClusterBlas (cluster
// BLAS built through VK_NV_cluster_acceleration_structure) and MotionInstance
// without knowing their concrete types.
class PrimitiveBase : public Resource {
public:
    enum class PrimTag {
        BLAS,
        MOTION_INSTANCE,
        // A cluster (mega-geometry) bottom-level acceleration structure. It is a
        // sibling of Blas, never a subclass: VK_NV_cluster_acceleration_structure
        // creates no VkAccelerationStructureKHR, so a ClusterBlas owns a byte pool
        // and a raw VkDeviceAddress instead of an acceleration-structure handle.
        CLUSTER_BLAS
    };

private:
    PrimTag _prim_tag;

public:
    PrimitiveBase(Device *device, PrimTag prim_tag) : Resource(device), _prim_tag(prim_tag) {}
    [[nodiscard]] auto prim_tag() const noexcept { return _prim_tag; }
    [[nodiscard]] bool is_motion_instance() const noexcept { return _prim_tag == PrimTag::MOTION_INSTANCE; }
    // Traceable device address of the acceleration structure this primitive
    // resolves to, i.e. the value a TLAS writes into
    // VkAccelerationStructureInstanceKHR::accelerationStructureReference.
    // `Blas` already declares `uint64_t get_accel_device_address() const` and
    // `bool has_motion() const noexcept` with exactly the signatures below, so
    // it satisfies these two pure virtuals implicitly and blas.h needs no edit
    // (see LC_CLUSTER_IMPL_SPEC §3.1/§3.5); `MotionInstance` forwards to its
    // child and `ClusterBlas` returns its cached cluster-BLAS address.
    [[nodiscard]] virtual uint64_t get_accel_device_address() const = 0;
    [[nodiscard]] virtual bool has_motion() const noexcept = 0;
};
}// namespace lc::vk
