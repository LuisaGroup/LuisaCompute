#pragma once
#include <volk.h>
#include "resource.h"
#include "default_buffer.h"
#include <luisa/runtime/rtx/accel.h>
namespace lc::vk {
class CommandBuffer;
using namespace luisa;
using namespace luisa::compute;
class MeshHandle;
class Blas;
class Tlas : public Resource {
    friend class Blas;
private:
    VkAccelerationStructureKHR _accel{nullptr};
    vstd::unique_ptr<DefaultBuffer> _accel_buffer;
    vstd::unique_ptr<DefaultBuffer> _instance_buffer;
    vstd::unique_ptr<DefaultBuffer> _motion_instance_buffer;  // 160-byte stride buffer for TLAS build when motion is enabled
    // VkAccelerationStructureMotionInstanceNV is 152 bytes, but the Vulkan driver
    // requires each motion instance to be 16-byte aligned in the instance buffer.
    // ceil(152 / 16) * 16 = 160.
    static constexpr size_t kMotionInstanceStride = 160u;
    VkAccelerationStructureBuildGeometryInfoKHR *_acceleration_build_geometry_info{nullptr};
    AccelOption _option;
    Buffer const *_scratch_buffer{nullptr};
    // Pending BLAS-address refreshes, keyed by TLAS instance index. The value
    // is the stable Blas (never the pooled MeshHandle) so that a destroyed or
    // recycled handle can never leave a dangling entry behind.
    vstd::unordered_map<uint64, Blas *> _set_map;
    uint64_t _scratch_buffer_offset{0};
    uint _last_instance_count = 0;
    struct Instance {
        MeshHandle *handle = nullptr;
        // Whether the instance was assigned a MotionInstance primitive (as
        // opposed to a plain Blas); required to keep the TLAS motion-capable
        // across builds whose modification list does not touch the primitive.
        bool is_motion_instance = false;
    };
    bool _require_rebuild = true;
    bool _has_motion = false;  // true if any child BLAS has motion or any MotionInstance is present
    bool _built_with_motion = false;  // motion flag of the currently built TLAS handle
    // Host-side shadow copies of the instance buffers used by the motion path;
    // updated incrementally and uploaded whole so untouched instances survive
    // partial modification lists and pending _set_map refreshes.
    luisa::vector<uint8_t> _motion_instance_cache;
    luisa::vector<uint8_t> _std_instance_cache;
    vstd::vector<Instance> _all_instance;
    void _resize_instance(size_t size);
    void _update_mesh(MeshHandle *handle);
    void _set_mesh(Blas *mesh, uint64 index, bool is_motion_instance, bool explicit_primitive);

public:
    Tlas(Device *device, AccelOption const &option);
    void pre_build(
        CommandBuffer &cmdbuffer,
        uint instance_count,
        luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
        luisa::vector<uint4> &cache,
        luisa::span<AccelBuildCommand::Modification const> modifications,
        AccelBuildRequest request);
    void build(
        CommandBuffer &cmdbuffer,
        uint instance_count);
    ~Tlas() override;
    [[nodiscard]] auto &accel() const { return _accel; }
    [[nodiscard]] auto instance_buffer() const { return _instance_buffer.get(); }
    [[nodiscard]] auto accel_buffer() const { return _accel_buffer.get(); }
    [[nodiscard]] bool has_motion() const noexcept { return _has_motion; }
};
}// namespace lc::vk
