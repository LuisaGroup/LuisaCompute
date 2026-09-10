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
    uint64_t _scratch_buffer_offset{0};
    uint _last_instance_count = 0;
    // Number of instance slots carrying a non-null refresh_blas. Mirrors the
    // size of the former refresh hash map so pre_build can skip the instance
    // rewrite in O(1) when nothing is pending. Kept in sync by
    // _queue_refresh/_drop_refresh and the pre_build/_resize_instance scans.
    size_t _pending_refresh_count = 0;
    struct Instance {
        MeshHandle *handle = nullptr;
        // Whether the instance was assigned a MotionInstance primitive (as
        // opposed to a plain Blas); required to keep the TLAS motion-capable
        // across builds whose modification list does not touch the primitive.
        bool is_motion_instance = false;
        // Pending BLAS-address refresh queued by a BLAS recreate (_sync_tlas).
        // Stores the stable Blas (never the pooled MeshHandle) so a destroyed
        // or recycled handle can never leave a dangling entry behind. It lives
        // in the slot itself because the key space (instance index) is exactly
        // the dense [0, _all_instance.size()) range of the contiguous instance
        // array: queue/consume/erase are O(1) array operations with no hash
        // lookups or per-entry node allocations, and shrinking the vector
        // drops entries of removed slots automatically.
        Blas *refresh_blas = nullptr;
    };
    bool _require_rebuild = true;
    bool _has_motion = false;  // true if any child BLAS has motion or any MotionInstance is present
    bool _built_with_motion = false;  // motion flag of the currently built TLAS handle
    // Host-side shadow copies of the instance buffers used by the motion path;
    // updated incrementally and uploaded whole so untouched instances survive
    // partial modification lists and pending refresh entries.
    luisa::vector<uint8_t> _motion_instance_cache;
    luisa::vector<uint8_t> _std_instance_cache;
    vstd::vector<Instance> _all_instance;
    void _resize_instance(size_t size);
    void _update_mesh(MeshHandle *handle);
    void _set_mesh(Blas *mesh, uint64 index, bool is_motion_instance, bool explicit_primitive);
    // Queues a refresh of the BLAS device address of one instance slot.
    void _queue_refresh(size_t instance_index, Blas *blas) noexcept;
    // Drops the pending refresh of one slot if it references `blas` (used by
    // ~Blas so a destroyed BLAS never leaks a refresh entry behind).
    void _drop_refresh(size_t instance_index, Blas *blas) noexcept;

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
