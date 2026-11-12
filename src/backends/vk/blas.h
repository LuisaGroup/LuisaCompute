#pragma once
#include <volk.h>
#include "primitive_base.h"
#include "default_buffer.h"
#include <luisa/runtime/rtx/accel.h>
namespace lc::vk {
class CommandBuffer;
class OpacityMicromap;
using namespace luisa;
using namespace luisa::compute;
class Tlas;
class Blas;
class MeshHandle {
public:
    Blas *mesh = nullptr;
    Tlas *accel = nullptr;
    size_t accel_index = 0;
    size_t mesh_index = 0;
    static MeshHandle *allocate_handle();
    static void destroy_handle(MeshHandle *handle);
};
class Blas : public PrimitiveBase {
    friend class Tlas;
private:
    luisa::spin_mutex _handle_mtx;
    VkAccelerationStructureKHR _accel{nullptr};
    vstd::unique_ptr<DefaultBuffer> _accel_buffer;
    VkAccelerationStructureBuildGeometryInfoKHR *_acceleration_build_geometry_info{nullptr};
    AccelOption _option;
    const Buffer *_scratch_buffer{nullptr};
    uint64_t _scratch_buffer_offset{0};
    uint32_t _last_primitive_count{0u};
    // Opacity micromap attached to this BLAS (VK_KHR_opacity_micromap), or null.
    // Set by `pre_build(MeshBuildCommand)` from the command's micromap binding (and
    // always assigned, so a build without a micromap detaches a stale one). Never
    // owned here: the micromap belongs to the frontend resource the user created and
    // must outlive every traced acceleration structure built from it, unless the
    // driver reports the build sizes as discardable.
    OpacityMicromap *_micromap{nullptr};
    // Source of the packed micro-triangle bits, recorded by pre_build() so that
    // build() can copy them into the micromap's usage-correct build-input buffer
    // right before vkCmdBuildMicromapsEXT.
    VkBuffer _micromap_data_buffer{VK_NULL_HANDLE};
    size_t _micromap_data_offset{0u};
    vstd::fixed_vector<MeshHandle *, 2> _handles;

    MeshHandle *_add_accel_ref(Tlas *accel, uint index);
    void _remove_accel_ref(MeshHandle *handle);
    void _sync_tlas();
    void _pre_build(
        CommandBuffer &cmdbuffer,
        VkAccelerationStructureGeometryKHR* acceleration_structure_geometry,
        uint32_t primitive_count,
        AccelBuildRequest request
    );
public:
    [[nodiscard]] auto &accel() const { return _accel; }
    [[nodiscard]] auto &option() const { return _option; }
    [[nodiscard]] bool has_motion() const noexcept { return _option.motion.is_enabled(); }
    // Backend-internal attachment of an opacity micromap to this BLAS
    // (VK_KHR_opacity_micromap).
    //
    // NOT reachable from the default build, intentionally: nothing in the public
    // LuisaCompute accelerator API can create an `OpacityMicromap` (there is no
    // AccelOption/MeshBuildCommand micromap input and no
    // DeviceInterface::create_micromap), so no caller exists and `_micromap`
    // stays null - which keeps `pre_build(MeshBuildCommand)` byte-identical to
    // the pre-extension build. Ownership stays with the caller: a micromap is
    // only discardable after the build when the driver says so, so this class
    // never destroys it. Passing nullptr detaches.
    void attach_opacity_micromap(OpacityMicromap *micromap) noexcept { _micromap = micromap; }
    [[nodiscard]] OpacityMicromap *opacity_micromap() const noexcept { return _micromap; }
    Blas(Device *device, AccelOption const &option);
    void pre_build(
        CommandBuffer &cmdbuffer,
        MeshBuildCommand const *cmd);
    void pre_build(
        CommandBuffer &cmdbuffer,
        ProceduralPrimitiveBuildCommand const *cmd);
    void build(
        CommandBuffer &cmdbuffer,
        MeshBuildCommand const *cmd);
    void build(
        CommandBuffer &cmdbuffer,
        ProceduralPrimitiveBuildCommand const *cmd);
    ~Blas() override;
    uint64_t get_accel_device_address() const;
};
}// namespace lc::vk
