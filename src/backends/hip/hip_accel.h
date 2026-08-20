#pragma once

#include <cstddef>

#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>

#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/rtx/accel.h>

namespace luisa::compute::hip {

class HIPDevice;
class HIPCommandEncoder;
class HIPPrimitive;

class HIPAccel {

public:
    struct Binding {
        uint64_t handle;
        uint64_t instance_buffer;
    };

    // Codegen-visible per-instance data. Must match the LLVM accel_instance_type layout:
    //   { [3 x <4 x float>] affine, u32 user_id, u32 sbt_offset, u32 mask, u32 flags, u64 handle }
    struct alignas(16) CodegenInstance {
        static constexpr uint32_t flag_opaque = 1u << 0u;
        static constexpr uint32_t visibility_mask_bits = 0xffu;
        // HIPRT copies visibility_mask into the gfx12 hardware instance node.
        // The public visibility domain is eight bits, so keep a coherent copy
        // of opacity in an otherwise unobservable bit. Native synchronous
        // queries can then transport it in their existing instance-id state
        // without another live value or per-candidate global load.
        static constexpr uint32_t packed_visibility_opaque_bit = 1u << 31u;

        float affine[3][4];// row-major 3x4
        uint32_t user_id;
        uint32_t sbt_offset;
        uint32_t visibility_mask;// low 8 bits public; bit 31 mirrors opacity
        uint32_t flags;
        uint64_t mesh_handle;
        uint64_t motion_data;
    };

    // Device-visible proof certificate stored immediately before the instance
    // array. `opacity_may_be_present` is monotone over an accel's lifetime:
    // every host/device transition to opaque sets it and no transition clears
    // it. Therefore zero proves that every current instance is non-opaque,
    // while nonzero deliberately means "unknown". Native effect-only any-hit
    // traversal consumes exactly that one-way implication and otherwise keeps
    // the ordinary resumable path.
    struct alignas(16) CodegenMetadata {
        uint32_t opacity_may_be_present;
        uint32_t reserved[3];
    };

    static_assert(sizeof(CodegenInstance) == 80u);
    static_assert(alignof(CodegenInstance) == 16u);
    static_assert(offsetof(CodegenInstance, affine) == 0u);
    static_assert(offsetof(CodegenInstance, user_id) == 48u);
    static_assert(offsetof(CodegenInstance, sbt_offset) == 52u);
    static_assert(offsetof(CodegenInstance, visibility_mask) == 56u);
    static_assert(offsetof(CodegenInstance, flags) == 60u);
    static_assert(offsetof(CodegenInstance, mesh_handle) == 64u);
    static_assert(offsetof(CodegenInstance, motion_data) == 72u);
    static_assert(sizeof(CodegenMetadata) == 16u);
    static_assert(alignof(CodegenMetadata) == 16u);
    static_assert(offsetof(CodegenMetadata, opacity_may_be_present) == 0u);
    static_assert(offsetof(hiprtFrameMatrix, matrix) == 0u);
    static_assert(sizeof(hiprtFrameMatrix::matrix) == sizeof(CodegenInstance::affine));
    static_assert(offsetof(hiprtFrameMatrix, time) == sizeof(CodegenInstance::affine));

private:
    AccelOption _option;
    hiprtContext _hiprt_ctx{nullptr};
    hiprtScene _scene{nullptr};
    bool _requires_rebuild{true};
    bool _hiprt_instances_dirty{true};
    mutable spin_mutex _mutex;

    // `_instance_allocation` owns [CodegenMetadata, CodegenInstance...]; the
    // public shader binding remains `_instance_buffer`, i.e. the first
    // CodegenInstance, so the existing accel ABI and instance indexing do not
    // change.
    hipDeviceptr_t _instance_allocation{};
    hipDeviceptr_t _instance_buffer{};
    size_t _instance_buffer_size{};
    hipDeviceptr_t _scene_build_buffer{};
    size_t _scene_build_capacity{};
    hipDeviceptr_t _scene_instances{};
    hipDeviceptr_t _scene_frames{};
    hipDeviceptr_t _scene_masks{};

    luisa::vector<CodegenInstance> _host_instances;
    luisa::vector<hiprtInstance> _hiprt_instances;
    luisa::vector<const HIPPrimitive *> _primitives;
    luisa::vector<const AccelBuildCommand::Modification *> _sorted_modifications;

    size_t _instance_count{};
    size_t _scene_instance_count{};

    [[nodiscard]] hiprtSceneBuildInput _make_scene_build_input(HIPCommandEncoder &encoder) noexcept;
    void _build(HIPCommandEncoder &encoder) noexcept;
    void _update(HIPCommandEncoder &encoder) noexcept;

public:
    explicit HIPAccel(hiprtContext ctx, const AccelOption &option) noexcept;
    ~HIPAccel() noexcept;
    void build(HIPCommandEncoder &encoder, AccelBuildCommand *command) noexcept;
    [[nodiscard]] Binding binding() const noexcept;
};

}// namespace luisa::compute::hip
