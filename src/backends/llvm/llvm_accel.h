//
// Created by Mike Smith on 2022/2/11.
//

#pragma once

#include <embree4/rtcore.h>

#include <core/stl.h>
#include <core/thread_pool.h>
#include <core/dirty_range.h>
#include <rtx/accel.h>
#include <backends/llvm/llvm_abi.h>

namespace luisa::compute::llvm {

class LLVMMesh;

class LLVMAccel {

public:
    struct alignas(16) Instance {
        float affine[12];
        bool visible;
        bool dirty;
        uint pad;
        RTCGeometry geometry;
    };
    static_assert(sizeof(Instance) == 64u);

    struct alignas(16) Handle {
        const LLVMAccel *accel;
        Instance *instances;
    };

private:
    RTCScene _handle;
    mutable luisa::vector<Instance> _instances;

private:
    [[nodiscard]] static std::array<float, 12> _compress(float4x4 m) noexcept;
    [[nodiscard]] static float4x4 _decompress(std::array<float, 12> m) noexcept;

public:
    LLVMAccel(RTCDevice device, AccelUsageHint hint) noexcept;
    ~LLVMAccel() noexcept;
    void build(ThreadPool &pool, size_t instance_count,
               luisa::span<const AccelBuildCommand::Modification> mods) noexcept;
    [[nodiscard]] Hit trace_closest(Ray ray) const noexcept;
    [[nodiscard]] bool trace_any(Ray ray) const noexcept;
    [[nodiscard]] auto handle() const noexcept { return Handle{this, _instances.data()}; }
};

[[nodiscard]] float32x4_t accel_trace_closest(const LLVMAccel *accel, int64_t r0, int64_t r1, int64_t r2, int64_t r3) noexcept;
[[nodiscard]] bool accel_trace_any(const LLVMAccel *accel, int64_t r0, int64_t r1, int64_t r2, int64_t r3) noexcept;

struct alignas(16) LLVMAccelInstance {
    float affine[12];
    bool visible;
    bool dirty;
    uint pad;
    uint geom0;
    uint geom1;
};

}// namespace luisa::compute::llvm

LUISA_STRUCT(luisa::compute::llvm::LLVMAccelInstance,
             affine, visible, dirty, pad, geom0, geom1) {};
