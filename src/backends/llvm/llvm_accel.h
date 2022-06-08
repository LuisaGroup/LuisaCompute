//
// Created by Mike Smith on 2022/2/11.
//

#pragma once

#include <embree3/rtcore.h>

#include <core/stl.h>
#include <core/thread_pool.h>
#include <core/dirty_range.h>
#include <rtx/accel.h>

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

private:
    RTCScene _handle;
    luisa::vector<Instance> _instances;

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
};

[[nodiscard]] Hit accel_trace_closest(const LLVMAccel *accel, const Ray &ray) noexcept;
[[nodiscard]] bool accel_trace_any(const LLVMAccel *accel, const Ray &ray) noexcept;

}// namespace luisa::compute::llvm
