//
// Created by Mike Smith on 2022/2/11.
//

#pragma once

#include <embree3/rtcore.h>

#include <core/stl.h>
#include <core/thread_pool.h>
#include <core/dirty_range.h>

namespace luisa::compute::ispc {

class ISPCMesh;

class ISPCAccel {

public:
    struct alignas(16) Instance {
        const ISPCMesh *mesh;
        std::array<float, 12> transform;
        bool visible;
    };
    static_assert(sizeof(Instance) == 64u);

    struct Handle {
        RTCScene scene;
        const float4x4 *transforms;
    };

private:
    RTCScene _handle;
    luisa::vector<Instance> _instances;
    luisa::vector<RTCGeometry> _committed_geometries;
    luisa::vector<float4x4> _committed_transforms;
    luisa::unordered_set<uint64_t> _resources;
    DirtyRange _dirty;

private:
    [[nodiscard]] static std::array<float, 12> _compress(float4x4 m) noexcept;
    [[nodiscard]] static float4x4 _decompress(std::array<float, 12> m) noexcept;

public:
    ISPCAccel(RTCDevice device, AccelBuildHint hint) noexcept;
    ~ISPCAccel() noexcept;
    [[nodiscard]] auto handle() const noexcept {
        return Handle{_handle, _committed_transforms.data()};
    }
    [[nodiscard]] auto uses_resource(uint64_t handle) const noexcept {
        return _resources.find(handle) != _resources.cend();
    }
    void push_mesh(const ISPCMesh *mesh, float4x4 transform, bool visible) noexcept;
    void pop_mesh() noexcept;
    void set_mesh(size_t index, const ISPCMesh *mesh, float4x4 transform, bool visible) noexcept;
    void set_visibility(size_t index, bool visible) noexcept;
    void set_transform(size_t index, float4x4 transform) noexcept;
    void build(ThreadPool &pool) noexcept;
    void update(ThreadPool &pool) noexcept;
};

}// namespace luisa::compute::ispc
