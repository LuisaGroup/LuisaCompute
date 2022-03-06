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

/**
 * @brief Strucure of acceleration 
 * 
 */
class ISPCAccel {

public:
    /**
     * @brief Instance
     * 
     */
    struct alignas(16) Instance {
        const ISPCMesh *mesh{nullptr};
        std::array<float, 12> transform{};
        bool visible{false};
    };
    static_assert(sizeof(Instance) == 64u);

    /**
     * @brief Handle for device usage
     * 
     */
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
    /**
     * @brief Construct a new ISPCAccel object
     * 
     * @param device RTCDevice where scene builds
     * @param hint scene build hint
     */
    ISPCAccel(RTCDevice device, AccelBuildHint hint) noexcept;
    ~ISPCAccel() noexcept;
    /**
     * @brief Generate handle for device usage
     * 
     * @return handle 
     */
    [[nodiscard]] auto handle() const noexcept {
        return Handle{_handle, _committed_transforms.data()};
    }
    /**
     * @brief If resoure is used in accel structure
     * 
     * @param handle handle of resource to be tested
     * @return true / false
     */
    [[nodiscard]] auto uses_resource(uint64_t handle) const noexcept {
        return _resources.find(handle) != _resources.cend();
    }
    /**
     * @brief Add a mesh to accel structure
     * 
     * @param mesh mesh to be added
     * @param transform mesh's transform
     * @param visible if mesh is visible
     */
    void push_mesh(const ISPCMesh *mesh, float4x4 transform, bool visible) noexcept;
    /**
     * @brief Pop the lastest mesh
     * 
     */
    void pop_mesh() noexcept;
    /**
     * @brief Set the mesh object
     * 
     * @param index place to set
     * @param mesh new mesh
     * @param transform new transform
     * @param visible new visibility
     */
    void set_mesh(size_t index, const ISPCMesh *mesh, float4x4 transform, bool visible) noexcept;
    /**
     * @brief Set visibility of mesh
     * 
     * @param index place to set
     * @param visible new visibility
     */
    void set_visibility(size_t index, bool visible) noexcept;
    /**
     * @brief Set transform of mesh
     * 
     * @param index place to set
     * @param transform new transform
     */
    void set_transform(size_t index, float4x4 transform) noexcept;
    /**
     * @brief Build the accel structure
     * 
     * @param pool thread pool
     */
    void build(ThreadPool &pool) noexcept;
    /**
     * @brief Update the accel structure
     * 
     * @param pool thread pool
     */
    void update(ThreadPool &pool) noexcept;
};

}// namespace luisa::compute::ispc
