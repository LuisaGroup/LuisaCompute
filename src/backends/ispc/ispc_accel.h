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
    struct alignas(16) Instance {
        float affine[12];
        bool visible;
        bool dirty;
        uint pad;
        RTCGeometry geometry;
    };
    static_assert(sizeof(Instance) == 64u);

    struct Handle {
        RTCScene scene;
        Instance *instances;
    };

private:
    RTCScene _handle;
    luisa::vector<Instance> _instances;

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
    [[nodiscard]] auto handle() noexcept {
        return Handle{.scene = _handle,
                      .instances = _instances.data()};
    }
    /**
     * @brief Build (or rebuild) the acceleration structure
     *
     * @param pool thread pool to perform the build
     * @param mesh_handles handles of the meshes to emplace in the acceleration structure
     * @param requests transform and visibility update requests from the host
     */
    void build(ThreadPool &pool, luisa::span<const uint64_t> mesh_handles,
               luisa::span<const AccelUpdateRequest> requests) noexcept;
    /**
     * @brief Update the acceleration structure
     * @param pool thread pool to perform the build
     * @param requests transform and visibility update requests from the host
     */
    void update(ThreadPool &pool, luisa::span<const AccelUpdateRequest> requests) noexcept;
};

}// namespace luisa::compute::ispc
