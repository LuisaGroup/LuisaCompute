//
// Created by Mike Smith on 2022/2/11.
//

#pragma once

#include <luisa/core/stl.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/dsl/struct.h>
#include <luisa/runtime/rtx/ray.h>
#include <luisa/runtime/rtx/hit.h>

#include "fallback_embree.h"
#include "fallback_device_api.h"

namespace luisa {
class ThreadPool;
}// namespace luisa

namespace luisa::compute::fallback {

class FallbackCommandQueue;
class FallbackPrimBase;
class FallbackMesh;

struct alignas(16) FallbackAccelRuntimeHeader {
    api::AccelInstance *instances{};
    const uint8_t *instance_flags{};
};

class alignas(16) FallbackAccel {

public:
    using Instance = api::AccelInstance;
    static_assert(sizeof(Instance) == 64u);
    using View = api::AccelView;

private:
    RTCScene _handle{};
    FallbackAccelRuntimeHeader _header{};
    luisa::vector<Instance> _instances;
    luisa::vector<uint8_t> _instance_flags;
    luisa::vector<RTCGeometry> _geometries;
    luisa::vector<const FallbackPrimBase *> _primitives;
    luisa::vector<const FallbackMesh *> _direct_meshes;
    luisa::vector<uint64_t> _mesh_revisions;

public:
    [[nodiscard]] RTCScene handle() const noexcept { return _handle; }
    FallbackAccel(RTCDevice device, const AccelOption &option) noexcept;
    ~FallbackAccel() noexcept;
    void build(luisa::unique_ptr<AccelBuildCommand> cmd) noexcept;
    [[nodiscard]] auto view() noexcept {
        return View{_handle, &_header.instances};
    }
};

using FallbackAccelView = FallbackAccel::View;

}// namespace luisa::compute::fallback
