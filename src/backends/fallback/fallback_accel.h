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
class FallbackMesh;

class alignas(16) FallbackAccel {

public:
    using Instance = api::AccelInstance;
    static_assert(sizeof(Instance) == 64u);
    using View = api::AccelView;

private:
    RTCScene _handle{};
    luisa::vector<Instance> _instances;
    Instance *_instance_data{};
    luisa::vector<RTCGeometry> _geometries;

public:
    [[nodiscard]] RTCScene handle() const noexcept { return _handle; }
    FallbackAccel(RTCDevice device, const AccelOption &option) noexcept;
    ~FallbackAccel() noexcept;
    void build(luisa::unique_ptr<AccelBuildCommand> cmd) noexcept;
    [[nodiscard]] auto view() noexcept {
        return View{_handle, &_instance_data};
    }
};

using FallbackAccelView = FallbackAccel::View;

}// namespace luisa::compute::fallback
