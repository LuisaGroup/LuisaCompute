#pragma once

#include <runtime/resource.h>
#include <runtime/buffer.h>

#ifndef LC_DISABLE_DSL
#include <dsl/syntax.h>
#endif

namespace luisa::compute {

class Device;
class DeviceInterface;

class LC_RUNTIME_API ProceduralPrimitive final : public Resource {

    friend class Device;

private:
    ProceduralPrimitive(DeviceInterface *device,
                        const AccelOption &option,
                        BufferView<AABB> buffer) noexcept;

public:
    ProceduralPrimitive() noexcept = default;
    using Resource::operator bool;
    [[nodiscard]] luisa::unique_ptr<Command> build(AccelBuildRequest request = AccelBuildRequest::PREFER_UPDATE) noexcept;
};

}// namespace luisa::compute
