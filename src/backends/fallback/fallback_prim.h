#pragma once

#include "fallback_embree.h"

namespace luisa::compute::fallback {

class FallbackPrimBase {
public:
    virtual ~FallbackPrimBase() noexcept = default;
    [[nodiscard]] virtual bool is_motion_instance() const noexcept = 0;
};

class FallbackPrim : public FallbackPrimBase {

private:
    RTCScene _handle{};
    RTCGeometry _geometry{};
    AccelOption _option;

protected:
    [[nodiscard]] bool _is_geometry_created() const noexcept { return _geometry != nullptr; }
    void _create_geometry(RTCGeometryType geom_type) noexcept;

public:
    FallbackPrim(RTCDevice device, RTCGeometryType geom_type, const AccelOption &option) noexcept;
    // FIXME: workaround for curve that cannot decide geometry type on construction
    FallbackPrim(RTCDevice device, const AccelOption &option) noexcept;
    [[nodiscard]] bool is_motion_instance() const noexcept final { return false; }
    [[nodiscard]] virtual bool is_curve() const noexcept { return false; }
    [[nodiscard]] virtual bool is_mesh() const noexcept { return false; }
    [[nodiscard]] virtual uint64_t build_revision() const noexcept { return 1u; }
    [[nodiscard]] virtual RTCScene committed_handle() const noexcept { return _handle; }
    ~FallbackPrim() noexcept override;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto geometry() const noexcept { return _geometry; }
    [[nodiscard]] const AccelOption &option() const noexcept { return _option; }
    [[nodiscard]] auto motion() const noexcept { return _option.motion; }
};

}// namespace luisa::compute::fallback
