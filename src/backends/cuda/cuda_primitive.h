#pragma once

#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/rhi/resource.h>
#include "optix_api.h"

namespace luisa::compute::cuda {

class CUDACommandEncoder;

class CUDAPrimitive {

public:
    enum struct Tag : uint8_t {
        MESH,
        PROCEDURAL,
        CURVE,
    };

private:
    Tag _tag;
    AccelOption _option;

protected:
    mutable spin_mutex _mutex;

private:
    CUdeviceptr _bvh_buffer_handle{};
    size_t _bvh_buffer_size{};
    size_t _update_buffer_size{};
    luisa::string _name;
    luisa::vector<CUdeviceptr> _motion_buffers{};

protected:
    optix::TraversableHandle _handle{};

protected:
    [[nodiscard]] virtual optix::BuildInput _make_build_input() noexcept = 0;
    void _build(CUDACommandEncoder &encoder) noexcept;
    void _update(CUDACommandEncoder &encoder) noexcept;

public:
    CUDAPrimitive(Tag tag, const AccelOption &option) noexcept;
    virtual ~CUDAPrimitive() noexcept;
    [[nodiscard]] optix::TraversableHandle handle() const noexcept;
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto option() const noexcept { return _option; }
    [[nodiscard]] auto pointer_to_handle() const noexcept { return &_handle; }
    void set_name(luisa::string &&name) noexcept;
};

[[nodiscard]] inline auto make_optix_build_options(const AccelOption &option, optix::BuildOperation op) noexcept {
    optix::AccelBuildOptions build_options{};
    build_options.operation = op;
    switch (option.hint) {
        case AccelOption::UsageHint::FAST_TRACE:
            build_options.buildFlags = optix::BUILD_FLAG_PREFER_FAST_TRACE;
            break;
        case AccelOption::UsageHint::FAST_BUILD:
            build_options.buildFlags = optix::BUILD_FLAG_PREFER_FAST_BUILD;
            break;
    }
    if (option.allow_compaction) {
        build_options.buildFlags |= optix::BUILD_FLAG_ALLOW_COMPACTION;
    }
    if (option.allow_update) {
        build_options.buildFlags |= optix::BUILD_FLAG_ALLOW_UPDATE;
    }
    auto &motion_options = option.motion_options;
    build_options.motionOptions.numKeys = motion_options.num_keys;
    build_options.motionOptions.timeBegin = motion_options.time_begin;
    build_options.motionOptions.timeEnd = motion_options.time_end;
    build_options.motionOptions.flags = static_cast<uint16_t>(motion_options.flag); // They share the same definition.
    return build_options;
}

}// namespace luisa::compute::cuda

