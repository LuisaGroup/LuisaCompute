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

    static constexpr auto max_motion_keyframe_count = 64u;

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

protected:
    optix::TraversableHandle _handle{};

protected:
    [[nodiscard]] virtual optix::BuildInput _make_build_input() const noexcept = 0;
    void _build(CUDACommandEncoder &encoder) noexcept;
    void _update(CUDACommandEncoder &encoder) noexcept;

    [[nodiscard]] const CUdeviceptr *_motion_buffer_pointers(CUdeviceptr base, size_t total_size) const noexcept;

public:
    CUDAPrimitive(Tag tag, const AccelOption &option) noexcept;
    virtual ~CUDAPrimitive() noexcept;
    [[nodiscard]] optix::TraversableHandle handle() const noexcept;
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto option() const noexcept { return _option; }
    [[nodiscard]] auto pointer_to_handle() const noexcept { return &_handle; }
    void set_name(luisa::string &&name) noexcept;
    [[nodiscard]] auto is_motion_blur() const noexcept { return static_cast<bool>(_option.motion); }
    [[nodiscard]] auto motion_keyframe_count() const noexcept { return std::max<uint>(1u, _option.motion.keyframe_count); }
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
    if (auto m = option.motion) {
        build_options.motionOptions.numKeys = m.keyframe_count;
        if (m.should_vanish_start) {
            build_options.motionOptions.flags |= optix::MOTION_FLAG_START_VANISH;
        }
        if (m.should_vanish_end) {
            build_options.motionOptions.flags |= optix::MOTION_FLAG_END_VANISH;
        }
        build_options.motionOptions.timeBegin = m.time_start;
        build_options.motionOptions.timeEnd = m.time_end;
    }
    return build_options;
}

}// namespace luisa::compute::cuda
