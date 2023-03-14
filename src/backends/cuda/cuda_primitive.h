//
// Created by Mike on 3/14/2023.
//

#pragma once

#include <runtime/rhi/resource.h>
#include <backends/cuda/optix_api.h>

namespace luisa::compute::cuda {

class CUDAPrimitive {

public:
    enum struct Tag : uint8_t {
        MESH,
        PROCEDURAL,
    };

private:
    Tag _tag;
    AccelOption _option;
    optix::TraversableHandle _handle;

protected:
    void _set_handle(optix::TraversableHandle handle) noexcept { _handle = handle; }

public:
    CUDAPrimitive(Tag tag, const AccelOption &option) noexcept
        : _tag{tag}, _option{option}, _handle{} {}
    virtual ~CUDAPrimitive() noexcept = default;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto option() const noexcept { return _option; }
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
    return build_options;
}

}// namespace luisa::compute::cuda
