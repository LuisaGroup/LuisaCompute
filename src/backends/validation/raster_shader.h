#pragma once

#include <luisa/ast/function.h>
#include <luisa/core/stl/vector.h>

#include "rw_resource.h"

namespace lc::validation {

class RasterShader final : public RWResource {

public:
    struct RootArgument {
        luisa::compute::Function::Binding binding{};
        luisa::compute::Usage usage{luisa::compute::Usage::NONE};
        bool is_bound{false};
    };

private:
    luisa::vector<RootArgument> _root_arguments;
    bool _conservative_aot;

public:
    RasterShader(uint64_t handle, luisa::vector<RootArgument> root_arguments,
                 bool conservative_aot) noexcept
        : RWResource{handle, Tag::RASTER_SHADER, false},
          _root_arguments{std::move(root_arguments)},
          _conservative_aot{conservative_aot} {}

    [[nodiscard]] luisa::span<const RootArgument> root_arguments() const noexcept {
        return _root_arguments;
    }

    [[nodiscard]] bool conservative_aot() const noexcept {
        return _conservative_aot;
    }

    static constexpr luisa::string_view validation_res_name{"RasterShader"};
};

}// namespace lc::validation
