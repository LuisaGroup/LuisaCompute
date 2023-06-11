#pragma once

#include "rw_resource.h"
#include <luisa/ast/function.h>

namespace lc::validation {

class Shader : public RWResource {
    luisa::vector<Function::Binding> _bound_arguments;

public:
    luisa::span<const Function::Binding> bound_arguments() const { return _bound_arguments; }
    Shader(
        uint64_t handle, luisa::span<const Function::Binding> bound_arguments) : RWResource(handle, Tag::SHADER, false), _bound_arguments{bound_arguments.begin(), bound_arguments.end()} {}
};

}// namespace lc::validation

