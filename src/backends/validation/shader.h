#pragma once

#include "rw_resource.h"
#include <ast/function.h>

namespace lc::validation {

class Shader : public RWResource {
    luisa::vector<Function::Binding> _bound_arguments;

public:
    luisa::span<const Function::Binding> bound_arguments() const { return _bound_arguments; }
    static luisa::vector<Function::Binding> fallback_binding(luisa::vector<Function::Binding> &bindings);
    Shader(
        uint64_t handle,
        luisa::vector<Function::Binding> bound_arguments);
};

}// namespace lc::validation
