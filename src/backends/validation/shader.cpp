#include "shader.h"
#include "resource.h"
#include <core/logging.h>
namespace lc::validation {
luisa::vector<Function::Binding> Shader::fallback_binding(luisa::vector<Function::Binding> &bindings) {
    luisa::vector<Function::Binding> old{bindings};
    for (auto &&i : bindings) {
        luisa::visit(
            [&]<typename T>(T &t) {
                if constexpr (std::is_same_v<T, luisa::monostate>) {
                    LUISA_ERROR("Binding Contain unwanted variable.");
                } else {
                    t.handle = reinterpret_cast<Resource *>(t.handle)->handle();
                }
            },
            i);
    }
    return old;
}
Shader::Shader(
    uint64_t handle,
    luisa::vector<Function::Binding> bound_arguments)
    : RWResource{handle, Tag::SHADER, false}, _bound_arguments{std::move(bound_arguments)} {
}
}// namespace lc::validation