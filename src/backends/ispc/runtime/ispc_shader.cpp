#pragma vengine_package ispc_vsproject
#include "ispc_shader.h"
namespace lc::ispc {
Shader::Shader(std::string const &str)
    : dllModule(str) {
    exportFunc = dllModule.function<void(uint, uint, uint, uint64)>("run");
}
void Shader::dispatch(
    uint x,
    uint y,
    uint z,
    vstd::vector<void const *> const &vec) const {
    exportFunc(x, y, z, (uint64)vec.data());
}
}// namespace lc::ispc