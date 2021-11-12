#pragma vengine_package ispc_vsproject

#include "ispc_shader.h"
namespace lc::ispc {
Shader::Shader(
    Function func,
    std::string const &str)
    : dllModule(str), func(func) {
    exportFunc = dllModule.function<void(uint, uint, uint, uint64)>("run");
    size_t sz = 0;
    for (auto &&i : func.arguments()) {
        varIdToArg.Emplace(i.uid(), sz);
        sz++;
    }
}
size_t Shader::GetArgIndex(uint varID) const {
    auto ite = varIdToArg.Find(varID);
    if (!ite) return std::numeric_limits<size_t>::max();
    return ite.Value();
}

void Shader::dispatch(
    uint3 sz,
    ArgVector const &vec) const {
    exportFunc(sz.x, sz.y, sz.z, (uint64)vec.data());
}
}// namespace lc::ispc