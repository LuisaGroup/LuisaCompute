#pragma vengine_package ispc_vsproject

#include "ispc_shader.h"
namespace lc::ispc {
Shader::Shader(
    Function func,
    std::string const &str)
    : dllModule(str), func(func) {
    exportFunc = dllModule.function<FuncType>("run");
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

ThreadTaskHandle Shader::dispatch(
    ThreadPool *tPool,
    uint3 sz,
    ArgVector const &vec) const {
    auto blockCount = uint3(8, 8, 8);
    auto threadCount = sz / blockCount;
    auto handle = tPool->GetParallelTask(
        [=](size_t i) {
            uint threadIdxZ = i / (threadCount.y * threadCount.x);
            i -= threadCount.y * threadCount.x * threadIdxZ;
            uint threadIdxY = i / threadCount.x;
            i -= threadIdxY * threadCount.x;
            uint threadIdxX = i;
            exportFunc(
                threadCount.x,
                threadCount.y,
                threadCount.z,
                blockCount.x,
                blockCount.y,
                blockCount.z,
                threadIdxX,
                threadIdxY,
                threadIdxZ,
                (uint64)vec.data());
        },
        threadCount.x * threadCount.y * threadCount.z,
        true);
    handle.Execute();
    return handle;
    //exportFunc(sz.x, sz.y, sz.z, (uint64)vec.data());
}
}// namespace lc::ispc