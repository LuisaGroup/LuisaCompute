#pragma vengine_package ispc_vsproject

#include "ispc_shader.h"
namespace lc::ispc {
Shader::Shader(
    Function func,
    JITModule module)
    : module{std::move(module)}, func(func) {
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
    auto blockCount = func.block_size();
    auto threadCount = sz / blockCount;
    auto handle = tPool->GetParallelTask(
        [=](size_t i) {
            uint threadIdxZ = i / (threadCount.y * threadCount.x);
            i -= threadCount.y * threadCount.x * threadIdxZ;
            uint threadIdxY = i / threadCount.x;
            i -= threadIdxY * threadCount.x;
            uint threadIdxX = i;
            module(threadCount, make_uint3(threadIdxX, threadIdxY, threadIdxZ), vec.data());
        },
        threadCount.x * threadCount.y * threadCount.z,
        true);
    handle.Complete();
    return handle;
    //exportFunc(sz.x, sz.y, sz.z, (uint64)vec.data());
}
}// namespace lc::ispc