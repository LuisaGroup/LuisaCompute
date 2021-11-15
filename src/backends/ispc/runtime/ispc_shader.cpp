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
    auto blockSize = func.block_size();
    auto blockCount = (sz + blockSize - 1u) / blockSize;
    auto totalCount = blockCount.x * blockCount.y * blockCount.z;
    auto sharedCounter = luisa::make_shared<std::atomic_uint>(0u);
    auto handle = tPool->GetParallelTask(
        [=](size_t) {
            auto &&counter = *sharedCounter;
            for (auto i = counter.fetch_add(1u); i < totalCount; i = counter.fetch_add(1u)) {
                uint blockIdxZ = i / (blockCount.y * blockCount.x);
                i -= blockCount.y * blockCount.x * blockIdxZ;
                uint blockIdxY = i / blockCount.x;
                i -= blockIdxY * blockCount.x;
                uint blockIdxX = i;
                module(blockCount, make_uint3(blockIdxX, blockIdxY, blockIdxZ), vec.data());
            }
        },
        std::thread::hardware_concurrency(),
        true);
    handle.Execute();
    return handle;
    //exportFunc(sz.x, sz.y, sz.z, (uint64)vec.data());
}
}// namespace lc::ispc