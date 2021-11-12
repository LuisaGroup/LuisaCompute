#pragma once
#include <vstl/Common.h>
#include <core/dynamic_module.h>
#include <vstl/ThreadPool.h>
namespace lc::ispc {
using namespace luisa;
class Shader {
private:
    DynamicModule dllModule;
    Function func;
    using FuncType = void(
        uint, // thd_cX,
        uint, //thd_cY,
        uint, //thd_cZ,
        uint, //blk_cX,
        uint, //blk_cY,
        uint, //blk_cZ,
        uint, // thd_idX,
        uint, //thd_idY,
        uint, // thd_idZ,
        uint64// arg
    );
    vstd::funcPtr_t<FuncType> exportFunc;
    vstd::HashMap<uint, uint> varIdToArg;

public:
    using ArgVector = vstd::vector<uint8_t, VEngine_AllocType::VEngine, true, 32>;
    Shader(
        Function func,
        std::string const &str);
    size_t GetArgIndex(uint varID) const;
    static constexpr size_t CalcAlign(size_t value, size_t align) {
        return (value + (align - 1)) & ~(align - 1);
    }
    template<typename T>
    static void PackArg(
        ArgVector &vec,
        T const &v) {
        size_t sz = CalcAlign(vec.size(), alignof(T));
        vec.resize(sz + sizeof(T));
        memcpy(vec.data() + sz, &v, sizeof(T));
    }
    static void PackArr(
        ArgVector &vec,
        void const *ptr,
        size_t arrSize,
        size_t arrAlign) {
        size_t sz = CalcAlign(vec.size(), arrAlign);
        vec.resize(sz + arrSize);
        memcpy(vec.data() + sz, ptr, arrSize);
    }
    ThreadTaskHandle dispatch(
        ThreadPool* tPool,
        uint3 sz,
        ArgVector const &vec) const;
};
}// namespace lc::ispc