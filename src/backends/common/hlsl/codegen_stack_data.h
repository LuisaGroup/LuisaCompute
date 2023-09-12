#pragma once
#include <luisa/vstl/common.h>
#include "hlsl_codegen.h"
#include "struct_generator.h"
#include "access_chain.h"
namespace lc::hlsl {

struct CodegenStackData : public vstd::IOperatorNewBase {
    CodegenUtility *util;
    vstd::StringBuilder *incrementalFunc;
    luisa::compute::Function kernel;
    vstd::unordered_map<Type const *, uint64> structTypes;
    vstd::unordered_map<uint64, uint64> constTypes;
    vstd::unordered_map<uint64_t /* hash */, uint64> funcTypes;
    vstd::HashMap<Type const *, vstd::unique_ptr<StructGenerator>> customStruct;
    vstd::unordered_map<uint, uint> arguments;
    vstd::unordered_map<Type const *, vstd::string> internalStruct;
    enum class FuncType : uint8_t {
        Kernel,
        Vert,
        Pixel,
        Callable
    };
    FuncType funcType;
    bool isRaster = false;
    bool isPixelShader = false;
    bool pixelFirstArgIsStruct = false;
    uint64 count = 0;
    uint64 constCount = 0;
    uint64 funcCount = 0;
    uint64 tempCount = 0;
    bool useTex2DBindless = false;
    bool useTex3DBindless = false;
    bool useBufferBindless = false;
    uint64 structCount = 0;
    uint64 argOffset = 0;
    int64_t appdataId = -1;
    int64 scopeCount = -1;

    vstd::function<void(Type const *)> generateStruct;
    vstd::unordered_map<vstd::string, vstd::string, vstd::hash<vstd::StringBuilder>> structReplaceName;
    vstd::unordered_map<uint64, Variable> sharedVariable;
    vstd::unordered_set<AccessChain, AccessHash> atomicsFuncs;
    Expression const *tempSwitchExpr;
    size_t tempSwitchCounter = 0;
    CodegenStackData();
    AccessChain const &GetAtomicFunc(
        CallOp op,
        Variable const &rootVar,
        Type const *retType,
        luisa::span<Expression const *const> exprs);
    void Clear();
    vstd::string_view CreateStruct(Type const *t);
    std::pair<uint64, bool> GetConstCount(uint64 data);
    uint64 GetFuncCount(Function f);
    uint64 GetTypeCount(Type const *t);
    ~CodegenStackData();
    static vstd::unique_ptr<CodegenStackData> Allocate(CodegenUtility *util);
    static void DeAllocate(vstd::unique_ptr<CodegenStackData> &&v);
    // static bool& ThreadLocalSpirv();
};
}// namespace lc::hlsl
