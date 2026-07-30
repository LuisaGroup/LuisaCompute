#pragma once
#include <luisa/vstl/common.h>
#include <luisa/ast/function_builder.h>
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
    vstd::unordered_map<uint64_t /* hash */, std::pair<uint64, luisa::string>> funcTypes;
    vstd::vector<StructGenerator *> customStructVector;
    vstd::HashMap<Type const *, vstd::unique_ptr<StructGenerator>> customStruct;
    vstd::vector<StructGenerator *> customStructVectorAliased;
    vstd::HashMap<Type const *, vstd::unique_ptr<StructGenerator>> customStructAliased;
    vstd::HashMap<luisa::compute::detail::FunctionBuilder const *, vstd::unordered_set<uint>> globallyCoherentBuffers;
    vstd::unordered_set<Type const *> originToAliasedTypes;
    vstd::unordered_set<Type const *> aliasedToOriginTypes;
    vstd::unordered_map<uint, uint> arguments;
    vstd::unordered_map<Type const *, vstd::string> internalStruct;
    vstd::vector<std::pair<vstd::string, Type const *>> printer;
    enum class FuncType : uint8_t {
        Kernel,
        Vert,
        Pixel,
        Callable
    };
    enum class CondOptValue : uint32_t {
        None = 0,
        Flatten = 1,
        Branch = 2,
        ForceCase = 4
    };
    FuncType funcType;
    CondOptValue cond_opt_value = CondOptValue::None;
    bool isRaster : 1 = false;
    bool isRayTracing : 1 = false;
    bool isSpirv : 1 = false;
    bool noRegister : 1 = false;
    bool isPixelShader : 1 = false;
    bool pixelFirstArgIsStruct : 1 = false;
    bool pixelUseBarycentric : 1 = false;
    bool useTex2DBindless : 1 = false;
    bool useTex3DBindless : 1 = false;
    bool useBufferBindless : 1 = false;
    bool enable_debug_info : 1 = false;
    bool use_8bit : 1 = false;
    // Key: pair of (function hash, variable uid) → validate index in _validate_N array
    struct ValidateKey {
        uint64_t func_hash;
        uint32_t uid;
        bool operator==(ValidateKey const &o) const noexcept { return func_hash == o.func_hash && uid == o.uid; }
    };
    struct ValidateKeyHash {
        uint64_t operator()(ValidateKey const &k) const noexcept {
            return luisa::hash<uint64_t>{}(k.func_hash, k.uid);
        }
    };
    luisa::unordered_map<ValidateKey, uint32_t, ValidateKeyHash> validate_index_map;
    uint64 count = 0;
    uint64 constCount = 0;
    uint64 funcCount = 0;
    uint64 tempCount = 0;
    uint64 structCount = 0;
    uint64 argOffset = 0;
    int64_t appdataId = -1;
    int64 scopeCount = -1;

    vstd::function<void(Type const *)> generateStruct;
    vstd::function<void(Type const *)> generateAliasedStruct;
    SharedVarSet sharedVariable;
    vstd::unordered_set<AccessChain, AccessHash> atomicsFuncs;
    Expression const *tempSwitchExpr;
    size_t tempSwitchCounter = 0;
    CodegenStackData();
    AccessChain const &GetAtomicFunc(
        Function func,
        CallOp op,
        Variable const &rootVar,
        Type const *retType,
        luisa::span<Expression const *const> exprs);
    void Clear();
    vstd::string_view CreateStruct(Type const *t);
    std::pair<vstd::string_view, bool> CreateAliasedStruct(Type const *t);
    std::pair<uint64, bool> GetConstCount(uint64 data);
    std::pair<uint64, luisa::string> const &GetFuncCountAndName(Function f);
    uint64 GetTypeCount(Type const *t);
    ~CodegenStackData();
    static vstd::unique_ptr<CodegenStackData> Allocate(CodegenUtility *util);
    static void DeAllocate(vstd::unique_ptr<CodegenStackData> &&v);
};
}// namespace lc::hlsl
