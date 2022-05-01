
#include <Codegen/CodegenStackData.h>
namespace toolhub::directx {
CodegenStackData::CodegenStackData()
    : generateStruct(
          [this](Type const *t) {
              return CreateStruct(t);
          }) {
    structReplaceName.Emplace(
        "float3"sv, "float4"sv);
    structReplaceName.Emplace(
        "int3"sv, "int4"sv);
    structReplaceName.Emplace(
        "uint3"sv, "uint4"sv);
}
void CodegenStackData::Clear() {
    rayDesc = nullptr;
    hitDesc = nullptr;
    arguments.Clear();
    scopeCount = -1;
    structTypes.Clear();
    constTypes.Clear();
    funcTypes.Clear();
    bindlessBufferTypes.Clear();
    customStruct.Clear();
    customStructVector.clear();
    generatedConstants.Clear();
    sharedVariable.clear();
    constCount = 0;
    count = 0;
    structCount = 0;
    funcCount = 0;
    tempCount = 0;
    bindlessBufferCount = 0;
}
uint CodegenStackData::AddBindlessType(Type const *type) {
    return bindlessBufferTypes
        .Emplace(
            type,
            vstd::MakeLazyEval([&] {
                return bindlessBufferCount++;
            }))
        .Value();
}
StructGenerator *CodegenStackData::CreateStruct(Type const *t) {
    bool isRayType = t->description() == CodegenUtility::rayTypeDesc;
    bool isHitType = t->description() == CodegenUtility::hitTypeDesc;
    auto ite = customStruct.Find(t);
    if (ite) {
        return ite.Value().get();
    }
    auto newPtr = new StructGenerator(
        t,
        structCount++,
        generateStruct);
    customStruct.ForceEmplace(
        t,
        vstd::create_unique(newPtr));

    if (isRayType) {
        rayDesc = newPtr;
        newPtr->SetStructName(vstd::string("LCRayDesc"sv));
    } else if (isHitType) {
        hitDesc = newPtr;
        newPtr->SetStructName(vstd::string("RayPayload"sv));
    } else {
        customStructVector.emplace_back(newPtr); 
    }
    return newPtr;
}
uint64 CodegenStackData::GetConstCount(uint64 data) {
    auto ite = constTypes.Emplace(
        data,
        vstd::MakeLazyEval(
            [&] {
                return constCount++;
            }));
    return ite.Value();
}
uint64 CodegenStackData::GetFuncCount(uint64 data) {
    auto ite = funcTypes.Emplace(
        data,
        vstd::MakeLazyEval(
            [&] {
                return funcCount++;
            }));
    return ite.Value();
}
uint64 CodegenStackData::GetTypeCount(Type const *t) {
    auto ite = structTypes.Emplace(
        t,
        vstd::MakeLazyEval(
            [&] {
                return count++;
            }));
    return ite.Value();
}
namespace detail {

struct CodegenGlobalPool {
    std::mutex mtx;
    vstd::vector<vstd::unique_ptr<CodegenStackData>> allCodegen;
    vstd::unique_ptr<CodegenStackData> Allocate() {
        std::lock_guard lck(mtx);
        if (!allCodegen.empty()) {
            auto ite = allCodegen.erase_last();
            ite->Clear();
            return ite;
        }
        return vstd::unique_ptr<CodegenStackData>(new CodegenStackData());
    }
    void DeAllocate(vstd::unique_ptr<CodegenStackData> &&v) {
        std::lock_guard lck(mtx);
        allCodegen.emplace_back(std::move(v));
    }
};
static CodegenGlobalPool codegenGlobalPool;
}// namespace detail
CodegenStackData::~CodegenStackData() {}
vstd::unique_ptr<CodegenStackData> CodegenStackData::Allocate() {
    return detail::codegenGlobalPool.Allocate();
}
void CodegenStackData::DeAllocate(vstd::unique_ptr<CodegenStackData>&& v) {
    detail::codegenGlobalPool.DeAllocate(std::move(v));
}
}// namespace toolhub::directx