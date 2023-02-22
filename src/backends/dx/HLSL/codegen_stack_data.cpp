#include "codegen_stack_data.h"
#include <runtime/rtx/ray.h>
#include <runtime/rtx/hit.h>
#include <iostream>
#include <ast/type_registry.h>
namespace toolhub::directx {
CodegenStackData::CodegenStackData()
    : generateStruct(
          [this](Type const *t) {
              return CreateStruct(t);
          }) {
    structReplaceName.try_emplace(
        "float3"sv, "float4"sv);
    structReplaceName.try_emplace(
        "int3"sv, "int4"sv);
    structReplaceName.try_emplace(
        "uint3"sv, "uint4"sv);
}
void CodegenStackData::Clear() {
    rayDesc = nullptr;
    hitDesc = nullptr;
    tempSwitchExpr = nullptr;
    arguments.clear();
    scopeCount = -1;
    tempSwitchCounter = 0;
    structTypes.clear();
    constTypes.clear();
    funcTypes.clear();
    customStruct.clear();
    customStructVector.clear();
    sharedVariable.clear();
    constCount = 0;
    argOffset = 0;
    appdataId = -1;
    count = 0;
    structCount = 0;
    funcCount = 0;
    tempCount = 0;
    bindlessBufferCount = 0;
}
void CodegenStackData::AddBindlessType(Type const *type) {
    bindlessBufferCount = 1;
}
/*
static thread_local bool gIsCodegenSpirv = false;
bool &CodegenStackData::ThreadLocalSpirv() {
    return gIsCodegenSpirv;
}*/

StructGenerator *CodegenStackData::CreateStruct(Type const *t) {
    bool isHitType = (t == Type::of<Hit>());
    StructGenerator *newPtr;
    auto ite = customStruct.try_emplace(
        t,
        vstd::lazy_eval([&] {
            newPtr = new StructGenerator(
                t,
                structCount++);
            return vstd::create_unique(newPtr);
        }));
    if (!ite.second) {
        return ite.first->second.get();
    } else {
        ite.first->second->Init(generateStruct);
    }
    if (isHitType) {
        hitDesc = newPtr;
        newPtr->SetStructName(vstd::string("RayPayload"sv));
    } else {
        customStructVector.emplace_back(newPtr);
    }
    return newPtr;
}
std::pair<uint64, bool> CodegenStackData::GetConstCount(uint64 data) {
    bool newValue = false;
    auto ite = constTypes.try_emplace(
        data,
        vstd::lazy_eval(
            [&] {
                newValue = true;
                return constCount++;
            }));
    return {ite.first->second, newValue};
}

uint64 CodegenStackData::GetFuncCount(void const* data) {
    auto ite = funcTypes.try_emplace(
        data,
        vstd::lazy_eval(
            [&] {
                return funcCount++;
            }));
    return ite.first->second;
}
uint64 CodegenStackData::GetTypeCount(Type const *t) {
    auto ite = structTypes.try_emplace(
        t,
        vstd::lazy_eval(
            [&] {
                return count++;
            }));
    return ite.first->second;
}
namespace detail {

struct CodegenGlobalPool {
    std::mutex mtx;
    vstd::vector<vstd::unique_ptr<CodegenStackData>> allCodegen;
    vstd::unique_ptr<CodegenStackData> Allocate() {
        std::lock_guard lck(mtx);
        if (!allCodegen.empty()) {
            auto ite = std::move(allCodegen.back());
            allCodegen.pop_back();
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
void CodegenStackData::DeAllocate(vstd::unique_ptr<CodegenStackData> &&v) {
    detail::codegenGlobalPool.DeAllocate(std::move(v));
}
}// namespace toolhub::directx