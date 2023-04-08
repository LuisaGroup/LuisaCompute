#include "codegen_stack_data.h"
#include <runtime/rtx/ray.h>
#include <runtime/rtx/hit.h>
#include <ast/type_registry.h>
namespace lc::dx {
CodegenStackData::CodegenStackData()
    : generateStruct(
          [this](Type const *t) {
              CreateStruct(t);
          }) {
    structReplaceName.try_emplace(
        "float3"sv, "float4"sv);
    structReplaceName.try_emplace(
        "int3"sv, "int4"sv);
    structReplaceName.try_emplace(
        "uint3"sv, "uint4"sv);
    internalStruct.emplace(Type::of<CommittedHit>(), "Hit0");
    internalStruct.emplace(Type::of<TriangleHit>(), "Hit1");
    internalStruct.emplace(Type::of<ProceduralHit>(), "Hit2");
}
void CodegenStackData::Clear() {
    tempSwitchExpr = nullptr;
    arguments.clear();
    scopeCount = -1;
    tempSwitchCounter = 0;
    structTypes.clear();
    constTypes.clear();
    funcTypes.clear();
    customStruct.clear();
    atomicsFuncs.clear();
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

vstd::string_view CodegenStackData::CreateStruct(Type const *t) {
    auto iter = internalStruct.find(t);
    if (iter != internalStruct.end())
        return iter->second;
    auto ite = customStruct.try_emplace(
        t,
        vstd::lazy_eval([&] {
            auto newPtr = new StructGenerator(
                t,
                structCount++,
                util);
            return vstd::create_unique(newPtr);
        }));
    if (ite.second) {
        auto newPtr = ite.first.value().get();
        newPtr->Init(generateStruct);
    }
    return ite.first.value()->GetStructName();
}
std::pair<uint64, bool> CodegenStackData::GetConstCount(uint64 data) {
    auto ite = constTypes.try_emplace(
        data,
        vstd::lazy_eval(
            [&] {
                return constCount++;
            }));
    return {ite.first->second, ite.second};
}

uint64 CodegenStackData::GetFuncCount(void const *data) {
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
vstd::unique_ptr<CodegenStackData> CodegenStackData::Allocate(CodegenUtility *util) {
    auto ptr = detail::codegenGlobalPool.Allocate();
    ptr->util = util;
    return ptr;
}
void CodegenStackData::DeAllocate(vstd::unique_ptr<CodegenStackData> &&v) {
    detail::codegenGlobalPool.DeAllocate(std::move(v));
}
// T for type, C for access, A for arguments
static vstd::string_view _atomic_exchange =
    R"(T r;InterlockedExchange(C,A,r);return r;)"sv;
static vstd::string_view _atomic_compare_exchange =
    R"(T r;InterlockedCompareExchange(C,A,r);return r;)"sv;
static vstd::string_view _atomic_compare_exchange_float =
    R"(T r;InterlockedCompareExchangeFloatBitwise(C,A,r);return r;)"sv;
static vstd::string_view _atomic_add =
    R"(T r;InterlockedAdd(C,A,r);return r;)"sv;
static vstd::string_view _atomic_add_float =
    R"(while(true){
T old=C;
T r;
InterlockedCompareExchangeFloatBitwise(C,old,old+A,r);
if(old==r)return old;
}
})"sv;
static vstd::string_view _atomic_sub =
    R"(T r;
InterlockedAdd(C,-A,r);
return r;)"sv;
static vstd::string_view _atomic_sub_float =
    R"(while(true){
T old=C;
T r;
InterlockedCompareExchangeFloatBitwise(C,old,old-A,r);
if(old==r)return old;
}
})"sv;
static vstd::string_view _atomic_and =
    R"(T r;InterlockedAnd(a[idx],-b,r);return r;)"sv;
static vstd::string_view _atomic_or =
    R"(T r;InterlockedOr(a[idx],-b,r);return r;)"sv;
static vstd::string_view _atomic_xor =
    R"(T r;InterlockedXor(a[idx],-b,r);return r;)"sv;
static vstd::string_view _atomic_min =
    R"(T r;InterlockedMin(a[idx],-b,r);return r;)"sv;
static vstd::string_view _atomic_min_float =
    R"(while(true){
T old=C;
if(old<=A){
T r;
InterlockedCompareExchangeFloatBitwise(C,old,A,r);
if(r==old) return old;
}})"sv;
static vstd::string_view _atomic_max =
    R"(T r;InterlockedMax(a[idx],-b,r);return r;)"sv;
static vstd::string_view _atomic_max_float =
    R"(while(true){
T old=C;
if(old>=A){
T r;
InterlockedCompareExchangeFloatBitwise(C,old,A,r);
if(r==old) return old;
}})"sv;
AccessChain const &CodegenStackData::GetAtomicFunc(
    CallOp op,
    Type const *rootType,
    Type const *retType,
    luisa::span<Expression const *const> exprs,
    vstd::StringBuilder &beforeCodeBlockSB) {
    size_t extra_arg_size = (op == CallOp::ATOMIC_COMPARE_EXCHANGE) ? 2 : 1;
    vstd::StringBuilder retTypeName;
    util->GetTypeName(*retType, retTypeName, Usage::NONE, true);
    TemplateFunction tmp{
        .ret_type = retTypeName.view(),
        .access_place = 'C',
        .args_place = 'A'};
    switch (op) {
        case CallOp::ATOMIC_EXCHANGE:
            tmp.body = _atomic_exchange;
            break;
        case CallOp::ATOMIC_COMPARE_EXCHANGE:
            if (retType->is_float32()) {
                tmp.body = _atomic_compare_exchange_float;
            } else {
                tmp.body = _atomic_compare_exchange;
            }
            break;
        case CallOp::ATOMIC_FETCH_ADD:
            if (retType->is_float32()) {
                tmp.body = _atomic_add_float;
            } else {
                tmp.body = _atomic_add;
            }
            break;
        case CallOp::ATOMIC_FETCH_SUB:
            if (retType->is_float32()) {
                tmp.body = _atomic_sub_float;
            } else {
                tmp.body = _atomic_sub;
            }
            break;
        case CallOp::ATOMIC_FETCH_AND:
            tmp.body = _atomic_and;
            break;
        case CallOp::ATOMIC_FETCH_OR:
            tmp.body = _atomic_or;
            break;
        case CallOp::ATOMIC_FETCH_XOR:
            tmp.body = _atomic_xor;
            break;
        case CallOp::ATOMIC_FETCH_MIN:
            if (retType->is_float32()) {
                tmp.body = _atomic_min_float;
            } else {
                tmp.body = _atomic_min;
            }
            break;
        case CallOp::ATOMIC_FETCH_MAX:
            if (retType->is_float32()) {
                tmp.body = _atomic_max_float;
            } else {
                tmp.body = _atomic_max;
            }
            break;
    }

    AccessChain chain{
        op,
        rootType,
        exprs.subspan(0, exprs.size() - extra_arg_size)};
    auto iter = atomicsFuncs.emplace(std::move(chain));
    if (iter.second) {
        iter.first->init_name();
        iter.first->gen_func_impl(util, tmp, exprs.subspan(exprs.size() - extra_arg_size, extra_arg_size), beforeCodeBlockSB);
    }
    return *iter.first;
}

}// namespace lc::dx