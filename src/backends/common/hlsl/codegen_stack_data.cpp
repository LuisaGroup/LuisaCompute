#include "codegen_stack_data.h"
#include <luisa/runtime/rtx/ray.h>
#include <luisa/runtime/rtx/hit.h>
#include <luisa/ast/type_registry.h>
namespace lc::hlsl {
CodegenStackData::CodegenStackData()
    : generateStruct(
          [this](Type const *t) {
              CreateStruct(t);
          }),
      generateAliasedStruct(
          [this](Type const *t) {
              CreateAliasedStruct(t);
          }) {
    Clear();
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
    customStructVector.clear();
    customStructAliased.clear();
    customStructVectorAliased.clear();
    originToAliasedTypes.clear();
    aliasedToOriginTypes.clear();
    atomicsFuncs.clear();
    sharedVariable.clear();
    printer.clear();
    constCount = 0;
    argOffset = 0;
    appdataId = -1;
    count = 0;
    structCount = 0;
    funcCount = 0;
    tempCount = 0;
    useTex2DBindless = false;
    useTex3DBindless = false;
    useBufferBindless = false;
    atomicFloatToInt = false;
    internalStruct.clear();
    internalStruct.emplace(Type::of<CommittedHit>(), "_Hit0");
    internalStruct.emplace(Type::of<TriangleHit>(), "_Hit1");
    internalStruct.emplace(Type::of<ProceduralHit>(), "_Hit2");
}

std::pair<vstd::string_view, bool> CodegenStackData::CreateAliasedStruct(Type const *t) {
    if (!util->TypeIsAliased(t)) {
        return {CreateStruct(t), false};
    }
    auto ite = customStructAliased.try_emplace(
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
        newPtr->InitAliased(generateAliasedStruct, isSpirv);
        customStructVectorAliased.emplace_back(ite.first.value().get());
    }
    return {ite.first.value()->GetStructName(), true};
}

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
        newPtr->Init(generateStruct, isSpirv);
        customStructVector.emplace_back(ite.first.value().get());
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

uint64 CodegenStackData::GetFuncCount(Function f) {
    auto ite = funcTypes.try_emplace(
        f.hash(),
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
            return ite;
        }
        return vstd::unique_ptr<CodegenStackData>(new CodegenStackData());
    }
    void DeAllocate(vstd::unique_ptr<CodegenStackData> &&v) {
        std::lock_guard lck(mtx);
        v->Clear();
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
// # for type, $ for access, @ for arguments
static vstd::string_view _atomic_exchange =
    R"(# r;InterlockedExchange($,@,r);return r;)"sv;
static vstd::string_view _atomic_compare_exchange =
    R"(# r;InterlockedCompareExchange($,@,r);return r;)"sv;
static vstd::string_view _atomic_compare_exchange_float =
    R"(# r;InterlockedCompareExchangeFloatBitwise($,@,r);return r;)"sv;
static vstd::string_view _atomic_compare_exchange_float_spirv =
    R"(# r;InterlockedCompareExchange($,asint(@),r);return asfloat(r);)"sv;
static vstd::string_view _atomic_add =
    R"(# r;InterlockedAdd($,@,r);return r;)"sv;
static vstd::string_view _atomic_add_float =
    R"(while(true){
# old=$;
# r;
InterlockedCompareExchangeFloatBitwise($,old,old+@,r);
if(old==r)return old;
})"sv;
static vstd::string_view _atomic_add_float_spirv =
    R"(while(true){
# old=asint($);
# r;
InterlockedCompareExchange($,old,asint(asfloat(old)+@),r);
if(old==r)return asfloat(old);
})"sv;
static vstd::string_view _atomic_sub =
    R"(# r;
InterlockedAdd($,-@,r);
return r;)"sv;
static vstd::string_view _atomic_sub_float =
    R"(while(true){
# old=$;
# r;
InterlockedCompareExchangeFloatBitwise($,old,old-@,r);
if(old==r)return old;
})"sv;
static vstd::string_view _atomic_sub_float_spirv =
    R"(while(true){
# old=asint($);
# r;
InterlockedCompareExchange($,old,asint(asfloat(old)-@),r);
if(old==r)return asfloat(old);
})"sv;
static vstd::string_view _atomic_and =
    R"(# r;InterlockedAnd($,@,r);return r;)"sv;
static vstd::string_view _atomic_or =
    R"(# r;InterlockedOr($,@,r);return r;)"sv;
static vstd::string_view _atomic_xor =
    R"(# r;InterlockedXor($,@,r);return r;)"sv;
static vstd::string_view _atomic_min =
    R"(# r;InterlockedMin($,@,r);return r;)"sv;
static vstd::string_view _atomic_min_float =
    R"(while(true){
# old=$;
if(old<=@){
# r;
InterlockedCompareExchangeFloatBitwise($,old,@,r);
if(r==old) return old;
}})"sv;
static vstd::string_view _atomic_min_float_spirv =
    R"(while(true){
# old=asint($);
if(asfloat(old)<=@){
# r;
InterlockedCompareExchange($,old,asint(@),r);
if(r==old) return asfloat(old);
}})"sv;
static vstd::string_view _atomic_max =
    R"(# r;InterlockedMax($,@,r);return r;)"sv;
static vstd::string_view _atomic_max_float =
    R"(while(true){
# old=$;
if(old>=@){
# r;
InterlockedCompareExchangeFloatBitwise($,old,@,r);
if(r==old) return old;
}})"sv;
static vstd::string_view _atomic_max_float_spirv =
    R"(while(true){
# old=asint($);
if(asfloat(old)>=@){
# r;
InterlockedCompareExchange($,old,asint(@),r);
if(r==old) return asfloat(old);
}})"sv;
AccessChain const &CodegenStackData::GetAtomicFunc(
    CallOp op,
    Variable const &rootVar,
    Type const *retType,
    luisa::span<Expression const *const> exprs) {
    size_t extra_arg_size = (op == CallOp::ATOMIC_COMPARE_EXCHANGE) ? 2 : 1;
    vstd::StringBuilder retTypeName;
    if (atomicFloatToInt && (retType->is_float32() || retType->is_float64())) {
        if (retType->is_float32()) {
            util->GetTypeName(*Type::of<int>(), retTypeName, Usage::NONE, true);
        } else {
            util->GetTypeName(*Type::of<int64_t>(), retTypeName, Usage::NONE, true);
        }
    } else {
        util->GetTypeName(*retType, retTypeName, Usage::NONE, true);
    }
    TemplateFunction tmp{
        .ret_type = retTypeName.view(),
        .tmp_type_name = retTypeName.view(),
        .access_place = '$',
        .args_place = '@',
        .temp_type_place = '#'};
    switch (op) {
        case CallOp::ATOMIC_EXCHANGE:
            tmp.body = _atomic_exchange;
            break;
        case CallOp::ATOMIC_COMPARE_EXCHANGE:
            if (retType->is_float32()) {
                tmp.body = isSpirv ? _atomic_compare_exchange_float_spirv : _atomic_compare_exchange_float;
            } else {
                tmp.body = _atomic_compare_exchange;
            }
            break;
        case CallOp::ATOMIC_FETCH_ADD:
            if (retType->is_float32()) {
                tmp.body = isSpirv ? _atomic_add_float_spirv : _atomic_add_float;
            } else {
                tmp.body = _atomic_add;
            }
            break;
        case CallOp::ATOMIC_FETCH_SUB:
            if (retType->is_float32()) {
                tmp.body = isSpirv ? _atomic_sub_float_spirv : _atomic_sub_float;
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
                tmp.body = isSpirv ? _atomic_min_float_spirv : _atomic_min_float;
            } else {
                tmp.body = _atomic_min;
            }
            break;
        case CallOp::ATOMIC_FETCH_MAX:
            if (retType->is_float32()) {
                tmp.body = isSpirv ? _atomic_max_float_spirv : _atomic_max_float;
            } else {
                tmp.body = _atomic_max;
            }
            break;
        default:
            LUISA_ERROR_WITH_LOCATION("Invalid atomic operator.");
    }

    AccessChain chain{
        op,
        rootVar,
        exprs.subspan(0, exprs.size() - extra_arg_size)};
    auto iter = atomicsFuncs.emplace(std::move(chain));
    if (iter.second) {
        auto &access_chain = const_cast<AccessChain &>(*iter.first);
        access_chain.init_name();
        access_chain.gen_func_impl(util, tmp, exprs.subspan(exprs.size() - extra_arg_size, extra_arg_size), *incrementalFunc);
    }
    return *iter.first;
}

}// namespace lc::hlsl
