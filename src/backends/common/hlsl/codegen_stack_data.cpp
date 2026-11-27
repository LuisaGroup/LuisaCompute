#include "codegen_stack_data.h"
#include "atomic_codegen_policy.h"
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
          }),
      util(nullptr),
      incrementalFunc(nullptr),
      funcType(FuncType::Kernel),
      tempSwitchExpr(nullptr) {
    Clear();
}
void CodegenStackData::Clear() {
    tempSwitchExpr = nullptr;
    arguments.clear();
    validate_index_map.clear();
    promoted_resource_args.clear();
    scopeCount = -1;
    tempSwitchCounter = 0;
    structTypes.clear();
    constTypes.clear();
    funcTypes.clear();
    structPacking.clear();
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
    enable_fast_math = false;
    use_8bit = false;
    use_atomic64 = false;
    emulate_8bit = false;
    default_warp_size.reset();
    pixelUseBarycentric = false;
    pixelUseFrontFacing = false;
    internalStruct.clear();
    internalStruct.emplace(Type::of<CommittedHit>(), "_Hit0");
    internalStruct.emplace(Type::of<TriangleHit>(), "_Hit1");
    internalStruct.emplace(Type::of<ProceduralHit>(), "_Hit2");
    globallyCoherentBuffers.clear();
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

StructPacking const &CodegenStackData::GetStructPacking(Type const *t) {
    auto ite = structPacking.try_emplace(
        t, vstd::lazy_eval([&] {
            // `vstd::make_unique` allocates through vengine, which is what the
            // vstd deleter frees with.
            return vstd::make_unique<StructPacking>(
                plan_struct_packing(t, bf16_fallback, emulate_8bit));
        }));
    return *ite.first.value();
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
            [this] {
                return constCount++;
            }));
    return {ite.first->second, ite.second};
}
std::pair<uint64, luisa::string> const &CodegenStackData::GetFuncCountAndName(Function f) {
    auto ite = funcTypes.try_emplace(
        f.hash(),
        vstd::lazy_eval(
            [this, &f] {
                return std::pair<uint64, luisa::string>{
                    funcCount++,
                    f.name()};
            }));
    auto &value = ite.first->second;
    if (value.second.empty() && (!f.name().empty())) {
        value.second = f.name();
    }
    return value;
}
uint64 CodegenStackData::GetTypeCount(Type const *t) {
    auto ite = structTypes.try_emplace(
        t,
        vstd::lazy_eval(
            [this] {
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
        return vstd::make_unique<CodegenStackData>();
    }
    void DeAllocate(vstd::unique_ptr<CodegenStackData> &&v) {
        std::lock_guard lck(mtx);
        v->Clear();
        allCodegen.emplace_back(std::move(v));
    }
};
namespace codegen_stack_data_detail_2 {
static CodegenGlobalPool codegenGlobalPool;
}  // namespace codegen_stack_data_detail_2
}// namespace detail
CodegenStackData::~CodegenStackData() = default;
vstd::unique_ptr<CodegenStackData> CodegenStackData::Allocate(CodegenUtility *util) {
    auto ptr = detail::codegen_stack_data_detail_2::codegenGlobalPool.Allocate();
    ptr->util = util;
    return ptr;
}
void CodegenStackData::DeAllocate(vstd::unique_ptr<CodegenStackData> &&v) {
    detail::codegen_stack_data_detail_2::codegenGlobalPool.DeAllocate(std::move(v));
}
// # for type, $ for access, @ for arguments
namespace codegen_stack_data_detail {
static vstd::string_view _atomic_exchange =
    R"(# r;InterlockedExchange($,@,r);return r;)"sv;
static vstd::string_view _atomic_compare_exchange =
    R"(# r;InterlockedCompareExchange($,@,r);return r;)"sv;
static vstd::string_view _atomic_compare_exchange_float =
    R"(# r;InterlockedCompareExchangeFloatBitwise($,@,r);return r;)"sv;
static vstd::string_view _atomic_add =
    R"(# r;InterlockedAdd($,@,r);return r;)"sv;
static vstd::string_view _atomic_add_float =
    R"(# old=$;
while(true){
# r;
InterlockedCompareExchangeFloatBitwise($,old,old+@,r);
if(asuint(old)==asuint(r))return old;
old=r;
})"sv;
static vstd::string_view _atomic_sub =
    R"(# r;
InterlockedAdd($,-@,r);
return r;)"sv;
static vstd::string_view _atomic_sub_float =
    R"(# old=$;
while(true){
# r;
InterlockedCompareExchangeFloatBitwise($,old,old-@,r);
if(asuint(old)==asuint(r))return old;
old=r;
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
    R"(# old=$;
while(true){
if(old<=@) return old;
# r;
InterlockedCompareExchangeFloatBitwise($,old,@,r);
if(asuint(r)==asuint(old)) return old;
old=r;
})"sv;
static vstd::string_view _atomic_max =
    R"(# r;InterlockedMax($,@,r);return r;)"sv;
// Float64 word forms: `$` is the uint64_t element the DX route declares for a
// scalar float64 buffer element, `@` the float64 operand, `#` a uint64_t temp.
static vstd::string_view _atomic_exchange_f64_word =
    R"(# r;InterlockedExchange($,_lc_f64_to_word(@),r);return _lc_f64_from_word(r);)"sv;
// The two operands need one conversion each, and the placeholder expansion
// cannot wrap them separately, so this body names the generated
// parameters directly: the access chain is a0 plus exactly one element
// index (a1), so the expected/desired operands are a2 and a3 (see the
// `float64_word_alias` shape check in GetAtomicFunc).
static vstd::string_view _atomic_compare_exchange_f64_word =
    R"(# r;InterlockedCompareExchange($,_lc_f64_to_word(a2),_lc_f64_to_word(a3),r);return _lc_f64_from_word(r);)"sv;
static vstd::string_view _atomic_add_f64_word =
    R"(# old=$;
while(true){
# r;
InterlockedCompareExchange($,old,_lc_f64_to_word(_lc_f64_from_word(old)+@),r);
if(old==r)return _lc_f64_from_word(old);
old=r;
})"sv;
static vstd::string_view _atomic_sub_f64_word =
    R"(# old=$;
while(true){
# r;
InterlockedCompareExchange($,old,_lc_f64_to_word(_lc_f64_from_word(old)-@),r);
if(old==r)return _lc_f64_from_word(old);
old=r;
})"sv;
static vstd::string_view _atomic_min_f64_word =
    R"(# old=$;
while(true){
if(_lc_f64_from_word(old)<=@) return _lc_f64_from_word(old);
# r;
InterlockedCompareExchange($,old,_lc_f64_to_word(@),r);
if(r==old) return _lc_f64_from_word(old);
old=r;
})"sv;
static vstd::string_view _atomic_max_f64_word =
    R"(# old=$;
while(true){
if(_lc_f64_from_word(old)>=@) return _lc_f64_from_word(old);
# r;
InterlockedCompareExchange($,old,_lc_f64_to_word(@),r);
if(r==old) return _lc_f64_from_word(old);
old=r;
})"sv;
static vstd::string_view _atomic_max_float =
    R"(# old=$;
while(true){
if(old>=@) return old;
# r;
InterlockedCompareExchangeFloatBitwise($,old,@,r);
if(asuint(r)==asuint(old)) return old;
old=r;
})"sv;
}  // namespace codegen_stack_data_detail
void CodegenUtility::CheckPackedAtomicTarget(
    luisa::span<Expression const *const> args) {
    for (auto arg : args) {
        if (arg == nullptr || arg->tag() != Expression::Tag::MEMBER) { continue; }
        auto member = static_cast<MemberExpr const *>(arg);
        if (member->is_swizzle()) { continue; }
        if (GetPackedStructMember(member->self()->type(), member->member_index())) {
            LUISA_ERROR_WITH_LOCATION(
                "Atomic access to a member of a packed struct word is not "
                "supported: a one-byte bool that shares a 32-bit word with 16-bit "
                "data is stored inside a single `uint`, so it has no address. Use "
                "a struct whose atomic member is 4-byte sized, or a separate "
                "buffer.");
        }
    }
}

AccessChain const &CodegenStackData::GetAtomicFunc(
    Function func,
    CallOp op,
    Variable const &rootVar,
    Type const *retType,
    luisa::span<Expression const *const> exprs) {
    util->CheckPackedAtomicTarget(exprs);
    if (retType->is_int64() || retType->is_uint64() || retType->is_float64()) {
        // Every 64-bit atomic needs DXIL 1.6: float64 is lowered through a
        // 64-bit integer view, so it needs the same SM 6.6 opcodes.
        use_atomic64 = true;
    }
    size_t extra_arg_size = (op == CallOp::ATOMIC_COMPARE_EXCHANGE) ? 2 : 1;
    vstd::StringBuilder retTypeName;
    util->GetTypeName(*retType, retTypeName, Usage::NONE, true);
    TemplateFunction tmp{
        .ret_type = retTypeName.view(),
        .tmp_type_name = retTypeName.view(),
        .access_place = '$',
        .args_place = '@',
        .temp_type_place = '#'};
    // A float64 atomic needs a 64-bit integer lvalue. The DX route declares a
    // scalar float64 buffer element as uint64_t (CodegenUtility::
    // IsFloat64WordElement), so only an atomic that addresses exactly one such
    // element has a word to update. Groupshared float64 storage and float64
    // members of a larger aggregate element have none.
    auto float64_word_alias =
        retType->is_float64() && !rootVar.is_shared() &&
        rootVar.type() != nullptr && rootVar.type()->is_buffer() &&
        rootVar.type()->element() != nullptr &&
        rootVar.type()->element()->is_float64() &&
        exprs.size() == extra_arg_size + 2u;// buffer + one element index
    // SPIR-V uses explicit atomic opcodes; HLSL float Interlocked overloads
    // are not consistently supported by the bundled Windows/Linux DXC builds.
    auto lowering = plan_hlsl_atomic_lowering(
        op, retType->is_float32(), retType->is_float64(),
        float64_word_alias, isSpirv);
    if (lowering == HlslAtomicLowering::UNSUPPORTED) [[unlikely]] {
        if (retType->is_float64()) {
            if (isSpirv) {
                LUISA_ERROR_WITH_LOCATION(
                    "Float64 atomic {} is unavailable in the HLSL-to-SPIR-V "
                    "route: DXC defines no float64 Interlocked* overload at "
                    "any shader model. Use native XIR-to-SPIR-V codegen for "
                    "this kernel.",
                    luisa::to_underlying(op));
            }
            if (rootVar.is_shared()) {
                LUISA_ERROR_WITH_LOCATION(
                    "Groupshared float64 atomic {} is not supported by the "
                    "HLSL backend: a groupshared allocation is not a 64-bit "
                    "integer word, and DXIL has no float64 Interlocked* "
                    "overload. Use a Buffer<double> atomic instead.",
                    luisa::to_underlying(op));
            }
            LUISA_ERROR_WITH_LOCATION(
                "Float64 atomic {} requires a scalar Buffer<double> element: "
                "the element is stored as one 64-bit integer word so SM 6.6 "
                "Interlocked* can update it, and a float64 member of a larger "
                "aggregate element has no such word. Split the float64 leaf "
                "into its own Buffer<double>.",
                luisa::to_underlying(op));
        }
        LUISA_ERROR_WITH_LOCATION(
            "Float atomic compare-exchange is unavailable in the "
            "HLSL-to-SPIR-V fallback because the bundled DXC only accepts "
            "the float CAS intrinsic on RWByteAddressBuffer. Use native "
            "XIR-to-SPIR-V codegen for this kernel.");
    }
    auto use_software_float_rmw =
        lowering == HlslAtomicLowering::FLOAT_CAS_LOOP;
    auto use_float64_word_cas =
        lowering == HlslAtomicLowering::FLOAT64_WORD_CAS_LOOP;
    if (retType->is_float64()) {
        // The word forms receive and return the uint64_t element.
        tmp.tmp_type_name = "uint64_t"sv;
    }
    switch (op) {
        case CallOp::ATOMIC_EXCHANGE:
            tmp.body = lowering == HlslAtomicLowering::FLOAT64_WORD_EXCHANGE ?
                           codegen_stack_data_detail::_atomic_exchange_f64_word :
                           codegen_stack_data_detail::_atomic_exchange;
            break;
        case CallOp::ATOMIC_COMPARE_EXCHANGE:
            tmp.body = lowering == HlslAtomicLowering::FLOAT64_WORD_COMPARE_EXCHANGE ?
                           codegen_stack_data_detail::_atomic_compare_exchange_f64_word :
                       lowering == HlslAtomicLowering::FLOAT_COMPARE_EXCHANGE ?
                           codegen_stack_data_detail::_atomic_compare_exchange_float :
                           codegen_stack_data_detail::_atomic_compare_exchange;
            break;
        case CallOp::ATOMIC_FETCH_ADD:
            tmp.body = use_float64_word_cas ?
                           codegen_stack_data_detail::_atomic_add_f64_word :
                       use_software_float_rmw ?
                           codegen_stack_data_detail::_atomic_add_float :
                           codegen_stack_data_detail::_atomic_add;
            break;
        case CallOp::ATOMIC_FETCH_SUB:
            tmp.body = use_float64_word_cas ?
                           codegen_stack_data_detail::_atomic_sub_f64_word :
                       use_software_float_rmw ?
                           codegen_stack_data_detail::_atomic_sub_float :
                           codegen_stack_data_detail::_atomic_sub;
            break;
        case CallOp::ATOMIC_FETCH_AND:
            tmp.body = codegen_stack_data_detail::_atomic_and;
            break;
        case CallOp::ATOMIC_FETCH_OR:
            tmp.body = codegen_stack_data_detail::_atomic_or;
            break;
        case CallOp::ATOMIC_FETCH_XOR:
            tmp.body = codegen_stack_data_detail::_atomic_xor;
            break;
        case CallOp::ATOMIC_FETCH_MIN:
            tmp.body = use_float64_word_cas ?
                           codegen_stack_data_detail::_atomic_min_f64_word :
                       use_software_float_rmw ?
                           codegen_stack_data_detail::_atomic_min_float :
                           codegen_stack_data_detail::_atomic_min;
            break;
        case CallOp::ATOMIC_FETCH_MAX:
            tmp.body = use_float64_word_cas ?
                           codegen_stack_data_detail::_atomic_max_f64_word :
                       use_software_float_rmw ?
                           codegen_stack_data_detail::_atomic_max_float :
                           codegen_stack_data_detail::_atomic_max;
            break;
        default:
            LUISA_ERROR_WITH_LOCATION("Invalid atomic operator.");
    }

    AccessChain chain{
        op,
        rootVar,
        exprs.subspan(0, exprs.size() - extra_arg_size),
        isSpirv};
    auto iter = atomicsFuncs.emplace(std::move(chain));
    if (iter.second) {
        auto &access_chain = const_cast<AccessChain &>(*iter.first);
        access_chain.init_name();
        access_chain.gen_func_impl(func, util, tmp, exprs.subspan(exprs.size() - extra_arg_size, extra_arg_size), *incrementalFunc);
    }
    return *iter.first;
}

}// namespace lc::hlsl
