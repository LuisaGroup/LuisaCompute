#include "cuda_codegen_llvm_optix_ir_legalize.h"

#include <string>

#include <luisa/core/logging.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>

namespace luisa::compute::cuda {

namespace {

[[nodiscard]] bool requires_legalization(llvm::Intrinsic::ID id) noexcept {
    switch (id) {
        case llvm::Intrinsic::rint:
        case llvm::Intrinsic::fabs:
        case llvm::Intrinsic::umin:
        case llvm::Intrinsic::umax:
        case llvm::Intrinsic::smin:
        case llvm::Intrinsic::smax:
        case llvm::Intrinsic::usub_sat:
        case llvm::Intrinsic::uadd_sat:
        case llvm::Intrinsic::sadd_sat:
        case llvm::Intrinsic::ssub_sat:
        case llvm::Intrinsic::abs:
        case llvm::Intrinsic::fshl:
        case llvm::Intrinsic::fshr:
        case llvm::Intrinsic::vector_reduce_fadd: return true;
        default: return false;
    }
}

[[nodiscard]] llvm::Constant *splat_int_constant(llvm::Type *type, const llvm::APInt &value) noexcept {
    auto *scalar = llvm::ConstantInt::get(type->getScalarType(), value);
    if (auto *vector = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
        return llvm::ConstantVector::getSplat(vector->getElementCount(), scalar);
    }
    LUISA_ASSERT(!type->isVectorTy(), "OptiX IR integer legalization does not support scalable vectors.");
    return scalar;
}

[[nodiscard]] llvm::Value *legalize_scalar_rint(llvm::IRBuilder<> &builder, llvm::Value *value) noexcept {
    auto type = value->getType();
    LUISA_ASSERT(type->isHalfTy() || type->isFloatTy() || type->isDoubleTy(),
                 "OptiX IR rint legalization requires half, float, or double.");
    // Every finite half value and its rounded integer are exactly representable
    // in float. The extension also preserves the sign of zero and infinity.
    if (type->isHalfTy()) {
        auto wide = builder.CreateFPExt(value, builder.getFloatTy());
        return builder.CreateFPTrunc(legalize_scalar_rint(builder, wide), type);
    }
    auto function_type = llvm::FunctionType::get(type, {type}, false);
    auto assembly = type->isFloatTy() ? "cvt.rni.f32.f32 $0, $1;" : "cvt.rni.f64.f64 $0, $1;";
    auto constraints = type->isFloatTy() ? "=f,f" : "=d,d";
    auto operation = llvm::InlineAsm::get(function_type, assembly, constraints, false);
    auto result = builder.CreateCall(function_type, operation, {value});
    result->setDoesNotThrow();
    result->setDoesNotAccessMemory();
    // No .ftz or .sat: retain subnormals, signed zero, infinities, and NaNs.
    // NVPTX's ordinary floating-point environment uses ties-to-even rounding.
    return result;
}

[[nodiscard]] llvm::Value *legalize_rint(llvm::IRBuilder<> &builder, llvm::Value *value) noexcept {
    auto type = value->getType();
    if (auto vector = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
        llvm::Value *result = llvm::UndefValue::get(type);
        for (auto i = 0u; i < vector->getNumElements(); i++) {
            auto element = builder.CreateExtractElement(value, i);
            result = builder.CreateInsertElement(result, legalize_scalar_rint(builder, element), i);
        }
        return result;
    }
    LUISA_ASSERT(!type->isVectorTy(), "OptiX IR rint legalization does not support scalable vectors.");
    return legalize_scalar_rint(builder, value);
}

[[nodiscard]] llvm::Value *legalize_fabs(llvm::IRBuilder<> &builder, llvm::Value *value) noexcept {
    auto type = value->getType();
    auto scalar_type = type->getScalarType();
    LUISA_ASSERT(scalar_type->isHalfTy() || scalar_type->isFloatTy() || scalar_type->isDoubleTy(),
                 "OptiX IR fabs legalization requires half, float, or double.");
    auto width = scalar_type->getScalarSizeInBits();
    auto integer_type = llvm::IntegerType::get(builder.getContext(), width);
    llvm::Type *bits_type = integer_type;
    llvm::Constant *mask = llvm::ConstantInt::get(integer_type, llvm::APInt::getSignedMaxValue(width));
    if (auto vector = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
        bits_type = llvm::FixedVectorType::get(integer_type, vector->getNumElements());
        mask = llvm::ConstantVector::getSplat(vector->getElementCount(), mask);
    } else {
        LUISA_ASSERT(!type->isVectorTy(), "OptiX IR fabs legalization does not support scalable vectors.");
    }
    // Clearing only the sign bit preserves the NaN payload and maps -0 to +0,
    // exactly as llvm.fabs; no floating comparison or NaN canonicalization.
    auto bits = builder.CreateBitCast(value, bits_type);
    return builder.CreateBitCast(builder.CreateAnd(bits, mask), type);
}

[[nodiscard]] unsigned legalize_bool_mask_comparisons(llvm::Module &module) noexcept {
    llvm::SmallVector<llvm::BitCastInst *, 8u> masks;
    for (auto &function : module) {
        for (auto &block : function) {
            for (auto &instruction : block) {
                auto cast = llvm::dyn_cast<llvm::BitCastInst>(&instruction);
                if (cast == nullptr) { continue; }
                auto vector = llvm::dyn_cast<llvm::FixedVectorType>(cast->getSrcTy());
                if (vector != nullptr && vector->getElementType()->isIntegerTy(1u) &&
                    cast->getDestTy()->isIntegerTy(vector->getNumElements())) {
                    masks.emplace_back(cast);
                }
            }
        }
    }
    auto replaced = 0u;
    for (auto mask : masks) {
        llvm::SmallVector<llvm::ICmpInst *, 4u> comparisons;
        auto supported = true;
        for (auto user : mask->users()) {
            auto compare = llvm::dyn_cast<llvm::ICmpInst>(user);
            if (compare == nullptr || !compare->isEquality()) {
                supported = false;
                break;
            }
            auto other = compare->getOperand(compare->getOperand(0) == mask ? 1u : 0u);
            auto constant = llvm::dyn_cast<llvm::ConstantInt>(other);
            if (constant == nullptr || (!constant->isZero() && !constant->isMinusOne())) {
                supported = false;
                break;
            }
            comparisons.emplace_back(compare);
        }
        if (!supported) { continue; }
        // The OptiX IR reader cannot handle narrow packed boolean masks such
        // as <3 x i1> bitcast to i3. EQ/NE comparisons with zero or all ones
        // are scalar any/all reductions, so no packed integer is needed.
        auto vector = llvm::cast<llvm::FixedVectorType>(mask->getSrcTy());
        for (auto compare : comparisons) {
            auto other = compare->getOperand(compare->getOperand(0) == mask ? 1u : 0u);
            auto all_ones = !llvm::cast<llvm::ConstantInt>(other)->isZero();
            llvm::IRBuilder<> builder{compare};
            builder.SetCurrentDebugLocation(compare->getDebugLoc());
            llvm::Value *reduction = builder.getInt1(all_ones);
            for (auto lane = 0u; lane < vector->getNumElements(); lane++) {
                auto value = builder.CreateExtractElement(mask->getOperand(0), lane);
                // Use bitwise operations rather than short-circuit selects
                // to preserve poison propagation from every boolean lane.
                reduction = all_ones ? builder.CreateAnd(reduction, value) : builder.CreateOr(reduction, value);
            }
            auto negate = compare->getPredicate() == llvm::CmpInst::ICMP_EQ ? !all_ones : all_ones;
            if (negate) { reduction = builder.CreateNot(reduction); }
            compare->replaceAllUsesWith(reduction);
            compare->eraseFromParent();
        }
        mask->eraseFromParent();
        replaced++;
    }
    return replaced;
}

struct SurfaceMemoryOperation {
    bool store;
    unsigned dimensions;
    unsigned lanes;
    unsigned bits;
};

[[nodiscard]] SurfaceMemoryOperation classify_surface_memory_operation(llvm::StringRef name) noexcept {
    auto store = name.consume_front("llvm.nvvm.sust.b.");
    auto recognized = store || name.consume_front("llvm.nvvm.suld.");
    LUISA_ASSERT(recognized, "Invalid OptiX IR surface operation.");
    auto dimensions = name.consume_front("2d.") ? 2u : name.consume_front("3d.") ? 3u :
                                                                                   0u;
    auto zero_boundary = name.consume_back(".zero");
    LUISA_ASSERT(dimensions != 0u && zero_boundary,
                 "OptiX IR surface lowering requires a 2D/3D zero-boundary operation.");
    auto lanes = name.consume_front("v2") ? 2u : name.consume_front("v4") ? 4u :
                                                                            1u;
    auto bits = name == "i8" ? 8u : name == "i16" ? 16u :
                                name == "i32"     ? 32u :
                                                    0u;
    LUISA_ASSERT(bits != 0u, "OptiX IR surface lowering requires i8/i16/i32 lanes.");
    return {store, dimensions, lanes, bits};
}

[[nodiscard]] unsigned legalize_surface_memory_operations(llvm::Module &module) noexcept {
    // CUDA 13.4 / driver 617.14 also miscompiles native NVRTC OptiX IR surface
    // intrinsics (BYTE4 reports misaligned local/shared accesses). Equivalent
    // inline surface instructions avoid that lowering bug; regular PTX emission
    // keeps its original intrinsics and never enters this compatibility pass.
    llvm::SmallVector<llvm::CallInst *, 32u> calls;
    llvm::SmallVector<llvm::Function *, 32u> declarations;
    for (auto &function : module) {
        if (function.getName().starts_with("llvm.nvvm.suld.") ||
            function.getName().starts_with("llvm.nvvm.sust.")) {
            declarations.emplace_back(&function);
        }
        for (auto &block : function) {
            for (auto &instruction : block) {
                auto call = llvm::dyn_cast<llvm::CallInst>(&instruction);
                auto callee = call == nullptr ? nullptr : call->getCalledFunction();
                if (callee != nullptr && (callee->getName().starts_with("llvm.nvvm.suld.") ||
                                          callee->getName().starts_with("llvm.nvvm.sust."))) {
                    calls.emplace_back(call);
                }
            }
        }
    }
    for (auto call : calls) {
        auto operation = classify_surface_memory_operation(call->getCalledFunction()->getName());
        auto coordinate_arguments = operation.dimensions + 1u;
        auto register_bits = operation.bits == 8u ? 16u : operation.bits;
        auto register_type = llvm::IntegerType::get(module.getContext(), register_bits);
        LUISA_ASSERT(call->arg_size() == coordinate_arguments + (operation.store ? operation.lanes : 0u) &&
                         call->getArgOperand(0)->getType()->isIntegerTy(64u),
                     "Invalid OptiX IR surface handle or argument count.");
        for (auto i = 1u; i < coordinate_arguments; i++) {
            LUISA_ASSERT(call->getArgOperand(i)->getType()->isIntegerTy(32u), "Invalid OptiX IR surface coordinate.");
        }
        if (operation.store) {
            LUISA_ASSERT(call->getType()->isVoidTy(), "Invalid OptiX IR surface store result.");
            for (auto i = 0u; i < operation.lanes; i++) {
                LUISA_ASSERT(call->getArgOperand(coordinate_arguments + i)->getType() == register_type,
                             "Invalid OptiX IR surface store lane.");
            }
        } else if (operation.lanes == 1u) {
            LUISA_ASSERT(call->getType() == register_type, "Invalid OptiX IR scalar surface load result.");
        } else {
            auto result = llvm::dyn_cast<llvm::StructType>(call->getType());
            LUISA_ASSERT(result != nullptr && result->getNumElements() == operation.lanes,
                         "Invalid OptiX IR vector surface load result.");
            for (auto element : result->elements()) {
                LUISA_ASSERT(element == register_type, "Invalid OptiX IR surface load lane.");
            }
        }
        llvm::IRBuilder<> builder{call};
        builder.SetCurrentDebugLocation(call->getDebugLoc());
        llvm::SmallVector<llvm::Value *, 9u> arguments;
        llvm::SmallVector<llvm::Type *, 9u> parameters;
        for (auto i = 0u; i < coordinate_arguments; i++) { arguments.emplace_back(call->getArgOperand(i)); }
        // PTX 3D surface addresses use four coordinates. NVVM's intrinsic
        // omits the final zero padding lane; x is already a byte coordinate.
        if (operation.dimensions == 3u) { arguments.emplace_back(builder.getInt32(0u)); }
        auto data_argument = static_cast<unsigned>(arguments.size());
        if (operation.store) {
            for (auto i = 0u; i < operation.lanes; i++) { arguments.emplace_back(call->getArgOperand(coordinate_arguments + i)); }
        }
        for (auto argument : arguments) { parameters.emplace_back(argument->getType()); }
        auto output_count = operation.store ? 0u : operation.lanes;
        auto reg = operation.bits == 32u ? "r" : "h";
        std::string constraints;
        for (auto i = 0u; i < output_count; i++) { constraints += std::string{"="} + reg + ","; }
        constraints += "l";
        for (auto i = 1u; i < data_argument; i++) { constraints += ",r"; }
        if (operation.store) {
            for (auto i = 0u; i < operation.lanes; i++) { constraints += std::string{","} + reg; }
        }
        constraints += ",~{memory}";
        auto operand = [](unsigned index) noexcept { return std::string{"$"} + std::to_string(index); };
        auto lanes = [&](unsigned first) noexcept {
            if (operation.lanes == 1u) { return operand(first); }
            std::string list{"{"};
            for (auto i = 0u; i < operation.lanes; i++) {
                if (i != 0u) { list += ", "; }
                list += operand(first + i);
            }
            return list + "}";
        };
        auto address = std::string{"["} + operand(output_count) + ", {";
        for (auto i = 1u; i < data_argument; i++) {
            if (i != 1u) { address += ", "; }
            address += operand(output_count + i);
        }
        address += "}]";
        auto assembly = std::string{operation.store ? "sust.b." : "suld.b."} +
                        std::to_string(operation.dimensions) + "d.";
        if (operation.lanes != 1u) { assembly += "v" + std::to_string(operation.lanes) + "."; }
        assembly += "b" + std::to_string(operation.bits) + ".zero ";
        assembly += operation.store ? address + ", " + lanes(data_argument) : lanes(0u) + ", " + address;
        assembly += ";";
        auto type = llvm::FunctionType::get(call->getType(), parameters, false);
        // Preserve the original integer/aggregate register ABI. Floating-point
        // texel interpretation remains in the existing surrounding bitcasts.
        // Both reads and writes stay ordered with other surface/OptiX calls.
        auto body = llvm::InlineAsm::get(type, assembly, constraints, true);
        auto replacement = builder.CreateCall(type, body, arguments);
        replacement->setDoesNotThrow();
        replacement->setDebugLoc(call->getDebugLoc());
        if (!operation.store) { call->replaceAllUsesWith(replacement); }
        call->eraseFromParent();
    }
    for (auto declaration : declarations) {
        if (declaration->use_empty() && !declaration->isUsedByMetadata()) { declaration->eraseFromParent(); }
    }
    return static_cast<unsigned>(calls.size());
}

[[nodiscard]] unsigned legalize_activemask_operations(llvm::Module &module) noexcept {
    llvm::SmallVector<llvm::CallInst *, 16u> calls;
    llvm::Function *activemask_decl = nullptr;
    for (auto &function : module) {
        if (function.getIntrinsicID() == llvm::Intrinsic::nvvm_activemask) {
            activemask_decl = &function;
            for (auto *user : function.users()) {
                if (auto *call = llvm::dyn_cast<llvm::CallInst>(user)) {
                    calls.emplace_back(call);
                }
            }
        }
    }
    if (!calls.empty()) {
        auto *i32 = llvm::Type::getInt32Ty(module.getContext());
        auto *fn_type = llvm::FunctionType::get(i32, false);
        auto *body = llvm::InlineAsm::get(fn_type, "activemask.b32 $0;", "=r", true);
        for (auto *call : calls) {
            llvm::IRBuilder<> builder{call};
            builder.SetCurrentDebugLocation(call->getDebugLoc());
            auto *replacement = builder.CreateCall(fn_type, body, {});
            replacement->setDoesNotThrow();
            replacement->setDebugLoc(call->getDebugLoc());
            replacement->takeName(call);
            call->replaceAllUsesWith(replacement);
            call->eraseFromParent();
        }
    }
    if (activemask_decl != nullptr && activemask_decl->use_empty() &&
        !activemask_decl->isUsedByMetadata()) {
        activemask_decl->eraseFromParent();
    }
    return static_cast<unsigned>(calls.size());
}

[[nodiscard]] unsigned legalize_lanemask_operations(llvm::Module &module) noexcept {
    llvm::SmallVector<llvm::CallInst *, 16u> calls;
    llvm::SmallVector<llvm::Function *, 5u> declarations;
    auto sreg_for_intrinsic = [](llvm::Intrinsic::ID id) noexcept -> const char * {
        switch (id) {
            case llvm::Intrinsic::nvvm_read_ptx_sreg_lanemask_eq: return "mov.u32 $0, %lanemask_eq;";
            case llvm::Intrinsic::nvvm_read_ptx_sreg_lanemask_le: return "mov.u32 $0, %lanemask_le;";
            case llvm::Intrinsic::nvvm_read_ptx_sreg_lanemask_lt: return "mov.u32 $0, %lanemask_lt;";
            case llvm::Intrinsic::nvvm_read_ptx_sreg_lanemask_ge: return "mov.u32 $0, %lanemask_ge;";
            case llvm::Intrinsic::nvvm_read_ptx_sreg_lanemask_gt: return "mov.u32 $0, %lanemask_gt;";
            default: return nullptr;
        }
    };
    for (auto &function : module) {
        if (sreg_for_intrinsic(function.getIntrinsicID()) != nullptr) {
            declarations.emplace_back(&function);
            for (auto *user : function.users()) {
                if (auto *call = llvm::dyn_cast<llvm::CallInst>(user)) {
                    calls.emplace_back(call);
                }
            }
        }
    }
    auto *i32 = llvm::Type::getInt32Ty(module.getContext());
    auto *fn_type = llvm::FunctionType::get(i32, false);
    for (auto *call : calls) {
        auto asm_str = sreg_for_intrinsic(call->getCalledFunction()->getIntrinsicID());
        auto *body = llvm::InlineAsm::get(fn_type, asm_str, "=r", true);
        llvm::IRBuilder<> builder{call};
        builder.SetCurrentDebugLocation(call->getDebugLoc());
        auto *replacement = builder.CreateCall(fn_type, body, {});
        replacement->setDoesNotThrow();
        replacement->setDebugLoc(call->getDebugLoc());
        replacement->takeName(call);
        call->replaceAllUsesWith(replacement);
        call->eraseFromParent();
    }
    for (auto *decl : declarations) {
        if (decl->use_empty() && !decl->isUsedByMetadata()) { decl->eraseFromParent(); }
    }
    return static_cast<unsigned>(calls.size());
}

[[nodiscard]] unsigned legalize_match_all_operations(llvm::Module &module) noexcept {
    llvm::SmallVector<llvm::CallInst *, 16u> calls;
    llvm::SmallVector<llvm::Function *, 2u> declarations;
    for (auto &function : module) {
        auto id = function.getIntrinsicID();
        if (id == llvm::Intrinsic::nvvm_match_all_sync_i32p ||
            id == llvm::Intrinsic::nvvm_match_all_sync_i64p) {
            declarations.emplace_back(&function);
            for (auto *user : function.users()) {
                if (auto *call = llvm::dyn_cast<llvm::CallInst>(user)) {
                    calls.emplace_back(call);
                }
            }
        }
    }
    auto *i1 = llvm::Type::getInt1Ty(module.getContext());
    auto *i32 = llvm::Type::getInt32Ty(module.getContext());
    auto *i64 = llvm::Type::getInt64Ty(module.getContext());
    auto *ret_type = llvm::StructType::get(module.getContext(), {i32, i1});
    for (auto *call : calls) {
        auto is_i64 = call->getCalledFunction()->getIntrinsicID() == llvm::Intrinsic::nvvm_match_all_sync_i64p;
        auto *val_type = is_i64 ? i64 : i32;
        auto *fn_type = llvm::FunctionType::get(ret_type, {i32, val_type}, false);
        auto callee = module.getOrInsertFunction(
            is_i64 ? "llvm.nvvm.match.all.sync.i64" : "llvm.nvvm.match.all.sync.i32",
            fn_type);
        if (auto *fn = llvm::dyn_cast<llvm::Function>(callee.getCallee())) {
            fn->addFnAttr(llvm::Attribute::NoUnwind);
        }
        llvm::IRBuilder<> builder{call};
        builder.SetCurrentDebugLocation(call->getDebugLoc());
        auto *mask = call->getArgOperand(0u);
        auto *val = call->getArgOperand(1u);
        auto *replacement = builder.CreateCall(fn_type, callee.getCallee(), {mask, val});
        replacement->setDoesNotThrow();
        replacement->setDebugLoc(call->getDebugLoc());
        replacement->takeName(call);
        call->replaceAllUsesWith(replacement);
        call->eraseFromParent();
    }
    for (auto *decl : declarations) {
        if (decl->use_empty() && !decl->isUsedByMetadata()) { decl->eraseFromParent(); }
    }
    return static_cast<unsigned>(calls.size());
}

[[nodiscard]] unsigned legalize_warp_sync_operations(llvm::Module &module) noexcept {
    llvm::SmallVector<llvm::CallInst *, 32u> calls;
    llvm::SmallVector<llvm::Function *, 16u> declarations;
    auto is_warp_sync = [](llvm::Intrinsic::ID id) noexcept {
        switch (id) {
            case llvm::Intrinsic::nvvm_vote_ballot_sync:
            case llvm::Intrinsic::nvvm_vote_all_sync:
            case llvm::Intrinsic::nvvm_vote_any_sync:
            case llvm::Intrinsic::nvvm_shfl_sync_idx_i32:
            case llvm::Intrinsic::nvvm_shfl_sync_bfly_i32:
            case llvm::Intrinsic::nvvm_shfl_sync_up_i32:
            case llvm::Intrinsic::nvvm_shfl_sync_down_i32:
            case llvm::Intrinsic::nvvm_redux_sync_add:
            case llvm::Intrinsic::nvvm_redux_sync_and:
            case llvm::Intrinsic::nvvm_redux_sync_or:
            case llvm::Intrinsic::nvvm_redux_sync_xor:
            case llvm::Intrinsic::nvvm_redux_sync_min:
            case llvm::Intrinsic::nvvm_redux_sync_max:
            case llvm::Intrinsic::nvvm_redux_sync_umin:
            case llvm::Intrinsic::nvvm_redux_sync_umax: return true;
            default: return false;
        }
    };
    for (auto &function : module) {
        if (is_warp_sync(function.getIntrinsicID())) {
            declarations.emplace_back(&function);
            for (auto *user : function.users()) {
                if (auto *call = llvm::dyn_cast<llvm::CallInst>(user)) {
                    calls.emplace_back(call);
                }
            }
        }
    }
    auto *i1 = llvm::Type::getInt1Ty(module.getContext());
    auto *i32 = llvm::Type::getInt32Ty(module.getContext());
    auto *pair_type = llvm::StructType::get(module.getContext(), {i32, i1});
    auto get_nvvm_fn = [&](llvm::StringRef name, llvm::FunctionType *fn_type, bool convergent) noexcept {
        auto callee = module.getOrInsertFunction(name, fn_type);
        if (auto *fn = llvm::dyn_cast<llvm::Function>(callee.getCallee())) {
            fn->addFnAttr(llvm::Attribute::NoUnwind);
            if (convergent) { fn->addFnAttr(llvm::Attribute::Convergent); }
        }
        return callee;
    };
    for (auto *call : calls) {
        auto id = call->getCalledFunction()->getIntrinsicID();
        llvm::IRBuilder<> builder{call};
        builder.SetCurrentDebugLocation(call->getDebugLoc());
        llvm::Value *replacement = nullptr;
        switch (id) {
            case llvm::Intrinsic::nvvm_vote_all_sync:
            case llvm::Intrinsic::nvvm_vote_any_sync:
            case llvm::Intrinsic::nvvm_vote_ballot_sync: {
                auto *fn_type = llvm::FunctionType::get(pair_type, {i32, i32, i1}, false);
                auto callee = get_nvvm_fn("llvm.nvvm.vote.sync", fn_type, true);
                auto mode = id == llvm::Intrinsic::nvvm_vote_all_sync ? 0u :
                            id == llvm::Intrinsic::nvvm_vote_any_sync ? 1u :
                                                                        3u;
                auto *vote_call = builder.CreateCall(
                    fn_type, callee.getCallee(),
                    {call->getArgOperand(0u), builder.getInt32(mode), call->getArgOperand(1u)});
                vote_call->setDoesNotThrow();
                vote_call->setConvergent();
                vote_call->setDebugLoc(call->getDebugLoc());
                auto elem_idx = id == llvm::Intrinsic::nvvm_vote_ballot_sync ? 0u : 1u;
                replacement = builder.CreateExtractValue(vote_call, elem_idx);
                break;
            }
            case llvm::Intrinsic::nvvm_shfl_sync_idx_i32:
            case llvm::Intrinsic::nvvm_shfl_sync_up_i32:
            case llvm::Intrinsic::nvvm_shfl_sync_down_i32:
            case llvm::Intrinsic::nvvm_shfl_sync_bfly_i32: {
                auto *fn_type = llvm::FunctionType::get(pair_type, {i32, i32, i32, i32, i32}, false);
                auto callee = get_nvvm_fn("llvm.nvvm.shfl.sync.i32", fn_type, true);
                auto mode = id == llvm::Intrinsic::nvvm_shfl_sync_idx_i32  ? 0u :
                            id == llvm::Intrinsic::nvvm_shfl_sync_up_i32   ? 1u :
                            id == llvm::Intrinsic::nvvm_shfl_sync_down_i32 ? 2u :
                                                                             3u;
                auto *shfl_call = builder.CreateCall(
                    fn_type, callee.getCallee(),
                    {call->getArgOperand(0u), builder.getInt32(mode), call->getArgOperand(1u),
                     call->getArgOperand(2u), call->getArgOperand(3u)});
                shfl_call->setDoesNotThrow();
                shfl_call->setConvergent();
                shfl_call->setDebugLoc(call->getDebugLoc());
                replacement = builder.CreateExtractValue(shfl_call, 0u);
                break;
            }
            case llvm::Intrinsic::nvvm_redux_sync_add:
            case llvm::Intrinsic::nvvm_redux_sync_umin:
            case llvm::Intrinsic::nvvm_redux_sync_min:
            case llvm::Intrinsic::nvvm_redux_sync_umax:
            case llvm::Intrinsic::nvvm_redux_sync_max:
            case llvm::Intrinsic::nvvm_redux_sync_and:
            case llvm::Intrinsic::nvvm_redux_sync_or:
            case llvm::Intrinsic::nvvm_redux_sync_xor: {
                auto *fn_type = llvm::FunctionType::get(i32, {i32, i32, i32}, false);
                auto callee = get_nvvm_fn("llvm.nvvm.redux.sync", fn_type, false);
                auto mode = id == llvm::Intrinsic::nvvm_redux_sync_add  ? 1u :
                            id == llvm::Intrinsic::nvvm_redux_sync_umin ? 2u :
                            id == llvm::Intrinsic::nvvm_redux_sync_min  ? 3u :
                            id == llvm::Intrinsic::nvvm_redux_sync_umax ? 4u :
                            id == llvm::Intrinsic::nvvm_redux_sync_max  ? 5u :
                            id == llvm::Intrinsic::nvvm_redux_sync_and  ? 6u :
                            id == llvm::Intrinsic::nvvm_redux_sync_or   ? 8u :
                                                                          10u;
                auto *redux_call = builder.CreateCall(
                    fn_type, callee.getCallee(),
                    {builder.getInt32(mode), call->getArgOperand(0u), call->getArgOperand(1u)});
                redux_call->setDoesNotThrow();
                redux_call->setDebugLoc(call->getDebugLoc());
                replacement = redux_call;
                break;
            }
            default: LUISA_ERROR_WITH_LOCATION("Unhandled warp-sync intrinsic in OptiX IR legalizer.");
        }
        replacement->takeName(call);
        call->replaceAllUsesWith(replacement);
        call->eraseFromParent();
    }
    for (auto *decl : declarations) {
        if (decl->use_empty() && !decl->isUsedByMetadata()) { decl->eraseFromParent(); }
    }
    return static_cast<unsigned>(calls.size());
}

[[nodiscard]] llvm::Value *freeze_value(llvm::IRBuilder<> &builder, llvm::Value *value) noexcept {
    auto *type = value->getType();
    if (llvm::isa<llvm::UndefValue>(value)) {
        return llvm::Constant::getNullValue(type);
    }
    if (llvm::isGuaranteedNotToBeUndefOrPoison(value)) {
        return value;
    }
    if (auto *vector = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
        llvm::Value *result = llvm::PoisonValue::get(type);
        for (auto i = 0u; i < vector->getNumElements(); i++) {
            auto *elem = freeze_value(builder, builder.CreateExtractElement(value, i));
            result = builder.CreateInsertElement(result, elem, i);
        }
        return result;
    }
    if (auto *structure = llvm::dyn_cast<llvm::StructType>(type)) {
        llvm::Value *result = llvm::PoisonValue::get(type);
        for (auto i = 0u; i < structure->getNumElements(); i++) {
            auto *field = freeze_value(builder, builder.CreateExtractValue(value, i));
            result = builder.CreateInsertValue(result, field, i);
        }
        return result;
    }
    if (auto *array = llvm::dyn_cast<llvm::ArrayType>(type)) {
        llvm::Value *result = llvm::PoisonValue::get(type);
        for (auto i = 0u; i < array->getNumElements(); i++) {
            auto *elem = freeze_value(builder, builder.CreateExtractValue(value, i));
            result = builder.CreateInsertValue(result, elem, i);
        }
        return result;
    }
    auto emit_asm_barrier = [&](llvm::Value *v, const char *asm_str, const char *constraint) noexcept -> llvm::Value * {
        auto *t = v->getType();
        auto *ft = llvm::FunctionType::get(t, {t}, false);
        auto *ia = llvm::InlineAsm::get(ft, asm_str, constraint, false);
        auto *call = builder.CreateCall(ft, ia, {v});
        call->setDoesNotThrow();
        call->setDoesNotAccessMemory();
        call->setDebugLoc(builder.getCurrentDebugLocation());
        return call;
    };
    if (type->isPointerTy()) {
        auto *i64 = builder.getInt64Ty();
        auto *as_int = builder.CreatePtrToInt(value, i64);
        auto *frozen_int = emit_asm_barrier(as_int, "mov.b64 $0, $1;", "=l,l");
        return builder.CreateIntToPtr(frozen_int, type);
    }
    if (type->isHalfTy()) {
        auto *i16 = builder.getInt16Ty();
        auto *i32 = builder.getInt32Ty();
        auto *bits = builder.CreateZExt(builder.CreateBitCast(value, i16), i32);
        auto *frozen = emit_asm_barrier(bits, "mov.b32 $0, $1;", "=r,r");
        return builder.CreateBitCast(builder.CreateTrunc(frozen, i16), type);
    }
    if (type->isFloatTy()) {
        return emit_asm_barrier(value, "mov.f32 $0, $1;", "=f,f");
    }
    if (type->isDoubleTy()) {
        return emit_asm_barrier(value, "mov.f64 $0, $1;", "=d,d");
    }
    if (type->isIntegerTy()) {
        auto bits = type->getIntegerBitWidth();
        if (bits == 64u) {
            return emit_asm_barrier(value, "mov.b64 $0, $1;", "=l,l");
        }
        if (bits == 32u) {
            return emit_asm_barrier(value, "mov.b32 $0, $1;", "=r,r");
        }
        if (bits < 32u) {
            auto *i32 = builder.getInt32Ty();
            auto *wide = builder.CreateZExt(value, i32);
            auto *frozen = emit_asm_barrier(wide, "mov.b32 $0, $1;", "=r,r");
            return builder.CreateTrunc(frozen, type);
        }
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported freeze operand type in OptiX IR legalizer.");
}

[[nodiscard]] unsigned legalize_freeze_instructions(llvm::Module &module) noexcept {
    llvm::SmallVector<llvm::FreezeInst *, 32u> worklist;
    for (auto &function : module) {
        for (auto &block : function) {
            for (auto &inst : block) {
                if (auto *freeze_inst = llvm::dyn_cast<llvm::FreezeInst>(&inst)) {
                    worklist.emplace_back(freeze_inst);
                }
            }
        }
    }
    for (auto *freeze_inst : worklist) {
        llvm::IRBuilder<> builder{freeze_inst};
        builder.SetCurrentDebugLocation(freeze_inst->getDebugLoc());
        auto *replacement = freeze_value(builder, freeze_inst->getOperand(0));
        replacement->takeName(freeze_inst);
        freeze_inst->replaceAllUsesWith(replacement);
        freeze_inst->eraseFromParent();
    }
    return static_cast<unsigned>(worklist.size());
}

}// namespace

void luisa_compute_cuda_llvm_legalize_optix_ir(llvm::Module &module) noexcept {
    llvm::SmallVector<llvm::IntrinsicInst *, 32u> worklist;
    llvm::SmallVector<llvm::Function *, 16u> declarations;
    for (auto &function : module) {
        if (function.isDeclaration() && requires_legalization(function.getIntrinsicID())) {
            declarations.emplace_back(&function);
        }
        for (auto &block : function) {
            for (auto &instruction : block) {
                if (auto intrinsic = llvm::dyn_cast<llvm::IntrinsicInst>(&instruction);
                    intrinsic != nullptr && requires_legalization(intrinsic->getIntrinsicID())) {
                    worklist.emplace_back(intrinsic);
                }
            }
        }
    }
    for (auto call : worklist) {
        llvm::IRBuilder<> builder{call};
        builder.SetCurrentDebugLocation(call->getDebugLoc());
        if (llvm::isa<llvm::FPMathOperator>(call)) { builder.setFastMathFlags(call->getFastMathFlags()); }
        auto id = call->getIntrinsicID();
        llvm::Value *replacement = nullptr;
        if (id == llvm::Intrinsic::rint) {
            replacement = legalize_rint(builder, call->getArgOperand(0));
        } else if (id == llvm::Intrinsic::fabs) {
            replacement = legalize_fabs(builder, call->getArgOperand(0));
        } else if (id == llvm::Intrinsic::vector_reduce_fadd) {
            auto vector = call->getArgOperand(1);
            auto type = llvm::dyn_cast<llvm::FixedVectorType>(vector->getType());
            LUISA_ASSERT(type != nullptr, "OptiX IR floating reduction requires a fixed vector.");
            // Keep the explicit start value and ordered left fold. When the
            // original permits reassociation, these adds carry the same FMF.
            replacement = call->getArgOperand(0);
            for (auto i = 0u; i < type->getNumElements(); i++) {
                replacement = builder.CreateFAdd(replacement, builder.CreateExtractElement(vector, i));
            }
        } else if (id == llvm::Intrinsic::usub_sat) {
            auto *type = call->getType();
            auto bits = type->getScalarSizeInBits();
            auto *a = call->getArgOperand(0);
            auto *b = call->getArgOperand(1);
            auto *zero = splat_int_constant(type, llvm::APInt::getZero(bits));
            replacement = builder.CreateSelect(builder.CreateICmpUGT(a, b), builder.CreateSub(a, b), zero);
        } else if (id == llvm::Intrinsic::uadd_sat) {
            auto *type = call->getType();
            auto bits = type->getScalarSizeInBits();
            auto *a = call->getArgOperand(0);
            auto *b = call->getArgOperand(1);
            auto *sum = builder.CreateAdd(a, b);
            auto *all_ones = splat_int_constant(type, llvm::APInt::getAllOnes(bits));
            replacement = builder.CreateSelect(builder.CreateICmpULT(sum, a), all_ones, sum);
        } else if (id == llvm::Intrinsic::sadd_sat) {
            auto *type = call->getType();
            auto bits = type->getScalarSizeInBits();
            auto *a = call->getArgOperand(0);
            auto *b = call->getArgOperand(1);
            auto *zero = splat_int_constant(type, llvm::APInt::getZero(bits));
            auto *smin = splat_int_constant(type, llvm::APInt::getSignedMinValue(bits));
            auto *smax = splat_int_constant(type, llvm::APInt::getSignedMaxValue(bits));
            auto *sum = builder.CreateAdd(a, b);
            auto *ov = builder.CreateICmpSLT(builder.CreateAnd(builder.CreateXor(a, sum), builder.CreateXor(b, sum)), zero);
            auto *sat = builder.CreateSelect(builder.CreateICmpSLT(a, zero), smin, smax);
            replacement = builder.CreateSelect(ov, sat, sum);
        } else if (id == llvm::Intrinsic::ssub_sat) {
            auto *type = call->getType();
            auto bits = type->getScalarSizeInBits();
            auto *a = call->getArgOperand(0);
            auto *b = call->getArgOperand(1);
            auto *zero = splat_int_constant(type, llvm::APInt::getZero(bits));
            auto *smin = splat_int_constant(type, llvm::APInt::getSignedMinValue(bits));
            auto *smax = splat_int_constant(type, llvm::APInt::getSignedMaxValue(bits));
            auto *diff = builder.CreateSub(a, b);
            auto *ov = builder.CreateICmpSLT(builder.CreateAnd(builder.CreateXor(a, b), builder.CreateXor(a, diff)), zero);
            auto *sat = builder.CreateSelect(builder.CreateICmpSLT(a, zero), smin, smax);
            replacement = builder.CreateSelect(ov, sat, diff);
        } else if (id == llvm::Intrinsic::abs) {
            auto *type = call->getType();
            auto bits = type->getScalarSizeInBits();
            auto *x = call->getArgOperand(0);
            auto *zero = splat_int_constant(type, llvm::APInt::getZero(bits));
            replacement = builder.CreateSelect(builder.CreateICmpSLT(x, zero), builder.CreateSub(zero, x), x);
        } else if (id == llvm::Intrinsic::fshl || id == llvm::Intrinsic::fshr) {
            auto *type = call->getType();
            auto bits = type->getScalarSizeInBits();
            auto *a = call->getArgOperand(0);
            auto *b = call->getArgOperand(1);
            auto *s = call->getArgOperand(2);
            auto *zero = splat_int_constant(type, llvm::APInt::getZero(bits));
            auto *width_c = splat_int_constant(type, llvm::APInt{bits, bits});
            auto is_pow2 = (bits & (bits - 1u)) == 0u;
            auto *shift = is_pow2 ?
                              builder.CreateAnd(s, splat_int_constant(type, llvm::APInt{bits, bits - 1u})) :
                              builder.CreateURem(s, width_c);
            auto *is_zero = builder.CreateICmpEQ(shift, zero);
            auto *inv_shift = is_pow2 ?
                                  builder.CreateAnd(builder.CreateSub(width_c, shift),
                                                    splat_int_constant(type, llvm::APInt{bits, bits - 1u})) :
                                  builder.CreateSelect(is_zero, zero, builder.CreateSub(width_c, shift));
            if (id == llvm::Intrinsic::fshl) {
                auto *combined = builder.CreateOr(builder.CreateShl(a, shift), builder.CreateLShr(b, inv_shift));
                replacement = builder.CreateSelect(is_zero, a, combined);
            } else {
                auto *combined = builder.CreateOr(builder.CreateLShr(b, shift), builder.CreateShl(a, inv_shift));
                replacement = builder.CreateSelect(is_zero, b, combined);
            }
        } else {
            auto lhs = call->getArgOperand(0);
            auto rhs = call->getArgOperand(1);
            auto predicate = id == llvm::Intrinsic::umin ? llvm::CmpInst::ICMP_ULT :
                             id == llvm::Intrinsic::umax ? llvm::CmpInst::ICMP_UGT :
                             id == llvm::Intrinsic::smin ? llvm::CmpInst::ICMP_SLT :
                                                           llvm::CmpInst::ICMP_SGT;
            replacement = builder.CreateSelect(builder.CreateICmp(predicate, lhs, rhs), lhs, rhs);
        }
        call->replaceAllUsesWith(replacement);
        call->eraseFromParent();
    }
    for (auto declaration : declarations) {
        // Metadata may name a declaration independently of its SSA uses.
        if (declaration->use_empty() && !declaration->isUsedByMetadata()) { declaration->eraseFromParent(); }
    }
    auto surface_operations = legalize_surface_memory_operations(module);
    auto bool_masks = legalize_bool_mask_comparisons(module);
    auto activemasks = legalize_activemask_operations(module);
    auto lanemasks = legalize_lanemask_operations(module);
    auto match_alls = legalize_match_all_operations(module);
    auto warp_syncs = legalize_warp_sync_operations(module);
    auto freezes = legalize_freeze_instructions(module);
    LUISA_VERBOSE("Legalized {} intrinsic call(s), {} surface operation(s), {} boolean mask(s), {} activemask(s), {} lanemask(s), {} match.all(s), {} warp-sync op(s), and {} freeze(s) for the OptiX IR reader.",
                  worklist.size(), surface_operations, bool_masks, activemasks, lanemasks, match_alls, warp_syncs, freezes);
}

}// namespace luisa::compute::cuda
