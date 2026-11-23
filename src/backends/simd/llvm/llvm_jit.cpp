#include "llvm_jit.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <optional>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <pthread.h>
#endif

#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectTransformLayer.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Verifier.h>
#include "../../common/llvm_pass_builder.h"
#include <llvm/Support/Error.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Vectorize/SLPVectorizer.h>

#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>

#include "../../common/env_flag.h"
#include "../../common/llvm_target_options.h"

namespace luisa::compute::simd {

namespace {

// Module keeps a non-owning reference to its LLVMContext. Function parameters
// are otherwise destroyed in the opposite order of this public (module,
// context) API, so every early return must explicitly release the module
// first. Moved-from pointers make the successful ORC path a no-op here.
struct ModuleContextLifetime {
    std::unique_ptr<::llvm::Module> &module;
    std::unique_ptr<::llvm::LLVMContext> &context;

    ~ModuleContextLifetime() noexcept {
        module.reset();
        context.reset();
    }
};

// Large vectorized kernels can recurse deeply inside LLVM -O2 / codegen passes
// and exceed the 1 MB default main-thread stack on Windows (or 512 KB worker
// stacks on macOS). Run heavy compiler steps on a 16 MB stack when needed.
template<typename Fn>
void run_with_compiler_stack(Fn &&fn) noexcept {
    constexpr size_t compiler_stack_bytes = 16u * 1024u * 1024u;
    struct Context {
        std::remove_reference_t<Fn> *fn;
    } ctx{&fn};
#ifdef _WIN32
    auto proc = [](LPVOID param) noexcept -> DWORD {
        (*static_cast<Context *>(param)->fn)();
        return 0u;
    };
    HANDLE thread = ::CreateThread(
        nullptr, compiler_stack_bytes, proc, &ctx,
        STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
    if (thread != nullptr) {
        ::WaitForSingleObject(thread, INFINITE);
        ::CloseHandle(thread);
        return;
    }
    fn();
#else
    auto proc = [](void *param) noexcept -> void * {
        (*static_cast<Context *>(param)->fn)();
        return nullptr;
    };
    pthread_attr_t attr{};
    if (::pthread_attr_init(&attr) == 0) {
        ::pthread_attr_setstacksize(&attr, compiler_stack_bytes);
        pthread_t thread{};
        if (::pthread_create(&thread, &attr, proc, &ctx) == 0) {
            ::pthread_join(thread, nullptr);
            ::pthread_attr_destroy(&attr);
            return;
        }
        ::pthread_attr_destroy(&attr);
    }
    fn();
#endif
}

// Strip <1 x T> pseudo-vectors in W=1 functions without touching <2 x T>,
// <3 x T>, <4 x T>, or wider vectors so native vector operations (e.g. SLP,
// W1 float2/3/4 loads, texture packets) remain intact while stack spills and
// residual 1-lane vectors optimize cleanly under SROA.
[[nodiscard]] bool is_v1_type(const ::llvm::Type *ty) noexcept {
    if (ty == nullptr) { return false; }
    if (auto *vt = ::llvm::dyn_cast<::llvm::FixedVectorType>(ty)) {
        return vt->getNumElements() == 1u;
    }
    return false;
}

[[nodiscard]] ::llvm::Value *constant_v1_element(::llvm::Value *v) noexcept {
    auto *vt = ::llvm::dyn_cast<::llvm::FixedVectorType>(v->getType());
    if (vt == nullptr || vt->getNumElements() != 1u) { return nullptr; }
    if (::llvm::isa<::llvm::PoisonValue>(v)) {
        return ::llvm::PoisonValue::get(vt->getElementType());
    }
    if (::llvm::isa<::llvm::UndefValue>(v)) {
        return ::llvm::UndefValue::get(vt->getElementType());
    }
    if (auto *c = ::llvm::dyn_cast<::llvm::Constant>(v)) {
        if (auto *splat = c->getSplatValue()) { return splat; }
        if (auto *agg = c->getAggregateElement(0u)) { return agg; }
    }
    return nullptr;
}

[[nodiscard]] ::llvm::Value *unwrap_v1(
    ::llvm::IRBuilder<> &b, ::llvm::Value *v) noexcept {
    if (!is_v1_type(v->getType())) { return v; }
    if (auto *ie = ::llvm::dyn_cast<::llvm::InsertElementInst>(v)) {
        if (auto *idx = ::llvm::dyn_cast<::llvm::ConstantInt>(ie->getOperand(2u));
            idx != nullptr && idx->isZero()) {
            return ie->getOperand(1u);
        }
    }
    if (auto *s = constant_v1_element(v)) { return s; }
    return b.CreateExtractElement(v, uint64_t{0u});
}

[[nodiscard]] ::llvm::Value *wrap_v1(
    ::llvm::IRBuilder<> &b, ::llvm::Value *s) noexcept {
    auto *v1_ty = ::llvm::FixedVectorType::get(s->getType(), 1u);
    return b.CreateInsertElement(
        ::llvm::PoisonValue::get(v1_ty), s, uint64_t{0u});
}

[[nodiscard]] bool is_const_v1_true(::llvm::Value *v) noexcept {
    if (auto *c = ::llvm::dyn_cast<::llvm::Constant>(v)) {
        return c->isAllOnesValue();
    }
    if (auto *ie = ::llvm::dyn_cast<::llvm::InsertElementInst>(v)) {
        if (auto *idx = ::llvm::dyn_cast<::llvm::ConstantInt>(ie->getOperand(2u));
            idx != nullptr && idx->isZero()) {
            if (auto *elem = ::llvm::dyn_cast<::llvm::ConstantInt>(ie->getOperand(1u));
                elem != nullptr && elem->isOne()) {
                return true;
            }
        }
    }
    return false;
}

[[nodiscard]] bool is_single_overloaded_elementwise_v1_intrinsic(
    ::llvm::Intrinsic::ID id) noexcept {
    switch (id) {
        case ::llvm::Intrinsic::fabs:
        case ::llvm::Intrinsic::sqrt:
        case ::llvm::Intrinsic::fma:
        case ::llvm::Intrinsic::fmuladd:
        case ::llvm::Intrinsic::minnum:
        case ::llvm::Intrinsic::maxnum:
        case ::llvm::Intrinsic::minimum:
        case ::llvm::Intrinsic::maximum:
        case ::llvm::Intrinsic::floor:
        case ::llvm::Intrinsic::ceil:
        case ::llvm::Intrinsic::trunc:
        case ::llvm::Intrinsic::round:
        case ::llvm::Intrinsic::roundeven:
        case ::llvm::Intrinsic::rint:
        case ::llvm::Intrinsic::nearbyint:
        case ::llvm::Intrinsic::copysign:
        case ::llvm::Intrinsic::sin:
        case ::llvm::Intrinsic::cos:
        case ::llvm::Intrinsic::exp:
        case ::llvm::Intrinsic::exp2:
        case ::llvm::Intrinsic::log:
        case ::llvm::Intrinsic::log2:
        case ::llvm::Intrinsic::log10:
        case ::llvm::Intrinsic::pow:
        case ::llvm::Intrinsic::ctpop:
        case ::llvm::Intrinsic::ctlz:
        case ::llvm::Intrinsic::cttz:
        case ::llvm::Intrinsic::bitreverse:
        case ::llvm::Intrinsic::bswap:
        case ::llvm::Intrinsic::fshl:
        case ::llvm::Intrinsic::fshr:
        case ::llvm::Intrinsic::abs:
        case ::llvm::Intrinsic::smin:
        case ::llvm::Intrinsic::smax:
        case ::llvm::Intrinsic::umin:
        case ::llvm::Intrinsic::umax:
            return true;
        default:
            return false;
    }
}

[[nodiscard]] bool is_v1_reduction_intrinsic(
    ::llvm::Intrinsic::ID id) noexcept {
    switch (id) {
        case ::llvm::Intrinsic::vector_reduce_or:
        case ::llvm::Intrinsic::vector_reduce_and:
        case ::llvm::Intrinsic::vector_reduce_xor:
        case ::llvm::Intrinsic::vector_reduce_add:
        case ::llvm::Intrinsic::vector_reduce_mul:
        case ::llvm::Intrinsic::vector_reduce_smin:
        case ::llvm::Intrinsic::vector_reduce_smax:
        case ::llvm::Intrinsic::vector_reduce_umin:
        case ::llvm::Intrinsic::vector_reduce_umax:
        case ::llvm::Intrinsic::vector_reduce_fmin:
        case ::llvm::Intrinsic::vector_reduce_fmax:
        case ::llvm::Intrinsic::vector_reduce_fminimum:
        case ::llvm::Intrinsic::vector_reduce_fmaximum:
            return true;
        default:
            return false;
    }
}

bool scalarize_v1_instructions(::llvm::Function &function) noexcept {
    bool any_changed = false;
    auto *module = function.getParent();
    for (unsigned pass = 0u; pass < 2u; ++pass) {
        bool changed = false;

        // Pass A: Rewrite <1 x T> PHI nodes to scalar T.
        struct PendingPhi {
            ::llvm::PHINode *old_phi;
            ::llvm::PHINode *new_phi;
        };
        ::llvm::SmallVector<PendingPhi, 16u> pending_phis;
        for (auto &&bb : function) {
            if (bb.isEHPad()) { continue; }
            for (auto &&phi : bb.phis()) {
                if (!is_v1_type(phi.getType())) { continue; }
                bool valid_preds = true;
                for (unsigned i = 0u; i < phi.getNumIncomingValues(); ++i) {
                    auto *term = phi.getIncomingBlock(i)->getTerminator();
#if LLVM_VERSION_MAJOR >= 24
                    if (term == nullptr ||
                        (!::llvm::isa<::llvm::UncondBrInst>(term) &&
                         !::llvm::isa<::llvm::CondBrInst>(term) &&
                         !::llvm::isa<::llvm::SwitchInst>(term))) {
#else
                    if (term == nullptr ||
                        (!::llvm::isa<::llvm::BranchInst>(term) &&
                         !::llvm::isa<::llvm::SwitchInst>(term))) {
#endif
                        valid_preds = false;
                        break;
                    }
                }
                if (!valid_preds) { continue; }
                auto *elt_ty = ::llvm::cast<::llvm::FixedVectorType>(phi.getType())
                                   ->getElementType();
                auto *new_phi = ::llvm::PHINode::Create(
                    elt_ty, phi.getNumIncomingValues(), phi.getName(),
#if LLVM_VERSION_MAJOR >= 24
                    phi.getIterator());
#else
                    &phi);
#endif
                ::llvm::IRBuilder<> b_wrap{&bb, bb.getFirstInsertionPt()};
                auto *wrapped = wrap_v1(b_wrap, new_phi);
                phi.replaceAllUsesWith(wrapped);
                pending_phis.push_back({&phi, new_phi});
                changed = true;
            }
        }
        for (auto &entry : pending_phis) {
            for (unsigned i = 0u; i < entry.old_phi->getNumIncomingValues(); ++i) {
                auto *pred_bb = entry.old_phi->getIncomingBlock(i);
                ::llvm::IRBuilder<> b{pred_bb->getTerminator()};
                entry.new_phi->addIncoming(
                    unwrap_v1(b, entry.old_phi->getIncomingValue(i)), pred_bb);
            }
        }
        for (auto &entry : pending_phis) {
            entry.old_phi->eraseFromParent();
        }

        // Pass B: Rewrite non-PHI <1 x T> instructions to scalar T.
        for (auto &&bb : function) {
            for (auto &inst : ::llvm::make_early_inc_range(bb)) {
                if (::llvm::isa<::llvm::PHINode>(&inst)) { continue; }

                if (auto *ee = ::llvm::dyn_cast<::llvm::ExtractElementInst>(&inst)) {
                    if (is_v1_type(ee->getVectorOperandType())) {
                        if (auto *ie = ::llvm::dyn_cast<::llvm::InsertElementInst>(
                                ee->getVectorOperand())) {
                            if (auto *idx = ::llvm::dyn_cast<::llvm::ConstantInt>(
                                    ie->getOperand(2u));
                                idx != nullptr && idx->isZero()) {
                                ee->replaceAllUsesWith(ie->getOperand(1u));
                                ee->eraseFromParent();
                                changed = true;
                                continue;
                            }
                        }
                        if (auto *s = constant_v1_element(ee->getVectorOperand())) {
                            ee->replaceAllUsesWith(s);
                            ee->eraseFromParent();
                            changed = true;
                        }
                    }
                    continue;
                }

                ::llvm::IRBuilder<> b{&inst};
                if (auto *li = ::llvm::dyn_cast<::llvm::LoadInst>(&inst)) {
                    if (!is_v1_type(li->getType()) || li->isAtomic()) { continue; }
                    auto *elt_ty = ::llvm::cast<::llvm::FixedVectorType>(li->getType())
                                       ->getElementType();
                    auto *new_li = b.CreateAlignedLoad(
                        elt_ty, li->getPointerOperand(), li->getAlign(),
                        li->isVolatile());
                    new_li->copyMetadata(*li);
                    li->replaceAllUsesWith(wrap_v1(b, new_li));
                    li->eraseFromParent();
                    changed = true;
                    continue;
                }

                if (auto *si = ::llvm::dyn_cast<::llvm::StoreInst>(&inst)) {
                    if (!is_v1_type(si->getValueOperand()->getType()) || si->isAtomic()) {
                        continue;
                    }
                    auto *scalar_val = unwrap_v1(b, si->getValueOperand());
                    auto *new_si = b.CreateAlignedStore(
                        scalar_val, si->getPointerOperand(), si->getAlign(),
                        si->isVolatile());
                    new_si->copyMetadata(*si);
                    si->eraseFromParent();
                    changed = true;
                    continue;
                }

                if (auto *gep = ::llvm::dyn_cast<::llvm::GetElementPtrInst>(&inst)) {
                    if (!is_v1_type(gep->getType())) { continue; }
                    auto *scalar_ptr = unwrap_v1(b, gep->getPointerOperand());
                    ::llvm::SmallVector<::llvm::Value *, 4u> indices;
                    indices.reserve(gep->getNumIndices());
                    for (auto &idx : gep->indices()) {
                        indices.emplace_back(unwrap_v1(b, idx.get()));
                    }
#if LLVM_VERSION_MAJOR >= 19
                    auto *scalar_gep = b.CreateGEP(
                        gep->getSourceElementType(), scalar_ptr, indices, "",
                        gep->getNoWrapFlags());
#else
                    auto *scalar_gep = b.CreateGEP(
                        gep->getSourceElementType(), scalar_ptr, indices, "",
                        gep->isInBounds());
#endif
                    gep->replaceAllUsesWith(wrap_v1(b, scalar_gep));
                    gep->eraseFromParent();
                    changed = true;
                    continue;
                }

                if (auto *bo = ::llvm::dyn_cast<::llvm::BinaryOperator>(&inst)) {
                    if (!is_v1_type(bo->getType())) { continue; }
                    auto *s0 = unwrap_v1(b, bo->getOperand(0u));
                    auto *s1 = unwrap_v1(b, bo->getOperand(1u));
                    auto *res = b.CreateBinOp(bo->getOpcode(), s0, s1);
                    if (auto *ri = ::llvm::dyn_cast<::llvm::Instruction>(res)) {
                        ri->copyIRFlags(bo);
                    }
                    bo->replaceAllUsesWith(wrap_v1(b, res));
                    bo->eraseFromParent();
                    changed = true;
                    continue;
                }

                if (auto *uo = ::llvm::dyn_cast<::llvm::UnaryOperator>(&inst)) {
                    if (!is_v1_type(uo->getType())) { continue; }
                    auto *s0 = unwrap_v1(b, uo->getOperand(0u));
                    auto *res = b.CreateUnOp(uo->getOpcode(), s0);
                    if (auto *ri = ::llvm::dyn_cast<::llvm::Instruction>(res)) {
                        ri->copyIRFlags(uo);
                    }
                    uo->replaceAllUsesWith(wrap_v1(b, res));
                    uo->eraseFromParent();
                    changed = true;
                    continue;
                }

                if (auto *ci = ::llvm::dyn_cast<::llvm::CastInst>(&inst)) {
                    auto *src = ci->getOperand(0u);
                    if (is_v1_type(ci->getType()) && is_v1_type(src->getType())) {
                        auto *dst_elt =
                            ::llvm::cast<::llvm::FixedVectorType>(ci->getType())
                                ->getElementType();
                        auto *s0 = unwrap_v1(b, src);
                        auto *res = b.CreateCast(ci->getOpcode(), s0, dst_elt);
                        if (auto *ri = ::llvm::dyn_cast<::llvm::Instruction>(res)) {
                            ri->copyIRFlags(ci);
                        }
                        ci->replaceAllUsesWith(wrap_v1(b, res));
                        ci->eraseFromParent();
                        changed = true;
                    } else if (ci->getOpcode() == ::llvm::Instruction::BitCast &&
                               is_v1_type(src->getType()) &&
                               !ci->getType()->isVectorTy()) {
                        auto *s0 = unwrap_v1(b, src);
                        auto *res = b.CreateBitCast(s0, ci->getType());
                        ci->replaceAllUsesWith(res);
                        ci->eraseFromParent();
                        changed = true;
                    } else if (ci->getOpcode() == ::llvm::Instruction::BitCast &&
                               is_v1_type(ci->getType()) &&
                               !src->getType()->isVectorTy()) {
                        auto *dst_elt =
                            ::llvm::cast<::llvm::FixedVectorType>(ci->getType())
                                ->getElementType();
                        auto *res = b.CreateBitCast(src, dst_elt);
                        ci->replaceAllUsesWith(wrap_v1(b, res));
                        ci->eraseFromParent();
                        changed = true;
                    }
                    continue;
                }

                if (auto *ic = ::llvm::dyn_cast<::llvm::ICmpInst>(&inst)) {
                    if (!is_v1_type(ic->getType()) ||
                        !is_v1_type(ic->getOperand(0u)->getType())) {
                        continue;
                    }
                    auto *s0 = unwrap_v1(b, ic->getOperand(0u));
                    auto *s1 = unwrap_v1(b, ic->getOperand(1u));
                    auto *res = b.CreateICmp(ic->getPredicate(), s0, s1);
                    ic->replaceAllUsesWith(wrap_v1(b, res));
                    ic->eraseFromParent();
                    changed = true;
                    continue;
                }

                if (auto *fc = ::llvm::dyn_cast<::llvm::FCmpInst>(&inst)) {
                    if (!is_v1_type(fc->getType()) ||
                        !is_v1_type(fc->getOperand(0u)->getType())) {
                        continue;
                    }
                    auto *s0 = unwrap_v1(b, fc->getOperand(0u));
                    auto *s1 = unwrap_v1(b, fc->getOperand(1u));
                    auto *res = b.CreateFCmp(fc->getPredicate(), s0, s1);
                    if (auto *ri = ::llvm::dyn_cast<::llvm::Instruction>(res);
                        ri != nullptr && ::llvm::isa<::llvm::FPMathOperator>(ri)) {
                        ri->copyFastMathFlags(fc);
                    }
                    fc->replaceAllUsesWith(wrap_v1(b, res));
                    fc->eraseFromParent();
                    changed = true;
                    continue;
                }

                if (auto *sel = ::llvm::dyn_cast<::llvm::SelectInst>(&inst)) {
                    if (!is_v1_type(sel->getType())) { continue; }
                    auto *sc = unwrap_v1(b, sel->getCondition());
                    auto *st = unwrap_v1(b, sel->getTrueValue());
                    auto *sf = unwrap_v1(b, sel->getFalseValue());
                    auto *res = b.CreateSelect(sc, st, sf);
                    if (auto *ri = ::llvm::dyn_cast<::llvm::Instruction>(res);
                        ri != nullptr && ::llvm::isa<::llvm::FPMathOperator>(sel) &&
                        ::llvm::isa<::llvm::FPMathOperator>(ri)) {
                        ri->copyFastMathFlags(sel);
                    }
                    sel->replaceAllUsesWith(wrap_v1(b, res));
                    sel->eraseFromParent();
                    changed = true;
                    continue;
                }

                if (auto *fr = ::llvm::dyn_cast<::llvm::FreezeInst>(&inst)) {
                    if (!is_v1_type(fr->getType())) { continue; }
                    auto *res = b.CreateFreeze(unwrap_v1(b, fr->getOperand(0u)));
                    fr->replaceAllUsesWith(wrap_v1(b, res));
                    fr->eraseFromParent();
                    changed = true;
                    continue;
                }

                if (auto *sv = ::llvm::dyn_cast<::llvm::ShuffleVectorInst>(&inst)) {
                    if (!is_v1_type(sv->getType())) { continue; }
                    auto *op0_vt = ::llvm::dyn_cast<::llvm::FixedVectorType>(
                        sv->getOperand(0u)->getType());
                    if (op0_vt == nullptr) { continue; }
                    auto mask = sv->getShuffleMask();
                    if (mask.size() != 1u) { continue; }
                    if (mask[0u] == ::llvm::PoisonMaskElem) {
                        sv->replaceAllUsesWith(::llvm::PoisonValue::get(sv->getType()));
                        sv->eraseFromParent();
                        changed = true;
                        continue;
                    }
                    const auto idx = static_cast<unsigned>(mask[0u]);
                    const auto n0 = op0_vt->getNumElements();
                    auto *src_vec = idx < n0 ? sv->getOperand(0u) : sv->getOperand(1u);
                    const auto elem_idx = idx < n0 ? idx : idx - n0;
                    ::llvm::Value *elem = nullptr;
                    if (is_v1_type(src_vec->getType()) && elem_idx == 0u) {
                        elem = unwrap_v1(b, src_vec);
                    } else {
                        elem = b.CreateExtractElement(
                            src_vec, static_cast<uint64_t>(elem_idx));
                    }
                    sv->replaceAllUsesWith(wrap_v1(b, elem));
                    sv->eraseFromParent();
                    changed = true;
                    continue;
                }

                if (auto *ii = ::llvm::dyn_cast<::llvm::IntrinsicInst>(&inst)) {
                    const auto id = ii->getIntrinsicID();
                    const bool has_explicit_align_arg = ii->arg_size() == 4u;
                    auto masked_align = [&](unsigned ptr_arg, unsigned align_arg) noexcept {
                        if (has_explicit_align_arg) {
                            if (auto *ci = ::llvm::dyn_cast<::llvm::ConstantInt>(
                                    ii->getArgOperand(align_arg))) {
                                if (ci->getZExtValue() > 0u) {
                                    return ::llvm::MaybeAlign{ci->getZExtValue()};
                                }
                            }
                            return ::llvm::MaybeAlign{1u};
                        }
                        return ::llvm::MaybeAlign{
                            ii->getParamAlign(ptr_arg).value_or(::llvm::Align{1u})};
                    };

                    if (id == ::llvm::Intrinsic::masked_gather &&
                        is_v1_type(ii->getType()) &&
                        (ii->arg_size() == 3u || ii->arg_size() == 4u)) {
                        const unsigned mask_idx = ii->arg_size() - 2u;
                        if (is_const_v1_true(ii->getArgOperand(mask_idx))) {
                            auto *elt_ty =
                                ::llvm::cast<::llvm::FixedVectorType>(ii->getType())
                                    ->getElementType();
                            auto *scalar_ptr = unwrap_v1(b, ii->getArgOperand(0u));
                            auto *load = b.CreateAlignedLoad(
                                elt_ty, scalar_ptr, masked_align(0u, 1u));
                            load->copyMetadata(*ii);
                            ii->replaceAllUsesWith(wrap_v1(b, load));
                            ii->eraseFromParent();
                            changed = true;
                        }
                        continue;
                    }

                    if (id == ::llvm::Intrinsic::masked_scatter &&
                        is_v1_type(ii->getArgOperand(0u)->getType()) &&
                        (ii->arg_size() == 3u || ii->arg_size() == 4u)) {
                        const unsigned mask_idx = ii->arg_size() - 1u;
                        if (is_const_v1_true(ii->getArgOperand(mask_idx))) {
                            auto *scalar_val = unwrap_v1(b, ii->getArgOperand(0u));
                            auto *scalar_ptr = unwrap_v1(b, ii->getArgOperand(1u));
                            auto *store = b.CreateAlignedStore(
                                scalar_val, scalar_ptr, masked_align(1u, 2u));
                            store->copyMetadata(*ii);
                            ii->eraseFromParent();
                            changed = true;
                        }
                        continue;
                    }

                    if (id == ::llvm::Intrinsic::masked_load &&
                        is_v1_type(ii->getType()) &&
                        (ii->arg_size() == 3u || ii->arg_size() == 4u)) {
                        const unsigned mask_idx = ii->arg_size() - 2u;
                        if (is_const_v1_true(ii->getArgOperand(mask_idx))) {
                            auto *elt_ty =
                                ::llvm::cast<::llvm::FixedVectorType>(ii->getType())
                                    ->getElementType();
                            auto *load = b.CreateAlignedLoad(
                                elt_ty, ii->getArgOperand(0u), masked_align(0u, 1u));
                            load->copyMetadata(*ii);
                            ii->replaceAllUsesWith(wrap_v1(b, load));
                            ii->eraseFromParent();
                            changed = true;
                        }
                        continue;
                    }

                    if (id == ::llvm::Intrinsic::masked_store &&
                        is_v1_type(ii->getArgOperand(0u)->getType()) &&
                        (ii->arg_size() == 3u || ii->arg_size() == 4u)) {
                        const unsigned mask_idx = ii->arg_size() - 1u;
                        if (is_const_v1_true(ii->getArgOperand(mask_idx))) {
                            auto *scalar_val = unwrap_v1(b, ii->getArgOperand(0u));
                            auto *store = b.CreateAlignedStore(
                                scalar_val, ii->getArgOperand(1u), masked_align(1u, 2u));
                            store->copyMetadata(*ii);
                            ii->eraseFromParent();
                            changed = true;
                        }
                        continue;
                    }

                    if (is_v1_reduction_intrinsic(id) &&
                        ii->arg_size() >= 1u &&
                        is_v1_type(ii->getArgOperand(0u)->getType()) &&
                        !ii->getType()->isVectorTy()) {
                        ii->replaceAllUsesWith(unwrap_v1(b, ii->getArgOperand(0u)));
                        ii->eraseFromParent();
                        changed = true;
                        continue;
                    }

                    if (module != nullptr &&
                        is_v1_type(ii->getType()) &&
                        is_single_overloaded_elementwise_v1_intrinsic(id)) {
                        auto *elt_ty =
                            ::llvm::cast<::llvm::FixedVectorType>(ii->getType())
                                ->getElementType();
                        ::llvm::SmallVector<::llvm::Value *, 4u> args;
                        args.reserve(ii->arg_size());
                        for (unsigned a = 0u; a < ii->arg_size(); ++a) {
                            auto *arg = ii->getArgOperand(a);
                            args.emplace_back(
                                is_v1_type(arg->getType()) ? unwrap_v1(b, arg) : arg);
                        }
#if LLVM_VERSION_MAJOR >= 20
                        auto *scalar_fn = ::llvm::Intrinsic::getOrInsertDeclaration(
                            module, id, {elt_ty});
#else
                        auto *scalar_fn = ::llvm::Intrinsic::getDeclaration(
                            module, id, {elt_ty});
#endif
                        auto *call = b.CreateCall(scalar_fn, args);
                        call->setTailCallKind(ii->getTailCallKind());
                        if (::llvm::isa<::llvm::FPMathOperator>(ii) &&
                            ::llvm::isa<::llvm::FPMathOperator>(call)) {
                            call->copyFastMathFlags(ii);
                        }
                        ii->replaceAllUsesWith(wrap_v1(b, call));
                        ii->eraseFromParent();
                        changed = true;
                        continue;
                    }
                }
            }
        }

        // Pass C: Clean up dead <1 x T> insertelement wrappers bottom-up.
        for (auto &&bb : function) {
            for (auto &inst : ::llvm::make_early_inc_range(::llvm::reverse(bb))) {
                if (auto *ie = ::llvm::dyn_cast<::llvm::InsertElementInst>(&inst)) {
                    if (is_v1_type(ie->getType()) && ie->use_empty()) {
                        ie->eraseFromParent();
                        changed = true;
                    }
                }
            }
        }

        any_changed |= changed;
        if (!changed) { break; }
    }
    return any_changed;
}

// Strip <1 x T> pseudo-vectors in W=1 functions after SROA/scalar opts and
// immediately before LoopVectorize/SLPVectorizer so LLVM SLP can pack
// independent scalar lanes (e.g. 8-corner 3D Perlin noise, Float3/Float4 math)
// into native SIMD registers on both x86-64 and AArch64 without touching
// explicit <2 x T> / <4 x T> operations.
struct ScalarizeW1FunctionPass {
    bool scalarize_all_functions{false};
    bool run_slp{true};

    // The pass-manager interface is stable even when PassInfoMixin is private.
    static ::llvm::StringRef name() noexcept { return "ScalarizeW1FunctionPass"; }
    static bool isRequired() noexcept { return false; }
    void printPipeline(::llvm::raw_ostream &os,
                       ::llvm::function_ref<::llvm::StringRef(::llvm::StringRef)> map_name) {
        os << map_name(name());
    }

    ::llvm::PreservedAnalyses run(
        ::llvm::Function &function,
        ::llvm::FunctionAnalysisManager &am) noexcept {
        if (!scalarize_all_functions &&
            !function.getName().ends_with(".simd_w1") &&
            !function.getName().ends_with(".live_w1")) {
            return ::llvm::PreservedAnalyses::all();
        }
        bool has_v1 = false;
        for (auto &&bb : function) {
            for (auto &&inst : bb) {
                if (is_v1_type(inst.getType())) {
                    has_v1 = true;
                    break;
                }
                for (auto &op : inst.operands()) {
                    if (is_v1_type(op->getType())) {
                        has_v1 = true;
                        break;
                    }
                }
                if (has_v1) { break; }
            }
            if (has_v1) { break; }
        }
        if (!has_v1 || !scalarize_v1_instructions(function)) {
            return ::llvm::PreservedAnalyses::all();
        }
        auto pa = ::llvm::PreservedAnalyses::none();
        am.invalidate(function, pa);
        auto run_fn_pass = [&](auto &&pass) noexcept {
            auto sub_pa = pass.run(function, am);
            am.invalidate(function, sub_pa);
        };
        run_fn_pass(::llvm::InstCombinePass{});
        run_fn_pass(::llvm::SROAPass{::llvm::SROAOptions::PreserveCFG});
        run_fn_pass(::llvm::InstCombinePass{});
        run_fn_pass(::llvm::EarlyCSEPass{/*UseMemorySSA=*/false});
        run_fn_pass(::llvm::SimplifyCFGPass{});
        if (run_slp) {
            run_fn_pass(::llvm::SLPVectorizerPass{});
        }
        return ::llvm::PreservedAnalyses::none();
    }
};

}// namespace

void LLVMJIT::_fail(std::string message) noexcept {
    if (_error.empty()) { _error = std::move(message); }
}

LLVMJIT::LLVMJIT(bool capture_object, bool enable_fast_math, bool scalarize_w1) noexcept
    : _enable_fast_math{enable_fast_math}, _scalarize_w1{scalarize_w1} {
    static std::once_flag llvm_target_init_once;
    static bool llvm_target_init_failed = false;
    std::call_once(llvm_target_init_once, [] {
        llvm_target_init_failed =
            ::llvm::InitializeNativeTarget() ||
            ::llvm::InitializeNativeTargetAsmPrinter();
    });
    if (llvm_target_init_failed) {
        _fail("failed to initialize the LLVM native target");
        return;
    }
    auto host = ::llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!host) {
        _fail("failed to detect the LLVM host target: " +
              ::llvm::toString(host.takeError()));
        return;
    }
    host->setCodeGenOptLevel(::llvm::CodeGenOptLevel::Aggressive);
    if (host->getTargetTriple().isAArch64()) {
        host->addFeatures({"+neon"});
    }
    // LLJIT selects PIC for an unspecified relocation model. Make that
    // choice explicit before creating the audit TargetMachine as well, so
    // captured assembly and ORC's final relocatable object have identical
    // instruction layout.
    host->setRelocationModel(::llvm::Reloc::PIC_);
    host->setCodeModel(::llvm::CodeModel::Small);
    auto &target_options = host->getOptions();
    luisa::compute::detail::apply_llvm_fp_target_options(
        target_options,
        enable_fast_math ? luisa::compute::detail::LLVMFPOpFusion::Fast :
                           luisa::compute::detail::LLVMFPOpFusion::Strict);
    if (!host->getTargetTriple().isOSBinFormatCOFF()) {
        luisa::compute::detail::enable_llvm_machine_function_splitting(
            target_options);
    }
    _target_triple = host->getTargetTriple().str();
    auto target_machine = host->createTargetMachine();
    if (!target_machine) {
        _fail("failed to create the LLVM host target machine: " +
              ::llvm::toString(target_machine.takeError()));
        return;
    }
    _target_machine = std::move(*target_machine);
    _target_machine->Options.MCOptions.AsmVerbose = true;
    ::llvm::orc::LLJITBuilder builder;
    // LLVM < 22 has no SectionMemoryManager reservation mode; keep its default layer.
#if LLVM_VERSION_MAJOR >= 22
    const auto &triple = host->getTargetTriple();
    if (triple.isOSBinFormatCOFF() && triple.isX86()) {
        // RuntimeDyldCOFFX86_64 requires all sections referenced by 32-bit
        // image-base-relative relocations (IMAGE_REL_AMD64_ADDR32NB) to be
        // allocated in a single monotonically ordered address range.
        builder.setObjectLinkingLayerCreator(
            [](::llvm::orc::ExecutionSession &es, auto &&...)
                -> ::llvm::Expected<std::unique_ptr<::llvm::orc::ObjectLayer>> {
                auto layer = std::make_unique<::llvm::orc::RTDyldObjectLinkingLayer>(
                    es, [](const ::llvm::MemoryBuffer &) {
                        return std::make_unique<::llvm::SectionMemoryManager>(
                            nullptr, /*ReserveAlloc=*/true);
                    });
                layer->setOverrideObjectFlagsWithResponsibilityFlags(true);
                layer->setAutoClaimResponsibilityForObjectSymbols(true);
                return layer;
            });
    }
#endif
    builder.setJITTargetMachineBuilder(std::move(*host));
    auto jit = builder.create();
    if (!jit) {
        _fail("failed to create LLVM ORC JIT: " +
              ::llvm::toString(jit.takeError()));
        _target_machine.reset();
        return;
    }
    _jit = std::move(*jit);
    // Uniform inverse hyperbolic operations call scalar libm functions.
    // Static CRTs do not export these symbols for ORC's process lookup.
    ::llvm::orc::SymbolMap math_symbols;
    auto bind_math_symbol = [this, &math_symbols](
                                const char *name, auto *function) noexcept {
        math_symbols.try_emplace(
            _jit->mangleAndIntern(name),
            ::llvm::orc::ExecutorSymbolDef{
                ::llvm::orc::ExecutorAddr::fromPtr(function),
                ::llvm::JITSymbolFlags::Callable});
    };
    using FloatMath = float (*)(float);
    using DoubleMath = double (*)(double);
    using FloatBinaryMath = float (*)(float, float);
    using DoubleBinaryMath = double (*)(double, double);
    bind_math_symbol("asinhf", static_cast<FloatMath>(&std::asinh));
    bind_math_symbol("acoshf", static_cast<FloatMath>(&std::acosh));
    bind_math_symbol("atanhf", static_cast<FloatMath>(&std::atanh));
    bind_math_symbol("asinh", static_cast<DoubleMath>(&std::asinh));
    bind_math_symbol("acosh", static_cast<DoubleMath>(&std::acosh));
    bind_math_symbol("atanh", static_cast<DoubleMath>(&std::atanh));
    bind_math_symbol("log2f", static_cast<FloatMath>(&std::log2));
    bind_math_symbol("log2", static_cast<DoubleMath>(&std::log2));
    bind_math_symbol("exp2f", static_cast<FloatMath>(&std::exp2));
    bind_math_symbol("exp2", static_cast<DoubleMath>(&std::exp2));
    bind_math_symbol("log10f", static_cast<FloatMath>(&std::log10));
    bind_math_symbol("log10", static_cast<DoubleMath>(&std::log10));
    bind_math_symbol("fmodf", static_cast<FloatBinaryMath>(&std::fmod));
    bind_math_symbol("fmod", static_cast<DoubleBinaryMath>(&std::fmod));
    bind_math_symbol("powf", static_cast<FloatBinaryMath>(&std::pow));
    bind_math_symbol("pow", static_cast<DoubleBinaryMath>(&std::pow));
    if (auto error = _jit->getMainJITDylib().define(
            ::llvm::orc::absoluteSymbols(std::move(math_symbols)))) {
        _fail("failed to define LLVM JIT scalar math symbols: " +
              ::llvm::toString(std::move(error)));
        return;
    }
    if (capture_object) {
        _object = std::make_shared<std::string>();
        _jit->getObjTransformLayer().setTransform(
            [capture = _object](
                std::unique_ptr<::llvm::MemoryBuffer> object)
                -> ::llvm::Expected<
                    std::unique_ptr<::llvm::MemoryBuffer>> {
                auto buffer = object->getBuffer();
                capture->assign(buffer.data(), buffer.size());
                return std::move(object);
            });
    }
}

LLVMJIT::~LLVMJIT() noexcept = default;
LLVMJIT::LLVMJIT(LLVMJIT &&) noexcept = default;
LLVMJIT &LLVMJIT::operator=(LLVMJIT &&) noexcept = default;

bool LLVMJIT::selects_size_bounded_pipeline(
    size_t block_count, size_t instruction_count) noexcept {
    static constexpr auto max_o2_blocks = size_t{65536u};
    static constexpr auto max_o2_instructions = size_t{1048576u};
    return block_count >= max_o2_blocks ||
           instruction_count >= max_o2_instructions;
}

bool LLVMJIT::add_module(
    std::unique_ptr<::llvm::Module> module,
    std::unique_ptr<::llvm::LLVMContext> context) noexcept {
    ModuleContextLifetime lifetime{module, context};
    if (!succeeded()) { return false; }
    if (module == nullptr || context == nullptr ||
        &module->getContext() != context.get()) {
        _fail("LLVM JIT module and context ownership do not match");
        return false;
    }
    auto size_bounded = false;
    if (!_prepare_module(*module, &size_bounded)) { return false; }

    if (size_bounded) {
        ::llvm::SmallVector<char, 0u> object_storage;
        ::llvm::raw_svector_ostream object_stream{object_storage};
        ::llvm::legacy::PassManager object_codegen;
        _target_machine->setOptLevel(::llvm::CodeGenOptLevel::Less);
        if (_target_machine->addPassesToEmitFile(
                object_codegen, object_stream, nullptr,
                ::llvm::CodeGenFileType::ObjectFile)) {
            _target_machine->setOptLevel(::llvm::CodeGenOptLevel::Aggressive);
            _fail("LLVM host target cannot emit relocatable object");
            return false;
        }
        object_codegen.run(*module);
        _target_machine->setOptLevel(::llvm::CodeGenOptLevel::Aggressive);
        if (_object != nullptr) {
            _object->assign(object_storage.data(), object_storage.size());
        }
        auto object_buffer = ::llvm::MemoryBuffer::getMemBufferCopy(
            ::llvm::StringRef{object_storage.data(), object_storage.size()},
            module->getModuleIdentifier());
        if (auto error = _jit->addObjectFile(std::move(object_buffer))) {
            _fail("failed to add size-bounded object to ORC JIT: " +
                  ::llvm::toString(std::move(error)));
            return false;
        }
        return true;
    }

    if (auto error = _jit->addIRModule(::llvm::orc::ThreadSafeModule(
            std::move(module), std::move(context)))) {
        _fail("failed to add LLVM IR module to ORC JIT: " +
              ::llvm::toString(std::move(error)));
        return false;
    }
    return true;
}

bool LLVMJIT::add_object(
    luisa::string_view object,
    luisa::string_view buffer_name) noexcept {
    if (!succeeded()) { return false; }
    if (object.empty()) {
        _fail("cannot add an empty object buffer to LLVM ORC JIT");
        return false;
    }
    auto object_buffer = ::llvm::MemoryBuffer::getMemBufferCopy(
        ::llvm::StringRef{object.data(), object.size()},
        ::llvm::StringRef{buffer_name.data(), buffer_name.size()});
    if (auto error = _jit->addObjectFile(std::move(object_buffer))) {
        _fail("failed to add cached object to ORC JIT: " +
              ::llvm::toString(std::move(error)));
        return false;
    }
    return true;
}

bool LLVMJIT::_prepare_module(
    ::llvm::Module &module, bool *out_size_bounded) noexcept {
    module.setDataLayout(_target_machine->createDataLayout());
#if LLVM_VERSION_MAJOR >= 21
    module.setTargetTriple(_target_machine->getTargetTriple());
#else
    module.setTargetTriple(_target_machine->getTargetTriple().str());
#endif
    if (::llvm::verifyModule(module, &::llvm::errs())) {
        _fail("refusing to process an invalid LLVM module");
        return false;
    }
    const auto fp_fusion = _enable_fast_math ?
                               luisa::compute::detail::LLVMFPOpFusion::Fast :
                               luisa::compute::detail::LLVMFPOpFusion::Strict;
    for (auto &&function : module) {
        luisa::compute::detail::apply_llvm_fp_function_options(
            function, fp_fusion);
    }

    ::llvm::LoopAnalysisManager loop_analyses;
    ::llvm::FunctionAnalysisManager function_analyses;
    ::llvm::CGSCCAnalysisManager cgscc_analyses;
    ::llvm::ModuleAnalysisManager module_analyses;
    ::llvm::PipelineTuningOptions tuning_options;
#if LLVM_VERSION_MAJOR >= 23
    tuning_options.MergeFunctions = false;
#endif
    auto pass_builder = luisa::compute::detail::make_llvm_pass_builder(
        module.getContext(), _target_machine.get(), tuning_options);
    if (!luisa::compute::detail::env_flag("LUISA_SIMD_DISABLE_W1_SCALARIZE")) {
        if (_scalarize_w1) {
            pass_builder.registerPipelineStartEPCallback(
                [](::llvm::ModulePassManager &mpm, ::llvm::OptimizationLevel) {
                    mpm.addPass(::llvm::AlwaysInlinerPass{});
                    ::llvm::FunctionPassManager fpm;
                    fpm.addPass(::llvm::SROAPass{::llvm::SROAOptions::PreserveCFG});
                    fpm.addPass(::llvm::InstCombinePass{});
                    fpm.addPass(ScalarizeW1FunctionPass{
                        /*scalarize_all_functions=*/true,
                        /*run_slp=*/false});
                    mpm.addPass(::llvm::createModuleToFunctionPassAdaptor(
                        std::move(fpm)));
                });
        }
        const bool scalarize_all = _scalarize_w1;
        pass_builder.registerVectorizerStartEPCallback(
            [scalarize_all](::llvm::FunctionPassManager &fpm, ::llvm::OptimizationLevel) {
                fpm.addPass(ScalarizeW1FunctionPass{
                    scalarize_all,
                    /*run_slp=*/!scalarize_all});
            });
    }
    pass_builder.registerModuleAnalyses(module_analyses);
    pass_builder.registerCGSCCAnalyses(cgscc_analyses);
    pass_builder.registerFunctionAnalyses(function_analyses);
    pass_builder.registerLoopAnalyses(loop_analyses);
    pass_builder.crossRegisterProxies(
        loop_analyses, function_analyses,
        cgscc_analyses, module_analyses);
    auto block_count = size_t{0u};
    auto instruction_count = size_t{0u};
    auto max_function_blocks = size_t{0u};
    auto max_function_instructions = size_t{0u};
    for (auto &&function : module) {
        if (function.isDeclaration()) { continue; }
        auto fn_blocks = size_t{0u};
        auto fn_instructions = size_t{0u};
        for (auto &&block : function) {
            fn_blocks++;
            fn_instructions += block.size();
        }
        block_count += fn_blocks;
        instruction_count += fn_instructions;
        max_function_blocks = std::max(max_function_blocks, fn_blocks);
        max_function_instructions =
            std::max(max_function_instructions, fn_instructions);
    }
    auto size_bounded = selects_size_bounded_pipeline(
        max_function_blocks, max_function_instructions);
    if (out_size_bounded != nullptr) {
        *out_size_bounded = size_bounded;
    }
    const auto full_o3_instruction_limit =
        _scalarize_w1 ? size_t{450'000u} : size_t{250'000u};
    const bool force_o2 =
        luisa::compute::detail::env_flag("LUISA_SIMD_OPT_LEVEL_O2") ||
        max_function_instructions > full_o3_instruction_limit;
    const auto opt_level = force_o2 ? ::llvm::OptimizationLevel::O2 :
                                      ::llvm::OptimizationLevel::O3;
    const char *pipeline_label =
        size_bounded ? "linear-SROA/O1-CG size-bounded" :
        force_o2     ? "O2" :
                       "O3";
    auto report = luisa::compute::detail::env_flag("LUISA_SIMD_REPORT_OPTIMIZATIONS");
    if (report) {
        LUISA_INFO(
            "SIMD LLVM pipeline: {}, blocks={}, instructions={}, "
            "max_fn_blocks={}, max_fn_instructions={}",
            pipeline_label,
            block_count, instruction_count,
            max_function_blocks, max_function_instructions);
    }
    // Outlining pure-IR native math helpers with FastCC avoids code duplication
    // in extreme cases, but on x86-64 FastCC clobbers all live ymm/zmm vector
    // registers across hot exp/log/pow/sin/cos calls (3-7% render regression).
    // Keep normal native math helpers inlined by default (matching the fallback
    // backend), and only force-outline when explicitly requested or when a
    // very large helper (>= 512 IR instructions) is heavily reused (>= 32 sites).
    const bool force_outline_native_math =
        luisa::compute::detail::env_flag("LUISA_SIMD_OUTLINE_NATIVE_MATH");
    if (instruction_count >= size_t{32768u} &&
        !luisa::compute::detail::env_flag("LUISA_SIMD_KEEP_INLINE_NATIVE_MATH")) {
        for (auto &&function : module) {
            if (function.isDeclaration() ||
                !function.hasFnAttribute("luisa.cpu.native_math")) {
                continue;
            }
            const auto fn_insns = function.getInstructionCount();
            if (!force_outline_native_math && fn_insns < size_t{512u}) {
                continue;
            }
            size_t call_count = 0u;
            bool all_direct_calls = true;
            for (auto *user : function.users()) {
                auto *cb = ::llvm::dyn_cast<::llvm::CallBase>(user);
                if (cb == nullptr || cb->getCalledFunction() != &function) {
                    all_direct_calls = false;
                    break;
                }
                ++call_count;
            }
            const size_t min_calls = force_outline_native_math ? size_t{8u} : size_t{32u};
            if (all_direct_calls && call_count >= min_calls) {
                function.removeFnAttr(::llvm::Attribute::AlwaysInline);
                function.addFnAttr(::llvm::Attribute::NoInline);
                function.setCallingConv(::llvm::CallingConv::Fast);
                for (auto *user : function.users()) {
                    ::llvm::cast<::llvm::CallBase>(user)->setCallingConv(
                        ::llvm::CallingConv::Fast);
                }
            }
        }
    }
    auto started = report ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto run_opt = [&]() noexcept {
        if (size_bounded) {
            ::llvm::ModulePassManager pipeline;
            pipeline.addPass(::llvm::AlwaysInlinerPass{});
            ::llvm::FunctionPassManager fpm;
            fpm.addPass(::llvm::SROAPass{::llvm::SROAOptions::PreserveCFG});
            fpm.addPass(::llvm::EarlyCSEPass{/*UseMemorySSA=*/false});
            fpm.addPass(::llvm::SimplifyCFGPass{});
            pipeline.addPass(
                ::llvm::createModuleToFunctionPassAdaptor(std::move(fpm)));
            pipeline.run(module, module_analyses);
        } else {
            auto pipeline = pass_builder.buildPerModuleDefaultPipeline(
                opt_level);
            pipeline.run(module, module_analyses);
        }
    };
    if (instruction_count >= size_t{32768u}) {
        run_with_compiler_stack(run_opt);
    } else {
        run_opt();
    }
    if (report) {
        auto elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        auto post_block_count = size_t{0u};
        auto post_instruction_count = size_t{0u};
        auto phi_count = size_t{0u};
        auto max_block_size = size_t{0u};
        auto max_block_phis = size_t{0u};
        for (auto &&function : module) {
            if (function.isDeclaration()) { continue; }
            for (auto &&block : function) {
                post_block_count++;
                post_instruction_count += block.size();
                max_block_size = std::max(max_block_size, block.size());
                auto block_phis = size_t{0u};
                for (auto &&instruction : block) {
                    block_phis += ::llvm::isa<::llvm::PHINode>(instruction);
                }
                phi_count += block_phis;
                max_block_phis = std::max(max_block_phis, block_phis);
            }
        }
        LUISA_INFO(
            "SIMD LLVM post-opt '{}': pipeline={}, optimize_ms={:.3f}, "
            "blocks={}, instructions={}, phis={}, max_block_size={}, max_block_phis={}",
            module.getModuleIdentifier(),
            size_bounded ? "O0-IR/O1-CG size-bounded" :
            force_o2     ? "O2" :
                           "O3",
            elapsed_ms,
            post_block_count, post_instruction_count, phi_count, max_block_size, max_block_phis);
    }
    return true;
}

std::string LLVMJIT::_emit_assembly(::llvm::Module &module) noexcept {
    auto size_bounded = false;
    if (!_prepare_module(module, &size_bounded)) { return {}; }
    ::llvm::SmallVector<char, 0u> storage;
    ::llvm::raw_svector_ostream output{storage};
    ::llvm::legacy::PassManager codegen;
    if (size_bounded) {
        _target_machine->setOptLevel(::llvm::CodeGenOptLevel::Less);
    }
    if (_target_machine->addPassesToEmitFile(
            codegen, output, nullptr,
            ::llvm::CodeGenFileType::AssemblyFile)) {
        if (size_bounded) {
            _target_machine->setOptLevel(::llvm::CodeGenOptLevel::Aggressive);
        }
        _fail("LLVM host target cannot emit assembly");
        return {};
    }
    codegen.run(module);
    if (size_bounded) {
        _target_machine->setOptLevel(::llvm::CodeGenOptLevel::Aggressive);
    }
    return std::string{storage.begin(), storage.end()};
}

std::string LLVMJIT::emit_assembly(
    std::unique_ptr<::llvm::Module> module,
    std::unique_ptr<::llvm::LLVMContext> context) noexcept {
    ModuleContextLifetime lifetime{module, context};
    if (!succeeded()) { return {}; }
    if (module == nullptr || context == nullptr ||
        &module->getContext() != context.get()) {
        _fail("LLVM assembly module and context ownership do not match");
        return {};
    }
    return _emit_assembly(*module);
}

std::string LLVMJIT::emit_assembly_copy(
    const ::llvm::Module &module) noexcept {
    if (!succeeded()) { return {}; }
    auto copy = ::llvm::CloneModule(module);
    return _emit_assembly(*copy);
}

void *LLVMJIT::lookup(luisa::string_view name) noexcept {
    if (!succeeded()) { return nullptr; }
    auto report = luisa::compute::detail::env_flag("LUISA_SIMD_REPORT_OPTIMIZATIONS");
    auto started = report ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::optional<::llvm::Expected<::llvm::orc::ExecutorAddr>> symbol_opt;
    run_with_compiler_stack([&]() noexcept {
        symbol_opt.emplace(_jit->lookup(::llvm::StringRef{name.data(), name.size()}));
    });
    auto &symbol = *symbol_opt;
    if (report) {
        auto elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        // ORC lookup includes materialization, native codegen and linking on
        // the first lookup. Later symbol lookups may reuse that materialization.
        LUISA_INFO("SIMD ORC lookup '{}': lookup_ms={:.3f}, succeeded={}", name, elapsed_ms, static_cast<bool>(symbol));
    }
    if (!symbol) {
        _fail("failed to look up LLVM JIT symbol '" +
              std::string{name} + "': " +
              ::llvm::toString(symbol.takeError()));
        return nullptr;
    }
    return symbol->toPtr<void *>();
}

bool LLVMJIT::supports_native_paired_leaf_gather(
    uint32_t width) const noexcept {
    // The measured rewrite is specific to one 512-bit vector of eight packed
    // 64-bit lanes. W4 was neutral and W16 requires two such instructions.
    // The generic legality/scalarization query does not establish relative
    // profitability against two 32-bit gathers. Keep that distinction in the
    // measured exact-W8 policy instead of treating legality as a cost proof.
    if (!succeeded() || width != 8u) { return false; }
    ::llvm::LLVMContext context;
    ::llvm::Module module{"simd-paired-gather-probe", context};
    module.setDataLayout(_target_machine->createDataLayout());
#if LLVM_VERSION_MAJOR >= 21
    module.setTargetTriple(_target_machine->getTargetTriple());
#else
    module.setTargetTriple(_target_machine->getTargetTriple().str());
#endif
    auto *type = ::llvm::FixedVectorType::get(
        ::llvm::Type::getInt64Ty(context), width);
    auto *function_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(context), false);
    auto *function = ::llvm::Function::Create(
        function_type, ::llvm::GlobalValue::PrivateLinkage,
        "simd_paired_gather_probe", module);
    auto target = _target_machine->getTargetTransformInfo(*function);
    auto fixed_register_bits = target.getRegisterBitWidth(
        ::llvm::TargetTransformInfo::RGK_FixedWidthVector);
    return fixed_register_bits.getKnownMinValue() >= 512u &&
           target.isLegalMaskedGather(type, ::llvm::Align{1u}) &&
           !target.forceScalarizeMaskedGather(type, ::llvm::Align{1u});
}

bool LLVMJIT::supports_native_biased_narrow_buffer_gather(
    uint32_t width) const noexcept {
    if (!succeeded() || (width != 8u && width != 16u)) {
        return false;
    }
    ::llvm::LLVMContext context;
    ::llvm::Module module{"simd-biased-narrow-gather-probe", context};
    auto data_layout = _target_machine->createDataLayout();
    // The address identity used by the rewrite is deliberately audited in
    // the backend's 64-bit host ABI. Do not silently generalize it to a
    // target with a different pointer-index width.
    if (data_layout.getPointerSizeInBits() != 64u ||
        data_layout.getIndexSizeInBits(0u) != 64u) {
        return false;
    }
    module.setDataLayout(data_layout);
#if LLVM_VERSION_MAJOR >= 21
    module.setTargetTriple(_target_machine->getTargetTriple());
#else
    module.setTargetTriple(_target_machine->getTargetTriple().str());
#endif
    auto *function_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(context), false);
    auto *function = ::llvm::Function::Create(
        function_type, ::llvm::GlobalValue::PrivateLinkage,
        "simd_biased_narrow_gather_probe", module);
    auto target = _target_machine->getTargetTransformInfo(*function);
    auto fixed_register_bits = target.getRegisterBitWidth(
        ::llvm::TargetTransformInfo::RGK_FixedWidthVector);
    // W16 is deliberately formed as two W8 gathers. A single W16 narrow
    // gather was slower than the existing wide-index lowering on the
    // reference host, while two independently schedulable halves preserve
    // the logical W16 mask and expose the same legal W8 machine shape.
    auto gather_width = std::min(width, 8u);
    auto *gather_type = ::llvm::FixedVectorType::get(
        ::llvm::Type::getInt32Ty(context), gather_width);
    auto alignment = ::llvm::Align{1u};
    return fixed_register_bits.getKnownMinValue() >= 512u &&
           target.isLegalMaskedGather(gather_type, alignment) &&
           !target.forceScalarizeMaskedGather(gather_type, alignment);
}

bool LLVMJIT::supports_native_predicated_loop(
    uint32_t width) const noexcept {
    if (!succeeded() || (width != 8u && width != 16u)) {
        return false;
    }
    ::llvm::LLVMContext context;
    ::llvm::Module module{"simd-predicated-loop-probe", context};
    module.setDataLayout(_target_machine->createDataLayout());
#if LLVM_VERSION_MAJOR >= 21
    module.setTargetTriple(_target_machine->getTargetTriple());
#else
    module.setTargetTriple(_target_machine->getTargetTriple().str());
#endif
    auto *function_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(context), false);
    auto *function = ::llvm::Function::Create(
        function_type, ::llvm::GlobalValue::PrivateLinkage,
        "simd_predicated_loop_probe", module);
    auto target = _target_machine->getTargetTransformInfo(*function);
    auto fixed_register_bits = target.getRegisterBitWidth(
        ::llvm::TargetTransformInfo::RGK_FixedWidthVector);
    auto *gather_type = ::llvm::FixedVectorType::get(
        ::llvm::Type::getInt32Ty(context), width);
    auto alignment = ::llvm::Align{alignof(uint32_t)};
    return fixed_register_bits.getKnownMinValue() >= 512u &&
           target.isLegalMaskedGather(gather_type, alignment) &&
           !target.forceScalarizeMaskedGather(gather_type, alignment);
}

bool LLVMJIT::supports_inlined_packet_batch(
    uint32_t width) const noexcept {
    if (!succeeded() || (width != 8u && width != 16u)) {
        return false;
    }
    ::llvm::LLVMContext context;
    ::llvm::Module module{"simd-packet-batch-probe", context};
    module.setDataLayout(_target_machine->createDataLayout());
#if LLVM_VERSION_MAJOR >= 21
    module.setTargetTriple(_target_machine->getTargetTriple());
#else
    module.setTargetTriple(_target_machine->getTargetTriple().str());
#endif
    auto *function_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(context), false);
    auto *function = ::llvm::Function::Create(
        function_type, ::llvm::GlobalValue::PrivateLinkage,
        "simd_packet_batch_probe", module);
    auto target = _target_machine->getTargetTransformInfo(*function);
    auto fixed_register_bits = target.getRegisterBitWidth(
        ::llvm::TargetTransformInfo::RGK_FixedWidthVector);
    auto *vector_type = ::llvm::FixedVectorType::get(
        ::llvm::Type::getInt32Ty(context), width);
    auto register_class = target.getRegisterClassForType(
        true, vector_type);
    return fixed_register_bits.getKnownMinValue() >= 512u &&
           target.getNumberOfRegisters(register_class) >= 32u;
}

bool LLVMJIT::supports_native_vector_compress(
    uint32_t width) const noexcept {
#if LLVM_VERSION_MAJOR < 18
    static_cast<void>(width);
    return false;
#else
    if (!succeeded() || width != 16u) { return false; }
    ::llvm::LLVMContext context;
    ::llvm::Module module{"simd-vector-compress-probe", context};
    module.setDataLayout(_target_machine->createDataLayout());
#if LLVM_VERSION_MAJOR >= 21
    module.setTargetTriple(_target_machine->getTargetTriple());
#else
    module.setTargetTriple(_target_machine->getTargetTriple().str());
#endif
    auto *function_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(context), false);
    auto *function = ::llvm::Function::Create(
        function_type, ::llvm::GlobalValue::PrivateLinkage,
        "simd_vector_compress_probe", module);
    auto target = _target_machine->getTargetTransformInfo(*function);
    auto fixed_register_bits = target.getRegisterBitWidth(
        ::llvm::TargetTransformInfo::RGK_FixedWidthVector);
    auto *vector_type = ::llvm::FixedVectorType::get(
        ::llvm::Type::getInt32Ty(context), width);
    return fixed_register_bits.getKnownMinValue() >= 512u &&
           target.isLegalMaskedCompressStore(
               vector_type, ::llvm::Align{16u});
#endif
}

bool LLVMJIT::supports_native_half_conversion(
    uint32_t width) const noexcept {
    if (!succeeded() ||
        (width != 4u && width != 8u && width != 16u)) {
        return false;
    }
    ::llvm::LLVMContext context;
    ::llvm::Module module{"simd-half-conversion-probe", context};
    module.setDataLayout(_target_machine->createDataLayout());
#if LLVM_VERSION_MAJOR >= 21
    module.setTargetTriple(_target_machine->getTargetTriple());
#else
    module.setTargetTriple(_target_machine->getTargetTriple().str());
#endif
    auto *function_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(context), false);
    auto *function = ::llvm::Function::Create(
        function_type, ::llvm::GlobalValue::PrivateLinkage,
        "simd_half_conversion_probe", module);
    auto target = _target_machine->getTargetTransformInfo(*function);
    auto *float_lanes = ::llvm::FixedVectorType::get(
        ::llvm::Type::getFloatTy(context), width);
    auto *half_lanes = ::llvm::FixedVectorType::get(
        ::llvm::Type::getHalfTy(context), width);
    auto extension_cost = target.getCastInstrCost(
        ::llvm::Instruction::FPExt, float_lanes, half_lanes,
        ::llvm::TTI::CastContextHint::None,
        ::llvm::TTI::TCK_RecipThroughput);
    auto truncation_cost = target.getCastInstrCost(
        ::llvm::Instruction::FPTrunc, half_lanes, float_lanes,
        ::llvm::TTI::CastContextHint::None,
        ::llvm::TTI::TCK_RecipThroughput);
    if (!extension_cost.isValid() || !truncation_cost.isValid()) {
        return false;
    }
    // One W8 cast or at most a bounded split of W16 is the measured useful
    // shape. Generic scalar/helper lowering is priced in the hundreds by TTI.
    auto maximum_packed_cost = static_cast<int64_t>(width / 4u);
    return extension_cost.getValue() <= maximum_packed_cost &&
           truncation_cost.getValue() <= maximum_packed_cost;
}

}// namespace luisa::compute::simd
