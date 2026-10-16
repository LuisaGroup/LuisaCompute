//
// Created by mike on 9/27/25.
//

#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

llvm::Value *CUDACodegenLLVMImpl::_create_llvm_vector(IB &b, llvm::ArrayRef<llvm::Value *> elems) noexcept {
    LUISA_DEBUG_ASSERT(!elems.empty());
    LUISA_DEBUG_ASSERT(std::all_of(elems.begin(), elems.end(), [&](auto e) { return e->getType() == elems.front()->getType(); }));
    auto llvm_vec_type = llvm::VectorType::get(elems.front()->getType(), static_cast<unsigned>(elems.size()), false);
    auto vec = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_vec_type));
    for (auto i = 0u; i < elems.size(); i++) { vec = b.CreateInsertElement(vec, elems[i], i); }
    return vec;
}

void CUDACodegenLLVMImpl::_translate_instruction(IB &b, FunctionContext &func_ctx, const xir::Instruction *inst) noexcept {
    llvm::Value *result{nullptr};
    auto handle_case = [&]<typename F>(F f) noexcept -> llvm::Value * {
        if constexpr (std::is_same_v<decltype(f()), void>) {
            f();
            return nullptr;
        } else {
            return f();
        }
    };
#define LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(inst_type, inst_method)      \
    case xir::inst_type::static_derived_instruction_tag(): {             \
        result = handle_case([&] {                                       \
            return _translate_##inst_method##_inst(                      \
                b, func_ctx, static_cast<const xir::inst_type *>(inst)); \
        });                                                              \
        break;                                                           \
    }
    switch (inst->derived_instruction_tag()) {
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(IfInst, if)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(SwitchInst, switch)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(IndexedBranchInst, indexed_branch)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(LoopInst, loop)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(SimpleLoopInst, simple_loop)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(BranchInst, branch)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(ConditionalBranchInst, conditional_branch)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(UnreachableInst, unreachable)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(BreakInst, break)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(ContinueInst, continue)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(ReturnInst, return)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(RasterDiscardInst, raster_discard)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(PhiInst, phi)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(AllocaInst, alloca)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(LoadInst, load)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(StoreInst, store)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(GEPInst, gep)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(AtomicInst, atomic)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(ArithmeticInst, arithmetic)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(ThreadGroupInst, thread_group)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(ResourceQueryInst, resource_query)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(ResourceReadInst, resource_read)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(ResourceWriteInst, resource_write)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(RayQueryLoopInst, ray_query_loop)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(RayQueryDispatchInst, ray_query_dispatch)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(RayQueryObjectReadInst, ray_query_object_read)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(RayQueryObjectWriteInst, ray_query_object_write)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(RayQueryPipelineInst, ray_query_pipeline)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(AutodiffScopeInst, autodiff_scope)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(AutodiffIntrinsicInst, autodiff_intrinsic)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(CallInst, call)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(CastInst, cast)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(PrintInst, print)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(ClockInst, clock)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(DebugBreakInst, debug_break)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(AssertInst, assert)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(AssumeInst, assume)
        LUISA_CUDA_LLVM_TRANSLATE_INST_CASE(OutlineInst, outline)
        default: LUISA_ERROR("Unknown instruction tag {}.", xir::to_string(inst->derived_instruction_tag()));
    }
#undef LUISA_CUDA_LLVM_TRANSLATE_INST_CASE
    if (result != nullptr) {
        func_ctx.local_values.try_emplace(inst, result);
    }
}

}// namespace luisa::compute::cuda
