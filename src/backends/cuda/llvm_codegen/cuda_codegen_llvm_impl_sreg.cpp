//
// Created by mike on 9/27/25.
//

#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

llvm::Value *CUDACodegenLLVMImpl::_read_special_register(IB &b, const FunctionContext &func_ctx,
                                                         xir::DerivedSpecialRegisterTag tag) noexcept {
    switch (tag) {
        case xir::DerivedSpecialRegisterTag::THREAD_ID: return _read_thread_id(b, func_ctx);
        case xir::DerivedSpecialRegisterTag::BLOCK_ID: return _read_block_id(b, func_ctx);
        case xir::DerivedSpecialRegisterTag::WARP_LANE_ID: return _read_warp_lane_id(b, func_ctx);
        case xir::DerivedSpecialRegisterTag::DISPATCH_ID: return _read_dispatch_id(b, func_ctx);
        case xir::DerivedSpecialRegisterTag::KERNEL_ID: return _read_kernel_id(b, func_ctx);
        case xir::DerivedSpecialRegisterTag::BLOCK_SIZE: return _read_block_size(b, func_ctx);
        case xir::DerivedSpecialRegisterTag::WARP_SIZE: return _read_warp_size(b, func_ctx);
        case xir::DerivedSpecialRegisterTag::DISPATCH_SIZE: return _read_dispatch_size(b, func_ctx);
        default: break;
    }
    LUISA_NOT_IMPLEMENTED("Special register {} not implemented.", xir::to_string(tag));
}

llvm::Value *CUDACodegenLLVMImpl::_read_block_id(IB &b, const FunctionContext &func_ctx) noexcept {
    if (_rt_analysis.uses_ray_tracing) {// block_id = (dispatch_id + block_size - uint3(1)) / block_size
        auto llvm_dispatch_id = _read_dispatch_id(b, func_ctx);
        auto llvm_block_size = _read_block_size(b, func_ctx);
        auto llvm_v3i32_one = llvm::ConstantDataVector::get(_llvm_context, std::array{1u, 1u, 1u});
        auto llvm_block_size_minus_one = b.CreateSub(llvm_block_size, llvm_v3i32_one);
        auto llvm_dispatch_id_padded = b.CreateAdd(llvm_dispatch_id, llvm_block_size_minus_one);
        auto llvm_block_id = b.CreateUDiv(llvm_dispatch_id_padded, llvm_block_size, "sreg.block.id");
        return llvm_block_id;
    }
    auto llvm_i32_type = b.getInt32Ty();
    auto llvm_bid_x = b.CreateIntrinsic(llvm_i32_type, llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {});
    auto llvm_bid_y = b.CreateIntrinsic(llvm_i32_type, llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y, {});
    auto llvm_bid_z = b.CreateIntrinsic(llvm_i32_type, llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z, {});
    auto llvm_bid = _create_llvm_vector(b, {llvm_bid_x, llvm_bid_y, llvm_bid_z});
    llvm_bid->setName("sreg.block.id");
    return llvm_bid;
}

llvm::Value *CUDACodegenLLVMImpl::_read_block_size(IB &, const FunctionContext &) noexcept {
    auto [bx, by, bz] = _config.block_size;
    auto is_power_of_two = [](auto x) noexcept { return x != 0u && (x & (x - 1u)) == 0u; };
    LUISA_ASSERT(is_power_of_two(bx) && is_power_of_two(by) && is_power_of_two(bz) && bx * by * bz <= 1024u,
                 "Block size must be power of two and not exceed 1024.");
    return llvm::ConstantDataVector::get(_llvm_context, std::array{bx, by, bz});
}

llvm::Value *CUDACodegenLLVMImpl::_read_thread_id(IB &b, const FunctionContext &func_ctx) noexcept {
    if (_rt_analysis.uses_ray_tracing) {// thread_id = dispatch_id % block_size
        auto llvm_dispatch_id = _read_dispatch_id(b, func_ctx);
        auto llvm_block_size = _read_block_size(b, func_ctx);
        auto llvm_tid = b.CreateURem(llvm_dispatch_id, llvm_block_size, "sreg.thread.id");
        return llvm_tid;
    }
    auto call = [&, llvm_i32_type = b.getInt32Ty()](uint32_t axis) noexcept -> llvm::Value * {
        // an axis must be 0 if the block size on that axis is 1
        if (_config.block_size[axis] <= 1u) { return b.getInt32(0); }
        // otherwise, read the thread id from special register
        constexpr std::array intrinsics = {
            llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x,
            llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y,
            llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z,
        };
        auto llvm_tid_axis = b.CreateIntrinsic(llvm_i32_type, intrinsics[axis], {});
        // the following assumption somehow breaks the optimization passes
        // auto llvm_tid_axis_lt_block_size = b.CreateICmpULT(llvm_tid_axis, b.getInt32(_config.block_size[axis]));
        // b.CreateAssumption(llvm_tid_axis_lt_block_size);
        return llvm_tid_axis;
    };
    auto llvm_tid_x = call(0);
    auto llvm_tid_y = call(1);
    auto llvm_tid_z = call(2);
    auto llvm_tid = _create_llvm_vector(b, {llvm_tid_x, llvm_tid_y, llvm_tid_z});
    llvm_tid->setName("sreg.thread.id");
    return llvm_tid;
}

llvm::Value *CUDACodegenLLVMImpl::_read_dispatch_size(IB &, const FunctionContext &func_ctx) noexcept {
    return func_ctx.llvm_dispatch_size;
}

llvm::Value *CUDACodegenLLVMImpl::_read_optix_launch_size(IB &b) noexcept {
    auto x = b.CreateCall(_get_inline_asm("call ($0), _optix_get_launch_dimension_x, ();", "=r", false), {});
    auto y = b.CreateCall(_get_inline_asm("call ($0), _optix_get_launch_dimension_y, ();", "=r", false), {});
    auto z = b.CreateCall(_get_inline_asm("call ($0), _optix_get_launch_dimension_z, ();", "=r", false), {});
    return _create_llvm_vector(b, {x, y, z});
}

llvm::Value *CUDACodegenLLVMImpl::_read_dispatch_id(IB &b, const FunctionContext &func_ctx) noexcept {
    if (_rt_analysis.uses_ray_tracing) {
        // asm("call (%0), _optix_get_launch_index_$axis, ();" : "=r"(out) : );
        auto call = [this, &b](uint32_t axis) noexcept -> llvm::Value * {
            if (_config.block_size[axis] <= 1u) { b.getInt32(0); }
            constexpr std::array axis_names = {"x", "y", "z"};
            auto llvm_asm_str = fmt::format("call ($0), _optix_get_launch_index_{}, ();", axis_names[axis]);
            auto llvm_asm = _get_inline_asm(llvm_asm_str, "=r", false);
            return b.CreateCall(llvm_asm, {}, std::string{"sreg.dispatch.id."}.append(axis_names[axis]));
        };
        auto llvm_did_x = call(0);
        auto llvm_did_y = call(1);
        auto llvm_did_z = call(2);
        auto llvm_dispatch_id = _create_llvm_vector(b, {llvm_did_x, llvm_did_y, llvm_did_z});
        return llvm_dispatch_id;
    }
    auto llvm_tid = _read_thread_id(b, func_ctx);
    auto llvm_bid = _read_block_id(b, func_ctx);
    auto llvm_block_size = _read_block_size(b, func_ctx);
    auto llvm_dispatch_offset = b.CreateMul(llvm_bid, llvm_block_size, "", true, true);
    auto llvm_dispatch_id = b.CreateAdd(llvm_tid, llvm_dispatch_offset, "sreg.dispatch.id", true, true);
    return llvm_dispatch_id;
}

llvm::Value *CUDACodegenLLVMImpl::_read_warp_size(IB &b, const FunctionContext &) const noexcept {
    if (_rt_analysis.uses_ray_tracing) { LUISA_NOT_IMPLEMENTED(); }
    return b.getInt32(32);
}

llvm::Value *CUDACodegenLLVMImpl::_read_warp_lane_id(IB &b, const FunctionContext &) const noexcept {
    if (_rt_analysis.uses_ray_tracing) { LUISA_NOT_IMPLEMENTED(); }
    return b.CreateIntrinsic(b.getInt32Ty(), llvm::Intrinsic::nvvm_read_ptx_sreg_laneid,
                             {}, {}, "sreg.warp.lane.id");
}

llvm::Value *CUDACodegenLLVMImpl::_read_kernel_id(IB &, const FunctionContext &func_ctx) noexcept {
    return func_ctx.llvm_kernel_id;
}

llvm::Value *CUDACodegenLLVMImpl::_read_warp_active_lane_mask(IB &b) const noexcept {
    if (_rt_analysis.uses_ray_tracing) { LUISA_NOT_IMPLEMENTED(); }
    auto mask = b.CreateIntrinsic(b.getInt32Ty(), llvm::Intrinsic::nvvm_activemask, {});
    mask->setName("sreg.warp.active.mask");
    return mask;
}

llvm::Value *CUDACodegenLLVMImpl::_read_warp_prefix_lane_mask(IB &b) const noexcept {
    if (_rt_analysis.uses_ray_tracing) { LUISA_NOT_IMPLEMENTED(); }
    return b.CreateIntrinsic(b.getInt32Ty(), llvm::Intrinsic::nvvm_read_ptx_sreg_lanemask_lt,
                             {}, {}, "sreg.warp.prefix.lane.mask");
}

}// namespace luisa::compute::cuda
