//
// Created by mike on 9/25/25.
//

#include <luisa/xir/passes/dom_tree.h>
#include <luisa/ast/function.h>
#include <luisa/runtime/rhi/pixel.h>

#include "../cuda_texture.h"
#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

llvm::Function *CUDACodegenLLVMImpl::_get_or_declare_llvm_function(const xir::Function *func) noexcept {
    if (auto iter = _xir_to_llvm_global.find(func); iter != _xir_to_llvm_global.end()) {
        LUISA_DEBUG_ASSERT(llvm::isa<llvm::Function>(iter->second), "Global is not a function.");
        return static_cast<llvm::Function *>(iter->second);
    }
    auto llvm_func = [this, func]() noexcept {
        switch (func->derived_function_tag()) {
            case xir::DerivedFunctionTag::KERNEL: return _declare_llvm_kernel_function(static_cast<const xir::KernelFunction *>(func));
            case xir::DerivedFunctionTag::CALLABLE: return _declare_llvm_callable_function(static_cast<const xir::CallableFunction *>(func));
            case xir::DerivedFunctionTag::EXTERNAL: return _declare_llvm_external_function(static_cast<const xir::ExternalFunction *>(func));
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Unsupported function type.");
    }();
    auto [iter, success] = _xir_to_llvm_global.try_emplace(func, llvm_func);
    LUISA_ASSERT(success, "Failed to insert LLVM function.");
    return llvm_func;
}

llvm::Function *CUDACodegenLLVMImpl::_declare_llvm_kernel_function(const xir::KernelFunction *func) noexcept {
    auto [llvm_func_type, llvm_func_name] = [&]() noexcept -> std::pair<llvm::FunctionType *, llvm::StringRef> {
        auto llvm_void_type = llvm::Type::getVoidTy(_llvm_context);
        if (_rt_analysis.uses_ray_tracing) {// ray tracing kernels use constant memory for args
            return std::make_pair(llvm::FunctionType::get(llvm_void_type, {}, false), "__raygen__main");
        }
        // normal kernels use direct arguments
        auto arg_struct_info = _get_kernel_argument_struct(func);
        return std::make_pair(llvm::FunctionType::get(llvm_void_type, {arg_struct_info->llvm_type}, false), "kernel_main");
    }();
    auto llvm_kernel = llvm::Function::Create(llvm_func_type, llvm::Function::ExternalLinkage, llvm_func_name, _llvm_module.get());
    llvm_kernel->setCallingConv(llvm::CallingConv::PTX_Kernel);
    return llvm_kernel;
}

llvm::Function *CUDACodegenLLVMImpl::_declare_llvm_callable_function(const xir::CallableFunction *func) noexcept {
    llvm::SmallVector<llvm::Type *> llvm_arg_types;
    for (auto arg : func->arguments()) {
        if (arg->is_reference()) {
            llvm_arg_types.emplace_back(llvm::PointerType::get(_llvm_context, 0));
        } else {
            llvm_arg_types.emplace_back(_get_llvm_type(arg->type())->reg_type);
        }
    }
    // the last two arguments are dispatch_size and kernel_id
    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
    auto llvm_i32x3_type = llvm::VectorType::get(llvm_i32_type, 3, false);
    llvm_arg_types.emplace_back(llvm_i32x3_type);// dispatch_size
    llvm_arg_types.emplace_back(llvm_i32_type);  // kernel_id
    auto llvm_ret_type = func->type() == nullptr ? llvm::Type::getVoidTy(_llvm_context) :
                                                   _get_llvm_type(func->type())->reg_type;
    auto llvm_func_type = llvm::FunctionType::get(llvm_ret_type, llvm_arg_types, false);
    auto llvm_func = llvm::Function::Create(llvm_func_type, llvm::Function::PrivateLinkage, 0,
                                            func->name().value_or("callable"), _llvm_module.get());
    llvm_func->setCallingConv(llvm::CallingConv::PTX_Device);
    return llvm_func;
}

llvm::Function *CUDACodegenLLVMImpl::_declare_llvm_external_function(const xir::ExternalFunction *func) noexcept {
    LUISA_NOT_IMPLEMENTED("External function declaration not implemented.");
}

llvm::Function *CUDACodegenLLVMImpl::_translate_function(const xir::FunctionDefinition *func) noexcept {
    switch (func->derived_function_tag()) {
        case xir::DerivedFunctionTag::KERNEL: return _translate_kernel_function(static_cast<const xir::KernelFunction *>(func));
        case xir::DerivedFunctionTag::CALLABLE: return _translate_callable_function(static_cast<const xir::CallableFunction *>(func));
        case xir::DerivedFunctionTag::EXTERNAL: LUISA_ERROR_WITH_LOCATION("Cannot translate external function.");
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported function type.");
}

llvm::Function *CUDACodegenLLVMImpl::_translate_kernel_function(const xir::KernelFunction *func) noexcept {
    auto arg_struct_info = _get_kernel_argument_struct(func);
    auto llvm_kernel = _get_or_declare_llvm_function(func);
    LUISA_DEBUG_ASSERT(llvm_kernel->isDeclaration(), "Kernel function already defined.");
    FunctionContext func_ctx{llvm_kernel};
    // load arguments
    IB b{func_ctx.llvm_entry_block};
    auto llvm_arg_struct = [&]() noexcept -> llvm::Value * {
        // ray tracing kernels use constant memory for args
        if (_rt_analysis.uses_ray_tracing) {
            auto llvm_global_arg = new ::llvm::GlobalVariable{
                *_llvm_module, arg_struct_info->llvm_type, true, llvm::GlobalValue::ExternalLinkage,
                nullptr, "params", nullptr, llvm::GlobalValue::NotThreadLocal,
                nvptx_address_space_constant, true};
            llvm::Align llvm_align{KernelArgumentStruct::argument_alignment};
            llvm_global_arg->setAlignment(llvm_align);
            return b.CreateAlignedLoad(arg_struct_info->llvm_type, llvm_global_arg, llvm_align, "params.load");
        }
        // normal kernels use direct arguments
        return llvm_kernel->getArg(0);
    }();
    // map arguments to local values
    auto arg_index = 0u;
    for (auto arg : func->arguments()) {
        auto member_index = arg_struct_info->argument_indices[arg_index];
        auto llvm_member_mem = b.CreateExtractValue(llvm_arg_struct, member_index, arg->name().value_or(""));
        auto llvm_member_reg = arg->is_value() ? _convert_llvm_mem_value_to_reg(b, llvm_member_mem, arg->type()) : llvm_member_mem;
        func_ctx.local_values.try_emplace(arg, llvm_member_reg);
        // create assumptions for bound textures' storage modes
        if (arg_index < _config.bindings.size() && arg->is_resource() && arg->type()->is_texture()) {
            if (auto binding = luisa::get_if<Function::TextureBinding>(&_config.bindings[arg_index])) {
                auto storage = reinterpret_cast<CUDATexture *>(binding->handle)->storage();
                auto llvm_storage = b.CreateExtractValue(llvm_member_reg, llvm_texture_type_storage_index);
                auto llvm_same_storage = b.CreateICmpEQ(llvm_storage, b.getInt64(luisa::to_underlying(storage)));
                b.CreateAssumption(llvm_same_storage);
            }
        }
        arg_index++;
    }
    // load dispatch_size_and_kernel_id
    auto llvm_dispatch_size_and_kernel_id = b.CreateExtractValue(llvm_arg_struct, arg_struct_info->dispatch_size_and_kernel_id_index);
    if (_rt_analysis.uses_ray_tracing) {// for OptiX kernels, we can use the built-in dispatch size
        auto llvm_dispatch_size_x = b.CreateCall(_get_inline_asm("call ($0), _optix_get_launch_dimension_x, ();", "=r", false), {});
        auto llvm_dispatch_size_y = b.CreateCall(_get_inline_asm("call ($0), _optix_get_launch_dimension_y, ();", "=r", false), {});
        auto llvm_dispatch_size_z = b.CreateCall(_get_inline_asm("call ($0), _optix_get_launch_dimension_z, ();", "=r", false), {});
        func_ctx.llvm_dispatch_size = _create_llvm_vector(b, {llvm_dispatch_size_x, llvm_dispatch_size_y, llvm_dispatch_size_z});
    } else {// for normal kernels, we read the dispatch size from arguments
        auto llvm_dispatch_size_x = b.CreateExtractValue(llvm_dispatch_size_and_kernel_id, 0);
        b.CreateAssumption(b.CreateICmpUGT(llvm_dispatch_size_x, b.getInt32(0)));
        auto llvm_dispatch_size_y = b.CreateExtractValue(llvm_dispatch_size_and_kernel_id, 1);
        b.CreateAssumption(b.CreateICmpUGT(llvm_dispatch_size_y, b.getInt32(0)));
        auto llvm_dispatch_size_z = b.CreateExtractValue(llvm_dispatch_size_and_kernel_id, 2);
        b.CreateAssumption(b.CreateICmpUGT(llvm_dispatch_size_z, b.getInt32(0)));
        func_ctx.llvm_dispatch_size = _create_llvm_vector(b, {llvm_dispatch_size_x, llvm_dispatch_size_y, llvm_dispatch_size_z});
        func_ctx.llvm_dispatch_size->setName("sreg.dispatch.size");
    }
    func_ctx.llvm_kernel_id = b.CreateExtractValue(llvm_dispatch_size_and_kernel_id, 3, "sreg.kernel.id");
    // translate body
    auto llvm_body = _translate_function_definition(func_ctx, func);
    // create guard for out-of-bounds threads if not ray tracing (OptiX will do this for us)
    if (!_rt_analysis.uses_ray_tracing) {
        auto llvm_dispatch_id = _read_dispatch_id(b, func_ctx);
        auto llvm_dispatch_id_in_bounds = b.CreateICmpULT(llvm_dispatch_id, func_ctx.llvm_dispatch_size, "dispatch.id.in.bounds");
        // if some axes of the block size is 1, we can assume those axes are always in bounds
        for (int i = 0; i < 3; i++) {
            if (_config.block_size[i] == 1) {
                llvm_dispatch_id_in_bounds = b.CreateInsertElement(llvm_dispatch_id_in_bounds, b.getInt1(true), i);
            }
        }
        auto llvm_dispatch_id_in_bounds_all = b.CreateAndReduce(llvm_dispatch_id_in_bounds);
        auto llvm_exit_block = llvm::BasicBlock::Create(_llvm_context, "exit.early", llvm_kernel);
        b.CreateCondBr(llvm_dispatch_id_in_bounds_all, llvm_body, llvm_exit_block);
        // exit block
        b.SetInsertPoint(llvm_exit_block);
        b.CreateRetVoid();
    } else {
        b.CreateBr(llvm_body);
    }
    // branch from entry to body
    return llvm_kernel;
}

llvm::Function *CUDACodegenLLVMImpl::_translate_callable_function(const xir::CallableFunction *func) noexcept {
    auto llvm_func = _get_or_declare_llvm_function(func);
    LUISA_DEBUG_ASSERT(llvm_func->isDeclaration(), "Callable function already defined.");
    FunctionContext func_ctx{llvm_func};
    auto llvm_arg_iter = llvm_func->arg_begin();
    for (auto arg : func->arguments()) {
        func_ctx.local_values.try_emplace(arg, llvm_arg_iter++);
    }
    func_ctx.llvm_dispatch_size = llvm_arg_iter++;
    func_ctx.llvm_dispatch_size->setName("sreg.dispatch.size");
    func_ctx.llvm_kernel_id = llvm_arg_iter++;
    func_ctx.llvm_kernel_id->setName("sreg.kernel.id");
    auto body = _translate_function_definition(func_ctx, func);
    // branch from entry to body
    IB b{func_ctx.llvm_entry_block};
    b.CreateBr(body);
    return llvm_func;
}

namespace {

template<typename F>
void luisa_compute_cuda_codegen_llvm_traverse_dom_tree_impl(luisa::unordered_set<const xir::DomTreeNode *> &visited,
                                                            const xir::DomTreeNode *node, const F &f) noexcept {
    if (visited.emplace(node).second) [[likely]] {
        f(node->block());
        for (auto child : node->children()) {
            luisa_compute_cuda_codegen_llvm_traverse_dom_tree_impl(visited, child, f);
        }
    }
}

template<typename F>
void luisa_compute_cuda_codegen_llvm_traverse_dom_tree(const xir::DomTree &tree, const F &f) noexcept {
    luisa::unordered_set<const xir::DomTreeNode *> visited;
    luisa_compute_cuda_codegen_llvm_traverse_dom_tree_impl(visited, tree.root(), f);
}

}// namespace

llvm::BasicBlock *CUDACodegenLLVMImpl::_translate_function_definition(FunctionContext &func_ctx, const xir::FunctionDefinition *f) noexcept {
    for (auto bb : f->basic_blocks()) {
        auto llvm_bb = llvm::BasicBlock::Create(_llvm_context, bb->name().value_or(""), func_ctx.llvm_func);
        func_ctx.local_values.try_emplace(bb, llvm_bb);
    }
    // generate code for each basic block in dominance order, so that
    // used values are always defined before use except for phi nodes,
    // which are handled separately after all blocks are generated
    auto dom_tree = xir::compute_dom_tree(const_cast<xir::FunctionDefinition *>(f));
    LUISA_ASSERT(dom_tree.root()->block() == f->body_block());
    luisa_compute_cuda_codegen_llvm_traverse_dom_tree(dom_tree, [this, &func_ctx](const xir::BasicBlock *bb) noexcept {
        auto llvm_bb = func_ctx.get_local_value<llvm::BasicBlock>(bb);
        IB b{llvm_bb};
        for (auto inst : bb->instructions()) {
            _translate_instruction(b, func_ctx, inst);
        }
    });
    // finalize phi nodes
    _finalize_pending_phi_nodes(func_ctx);
    // Assert that all XIR blocks have been properly translated with terminators.
    // If this fires, it indicates an upstream XIR pass left an empty/orphaned block.
    for (auto bb : f->basic_blocks()) {
        auto llvm_bb = func_ctx.get_local_value<llvm::BasicBlock>(bb);
        LUISA_ASSERT(llvm_bb->getTerminator() != nullptr,
                     "LLVM basic block has no terminator after translation. "
                     "This likely indicates a bug in an XIR optimization pass "
                     "(e.g. DCE incorrectly removing break/continue instructions).");
    }
    return func_ctx.get_local_value<llvm::BasicBlock>(f->body_block());
}

void CUDACodegenLLVMImpl::_mark_llvm_function_as_pure(llvm::Function *func) noexcept {
    func->addFnAttr(llvm::Attribute::NoCallback);
    func->setMustProgress();
    func->setDoesNotFreeMemory();
    func->setNoSync();
    func->setDoesNotThrow();
    func->setSpeculatable();
    func->setWillReturn();
    func->setDoesNotAccessMemory();
}

llvm::Function *CUDACodegenLLVMImpl::_get_assert_function() noexcept {
    if (auto llvm_f = _llvm_module->getFunction("luisa.assert")) {
        return llvm_f;
    }
    auto llvm_i1_type = llvm::Type::getInt1Ty(_llvm_context);
    auto llvm_const_ptr_type = llvm::PointerType::get(_llvm_context, nvptx_address_space_constant);
    auto llvm_func_type = llvm::FunctionType::get(llvm::Type::getVoidTy(_llvm_context),
                                                  {llvm_i1_type, llvm_const_ptr_type}, false);
    auto llvm_f = llvm::Function::Create(llvm_func_type,
                                         llvm::Function::PrivateLinkage,
                                         "luisa.assert", *_llvm_module);
    auto llvm_entry = llvm::BasicBlock::Create(_llvm_context, "entry", llvm_f);
    IB b{llvm_entry};
    auto llvm_cond = llvm_f->getArg(0);
    auto llvm_msg = llvm_f->getArg(1);
    llvm_cond->setName("cond");
    llvm_msg->setName("message");
    auto llvm_then_bb = llvm::BasicBlock::Create(_llvm_context, "then", llvm_f);
    auto llvm_trap_bb = llvm::BasicBlock::Create(_llvm_context, "trap", llvm_f);
    b.CreateCondBr(llvm_cond, llvm_then_bb, llvm_trap_bb);
    // then block
    b.SetInsertPoint(llvm_then_bb);
    b.CreateRetVoid();
    // trap block
    b.SetInsertPoint(llvm_trap_bb);
    auto llvm_vprintf = _get_vprintf_function();
    auto llvm_generic_ptr_type = llvm::PointerType::get(_llvm_context, 0);
    auto llvm_msg_p0 = b.CreateAddrSpaceCast(llvm_msg, llvm_generic_ptr_type);
    auto llvm_null_p0 = llvm::ConstantPointerNull::get(llvm_generic_ptr_type);
    b.CreateCall(llvm_vprintf, {llvm_msg_p0, llvm_null_p0});
    b.CreateIntrinsic(b.getVoidTy(), llvm::Intrinsic::trap, {});
    b.CreateUnreachable();
    return llvm_f;
}

llvm::Function *CUDACodegenLLVMImpl::_get_vprintf_function() noexcept {
    if (auto llvm_f = _llvm_module->getFunction("vprintf")) { return llvm_f; }
    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
    auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, 0);
    auto llvm_func_type = llvm::FunctionType::get(llvm_i32_type, {llvm_ptr_type, llvm_ptr_type}, false);
    return llvm::Function::Create(llvm_func_type, llvm::Function::ExternalLinkage, "vprintf", *_llvm_module);
}

// <4 x type> luisa.cuda.texture.2d.read.<type>(i64 handle, i64 storage, <2 x i32> coord)
//
// i16 @llvm.nvvm.suld.2d.i8.<clamp>(i64 %tex, i32 %x, i32 %y)
// i16 @llvm.nvvm.suld.2d.i16.<clamp>(i64 %tex, i32 %x, i32 %y)
// i32 @llvm.nvvm.suld.2d.i32.<clamp>(i64 %tex, i32 %x, i32 %y)
// i64 @llvm.nvvm.suld.2d.i64.<clamp>(i64 %tex, i32 %x, i32 %y)
//
// %short2 @llvm.nvvm.suld.2d.v2i8.<clamp>(i64 %tex, i32 %x, i32 %y)
// %short2 @llvm.nvvm.suld.2d.v2i16.<clamp>(i64 %tex, i32 %x, i32 %y)
// %int2 @llvm.nvvm.suld.2d.v2i32.<clamp>(i64 %tex, i32 %x, i32 %y)
// %long2 @llvm.nvvm.suld.2d.v2i64.<clamp>(i64 %tex, i32 %x, i32 %y)
//
// %short4 @llvm.nvvm.suld.2d.v4i8.<clamp>(i64 %tex, i32 %x, i32 %y)
// %short4 @llvm.nvvm.suld.2d.v4i16.<clamp>(i64 %tex, i32 %x, i32 %y)
// %int4 @llvm.nvvm.suld.2d.v4i32.<clamp>(i64 %tex, i32 %x, i32 %y)
llvm::Function *CUDACodegenLLVMImpl::_get_texture2d_read_function(llvm::VectorType *llvm_value_type) noexcept {
    auto name = fmt::format("luisa.cuda.texture.2d.read.{}", _to_string(llvm_value_type->getElementType()));
    if (auto llvm_func = _llvm_module->getFunction(name)) { return llvm_func; }
    auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
    auto llvm_coord_type = llvm::VectorType::get(llvm_i32_type, 2, false);
    auto llvm_func_type = llvm::FunctionType::get(
        llvm_value_type, {llvm_i64_type, llvm_i64_type, llvm_coord_type}, false);
    auto llvm_func = llvm::Function::Create(llvm_func_type, llvm::Function::PrivateLinkage, name, *_llvm_module);
    llvm_func->addFnAttr(llvm::Attribute::AlwaysInline);
    auto llvm_entry = llvm::BasicBlock::Create(_llvm_context, "entry", llvm_func);
    IB b{llvm_entry};
    auto llvm_handle = llvm_func->getArg(0);
    llvm_handle->setName("surface.handle");
    auto llvm_storage = llvm_func->getArg(1);
    llvm_storage->setName("surface.storage");
    auto llvm_coord = llvm_func->getArg(2);
    llvm_coord->setName("coord");
    auto llvm_coord_x = b.CreateExtractElement(llvm_coord, b.getInt64(0), "coord.x");
    auto llvm_coord_y = b.CreateExtractElement(llvm_coord, b.getInt64(1), "coord.y");
    auto llvm_default_block = llvm::BasicBlock::Create(_llvm_context, "switch.default", llvm_func);
    auto llvm_switch = b.CreateSwitch(llvm_storage, llvm_default_block, 16);
    auto create_case = [&](PixelStorage storage, llvm::Intrinsic::ID intrinsic, llvm::Type *llvm_channel_type) noexcept {
        auto llvm_case_block = llvm::BasicBlock::Create(_llvm_context, fmt::format("switch.case.{}", luisa::to_string(storage)), llvm_func);
        llvm_switch->addCase(b.getInt64(luisa::to_underlying(storage)), llvm_case_block);
        b.SetInsertPoint(llvm_case_block);
        // call suld intrinsic
        auto pixel_size = pixel_storage_size(storage, luisa::make_uint3(1));
        auto llvm_coord_x_bytes = b.CreateMul(llvm_coord_x, b.getInt32(pixel_size), "coord.x.bytes", true, true);
        auto llvm_raw = b.CreateIntrinsic(intrinsic, {llvm_handle, llvm_coord_x_bytes, llvm_coord_y});
        // convert raw to vector type
        auto channel_count = pixel_storage_channel_count(storage);
        LUISA_DEBUG_ASSERT(channel_count == 1 || channel_count == 2 || channel_count == 4);
        auto llvm_src_type = llvm::VectorType::get(llvm_channel_type, 4, false);
        auto llvm_src = static_cast<llvm::Value *>(llvm::Constant::getNullValue(llvm_src_type));
        if (channel_count == 1) {
            auto llvm_channel = llvm_channel_type->isIntegerTy(8) ?
                                    b.CreateTrunc(llvm_raw, llvm_channel_type) :
                                    b.CreateBitCast(llvm_raw, llvm_channel_type);
            llvm_src = b.CreateInsertElement(llvm_src, llvm_channel, b.getInt64(0));
        } else {
            for (auto i = 0; i < channel_count; i++) {
                auto llvm_channel = b.CreateExtractValue(llvm_raw, i);
                llvm_channel = llvm_channel_type->isIntegerTy(8) ?
                                   b.CreateTrunc(llvm_channel, llvm_channel_type) :
                                   b.CreateBitCast(llvm_channel, llvm_channel_type);
                llvm_src = b.CreateInsertElement(llvm_src, llvm_channel, i);
            }
        }
        // cast to the dst type
        auto llvm_dst = _texel_cast(b, llvm_src, llvm_value_type);
        b.CreateRet(llvm_dst);
    };
    auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
    auto llvm_i16_type = llvm::Type::getInt16Ty(_llvm_context);
    auto llvm_f16_type = llvm::Type::getHalfTy(_llvm_context);
    auto llvm_f32_type = llvm::Type::getFloatTy(_llvm_context);
    create_case(PixelStorage::BYTE1, llvm::Intrinsic::nvvm_suld_2d_i8_zero, llvm_i8_type);
    create_case(PixelStorage::BYTE2, llvm::Intrinsic::nvvm_suld_2d_v2i8_zero, llvm_i8_type);
    create_case(PixelStorage::BYTE4, llvm::Intrinsic::nvvm_suld_2d_v4i8_zero, llvm_i8_type);
    create_case(PixelStorage::SHORT1, llvm::Intrinsic::nvvm_suld_2d_i16_zero, llvm_i16_type);
    create_case(PixelStorage::SHORT2, llvm::Intrinsic::nvvm_suld_2d_v2i16_zero, llvm_i16_type);
    create_case(PixelStorage::SHORT4, llvm::Intrinsic::nvvm_suld_2d_v4i16_zero, llvm_i16_type);
    create_case(PixelStorage::INT1, llvm::Intrinsic::nvvm_suld_2d_i32_zero, llvm_i32_type);
    create_case(PixelStorage::INT2, llvm::Intrinsic::nvvm_suld_2d_v2i32_zero, llvm_i32_type);
    create_case(PixelStorage::INT4, llvm::Intrinsic::nvvm_suld_2d_v4i32_zero, llvm_i32_type);
    create_case(PixelStorage::HALF1, llvm::Intrinsic::nvvm_suld_2d_i16_zero, llvm_f16_type);
    create_case(PixelStorage::HALF2, llvm::Intrinsic::nvvm_suld_2d_v2i16_zero, llvm_f16_type);
    create_case(PixelStorage::HALF4, llvm::Intrinsic::nvvm_suld_2d_v4i16_zero, llvm_f16_type);
    create_case(PixelStorage::FLOAT1, llvm::Intrinsic::nvvm_suld_2d_i32_zero, llvm_f32_type);
    create_case(PixelStorage::FLOAT2, llvm::Intrinsic::nvvm_suld_2d_v2i32_zero, llvm_f32_type);
    create_case(PixelStorage::FLOAT4, llvm::Intrinsic::nvvm_suld_2d_v4i32_zero, llvm_f32_type);
    // default block is unreachable
    b.SetInsertPoint(llvm_default_block);
    b.CreateUnreachable();
    return llvm_func;
}

// <4 x type> luisa.cuda.texture.3d.read.<type>(i64 handle, i64 storage, <3 x i32> coord)
//
// i16 @llvm.nvvm.suld.3d.i8.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %z)
// i16 @llvm.nvvm.suld.3d.i16.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %z)
// i32 @llvm.nvvm.suld.3d.i32.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %z)
// i64 @llvm.nvvm.suld.3d.i64.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %z)
//
// %short2 @llvm.nvvm.suld.3d.v2i8.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %z)
// %short2 @llvm.nvvm.suld.3d.v2i16.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %z)
// %int2 @llvm.nvvm.suld.3d.v2i32.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %z)
// %long2 @llvm.nvvm.suld.3d.v2i64.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %z)
//
// %short4 @llvm.nvvm.suld.3d.v4i8.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %z)
// %short4 @llvm.nvvm.suld.3d.v4i16.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %z)
// %int4 @llvm.nvvm.suld.3d.v4i32.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %z)
llvm::Function *CUDACodegenLLVMImpl::_get_texture3d_read_function(llvm::VectorType *llvm_value_type) noexcept {
    auto name = fmt::format("luisa.cuda.texture.3d.read.{}", _to_string(llvm_value_type->getElementType()));
    if (auto llvm_func = _llvm_module->getFunction(name)) { return llvm_func; }
    auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
    auto llvm_coord_type = llvm::VectorType::get(llvm_i32_type, 3, false);
    auto llvm_func_type = llvm::FunctionType::get(
        llvm_value_type, {llvm_i64_type, llvm_i64_type, llvm_coord_type}, false);
    auto llvm_func = llvm::Function::Create(llvm_func_type, llvm::Function::PrivateLinkage, name, *_llvm_module);
    llvm_func->addFnAttr(llvm::Attribute::AlwaysInline);
    auto llvm_entry = llvm::BasicBlock::Create(_llvm_context, "entry", llvm_func);
    IB b{llvm_entry};
    auto llvm_handle = llvm_func->getArg(0);
    llvm_handle->setName("surface.handle");
    auto llvm_storage = llvm_func->getArg(1);
    llvm_storage->setName("surface.storage");
    auto llvm_coord = llvm_func->getArg(2);
    llvm_coord->setName("coord");
    auto llvm_coord_x = b.CreateExtractElement(llvm_coord, b.getInt64(0), "coord.x");
    auto llvm_coord_y = b.CreateExtractElement(llvm_coord, b.getInt64(1), "coord.y");
    auto llvm_coord_z = b.CreateExtractElement(llvm_coord, b.getInt64(2), "coord.z");
    auto llvm_default_block = llvm::BasicBlock::Create(_llvm_context, "switch.default", llvm_func);
    auto llvm_switch = b.CreateSwitch(llvm_storage, llvm_default_block, 16);
    auto create_case = [&](PixelStorage storage, llvm::Intrinsic::ID intrinsic, llvm::Type *llvm_channel_type) noexcept {
        auto llvm_case_block = llvm::BasicBlock::Create(_llvm_context, fmt::format("switch.case.{}", luisa::to_string(storage)), llvm_func);
        llvm_switch->addCase(b.getInt64(luisa::to_underlying(storage)), llvm_case_block);
        b.SetInsertPoint(llvm_case_block);
        // call suld intrinsic
        auto pixel_size = pixel_storage_size(storage, luisa::make_uint3(1));
        auto llvm_coord_x_bytes = b.CreateMul(llvm_coord_x, b.getInt32(pixel_size), "coord.x.bytes", true, true);
        auto llvm_raw = b.CreateIntrinsic(intrinsic, {llvm_handle, llvm_coord_x_bytes, llvm_coord_y, llvm_coord_z});
        // convert raw to vector type
        auto channel_count = pixel_storage_channel_count(storage);
        LUISA_DEBUG_ASSERT(channel_count == 1 || channel_count == 2 || channel_count == 4);
        auto llvm_src_type = llvm::VectorType::get(llvm_channel_type, 4, false);
        auto llvm_src = static_cast<llvm::Value *>(llvm::Constant::getNullValue(llvm_src_type));
        if (channel_count == 1) {
            auto llvm_channel = llvm_channel_type->isIntegerTy(8) ?
                                    b.CreateTrunc(llvm_raw, llvm_channel_type) :
                                    b.CreateBitCast(llvm_raw, llvm_channel_type);
            llvm_src = b.CreateInsertElement(llvm_src, llvm_channel, b.getInt64(0));
        } else {
            for (auto i = 0; i < channel_count; i++) {
                auto llvm_channel = b.CreateExtractValue(llvm_raw, i);
                llvm_channel = llvm_channel_type->isIntegerTy(8) ?
                                   b.CreateTrunc(llvm_channel, llvm_channel_type) :
                                   b.CreateBitCast(llvm_channel, llvm_channel_type);
                llvm_src = b.CreateInsertElement(llvm_src, llvm_channel, i);
            }
        }
        // cast to the dst type
        auto llvm_dst = _texel_cast(b, llvm_src, llvm_value_type);
        b.CreateRet(llvm_dst);
    };
    auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
    auto llvm_i16_type = llvm::Type::getInt16Ty(_llvm_context);
    auto llvm_f16_type = llvm::Type::getHalfTy(_llvm_context);
    auto llvm_f32_type = llvm::Type::getFloatTy(_llvm_context);
    create_case(PixelStorage::BYTE1, llvm::Intrinsic::nvvm_suld_3d_i8_zero, llvm_i8_type);
    create_case(PixelStorage::BYTE2, llvm::Intrinsic::nvvm_suld_3d_v2i8_zero, llvm_i8_type);
    create_case(PixelStorage::BYTE4, llvm::Intrinsic::nvvm_suld_3d_v4i8_zero, llvm_i8_type);
    create_case(PixelStorage::SHORT1, llvm::Intrinsic::nvvm_suld_3d_i16_zero, llvm_i16_type);
    create_case(PixelStorage::SHORT2, llvm::Intrinsic::nvvm_suld_3d_v2i16_zero, llvm_i16_type);
    create_case(PixelStorage::SHORT4, llvm::Intrinsic::nvvm_suld_3d_v4i16_zero, llvm_i16_type);
    create_case(PixelStorage::INT1, llvm::Intrinsic::nvvm_suld_3d_i32_zero, llvm_i32_type);
    create_case(PixelStorage::INT2, llvm::Intrinsic::nvvm_suld_3d_v2i32_zero, llvm_i32_type);
    create_case(PixelStorage::INT4, llvm::Intrinsic::nvvm_suld_3d_v4i32_zero, llvm_i32_type);
    create_case(PixelStorage::HALF1, llvm::Intrinsic::nvvm_suld_3d_i16_zero, llvm_f16_type);
    create_case(PixelStorage::HALF2, llvm::Intrinsic::nvvm_suld_3d_v2i16_zero, llvm_f16_type);
    create_case(PixelStorage::HALF4, llvm::Intrinsic::nvvm_suld_3d_v4i16_zero, llvm_f16_type);
    create_case(PixelStorage::FLOAT1, llvm::Intrinsic::nvvm_suld_3d_i32_zero, llvm_f32_type);
    create_case(PixelStorage::FLOAT2, llvm::Intrinsic::nvvm_suld_3d_v2i32_zero, llvm_f32_type);
    create_case(PixelStorage::FLOAT4, llvm::Intrinsic::nvvm_suld_3d_v4i32_zero, llvm_f32_type);
    // default block is unreachable
    b.SetInsertPoint(llvm_default_block);
    b.CreateUnreachable();
    return llvm_func;
}

// void luisa.cuda.texture.2d.write.<type>(i64 handle, i64 storage, <2 x i32> coord, <4 x type> value)
//
// void @llvm.nvvm.sust.b.2d.i8.<clamp>(i64 %tex, i32 %x, i32 %y, i16 %r)
// void @llvm.nvvm.sust.b.2d.i16.<clamp>(i64 %tex, i32 %x, i32 %y, i16 %r)
// void @llvm.nvvm.sust.b.2d.i32.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %r)
// void @llvm.nvvm.sust.b.2d.i64.<clamp>(i64 %tex, i32 %x, i32 %y, i64 %r)
//
// void @llvm.nvvm.sust.b.2d.v2i8.<clamp>(i64 %tex, i32 %x, i32 %y,
//                                        i16 %r, i16 %g)
// void @llvm.nvvm.sust.b.2d.v2i16.<clamp>(i64 %tex, i32 %x, i32 %y,
//                                         i16 %r, i16 %g)
// void @llvm.nvvm.sust.b.2d.v2i32.<clamp>(i64 %tex, i32 %x, i32 %y,
//                                         i32 %r, i32 %g)
// void @llvm.nvvm.sust.b.2d.v2i64.<clamp>(i64 %tex, i32 %x, i32 %y,
//                                         i64 %r, i64 %g)
//
// void @llvm.nvvm.sust.b.2d.v4i8.<clamp>(i64 %tex, i32 %x, i32 %y,
//                                        i16 %r, i16 %g, i16 %b, i16 %a)
// void @llvm.nvvm.sust.b.2d.v4i16.<clamp>(i64 %tex, i32 %x, i32 %y,
//                                         i16 %r, i16 %g, i16 %b, i16 %a)
// void @llvm.nvvm.sust.b.2d.v4i32.<clamp>(i64 %tex, i32 %x, i32 %y,
//                                         i32 %r, i32 %g, i32 %b, i32 %a)
llvm::Function *CUDACodegenLLVMImpl::_get_texture2d_write_function(llvm::VectorType *llvm_value_type) noexcept {
    auto name = fmt::format("luisa.cuda.texture.2d.write.{}", _to_string(llvm_value_type->getElementType()));
    if (auto llvm_func = _llvm_module->getFunction(name)) { return llvm_func; }
    auto llvm_void_type = llvm::Type::getVoidTy(_llvm_context);
    auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
    auto llvm_coord_type = llvm::VectorType::get(llvm_i32_type, 2, false);
    auto llvm_func_type = llvm::FunctionType::get(
        llvm_void_type, {llvm_i64_type, llvm_i64_type, llvm_coord_type, llvm_value_type}, false);
    auto llvm_func = llvm::Function::Create(llvm_func_type, llvm::Function::PrivateLinkage, name, *_llvm_module);
    llvm_func->addFnAttr(llvm::Attribute::AlwaysInline);
    auto llvm_entry = llvm::BasicBlock::Create(_llvm_context, "entry", llvm_func);
    IB b{llvm_entry};
    auto llvm_handle = llvm_func->getArg(0);
    llvm_handle->setName("surface.handle");
    auto llvm_storage = llvm_func->getArg(1);
    llvm_storage->setName("surface.storage");
    auto llvm_coord = llvm_func->getArg(2);
    llvm_coord->setName("coord");
    auto llvm_coord_x = b.CreateExtractElement(llvm_coord, b.getInt64(0), "coord.x");
    auto llvm_coord_y = b.CreateExtractElement(llvm_coord, b.getInt64(1), "coord.y");
    auto llvm_value = llvm_func->getArg(3);
    llvm_value->setName("value");
    auto llvm_default_block = llvm::BasicBlock::Create(_llvm_context, "switch.default", llvm_func);
    auto llvm_switch = b.CreateSwitch(llvm_storage, llvm_default_block, 16);
    auto create_case = [&](PixelStorage storage, llvm::Intrinsic::ID intrinsic, llvm::Type *llvm_channel_type, llvm::Type *llvm_storage_channel_type) noexcept {
        auto llvm_case_block = llvm::BasicBlock::Create(_llvm_context, fmt::format("switch.case.{}", luisa::to_string(storage)), llvm_func);
        llvm_switch->addCase(b.getInt64(luisa::to_underlying(storage)), llvm_case_block);
        b.SetInsertPoint(llvm_case_block);
        // cast value to pixel format
        auto channel_count = pixel_storage_channel_count(storage);
        LUISA_DEBUG_ASSERT(channel_count == 1 || channel_count == 2 || channel_count == 4);
        auto llvm_dst_type = llvm::VectorType::get(llvm_channel_type, 4, false);
        auto llvm_dst = _texel_cast(b, llvm_value, llvm_dst_type);
        auto llvm_raw_type = llvm::VectorType::get(llvm_storage_channel_type, 4, false);
        auto llvm_raw = llvm_channel_type->isIntegerTy(8) ? b.CreateZExt(llvm_dst, llvm_raw_type) : b.CreateBitCast(llvm_dst, llvm_raw_type);
        llvm::SmallVector<llvm::Value *, 7> llvm_args;
        llvm_args.emplace_back(llvm_handle);
        auto pixel_size = pixel_storage_size(storage, luisa::make_uint3(1));
        auto llvm_coord_x_bytes = b.CreateMul(llvm_coord_x, b.getInt32(pixel_size), "coord.x.bytes", true, true);
        llvm_args.emplace_back(llvm_coord_x_bytes);
        llvm_args.emplace_back(llvm_coord_y);
        for (auto i = 0; i < channel_count; i++) {
            auto llvm_channel = b.CreateExtractElement(llvm_raw, i);
            llvm_args.emplace_back(llvm_channel);
        }
        b.CreateIntrinsic(intrinsic, llvm_args);
        b.CreateRetVoid();
    };
    auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
    auto llvm_i16_type = llvm::Type::getInt16Ty(_llvm_context);
    auto llvm_f16_type = llvm::Type::getHalfTy(_llvm_context);
    auto llvm_f32_type = llvm::Type::getFloatTy(_llvm_context);
    create_case(PixelStorage::BYTE1, llvm::Intrinsic::nvvm_sust_b_2d_i8_zero, llvm_i8_type, llvm_i16_type);
    create_case(PixelStorage::BYTE2, llvm::Intrinsic::nvvm_sust_b_2d_v2i8_zero, llvm_i8_type, llvm_i16_type);
    create_case(PixelStorage::BYTE4, llvm::Intrinsic::nvvm_sust_b_2d_v4i8_zero, llvm_i8_type, llvm_i16_type);
    create_case(PixelStorage::SHORT1, llvm::Intrinsic::nvvm_sust_b_2d_i16_zero, llvm_i16_type, llvm_i16_type);
    create_case(PixelStorage::SHORT2, llvm::Intrinsic::nvvm_sust_b_2d_v2i16_zero, llvm_i16_type, llvm_i16_type);
    create_case(PixelStorage::SHORT4, llvm::Intrinsic::nvvm_sust_b_2d_v4i16_zero, llvm_i16_type, llvm_i16_type);
    create_case(PixelStorage::INT1, llvm::Intrinsic::nvvm_sust_b_2d_i32_zero, llvm_i32_type, llvm_i32_type);
    create_case(PixelStorage::INT2, llvm::Intrinsic::nvvm_sust_b_2d_v2i32_zero, llvm_i32_type, llvm_i32_type);
    create_case(PixelStorage::INT4, llvm::Intrinsic::nvvm_sust_b_2d_v4i32_zero, llvm_i32_type, llvm_i32_type);
    create_case(PixelStorage::HALF1, llvm::Intrinsic::nvvm_sust_b_2d_i16_zero, llvm_f16_type, llvm_i16_type);
    create_case(PixelStorage::HALF2, llvm::Intrinsic::nvvm_sust_b_2d_v2i16_zero, llvm_f16_type, llvm_i16_type);
    create_case(PixelStorage::HALF4, llvm::Intrinsic::nvvm_sust_b_2d_v4i16_zero, llvm_f16_type, llvm_i16_type);
    create_case(PixelStorage::FLOAT1, llvm::Intrinsic::nvvm_sust_b_2d_i32_zero, llvm_f32_type, llvm_i32_type);
    create_case(PixelStorage::FLOAT2, llvm::Intrinsic::nvvm_sust_b_2d_v2i32_zero, llvm_f32_type, llvm_i32_type);
    create_case(PixelStorage::FLOAT4, llvm::Intrinsic::nvvm_sust_b_2d_v4i32_zero, llvm_f32_type, llvm_i32_type);
    // default block is unreachable
    b.SetInsertPoint(llvm_default_block);
    b.CreateUnreachable();
    return llvm_func;
}

// void luisa.cuda.texture.3d.write.<type>(i64 handle, i64 storage, <3 x i32> coord, <4 x type> value)
//
// void @llvm.nvvm.sust.b.3d.i8.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %z, i16 %r)
// void @llvm.nvvm.sust.b.3d.i16.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %z, i16 %r)
// void @llvm.nvvm.sust.b.3d.i32.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %z, i32 %r)
// void @llvm.nvvm.sust.b.3d.i64.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %z, i64 %r)
//
// void @llvm.nvvm.sust.b.3d.v2i8.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %z, i16 %r, i16 %g)
// void @llvm.nvvm.sust.b.3d.v2i16.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %z, i16 %r, i16 %g)
// void @llvm.nvvm.sust.b.3d.v2i32.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %z, i32 %r, i32 %g)
// void @llvm.nvvm.sust.b.3d.v2i64.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %z, i64 %r, i64 %g)
//
// void @llvm.nvvm.sust.b.3d.v4i8.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %z, i16 %r, i16 %g, i16 %b, i16 %a)
// void @llvm.nvvm.sust.b.3d.v4i16.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %z, i16 %r, i16 %g, i16 %b, i16 %a)
// void @llvm.nvvm.sust.b.3d.v4i32.<clamp>(i64 %tex, i32 %x, i32 %y, i32 %z, i32 %r, i32 %g, i32 %b, i32 %a)
llvm::Function *CUDACodegenLLVMImpl::_get_texture3d_write_function(llvm::VectorType *llvm_value_type) noexcept {
    auto name = fmt::format("luisa.cuda.texture.3d.write.{}", _to_string(llvm_value_type->getElementType()));
    if (auto llvm_func = _llvm_module->getFunction(name)) { return llvm_func; }
    auto llvm_void_type = llvm::Type::getVoidTy(_llvm_context);
    auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
    auto llvm_coord_type = llvm::VectorType::get(llvm_i32_type, 3, false);
    auto llvm_func_type = llvm::FunctionType::get(
        llvm_void_type, {llvm_i64_type, llvm_i64_type, llvm_coord_type, llvm_value_type}, false);
    auto llvm_func = llvm::Function::Create(llvm_func_type, llvm::Function::PrivateLinkage, name, *_llvm_module);
    llvm_func->addFnAttr(llvm::Attribute::AlwaysInline);
    auto llvm_entry = llvm::BasicBlock::Create(_llvm_context, "entry", llvm_func);
    IB b{llvm_entry};
    auto llvm_handle = llvm_func->getArg(0);
    llvm_handle->setName("surface.handle");
    auto llvm_storage = llvm_func->getArg(1);
    llvm_storage->setName("surface.storage");
    auto llvm_coord = llvm_func->getArg(2);
    llvm_coord->setName("coord");
    auto llvm_coord_x = b.CreateExtractElement(llvm_coord, b.getInt64(0), "coord.x");
    auto llvm_coord_y = b.CreateExtractElement(llvm_coord, b.getInt64(1), "coord.y");
    auto llvm_coord_z = b.CreateExtractElement(llvm_coord, b.getInt64(2), "coord.z");
    auto llvm_value = llvm_func->getArg(3);
    llvm_value->setName("value");
    auto llvm_default_block = llvm::BasicBlock::Create(_llvm_context, "switch.default", llvm_func);
    auto llvm_switch = b.CreateSwitch(llvm_storage, llvm_default_block, 16);
    auto create_case = [&](PixelStorage storage, llvm::Intrinsic::ID intrinsic, llvm::Type *llvm_channel_type, llvm::Type *llvm_storage_channel_type) noexcept {
        auto llvm_case_block = llvm::BasicBlock::Create(_llvm_context, fmt::format("switch.case.{}", luisa::to_string(storage)), llvm_func);
        llvm_switch->addCase(b.getInt64(luisa::to_underlying(storage)), llvm_case_block);
        b.SetInsertPoint(llvm_case_block);
        // cast value to pixel format
        auto channel_count = pixel_storage_channel_count(storage);
        LUISA_DEBUG_ASSERT(channel_count == 1 || channel_count == 2 || channel_count == 4);
        auto llvm_dst_type = llvm::VectorType::get(llvm_channel_type, 4, false);
        auto llvm_dst = _texel_cast(b, llvm_value, llvm_dst_type);
        auto llvm_raw_type = llvm::VectorType::get(llvm_storage_channel_type, 4, false);
        auto llvm_raw = llvm_channel_type->isIntegerTy(8) ? b.CreateZExt(llvm_dst, llvm_raw_type) : b.CreateBitCast(llvm_dst, llvm_raw_type);
        llvm::SmallVector<llvm::Value *, 8> llvm_args;
        llvm_args.emplace_back(llvm_handle);
        auto pixel_size = pixel_storage_size(storage, luisa::make_uint3(1));
        auto llvm_coord_x_bytes = b.CreateMul(llvm_coord_x, b.getInt32(pixel_size), "coord.x.bytes", true, true);
        llvm_args.emplace_back(llvm_coord_x_bytes);
        llvm_args.emplace_back(llvm_coord_y);
        llvm_args.emplace_back(llvm_coord_z);
        for (auto i = 0; i < channel_count; i++) {
            auto llvm_channel = b.CreateExtractElement(llvm_raw, i);
            llvm_args.emplace_back(llvm_channel);
        }
        b.CreateIntrinsic(intrinsic, llvm_args);
        b.CreateRetVoid();
    };
    auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
    auto llvm_i16_type = llvm::Type::getInt16Ty(_llvm_context);
    auto llvm_f16_type = llvm::Type::getHalfTy(_llvm_context);
    auto llvm_f32_type = llvm::Type::getFloatTy(_llvm_context);
    create_case(PixelStorage::BYTE1, llvm::Intrinsic::nvvm_sust_b_3d_i8_zero, llvm_i8_type, llvm_i16_type);
    create_case(PixelStorage::BYTE2, llvm::Intrinsic::nvvm_sust_b_3d_v2i8_zero, llvm_i8_type, llvm_i16_type);
    create_case(PixelStorage::BYTE4, llvm::Intrinsic::nvvm_sust_b_3d_v4i8_zero, llvm_i8_type, llvm_i16_type);
    create_case(PixelStorage::SHORT1, llvm::Intrinsic::nvvm_sust_b_3d_i16_zero, llvm_i16_type, llvm_i16_type);
    create_case(PixelStorage::SHORT2, llvm::Intrinsic::nvvm_sust_b_3d_v2i16_zero, llvm_i16_type, llvm_i16_type);
    create_case(PixelStorage::SHORT4, llvm::Intrinsic::nvvm_sust_b_3d_v4i16_zero, llvm_i16_type, llvm_i16_type);
    create_case(PixelStorage::INT1, llvm::Intrinsic::nvvm_sust_b_3d_i32_zero, llvm_i32_type, llvm_i32_type);
    create_case(PixelStorage::INT2, llvm::Intrinsic::nvvm_sust_b_3d_v2i32_zero, llvm_i32_type, llvm_i32_type);
    create_case(PixelStorage::INT4, llvm::Intrinsic::nvvm_sust_b_3d_v4i32_zero, llvm_i32_type, llvm_i32_type);
    create_case(PixelStorage::HALF1, llvm::Intrinsic::nvvm_sust_b_3d_i16_zero, llvm_f16_type, llvm_i16_type);
    create_case(PixelStorage::HALF2, llvm::Intrinsic::nvvm_sust_b_3d_v2i16_zero, llvm_f16_type, llvm_i16_type);
    create_case(PixelStorage::HALF4, llvm::Intrinsic::nvvm_sust_b_3d_v4i16_zero, llvm_f16_type, llvm_i16_type);
    create_case(PixelStorage::FLOAT1, llvm::Intrinsic::nvvm_sust_b_3d_i32_zero, llvm_f32_type, llvm_i32_type);
    create_case(PixelStorage::FLOAT2, llvm::Intrinsic::nvvm_sust_b_3d_v2i32_zero, llvm_f32_type, llvm_i32_type);
    create_case(PixelStorage::FLOAT4, llvm::Intrinsic::nvvm_sust_b_3d_v4i32_zero, llvm_f32_type, llvm_i32_type);
    // default block is unreachable
    b.SetInsertPoint(llvm_default_block);
    b.CreateUnreachable();
    return llvm_func;
}

llvm::InlineAsm *CUDACodegenLLVMImpl::_get_inline_asm(std::string_view asm_string, std::string_view constraints, bool has_side_effects) noexcept {
    auto map_type = [this](char type) noexcept -> llvm::Type * {
        // "h" = .u16 reg
        // "r" = .u32 reg
        // "l" = .u64 reg
        // "q" = .u128 reg
        // "f" = .f32 reg
        // "d" = .f64 reg
        switch (type) {
            case 'h': return llvm::Type::getInt16Ty(_llvm_context);
            case 'r': return llvm::Type::getInt32Ty(_llvm_context);
            case 'l': return llvm::Type::getInt64Ty(_llvm_context);
            case 'q': return llvm::Type::getInt128Ty(_llvm_context);
            case 'f': return llvm::Type::getFloatTy(_llvm_context);
            case 'd': return llvm::Type::getDoubleTy(_llvm_context);
            default: LUISA_ERROR_WITH_LOCATION("Unsupported inline asm type constraint '{}'.", type);
        }
    };
    llvm::SmallVector<llvm::Type *, 4> param_types;
    llvm::SmallVector<llvm::Type *, 4> return_types;
    auto next_is_output = false;
    for (auto c : constraints) {
        if (c == '=') {
            next_is_output = true;
        } else if (c == ',') {
            next_is_output = false;
        } else {
            auto type = map_type(c);
            if (next_is_output) {
                return_types.emplace_back(type);
            } else {
                param_types.emplace_back(type);
            }
        }
    }
    auto return_type = return_types.empty()     ? llvm::Type::getVoidTy(_llvm_context) :
                       return_types.size() == 1 ? return_types.front() :
                                                  llvm::StructType::get(_llvm_context, return_types);
    auto func_type = llvm::FunctionType::get(return_type, param_types, false);
    return llvm::InlineAsm::get(func_type, asm_string, constraints, has_side_effects);
}

llvm::Value *CUDACodegenLLVMImpl::_translate_call_inst(IB &b, FunctionContext &func_ctx, const xir::CallInst *inst) noexcept {
    auto llvm_callee = _get_or_declare_llvm_function(inst->callee());
    llvm::SmallVector<llvm::Value *> llvm_args;
    llvm_args.reserve(inst->argument_count() + 2u);
    for (auto i = 0u; i < inst->argument_count(); i++) {
        auto llvm_arg = _get_llvm_value(b, func_ctx, inst->argument(i));
        // cast pointer arguments to address space 0 if not already
        if (auto llvm_arg_type = llvm_arg->getType();
            llvm_arg_type->isPointerTy() && llvm_arg_type->getPointerAddressSpace() != 0) {
            llvm_arg = b.CreateAddrSpaceCast(llvm_arg, b.getPtrTy());
        }
        llvm_args.emplace_back(llvm_arg);
    }
    // append dispatch_size and kernel_id arguments
    llvm_args.emplace_back(_read_dispatch_size(b, func_ctx));
    llvm_args.emplace_back(_read_kernel_id(b, func_ctx));
    // create call instruction
    auto call_inst = b.CreateCall(llvm_callee, llvm_args, inst->name().value_or(""));
    call_inst->setCallingConv(llvm_callee->getCallingConv());
    return inst->type() == nullptr ? nullptr : call_inst;
}

void CUDACodegenLLVMImpl::_translate_outline_inst(IB &b, FunctionContext &func_ctx, const xir::OutlineInst *inst) noexcept {
    LUISA_ERROR_WITH_LOCATION("Outline instruction should have been lowered.");
}

}// namespace luisa::compute::cuda
