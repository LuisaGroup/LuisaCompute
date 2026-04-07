//
// Created by mike on 3/18/26.
//

#include <luisa/xir/passes/dom_tree.h>
#include <luisa/ast/function.h>
#include <luisa/runtime/rhi/pixel.h>

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

llvm::Function *HIPCodegenLLVMImpl::_get_or_declare_llvm_function(const xir::Function *func) noexcept {
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

llvm::Function *HIPCodegenLLVMImpl::_declare_llvm_kernel_function(const xir::KernelFunction *func) noexcept {
    auto arg_struct_info = _get_kernel_argument_struct(func);
    auto [llvm_func_type, llvm_func_name] = [&]() noexcept -> std::pair<llvm::FunctionType *, llvm::StringRef> {
        auto llvm_void_type = llvm::Type::getVoidTy(_llvm_context);
        return std::make_pair(llvm::FunctionType::get(llvm_void_type, {arg_struct_info->llvm_type}, false), "kernel_main");
    }();
    auto llvm_kernel = llvm::Function::Create(llvm_func_type, llvm::Function::ExternalLinkage, llvm_func_name, _llvm_module.get());
    llvm_kernel->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
    llvm_kernel->addFnAttr("amdgpu-unsafe-fp-atomics", "true");

    // Set occupancy hints: tell the AMDGPU backend the exact workgroup size
    // so it can optimize register allocation and occupancy accordingly.
    auto block_size = _config.block_size[0] * _config.block_size[1] * _config.block_size[2];
    auto block_size_str = std::to_string(block_size);
    llvm_kernel->addFnAttr("amdgpu-flat-work-group-size", block_size_str + "," + block_size_str);

    return llvm_kernel;
}

llvm::Function *HIPCodegenLLVMImpl::_declare_llvm_callable_function(const xir::CallableFunction *func) noexcept {
    llvm::SmallVector<llvm::Type *> llvm_arg_types;
    for (auto arg : func->arguments()) {
        if (arg->is_reference()) {
            llvm_arg_types.emplace_back(llvm::PointerType::get(_llvm_context, 0));
        } else {
            llvm_arg_types.emplace_back(_get_llvm_type(arg->type())->reg_type);
        }
    }
    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
    auto llvm_i32x3_type = llvm::FixedVectorType::get(llvm_i32_type, 3);
    llvm_arg_types.emplace_back(llvm_i32x3_type);
    llvm_arg_types.emplace_back(llvm_i32_type);
    if (_rt_analysis.uses_ray_tracing) {
        llvm_arg_types.emplace_back(llvm_i32_type);
        llvm_arg_types.emplace_back(llvm_i32_type);
        llvm_arg_types.emplace_back(llvm::PointerType::get(_llvm_context, 0));
    }
    auto llvm_ret_type = func->type() == nullptr ? llvm::Type::getVoidTy(_llvm_context) :
                                                   _get_llvm_type(func->type())->reg_type;
    auto llvm_func_type = llvm::FunctionType::get(llvm_ret_type, llvm_arg_types, false);
    auto llvm_func = llvm::Function::Create(llvm_func_type, llvm::Function::PrivateLinkage, 0,
                                            func->name().value_or("callable"), _llvm_module.get());
    llvm_func->addFnAttr("amdgpu-unsafe-fp-atomics", "true");
    return llvm_func;
}

llvm::Function *HIPCodegenLLVMImpl::_declare_llvm_external_function(const xir::ExternalFunction *func) noexcept {
    LUISA_NOT_IMPLEMENTED("External function declaration not implemented.");
}

llvm::Function *HIPCodegenLLVMImpl::_translate_function(const xir::FunctionDefinition *func) noexcept {
    switch (func->derived_function_tag()) {
        case xir::DerivedFunctionTag::KERNEL: return _translate_kernel_function(static_cast<const xir::KernelFunction *>(func));
        case xir::DerivedFunctionTag::CALLABLE: return _translate_callable_function(static_cast<const xir::CallableFunction *>(func));
        case xir::DerivedFunctionTag::EXTERNAL: LUISA_ERROR_WITH_LOCATION("Cannot translate external function.");
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported function type.");
}

llvm::Function *HIPCodegenLLVMImpl::_translate_kernel_function(const xir::KernelFunction *func) noexcept {
    auto arg_struct_info = _get_kernel_argument_struct(func);
    auto llvm_kernel = _get_or_declare_llvm_function(func);
    LUISA_DEBUG_ASSERT(llvm_kernel->isDeclaration(), "Kernel function already defined.");
    FunctionContext func_ctx{llvm_kernel};
    IB b{func_ctx.llvm_entry_block};
    auto llvm_arg_struct = llvm_kernel->getArg(0);
    auto arg_index = 0u;
    for (auto arg : func->arguments()) {
        auto member_index = arg_struct_info->argument_indices[arg_index];
        auto llvm_member_mem = b.CreateExtractValue(llvm_arg_struct, member_index, arg->name().value_or(""));
        auto llvm_member_reg = arg->is_value() ? _convert_llvm_mem_value_to_reg(b, llvm_member_mem, arg->type()) : llvm_member_mem;
        func_ctx.local_values.try_emplace(arg, llvm_member_reg);
        arg_index++;
    }
    auto llvm_dispatch_size_and_kernel_id = b.CreateExtractValue(llvm_arg_struct, arg_struct_info->dispatch_size_and_kernel_id_index);
    auto llvm_dispatch_size_x = b.CreateExtractValue(llvm_dispatch_size_and_kernel_id, 0);
    b.CreateAssumption(b.CreateICmpUGT(llvm_dispatch_size_x, b.getInt32(0)));
    auto llvm_dispatch_size_y = b.CreateExtractValue(llvm_dispatch_size_and_kernel_id, 1);
    b.CreateAssumption(b.CreateICmpUGT(llvm_dispatch_size_y, b.getInt32(0)));
    auto llvm_dispatch_size_z = b.CreateExtractValue(llvm_dispatch_size_and_kernel_id, 2);
    b.CreateAssumption(b.CreateICmpUGT(llvm_dispatch_size_z, b.getInt32(0)));
    func_ctx.llvm_dispatch_size = _create_llvm_vector(b, {llvm_dispatch_size_x, llvm_dispatch_size_y, llvm_dispatch_size_z});
    func_ctx.llvm_dispatch_size->setName("sreg.dispatch.size");
    func_ctx.llvm_kernel_id = b.CreateExtractValue(llvm_dispatch_size_and_kernel_id, 3, "sreg.kernel.id");
    if (arg_struct_info->has_rt_global_stack_buffer) {
        auto idx = arg_struct_info->rt_global_stack_buffer_index;
        func_ctx.llvm_rt_stack_size = b.CreateExtractValue(llvm_arg_struct, idx, "rt.stack.size");
        func_ctx.llvm_rt_stack_count = b.CreateExtractValue(llvm_arg_struct, idx + 1, "rt.stack.count");
        func_ctx.llvm_rt_stack_data = b.CreateExtractValue(llvm_arg_struct, idx + 2, "rt.stack.data");
    } else if (_rt_analysis.uses_ray_tracing) {
        func_ctx.llvm_rt_stack_size = b.getInt32(0);
        func_ctx.llvm_rt_stack_count = b.getInt32(0);
        func_ctx.llvm_rt_stack_data = llvm::ConstantPointerNull::get(b.getPtrTy(0));
    }
    auto llvm_body = _translate_function_definition(func_ctx, func);
    auto llvm_dispatch_id = _read_dispatch_id(b, func_ctx);
    auto llvm_dispatch_id_in_bounds = b.CreateICmpULT(llvm_dispatch_id, func_ctx.llvm_dispatch_size, "dispatch.id.in.bounds");
    for (int i = 0; i < 3; i++) {
        if (_config.block_size[i] == 1) {
            llvm_dispatch_id_in_bounds = b.CreateInsertElement(llvm_dispatch_id_in_bounds, b.getInt1(true), i);
        }
    }
    auto llvm_dispatch_id_in_bounds_all = b.CreateAndReduce(llvm_dispatch_id_in_bounds);
    auto llvm_exit_block = llvm::BasicBlock::Create(_llvm_context, "exit.early", llvm_kernel);
    b.CreateCondBr(llvm_dispatch_id_in_bounds_all, llvm_body, llvm_exit_block);
    b.SetInsertPoint(llvm_exit_block);
    b.CreateRetVoid();
    return llvm_kernel;
}

llvm::Function *HIPCodegenLLVMImpl::_translate_callable_function(const xir::CallableFunction *func) noexcept {
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
    if (_rt_analysis.uses_ray_tracing) {
        func_ctx.llvm_rt_stack_size = llvm_arg_iter++;
        func_ctx.llvm_rt_stack_size->setName("rt.stack.size");
        func_ctx.llvm_rt_stack_count = llvm_arg_iter++;
        func_ctx.llvm_rt_stack_count->setName("rt.stack.count");
        func_ctx.llvm_rt_stack_data = llvm_arg_iter++;
        func_ctx.llvm_rt_stack_data->setName("rt.stack.data");
    }
    auto body = _translate_function_definition(func_ctx, func);
    IB b{func_ctx.llvm_entry_block};
    b.CreateBr(body);
    return llvm_func;
}

namespace {

template<typename F>
void luisa_compute_hip_codegen_llvm_traverse_dom_tree_impl(luisa::unordered_set<const xir::DomTreeNode *> &visited,
                                                           const xir::DomTreeNode *node, const F &f) noexcept {
    if (visited.emplace(node).second) [[likely]] {
        f(node->block());
        for (auto child : node->children()) {
            luisa_compute_hip_codegen_llvm_traverse_dom_tree_impl(visited, child, f);
        }
    }
}

template<typename F>
void luisa_compute_hip_codegen_llvm_traverse_dom_tree(const xir::DomTree &tree, const F &f) noexcept {
    luisa::unordered_set<const xir::DomTreeNode *> visited;
    luisa_compute_hip_codegen_llvm_traverse_dom_tree_impl(visited, tree.root(), f);
}

}// namespace

llvm::BasicBlock *HIPCodegenLLVMImpl::_translate_function_definition(FunctionContext &func_ctx, const xir::FunctionDefinition *f) noexcept {
    for (auto bb : f->basic_blocks()) {
        auto llvm_bb = llvm::BasicBlock::Create(_llvm_context, bb->name().value_or(""), func_ctx.llvm_func);
        func_ctx.local_values.try_emplace(bb, llvm_bb);
    }
    auto dom_tree = xir::compute_dom_tree(const_cast<xir::FunctionDefinition *>(f));
    LUISA_ASSERT(dom_tree.root()->block() == f->body_block());
    luisa_compute_hip_codegen_llvm_traverse_dom_tree(dom_tree, [this, &func_ctx](const xir::BasicBlock *bb) noexcept {
        auto llvm_bb = func_ctx.get_local_value<llvm::BasicBlock>(bb);
        IB b{llvm_bb};
        for (auto inst : bb->instructions()) {
            _translate_instruction(b, func_ctx, inst);
        }
    });
    _finalize_pending_phi_nodes(func_ctx);
    return func_ctx.get_local_value<llvm::BasicBlock>(f->body_block());
}

void HIPCodegenLLVMImpl::_mark_llvm_function_as_pure(llvm::Function *func) noexcept {
    func->addFnAttr(llvm::Attribute::NoCallback);
    func->setMustProgress();
    func->setDoesNotFreeMemory();
    func->setNoSync();
    func->setDoesNotThrow();
    func->setSpeculatable();
    func->setWillReturn();
    func->setDoesNotAccessMemory();
}

llvm::Function *HIPCodegenLLVMImpl::_get_assert_function() noexcept {
    if (auto llvm_f = _llvm_module->getFunction("luisa.assert")) {
        return llvm_f;
    }
    auto llvm_i1_type = llvm::Type::getInt1Ty(_llvm_context);
    auto llvm_const_ptr_type = llvm::PointerType::get(_llvm_context, amdgpu_address_space_constant);
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
    b.SetInsertPoint(llvm_then_bb);
    b.CreateRetVoid();
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

llvm::Function *HIPCodegenLLVMImpl::_get_vprintf_function() noexcept {
    if (auto llvm_f = _llvm_module->getFunction("vprintf")) { return llvm_f; }
    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
    auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, 0);
    auto llvm_func_type = llvm::FunctionType::get(llvm_i32_type, {llvm_ptr_type, llvm_ptr_type}, false);
    return llvm::Function::Create(llvm_func_type, llvm::Function::ExternalLinkage, "vprintf", *_llvm_module);
}

llvm::Function *HIPCodegenLLVMImpl::_get_texture2d_read_function(llvm::VectorType *llvm_value_type) noexcept {
    auto name = fmt::format("luisa.hip.texture.2d.read.{}", _to_string(llvm_value_type->getElementType()));
    if (auto llvm_func = _llvm_module->getFunction(name)) { return llvm_func; }

    auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
    auto llvm_i16_type = llvm::Type::getInt16Ty(_llvm_context);
    auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
    auto llvm_f32_type = llvm::Type::getFloatTy(_llvm_context);
    auto llvm_f16_type = llvm::Type::getHalfTy(_llvm_context);
    auto llvm_coord_type = llvm::FixedVectorType::get(llvm_i32_type, 2);
    auto llvm_v4f32_type = llvm::FixedVectorType::get(llvm_f32_type, 4);
    auto llvm_v8i32_type = llvm::FixedVectorType::get(llvm_i32_type, 8);

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

    auto create_case = [&](PixelStorage storage, llvm::Type *llvm_channel_type) noexcept {
        auto llvm_case_block = llvm::BasicBlock::Create(_llvm_context, fmt::format("switch.case.{}", luisa::to_string(storage)), llvm_func);
        llvm_switch->addCase(b.getInt64(luisa::to_underlying(storage)), llvm_case_block);
        b.SetInsertPoint(llvm_case_block);

        // Image descriptors must be loaded from constant address space (4) on AMDGPU
        auto llvm_const_ptr = b.CreateIntToPtr(llvm_handle, llvm::PointerType::get(_llvm_context, amdgpu_address_space_constant), "rsrc.ptr");
        auto llvm_rsrc = b.CreateLoad(llvm_v8i32_type, llvm_const_ptr, "rsrc");

        auto llvm_raw = b.CreateIntrinsic(
            llvm::Intrinsic::amdgcn_image_load_2d,
            {llvm_v4f32_type, llvm_i32_type, llvm_v8i32_type},
            {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_rsrc, b.getInt32(0), b.getInt32(0)});

        auto channel_count = pixel_storage_channel_count(storage);
        LUISA_DEBUG_ASSERT(channel_count == 1 || channel_count == 2 || channel_count == 4);

        llvm::Value *llvm_src;
        if (llvm_channel_type->isIntegerTy()) {
            auto llvm_v4i32_type = llvm::FixedVectorType::get(llvm_i32_type, 4);
            auto llvm_raw_i32 = b.CreateBitCast(llvm_raw, llvm_v4i32_type, "raw.i32");
            auto llvm_src_type = llvm::FixedVectorType::get(llvm_channel_type, 4);
            llvm_src = static_cast<llvm::Value *>(llvm::Constant::getNullValue(llvm_src_type));
            for (auto i = 0u; i < channel_count; i++) {
                auto llvm_channel = b.CreateExtractElement(llvm_raw_i32, static_cast<uint64_t>(i));
                llvm_channel = b.CreateTrunc(llvm_channel, llvm_channel_type);
                llvm_src = b.CreateInsertElement(llvm_src, llvm_channel, static_cast<uint64_t>(i));
            }
        } else if (llvm_channel_type->isHalfTy()) {
            auto llvm_src_type = llvm::FixedVectorType::get(llvm_f16_type, 4);
            llvm_src = static_cast<llvm::Value *>(llvm::Constant::getNullValue(llvm_src_type));
            for (auto i = 0u; i < channel_count; i++) {
                auto llvm_channel = b.CreateExtractElement(llvm_raw, static_cast<uint64_t>(i));
                llvm_channel = b.CreateFPTrunc(llvm_channel, llvm_f16_type);
                llvm_src = b.CreateInsertElement(llvm_src, llvm_channel, static_cast<uint64_t>(i));
            }
        } else {
            auto llvm_src_type = llvm::FixedVectorType::get(llvm_f32_type, 4);
            llvm_src = static_cast<llvm::Value *>(llvm::Constant::getNullValue(llvm_src_type));
            for (auto i = 0u; i < channel_count; i++) {
                auto llvm_channel = b.CreateExtractElement(llvm_raw, static_cast<uint64_t>(i));
                llvm_src = b.CreateInsertElement(llvm_src, llvm_channel, static_cast<uint64_t>(i));
            }
        }

        auto llvm_dst = _texel_cast(b, llvm_src, llvm_value_type);
        b.CreateRet(llvm_dst);
    };

    create_case(PixelStorage::BYTE1, llvm_i8_type);
    create_case(PixelStorage::BYTE2, llvm_i8_type);
    create_case(PixelStorage::BYTE4, llvm_i8_type);
    create_case(PixelStorage::SHORT1, llvm_i16_type);
    create_case(PixelStorage::SHORT2, llvm_i16_type);
    create_case(PixelStorage::SHORT4, llvm_i16_type);
    create_case(PixelStorage::INT1, llvm_i32_type);
    create_case(PixelStorage::INT2, llvm_i32_type);
    create_case(PixelStorage::INT4, llvm_i32_type);
    create_case(PixelStorage::HALF1, llvm_f16_type);
    create_case(PixelStorage::HALF2, llvm_f16_type);
    create_case(PixelStorage::HALF4, llvm_f16_type);
    create_case(PixelStorage::FLOAT1, llvm_f32_type);
    create_case(PixelStorage::FLOAT2, llvm_f32_type);
    create_case(PixelStorage::FLOAT4, llvm_f32_type);

    b.SetInsertPoint(llvm_default_block);
    b.CreateUnreachable();

    return llvm_func;
}

llvm::Function *HIPCodegenLLVMImpl::_get_texture2d_write_function(llvm::VectorType *llvm_value_type) noexcept {
    auto element_type = _to_string(llvm_value_type->getElementType());
    auto name = fmt::format("luisa.hip.texture.2d.write.{}", element_type);
    if (auto llvm_func = _llvm_module->getFunction(name)) { return llvm_func; }

    auto llvm_void_type = llvm::Type::getVoidTy(_llvm_context);
    auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
    auto llvm_i16_type = llvm::Type::getInt16Ty(_llvm_context);
    auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
    auto llvm_f32_type = llvm::Type::getFloatTy(_llvm_context);
    auto llvm_f16_type = llvm::Type::getHalfTy(_llvm_context);
    auto llvm_coord_type = llvm::VectorType::get(llvm_i32_type, 2, false);
    auto llvm_v4f32_type = llvm::FixedVectorType::get(llvm_f32_type, 4);
    auto llvm_v8i32_type = llvm::FixedVectorType::get(llvm_i32_type, 8);

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
    auto llvm_value = llvm_func->getArg(3);
    llvm_value->setName("value");

    auto llvm_coord_x = b.CreateExtractElement(llvm_coord, b.getInt64(0), "coord.x");
    auto llvm_coord_y = b.CreateExtractElement(llvm_coord, b.getInt64(1), "coord.y");

    auto llvm_default_block = llvm::BasicBlock::Create(_llvm_context, "switch.default", llvm_func);
    auto llvm_switch = b.CreateSwitch(llvm_storage, llvm_default_block, 16);

    auto create_case = [&](PixelStorage storage, llvm::Type *llvm_channel_type) noexcept {
        auto llvm_case_block = llvm::BasicBlock::Create(_llvm_context, fmt::format("switch.case.{}", luisa::to_string(storage)), llvm_func);
        llvm_switch->addCase(b.getInt64(luisa::to_underlying(storage)), llvm_case_block);
        b.SetInsertPoint(llvm_case_block);

        auto channel_count = pixel_storage_channel_count(storage);
        LUISA_DEBUG_ASSERT(channel_count == 1 || channel_count == 2 || channel_count == 4);

        auto llvm_dst_type = llvm::FixedVectorType::get(llvm_channel_type, 4);
        auto llvm_dst = _texel_cast(b, llvm_value, llvm_dst_type);

        llvm::Value *llvm_data;
        if (llvm_channel_type->isIntegerTy()) {
            auto llvm_v4i32_type = llvm::FixedVectorType::get(llvm_i32_type, 4);
            auto llvm_ext = b.CreateZExt(llvm_dst, llvm_v4i32_type, "ext.i32");
            llvm_data = b.CreateBitCast(llvm_ext, llvm_v4f32_type, "data.f32");
        } else if (llvm_channel_type->isHalfTy()) {
            llvm_data = static_cast<llvm::Value *>(llvm::Constant::getNullValue(llvm_v4f32_type));
            for (auto i = 0u; i < 4u; i++) {
                auto llvm_channel = b.CreateExtractElement(llvm_dst, static_cast<uint64_t>(i));
                llvm_channel = b.CreateFPExt(llvm_channel, llvm_f32_type);
                llvm_data = b.CreateInsertElement(llvm_data, llvm_channel, static_cast<uint64_t>(i));
            }
        } else {
            llvm_data = llvm_dst;
        }

        auto llvm_const_ptr = b.CreateIntToPtr(llvm_handle, llvm::PointerType::get(_llvm_context, amdgpu_address_space_constant), "rsrc.ptr");
        auto llvm_rsrc = b.CreateLoad(llvm_v8i32_type, llvm_const_ptr, "rsrc");

        b.CreateIntrinsic(
            llvm::Intrinsic::amdgcn_image_store_2d,
            {llvm_v4f32_type, llvm_i32_type, llvm_v8i32_type},
            {llvm_data, b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_rsrc, b.getInt32(0), b.getInt32(0)});

        b.CreateRetVoid();
    };

    create_case(PixelStorage::BYTE1, llvm_i8_type);
    create_case(PixelStorage::BYTE2, llvm_i8_type);
    create_case(PixelStorage::BYTE4, llvm_i8_type);
    create_case(PixelStorage::SHORT1, llvm_i16_type);
    create_case(PixelStorage::SHORT2, llvm_i16_type);
    create_case(PixelStorage::SHORT4, llvm_i16_type);
    create_case(PixelStorage::INT1, llvm_i32_type);
    create_case(PixelStorage::INT2, llvm_i32_type);
    create_case(PixelStorage::INT4, llvm_i32_type);
    create_case(PixelStorage::HALF1, llvm_f16_type);
    create_case(PixelStorage::HALF2, llvm_f16_type);
    create_case(PixelStorage::HALF4, llvm_f16_type);
    create_case(PixelStorage::FLOAT1, llvm_f32_type);
    create_case(PixelStorage::FLOAT2, llvm_f32_type);
    create_case(PixelStorage::FLOAT4, llvm_f32_type);

    b.SetInsertPoint(llvm_default_block);
    b.CreateUnreachable();

    return llvm_func;
}

llvm::Function *HIPCodegenLLVMImpl::_get_texture3d_read_function(llvm::VectorType *llvm_value_type) noexcept {
    auto name = fmt::format("luisa.hip.texture.3d.read.{}", _to_string(llvm_value_type->getElementType()));
    if (auto llvm_func = _llvm_module->getFunction(name)) { return llvm_func; }

    auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
    auto llvm_i16_type = llvm::Type::getInt16Ty(_llvm_context);
    auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
    auto llvm_f32_type = llvm::Type::getFloatTy(_llvm_context);
    auto llvm_f16_type = llvm::Type::getHalfTy(_llvm_context);
    auto llvm_coord_type = llvm::FixedVectorType::get(llvm_i32_type, 3);
    auto llvm_v4f32_type = llvm::FixedVectorType::get(llvm_f32_type, 4);
    auto llvm_v8i32_type = llvm::FixedVectorType::get(llvm_i32_type, 8);

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

    auto create_case = [&](PixelStorage storage, llvm::Type *llvm_channel_type) noexcept {
        auto llvm_case_block = llvm::BasicBlock::Create(_llvm_context, fmt::format("switch.case.{}", luisa::to_string(storage)), llvm_func);
        llvm_switch->addCase(b.getInt64(luisa::to_underlying(storage)), llvm_case_block);
        b.SetInsertPoint(llvm_case_block);

        // Image descriptors must be loaded from constant address space (4) on AMDGPU
        auto llvm_const_ptr = b.CreateIntToPtr(llvm_handle, llvm::PointerType::get(_llvm_context, amdgpu_address_space_constant), "rsrc.ptr");
        auto llvm_rsrc = b.CreateLoad(llvm_v8i32_type, llvm_const_ptr, "rsrc");

        auto llvm_raw = b.CreateIntrinsic(
            llvm::Intrinsic::amdgcn_image_load_3d,
            {llvm_v4f32_type, llvm_i32_type, llvm_v8i32_type},
            {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_coord_z, llvm_rsrc, b.getInt32(0), b.getInt32(0)});

        auto channel_count = pixel_storage_channel_count(storage);
        LUISA_DEBUG_ASSERT(channel_count == 1 || channel_count == 2 || channel_count == 4);

        llvm::Value *llvm_src;
        if (llvm_channel_type->isIntegerTy()) {
            auto llvm_v4i32_type = llvm::FixedVectorType::get(llvm_i32_type, 4);
            auto llvm_raw_i32 = b.CreateBitCast(llvm_raw, llvm_v4i32_type, "raw.i32");
            auto llvm_src_type = llvm::FixedVectorType::get(llvm_channel_type, 4);
            llvm_src = static_cast<llvm::Value *>(llvm::Constant::getNullValue(llvm_src_type));
            for (auto i = 0u; i < channel_count; i++) {
                auto llvm_channel = b.CreateExtractElement(llvm_raw_i32, static_cast<uint64_t>(i));
                llvm_channel = b.CreateTrunc(llvm_channel, llvm_channel_type);
                llvm_src = b.CreateInsertElement(llvm_src, llvm_channel, static_cast<uint64_t>(i));
            }
        } else if (llvm_channel_type->isHalfTy()) {
            auto llvm_src_type = llvm::FixedVectorType::get(llvm_f16_type, 4);
            llvm_src = static_cast<llvm::Value *>(llvm::Constant::getNullValue(llvm_src_type));
            for (auto i = 0u; i < channel_count; i++) {
                auto llvm_channel = b.CreateExtractElement(llvm_raw, static_cast<uint64_t>(i));
                llvm_channel = b.CreateFPTrunc(llvm_channel, llvm_f16_type);
                llvm_src = b.CreateInsertElement(llvm_src, llvm_channel, static_cast<uint64_t>(i));
            }
        } else {
            auto llvm_src_type = llvm::FixedVectorType::get(llvm_f32_type, 4);
            llvm_src = static_cast<llvm::Value *>(llvm::Constant::getNullValue(llvm_src_type));
            for (auto i = 0u; i < channel_count; i++) {
                auto llvm_channel = b.CreateExtractElement(llvm_raw, static_cast<uint64_t>(i));
                llvm_src = b.CreateInsertElement(llvm_src, llvm_channel, static_cast<uint64_t>(i));
            }
        }

        auto llvm_dst = _texel_cast(b, llvm_src, llvm_value_type);
        b.CreateRet(llvm_dst);
    };

    create_case(PixelStorage::BYTE1, llvm_i8_type);
    create_case(PixelStorage::BYTE2, llvm_i8_type);
    create_case(PixelStorage::BYTE4, llvm_i8_type);
    create_case(PixelStorage::SHORT1, llvm_i16_type);
    create_case(PixelStorage::SHORT2, llvm_i16_type);
    create_case(PixelStorage::SHORT4, llvm_i16_type);
    create_case(PixelStorage::INT1, llvm_i32_type);
    create_case(PixelStorage::INT2, llvm_i32_type);
    create_case(PixelStorage::INT4, llvm_i32_type);
    create_case(PixelStorage::HALF1, llvm_f16_type);
    create_case(PixelStorage::HALF2, llvm_f16_type);
    create_case(PixelStorage::HALF4, llvm_f16_type);
    create_case(PixelStorage::FLOAT1, llvm_f32_type);
    create_case(PixelStorage::FLOAT2, llvm_f32_type);
    create_case(PixelStorage::FLOAT4, llvm_f32_type);

    b.SetInsertPoint(llvm_default_block);
    b.CreateUnreachable();

    return llvm_func;
}

llvm::Function *HIPCodegenLLVMImpl::_get_texture3d_write_function(llvm::VectorType *llvm_value_type) noexcept {
    auto element_type = _to_string(llvm_value_type->getElementType());
    auto name = fmt::format("luisa.hip.texture.3d.write.{}", element_type);
    if (auto llvm_func = _llvm_module->getFunction(name)) { return llvm_func; }

    auto llvm_void_type = llvm::Type::getVoidTy(_llvm_context);
    auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
    auto llvm_i16_type = llvm::Type::getInt16Ty(_llvm_context);
    auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
    auto llvm_f32_type = llvm::Type::getFloatTy(_llvm_context);
    auto llvm_f16_type = llvm::Type::getHalfTy(_llvm_context);
    auto llvm_coord_type = llvm::FixedVectorType::get(llvm_i32_type, 3);
    auto llvm_v4f32_type = llvm::FixedVectorType::get(llvm_f32_type, 4);
    auto llvm_v8i32_type = llvm::FixedVectorType::get(llvm_i32_type, 8);

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
    auto llvm_value = llvm_func->getArg(3);
    llvm_value->setName("value");

    auto llvm_coord_x = b.CreateExtractElement(llvm_coord, b.getInt64(0), "coord.x");
    auto llvm_coord_y = b.CreateExtractElement(llvm_coord, b.getInt64(1), "coord.y");
    auto llvm_coord_z = b.CreateExtractElement(llvm_coord, b.getInt64(2), "coord.z");

    auto llvm_default_block = llvm::BasicBlock::Create(_llvm_context, "switch.default", llvm_func);
    auto llvm_switch = b.CreateSwitch(llvm_storage, llvm_default_block, 16);

    auto create_case = [&](PixelStorage storage, llvm::Type *llvm_channel_type) noexcept {
        auto llvm_case_block = llvm::BasicBlock::Create(_llvm_context, fmt::format("switch.case.{}", luisa::to_string(storage)), llvm_func);
        llvm_switch->addCase(b.getInt64(luisa::to_underlying(storage)), llvm_case_block);
        b.SetInsertPoint(llvm_case_block);

        auto channel_count = pixel_storage_channel_count(storage);
        LUISA_DEBUG_ASSERT(channel_count == 1 || channel_count == 2 || channel_count == 4);

        auto llvm_dst_type = llvm::FixedVectorType::get(llvm_channel_type, 4);
        auto llvm_dst = _texel_cast(b, llvm_value, llvm_dst_type);

        llvm::Value *llvm_data;
        if (llvm_channel_type->isIntegerTy()) {
            auto llvm_v4i32_type = llvm::FixedVectorType::get(llvm_i32_type, 4);
            auto llvm_ext = b.CreateZExt(llvm_dst, llvm_v4i32_type, "ext.i32");
            llvm_data = b.CreateBitCast(llvm_ext, llvm_v4f32_type, "data.f32");
        } else if (llvm_channel_type->isHalfTy()) {
            llvm_data = static_cast<llvm::Value *>(llvm::Constant::getNullValue(llvm_v4f32_type));
            for (auto i = 0u; i < 4u; i++) {
                auto llvm_channel = b.CreateExtractElement(llvm_dst, static_cast<uint64_t>(i));
                llvm_channel = b.CreateFPExt(llvm_channel, llvm_f32_type);
                llvm_data = b.CreateInsertElement(llvm_data, llvm_channel, static_cast<uint64_t>(i));
            }
        } else {
            llvm_data = llvm_dst;
        }

        auto llvm_const_ptr = b.CreateIntToPtr(llvm_handle, llvm::PointerType::get(_llvm_context, amdgpu_address_space_constant), "rsrc.ptr");
        auto llvm_rsrc = b.CreateLoad(llvm_v8i32_type, llvm_const_ptr, "rsrc");

        b.CreateIntrinsic(
            llvm::Intrinsic::amdgcn_image_store_3d,
            {llvm_v4f32_type, llvm_i32_type, llvm_v8i32_type},
            {llvm_data, b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_coord_z, llvm_rsrc, b.getInt32(0), b.getInt32(0)});

        b.CreateRetVoid();
    };

    create_case(PixelStorage::BYTE1, llvm_i8_type);
    create_case(PixelStorage::BYTE2, llvm_i8_type);
    create_case(PixelStorage::BYTE4, llvm_i8_type);
    create_case(PixelStorage::SHORT1, llvm_i16_type);
    create_case(PixelStorage::SHORT2, llvm_i16_type);
    create_case(PixelStorage::SHORT4, llvm_i16_type);
    create_case(PixelStorage::INT1, llvm_i32_type);
    create_case(PixelStorage::INT2, llvm_i32_type);
    create_case(PixelStorage::INT4, llvm_i32_type);
    create_case(PixelStorage::HALF1, llvm_f16_type);
    create_case(PixelStorage::HALF2, llvm_f16_type);
    create_case(PixelStorage::HALF4, llvm_f16_type);
    create_case(PixelStorage::FLOAT1, llvm_f32_type);
    create_case(PixelStorage::FLOAT2, llvm_f32_type);
    create_case(PixelStorage::FLOAT4, llvm_f32_type);

    b.SetInsertPoint(llvm_default_block);
    b.CreateUnreachable();

    return llvm_func;
}

llvm::InlineAsm *HIPCodegenLLVMImpl::_get_inline_asm(std::string_view asm_string, std::string_view constraints, bool has_side_effects) noexcept {
    auto map_type = [this](char type) noexcept -> llvm::Type * {
        switch (type) {
            case 'h': return llvm::Type::getInt16Ty(_llvm_context);
            case 'r': return llvm::Type::getInt32Ty(_llvm_context);
            case 'l': return llvm::Type::getInt64Ty(_llvm_context);
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

llvm::Value *HIPCodegenLLVMImpl::_translate_call_inst(IB &b, FunctionContext &func_ctx, const xir::CallInst *inst) noexcept {
    auto llvm_callee = _get_or_declare_llvm_function(inst->callee());
    llvm::SmallVector<llvm::Value *> llvm_args;
    llvm_args.reserve(inst->argument_count() + 5u);
    for (auto i = 0u; i < inst->argument_count(); i++) {
        auto llvm_arg = _get_llvm_value(b, func_ctx, inst->argument(i));
        if (auto llvm_arg_type = llvm_arg->getType();
            llvm_arg_type->isPointerTy() && llvm_arg_type->getPointerAddressSpace() != 0) {
            llvm_arg = b.CreateAddrSpaceCast(llvm_arg, b.getPtrTy());
        }
        llvm_args.emplace_back(llvm_arg);
    }
    llvm_args.emplace_back(_read_dispatch_size(b, func_ctx));
    llvm_args.emplace_back(_read_kernel_id(b, func_ctx));
    if (_rt_analysis.uses_ray_tracing) {
        llvm_args.emplace_back(func_ctx.llvm_rt_stack_size);
        llvm_args.emplace_back(func_ctx.llvm_rt_stack_count);
        llvm_args.emplace_back(func_ctx.llvm_rt_stack_data);
    }
    auto call_inst = b.CreateCall(llvm_callee, llvm_args, inst->name().value_or(""));
    call_inst->setCallingConv(llvm_callee->getCallingConv());
    return inst->type() == nullptr ? nullptr : call_inst;
}

void HIPCodegenLLVMImpl::_translate_outline_inst(IB &b, FunctionContext &func_ctx, const xir::OutlineInst *inst) noexcept {
    LUISA_ERROR_WITH_LOCATION("Outline instruction should have been lowered.");
}

}// namespace luisa::compute::hip
