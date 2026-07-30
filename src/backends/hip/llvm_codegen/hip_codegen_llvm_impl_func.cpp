//
// Created by mike on 3/18/26.
//

#include <luisa/xir/passes/dom_tree.h>
#include <luisa/ast/function.h>
#include <luisa/runtime/rhi/pixel.h>

#include "hip_codegen_llvm_impl.h"

#include <algorithm>

namespace luisa::compute::hip {

namespace {

constexpr auto hip_hardware_ray_query_state_size = 448u;
constexpr auto hip_software_ray_query_state_size = 576u;

}// namespace

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
    if (_config.max_register_count != 0u) {
        auto max_vgpr_count = std::min(_config.max_register_count, 256u);
        llvm_kernel->addFnAttr("amdgpu-num-vgpr", std::to_string(max_vgpr_count));
    }

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
    if (_config.requires_printing) {
        llvm_arg_types.emplace_back(_get_llvm_print_buffer_type());
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
    auto name = func->name();
    LUISA_ASSERT(name.has_value() && !name->empty(),
                 "HIP external functions must have a non-empty symbol name.");
    // ExternalCallable is a module ABI boundary: value arguments use Luisa's
    // LLVM register types, mutable references use generic pointers, and no
    // internal dispatch/printing/ray-tracing context parameters are appended.
    llvm::SmallVector<llvm::Type *> llvm_arg_types;
    llvm_arg_types.reserve(func->arguments().count_size());
    for (auto arg : func->arguments()) {
        llvm_arg_types.emplace_back(
            arg->is_reference() ? llvm::PointerType::get(_llvm_context, 0) :
                                  _get_llvm_type(arg->type())->reg_type);
    }
    auto llvm_ret_type = func->type() == nullptr ?
                             llvm::Type::getVoidTy(_llvm_context) :
                             _get_llvm_type(func->type())->reg_type;
    auto llvm_func_type = llvm::FunctionType::get(
        llvm_ret_type, llvm_arg_types, false);
    if (auto existing = _llvm_module->getFunction(*name)) {
        LUISA_ASSERT(existing->getFunctionType() == llvm_func_type,
                     "HIP external function '{}' has an incompatible LLVM ABI.",
                     *name);
        return existing;
    }
    return llvm::Function::Create(
        llvm_func_type, llvm::Function::ExternalLinkage,
        llvm::StringRef{name->data(), name->size()}, _llvm_module.get());
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
    if (arg_struct_info->has_print_buffer) {
        auto llvm_print_buffer = b.CreateExtractValue(
            llvm_arg_struct, arg_struct_info->print_buffer_index,
            "print.buffer");
        func_ctx.llvm_print_buffer_capacity = b.CreateExtractValue(
            llvm_print_buffer, 0u, "print.buffer.capacity");
        func_ctx.llvm_print_buffer_content = b.CreateExtractValue(
            llvm_print_buffer, 1u, "print.buffer.content");
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
    if (_rt_analysis.uses_ray_query) {
        // gfx12 uses the compact flat-traversal state guarded by a matching
        // exact-size static_assert in hiprt_device_wrapper.hip. Keep this
        // allocation exact: it is address-taken across the out-of-line
        // traversal calls, so every unused byte becomes per-thread scratch.
        // Generic HIPRT traversal with a private instance stack uses the
        // exact 576-byte state checked in hiprt_device_wrapper.hip.
        auto llvm_rq_state_size = _uses_hardware_rt_stack ?
                                      hip_hardware_ray_query_state_size :
                                      hip_software_ray_query_state_size;
        auto llvm_rq_state_type = llvm::ArrayType::get(b.getInt8Ty(), llvm_rq_state_size);
        IB alloca_b{func_ctx.llvm_alloca_block->getTerminator()};
        auto alloca_inst = alloca_b.CreateAlloca(llvm_rq_state_type, nullptr, "rq.state");
        alloca_inst->setAlignment(llvm::Align(16));
        func_ctx.llvm_rq_state = alloca_inst;
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
    if (_config.requires_printing) {
        auto llvm_print_buffer = llvm_arg_iter++;
        llvm_print_buffer->setName("print.buffer");
        IB b{func_ctx.llvm_entry_block};
        func_ctx.llvm_print_buffer_capacity = b.CreateExtractValue(
            llvm_print_buffer, 0u, "print.buffer.capacity");
        func_ctx.llvm_print_buffer_content = b.CreateExtractValue(
            llvm_print_buffer, 1u, "print.buffer.content");
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
    if (_rt_analysis.uses_ray_query) {
        auto llvm_rq_state_size = _uses_hardware_rt_stack ?
                                      hip_hardware_ray_query_state_size :
                                      hip_software_ray_query_state_size;
        auto llvm_rq_state_type = llvm::ArrayType::get(
            llvm::Type::getInt8Ty(_llvm_context), llvm_rq_state_size);
        IB alloca_b{func_ctx.llvm_alloca_block->getTerminator()};
        auto alloca_inst = alloca_b.CreateAlloca(llvm_rq_state_type, nullptr, "rq.state");
        alloca_inst->setAlignment(llvm::Align(16));
        func_ctx.llvm_rq_state = alloca_inst;
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
    luisa::unordered_set<const xir::BasicBlock *> translated_blocks;
    luisa_compute_hip_codegen_llvm_traverse_dom_tree(dom_tree, [this, &func_ctx, &translated_blocks](const xir::BasicBlock *bb) noexcept {
        translated_blocks.emplace(bb);
        auto llvm_bb = func_ctx.get_local_value<llvm::BasicBlock>(bb);
        IB b{llvm_bb};
        for (auto inst : bb->instructions()) {
            _translate_instruction(b, func_ctx, inst);
        }
    });
    _finalize_pending_phi_nodes(func_ctx, translated_blocks);
    // Dominator traversal skips unreachable structured merge blocks, but LLVM still requires terminators.
    for (auto bb : f->basic_blocks()) {
        auto llvm_bb = func_ctx.get_local_value<llvm::BasicBlock>(bb);
        if (!translated_blocks.contains(bb)) {
            IB b{llvm_bb};
            b.CreateUnreachable();
        }
        LUISA_ASSERT(llvm_bb->getTerminator() != nullptr,
                     "LLVM basic block has no terminator after translation.");
    }
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
    if (auto llvm_f = _llvm_module->getFunction("luisa.vprintf")) { return llvm_f; }
    auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
    auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
    auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, 0);
    auto llvm_func_type = llvm::FunctionType::get(llvm_i32_type, {llvm_ptr_type, llvm_ptr_type}, false);
    auto llvm_f = llvm::Function::Create(
        llvm_func_type, llvm::Function::PrivateLinkage,
        "luisa.vprintf", *_llvm_module);
    llvm_f->addFnAttr(llvm::Attribute::NoUnwind);

    auto llvm_printf_begin_type = llvm::FunctionType::get(llvm_i64_type, {llvm_i64_type}, false);
    auto llvm_printf_begin = _llvm_module->getOrInsertFunction(
        "__ockl_printf_begin", llvm_printf_begin_type);
    auto llvm_printf_append_string_type = llvm::FunctionType::get(
        llvm_i64_type, {llvm_i64_type, llvm_ptr_type, llvm_i64_type, llvm_i32_type}, false);
    auto llvm_printf_append_string = _llvm_module->getOrInsertFunction(
        "__ockl_printf_append_string_n", llvm_printf_append_string_type);
    auto llvm_printf_append_args_type = llvm::FunctionType::get(
        llvm_i64_type,
        {llvm_i64_type, llvm_i32_type,
         llvm_i64_type, llvm_i64_type, llvm_i64_type, llvm_i64_type,
         llvm_i64_type, llvm_i64_type, llvm_i64_type, llvm_i32_type},
        false);
    auto llvm_printf_append_args = _llvm_module->getOrInsertFunction(
        "__ockl_printf_append_args", llvm_printf_append_args_type);

    auto llvm_entry = llvm::BasicBlock::Create(_llvm_context, "entry", llvm_f);
    auto llvm_strlen_loop = llvm::BasicBlock::Create(_llvm_context, "strlen.loop", llvm_f);
    auto llvm_print = llvm::BasicBlock::Create(_llvm_context, "print", llvm_f);
    auto llvm_args_loop = llvm::BasicBlock::Create(_llvm_context, "args.loop", llvm_f);
    auto llvm_arg_scalar = llvm::BasicBlock::Create(_llvm_context, "arg.scalar", llvm_f);
    auto llvm_arg_string_strlen = llvm::BasicBlock::Create(_llvm_context, "arg.string.strlen", llvm_f);
    auto llvm_arg_string = llvm::BasicBlock::Create(_llvm_context, "arg.string", llvm_f);
    auto llvm_args_continue = llvm::BasicBlock::Create(_llvm_context, "args.continue", llvm_f);
    auto llvm_exit = llvm::BasicBlock::Create(_llvm_context, "exit", llvm_f);

    auto llvm_format = llvm_f->getArg(0);
    auto llvm_argument_pack = llvm_f->getArg(1);
    llvm_format->setName("format");
    llvm_argument_pack->setName("argument.pack");

    IB b{llvm_entry};
    b.CreateBr(llvm_strlen_loop);

    // OCKL's append_string_n expects the terminating null byte to be included.
    b.SetInsertPoint(llvm_strlen_loop);
    auto llvm_string_index = b.CreatePHI(llvm_i64_type, 2u, "strlen.index");
    llvm_string_index->addIncoming(b.getInt64(0u), llvm_entry);
    auto llvm_character_pointer = b.CreateInBoundsGEP(
        llvm_i8_type, llvm_format, llvm_string_index);
    auto llvm_character = b.CreateLoad(llvm_i8_type, llvm_character_pointer);
    auto llvm_next_string_index = b.CreateAdd(llvm_string_index, b.getInt64(1u));
    llvm_string_index->addIncoming(llvm_next_string_index, llvm_strlen_loop);
    auto llvm_string_done = b.CreateICmpEQ(llvm_character, b.getInt8(0u));
    b.CreateCondBr(llvm_string_done, llvm_print, llvm_strlen_loop);

    b.SetInsertPoint(llvm_print);
    auto llvm_printf_handle = b.CreateCall(llvm_printf_begin, {b.getInt64(0u)});
    auto llvm_has_arguments = b.CreateICmpNE(
        llvm_argument_pack, llvm::ConstantPointerNull::get(llvm_ptr_type));
    auto llvm_string_is_last = b.CreateZExt(b.CreateNot(llvm_has_arguments), llvm_i32_type);
    auto llvm_string_handle = b.CreateCall(
        llvm_printf_append_string,
        {llvm_printf_handle, llvm_format, llvm_next_string_index, llvm_string_is_last});
    b.CreateCondBr(llvm_has_arguments, llvm_args_loop, llvm_exit);

    // Argument packs contain one i64 count followed by tagged {kind, payload}
    // pairs. Sending one argument per append call avoids ABI-dependent
    // aggregate layout while still supporting arbitrarily large XIR values.
    b.SetInsertPoint(llvm_args_loop);
    auto llvm_argument_index = b.CreatePHI(llvm_i64_type, 2u, "argument.index");
    llvm_argument_index->addIncoming(b.getInt64(0u), llvm_print);
    auto llvm_previous_handle = b.CreatePHI(llvm_i64_type, 2u, "printf.handle");
    llvm_previous_handle->addIncoming(llvm_string_handle, llvm_print);
    auto llvm_argument_count = b.CreateLoad(llvm_i64_type, llvm_argument_pack, "argument.count");
    auto llvm_kind_index = b.CreateAdd(
        b.CreateMul(llvm_argument_index, b.getInt64(2u)),
        b.getInt64(1u));
    auto llvm_payload_index = b.CreateAdd(llvm_kind_index, b.getInt64(1u));
    auto llvm_kind_pointer = b.CreateInBoundsGEP(
        llvm_i64_type, llvm_argument_pack, llvm_kind_index);
    auto llvm_argument_pointer = b.CreateInBoundsGEP(
        llvm_i64_type, llvm_argument_pack, llvm_payload_index);
    auto llvm_argument_kind = b.CreateLoad(llvm_i64_type, llvm_kind_pointer);
    auto llvm_argument = b.CreateLoad(llvm_i64_type, llvm_argument_pointer);
    auto llvm_next_argument_index = b.CreateAdd(llvm_argument_index, b.getInt64(1u));
    auto llvm_arguments_done = b.CreateICmpEQ(llvm_next_argument_index, llvm_argument_count);
    auto llvm_argument_is_last = b.CreateZExt(llvm_arguments_done, llvm_i32_type);
    auto llvm_argument_is_string = b.CreateICmpNE(llvm_argument_kind, b.getInt64(0u));
    b.CreateCondBr(llvm_argument_is_string, llvm_arg_string_strlen, llvm_arg_scalar);

    b.SetInsertPoint(llvm_arg_scalar);
    llvm::SmallVector<llvm::Value *, 10> llvm_append_arguments{
        llvm_previous_handle, b.getInt32(1u), llvm_argument,
        b.getInt64(0u), b.getInt64(0u), b.getInt64(0u),
        b.getInt64(0u), b.getInt64(0u), b.getInt64(0u),
        llvm_argument_is_last};
    auto llvm_scalar_handle = b.CreateCall(llvm_printf_append_args, llvm_append_arguments);
    b.CreateBr(llvm_args_continue);

    // OCKL copies %s payloads through append_string_n. Passing the pointer as a
    // scalar would leave the host trying to dereference a device address.
    b.SetInsertPoint(llvm_arg_string_strlen);
    auto llvm_argument_string_index = b.CreatePHI(llvm_i64_type, 2u, "argument.string.index");
    llvm_argument_string_index->addIncoming(b.getInt64(0u), llvm_args_loop);
    auto llvm_argument_string = b.CreateIntToPtr(llvm_argument, llvm_ptr_type);
    auto llvm_argument_character_pointer = b.CreateInBoundsGEP(
        llvm_i8_type, llvm_argument_string, llvm_argument_string_index);
    auto llvm_argument_character = b.CreateLoad(llvm_i8_type, llvm_argument_character_pointer);
    auto llvm_next_argument_string_index = b.CreateAdd(
        llvm_argument_string_index, b.getInt64(1u));
    llvm_argument_string_index->addIncoming(
        llvm_next_argument_string_index, llvm_arg_string_strlen);
    auto llvm_argument_string_done = b.CreateICmpEQ(
        llvm_argument_character, b.getInt8(0u));
    b.CreateCondBr(
        llvm_argument_string_done, llvm_arg_string,
        llvm_arg_string_strlen);

    b.SetInsertPoint(llvm_arg_string);
    auto llvm_argument_string_handle = b.CreateCall(
        llvm_printf_append_string,
        {llvm_previous_handle, llvm_argument_string,
         llvm_next_argument_string_index, llvm_argument_is_last});
    b.CreateBr(llvm_args_continue);

    b.SetInsertPoint(llvm_args_continue);
    auto llvm_next_handle = b.CreatePHI(llvm_i64_type, 2u, "printf.next.handle");
    llvm_next_handle->addIncoming(llvm_scalar_handle, llvm_arg_scalar);
    llvm_next_handle->addIncoming(llvm_argument_string_handle, llvm_arg_string);
    llvm_argument_index->addIncoming(llvm_next_argument_index, llvm_args_continue);
    llvm_previous_handle->addIncoming(llvm_next_handle, llvm_args_continue);
    b.CreateCondBr(llvm_arguments_done, llvm_exit, llvm_args_loop);

    b.SetInsertPoint(llvm_exit);
    auto llvm_final_handle = b.CreatePHI(llvm_i64_type, 2u, "printf.result");
    llvm_final_handle->addIncoming(llvm_string_handle, llvm_print);
    llvm_final_handle->addIncoming(llvm_next_handle, llvm_args_continue);
    b.CreateRet(b.CreateTrunc(llvm_final_handle, llvm_i32_type));
    return llvm_f;
}

llvm::Value *HIPCodegenLLVMImpl::_unpack_r10g10b10a2(
    IB &b, llvm::Value *packed, llvm::VectorType *dst_type) noexcept {
    LUISA_DEBUG_ASSERT(packed->getType()->isIntegerTy(32u));
    LUISA_DEBUG_ASSERT(dst_type->getElementCount().getKnownMinValue() == 4u);
    constexpr uint32_t shifts[]{0u, 10u, 20u, 30u};
    constexpr uint32_t masks[]{0x3ffu, 0x3ffu, 0x3ffu, 0x3u};
    constexpr float scales[]{1.0f / 1023.0f, 1.0f / 1023.0f,
                             1.0f / 1023.0f, 1.0f / 3.0f};
    auto dst_element_type = dst_type->getElementType();
    auto result = static_cast<llvm::Value *>(
        llvm::Constant::getNullValue(dst_type));
    for (auto i = 0u; i < 4u; i++) {
        auto bits = shifts[i] == 0u ? packed :
                                      b.CreateLShr(packed, b.getInt32(shifts[i]));
        bits = b.CreateAnd(bits, b.getInt32(masks[i]));
        llvm::Value *channel;
        if (dst_element_type->isFloatingPointTy()) {
            channel = b.CreateFMul(
                b.CreateUIToFP(bits, b.getFloatTy()),
                llvm::ConstantFP::get(b.getFloatTy(), scales[i]));
            channel = _safe_fp_cast(b, channel, dst_element_type);
        } else {
            LUISA_DEBUG_ASSERT(dst_element_type->isIntegerTy());
            channel = b.CreateZExtOrTrunc(bits, dst_element_type);
        }
        result = b.CreateInsertElement(result, channel, b.getInt64(i));
    }
    return result;
}

llvm::Value *HIPCodegenLLVMImpl::_pack_r10g10b10a2(
    IB &b, llvm::Value *value) noexcept {
    auto value_type = llvm::cast<llvm::VectorType>(value->getType());
    LUISA_DEBUG_ASSERT(value_type->getElementCount().getKnownMinValue() == 4u);
    constexpr uint32_t shifts[]{0u, 10u, 20u, 30u};
    constexpr uint32_t masks[]{0x3ffu, 0x3ffu, 0x3ffu, 0x3u};
    constexpr float scales[]{1023.0f, 1023.0f, 1023.0f, 3.0f};
    auto element_type = value_type->getElementType();
    auto packed = static_cast<llvm::Value *>(b.getInt32(0u));
    for (auto i = 0u; i < 4u; i++) {
        auto channel = b.CreateExtractElement(value, b.getInt64(i));
        llvm::Value *bits;
        if (element_type->isFloatingPointTy()) {
            channel = _safe_fp_cast(b, channel, b.getFloatTy());
            channel = b.CreateMinNum(
                b.CreateMaxNum(
                    channel,
                    llvm::ConstantFP::get(b.getFloatTy(), 0.0)),
                llvm::ConstantFP::get(b.getFloatTy(), 1.0));
            channel = b.CreateFMul(
                channel,
                llvm::ConstantFP::get(b.getFloatTy(), scales[i]));
            channel = b.CreateUnaryIntrinsic(llvm::Intrinsic::round, channel);
            bits = b.CreateFPToUI(channel, b.getInt32Ty());
        } else {
            LUISA_DEBUG_ASSERT(element_type->isIntegerTy());
            bits = b.CreateZExtOrTrunc(channel, b.getInt32Ty());
        }
        bits = b.CreateAnd(bits, b.getInt32(masks[i]));
        if (shifts[i] != 0u) {
            bits = b.CreateShl(bits, b.getInt32(shifts[i]));
        }
        packed = b.CreateOr(packed, bits);
    }
    return packed;
}

llvm::Function *HIPCodegenLLVMImpl::_get_texture2d_read_function(llvm::VectorType *llvm_value_type) noexcept {
    // LLVM integer types erase signedness. Keep it as an explicit runtime
    // argument so one cached helper can serve both Image<int> and Image<uint>.
    auto name = fmt::format("luisa.hip.texture.2d.read.{}", _to_string(llvm_value_type->getElementType()));
    if (auto llvm_func = _llvm_module->getFunction(name)) { return llvm_func; }

    auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
    auto llvm_i1_type = llvm::Type::getInt1Ty(_llvm_context);
    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
    auto llvm_i16_type = llvm::Type::getInt16Ty(_llvm_context);
    auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
    auto llvm_f32_type = llvm::Type::getFloatTy(_llvm_context);
    auto llvm_f16_type = llvm::Type::getHalfTy(_llvm_context);
    auto llvm_coord_type = llvm::FixedVectorType::get(llvm_i32_type, 2);
    auto llvm_v4f32_type = llvm::FixedVectorType::get(llvm_f32_type, 4);
    auto llvm_v8i32_type = llvm::FixedVectorType::get(llvm_i32_type, 8);

    auto llvm_func_type = llvm::FunctionType::get(
        llvm_value_type,
        {llvm_i64_type, llvm_i64_type, llvm_i1_type, llvm_coord_type}, false);
    auto llvm_func = llvm::Function::Create(llvm_func_type, llvm::Function::PrivateLinkage, name, *_llvm_module);
    llvm_func->addFnAttr(llvm::Attribute::AlwaysInline);

    auto llvm_entry = llvm::BasicBlock::Create(_llvm_context, "entry", llvm_func);
    IB b{llvm_entry};

    auto llvm_handle = llvm_func->getArg(0);
    llvm_handle->setName("surface.handle");
    auto llvm_storage = llvm_func->getArg(1);
    llvm_storage->setName("surface.storage");
    auto llvm_is_signed = llvm_func->getArg(2);
    llvm_is_signed->setName("surface.is.signed.integer");
    auto llvm_coord = llvm_func->getArg(3);
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
        if (llvm_channel_type->isIntegerTy() &&
            llvm_value_type->getElementType()->isIntegerTy()) {
            auto llvm_signed_dst = b.CreateIntCast(
                llvm_src, llvm_value_type, true,
                "texel.cast.signed.int.to.int");
            llvm_dst = b.CreateSelect(
                llvm_is_signed, llvm_signed_dst, llvm_dst,
                "texel.cast.int");
        }
        b.CreateRet(llvm_dst);
    };

    auto create_packed_case = [&]() noexcept {
        auto llvm_case_block = llvm::BasicBlock::Create(
            _llvm_context, "switch.case.r10g10b10a2", llvm_func);
        llvm_switch->addCase(
            b.getInt64(to_underlying(PixelStorage::R10G10B10A2)),
            llvm_case_block);
        b.SetInsertPoint(llvm_case_block);
        auto llvm_const_ptr = b.CreateIntToPtr(
            llvm_handle,
            llvm::PointerType::get(
                _llvm_context, amdgpu_address_space_constant),
            "rsrc.ptr");
        auto llvm_rsrc = b.CreateLoad(
            llvm_v8i32_type, llvm_const_ptr, "rsrc");
        auto llvm_raw = b.CreateIntrinsic(
            llvm::Intrinsic::amdgcn_image_load_2d,
            {llvm_v4f32_type, llvm_i32_type, llvm_v8i32_type},
            {b.getInt32(15), llvm_coord_x, llvm_coord_y,
             llvm_rsrc, b.getInt32(0), b.getInt32(0)});
        auto llvm_raw_i32 = b.CreateBitCast(
            llvm_raw, llvm::FixedVectorType::get(llvm_i32_type, 4u));
        auto llvm_packed = b.CreateExtractElement(
            llvm_raw_i32, b.getInt64(0u));
        b.CreateRet(_unpack_r10g10b10a2(
            b, llvm_packed, llvm_value_type));
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
    create_packed_case();

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
    auto llvm_i1_type = llvm::Type::getInt1Ty(_llvm_context);
    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
    auto llvm_i16_type = llvm::Type::getInt16Ty(_llvm_context);
    auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
    auto llvm_f32_type = llvm::Type::getFloatTy(_llvm_context);
    auto llvm_f16_type = llvm::Type::getHalfTy(_llvm_context);
    auto llvm_coord_type = llvm::VectorType::get(llvm_i32_type, 2, false);
    auto llvm_v4f32_type = llvm::FixedVectorType::get(llvm_f32_type, 4);
    auto llvm_v8i32_type = llvm::FixedVectorType::get(llvm_i32_type, 8);

    auto llvm_func_type = llvm::FunctionType::get(
        llvm_void_type,
        {llvm_i64_type, llvm_i64_type, llvm_i1_type,
         llvm_coord_type, llvm_value_type},
        false);
    auto llvm_func = llvm::Function::Create(llvm_func_type, llvm::Function::PrivateLinkage, name, *_llvm_module);
    llvm_func->addFnAttr(llvm::Attribute::AlwaysInline);

    auto llvm_entry = llvm::BasicBlock::Create(_llvm_context, "entry", llvm_func);
    IB b{llvm_entry};

    auto llvm_handle = llvm_func->getArg(0);
    llvm_handle->setName("surface.handle");
    auto llvm_storage = llvm_func->getArg(1);
    llvm_storage->setName("surface.storage");
    auto llvm_is_signed = llvm_func->getArg(2);
    llvm_is_signed->setName("surface.is.signed.integer");
    auto llvm_coord = llvm_func->getArg(3);
    llvm_coord->setName("coord");
    auto llvm_value = llvm_func->getArg(4);
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
            auto llvm_unsigned_ext = b.CreateIntCast(
                llvm_dst, llvm_v4i32_type, false, "ext.unsigned.i32");
            auto llvm_signed_ext = b.CreateIntCast(
                llvm_dst, llvm_v4i32_type, true, "ext.signed.i32");
            auto llvm_ext = b.CreateSelect(
                llvm_is_signed, llvm_signed_ext, llvm_unsigned_ext,
                "ext.i32");
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

    auto create_packed_case = [&]() noexcept {
        auto llvm_case_block = llvm::BasicBlock::Create(
            _llvm_context, "switch.case.r10g10b10a2", llvm_func);
        llvm_switch->addCase(
            b.getInt64(to_underlying(PixelStorage::R10G10B10A2)),
            llvm_case_block);
        b.SetInsertPoint(llvm_case_block);
        auto llvm_packed = _pack_r10g10b10a2(b, llvm_value);
        auto llvm_data_i32 = static_cast<llvm::Value *>(
            llvm::Constant::getNullValue(
                llvm::FixedVectorType::get(llvm_i32_type, 4u)));
        llvm_data_i32 = b.CreateInsertElement(
            llvm_data_i32, llvm_packed, b.getInt64(0u));
        auto llvm_data = b.CreateBitCast(llvm_data_i32, llvm_v4f32_type);
        auto llvm_const_ptr = b.CreateIntToPtr(
            llvm_handle,
            llvm::PointerType::get(
                _llvm_context, amdgpu_address_space_constant),
            "rsrc.ptr");
        auto llvm_rsrc = b.CreateLoad(
            llvm_v8i32_type, llvm_const_ptr, "rsrc");
        b.CreateIntrinsic(
            llvm::Intrinsic::amdgcn_image_store_2d,
            {llvm_v4f32_type, llvm_i32_type, llvm_v8i32_type},
            {llvm_data, b.getInt32(15), llvm_coord_x, llvm_coord_y,
             llvm_rsrc, b.getInt32(0), b.getInt32(0)});
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
    create_packed_case();

    b.SetInsertPoint(llvm_default_block);
    b.CreateUnreachable();

    return llvm_func;
}

llvm::Function *HIPCodegenLLVMImpl::_get_texture3d_read_function(llvm::VectorType *llvm_value_type) noexcept {
    auto name = fmt::format("luisa.hip.texture.3d.read.{}", _to_string(llvm_value_type->getElementType()));
    if (auto llvm_func = _llvm_module->getFunction(name)) { return llvm_func; }

    auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
    auto llvm_i1_type = llvm::Type::getInt1Ty(_llvm_context);
    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
    auto llvm_i16_type = llvm::Type::getInt16Ty(_llvm_context);
    auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
    auto llvm_f32_type = llvm::Type::getFloatTy(_llvm_context);
    auto llvm_f16_type = llvm::Type::getHalfTy(_llvm_context);
    auto llvm_coord_type = llvm::FixedVectorType::get(llvm_i32_type, 3);
    auto llvm_v4f32_type = llvm::FixedVectorType::get(llvm_f32_type, 4);
    auto llvm_v8i32_type = llvm::FixedVectorType::get(llvm_i32_type, 8);

    auto llvm_func_type = llvm::FunctionType::get(
        llvm_value_type,
        {llvm_i64_type, llvm_i64_type, llvm_i1_type, llvm_coord_type}, false);
    auto llvm_func = llvm::Function::Create(llvm_func_type, llvm::Function::PrivateLinkage, name, *_llvm_module);
    llvm_func->addFnAttr(llvm::Attribute::AlwaysInline);

    auto llvm_entry = llvm::BasicBlock::Create(_llvm_context, "entry", llvm_func);
    IB b{llvm_entry};

    auto llvm_handle = llvm_func->getArg(0);
    llvm_handle->setName("surface.handle");
    auto llvm_storage = llvm_func->getArg(1);
    llvm_storage->setName("surface.storage");
    auto llvm_is_signed = llvm_func->getArg(2);
    llvm_is_signed->setName("surface.is.signed.integer");
    auto llvm_coord = llvm_func->getArg(3);
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
        if (llvm_channel_type->isIntegerTy() &&
            llvm_value_type->getElementType()->isIntegerTy()) {
            auto llvm_signed_dst = b.CreateIntCast(
                llvm_src, llvm_value_type, true,
                "texel.cast.signed.int.to.int");
            llvm_dst = b.CreateSelect(
                llvm_is_signed, llvm_signed_dst, llvm_dst,
                "texel.cast.int");
        }
        b.CreateRet(llvm_dst);
    };

    auto create_packed_case = [&]() noexcept {
        auto llvm_case_block = llvm::BasicBlock::Create(
            _llvm_context, "switch.case.r10g10b10a2", llvm_func);
        llvm_switch->addCase(
            b.getInt64(to_underlying(PixelStorage::R10G10B10A2)),
            llvm_case_block);
        b.SetInsertPoint(llvm_case_block);
        auto llvm_const_ptr = b.CreateIntToPtr(
            llvm_handle,
            llvm::PointerType::get(
                _llvm_context, amdgpu_address_space_constant),
            "rsrc.ptr");
        auto llvm_rsrc = b.CreateLoad(
            llvm_v8i32_type, llvm_const_ptr, "rsrc");
        auto llvm_raw = b.CreateIntrinsic(
            llvm::Intrinsic::amdgcn_image_load_3d,
            {llvm_v4f32_type, llvm_i32_type, llvm_v8i32_type},
            {b.getInt32(15), llvm_coord_x, llvm_coord_y, llvm_coord_z,
             llvm_rsrc, b.getInt32(0), b.getInt32(0)});
        auto llvm_raw_i32 = b.CreateBitCast(
            llvm_raw, llvm::FixedVectorType::get(llvm_i32_type, 4u));
        auto llvm_packed = b.CreateExtractElement(
            llvm_raw_i32, b.getInt64(0u));
        b.CreateRet(_unpack_r10g10b10a2(
            b, llvm_packed, llvm_value_type));
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
    create_packed_case();

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
    auto llvm_i1_type = llvm::Type::getInt1Ty(_llvm_context);
    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
    auto llvm_i16_type = llvm::Type::getInt16Ty(_llvm_context);
    auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
    auto llvm_f32_type = llvm::Type::getFloatTy(_llvm_context);
    auto llvm_f16_type = llvm::Type::getHalfTy(_llvm_context);
    auto llvm_coord_type = llvm::FixedVectorType::get(llvm_i32_type, 3);
    auto llvm_v4f32_type = llvm::FixedVectorType::get(llvm_f32_type, 4);
    auto llvm_v8i32_type = llvm::FixedVectorType::get(llvm_i32_type, 8);

    auto llvm_func_type = llvm::FunctionType::get(
        llvm_void_type,
        {llvm_i64_type, llvm_i64_type, llvm_i1_type,
         llvm_coord_type, llvm_value_type},
        false);
    auto llvm_func = llvm::Function::Create(llvm_func_type, llvm::Function::PrivateLinkage, name, *_llvm_module);
    llvm_func->addFnAttr(llvm::Attribute::AlwaysInline);

    auto llvm_entry = llvm::BasicBlock::Create(_llvm_context, "entry", llvm_func);
    IB b{llvm_entry};

    auto llvm_handle = llvm_func->getArg(0);
    llvm_handle->setName("surface.handle");
    auto llvm_storage = llvm_func->getArg(1);
    llvm_storage->setName("surface.storage");
    auto llvm_is_signed = llvm_func->getArg(2);
    llvm_is_signed->setName("surface.is.signed.integer");
    auto llvm_coord = llvm_func->getArg(3);
    llvm_coord->setName("coord");
    auto llvm_value = llvm_func->getArg(4);
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
            auto llvm_unsigned_ext = b.CreateIntCast(
                llvm_dst, llvm_v4i32_type, false, "ext.unsigned.i32");
            auto llvm_signed_ext = b.CreateIntCast(
                llvm_dst, llvm_v4i32_type, true, "ext.signed.i32");
            auto llvm_ext = b.CreateSelect(
                llvm_is_signed, llvm_signed_ext, llvm_unsigned_ext,
                "ext.i32");
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

    auto create_packed_case = [&]() noexcept {
        auto llvm_case_block = llvm::BasicBlock::Create(
            _llvm_context, "switch.case.r10g10b10a2", llvm_func);
        llvm_switch->addCase(
            b.getInt64(to_underlying(PixelStorage::R10G10B10A2)),
            llvm_case_block);
        b.SetInsertPoint(llvm_case_block);
        auto llvm_packed = _pack_r10g10b10a2(b, llvm_value);
        auto llvm_data_i32 = static_cast<llvm::Value *>(
            llvm::Constant::getNullValue(
                llvm::FixedVectorType::get(llvm_i32_type, 4u)));
        llvm_data_i32 = b.CreateInsertElement(
            llvm_data_i32, llvm_packed, b.getInt64(0u));
        auto llvm_data = b.CreateBitCast(llvm_data_i32, llvm_v4f32_type);
        auto llvm_const_ptr = b.CreateIntToPtr(
            llvm_handle,
            llvm::PointerType::get(
                _llvm_context, amdgpu_address_space_constant),
            "rsrc.ptr");
        auto llvm_rsrc = b.CreateLoad(
            llvm_v8i32_type, llvm_const_ptr, "rsrc");
        b.CreateIntrinsic(
            llvm::Intrinsic::amdgcn_image_store_3d,
            {llvm_v4f32_type, llvm_i32_type, llvm_v8i32_type},
            {llvm_data, b.getInt32(15), llvm_coord_x, llvm_coord_y,
             llvm_coord_z, llvm_rsrc, b.getInt32(0), b.getInt32(0)});
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
    create_packed_case();

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
    auto is_external = inst->callee()->isa<xir::ExternalFunction>();
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
    if (!is_external && _config.requires_printing) {
        auto llvm_print_buffer = static_cast<llvm::Value *>(
            llvm::PoisonValue::get(_get_llvm_print_buffer_type()));
        llvm_print_buffer = b.CreateInsertValue(
            llvm_print_buffer, func_ctx.llvm_print_buffer_capacity, 0u);
        llvm_print_buffer = b.CreateInsertValue(
            llvm_print_buffer, func_ctx.llvm_print_buffer_content, 1u);
        llvm_args.emplace_back(llvm_print_buffer);
    }
    if (!is_external) {
        llvm_args.emplace_back(_read_dispatch_size(b, func_ctx));
        llvm_args.emplace_back(_read_kernel_id(b, func_ctx));
        if (_rt_analysis.uses_ray_tracing) {
            llvm_args.emplace_back(func_ctx.llvm_rt_stack_size);
            llvm_args.emplace_back(func_ctx.llvm_rt_stack_count);
            llvm_args.emplace_back(func_ctx.llvm_rt_stack_data);
        }
    }
    auto call_inst = b.CreateCall(llvm_callee, llvm_args, inst->name().value_or(""));
    call_inst->setCallingConv(llvm_callee->getCallingConv());
    return inst->type() == nullptr ? nullptr : call_inst;
}

void HIPCodegenLLVMImpl::_translate_outline_inst(IB &b, FunctionContext &func_ctx, const xir::OutlineInst *inst) noexcept {
    LUISA_ERROR_WITH_LOCATION("Outline instruction should have been lowered.");
}

}// namespace luisa::compute::hip
