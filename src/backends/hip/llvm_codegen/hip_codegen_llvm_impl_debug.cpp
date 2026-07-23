//
// Created by mike on 3/18/26.
//

#include <limits>

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

void HIPCodegenLLVMImpl::_translate_print_inst(
    IB &b, FunctionContext &func_ctx,
    const xir::PrintInst *inst) noexcept {
    auto info_iter = _print_info.find(inst);
    LUISA_ASSERT(info_iter != _print_info.end(),
                 "Missing HIP print metadata for XIR PrintInst.");
    LUISA_ASSERT(func_ctx.llvm_print_buffer_capacity != nullptr &&
                     func_ctx.llvm_print_buffer_content != nullptr,
                 "HIP PrintInst has no print-buffer context.");

    auto [argument_pack_type, format_index] = info_iter->second;
    LUISA_ASSERT(argument_pack_type->size() <=
                     std::numeric_limits<uint32_t>::max(),
                 "HIP print record is too large ({} bytes).",
                 argument_pack_type->size());
    auto llvm_argument_pack_info = _get_llvm_type(argument_pack_type);
    auto llvm_argument_pack = static_cast<llvm::Value *>(
        llvm::Constant::getNullValue(llvm_argument_pack_info->mem_type));
    llvm_argument_pack = b.CreateInsertValue(
        llvm_argument_pack,
        b.getInt32(static_cast<uint32_t>(argument_pack_type->size())),
        llvm_argument_pack_info->member_indices[0u]);
    llvm_argument_pack = b.CreateInsertValue(
        llvm_argument_pack, b.getInt32(format_index),
        llvm_argument_pack_info->member_indices[1u]);

    auto operand_uses = inst->operand_uses();
    LUISA_ASSERT(argument_pack_type->members().size() ==
                     operand_uses.size() + 2u,
                 "HIP print argument-pack metadata mismatch.");
    for (auto i = 0u; i < operand_uses.size(); i++) {
        auto operand = operand_uses[i]->value();
        LUISA_ASSERT(operand != nullptr, "HIP print operand is null.");
        auto llvm_operand = _get_llvm_value(b, func_ctx, operand);
        auto llvm_mem_operand = _convert_llvm_reg_value_to_mem(
            b, llvm_operand, operand->type());
        llvm_argument_pack = b.CreateInsertValue(
            llvm_argument_pack, llvm_mem_operand,
            llvm_argument_pack_info->member_indices[i + 2u]);
    }

    auto llvm_temp = _create_temp_in_alloca_block(
        func_ctx, llvm_argument_pack_info->mem_type,
        argument_pack_type->alignment());
    b.CreateAlignedStore(
        llvm_argument_pack, llvm_temp,
        llvm::Align{argument_pack_type->alignment()});
    auto llvm_generic_ptr_type = llvm::PointerType::get(_llvm_context, 0u);
    auto llvm_temp_generic = llvm_temp;
    if (llvm_temp->getType()->getPointerAddressSpace() != 0u) {
        llvm_temp_generic = b.CreateAddrSpaceCast(
            llvm_temp, llvm_generic_ptr_type);
    }

    auto llvm_append = [this]() noexcept {
        static constexpr auto name = "luisa.hip.print.append";
        if (auto function = _llvm_module->getFunction(name)) {
            return function;
        }
        auto llvm_void_type = llvm::Type::getVoidTy(_llvm_context);
        auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
        auto llvm_i64_type = llvm::Type::getInt64Ty(_llvm_context);
        auto llvm_global_ptr_type = llvm::PointerType::get(
            _llvm_context, amdgpu_address_space_global);
        auto llvm_generic_ptr_type = llvm::PointerType::get(
            _llvm_context, 0u);
        auto llvm_function_type = llvm::FunctionType::get(
            llvm_void_type,
            {llvm_global_ptr_type, llvm_i64_type,
             llvm_generic_ptr_type, llvm_i64_type},
            false);
        auto llvm_function = llvm::Function::Create(
            llvm_function_type, llvm::Function::PrivateLinkage,
            name, *_llvm_module);
        llvm_function->addFnAttr(llvm::Attribute::NoInline);
        llvm_function->addFnAttr(llvm::Attribute::NoUnwind);

        auto llvm_entry = llvm::BasicBlock::Create(
            _llvm_context, "entry", llvm_function);
        auto llvm_reserve = llvm::BasicBlock::Create(
            _llvm_context, "reserve", llvm_function);
        auto llvm_write = llvm::BasicBlock::Create(
            _llvm_context, "write", llvm_function);
        auto llvm_exit = llvm::BasicBlock::Create(
            _llvm_context, "exit", llvm_function);
        auto llvm_content = llvm_function->getArg(0u);
        auto llvm_capacity = llvm_function->getArg(1u);
        auto llvm_item = llvm_function->getArg(2u);
        auto llvm_item_size = llvm_function->getArg(3u);
        llvm_content->setName("content");
        llvm_capacity->setName("capacity");
        llvm_item->setName("item");
        llvm_item_size->setName("item.size");

        IB append_builder{llvm_entry};
        auto llvm_content_valid = append_builder.CreateICmpNE(
            llvm_content,
            llvm::ConstantPointerNull::get(llvm_global_ptr_type));
        auto llvm_has_capacity = append_builder.CreateICmpUGE(
            llvm_capacity, llvm_item_size);
        append_builder.CreateCondBr(
            append_builder.CreateAnd(
                llvm_content_valid, llvm_has_capacity),
            llvm_reserve, llvm_exit);

        append_builder.SetInsertPoint(llvm_reserve);
        auto llvm_offset = append_builder.CreateAtomicRMW(
            llvm::AtomicRMWInst::Add, llvm_content, llvm_item_size,
            llvm::MaybeAlign{alignof(size_t)},
            llvm::AtomicOrdering::Monotonic);
        llvm_offset->setSyncScopeID(llvm::SyncScope::System);
        auto llvm_item_end = append_builder.CreateAdd(
            llvm_offset, llvm_item_size);
        append_builder.CreateCondBr(
            append_builder.CreateICmpULE(
                llvm_item_end, llvm_capacity),
            llvm_write, llvm_exit);

        append_builder.SetInsertPoint(llvm_write);
        auto llvm_data_offset = append_builder.CreateAdd(
            llvm_offset, append_builder.getInt64(sizeof(size_t)));
        auto llvm_destination = append_builder.CreateInBoundsGEP(
            llvm_i8_type, llvm_content, llvm_data_offset);
        append_builder.CreateMemCpy(
            llvm_destination, llvm::MaybeAlign{1u},
            llvm_item, llvm::MaybeAlign{1u}, llvm_item_size);
        append_builder.CreateBr(llvm_exit);

        append_builder.SetInsertPoint(llvm_exit);
        append_builder.CreateRetVoid();
        return llvm_function;
    }();

    auto llvm_call = b.CreateCall(
        llvm_append,
        {func_ctx.llvm_print_buffer_content,
         func_ctx.llvm_print_buffer_capacity,
         llvm_temp_generic,
         b.getInt64(argument_pack_type->size())});
    llvm_call->addFnAttr(llvm::Attribute::NoUnwind);
}

llvm::Value *HIPCodegenLLVMImpl::_translate_clock_inst(
    IB &b, FunctionContext &func_ctx,
    const xir::ClockInst *inst) noexcept {
    static_cast<void>(func_ctx);
    static_cast<void>(inst);
    return b.CreateIntrinsic(
        b.getInt64Ty(), llvm::Intrinsic::readcyclecounter, {});
}

void HIPCodegenLLVMImpl::_translate_debug_break_inst(
    IB &b, FunctionContext &func_ctx,
    const xir::DebugBreakInst *inst) noexcept {
    static_cast<void>(func_ctx);
    static_cast<void>(inst);
    b.CreateIntrinsic(b.getVoidTy(), llvm::Intrinsic::debugtrap, {});
}

void HIPCodegenLLVMImpl::_translate_assert_inst(
    IB &b, FunctionContext &func_ctx,
    const xir::AssertInst *inst) noexcept {
    auto llvm_condition = _get_llvm_value(
        b, func_ctx, inst->condition());
    _create_assertion_with_message(
        b, llvm_condition,
        fmt::format("Assertion failed: {}\n", inst->message()));
}

void HIPCodegenLLVMImpl::_translate_assume_inst(
    IB &b, FunctionContext &func_ctx,
    const xir::AssumeInst *inst) noexcept {
    b.CreateAssumption(_get_llvm_value(
        b, func_ctx, inst->condition()));
}

void HIPCodegenLLVMImpl::_create_assertion_with_message(
    IB &b, llvm::Value *cond,
    luisa::string_view message) noexcept {
    if (_config.enable_debug_info) {
        auto llvm_message = llvm::ConstantDataArray::getString(
            _llvm_context, message);
        // ReSharper disable once CppDFAMemoryLeak
        auto llvm_message_global = new llvm::GlobalVariable(
            *_llvm_module, llvm_message->getType(), true,
            llvm::GlobalValue::PrivateLinkage, llvm_message,
            "luisa.assert.message", nullptr,
            llvm::GlobalValue::NotThreadLocal,
            amdgpu_address_space_constant);
        b.CreateCall(_get_assert_function(),
                     {cond, llvm_message_global});
    }
}

}// namespace luisa::compute::hip
