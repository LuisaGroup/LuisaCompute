#include "metal_codegen_llvm_impl.h"

namespace luisa::compute::metal::detail {

llvm::Function *MetalCodegenLLVMImpl::_function(const xir::Function *function) noexcept {
    if (auto iter = _functions.find(function); iter != _functions.end()) { return iter->second; }
    auto llvm_function = [&]() noexcept -> llvm::Function * {
        switch (function->derived_function_tag()) {
            case xir::DerivedFunctionTag::KERNEL: return _declare_kernel(static_cast<const xir::KernelFunction *>(function));
            case xir::DerivedFunctionTag::RASTER_STAGE: return _declare_raster_stage(static_cast<const xir::RasterStageFunction *>(function));
            case xir::DerivedFunctionTag::CALLABLE: return _declare_callable(static_cast<const xir::CallableFunction *>(function));
            case xir::DerivedFunctionTag::EXTERNAL: return _declare_external(static_cast<const xir::ExternalFunction *>(function));
        }
        LUISA_ERROR_WITH_LOCATION("Invalid XIR function kind.");
    }();
    auto [iter, inserted] = _functions.try_emplace(function, llvm_function);
    LUISA_ASSERT(inserted, "Failed to cache a Metal LLVM function.");
    return iter->second;
}

llvm::Function *MetalCodegenLLVMImpl::_declare_kernel(const xir::KernelFunction *function) noexcept {
    llvm::SmallVector<llvm::Type *> arguments;
    for (auto argument : function->arguments()) {
        auto llvm_type = argument->is_reference() &&
                                 !is_indirect_dispatch_buffer_type(argument->type()) ?
                             llvm::PointerType::get(
                                 _context, air_address_space_generic) :
                             _type(argument->type())->reg_type;
        arguments.emplace_back(llvm_type);
        if (argument->type()->is_texture() &&
            _texture_needs_sampled_split(argument)) {
            arguments.emplace_back(llvm_type);
        }
    }
    auto i32 = llvm::Type::getInt32Ty(_context);
    auto i32x3 = llvm::FixedVectorType::get(i32, 3u);
    arguments.append({i32x3, i32, i32x3, i32x3, i32x3, i32x3, i32, i32});
    auto function_type = llvm::FunctionType::get(llvm::Type::getVoidTy(_context), arguments, false);
    auto llvm_function = llvm::Function::Create(function_type, llvm::GlobalValue::PrivateLinkage,
                                                "kernel_main_impl", _module);
    llvm_function->addFnAttr(llvm::Attribute::AlwaysInline);
    llvm_function->addFnAttr(llvm::Attribute::Convergent);
    _set_float_control_attributes(llvm_function);
    return llvm_function;
}

llvm::Function *MetalCodegenLLVMImpl::_declare_raster_stage(const xir::RasterStageFunction *function) noexcept {
    llvm::SmallVector<llvm::Type *> arguments;
    for (auto argument : function->arguments()) {
        auto llvm_type = argument->is_reference() ?
                             llvm::PointerType::get(
                                 _context, air_address_space_generic) :
                             _type(argument->type())->reg_type;
        arguments.emplace_back(llvm_type);
        if (argument->type()->is_texture() &&
            _texture_needs_sampled_split(argument)) {
            arguments.emplace_back(llvm_type);
        }
    }
    auto i32 = llvm::Type::getInt32Ty(_context);
    auto f32x3 = llvm::FixedVectorType::get(llvm::Type::getFloatTy(_context), 3u);
    arguments.append({i32, i32, f32x3});
    auto return_type = function->type() == nullptr ?
                           llvm::Type::getVoidTy(_context) :
                           _type(function->type())->reg_type;
    auto function_type = llvm::FunctionType::get(return_type, arguments, false);
    auto name = function->stage() == xir::RasterStage::VERTEX ?
                    "vertex_main_impl" :
                    "fragment_main_impl";
    auto llvm_function = llvm::Function::Create(
        function_type, llvm::GlobalValue::PrivateLinkage, name, _module);
    llvm_function->addFnAttr(llvm::Attribute::AlwaysInline);
    llvm_function->addFnAttr(llvm::Attribute::Convergent);
    _set_float_control_attributes(llvm_function);
    return llvm_function;
}

llvm::Function *MetalCodegenLLVMImpl::_declare_callable(const xir::CallableFunction *function) noexcept {
    llvm::SmallVector<llvm::Type *> arguments;
    for (auto argument : function->arguments()) {
        auto llvm_type = argument->is_reference() &&
                                 !is_indirect_dispatch_buffer_type(argument->type()) ?
                             llvm::PointerType::get(
                                 _context, air_address_space_generic) :
                             _type(argument->type())->reg_type;
        arguments.emplace_back(llvm_type);
        if (argument->type()->is_texture() &&
            _texture_needs_sampled_split(argument)) {
            arguments.emplace_back(llvm_type);
        }
    }
    auto i32 = llvm::Type::getInt32Ty(_context);
    if (_config.program == MetalAIRProgram::COMPUTE) {
        auto i32x3 = llvm::FixedVectorType::get(i32, 3u);
        arguments.append({i32x3, i32, i32x3, i32x3, i32x3, i32x3, i32, i32});
    } else {
        auto f32x3 = llvm::FixedVectorType::get(llvm::Type::getFloatTy(_context), 3u);
        arguments.append({i32, i32, f32x3});
    }
    auto return_type = function->type() == nullptr ? llvm::Type::getVoidTy(_context) : _type(function->type())->reg_type;
    auto function_type = llvm::FunctionType::get(return_type, arguments, false);
    auto llvm_function = llvm::Function::Create(function_type, llvm::GlobalValue::PrivateLinkage,
                                                function->name().value_or("callable"), _module);
    llvm_function->addFnAttr(llvm::Attribute::AlwaysInline);
    llvm_function->addFnAttr(llvm::Attribute::Convergent);
    _set_float_control_attributes(llvm_function);
    return llvm_function;
}

llvm::Function *MetalCodegenLLVMImpl::_declare_external(
    const xir::ExternalFunction *function) noexcept {
    auto name = function->name();
    LUISA_ASSERT(name.has_value() && !name->empty(),
                 "Metal AIR external functions must have a non-empty symbol name.");
    // ExternalCallable is an LLVM module ABI boundary. Unlike generated
    // callables, it receives no hidden dispatch/raster state arguments.
    llvm::SmallVector<llvm::Type *> arguments;
    arguments.reserve(function->arguments().count_size());
    for (auto argument : function->arguments()) {
        arguments.emplace_back(
            argument->is_reference() ?
                llvm::PointerType::get(_context, air_address_space_generic) :
                _type(argument->type())->reg_type);
    }
    auto return_type = function->type() == nullptr ?
                           llvm::Type::getVoidTy(_context) :
                           _type(function->type())->reg_type;
    auto function_type = llvm::FunctionType::get(
        return_type, arguments, false);
    auto symbol = llvm::StringRef{name->data(), name->size()};
    if (auto existing = _module.getFunction(symbol)) {
        LUISA_ASSERT(existing->getFunctionType() == function_type,
                     "Metal AIR external function '{}' has an incompatible LLVM ABI.",
                     *name);
        return existing;
    }
    return llvm::Function::Create(
        function_type, llvm::GlobalValue::ExternalLinkage,
        symbol, _module);
}

void MetalCodegenLLVMImpl::_set_float_control_attributes(
    llvm::Function *function) const noexcept {
    function->addFnAttr("no-trapping-math", "true");
    if (_config.enable_fast_math) {
        function->addFnAttr("approx-func-fp-math", "true");
        function->addFnAttr("no-infs-fp-math", "true");
        function->addFnAttr("no-nans-fp-math", "true");
        function->addFnAttr("no-signed-zeros-fp-math", "true");
        function->addFnAttr("unsafe-fp-math", "true");
    }
}

void MetalCodegenLLVMImpl::_bind_state_parameters(FunctionContext &context, llvm::Function::arg_iterator iterator) noexcept {
    if (_config.program != MetalAIRProgram::COMPUTE) {
        context.kernel_id = iterator++;
        context.raster_object_id = iterator++;
        context.raster_barycentrics = iterator++;
        LUISA_ASSERT(iterator == context.function->arg_end(), "Unexpected Metal raster state parameter count.");
        context.kernel_id->setName("sreg.primitive.id");
        context.raster_object_id->setName("sreg.object.id");
        context.raster_barycentrics->setName("sreg.barycentrics");
        return;
    }
    context.dispatch_size = iterator++;
    context.kernel_id = iterator++;
    context.thread_id = iterator++;
    context.block_id = iterator++;
    context.dispatch_id = iterator++;
    context.block_size = iterator++;
    context.warp_size = iterator++;
    context.warp_lane_id = iterator++;
    LUISA_ASSERT(iterator == context.function->arg_end(), "Unexpected Metal kernel state parameter count.");
    context.dispatch_size->setName("sreg.dispatch.size");
    context.kernel_id->setName("sreg.kernel.id");
    context.thread_id->setName("sreg.thread.id");
    context.block_id->setName("sreg.block.id");
    context.dispatch_id->setName("sreg.dispatch.id");
    context.block_size->setName("sreg.block.size");
    context.warp_size->setName("sreg.warp.size");
    context.warp_lane_id->setName("sreg.warp.lane.id");
}

void MetalCodegenLLVMImpl::_append_state_arguments(const FunctionContext &context, llvm::SmallVectorImpl<llvm::Value *> &arguments) noexcept {
    if (_config.program != MetalAIRProgram::COMPUTE) {
        arguments.append({context.kernel_id, context.raster_object_id,
                          context.raster_barycentrics});
        return;
    }
    arguments.append({context.dispatch_size, context.kernel_id,
                      context.thread_id, context.block_id, context.dispatch_id, context.block_size,
                      context.warp_size, context.warp_lane_id});
}

llvm::Function *MetalCodegenLLVMImpl::_translate_kernel(const xir::KernelFunction *function) noexcept {
    auto llvm_function = _function(function);
    LUISA_ASSERT(llvm_function->isDeclaration(), "Metal LLVM kernel implementation was translated twice.");
    FunctionContext context{llvm_function};
    auto iterator = llvm_function->arg_begin();
    for (auto argument : function->arguments()) {
        context.values.try_emplace(argument, iterator++);
        if (argument->type()->is_texture() &&
            _texture_needs_sampled_split(argument)) {
            context.sampled_textures.try_emplace(argument, iterator++);
        }
    }
    _bind_state_parameters(context, iterator);
    auto body = _translate_function(context, function);
    IB builder{context.entry_block};
    builder.CreateBr(body);
    _emit_kernel_entry(function, llvm_function, _config.entry == MetalAIRKernelEntry::INDIRECT);
    return llvm_function;
}

llvm::Function *MetalCodegenLLVMImpl::_translate_raster_stage(const xir::RasterStageFunction *function) noexcept {
    auto llvm_function = _function(function);
    LUISA_ASSERT(llvm_function->isDeclaration(), "Metal LLVM raster implementation was translated twice.");
    FunctionContext context{llvm_function};
    auto iterator = llvm_function->arg_begin();
    for (auto argument : function->arguments()) {
        context.values.try_emplace(argument, iterator++);
        if (argument->type()->is_texture() &&
            _texture_needs_sampled_split(argument)) {
            context.sampled_textures.try_emplace(argument, iterator++);
        }
    }
    _bind_state_parameters(context, iterator);
    auto body = _translate_function(context, function);
    IB builder{context.entry_block};
    builder.CreateBr(body);
    if (function->stage() == xir::RasterStage::VERTEX) {
        _emit_raster_vertex_entry(function, llvm_function);
    } else {
        _emit_raster_fragment_entry(function, llvm_function);
    }
    return llvm_function;
}

llvm::Function *MetalCodegenLLVMImpl::_translate_callable(const xir::CallableFunction *function) noexcept {
    auto llvm_function = _function(function);
    LUISA_ASSERT(llvm_function->isDeclaration(), "Metal LLVM callable was translated twice.");
    FunctionContext context{llvm_function};
    auto iterator = llvm_function->arg_begin();
    for (auto argument : function->arguments()) {
        context.values.try_emplace(argument, iterator++);
        if (argument->type()->is_texture() &&
            _texture_needs_sampled_split(argument)) {
            context.sampled_textures.try_emplace(argument, iterator++);
        }
    }
    _bind_state_parameters(context, iterator);
    auto body = _translate_function(context, function);
    IB builder{context.entry_block};
    builder.CreateBr(body);
    return llvm_function;
}

namespace {

template<typename F>
void traverse_metal_dom_tree(luisa::unordered_set<const xir::DomTreeNode *> &visited,
                             const xir::DomTreeNode *node, const F &visit) noexcept {
    if (visited.emplace(node).second) {
        visit(node->block());
        for (auto child : node->children()) { traverse_metal_dom_tree(visited, child, visit); }
    }
}

}// namespace

llvm::BasicBlock *MetalCodegenLLVMImpl::_translate_function(FunctionContext &context, const xir::FunctionDefinition *function) noexcept {
    for (auto block : function->basic_blocks()) {
        auto llvm_block = llvm::BasicBlock::Create(_context, block->name().value_or(""), context.function);
        context.values.try_emplace(block, llvm_block);
    }
    auto tree = xir::compute_dom_tree(const_cast<xir::FunctionDefinition *>(function));
    LUISA_ASSERT(tree.root() != nullptr && tree.root()->block() == function->body_block(), "Invalid XIR dominance tree.");
    luisa::unordered_set<const xir::DomTreeNode *> visited;
    traverse_metal_dom_tree(visited, tree.root(), [this, &context](const xir::BasicBlock *block) noexcept {
        auto llvm_block = context.value<llvm::BasicBlock>(block);
        IB builder{llvm_block};
        if (_config.enable_fast_math) {
            builder.setFastMathFlags(llvm::FastMathFlags::getFast());
        }
        for (auto instruction : block->instructions()) { _translate_instruction(builder, context, instruction); }
        context.block_exits.insert_or_assign(block, builder.GetInsertBlock());
    });
    IB detached_builder{_context};
    for (auto phi : context.pending_phi_nodes) {
        auto llvm_phi = context.value<llvm::PHINode>(phi);
        for (auto i = 0u; i < phi->incoming_count(); i++) {
            auto [value, block] = phi->incoming(i);
            llvm_phi->addIncoming(_value(detached_builder, context, value), context.block_exit(block));
        }
    }
    for (auto block : function->basic_blocks()) {
        auto llvm_block = context.block_exit(block);
        LUISA_ASSERT(llvm_block->getTerminator() != nullptr,
                     "XIR block '{}' has no terminator after Metal LLVM translation.", block->name().value_or(""));
    }
    return context.value<llvm::BasicBlock>(function->body_block());
}

void MetalCodegenLLVMImpl::_emit_kernel_entry(const xir::KernelFunction *kernel, llvm::Function *implementation, bool indirect) noexcept {
    auto i32 = llvm::Type::getInt32Ty(_context);
    auto i32x3 = llvm::FixedVectorType::get(i32, 3u);
    auto i32x4 = llvm::FixedVectorType::get(i32, 4u);
    auto args_pointer = llvm::PointerType::get(_context, air_address_space_constant);
    auto dispatch_pointer = llvm::PointerType::get(_context, indirect ? air_address_space_device : air_address_space_constant);
    auto function_type = llvm::FunctionType::get(llvm::Type::getVoidTy(_context),
                                                 {args_pointer, dispatch_pointer,
                                                  i32x3, i32x3, i32x3, i32x3, i32, i32},
                                                 false);
    auto name = indirect ? "kernel_main_indirect" : "kernel_main";
    auto function = llvm::Function::Create(function_type, llvm::GlobalValue::ExternalLinkage, name, _module);
    function->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);
    function->setMustProgress();
    function->setDoesNotFreeMemory();
    function->setDoesNotThrow();
    function->setWillReturn();
    function->addFnAttr(llvm::Attribute::Convergent);
    function->addFnAttr("no-builtins");
    function->addFnAttr("frame-pointer", "all");
    _set_float_control_attributes(function);
    auto args = function->getArg(0);
    auto dispatch_data = function->getArg(1);
    args->setName("args");
    dispatch_data->setName(indirect ? "dispatch_size_and_kernel_id" : "dispatch_size");
    auto layout = _root_argument_layout();
    auto dispatch_type = indirect ? static_cast<llvm::Type *>(i32x4) : i32x3;
    function->setMetadata(
        "arg_eltypes",
        llvm::MDNode::get(
            _context,
            {md_i32(_context, 0u), llvm::ValueAsMetadata::get(llvm::UndefValue::get(layout.type)),
             md_i32(_context, 1u), llvm::ValueAsMetadata::get(llvm::UndefValue::get(dispatch_type))}));
    args->addAttr(llvm::Attribute::NoUndef);
    args->addAttr(llvm::Attribute::ReadOnly);
    args->addAttr(llvm::Attribute::getWithAlignment(_context, llvm::Align{kernel_argument_alignment}));
    args->addAttr(llvm::Attribute::getWithDereferenceableBytes(_context, layout.size));
    args->addAttr(llvm::Attribute::get(_context, "air-buffer-no-alias"));
    dispatch_data->addAttr(llvm::Attribute::NoUndef);
    dispatch_data->addAttr(llvm::Attribute::ReadOnly);
    dispatch_data->addAttr(llvm::Attribute::getWithAlignment(_context, llvm::Align{16u}));
    dispatch_data->addAttr(llvm::Attribute::getWithDereferenceableBytes(_context, 16u));
    dispatch_data->addAttr(llvm::Attribute::get(_context, "air-buffer-no-alias"));
    for (auto i = 2u; i < function->arg_size(); i++) { function->getArg(i)->addAttr(llvm::Attribute::NoUndef); }

    auto entry = llvm::BasicBlock::Create(_context, "entry", function);
    IB builder{entry};
    auto dispatch_raw = builder.CreateAlignedLoad(dispatch_type, dispatch_data, llvm::Align{16u});
    llvm::Value *dispatch_size = dispatch_raw;
    llvm::Value *kernel_id = builder.getInt32(0u);
    if (indirect) {
        dispatch_size = builder.CreateShuffleVector(dispatch_raw, {0, 1, 2});
        kernel_id = builder.CreateExtractElement(dispatch_raw, 3u);
    }
    auto thread_id = function->getArg(2u);
    auto block_id = function->getArg(3u);
    auto dispatch_id = function->getArg(4u);
    auto block_size = function->getArg(5u);
    auto warp_size = function->getArg(6u);
    auto warp_lane_id = function->getArg(7u);
    auto in_bounds_components = builder.CreateICmpUGT(dispatch_size, dispatch_id);
    auto all_type = llvm::FunctionType::get(builder.getInt1Ty(), {in_bounds_components->getType()}, false);
    auto all_function = llvm::cast<llvm::Function>(
        _module.getOrInsertFunction("air.all.v3i1", all_type).getCallee());
    all_function->setDoesNotAccessMemory();
    all_function->setDoesNotFreeMemory();
    all_function->setDoesNotThrow();
    all_function->setNoSync();
    all_function->setWillReturn();
    auto in_bounds = builder.CreateCall(all_function, {in_bounds_components});
    auto body = llvm::BasicBlock::Create(_context, "in_bounds", function);
    auto exit = llvm::BasicBlock::Create(_context, "exit", function);
    builder.CreateCondBr(in_bounds, body, exit);
    builder.SetInsertPoint(body);
    llvm::SmallVector<llvm::Value *> arguments;
    auto argument_index = 0u;
    for (auto argument : kernel->arguments()) {
        auto field_pointer = builder.CreateStructGEP(
            layout.type, args, layout.member_indices[argument_index]);
        if (argument->is_reference()) {
            LUISA_ASSERT(
                is_indirect_dispatch_buffer_type(argument->type()),
                "Unsupported Metal AIR reference kernel argument.");
            arguments.emplace_back(
                _load(builder, field_pointer, argument->type()));
        } else if (argument->type()->is_buffer()) {
            auto buffer_type = _buffer(argument->type()->element());
            auto device_pointer_field = builder.CreateStructGEP(buffer_type, field_pointer, 0u);
            auto device_pointer = builder.CreateAlignedLoad(
                buffer_type->getElementType(0u), device_pointer_field, llvm::Align{16u});
            auto size_pointer = builder.CreateStructGEP(buffer_type, field_pointer, 1u);
            auto size = builder.CreateAlignedLoad(buffer_type->getElementType(1u), size_pointer, llvm::Align{8u});
            auto buffer = static_cast<llvm::Value *>(llvm::PoisonValue::get(buffer_type));
            buffer = builder.CreateInsertValue(buffer, device_pointer, 0u);
            buffer = builder.CreateInsertValue(buffer, size, 1u);
            arguments.emplace_back(buffer);
        } else {
            arguments.emplace_back(_load(builder, field_pointer, argument->type()));
        }
        if (argument->type()->is_texture() &&
            _texture_needs_sampled_split(argument)) {
            arguments.emplace_back(_load_root_argument(
                builder, args, argument, argument_index, true));
        }
        argument_index++;
    }
    arguments.append({dispatch_size, kernel_id, thread_id, block_id, dispatch_id, block_size, warp_size, warp_lane_id});
    auto implementation_call = builder.CreateCall(implementation, arguments);
    implementation_call->setConvergent();
    builder.CreateBr(exit);
    builder.SetInsertPoint(exit);
    builder.CreateRetVoid();
    _add_kernel_metadata(function, layout.size, indirect);
}

void MetalCodegenLLVMImpl::_emit_raster_vertex_entry(
    const xir::RasterStageFunction *stage,
    llvm::Function *implementation) noexcept {
    LUISA_ASSERT(
        _config.program == MetalAIRProgram::RASTER_VERTEX &&
            stage->stage() == xir::RasterStage::VERTEX,
        "Invalid Metal AIR vertex-stage entry request.");
    luisa::vector<const xir::Argument *> stage_arguments;
    for (auto argument : stage->arguments()) {
        stage_arguments.emplace_back(argument);
    }
    LUISA_ASSERT(!stage_arguments.empty(),
                 "Metal AIR vertex stage requires an AppData payload argument.");
    llvm::SmallVector<llvm::Type *> parameter_types;
    parameter_types.reserve(_config.raster.vertex_attributes.size() + 4u);
    for (auto attribute : _config.raster.vertex_attributes) {
        parameter_types.emplace_back(_raster_vertex_input(attribute.format).type);
    }
    auto i32 = llvm::Type::getInt32Ty(_context);
    auto f32 = llvm::Type::getFloatTy(_context);
    auto f32x3 = llvm::FixedVectorType::get(f32, 3u);
    parameter_types.append(
        {i32, i32,
         llvm::PointerType::get(_context, air_address_space_constant),
         llvm::PointerType::get(_context, air_address_space_constant)});

    auto return_type = stage->type();
    llvm::SmallVector<llvm::Type *> output_types;
    if (return_type->is_structure()) {
        for (auto member : return_type->members()) {
            output_types.emplace_back(_type(member)->reg_type);
        }
    } else {
        output_types.emplace_back(_type(return_type)->reg_type);
    }
    auto entry_return_type = output_types.size() == 1u ?
                                 output_types.front() :
                                 static_cast<llvm::Type *>(
                                     llvm::StructType::get(_context, output_types, true));
    auto function_type = llvm::FunctionType::get(
        entry_return_type, parameter_types, false);
    auto function = llvm::Function::Create(
        function_type, llvm::GlobalValue::ExternalLinkage,
        "vertex_main", _module);
    function->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);
    function->setMustProgress();
    function->setDoesNotFreeMemory();
    function->setDoesNotThrow();
    function->setWillReturn();
    function->addFnAttr(llvm::Attribute::Convergent);
    function->addFnAttr("no-builtins");
    function->addFnAttr("frame-pointer", "all");
    _set_float_control_attributes(function);

    auto attribute_count = _config.raster.vertex_attributes.size();
    auto vertex_id = function->getArg(attribute_count);
    auto instance_id = function->getArg(attribute_count + 1u);
    auto root = function->getArg(attribute_count + 2u);
    auto object_id_pointer = function->getArg(attribute_count + 3u);
    vertex_id->setName("vertex_id");
    instance_id->setName("instance_id");
    root->setName("args");
    object_id_pointer->setName("object_id");
    vertex_id->addAttr(llvm::Attribute::NoUndef);
    instance_id->addAttr(llvm::Attribute::NoUndef);
    auto layout = _root_argument_layout();
    root->addAttr(llvm::Attribute::NoUndef);
    root->addAttr(llvm::Attribute::ReadOnly);
    root->addAttr(llvm::Attribute::getWithAlignment(
        _context, llvm::Align{kernel_argument_alignment}));
    root->addAttr(llvm::Attribute::getWithDereferenceableBytes(
        _context, layout.size));
    root->addAttr(llvm::Attribute::get(_context, "air-buffer-no-alias"));
    object_id_pointer->addAttr(llvm::Attribute::NoUndef);
    object_id_pointer->addAttr(llvm::Attribute::ReadOnly);
    object_id_pointer->addAttr(llvm::Attribute::getWithAlignment(
        _context, llvm::Align{4u}));
    object_id_pointer->addAttr(
        llvm::Attribute::getWithDereferenceableBytes(_context, 4u));
    object_id_pointer->addAttr(
        llvm::Attribute::get(_context, "air-buffer-no-alias"));
    _set_air_pointer_element_types(
        function,
        {{static_cast<unsigned>(attribute_count + 2u), layout.type},
         {static_cast<unsigned>(attribute_count + 3u), i32}});

    auto entry = llvm::BasicBlock::Create(_context, "entry", function);
    IB builder{entry};
    auto app_data_type = stage_arguments[0u]->type();
    auto app_data_info = _type(app_data_type);
    auto app_data = static_cast<llvm::Value *>(
        llvm::Constant::getNullValue(app_data_info->reg_type));
    auto convert_attribute = [&](llvm::Value *value,
                                 const RasterVertexInput &input,
                                 uint32_t destination_dimension) noexcept {
        auto destination_type = llvm::FixedVectorType::get(
            f32, destination_dimension);
        auto result = static_cast<llvm::Value *>(
            llvm::Constant::getNullValue(destination_type));
        for (auto lane = 0u;
             lane < std::min(input.dimension, destination_dimension);
             lane++) {
            auto component = input.dimension == 1u ?
                                 value :
                                 builder.CreateExtractElement(value, lane);
            if (component->getType()->isIntegerTy()) {
                component = input.signed_integer ?
                                builder.CreateSIToFP(component, f32) :
                                builder.CreateUIToFP(component, f32);
            } else if (component->getType()->isHalfTy()) {
                component = builder.CreateFPExt(component, f32);
            }
            result = builder.CreateInsertElement(result, component, lane);
        }
        return result;
    };
    auto set_member = [&](uint32_t member, llvm::Value *value) noexcept {
        app_data = builder.CreateInsertValue(app_data, value, member);
    };
    for (auto index = 0u; index < attribute_count; index++) {
        auto descriptor = _config.raster.vertex_attributes[index];
        auto input = _raster_vertex_input(descriptor.format);
        auto value = function->getArg(index);
        switch (descriptor.semantic) {
            case VertexAttributeType::Position:
                set_member(0u, convert_attribute(value, input, 3u));
                break;
            case VertexAttributeType::Normal:
                set_member(1u, convert_attribute(value, input, 3u));
                break;
            case VertexAttributeType::Tangent:
                set_member(2u, convert_attribute(value, input, 4u));
                break;
            case VertexAttributeType::Color:
                set_member(3u, convert_attribute(value, input, 4u));
                break;
            case VertexAttributeType::UV0: [[fallthrough]];
            case VertexAttributeType::UV1: [[fallthrough]];
            case VertexAttributeType::UV2: [[fallthrough]];
            case VertexAttributeType::UV3: {
                auto uv_index = static_cast<uint32_t>(descriptor.semantic) -
                                static_cast<uint32_t>(VertexAttributeType::UV0);
                auto uv = builder.CreateExtractValue(app_data, 4u);
                uv = builder.CreateInsertValue(
                    uv, convert_attribute(value, input, 2u), uv_index);
                set_member(4u, uv);
                break;
            }
        }
    }
    set_member(5u, vertex_id);
    set_member(6u, instance_id);

    llvm::SmallVector<llvm::Value *> arguments{app_data};
    auto root_index = _config.raster.stage_root_argument_offset;
    for (auto i = 1u; i < stage_arguments.size(); i++, root_index++) {
        auto argument = stage_arguments[i];
        arguments.emplace_back(_load_root_argument(
            builder, root, argument, root_index));
        if (argument->type()->is_texture() &&
            _texture_needs_sampled_split(argument)) {
            arguments.emplace_back(_load_root_argument(
                builder, root, argument, root_index, true));
        }
    }
    auto object_id = builder.CreateAlignedLoad(
        i32, object_id_pointer, llvm::Align{4u});
    arguments.append(
        {builder.getInt32(0u), object_id,
         llvm::Constant::getNullValue(f32x3)});
    auto result = builder.CreateCall(implementation, arguments);
    result->setConvergent();
    if (output_types.size() == 1u) {
        builder.CreateRet(result);
    } else {
        auto output = static_cast<llvm::Value *>(
            llvm::PoisonValue::get(entry_return_type));
        for (auto i = 0u; i < output_types.size(); i++) {
            output = builder.CreateInsertValue(
                output, builder.CreateExtractValue(result, i), i);
        }
        builder.CreateRet(output);
    }
    _add_raster_vertex_metadata(function, output_types);
}

void MetalCodegenLLVMImpl::_emit_raster_fragment_entry(
    const xir::RasterStageFunction *stage,
    llvm::Function *implementation) noexcept {
    LUISA_ASSERT(
        _config.program == MetalAIRProgram::RASTER_FRAGMENT &&
            stage->stage() == xir::RasterStage::FRAGMENT,
        "Invalid Metal AIR fragment-stage entry request.");
    luisa::vector<const xir::Argument *> stage_arguments;
    for (auto argument : stage->arguments()) {
        stage_arguments.emplace_back(argument);
    }
    LUISA_ASSERT(!stage_arguments.empty(),
                 "Metal AIR fragment stage requires a vertex payload argument.");
    auto payload_type = stage_arguments[0u]->type();
    llvm::SmallVector<llvm::Type *> payload_members;
    if (payload_type->is_structure()) {
        for (auto member : payload_type->members()) {
            payload_members.emplace_back(_type(member)->reg_type);
        }
    } else {
        payload_members.emplace_back(_type(payload_type)->reg_type);
    }
    auto i32 = llvm::Type::getInt32Ty(_context);
    auto f32x3 = llvm::FixedVectorType::get(
        llvm::Type::getFloatTy(_context), 3u);
    llvm::SmallVector<llvm::Type *> parameter_types{payload_members};
    parameter_types.append(
        {i32, f32x3,
         llvm::PointerType::get(_context, air_address_space_constant),
         llvm::PointerType::get(_context, air_address_space_constant)});

    auto return_type = stage->type();
    llvm::SmallVector<llvm::Type *> output_types;
    if (return_type->is_structure()) {
        for (auto member : return_type->members()) {
            output_types.emplace_back(_type(member)->reg_type);
        }
    } else {
        output_types.emplace_back(_type(return_type)->reg_type);
    }
    LUISA_ASSERT(!output_types.empty() && output_types.size() <= 8u,
                 "Metal AIR fragment stages must return between 1 and 8 color targets.");
    _result.fragment_output_count = static_cast<uint32_t>(output_types.size());
    auto entry_return_type = output_types.size() == 1u ?
                                 output_types.front() :
                                 static_cast<llvm::Type *>(
                                     llvm::StructType::get(_context, output_types, true));
    auto function_type = llvm::FunctionType::get(
        entry_return_type, parameter_types, false);
    auto function = llvm::Function::Create(
        function_type, llvm::GlobalValue::ExternalLinkage,
        "fragment_main", _module);
    function->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);
    function->setMustProgress();
    function->setDoesNotFreeMemory();
    function->setDoesNotThrow();
    function->setWillReturn();
    function->addFnAttr(llvm::Attribute::Convergent);
    function->addFnAttr("no-builtins");
    function->addFnAttr("frame-pointer", "all");
    _set_float_control_attributes(function);

    auto primitive_index = payload_members.size();
    auto barycentrics_index = primitive_index + 1u;
    auto root_index_in_function = primitive_index + 2u;
    auto object_id_index = primitive_index + 3u;
    auto primitive_id = function->getArg(primitive_index);
    auto barycentrics = function->getArg(barycentrics_index);
    auto root = function->getArg(root_index_in_function);
    auto object_id_pointer = function->getArg(object_id_index);
    primitive_id->setName("primitive_id");
    barycentrics->setName("barycentrics");
    root->setName("args");
    object_id_pointer->setName("object_id");
    primitive_id->addAttr(llvm::Attribute::NoUndef);
    barycentrics->addAttr(llvm::Attribute::NoUndef);
    auto layout = _root_argument_layout();
    root->addAttr(llvm::Attribute::NoUndef);
    root->addAttr(llvm::Attribute::ReadOnly);
    root->addAttr(llvm::Attribute::getWithAlignment(
        _context, llvm::Align{kernel_argument_alignment}));
    root->addAttr(llvm::Attribute::getWithDereferenceableBytes(
        _context, layout.size));
    root->addAttr(llvm::Attribute::get(_context, "air-buffer-no-alias"));
    object_id_pointer->addAttr(llvm::Attribute::NoUndef);
    object_id_pointer->addAttr(llvm::Attribute::ReadOnly);
    object_id_pointer->addAttr(llvm::Attribute::getWithAlignment(
        _context, llvm::Align{4u}));
    object_id_pointer->addAttr(
        llvm::Attribute::getWithDereferenceableBytes(_context, 4u));
    object_id_pointer->addAttr(
        llvm::Attribute::get(_context, "air-buffer-no-alias"));
    _set_air_pointer_element_types(
        function,
        {{static_cast<unsigned>(root_index_in_function), layout.type},
         {static_cast<unsigned>(object_id_index), i32}});

    auto entry = llvm::BasicBlock::Create(_context, "entry", function);
    IB builder{entry};
    llvm::Value *payload = nullptr;
    if (payload_type->is_structure()) {
        payload = llvm::PoisonValue::get(_type(payload_type)->reg_type);
        for (auto i = 0u; i < payload_members.size(); i++) {
            payload = builder.CreateInsertValue(
                payload, function->getArg(i), i);
        }
    } else {
        payload = function->getArg(0u);
    }
    auto object_id = builder.CreateAlignedLoad(
        i32, object_id_pointer, llvm::Align{4u});
    llvm::SmallVector<llvm::Value *> arguments{payload};
    auto root_argument_index = _config.raster.stage_root_argument_offset;
    for (auto i = 1u; i < stage_arguments.size();
         i++, root_argument_index++) {
        auto argument = stage_arguments[i];
        arguments.emplace_back(_load_root_argument(
            builder, root, argument, root_argument_index));
        if (argument->type()->is_texture() &&
            _texture_needs_sampled_split(argument)) {
            arguments.emplace_back(_load_root_argument(
                builder, root, argument, root_argument_index, true));
        }
    }
    arguments.append({primitive_id, object_id, barycentrics});
    auto result = builder.CreateCall(implementation, arguments);
    result->setConvergent();
    if (output_types.size() == 1u) {
        builder.CreateRet(result);
    } else {
        auto output = static_cast<llvm::Value *>(
            llvm::PoisonValue::get(entry_return_type));
        for (auto i = 0u; i < output_types.size(); i++) {
            output = builder.CreateInsertValue(
                output, builder.CreateExtractValue(result, i), i);
        }
        builder.CreateRet(output);
    }
    _add_raster_fragment_metadata(function, output_types);
}

void MetalCodegenLLVMImpl::_translate_instruction(IB &builder, FunctionContext &function, const xir::Instruction *inst) noexcept {
    llvm::Value *result{nullptr};
    switch (inst->derived_instruction_tag()) {
        case xir::DerivedInstructionTag::IF: {
            auto instruction = static_cast<const xir::IfInst *>(inst);
            builder.CreateCondBr(_value(builder, function, instruction->condition()),
                                 function.value<llvm::BasicBlock>(instruction->true_block()),
                                 function.value<llvm::BasicBlock>(instruction->false_block()));
            break;
        }
        case xir::DerivedInstructionTag::SWITCH: {
            auto instruction = static_cast<const xir::SwitchInst *>(inst);
            auto selector = _value(builder, function, instruction->value());
            auto selector_type = llvm::cast<llvm::IntegerType>(selector->getType());
            auto llvm_switch = builder.CreateSwitch(selector,
                                                    function.value<llvm::BasicBlock>(instruction->default_block()),
                                                    instruction->case_count());
            for (auto i = 0u; i < instruction->case_count(); i++) {
                llvm_switch->addCase(llvm::ConstantInt::get(selector_type, instruction->case_value(i)),
                                     function.value<llvm::BasicBlock>(instruction->case_block(i)));
            }
            break;
        }
        case xir::DerivedInstructionTag::INDEXED_BRANCH: {
            auto instruction = static_cast<const xir::IndexedBranchInst *>(inst);
            auto selector = _value(builder, function, instruction->value());
            auto selector_type = llvm::cast<llvm::IntegerType>(selector->getType());
            auto llvm_switch = builder.CreateSwitch(
                selector,
                function.value<llvm::BasicBlock>(instruction->default_block()),
                instruction->case_count());
            for (auto i = 0u; i < instruction->case_count(); i++) {
                llvm_switch->addCase(
                    llvm::ConstantInt::get(selector_type, instruction->case_value(i)),
                    function.value<llvm::BasicBlock>(instruction->case_block(i)));
            }
            break;
        }
        case xir::DerivedInstructionTag::LOOP: {
            auto instruction = static_cast<const xir::LoopInst *>(inst);
            builder.CreateBr(function.value<llvm::BasicBlock>(instruction->prepare_block()));
            break;
        }
        case xir::DerivedInstructionTag::SIMPLE_LOOP: {
            auto instruction = static_cast<const xir::SimpleLoopInst *>(inst);
            builder.CreateBr(function.value<llvm::BasicBlock>(instruction->body_block()));
            break;
        }
        case xir::DerivedInstructionTag::BRANCH: {
            auto instruction = static_cast<const xir::BranchInst *>(inst);
            builder.CreateBr(function.value<llvm::BasicBlock>(instruction->target_block()));
            break;
        }
        case xir::DerivedInstructionTag::CONDITIONAL_BRANCH: {
            auto instruction = static_cast<const xir::ConditionalBranchInst *>(inst);
            builder.CreateCondBr(_value(builder, function, instruction->condition()),
                                 function.value<llvm::BasicBlock>(instruction->true_block()),
                                 function.value<llvm::BasicBlock>(instruction->false_block()));
            break;
        }
        case xir::DerivedInstructionTag::BREAK: {
            auto instruction = static_cast<const xir::BreakInst *>(inst);
            builder.CreateBr(function.value<llvm::BasicBlock>(instruction->target_block()));
            break;
        }
        case xir::DerivedInstructionTag::CONTINUE: {
            auto instruction = static_cast<const xir::ContinueInst *>(inst);
            builder.CreateBr(function.value<llvm::BasicBlock>(instruction->target_block()));
            break;
        }
        case xir::DerivedInstructionTag::RETURN: {
            auto instruction = static_cast<const xir::ReturnInst *>(inst);
            _deallocate_ray_queries(builder, function);
            if (auto value = instruction->return_value()) {
                builder.CreateRet(_value(builder, function, value));
            } else {
                builder.CreateRetVoid();
            }
            break;
        }
        case xir::DerivedInstructionTag::UNREACHABLE: {
            _deallocate_ray_queries(builder, function);
            auto return_type = function.function->getReturnType();
            if (return_type->isVoidTy()) {
                builder.CreateRetVoid();
            } else {
                builder.CreateRet(llvm::PoisonValue::get(return_type));
            }
            break;
        }
        case xir::DerivedInstructionTag::PHI: {
            auto instruction = static_cast<const xir::PhiInst *>(inst);
            function.pending_phi_nodes.emplace_back(instruction);
            result = builder.CreatePHI(_type(inst->type())->reg_type, instruction->incoming_count(), inst->name().value_or(""));
            break;
        }
        case xir::DerivedInstructionTag::ALLOCA: {
            auto instruction = static_cast<const xir::AllocaInst *>(inst);
            if (is_ray_query_type(inst->type())) {
                LUISA_ASSERT(instruction->is_local(),
                             "Metal AIR ray-query objects must be local.");
                auto config = _air_ray_tracing_config(instruction);
                llvm::SmallVector<llvm::Value *, 0u> arguments;
                auto allocation = _air_ray_query_call(
                    builder, "allocate_intersection_query",
                    llvm::PointerType::get(
                        _context, air_address_space_generic),
                    arguments, config);
                allocation->setName(inst->name().value_or("ray.query"));
                function.ray_query_allocations.emplace_back(
                    FunctionContext::RayQueryAllocation{
                        allocation, config});
                result = allocation;
                break;
            }
            auto type = _type(inst->type())->mem_type;
            if (instruction->is_local()) {
                auto allocation = _temporary(function, type, _type_alignment(inst->type()));
                allocation->setName(inst->name().value_or(""));
                result = allocation;
            } else {
                auto global = new llvm::GlobalVariable(
                    _module, type, false, llvm::GlobalValue::InternalLinkage,
                    llvm::PoisonValue::get(type), inst->name().value_or("shared"), nullptr,
                    llvm::GlobalValue::NotThreadLocal, air_address_space_threadgroup, false);
                global->setAlignment(llvm::Align{_type_alignment(inst->type())});
                result = global;
            }
            break;
        }
        case xir::DerivedInstructionTag::LOAD: {
            auto instruction = static_cast<const xir::LoadInst *>(inst);
            result = _load(builder, _value(builder, function, instruction->variable()), inst->type());
            break;
        }
        case xir::DerivedInstructionTag::STORE: {
            auto instruction = static_cast<const xir::StoreInst *>(inst);
            auto value = _value(builder, function, instruction->value());
            if (is_ray_query_type(instruction->value()->type())) {
                LUISA_ASSERT(
                    is_ray_query_type(instruction->variable()->type()) &&
                        _value(builder, function,
                               instruction->variable()) == value,
                    "Invalid Metal AIR ray-query initialization store.");
            } else {
                _store(builder,
                       _value(builder, function,
                              instruction->variable()),
                       value, instruction->value()->type());
            }
            break;
        }
        case xir::DerivedInstructionTag::GEP: result = _translate_gep(builder, function, static_cast<const xir::GEPInst *>(inst)); break;
        case xir::DerivedInstructionTag::ARITHMETIC: result = _translate_arithmetic(builder, function, static_cast<const xir::ArithmeticInst *>(inst)); break;
        case xir::DerivedInstructionTag::CAST: result = _translate_cast(builder, function, static_cast<const xir::CastInst *>(inst)); break;
        case xir::DerivedInstructionTag::CALL: {
            auto instruction = static_cast<const xir::CallInst *>(inst);
            auto callee = instruction->callee();
            auto external = callee->derived_function_tag() ==
                            xir::DerivedFunctionTag::EXTERNAL;
            llvm::SmallVector<llvm::Value *> arguments;
            auto callee_argument = callee->arguments().begin();
            for (auto i = 0u; i < instruction->argument_count(); i++, ++callee_argument) {
                auto argument = _value(builder, function, instruction->argument(i));
                if (callee_argument->is_reference()) {
                    auto generic_pointer = llvm::PointerType::get(_context, air_address_space_generic);
                    if (argument->getType() != generic_pointer) { argument = builder.CreateAddrSpaceCast(argument, generic_pointer); }
                }
                arguments.emplace_back(argument);
                if (!external &&
                    callee_argument->type()->is_texture() &&
                    _texture_needs_sampled_split(*callee_argument)) {
                    arguments.emplace_back(function.sampled_texture(
                        instruction->argument(i)));
                }
            }
            if (!external) { _append_state_arguments(function, arguments); }
            auto llvm_callee = _function(callee);
            auto call = builder.CreateCall(llvm_callee, arguments);
            call->setCallingConv(llvm_callee->getCallingConv());
            if (!external) { call->setConvergent(); }
            if (!call->getType()->isVoidTy()) { result = call; }
            break;
        }
        case xir::DerivedInstructionTag::RESOURCE_QUERY: result = _translate_resource_query(builder, function, static_cast<const xir::ResourceQueryInst *>(inst)); break;
        case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
            result = _translate_ray_query_object_read(
                builder, function,
                static_cast<const xir::RayQueryObjectReadInst *>(inst));
            break;
        case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
            _translate_ray_query_object_write(
                builder, function,
                static_cast<const xir::RayQueryObjectWriteInst *>(inst));
            break;
        case xir::DerivedInstructionTag::RESOURCE_READ: result = _translate_resource_read(builder, function, static_cast<const xir::ResourceReadInst *>(inst)); break;
        case xir::DerivedInstructionTag::RESOURCE_WRITE: _translate_resource_write(builder, function, static_cast<const xir::ResourceWriteInst *>(inst)); break;
        case xir::DerivedInstructionTag::ATOMIC: result = _translate_atomic(builder, function, static_cast<const xir::AtomicInst *>(inst)); break;
        case xir::DerivedInstructionTag::THREAD_GROUP:
            result = _translate_thread_group(builder, function, static_cast<const xir::ThreadGroupInst *>(inst));
            break;
        case xir::DerivedInstructionTag::PRINT:
            _translate_print(builder, function, static_cast<const xir::PrintInst *>(inst));
            break;
        case xir::DerivedInstructionTag::DEBUG_BREAK:
            // Watch operands and the callback belong to debugger-side XIR
            // state. AIR, like the CUDA/HIP LLVM paths, receives only the
            // target trap intrinsic.
            builder.CreateIntrinsic(builder.getVoidTy(), llvm::Intrinsic::debugtrap, {});
            break;
        case xir::DerivedInstructionTag::RASTER_DISCARD: {
            LUISA_ASSERT(
                _config.program == MetalAIRProgram::RASTER_FRAGMENT,
                "Raster discard is only valid in a Metal AIR fragment stage.");
            auto function_type = llvm::FunctionType::get(
                builder.getVoidTy(), {}, false);
            auto callee = _module.getOrInsertFunction(
                "air.discard_fragment", function_type);
            if (auto intrinsic = llvm::dyn_cast<llvm::Function>(callee.getCallee())) {
                intrinsic->setMustProgress();
                intrinsic->setDoesNotThrow();
                intrinsic->setWillReturn();
            }
            builder.CreateCall(callee);
            auto return_type = function.function->getReturnType();
            if (return_type->isVoidTy()) {
                builder.CreateRetVoid();
            } else {
                builder.CreateRet(llvm::PoisonValue::get(return_type));
            }
            break;
        }
        case xir::DerivedInstructionTag::ASSUME: {
            auto instruction = static_cast<const xir::AssumeInst *>(inst);
            builder.CreateAssumption(_value(builder, function, instruction->condition()));
            break;
        }
        case xir::DerivedInstructionTag::ASSERT:
            // Keep parity with the MSL backend, whose lc_assert helper is a
            // no-op until Metal device-side assertion reporting is defined.
            break;
        default: _unsupported_instruction(inst);
    }
    if (result != nullptr) { function.values.try_emplace(inst, result); }
}

}// namespace luisa::compute::metal::detail
