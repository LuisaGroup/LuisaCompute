#include "metal_codegen_llvm_impl.h"

namespace luisa::compute::metal::detail {

namespace {

[[nodiscard]] llvm::Value *vectorize_float3(
    llvm::IRBuilder<> &builder, llvm::Value *value) noexcept {
    if (value->getType()->isVectorTy()) { return value; }
    LUISA_ASSERT(
        value->getType()->isArrayTy() &&
            llvm::cast<llvm::ArrayType>(value->getType())
                    ->getNumElements() == 3u,
        "Invalid Metal AIR ray float3 storage type.");
    auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(
        llvm::FixedVectorType::get(builder.getFloatTy(), 3u)));
    for (auto i = 0u; i < 3u; i++) {
        result = builder.CreateInsertElement(
            result, builder.CreateExtractValue(value, i), i);
    }
    return result;
}

[[nodiscard]] llvm::Value *float3_to_storage(
    llvm::IRBuilder<> &builder, llvm::Value *value,
    llvm::Type *storage_type) noexcept {
    if (storage_type->isVectorTy()) { return value; }
    LUISA_ASSERT(
        storage_type->isArrayTy() &&
            llvm::cast<llvm::ArrayType>(storage_type)
                    ->getNumElements() == 3u,
        "Invalid Metal AIR ray float3 destination type.");
    auto storage = static_cast<llvm::Value *>(
        llvm::PoisonValue::get(storage_type));
    for (auto i = 0u; i < 3u; i++) {
        storage = builder.CreateInsertValue(
            storage, builder.CreateExtractElement(value, i), i);
    }
    return storage;
}

}// namespace

void MetalCodegenLLVMImpl::_collect_ray_query_pipelines(
    const xir::Module &module) noexcept {
    auto pipeline_count = 0u;
    for (auto function : module.function_list()) {
        if (auto definition = function->definition()) {
            definition->traverse_instructions(
                [&pipeline_count](const xir::Instruction *instruction) noexcept {
                    pipeline_count += instruction->isa<xir::RayQueryPipelineInst>();
                });
        }
    }
    _ray_query_pipelines.reserve(pipeline_count);
    for (auto function : module.function_list()) {
        if (auto definition = function->definition()) {
            definition->traverse_instructions(
                [this](const xir::Instruction *instruction) noexcept {
                    if (!instruction->isa<xir::RayQueryPipelineInst>()) { return; }
                    auto pipeline = static_cast<const xir::RayQueryPipelineInst *>(
                        instruction);
                    auto query_object = pipeline->query_object();
                    LUISA_ASSERT(
                        query_object != nullptr &&
                            query_object->isa<xir::AllocaInst>(),
                        "Metal AIR ray-query pipeline must target a local query object.");
                    const xir::ResourceQueryInst *constructor{nullptr};
                    for (auto use : query_object->use_list()) {
                        auto user = use->user();
                        if (user == nullptr || !user->isa<xir::StoreInst>()) {
                            continue;
                        }
                        auto store = static_cast<const xir::StoreInst *>(user);
                        if (store->variable() == query_object &&
                            store->value()->isa<xir::ResourceQueryInst>()) {
                            LUISA_ASSERT(
                                constructor == nullptr,
                                "Metal AIR ray-query pipeline has multiple constructors.");
                            constructor = static_cast<const xir::ResourceQueryInst *>(
                                store->value());
                        }
                    }
                    LUISA_ASSERT(
                        constructor != nullptr,
                        "Metal AIR ray-query pipeline has no constructor.");
                    auto surface = pipeline->on_surface_function();
                    auto procedural = pipeline->on_procedural_function();
                    LUISA_ASSERT(
                        surface != nullptr && procedural != nullptr &&
                            surface->isa<xir::CallableFunction>() &&
                            procedural->isa<xir::CallableFunction>(),
                        "Metal AIR ray-query pipeline handlers must be callables.");
                    auto surface_callable =
                        static_cast<const xir::CallableFunction *>(surface);
                    auto procedural_callable =
                        static_cast<const xir::CallableFunction *>(procedural);
                    auto index = _ray_query_pipelines.size();
                    auto i8 = llvm::Type::getInt8Ty(_context);
                    auto i32 = llvm::Type::getInt32Ty(_context);
                    auto f32 = llvm::Type::getFloatTy(_context);
                    auto i32x3 = llvm::FixedVectorType::get(i32, 3u);
                    auto f32x2 = llvm::FixedVectorType::get(f32, 2u);
                    auto f32x3 = llvm::FixedVectorType::get(f32, 3u);
                    auto payload_field_mask =
                        (1u << ray_query_payload_accept) |
                        (1u << ray_query_payload_continue);
                    auto use_payload_field =
                        [&payload_field_mask](RayQueryPayloadField field) noexcept {
                            payload_field_mask |=
                                1u << static_cast<unsigned>(field);
                        };
                    static_cast<const xir::FunctionDefinition *>(surface_callable)
                        ->traverse_instructions(
                            [&](const xir::Instruction *handler_instruction) noexcept {
                                if (handler_instruction->isa<
                                        xir::RayQueryObjectReadInst>()) {
                                    auto read = static_cast<
                                        const xir::RayQueryObjectReadInst *>(
                                        handler_instruction);
                                    switch (read->op()) {
                                        case xir::RayQueryObjectReadOp::
                                            RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT:
                                            use_payload_field(
                                                ray_query_payload_instance);
                                            use_payload_field(
                                                ray_query_payload_primitive);
                                            use_payload_field(
                                                ray_query_payload_barycentrics);
                                            use_payload_field(
                                                ray_query_payload_distance);
                                            break;
                                        case xir::RayQueryObjectReadOp::
                                            RAY_QUERY_OBJECT_WORLD_SPACE_RAY:
                                            use_payload_field(
                                                ray_query_payload_world_origin);
                                            use_payload_field(
                                                ray_query_payload_world_direction);
                                            use_payload_field(
                                                ray_query_payload_min_distance);
                                            use_payload_field(
                                                ray_query_payload_max_distance);
                                            break;
                                        default: break;
                                    }
                                } else if (handler_instruction->isa<
                                               xir::RayQueryObjectWriteInst>()) {
                                    auto write = static_cast<
                                        const xir::RayQueryObjectWriteInst *>(
                                        handler_instruction);
                                    if (write->op() == xir::RayQueryObjectWriteOp::
                                                           RAY_QUERY_OBJECT_COMMIT_TRIANGLE) {
                                        // Stateful ray queries expose the committed
                                        // candidate distance as the current ray t_max.
                                        use_payload_field(ray_query_payload_distance);
                                    }
                                }
                                for (auto i = 0u;
                                     i < handler_instruction->operand_count(); i++) {
                                    auto operand = handler_instruction->operand(i);
                                    if (operand == nullptr ||
                                        !operand->isa<xir::SpecialRegister>()) {
                                        continue;
                                    }
                                    auto tag = static_cast<
                                                   const xir::SpecialRegister *>(operand)
                                                   ->derived_special_register_tag();
                                    switch (tag) {
                                        case xir::DerivedSpecialRegisterTag::THREAD_ID:
                                            use_payload_field(
                                                ray_query_payload_thread_id);
                                            break;
                                        case xir::DerivedSpecialRegisterTag::BLOCK_ID:
                                            use_payload_field(
                                                ray_query_payload_block_id);
                                            break;
                                        case xir::DerivedSpecialRegisterTag::WARP_LANE_ID:
                                            use_payload_field(
                                                ray_query_payload_warp_lane);
                                            break;
                                        case xir::DerivedSpecialRegisterTag::DISPATCH_ID:
                                            use_payload_field(
                                                ray_query_payload_dispatch_id);
                                            break;
                                        case xir::DerivedSpecialRegisterTag::KERNEL_ID:
                                            use_payload_field(
                                                ray_query_payload_kernel_id);
                                            break;
                                        case xir::DerivedSpecialRegisterTag::BLOCK_SIZE:
                                            use_payload_field(
                                                ray_query_payload_block_size);
                                            break;
                                        case xir::DerivedSpecialRegisterTag::WARP_SIZE:
                                            use_payload_field(
                                                ray_query_payload_warp_size);
                                            break;
                                        case xir::DerivedSpecialRegisterTag::DISPATCH_SIZE:
                                            use_payload_field(
                                                ray_query_payload_dispatch_size);
                                            break;
                                        case xir::DerivedSpecialRegisterTag::
                                            RASTER_OBJECT_ID:
                                        case xir::DerivedSpecialRegisterTag::
                                            RASTER_BARYCENTRICS:
                                        case xir::DerivedSpecialRegisterTag::
                                            RASTER_FRONT_FACING:
                                        case xir::DerivedSpecialRegisterTag::
                                            RASTER_BASE_INSTANCE:
                                            LUISA_ERROR_WITH_LOCATION(
                                                "Raster special register '{}' reached a "
                                                "Metal AIR compute intersection handler.",
                                                xir::to_string(tag));
                                    }
                                }
                            });
                    auto payload_type = llvm::StructType::create(
                        _context, luisa::format("LuisaRayQueryPayload{}", index));
                    llvm::SmallVector<llvm::Type *, 32u> payload_fields{
                        i8, i8, i32, i32, i32, f32x2, f32,
                        f32x3, f32x3, f32, f32,
                        i32x3, i32, i32x3, i32x3, i32x3, i32x3,
                        i32, i32};
                    auto omitted_field = llvm::ArrayType::get(i8, 0u);
                    for (auto i = 0u; i < ray_query_payload_field_count; i++) {
                        if ((payload_field_mask & (1u << i)) == 0u) {
                            payload_fields[i] = omitted_field;
                        }
                    }
                    luisa::vector<RayQueryPayloadCapture> captures;
                    captures.reserve(pipeline->captured_argument_count());
                    for (auto i = 0u;
                         i < pipeline->captured_argument_count(); i++) {
                        auto value = pipeline->captured_argument(i);
                        LUISA_ASSERT(
                            value != nullptr && value->type() != nullptr,
                            "Metal AIR ray-query pipeline has a null capture.");
                        auto payload_index = static_cast<unsigned>(
                            payload_fields.size());
                        payload_fields.emplace_back(
                            _type(value->type())->mem_type);
                        captures.emplace_back(RayQueryPayloadCapture{
                            .value = value,
                            .type = value->type(),
                            .payload_index = payload_index,
                            .reference = value->is_lvalue()});
                    }
                    payload_type->setBody(payload_fields, false);
                    auto motion =
                        constructor->op() == xir::ResourceQueryOp::
                                                 RAY_TRACING_QUERY_ALL_MOTION_BLUR ||
                        constructor->op() == xir::ResourceQueryOp::
                                                 RAY_TRACING_QUERY_ANY_MOTION_BLUR;
                    auto config = _air_ray_tracing_config(query_object, motion);
                    LUISA_ASSERT(
                        !config.curves,
                        "Metal AIR loop-to-IFT currently requires triangle-only "
                        "acceleration structures.");
                    auto function_name = luisa::format(
                        "luisa_ray_query_surface_{}", index);
                    _ray_query_pipelines.emplace_back(RayQueryPipeline{
                        .instruction = pipeline,
                        .query_object =
                            static_cast<const xir::AllocaInst *>(query_object),
                        .constructor = constructor,
                        .surface_handler = surface_callable,
                        .index = index,
                        .config = config,
                        .payload_type = payload_type,
                        .payload_field_mask = payload_field_mask,
                        .captures = std::move(captures),
                        .function_name = function_name});
                    auto &info = _ray_query_pipelines.back();
                    _ray_query_pipeline_indices.try_emplace(pipeline, index);
                    _pipeline_query_objects.insert(query_object);
                    auto [surface_iter, surface_inserted] =
                        _ray_query_pipeline_handlers.try_emplace(
                            surface_callable,
                            RayQueryPipelineHandler{&info, true});
                    LUISA_ASSERT(
                        surface_inserted,
                        "Metal AIR ray-query surface handler is shared by pipelines.");
                    static_cast<void>(surface_iter);
                    auto [procedural_iter, procedural_inserted] =
                        _ray_query_pipeline_handlers.try_emplace(
                            procedural_callable,
                            RayQueryPipelineHandler{&info, false});
                    LUISA_ASSERT(
                        procedural_inserted,
                        "Metal AIR ray-query procedural handler is shared by pipelines.");
                    static_cast<void>(procedural_iter);
                    _result.intersection_functions.emplace_back(
                        std::move(function_name));
                });
        }
    }
}

const MetalCodegenLLVMImpl::RayQueryPipeline &
MetalCodegenLLVMImpl::_ray_query_pipeline(
    const xir::RayQueryPipelineInst *instruction) const noexcept {
    auto iter = _ray_query_pipeline_indices.find(instruction);
    LUISA_ASSERT(
        iter != _ray_query_pipeline_indices.end(),
        "Metal AIR ray-query pipeline was not collected.");
    return _ray_query_pipelines[iter->second];
}

llvm::Value *MetalCodegenLLVMImpl::_pipeline_payload_pointer(
    IB &builder, const FunctionContext &function,
    RayQueryPayloadField field) const noexcept {
    LUISA_ASSERT(
        function.pipeline_handler != nullptr &&
            function.pipeline_payload != nullptr,
        "Metal AIR pipeline payload requested outside an intersection handler.");
    LUISA_ASSERT(
        function.pipeline_handler->pipeline->uses_payload_field(field),
        "Metal AIR intersection handler requested an omitted payload field.");
    return builder.CreateStructGEP(
        function.pipeline_handler->pipeline->payload_type,
        function.pipeline_payload, static_cast<unsigned>(field));
}

void MetalCodegenLLVMImpl::_emit_ray_query_intersection_functions() noexcept {
    for (auto &&pipeline : _ray_query_pipelines) {
        auto i1 = llvm::Type::getInt1Ty(_context);
        auto i32 = llvm::Type::getInt32Ty(_context);
        auto f32 = llvm::Type::getFloatTy(_context);
        auto f32x2 = llvm::FixedVectorType::get(f32, 2u);
        auto payload_pointer = llvm::PointerType::get(
            _context, 5u /* AIR ray-data address space */);
        auto decision_type = llvm::StructType::get(
            _context, {i1, i1}, true);
        auto function_type = llvm::FunctionType::get(
            decision_type,
            {payload_pointer, i32, i32, i32, f32x2, f32}, false);
        auto function = llvm::Function::Create(
            function_type, llvm::GlobalValue::ExternalLinkage,
            pipeline.function_name, _module);
        function->setMustProgress();
        function->setDoesNotFreeMemory();
        function->setDoesNotThrow();
        function->setWillReturn();
        function->addFnAttr("no-builtins");
        function->addFnAttr("frame-pointer", "all");
        _set_float_control_attributes(function);
        _set_air_pointer_element_types(
            function,
            {std::pair<unsigned, llvm::Type *>{
                0u, pipeline.payload_type}});
        auto payload = function->getArg(0u);
        payload->setName("payload");
        payload->addAttr(llvm::Attribute::NoUndef);
        auto payload_size = _data_layout.getTypeAllocSize(
            pipeline.payload_type);
        auto payload_alignment = _data_layout.getABITypeAlign(
            pipeline.payload_type);
        payload->addAttr(llvm::Attribute::getWithAlignment(
            _context, payload_alignment));
        payload->addAttr(llvm::Attribute::getWithDereferenceableBytes(
            _context, payload_size));
        for (auto i = 1u; i < function->arg_size(); i++) {
            function->getArg(i)->addAttr(llvm::Attribute::NoUndef);
        }

        auto entry = llvm::BasicBlock::Create(
            _context, "entry", function);
        IB builder{entry};
        auto field = [&](RayQueryPayloadField index) noexcept {
            return builder.CreateStructGEP(
                pipeline.payload_type, payload,
                static_cast<unsigned>(index));
        };
        auto store = [&](llvm::Value *value,
                         RayQueryPayloadField index) noexcept {
            if (!pipeline.uses_payload_field(index)) { return; }
            auto pointer = field(index);
            auto instruction = builder.CreateStore(value, pointer);
            instruction->setAlignment(
                _data_layout.getABITypeAlign(value->getType()));
        };
        store(builder.getInt8(0u), ray_query_payload_accept);
        store(builder.getInt8(1u), ray_query_payload_continue);
        store(builder.getInt32(1u), ray_query_payload_candidate_kind);
        store(function->getArg(3u), ray_query_payload_instance);
        store(function->getArg(1u), ray_query_payload_primitive);
        store(function->getArg(4u), ray_query_payload_barycentrics);
        store(function->getArg(5u), ray_query_payload_distance);

        auto handler = _function(pipeline.surface_handler);
        llvm::SmallVector<llvm::Value *> arguments;
        arguments.emplace_back(payload);
        for (auto &&capture : pipeline.captures) {
            auto pointer = builder.CreateStructGEP(
                pipeline.payload_type, payload,
                capture.payload_index);
            if (capture.reference) {
                arguments.emplace_back(pointer);
            } else {
                arguments.emplace_back(
                    _load(builder, pointer, capture.type));
            }
        }
        for (auto index : {
                 ray_query_payload_dispatch_size,
                 ray_query_payload_kernel_id,
                 ray_query_payload_thread_id,
                 ray_query_payload_block_id,
                 ray_query_payload_dispatch_id,
                 ray_query_payload_block_size,
                 ray_query_payload_warp_size,
                 ray_query_payload_warp_lane}) {
            if (pipeline.uses_payload_field(index)) {
                auto pointer = field(index);
                auto element_type = pipeline.payload_type->getElementType(
                    static_cast<unsigned>(index));
                auto load = builder.CreateLoad(element_type, pointer);
                load->setAlignment(
                    _data_layout.getABITypeAlign(element_type));
                arguments.emplace_back(load);
            } else {
                arguments.emplace_back(llvm::PoisonValue::get(
                    handler->getFunctionType()->getParamType(
                        arguments.size())));
            }
        }
        auto call = builder.CreateCall(handler, arguments);
        call->setCallingConv(handler->getCallingConv());
        call->setConvergent();

        auto load_flag = [&](RayQueryPayloadField index) noexcept {
            auto load = builder.CreateLoad(
                builder.getInt8Ty(), field(index));
            load->setAlignment(llvm::Align{1u});
            return builder.CreateICmpNE(load, builder.getInt8(0u));
        };
        auto decision = static_cast<llvm::Value *>(
            llvm::PoisonValue::get(decision_type));
        decision = builder.CreateInsertValue(
            decision, load_flag(ray_query_payload_accept), 0u);
        decision = builder.CreateInsertValue(
            decision, load_flag(ray_query_payload_continue), 1u);
        builder.CreateRet(decision);
        _add_ray_query_intersection_metadata(function, pipeline);
    }
}

void MetalCodegenLLVMImpl::_translate_ray_query_pipeline(
    IB &builder, FunctionContext &function,
    const xir::RayQueryPipelineInst *instruction) noexcept {
    auto &&pipeline = _ray_query_pipeline(instruction);
    LUISA_ASSERT(
        pipeline.index < function.intersection_tables.size(),
        "Metal AIR ray-query pipeline has no bound IFT argument.");
    auto payload_alignment = _data_layout.getABITypeAlign(
        pipeline.payload_type);
    auto payload = _temporary(
        function, pipeline.payload_type,
        payload_alignment.value());
    payload->setName(luisa::format(
        "ray.query.payload.{}", pipeline.index));
    auto store = [&](llvm::Value *value,
                     RayQueryPayloadField index) noexcept {
        if (!pipeline.uses_payload_field(index)) { return; }
        auto pointer = builder.CreateStructGEP(
            pipeline.payload_type, payload,
            static_cast<unsigned>(index));
        auto instruction = builder.CreateStore(value, pointer);
        instruction->setAlignment(
            _data_layout.getABITypeAlign(value->getType()));
    };
    store(builder.getInt8(0u), ray_query_payload_accept);
    store(builder.getInt8(1u), ray_query_payload_continue);

    auto constructor = pipeline.constructor;
    auto accel = _value(builder, function, constructor->operand(0u));
    auto ray = _value(builder, function, constructor->operand(1u));
    auto time = pipeline.config.motion ?
                    _value(builder, function, constructor->operand(2u)) :
                    nullptr;
    auto mask = _value(
        builder, function,
        constructor->operand(pipeline.config.motion ? 3u : 2u));
    auto origin = vectorize_float3(
        builder, builder.CreateExtractValue(ray, 0u));
    auto t_min = builder.CreateExtractValue(ray, 1u);
    auto direction = vectorize_float3(
        builder, builder.CreateExtractValue(ray, 2u));
    auto t_max = builder.CreateExtractValue(ray, 3u);
    store(origin, ray_query_payload_world_origin);
    store(direction, ray_query_payload_world_direction);
    store(t_min, ray_query_payload_min_distance);
    store(t_max, ray_query_payload_max_distance);
    store(function.dispatch_size, ray_query_payload_dispatch_size);
    store(function.kernel_id, ray_query_payload_kernel_id);
    store(function.thread_id, ray_query_payload_thread_id);
    store(function.block_id, ray_query_payload_block_id);
    store(function.dispatch_id, ray_query_payload_dispatch_id);
    store(function.block_size, ray_query_payload_block_size);
    store(function.warp_size, ray_query_payload_warp_size);
    store(function.warp_lane_id, ray_query_payload_warp_lane);
    for (auto &&capture : pipeline.captures) {
        auto pointer = builder.CreateStructGEP(
            pipeline.payload_type, payload,
            capture.payload_index);
        auto value = _value(builder, function, capture.value);
        if (capture.reference) {
            value = _load(builder, value, capture.type);
        }
        _store(builder, pointer, value, capture.type);
    }

    auto accept_any =
        constructor->op() == xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY ||
        constructor->op() == xir::ResourceQueryOp::
                                 RAY_TRACING_QUERY_ANY_MOTION_BLUR;
    auto intersection = _air_trace(
        builder, accel, ray, mask, time, pipeline.config,
        accept_any, function.intersection_tables[pipeline.index],
        payload,
        _data_layout.getTypeAllocSize(pipeline.payload_type),
        false,
        // Apple's triangle-data intersector oracle enables both the
        // triangle and bounding-box geometry bits (3), even when the
        // intersection function itself is triangle-only. With the opaque
        // triangle value (1), hardware traversal succeeds but the IFT entry
        // is never invoked.
        true);
    for (auto &&capture : pipeline.captures) {
        if (!capture.reference) { continue; }
        auto pointer = builder.CreateStructGEP(
            pipeline.payload_type, payload,
            capture.payload_index);
        auto value = _load(builder, pointer, capture.type);
        _store(builder,
               _value(builder, function, capture.value),
               value, capture.type);
    }
    function.pipeline_query_results.insert_or_assign(
        pipeline.query_object, intersection);
}

llvm::Value *MetalCodegenLLVMImpl::_translate_pipeline_ray_query_read(
    IB &builder, FunctionContext &function,
    const xir::RayQueryObjectReadInst *instruction) noexcept {
    auto i32 = builder.getInt32Ty();
    auto f32 = builder.getFloatTy();
    auto f32x2 = llvm::FixedVectorType::get(f32, 2u);
    auto load_payload = [&](RayQueryPayloadField index) noexcept {
        auto pointer = _pipeline_payload_pointer(
            builder, function, index);
        auto type = function.pipeline_handler->pipeline->payload_type
                        ->getElementType(static_cast<unsigned>(index));
        auto load = builder.CreateLoad(type, pointer);
        load->setAlignment(_data_layout.getABITypeAlign(type));
        return static_cast<llvm::Value *>(load);
    };
    if (function.pipeline_handler != nullptr) {
        switch (instruction->op()) {
            case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED: {
                auto flag = load_payload(ray_query_payload_continue);
                return builder.CreateICmpEQ(flag, builder.getInt8(0u));
            }
            case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE:
                return builder.getInt1(
                    function.pipeline_handler->surface);
            case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE:
                return builder.getInt1(
                    !function.pipeline_handler->surface);
            case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY: {
                auto result_type = llvm::cast<llvm::StructType>(
                    _type(instruction->type())->reg_type);
                auto result = static_cast<llvm::Value *>(
                    llvm::PoisonValue::get(result_type));
                auto origin = load_payload(ray_query_payload_world_origin);
                auto direction = load_payload(
                    ray_query_payload_world_direction);
                result = builder.CreateInsertValue(
                    result,
                    float3_to_storage(
                        builder, origin,
                        result_type->getElementType(0u)),
                    0u);
                result = builder.CreateInsertValue(
                    result,
                    load_payload(ray_query_payload_min_distance), 1u);
                result = builder.CreateInsertValue(
                    result,
                    float3_to_storage(
                        builder, direction,
                        result_type->getElementType(2u)),
                    2u);
                result = builder.CreateInsertValue(
                    result,
                    load_payload(ray_query_payload_max_distance), 3u);
                return result;
            }
            case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT: {
                auto result = static_cast<llvm::Value *>(
                    llvm::PoisonValue::get(
                        _type(instruction->type())->reg_type));
                result = builder.CreateInsertValue(
                    result, load_payload(ray_query_payload_instance), 0u);
                result = builder.CreateInsertValue(
                    result, load_payload(ray_query_payload_primitive), 1u);
                result = builder.CreateInsertValue(
                    result, load_payload(ray_query_payload_barycentrics), 2u);
                result = builder.CreateInsertValue(
                    result, load_payload(ray_query_payload_distance), 3u);
                return result;
            }
            default:
                LUISA_ERROR_WITH_LOCATION(
                    "Unsupported Metal AIR intersection-handler ray-query read '{}'.",
                    xir::to_string(instruction->op()));
        }
    }

    LUISA_ASSERT(
        instruction->op() ==
            xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT,
        "Only committed-hit reads may consume a completed Metal AIR pipeline.");
    auto iter = function.pipeline_query_results.find(
        instruction->operand(0u));
    LUISA_ASSERT(
        iter != function.pipeline_query_results.end(),
        "Metal AIR pipeline committed hit was read before traversal.");
    auto intersection = iter->second;
    auto intersection_type = builder.CreateExtractValue(
        intersection, 0u);
    auto is_none = builder.CreateICmpEQ(
        intersection_type, builder.getInt32(0u));
    auto is_triangle = builder.CreateICmpEQ(
        intersection_type, builder.getInt32(1u));
    auto kind = builder.CreateSelect(
        is_none, builder.getInt32(0u),
        builder.CreateSelect(
            is_triangle, builder.getInt32(1u), builder.getInt32(2u)));
    auto instance = builder.CreateSelect(
        is_none, builder.getInt32(~0u),
        builder.CreateExtractValue(intersection, 5u));
    auto primitive = builder.CreateExtractValue(intersection, 2u);
    auto barycentrics = builder.CreateSelect(
        is_triangle, builder.CreateExtractValue(intersection, 7u),
        llvm::Constant::getNullValue(f32x2));
    auto distance = builder.CreateExtractValue(intersection, 1u);
    auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(
        _type(instruction->type())->reg_type));
    result = builder.CreateInsertValue(result, instance, 0u);
    result = builder.CreateInsertValue(result, primitive, 1u);
    result = builder.CreateInsertValue(result, barycentrics, 2u);
    result = builder.CreateInsertValue(result, kind, 3u);
    result = builder.CreateInsertValue(result, distance, 4u);
    return result;
}

void MetalCodegenLLVMImpl::_translate_pipeline_ray_query_write(
    IB &builder, FunctionContext &function,
    const xir::RayQueryObjectWriteInst *instruction) noexcept {
    auto store_flag = [&](RayQueryPayloadField field,
                          bool value) noexcept {
        auto pointer = _pipeline_payload_pointer(
            builder, function, field);
        auto store = builder.CreateStore(
            builder.getInt8(value ? 1u : 0u), pointer);
        store->setAlignment(llvm::Align{1u});
    };
    switch (instruction->op()) {
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE:
            LUISA_ASSERT(
                function.pipeline_handler->surface,
                "Triangle commit occurred in a procedural AIR handler.");
            store_flag(ray_query_payload_accept, true);
            if (function.pipeline_handler->pipeline->uses_payload_field(
                    ray_query_payload_max_distance)) {
                auto candidate_pointer = _pipeline_payload_pointer(
                    builder, function, ray_query_payload_distance);
                auto candidate_distance = builder.CreateLoad(
                    builder.getFloatTy(), candidate_pointer);
                candidate_distance->setAlignment(
                    _data_layout.getABITypeAlign(builder.getFloatTy()));
                auto max_distance_pointer = _pipeline_payload_pointer(
                    builder, function, ray_query_payload_max_distance);
                auto store = builder.CreateStore(
                    candidate_distance, max_distance_pointer);
                store->setAlignment(
                    _data_layout.getABITypeAlign(builder.getFloatTy()));
            }
            return;
        case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE:
            store_flag(ray_query_payload_continue, false);
            return;
        default:
            LUISA_ERROR_WITH_LOCATION(
                "Unsupported Metal AIR intersection-handler ray-query write '{}'.",
                xir::to_string(instruction->op()));
    }
}

void MetalCodegenLLVMImpl::_add_ray_query_intersection_metadata(
    llvm::Function *function,
    const RayQueryPipeline &pipeline) noexcept {
    auto node = [this](llvm::ArrayRef<llvm::Metadata *> operands) noexcept {
        return llvm::MDNode::get(_context, operands);
    };
    auto layout = _data_layout.getStructLayout(pipeline.payload_type);
    static constexpr std::array<luisa::string_view,
                                ray_query_payload_field_count>
        type_names{
            "uchar", "uchar", "uint", "uint", "uint", "float2",
            "float", "float3", "float3", "float", "float",
            "uint3", "uint", "uint3", "uint3", "uint3", "uint3",
            "uint", "uint"};
    static constexpr std::array<luisa::string_view,
                                ray_query_payload_field_count>
        field_names{
            "accept", "continue_search", "candidate_kind", "instance_id",
            "primitive_id", "barycentrics", "distance", "world_origin",
            "world_direction", "min_distance", "max_distance",
            "dispatch_size", "kernel_id", "thread_id", "block_id",
            "dispatch_id", "block_size", "warp_size", "warp_lane"};
    llvm::SmallVector<llvm::Metadata *> payload_fields;
    for (auto i = 0u; i < ray_query_payload_field_count; i++) {
        if (!pipeline.uses_payload_field(
                static_cast<RayQueryPayloadField>(i))) {
            continue;
        }
        auto type = pipeline.payload_type->getElementType(i);
        payload_fields.append({md_i32(_context, static_cast<uint32_t>(
                                                    layout->getElementOffset(i))),
                               md_i32(_context, static_cast<uint32_t>(
                                                    _data_layout.getTypeAllocSize(type))),
                               md_i32(_context, 0u), md_string(_context, type_names[i]),
                               md_string(_context, field_names[i])});
    }
    auto append_capture_field =
        [&](const RayQueryPayloadCapture &capture,
            size_t capture_index) noexcept {
            auto offset = static_cast<uint32_t>(
                layout->getElementOffset(capture.payload_index));
            auto size = static_cast<uint32_t>(
                _data_layout.getTypeAllocSize(
                    pipeline.payload_type->getElementType(
                        capture.payload_index)));
            auto name = luisa::format("capture.{}", capture_index);
            if (capture.type->is_buffer()) {
                auto element = capture.type->element();
                auto element_name = element == nullptr ?
                                        luisa::string{"uchar"} :
                                        _air_type_name(element);
                auto buffer_info = node({md_i32(_context, 0u), md_i32(_context, 8u),
                                         md_i32(_context, 0u), md_string(_context, element_name),
                                         md_string(_context, "data"), md_i32(_context, 8u),
                                         md_i32(_context, 8u), md_i32(_context, 0u),
                                         md_string(_context, "ulong"),
                                         md_string(_context, "size")});
                payload_fields.append({md_string(_context, "air.struct_type_info"), buffer_info,
                                       md_i32(_context, offset), md_i32(_context, size),
                                       md_i32(_context, 0u),
                                       md_string(_context,
                                                 luisa::format("LCBuffer.{}", element_name)),
                                       md_string(_context, name)});
                return;
            }
            if (capture.type->is_bindless_array()) {
                auto bindless_info = node({md_i32(_context, 0u), md_i32(_context, 8u),
                                           md_i32(_context, 0u),
                                           md_string(_context, "LCBindlessItem"),
                                           md_string(_context, "items")});
                payload_fields.append({md_string(_context, "air.struct_type_info"), bindless_info,
                                       md_i32(_context, offset), md_i32(_context, size),
                                       md_i32(_context, 0u),
                                       md_string(_context, "LCBindlessArray"),
                                       md_string(_context, name)});
                return;
            }
            auto base = capture.type;
            auto array_count = 0u;
            while (base->is_array()) {
                array_count = array_count == 0u ?
                                  base->dimension() :
                                  array_count * base->dimension();
                base = base->element();
            }
            if (base->is_structure()) {
                payload_fields.append({md_string(_context, "air.struct_type_info"),
                                       _air_struct_type_info(base)});
            }
            payload_fields.append({md_i32(_context, offset),
                                   md_i32(_context, static_cast<uint32_t>(base->size())),
                                   md_i32(_context, array_count),
                                   md_string(_context, _air_type_name(base)),
                                   md_string(_context, name)});
        };
    for (auto i = 0u; i < pipeline.captures.size(); i++) {
        append_capture_field(pipeline.captures[i], i);
    }
    auto payload_info = node({md_i32(_context, 0u), md_string(_context, "air.payload"),
                              md_string(_context, "air.struct_type_info"), node(payload_fields),
                              md_string(_context, "air.arg_type_size"),
                              md_i32(_context, static_cast<uint32_t>(
                                                   _data_layout.getTypeAllocSize(
                                                       pipeline.payload_type))),
                              md_string(_context, "air.arg_type_align_size"),
                              md_i32(_context, static_cast<uint32_t>(
                                                   _data_layout.getABITypeAlign(
                                                                   pipeline.payload_type)
                                                       .value())),
                              md_string(_context, "air.arg_type_name"),
                              md_string(_context, pipeline.payload_type->getName()),
                              md_string(_context, "air.arg_name"), md_string(_context, "payload")});
    auto argument = [&](uint32_t index, luisa::string_view semantic,
                        luisa::string_view type,
                        luisa::string_view name) noexcept {
        return node({md_i32(_context, index), md_string(_context, semantic),
                     md_string(_context, "air.arg_type_name"),
                     md_string(_context, type),
                     md_string(_context, "air.arg_name"),
                     md_string(_context, name)});
    };
    auto return_info = node({node({md_string(_context, "air.accept_intersection"),
                                   md_string(_context, "air.arg_type_name"),
                                   md_string(_context, "bool"),
                                   md_string(_context, "air.arg_name"),
                                   md_string(_context, "accept")}),
                             node({md_string(_context, "air.continue_search"),
                                   md_string(_context, "air.arg_type_name"),
                                   md_string(_context, "bool"),
                                   md_string(_context, "air.arg_name"),
                                   md_string(_context, "continue_search")})});
    auto argument_info = node({payload_info,
                               argument(1u, "air.primitive_id", "uint", "primitive_id"),
                               argument(2u, "air.geometry_id", "uint", "geometry_id"),
                               argument(3u, "air.instance_id", "uint", "instance_id"),
                               argument(4u, "air.barycentric_coord", "float2", "barycentrics"),
                               argument(5u, "air.distance", "float", "distance")});
    auto info = node({llvm::ValueAsMetadata::get(function), return_info,
                      argument_info, md_string(_context, "air.triangle"),
                      md_string(_context, "air.instancing"),
                      md_string(_context, "air.triangle_data")});
    _module.getOrInsertNamedMetadata("air.intersection")->addOperand(info);
}

}// namespace luisa::compute::metal::detail
