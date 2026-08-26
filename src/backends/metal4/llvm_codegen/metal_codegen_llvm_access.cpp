#include "metal_codegen_llvm_impl.h"

namespace luisa::compute::metal::detail {

std::pair<llvm::Value *, const Type *> MetalCodegenLLVMImpl::_access_chain(
    IB &builder, FunctionContext &function, llvm::Value *pointer,
    const Type *type, luisa::span<const xir::Use *const> indices) noexcept {
    LUISA_ASSERT(pointer->getType()->isPointerTy(), "XIR access chain base is not a pointer.");
    for (auto index_use : indices) {
        auto index = _value(builder, function, index_use->value());
        switch (type->tag()) {
            case Type::Tag::VECTOR: {
                type = type->element();
                pointer = builder.CreateInBoundsGEP(_type(type)->mem_type, pointer, index);
                break;
            }
            case Type::Tag::MATRIX: {
                type = Type::vector(type->element(), type->dimension());
                pointer = builder.CreateInBoundsGEP(_type(type)->mem_type, pointer, index);
                break;
            }
            case Type::Tag::ARRAY: [[fallthrough]];
            case Type::Tag::COOPERATIVE_VECTOR: {
                type = type->element();
                pointer = builder.CreateInBoundsGEP(_type(type)->mem_type, pointer, index);
                break;
            }
            case Type::Tag::STRUCTURE: {
                LUISA_ASSERT(llvm::isa<llvm::ConstantInt>(index), "XIR structure index must be constant.");
                auto member_index = llvm::cast<llvm::ConstantInt>(index)->getZExtValue();
                LUISA_ASSERT(member_index < type->members().size(), "XIR structure index is out of range.");
                auto offset = _type(type)->member_offsets[member_index];
                pointer = builder.CreateConstInBoundsGEP1_64(builder.getInt8Ty(), pointer, offset);
                type = type->members()[member_index];
                break;
            }
            default: LUISA_ERROR_WITH_LOCATION("Invalid XIR access chain base type '{}'.", type->description());
        }
    }
    return {pointer, type};
}

llvm::Value *MetalCodegenLLVMImpl::_translate_gep(IB &builder, FunctionContext &function, const xir::GEPInst *inst) noexcept {
    auto [pointer, type] = _access_chain(builder, function, _value(builder, function, inst->base()),
                                         inst->base()->type(), inst->index_uses());
    LUISA_ASSERT(type == inst->type(), "XIR GEP result type mismatch.");
    return pointer;
}

llvm::Value *MetalCodegenLLVMImpl::_static_cast(IB &builder, llvm::Value *value, const Type *source, const Type *target) noexcept {
    if (source == target) { return value; }
    auto source_element = source->is_vector() ? source->element() : source;
    auto target_element = target->is_vector() ? target->element() : target;
    LUISA_ASSERT((source->is_scalar() || source->is_vector()) && (target->is_scalar() || target->is_vector()),
                 "Metal LLVM static casts require scalar or vector operands.");
    if (source->is_vector() && target->is_vector()) {
        LUISA_ASSERT(source->dimension() == target->dimension(), "Vector cast dimension mismatch.");
    }
    auto llvm_target = _type(target)->reg_type;
    if (source->is_scalar() && target->is_vector()) {
        auto scalar_target = _type(target_element)->reg_type;
        value = _static_cast(builder, value, source, target_element);
        return builder.CreateVectorSplat(target->dimension(), value);
    }
    if (target_element->is_bool()) {
        return builder.CreateICmpNE(value, llvm::Constant::getNullValue(value->getType()));
    }
    if (source_element->is_bool()) {
        auto zero = llvm::Constant::getNullValue(llvm_target);
        auto one = target_element->is_float() ? llvm::ConstantFP::get(llvm_target, 1.0) : llvm::ConstantInt::get(llvm_target, 1u);
        return builder.CreateSelect(value, one, zero);
    }
    auto source_is_float = source_element->is_float();
    auto target_is_float = target_element->is_float();
    auto source_is_signed = source_element->is_int();
    if (source_is_float && target_is_float) { return builder.CreateFPCast(value, llvm_target); }
    if (source_is_float) { return target_element->is_int() ? builder.CreateFPToSI(value, llvm_target) : builder.CreateFPToUI(value, llvm_target); }
    if (target_is_float) { return source_is_signed ? builder.CreateSIToFP(value, llvm_target) : builder.CreateUIToFP(value, llvm_target); }
    return builder.CreateIntCast(value, llvm_target, source_is_signed);
}

llvm::Value *MetalCodegenLLVMImpl::_bitwise_cast(IB &builder, FunctionContext &function, llvm::Value *value,
                                                 const Type *source, const Type *target) noexcept {
    if (source == target) { return value; }
    if ((source->is_scalar() || source->is_vector()) && (target->is_scalar() || target->is_vector()) &&
        !source->is_bool_or_bool_vector() && !target->is_bool_or_bool_vector()) {
        return builder.CreateBitCast(value, _type(target)->reg_type);
    }
    auto memory = _reg_to_mem(builder, value, source);
    auto temporary = _temporary(
        function, memory->getType(),
        std::max(source->alignment(), target->alignment()));
    builder.CreateAlignedStore(memory, temporary, llvm::Align{source->alignment()});
    auto target_memory = builder.CreateAlignedLoad(_type(target)->mem_type, temporary, llvm::Align{target->alignment()});
    return _mem_to_reg(builder, target_memory, target);
}

llvm::Value *MetalCodegenLLVMImpl::_translate_cast(IB &builder, FunctionContext &function, const xir::CastInst *inst) noexcept {
    auto source = inst->value()->type();
    auto target = inst->type();
    auto value = _value(builder, function, inst->value());
    switch (inst->op()) {
        case xir::CastOp::STATIC_CAST: return _static_cast(builder, value, source, target);
        case xir::CastOp::BITWISE_CAST: return _bitwise_cast(builder, function, value, source, target);
    }
    LUISA_ERROR_WITH_LOCATION("Invalid XIR cast operation.");
}

llvm::Value *MetalCodegenLLVMImpl::_buffer_pointer(IB &builder, llvm::Value *buffer, llvm::Value *index, size_t stride) noexcept {
    auto pointer = builder.CreateExtractValue(buffer, 0u);
    auto index64 = builder.CreateZExtOrTrunc(index, builder.getInt64Ty());
    auto offset = stride == 1u ? index64 : builder.CreateMul(index64, builder.getInt64(stride));
    return builder.CreateInBoundsGEP(builder.getInt8Ty(), pointer, offset);
}

llvm::Value *MetalCodegenLLVMImpl::_bindless_slot(
    IB &builder, llvm::Value *array, llvm::Value *index) noexcept {
    LUISA_ASSERT(array->getType() == _bindless_array(),
                 "Invalid Metal bindless-array LLVM value.");
    auto items = builder.CreateExtractValue(array, 0u);
    index = builder.CreateZExtOrTrunc(index, builder.getInt64Ty());
    return builder.CreateInBoundsGEP(_bindless_item(), items, index);
}

llvm::Value *MetalCodegenLLVMImpl::_bindless_slot_field(
    IB &builder, llvm::Value *slot, unsigned field) noexcept {
    auto type = _bindless_item()->getElementType(field);
    auto pointer = builder.CreateStructGEP(_bindless_item(), slot, field);
    auto alignment = field == 0u || field == 2u ? 16u : 8u;
    return builder.CreateAlignedLoad(type, pointer, llvm::Align{alignment});
}

llvm::Value *MetalCodegenLLVMImpl::_bindless_buffer_size(
    IB &builder, llvm::Value *slot) noexcept {
    auto packed = _bindless_slot_field(builder, slot, 1u);
    return builder.CreateAnd(packed, builder.getInt64(0x0000ffffffffffffull));
}

llvm::Value *MetalCodegenLLVMImpl::_bindless_texture(
    IB &builder, llvm::Value *slot, unsigned dimension) noexcept {
    LUISA_ASSERT(dimension == 2u || dimension == 3u,
                 "Metal bindless textures must be two- or three-dimensional.");
    auto wrapper = _bindless_slot_field(builder, slot, dimension);
    return builder.CreateExtractValue(wrapper, 0u);
}

llvm::Value *MetalCodegenLLVMImpl::_bindless_sampler_code(
    IB &builder, llvm::Value *slot, unsigned dimension) noexcept {
    LUISA_ASSERT(dimension == 2u || dimension == 3u,
                 "Metal bindless samplers must be two- or three-dimensional.");
    auto packed = _bindless_slot_field(builder, slot, 1u);
    auto shift = dimension == 2u ? 48u : 56u;
    return builder.CreateTrunc(builder.CreateLShr(packed, shift), builder.getInt32Ty());
}

llvm::Value *MetalCodegenLLVMImpl::_device_pointer_offset(
    IB &builder, llvm::Value *pointer, llvm::Value *offset, size_t stride) noexcept {
    offset = builder.CreateZExtOrTrunc(offset, builder.getInt64Ty());
    if (stride != 1u) { offset = builder.CreateMul(offset, builder.getInt64(stride)); }
    return builder.CreateInBoundsGEP(builder.getInt8Ty(), pointer, offset);
}

llvm::Value *MetalCodegenLLVMImpl::_accel_instance_pointer(
    IB &builder, llvm::Value *accel, llvm::Value *index) noexcept {
    LUISA_ASSERT(accel->getType() == _accel(),
                 "Invalid Metal acceleration-structure LLVM value.");
    auto instances = builder.CreateExtractValue(accel, 1u);
    index = builder.CreateZExtOrTrunc(index, builder.getInt64Ty());
    return builder.CreateInBoundsGEP(_accel_instance(), instances, index);
}

AIRRayTracingConfig MetalCodegenLLVMImpl::_air_ray_tracing_config(
    const xir::Value *value, bool motion) const noexcept {
    AIRRayTracingConfig config{.motion = motion};
    const xir::Instruction *source = nullptr;
    if (value != nullptr && value->isa<xir::Instruction>()) {
        source = static_cast<const xir::Instruction *>(value);
    }
    if (value != nullptr && value->isa<xir::AllocaInst>()) {
        for (auto use : value->use_list()) {
            auto user = use->user();
            if (user == nullptr || !user->isa<xir::StoreInst>()) {
                continue;
            }
            auto store = static_cast<const xir::StoreInst *>(user);
            if (store->variable() == value &&
                store->value()->isa<xir::ResourceQueryInst>()) {
                source = static_cast<const xir::ResourceQueryInst *>(
                    store->value());
                break;
            }
        }
    }
    if (source == nullptr) { return config; }
    auto metadata = source->find_metadata<xir::CurveBasisMD>();
    if (metadata == nullptr || metadata->curve_basis_set().none()) {
        return config;
    }
    auto bases = metadata->curve_basis_set();
    config.curves = true;
    if (bases.count() != 1u) {
        config.curve_basis = ~0u;
        return config;
    }
    if (bases.test(CurveBasis::PIECEWISE_LINEAR)) {
        config.curve_basis = 2u;
        config.curve_control_point_count = 2u;
    } else if (bases.test(CurveBasis::CUBIC_BSPLINE)) {
        config.curve_basis = 0u;
        config.curve_control_point_count = 4u;
    } else if (bases.test(CurveBasis::CATMULL_ROM)) {
        config.curve_basis = 1u;
        config.curve_control_point_count = 4u;
    } else {
        LUISA_ASSERT(bases.test(CurveBasis::BEZIER),
                     "Invalid Metal AIR curve-basis metadata.");
        config.curve_basis = 3u;
        config.curve_control_point_count = 4u;
    }
    return config;
}

llvm::Value *MetalCodegenLLVMImpl::_air_trace(
    IB &builder, llvm::Value *accel, llvm::Value *ray,
    llvm::Value *mask, llvm::Value *time,
    AIRRayTracingConfig config, bool accept_any) noexcept {
    LUISA_ASSERT(accel->getType() == _accel() && ray->getType()->isStructTy(),
                 "Invalid Metal AIR trace operands.");
    LUISA_ASSERT(config.motion == (time != nullptr),
                 "Metal AIR motion trace requires exactly one time operand.");
    auto handle_wrapper = builder.CreateExtractValue(accel, 0u);
    auto handle = builder.CreateExtractValue(handle_wrapper, 0u);
    auto origin = builder.CreateExtractValue(ray, 0u);
    auto t_min = builder.CreateExtractValue(ray, 1u);
    auto direction = builder.CreateExtractValue(ray, 2u);
    auto t_max = builder.CreateExtractValue(ray, 3u);
    auto vectorize_float3 = [&builder](llvm::Value *value) noexcept {
        if (value->getType()->isVectorTy()) { return value; }
        LUISA_ASSERT(value->getType()->isArrayTy() &&
                         llvm::cast<llvm::ArrayType>(value->getType())->getNumElements() == 3u,
                     "Invalid Metal AIR ray vector storage type.");
        auto result = static_cast<llvm::Value *>(
            llvm::PoisonValue::get(
                llvm::FixedVectorType::get(builder.getFloatTy(), 3u)));
        for (auto i = 0u; i < 3u; i++) {
            result = builder.CreateInsertElement(
                result, builder.CreateExtractValue(value, i), i);
        }
        return result;
    };
    origin = vectorize_float3(origin);
    direction = vectorize_float3(direction);
    mask = builder.CreateZExtOrTrunc(mask, builder.getInt32Ty());

    auto ift_pointer_type = llvm::PointerType::get(
        _context, air_address_space_device);
    auto null_ift_type = llvm::FunctionType::get(
        ift_pointer_type, {}, false);
    auto null_ift = _module.getOrInsertFunction(
        "air.get_null_intersection_function_table", null_ift_type);
    if (auto f = llvm::dyn_cast<llvm::Function>(null_ift.getCallee())) {
        _set_air_pointer_element_types(
            f, {}, _air_intersection_function_table());
        f->setMustProgress();
        f->setDoesNotFreeMemory();
        f->setDoesNotThrow();
        f->setWillReturn();
        f->setOnlyReadsMemory();
        f->setOnlyAccessesInaccessibleMemory();
    }
    auto ift = builder.CreateCall(null_ift);

    auto payload_pointer_type = llvm::PointerType::get(
        _context, air_address_space_generic);
    auto payload = llvm::ConstantPointerNull::get(
        llvm::cast<llvm::PointerType>(payload_pointer_type));
    llvm::SmallVector<llvm::Value *, 21u> arguments{
        origin, direction, t_min, t_max, handle, mask};
    if (config.motion) {
        LUISA_ASSERT(time->getType()->isFloatTy(),
                     "Metal AIR motion time must be float.");
        arguments.emplace_back(time);
    }
    auto ift_index = static_cast<unsigned>(arguments.size());
    arguments.emplace_back(ift);
    auto payload_index = static_cast<unsigned>(arguments.size());
    arguments.append({
        payload, builder.getInt64(0u),
        builder.getInt32(0u),// winding
        builder.getInt32(0u),// triangle culling
        builder.getInt32(0u),// geometry culling
        builder.getInt32(0u),// opacity culling
        builder.getInt32(1u),// force opaque
        builder.getInt32(config.geometry_type()),
        builder.getInt32(config.curve_basis),
        builder.getInt32(0u),// round curves
        builder.getInt32(config.curve_control_point_count),
        builder.getInt1(false),// do not assume identity transforms
        builder.getInt1(accept_any)});
    llvm::SmallVector<llvm::Type *, 21u> parameter_types;
    parameter_types.reserve(arguments.size());
    for (auto argument : arguments) {
        parameter_types.emplace_back(argument->getType());
    }
    auto function_type = llvm::FunctionType::get(
        _air_intersection_result(config.curves), parameter_types, false);
    std::string name{"air.intersect"};
    auto suffix = config.intrinsic_suffix();
    name.append(suffix.data(), suffix.size());
    auto callee = _module.getOrInsertFunction(name, function_type);
    if (auto f = llvm::dyn_cast<llvm::Function>(callee.getCallee())) {
        llvm::SmallVector<std::pair<unsigned, llvm::Type *>, 3u> pointer_types{
            std::pair<unsigned, llvm::Type *>{4u, _air_accel_handle()},
            std::pair<unsigned, llvm::Type *>{
                ift_index, _air_intersection_function_table()},
            std::pair<unsigned, llvm::Type *>{
                payload_index, builder.getInt8Ty()}};
        _set_air_pointer_element_types(f, pointer_types);
        f->setMustProgress();
        f->setDoesNotThrow();
        f->setWillReturn();
    }
    auto call = builder.CreateCall(callee, arguments);
    call->setDoesNotThrow();
    call->addFnAttr(llvm::Attribute::WillReturn);
    return call;
}

llvm::CallInst *MetalCodegenLLVMImpl::_air_ray_query_call(
    IB &builder, luisa::string_view operation,
    llvm::Type *return_type,
    llvm::ArrayRef<llvm::Value *> arguments,
    AIRRayTracingConfig config,
    bool read_only,
    llvm::ArrayRef<std::pair<unsigned, llvm::Type *>> extra_pointer_types) noexcept {
    std::string name{"air."};
    name.append(operation.data(), operation.size());
    auto suffix = config.intrinsic_suffix();
    name.append(suffix.data(), suffix.size());
    llvm::SmallVector<llvm::Type *> parameter_types;
    parameter_types.reserve(arguments.size());
    for (auto argument : arguments) {
        parameter_types.emplace_back(argument->getType());
    }
    auto function_type = llvm::FunctionType::get(
        return_type, parameter_types, false);
    auto callee = _module.getOrInsertFunction(name, function_type);
    if (auto f = llvm::dyn_cast<llvm::Function>(callee.getCallee())) {
        llvm::SmallVector<std::pair<unsigned, llvm::Type *>> pointer_types;
        if (!arguments.empty()) {
            LUISA_ASSERT(arguments.front()->getType()->isPointerTy(),
                         "Metal AIR ray-query intrinsic first argument must be the query pointer.");
            pointer_types.emplace_back(0u, _air_intersection_query());
        }
        pointer_types.append(extra_pointer_types.begin(),
                             extra_pointer_types.end());
        auto returns_query = arguments.empty() && return_type->isPointerTy();
        _set_air_pointer_element_types(
            f, pointer_types,
            returns_query ? _air_intersection_query() : nullptr);
        f->setMustProgress();
        f->setDoesNotThrow();
        f->setWillReturn();
        for (auto [index, _] : extra_pointer_types) {
            f->getArg(index)->addAttr(llvm::Attribute::ReadOnly);
        }
        if (read_only) {
            f->setDoesNotFreeMemory();
            f->setOnlyReadsMemory();
            f->setOnlyAccessesArgMemory();
            auto query_argument = f->getArg(0u);
            query_argument->addAttr(
                llvm::Attribute::getWithCaptureInfo(
                    _context, llvm::CaptureInfo::none()));
            query_argument->addAttr(llvm::Attribute::ReadOnly);
        }
    }
    auto call = builder.CreateCall(callee, arguments);
    call->setDoesNotThrow();
    call->addFnAttr(llvm::Attribute::WillReturn);
    for (auto [index, _] : extra_pointer_types) {
        call->addParamAttr(index, llvm::Attribute::ReadOnly);
    }
    if (read_only) { call->setOnlyReadsMemory(); }
    return call;
}

void MetalCodegenLLVMImpl::_deallocate_ray_queries(
    IB &builder, const FunctionContext &function) noexcept {
    for (auto iter = function.ray_query_allocations.rbegin();
         iter != function.ray_query_allocations.rend(); iter++) {
        llvm::SmallVector<llvm::Value *, 1u> arguments{iter->value};
        static_cast<void>(_air_ray_query_call(
            builder, "deallocate_intersection_query",
            builder.getVoidTy(), arguments, iter->config));
    }
}

uint32_t MetalCodegenLLVMImpl::_texture_access(const xir::Value *texture) const noexcept {
    LUISA_ASSERT(texture != nullptr && texture->type() != nullptr && texture->type()->is_texture(),
                 "Texture access requested for a non-texture XIR value.");
    auto access = 0u;
    auto sampled = false;
    for (auto use : texture->use_list()) {
        auto user = use->user();
        if (user == nullptr) { continue; }
        if (user->isa<xir::ResourceWriteInst>()) {
            access |= air_texture_access_write;
        } else if (user->isa<xir::ResourceReadInst>()) {
            access |= air_texture_access_read;
        } else if (user->isa<xir::ResourceQueryInst>()) {
            auto query = static_cast<const xir::ResourceQueryInst *>(user);
            if (is_direct_texture_sample(query->op())) {
                sampled = true;
            }
        } else {
            // Resource values are expected to be direct operands after the
            // Metal inline-all pass. Stay conservative if that invariant is
            // ever relaxed so intrinsic access modes still match metadata.
            return air_texture_access_read_write;
        }
    }
    if (access != 0u) { return access; }
    return sampled ? air_texture_access_sample : air_texture_access_read;
}

bool MetalCodegenLLVMImpl::_texture_needs_sampled_split(
    const xir::Value *texture) const noexcept {
    LUISA_ASSERT(
        texture != nullptr && texture->type() != nullptr &&
            texture->type()->is_texture(),
        "Texture split requested for a non-texture XIR value.");
    auto sampled = false;
    auto read_or_written = false;
    for (auto use : texture->use_list()) {
        auto user = use->user();
        if (user == nullptr) { continue; }
        if (user->isa<xir::ResourceReadInst>() ||
            user->isa<xir::ResourceWriteInst>()) {
            read_or_written = true;
        } else if (user->isa<xir::ResourceQueryInst>()) {
            auto query = static_cast<const xir::ResourceQueryInst *>(user);
            sampled |= is_direct_texture_sample(query->op());
        }
    }
    return sampled && read_or_written;
}

}// namespace luisa::compute::metal::detail
