#include "metal_codegen_llvm_impl.h"

namespace luisa::compute::metal::detail {

llvm::Value *MetalCodegenLLVMImpl::_translate_atomic(
    IB &builder, FunctionContext &function, const xir::AtomicInst *inst) noexcept {
    auto indices = inst->index_uses();
    auto base = inst->base();
    auto base_type = base->type();
    auto pointer = _value(builder, function, base);
    if (base_type->is_buffer()) {
        LUISA_ASSERT(!indices.empty(), "Metal AIR buffer atomic requires an element index.");
        auto element_type = base_type->element();
        pointer = _buffer_pointer(
            builder, pointer, _value(builder, function, indices.front()->value()),
            element_type->size());
        base_type = element_type;
        indices = indices.subspan(1u);
    }
    auto [element_pointer, element_type] =
        _access_chain(builder, function, pointer, base_type, indices);
    LUISA_ASSERT(element_type == inst->type() && element_type->is_scalar(),
                 "Metal AIR atomic result type mismatch.");
    auto address_space = llvm::cast<llvm::PointerType>(element_pointer->getType())->getAddressSpace();
    LUISA_ASSERT(address_space == air_address_space_device ||
                     address_space == air_address_space_threadgroup,
                 "Metal AIR atomics require device or threadgroup memory.");
    auto is_threadgroup = address_space == air_address_space_threadgroup;
    auto scope = builder.getInt32(is_threadgroup ? 1u : 2u);
    auto prefix = std::string{"air.atomic."};
    prefix.append(is_threadgroup ? "local." : "global.");

    auto intrinsic = [&](std::string name, llvm::Type *return_type,
                         llvm::ArrayRef<llvm::Value *> arguments) noexcept {
        llvm::SmallVector<llvm::Type *> parameter_types;
        parameter_types.reserve(arguments.size());
        for (auto argument : arguments) { parameter_types.emplace_back(argument->getType()); }
        auto function_type = llvm::FunctionType::get(return_type, parameter_types, false);
        auto callee = _module.getOrInsertFunction(name, function_type);
        if (auto intrinsic_function = llvm::dyn_cast<llvm::Function>(callee.getCallee())) {
            intrinsic_function->setMustProgress();
            intrinsic_function->setDoesNotThrow();
            intrinsic_function->setWillReturn();
        }
        return builder.CreateCall(callee, arguments);
    };
    auto atomic_load_i32 = [&]() noexcept {
        llvm::SmallVector<llvm::Value *> arguments{
            element_pointer, builder.getInt32(0u), scope, builder.getInt1(true)};
        return intrinsic(prefix + "load.i32", builder.getInt32Ty(), arguments);
    };
    auto atomic_rmw_i32 = [&](llvm::StringRef operation, llvm::Value *value,
                              bool signed_operation, bool signed_suffix) noexcept {
        auto name = prefix;
        name.append(operation.data(), operation.size());
        if (signed_suffix) { name.append(signed_operation ? ".s" : ".u"); }
        name.append(".i32");
        llvm::SmallVector<llvm::Value *> arguments{
            element_pointer, value, builder.getInt32(0u), scope, builder.getInt1(true)};
        return intrinsic(std::move(name), builder.getInt32Ty(), arguments);
    };
    auto atomic_rmw_f32 = [&](llvm::StringRef operation, llvm::Value *value) noexcept {
        auto name = prefix;
        name.append(operation.data(), operation.size());
        name.append(".f32");
        llvm::SmallVector<llvm::Value *> arguments{
            element_pointer, value, builder.getInt32(0u), scope, builder.getInt1(true)};
        return intrinsic(std::move(name), builder.getFloatTy(), arguments);
    };
    auto atomic_compare_exchange = [&](llvm::Value *expected, llvm::Value *desired,
                                       bool floating_point) noexcept {
        auto value_type = expected->getType();
        auto expected_pointer = _temporary(function, value_type, 4u);
        builder.CreateAlignedStore(expected, expected_pointer, llvm::Align{4u});
        auto name = prefix + "cmpxchg.weak." + (floating_point ? "f32" : "i32");
        llvm::SmallVector<llvm::Value *> arguments{
            element_pointer, expected_pointer, desired,
            builder.getInt32(0u), builder.getInt32(0u), scope, builder.getInt1(true)};
        return intrinsic(std::move(name), value_type, arguments);
    };

    llvm::SmallVector<llvm::Value *, 2u> values;
    for (auto value_use : inst->value_uses()) {
        values.emplace_back(_value(builder, function, value_use->value()));
    }
    if (element_type->is_int() || element_type->is_uint()) {
        LUISA_ASSERT(element_type->size() == 4u, "Metal AIR only supports 32-bit integer atomics.");
        auto is_signed = element_type->is_int();
        switch (inst->op()) {
            case xir::AtomicOp::EXCHANGE:
                return atomic_rmw_i32("xchg", values.front(), is_signed, false);
            case xir::AtomicOp::COMPARE_EXCHANGE:
                return atomic_compare_exchange(values[0u], values[1u], false);
            case xir::AtomicOp::FETCH_ADD:
                return atomic_rmw_i32("add", values.front(), is_signed, true);
            case xir::AtomicOp::FETCH_SUB:
                return atomic_rmw_i32("sub", values.front(), is_signed, true);
            case xir::AtomicOp::FETCH_AND:
                return atomic_rmw_i32("and", values.front(), is_signed, true);
            case xir::AtomicOp::FETCH_OR:
                return atomic_rmw_i32("or", values.front(), is_signed, true);
            case xir::AtomicOp::FETCH_XOR:
                return atomic_rmw_i32("xor", values.front(), is_signed, true);
            case xir::AtomicOp::FETCH_MIN:
                return atomic_rmw_i32("min", values.front(), is_signed, true);
            case xir::AtomicOp::FETCH_MAX:
                return atomic_rmw_i32("max", values.front(), is_signed, true);
        }
    }
    LUISA_ASSERT(element_type->is_float() && element_type->size() == 4u,
                 "Metal AIR only supports 32-bit floating-point atomics.");
    auto value = values.front();
    if (!is_threadgroup) {
        switch (inst->op()) {
            case xir::AtomicOp::EXCHANGE: return atomic_rmw_f32("xchg", value);
            case xir::AtomicOp::COMPARE_EXCHANGE:
                return atomic_compare_exchange(values[0u], values[1u], true);
            case xir::AtomicOp::FETCH_ADD: return atomic_rmw_f32("add", value);
            case xir::AtomicOp::FETCH_SUB: return atomic_rmw_f32("sub", value);
            default: break;
        }
    } else {
        switch (inst->op()) {
            case xir::AtomicOp::EXCHANGE: {
                auto bits = builder.CreateBitCast(value, builder.getInt32Ty());
                return builder.CreateBitCast(
                    atomic_rmw_i32("xchg", bits, true, false), builder.getFloatTy());
            }
            case xir::AtomicOp::COMPARE_EXCHANGE: {
                auto expected = builder.CreateBitCast(values[0u], builder.getInt32Ty());
                auto desired = builder.CreateBitCast(values[1u], builder.getInt32Ty());
                return builder.CreateBitCast(
                    atomic_compare_exchange(expected, desired, false), builder.getFloatTy());
            }
            default: break;
        }
    }

    LUISA_ASSERT(inst->op() == xir::AtomicOp::FETCH_ADD ||
                     inst->op() == xir::AtomicOp::FETCH_SUB ||
                     inst->op() == xir::AtomicOp::FETCH_MIN ||
                     inst->op() == xir::AtomicOp::FETCH_MAX,
                 "Invalid floating-point atomic operation.");
    auto initial_block = builder.GetInsertBlock();
    auto loop_block = llvm::BasicBlock::Create(_context, "atomic.loop", function.function);
    auto attempt_block = llvm::BasicBlock::Create(_context, "atomic.try", function.function);
    auto exit_block = llvm::BasicBlock::Create(_context, "atomic.exit", function.function);
    auto initial_bits = atomic_load_i32();
    builder.CreateBr(loop_block);

    builder.SetInsertPoint(loop_block);
    auto old_bits = builder.CreatePHI(builder.getInt32Ty(), 2u, "atomic.old.bits");
    old_bits->addIncoming(initial_bits, initial_block);
    auto old_value = builder.CreateBitCast(old_bits, builder.getFloatTy());
    if (inst->op() == xir::AtomicOp::FETCH_MIN || inst->op() == xir::AtomicOp::FETCH_MAX) {
        auto keep_old = inst->op() == xir::AtomicOp::FETCH_MIN ?
                            builder.CreateFCmpOLE(old_value, value) :
                            builder.CreateFCmpOGE(old_value, value);
        builder.CreateCondBr(keep_old, exit_block, attempt_block);
    } else {
        builder.CreateBr(attempt_block);
    }

    builder.SetInsertPoint(attempt_block);
    auto desired_value = inst->op() == xir::AtomicOp::FETCH_ADD ?
                             builder.CreateFAdd(old_value, value) :
                         inst->op() == xir::AtomicOp::FETCH_SUB ?
                             builder.CreateFSub(old_value, value) :
                             value;
    auto desired_bits = builder.CreateBitCast(desired_value, builder.getInt32Ty());
    auto observed_bits = atomic_compare_exchange(old_bits, desired_bits, false);
    auto exchanged = builder.CreateICmpEQ(observed_bits, old_bits);
    builder.CreateCondBr(exchanged, exit_block, loop_block);
    old_bits->addIncoming(observed_bits, attempt_block);

    builder.SetInsertPoint(exit_block);
    return builder.CreateBitCast(old_bits, builder.getFloatTy());
}

llvm::Value *MetalCodegenLLVMImpl::_translate_thread_group(
    IB &builder, FunctionContext &function, const xir::ThreadGroupInst *inst) noexcept {
    auto convergent_call = [&](llvm::StringRef name, llvm::Type *return_type,
                               llvm::ArrayRef<llvm::Value *> arguments) noexcept {
        llvm::SmallVector<llvm::Type *, 3u> parameter_types;
        parameter_types.reserve(arguments.size());
        for (auto argument : arguments) { parameter_types.emplace_back(argument->getType()); }
        auto function_type = llvm::FunctionType::get(return_type, parameter_types, false);
        auto callee = _module.getOrInsertFunction(name, function_type);
        if (auto intrinsic = llvm::dyn_cast<llvm::Function>(callee.getCallee())) {
            intrinsic->setMustProgress();
            intrinsic->setDoesNotThrow();
            intrinsic->setWillReturn();
            intrinsic->addFnAttr(llvm::Attribute::Convergent);
        }
        auto call = builder.CreateCall(callee, arguments);
        call->setConvergent();
        return call;
    };
    auto ballot = [&](llvm::Value *predicate) noexcept {
        return convergent_call("air.simd_ballot.i64", builder.getInt64Ty(), {predicate});
    };
    auto shuffle = [&](auto &&self, llvm::Value *value, const Type *type,
                       llvm::Value *lane, bool first) noexcept -> llvm::Value * {
        auto intrinsic = first ? llvm::StringRef{"simd_broadcast_first"} : llvm::StringRef{"simd_shuffle"};
        llvm::SmallVector<llvm::Value *, 1u> extra;
        if (!first) { extra.emplace_back(lane); }
        if (type->is_bool_or_bool_vector()) {
            auto bool_type = value->getType();
            auto integer_type = type->is_vector() ?
                                    static_cast<llvm::Type *>(llvm::FixedVectorType::get(
                                        builder.getInt32Ty(), type->dimension())) :
                                    builder.getInt32Ty();
            auto integer = builder.CreateZExt(value, integer_type);
            auto shuffled = _air_simd_call(builder, intrinsic, integer, false, extra);
            return builder.CreateTrunc(shuffled, bool_type);
        }
        if (type->is_scalar()) {
            if (type->size() == 8u) {
                auto low = builder.CreateTrunc(value, builder.getInt32Ty());
                auto high = builder.CreateTrunc(
                    builder.CreateLShr(value, builder.getInt64(32u)), builder.getInt32Ty());
                low = _air_simd_call(builder, intrinsic, low, false, extra);
                high = _air_simd_call(builder, intrinsic, high, false, extra);
                auto bits = builder.CreateOr(
                    builder.CreateZExt(low, builder.getInt64Ty()),
                    builder.CreateShl(builder.CreateZExt(high, builder.getInt64Ty()), 32u));
                return bits;
            }
            return _air_simd_call(builder, intrinsic, value, type->is_int(), extra);
        }
        if (type->is_vector()) {
            if (type->element()->size() != 8u) {
                return _air_simd_call(builder, intrinsic, value, type->is_int_vector(), extra);
            }
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(value->getType()));
            for (auto i = 0u; i < type->dimension(); i++) {
                auto element = self(self, builder.CreateExtractElement(value, i), type->element(), lane, first);
                result = builder.CreateInsertElement(result, element, i);
            }
            return result;
        }
        auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(value->getType()));
        if (type->is_matrix() || type->is_array()) {
            auto element_type = type->is_matrix() ?
                                    Type::vector(type->element(), type->dimension()) :
                                    type->element();
            for (auto i = 0u; i < type->dimension(); i++) {
                auto element = self(self, builder.CreateExtractValue(value, i), element_type, lane, first);
                result = builder.CreateInsertValue(result, element, i);
            }
            return result;
        }
        LUISA_ASSERT(type->is_structure(), "Invalid aggregate type for AIR SIMD shuffle.");
        for (auto i = 0u; i < type->members().size(); i++) {
            auto member = self(self, builder.CreateExtractValue(value, i), type->members()[i], lane, first);
            result = builder.CreateInsertValue(result, member, i);
        }
        return result;
    };

    switch (inst->op()) {
        case xir::ThreadGroupOp::SHADER_EXECUTION_REORDER:
            // The existing Metal source backend implements SER as a no-op.
            return nullptr;
        case xir::ThreadGroupOp::RASTER_QUAD_DDX: [[fallthrough]];
        case xir::ThreadGroupOp::RASTER_QUAD_DDY: {
            LUISA_ASSERT(
                _config.program == MetalAIRProgram::RASTER_FRAGMENT,
                "Raster derivatives are only valid in a Metal AIR fragment stage.");
            auto operand = _value(builder, function, inst->operand(0u));
            auto type = operand->getType();
            auto element = type->isVectorTy() ?
                               llvm::cast<llvm::FixedVectorType>(type)->getElementType() :
                               type;
            LUISA_ASSERT(element->isHalfTy() || element->isFloatTy(),
                         "Metal AIR raster derivatives require f16 or f32 values.");
            auto name = luisa::string{
                inst->op() == xir::ThreadGroupOp::RASTER_QUAD_DDX ?
                    "air.dfdx." :
                    "air.dfdy."};
            if (type->isVectorTy()) {
                name.append("v");
                name.append(std::to_string(
                    llvm::cast<llvm::FixedVectorType>(type)->getNumElements()));
            }
            name.append(element->isHalfTy() ? "f16" : "f32");
            return convergent_call(
                llvm::StringRef{name.data(), name.size()}, type, {operand});
        }
        case xir::ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE:
            return convergent_call("air.simd_is_first", builder.getInt1Ty(), {});
        case xir::ThreadGroupOp::WARP_FIRST_ACTIVE_LANE: {
            auto active = convergent_call(
                "air.simd_active_threads_mask.i64", builder.getInt64Ty(), {});
            auto first = _air_integer_call(builder, "ctz", active, false);
            return builder.CreateZExtOrTrunc(first, _type(inst->type())->reg_type);
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL: {
            auto operand = inst->operand(0u);
            auto value = _value(builder, function, operand);
            auto first = shuffle(shuffle, value, operand->type(), nullptr, true);
            auto equal = operand->type()->is_float_or_float_vector() ?
                             builder.CreateFCmpOEQ(value, first) :
                             builder.CreateICmpEQ(value, first);
            if (!equal->getType()->isVectorTy()) {
                return convergent_call("air.simd_all", builder.getInt1Ty(), {equal});
            }
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(equal->getType()));
            for (auto i = 0u; i < operand->type()->dimension(); i++) {
                auto component = convergent_call(
                    "air.simd_all", builder.getInt1Ty(), {builder.CreateExtractElement(equal, i)});
                result = builder.CreateInsertElement(result, component, i);
            }
            return result;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_AND: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_OR: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_XOR: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_MAX: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_MIN: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_PRODUCT: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_SUM: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_PREFIX_SUM: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_PREFIX_PRODUCT: {
            auto operand = inst->operand(0u);
            auto value = _value(builder, function, operand);
            auto name = [&]() noexcept -> llvm::StringRef {
                switch (inst->op()) {
                    case xir::ThreadGroupOp::WARP_ACTIVE_BIT_AND: return "simd_and";
                    case xir::ThreadGroupOp::WARP_ACTIVE_BIT_OR: return "simd_or";
                    case xir::ThreadGroupOp::WARP_ACTIVE_BIT_XOR: return "simd_xor";
                    case xir::ThreadGroupOp::WARP_ACTIVE_MAX: return "simd_max";
                    case xir::ThreadGroupOp::WARP_ACTIVE_MIN: return "simd_min";
                    case xir::ThreadGroupOp::WARP_ACTIVE_PRODUCT: return "simd_product";
                    case xir::ThreadGroupOp::WARP_ACTIVE_SUM: return "simd_sum";
                    case xir::ThreadGroupOp::WARP_PREFIX_SUM: return "simd_prefix_exclusive_sum";
                    case xir::ThreadGroupOp::WARP_PREFIX_PRODUCT: return "simd_prefix_exclusive_product";
                    default: break;
                }
                LUISA_ERROR_WITH_LOCATION("Invalid AIR SIMD reduction operation.");
            }();
            return _air_simd_call(builder, name, value, operand->type()->is_int_or_int_vector());
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_COUNT_BITS: {
            auto mask = ballot(_value(builder, function, inst->operand(0u)));
            auto count = _air_integer_call(builder, "popcount", mask, false);
            return builder.CreateZExtOrTrunc(count, _type(inst->type())->reg_type);
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL:
            return convergent_call(
                "air.simd_all", builder.getInt1Ty(), {_value(builder, function, inst->operand(0u))});
        case xir::ThreadGroupOp::WARP_ACTIVE_ANY:
            return convergent_call(
                "air.simd_any", builder.getInt1Ty(), {_value(builder, function, inst->operand(0u))});
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_MASK: {
            auto mask = ballot(_value(builder, function, inst->operand(0u)));
            auto result = static_cast<llvm::Value *>(llvm::Constant::getNullValue(_type(inst->type())->reg_type));
            result = builder.CreateInsertElement(
                result, builder.CreateTrunc(mask, builder.getInt32Ty()), builder.getInt32(0u));
            auto high = builder.CreateTrunc(builder.CreateLShr(mask, 32u), builder.getInt32Ty());
            return builder.CreateInsertElement(result, high, builder.getInt32(1u));
        }
        case xir::ThreadGroupOp::WARP_PREFIX_COUNT_BITS: {
            auto predicate = _value(builder, function, inst->operand(0u));
            auto value = builder.CreateZExt(predicate, builder.getInt16Ty());
            auto count = _air_simd_call(builder, "simd_prefix_exclusive_sum", value, false);
            return builder.CreateZExtOrTrunc(count, _type(inst->type())->reg_type);
        }
        case xir::ThreadGroupOp::WARP_READ_LANE: {
            auto operand = inst->operand(0u);
            auto value = _value(builder, function, operand);
            auto lane = builder.CreateTrunc(
                _value(builder, function, inst->operand(1u)), builder.getInt16Ty());
            return shuffle(shuffle, value, operand->type(), lane, false);
        }
        case xir::ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE: {
            auto operand = inst->operand(0u);
            return shuffle(
                shuffle, _value(builder, function, operand), operand->type(), nullptr, true);
        }
        case xir::ThreadGroupOp::SYNCHRONIZE_BLOCK: {
            auto function_type = llvm::FunctionType::get(
                builder.getVoidTy(), {builder.getInt32Ty(), builder.getInt32Ty()}, false);
            auto barrier = _module.getOrInsertFunction("air.wg.barrier", function_type);
            if (auto f = llvm::dyn_cast<llvm::Function>(barrier.getCallee())) {
                f->setMustProgress();
                f->setDoesNotThrow();
                f->setWillReturn();
                f->addFnAttr(llvm::Attribute::Convergent);
            }
            auto call = builder.CreateCall(barrier, {builder.getInt32(2u), builder.getInt32(1u)});
            call->setConvergent();
            return nullptr;
        }
    }
    _unsupported_instruction(inst);
}

llvm::Value *MetalCodegenLLVMImpl::_air_scalar_call(IB &builder, llvm::StringRef name, llvm::ArrayRef<llvm::Value *> arguments) noexcept {
    LUISA_ASSERT(!arguments.empty(), "AIR intrinsic requires at least one argument.");
    auto type = arguments.front()->getType();
    LUISA_ASSERT(type->isHalfTy() || type->isFloatTy(), "AIR math intrinsic requires f16 or f32 scalar arguments.");
    auto use_fast = _config.enable_fast_math && type->isFloatTy() &&
                    name != "fma" && name != "fabs" && name != "copysign" &&
                    name != "fmin" && name != "fmax" &&
                    name != "clamp" && name != "saturate";
    auto function_name = std::string{"air."};
    if (use_fast) { function_name.append("fast_"); }
    function_name.append(name.data(), name.size());
    function_name.append(type->isHalfTy() ? ".f16" : ".f32");
    llvm::SmallVector<llvm::Type *> parameter_types(arguments.size(), type);
    auto function_type = llvm::FunctionType::get(type, parameter_types, false);
    auto callee = _module.getOrInsertFunction(function_name, function_type);
    if (auto function = llvm::dyn_cast<llvm::Function>(callee.getCallee())) {
        function->setMustProgress();
        function->setDoesNotFreeMemory();
        function->setNoSync();
        function->setDoesNotThrow();
        function->setWillReturn();
        function->setDoesNotAccessMemory();
        function->setSpeculatable();
    }
    return builder.CreateCall(callee, arguments);
}

void MetalCodegenLLVMImpl::_air_atomic_fence(IB &builder, uint32_t memory_flags) noexcept {
    auto i32 = builder.getInt32Ty();
    auto function_type = llvm::FunctionType::get(builder.getVoidTy(), {i32, i32, i32}, false);
    auto callee = _module.getOrInsertFunction("air.atomic.fence", function_type);
    if (auto function = llvm::dyn_cast<llvm::Function>(callee.getCallee())) {
        function->setMustProgress();
        function->setDoesNotThrow();
        function->setWillReturn();
    }
    // Apple Metal lowers atomic_thread_fence(flags, memory_order_seq_cst) to
    // air.atomic.fence(flags, 5, 2). The final operand is the device scope.
    builder.CreateCall(callee, {builder.getInt32(memory_flags),
                                builder.getInt32(5u), builder.getInt32(2u)});
}

llvm::Value *MetalCodegenLLVMImpl::_air_sampler(
    IB &builder, llvm::Value *filter, llvm::Value *address) noexcept {
    filter = builder.CreateZExtOrTrunc(filter, builder.getInt32Ty());
    address = builder.CreateZExtOrTrunc(address, builder.getInt32Ty());
    auto code = builder.CreateOr(builder.CreateShl(filter, 2u), address);
    return _air_sampler_code(builder, code);
}

llvm::Value *MetalCodegenLLVMImpl::_air_sampler_code(
    IB &builder, llvm::Value *code) noexcept {
    auto sampler_pointer = llvm::PointerType::get(_context, air_address_space_constant);
    if (_sampler_table == nullptr) {
        // These 64-bit descriptors and their duplicate mapping are the exact
        // sampler-state constants emitted by Apple Metal 32023.883 for
        // Luisa's 4 filters x 4 address modes. Anisotropic repeat/mirror/zero
        // intentionally reuse the linear-linear states, matching the MSL path.
        constexpr std::array<uint64_t, 13u> descriptors{
            34901797601017929ull, 34901797601018002ull,
            34901797601018075ull, 34901797601017856ull,
            34901797601028681ull, 34901797601028754ull,
            34901797601028827ull, 34901797601028608ull,
            34901797601036873ull, 34901797601036946ull,
            34901797601037019ull, 34901797601036800ull,
            34901797616765513ull};
        constexpr std::array<uint8_t, 16u> table_indices{
            0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u,
            8u, 9u, 10u, 11u, 12u, 9u, 10u, 11u};
        auto i64 = builder.getInt64Ty();
        auto modern_sampler_state = _config.air_version.major > 2u ||
                                    (_config.air_version.major == 2u &&
                                     _config.air_version.minor >= 7u);
        auto state_type = modern_sampler_state ?
                              static_cast<llvm::Type *>(llvm::ArrayType::get(i64, 2u)) :
                              static_cast<llvm::Type *>(i64);
        std::array<llvm::GlobalVariable *, descriptors.size()> states{};
        auto sampler_metadata = _module.getOrInsertNamedMetadata("air.sampler_states");
        for (auto i = 0u; i < descriptors.size(); i++) {
            auto initializer = modern_sampler_state ?
                                   static_cast<llvm::Constant *>(llvm::ConstantArray::get(
                                       llvm::cast<llvm::ArrayType>(state_type),
                                       {llvm::ConstantInt::get(i64, descriptors[i]),
                                        llvm::ConstantInt::get(i64, 0u)})) :
                                   static_cast<llvm::Constant *>(llvm::ConstantInt::get(
                                       i64, descriptors[i] | (1ull << 63u)));
            states[i] = new llvm::GlobalVariable(
                _module, state_type, true, llvm::GlobalValue::InternalLinkage,
                initializer, i == 0u ? "__air_sampler_state" : "__air_sampler_state." + std::to_string(i),
                nullptr, llvm::GlobalVariable::NotThreadLocal, air_address_space_constant);
            states[i]->setAlignment(llvm::Align{8u});
            sampler_metadata->addOperand(llvm::MDNode::get(
                _context, {md_string(_context, "air.sampler_state"),
                           llvm::ValueAsMetadata::get(states[i])}));
        }
        _air_sampler_wrapper_type = llvm::StructType::create(
            _context, {sampler_pointer}, "luisa.air.sampler");
        _set_struct_pointer_element_type(
            _air_sampler_wrapper_type, 0u, state_type);
        llvm::SmallVector<llvm::Constant *, 16u> samplers;
        for (auto index : table_indices) {
            auto pointer = llvm::ConstantExpr::getPointerCast(
                states[index], sampler_pointer);
            samplers.emplace_back(llvm::ConstantStruct::get(
                _air_sampler_wrapper_type, {pointer}));
        }
        auto table_type = llvm::ArrayType::get(
            _air_sampler_wrapper_type, samplers.size());
        _sampler_table = new llvm::GlobalVariable(
            _module, table_type, true, llvm::GlobalValue::InternalLinkage,
            llvm::ConstantArray::get(table_type, samplers), "luisa.air.samplers",
            nullptr, llvm::GlobalVariable::NotThreadLocal, air_address_space_constant);
        _sampler_table->setAlignment(llvm::Align{8u});
    }
    code = builder.CreateAnd(
        builder.CreateZExtOrTrunc(code, builder.getInt32Ty()), builder.getInt32(15u));
    auto table_type = llvm::cast<llvm::ArrayType>(_sampler_table->getValueType());
    auto pointer = builder.CreateInBoundsGEP(
        table_type, _sampler_table,
        {builder.getInt64(0u), builder.CreateZExt(code, builder.getInt64Ty())});
    auto sampler_field = builder.CreateStructGEP(
        _air_sampler_wrapper_type, pointer, 0u);
    return builder.CreateAlignedLoad(
        sampler_pointer, sampler_field, llvm::Align{8u});
}

llvm::Value *MetalCodegenLLVMImpl::_air_integer_call(
    IB &builder, llvm::StringRef name, llvm::Value *value, bool zero_is_undefined) noexcept {
    auto type = value->getType();
    LUISA_ASSERT(type->isIntOrIntVectorTy(), "AIR integer intrinsic requires an integer scalar or vector.");
    auto scalar_width = type->getScalarSizeInBits();
    auto function_name = std::string{"air."};
    function_name.append(name.data(), name.size());
    function_name.push_back('.');
    if (auto vector_type = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
        function_name.push_back('v');
        function_name.append(std::to_string(vector_type->getNumElements()));
    }
    function_name.push_back('i');
    function_name.append(std::to_string(scalar_width));
    llvm::SmallVector<llvm::Value *, 2u> arguments{value};
    if (name == "clz" || name == "ctz") {
        arguments.emplace_back(builder.getInt1(zero_is_undefined));
    }
    llvm::SmallVector<llvm::Type *, 2u> parameter_types;
    for (auto argument : arguments) { parameter_types.emplace_back(argument->getType()); }
    auto function_type = llvm::FunctionType::get(type, parameter_types, false);
    auto callee = _module.getOrInsertFunction(function_name, function_type);
    if (auto function = llvm::dyn_cast<llvm::Function>(callee.getCallee())) {
        function->setMustProgress();
        function->setDoesNotFreeMemory();
        function->setNoSync();
        function->setDoesNotThrow();
        function->setWillReturn();
        function->setDoesNotAccessMemory();
    }
    return builder.CreateCall(callee, arguments);
}

llvm::Value *MetalCodegenLLVMImpl::_air_simd_call(
    IB &builder, llvm::StringRef name, llvm::Value *value,
    bool signed_integer, llvm::ArrayRef<llvm::Value *> extra_arguments) noexcept {
    auto type = value->getType();
    auto scalar = type->getScalarType();
    LUISA_ASSERT((scalar->isIntegerTy() && !scalar->isIntegerTy(1u)) ||
                     scalar->isHalfTy() || scalar->isFloatTy(),
                 "AIR SIMD intrinsic requires an integer, half, or float scalar/vector.");
    auto function_name = std::string{"air."};
    function_name.append(name.data(), name.size());
    if (scalar->isIntegerTy()) {
        function_name.append(signed_integer ? ".s." : ".u.");
    } else {
        function_name.push_back('.');
    }
    if (auto vector_type = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
        function_name.push_back('v');
        function_name.append(std::to_string(vector_type->getNumElements()));
    }
    if (scalar->isIntegerTy()) {
        function_name.push_back('i');
        function_name.append(std::to_string(scalar->getIntegerBitWidth()));
    } else {
        function_name.push_back('f');
        function_name.append(std::to_string(scalar->getPrimitiveSizeInBits()));
    }
    llvm::SmallVector<llvm::Value *, 3u> arguments{value};
    arguments.append(extra_arguments.begin(), extra_arguments.end());
    llvm::SmallVector<llvm::Type *, 3u> parameter_types;
    parameter_types.reserve(arguments.size());
    for (auto argument : arguments) { parameter_types.emplace_back(argument->getType()); }
    auto function_type = llvm::FunctionType::get(type, parameter_types, false);
    auto callee = _module.getOrInsertFunction(function_name, function_type);
    if (auto function = llvm::dyn_cast<llvm::Function>(callee.getCallee())) {
        function->setMustProgress();
        function->setDoesNotThrow();
        function->setWillReturn();
        function->addFnAttr(llvm::Attribute::Convergent);
    }
    auto call = builder.CreateCall(callee, arguments);
    call->setConvergent();
    return call;
}

llvm::Value *MetalCodegenLLVMImpl::_air_unary(IB &builder, llvm::StringRef name, llvm::Value *value) noexcept {
    if (!value->getType()->isVectorTy()) { return _air_scalar_call(builder, name, {value}); }
    auto vector_type = llvm::cast<llvm::FixedVectorType>(value->getType());
    auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(vector_type));
    for (auto i = 0u; i < vector_type->getNumElements(); i++) {
        auto scalar = _air_scalar_call(builder, name, {builder.CreateExtractElement(value, i)});
        result = builder.CreateInsertElement(result, scalar, i);
    }
    return result;
}

llvm::Value *MetalCodegenLLVMImpl::_air_binary(IB &builder, llvm::StringRef name, llvm::Value *lhs, llvm::Value *rhs) noexcept {
    if (!lhs->getType()->isVectorTy()) { return _air_scalar_call(builder, name, {lhs, rhs}); }
    auto vector_type = llvm::cast<llvm::FixedVectorType>(lhs->getType());
    auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(vector_type));
    for (auto i = 0u; i < vector_type->getNumElements(); i++) {
        auto scalar = _air_scalar_call(builder, name,
                                       {builder.CreateExtractElement(lhs, i), builder.CreateExtractElement(rhs, i)});
        result = builder.CreateInsertElement(result, scalar, i);
    }
    return result;
}

llvm::Value *MetalCodegenLLVMImpl::_air_ternary(IB &builder, llvm::StringRef name, llvm::Value *a, llvm::Value *b, llvm::Value *c) noexcept {
    if (!a->getType()->isVectorTy()) { return _air_scalar_call(builder, name, {a, b, c}); }
    auto vector_type = llvm::cast<llvm::FixedVectorType>(a->getType());
    auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(vector_type));
    for (auto i = 0u; i < vector_type->getNumElements(); i++) {
        auto scalar = _air_scalar_call(builder, name,
                                       {builder.CreateExtractElement(a, i),
                                        builder.CreateExtractElement(b, i),
                                        builder.CreateExtractElement(c, i)});
        result = builder.CreateInsertElement(result, scalar, i);
    }
    return result;
}

}// namespace luisa::compute::metal::detail
