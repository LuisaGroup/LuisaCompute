#include "metal_codegen_llvm_impl.h"

namespace luisa::compute::metal::detail {

namespace {

[[nodiscard]] uint64_t metal_constant_index(const xir::Constant *constant) noexcept {
    switch (constant->type()->tag()) {
        case Type::Tag::INT8: return static_cast<uint64_t>(constant->as<int8_t>());
        case Type::Tag::UINT8: return constant->as<uint8_t>();
        case Type::Tag::INT16: return static_cast<uint64_t>(constant->as<int16_t>());
        case Type::Tag::UINT16: return constant->as<uint16_t>();
        case Type::Tag::INT32: return static_cast<uint64_t>(constant->as<int32_t>());
        case Type::Tag::UINT32: return constant->as<uint32_t>();
        case Type::Tag::INT64: return static_cast<uint64_t>(constant->as<int64_t>());
        case Type::Tag::UINT64: return constant->as<uint64_t>();
        default: LUISA_ERROR_WITH_LOCATION("Invalid XIR aggregate index type '{}'.", constant->type()->description());
    }
}

[[nodiscard]] llvm::Value *metal_matrix_determinant(
    llvm::IRBuilder<> &builder, llvm::Value *matrix,
    int skipped_row = -1, int skipped_column = -1) noexcept {
    auto matrix_type = llvm::cast<llvm::ArrayType>(matrix->getType());
    auto dimension = static_cast<unsigned>(matrix_type->getNumElements());
    llvm::SmallVector<unsigned, 4u> rows;
    llvm::SmallVector<unsigned, 4u> columns;
    for (auto i = 0u; i < dimension; i++) {
        if (static_cast<int>(i) != skipped_row) { rows.emplace_back(i); }
        if (static_cast<int>(i) != skipped_column) { columns.emplace_back(i); }
    }
    LUISA_ASSERT(rows.size() == columns.size() && !rows.empty(), "Invalid matrix minor.");
    llvm::SmallVector<unsigned, 4u> permutation;
    for (auto i = 0u; i < columns.size(); i++) { permutation.emplace_back(i); }
    auto scalar_type = matrix_type->getElementType()->getScalarType();
    auto result = static_cast<llvm::Value *>(llvm::ConstantFP::get(scalar_type, 0.0));
    do {
        auto product = static_cast<llvm::Value *>(llvm::ConstantFP::get(scalar_type, 1.0));
        auto inversion_count = 0u;
        for (auto i = 0u; i < permutation.size(); i++) {
            for (auto j = i + 1u; j < permutation.size(); j++) {
                inversion_count += permutation[i] > permutation[j] ? 1u : 0u;
            }
            auto column = builder.CreateExtractValue(matrix, columns[permutation[i]]);
            auto element = builder.CreateExtractElement(column, static_cast<uint64_t>(rows[i]));
            product = builder.CreateFMul(product, element);
        }
        result = inversion_count % 2u == 0u ?
                     builder.CreateFAdd(result, product) :
                     builder.CreateFSub(result, product);
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    return result;
}

[[nodiscard]] llvm::Function *get_or_create_strict_fmod_f32(llvm::Module &module) noexcept {
    constexpr llvm::StringLiteral fn_name{"__luisa_strict_fmod_f32"};
    if (auto *existing = module.getFunction(fn_name)) {
        return existing;
    }
    auto &ctx = module.getContext();
    auto *f32 = llvm::Type::getFloatTy(ctx);
    auto *i32 = llvm::Type::getInt32Ty(ctx);
    auto *i1 = llvm::Type::getInt1Ty(ctx);
    auto *fn_ty = llvm::FunctionType::get(f32, {f32, f32}, false);
    auto *fn = llvm::Function::Create(
        fn_ty, llvm::GlobalValue::PrivateLinkage, fn_name, module);
    fn->addFnAttr(llvm::Attribute::AlwaysInline);
    fn->addFnAttr("no-trapping-math", "true");
    fn->setDoesNotAccessMemory();
    fn->setDoesNotThrow();
    fn->setNoSync();
    fn->setWillReturn();

    auto *entry_bb = llvm::BasicBlock::Create(ctx, "entry", fn);
    auto *check_lt_bb = llvm::BasicBlock::Create(ctx, "check.lt", fn);
    auto *check_eq_bb = llvm::BasicBlock::Create(ctx, "check.eq", fn);
    auto *try_native_bb = llvm::BasicBlock::Create(ctx, "try.native", fn);
    auto *slow_bb = llvm::BasicBlock::Create(ctx, "slow", fn);
    auto *loop_cond_bb = llvm::BasicBlock::Create(ctx, "loop.cond", fn);
    auto *loop_body_bb = llvm::BasicBlock::Create(ctx, "loop.body", fn);
    auto *loop_step_bb = llvm::BasicBlock::Create(ctx, "loop.step", fn);
    auto *after_loop_bb = llvm::BasicBlock::Create(ctx, "after.loop", fn);
    auto *renorm_bb = llvm::BasicBlock::Create(ctx, "renorm", fn);
    auto *ret_zero_bb = llvm::BasicBlock::Create(ctx, "ret.zero", fn);

    auto *x = fn->getArg(0u);
    auto *y = fn->getArg(1u);
    auto c32 = [&](uint32_t v) { return llvm::ConstantInt::get(i32, v); };

    llvm::IRBuilder<> b{entry_bb};
    auto *x_bits = b.CreateBitCast(x, i32);
    auto *y_bits = b.CreateBitCast(y, i32);
    auto *x_mag = b.CreateAnd(x_bits, c32(0x7fffffffu));
    auto *y_mag = b.CreateAnd(y_bits, c32(0x7fffffffu));
    auto *sign = b.CreateAnd(x_bits, c32(0x80000000u));

    auto *bad_y0 = b.CreateICmpEQ(y_mag, c32(0u));
    auto *bad_x_inf_nan = b.CreateICmpUGE(x_mag, c32(0x7f800000u));
    auto *bad_y_nan = b.CreateICmpUGT(y_mag, c32(0x7f800000u));
    auto *is_nan_out = b.CreateOr(bad_y0, b.CreateOr(bad_x_inf_nan, bad_y_nan));
    auto *ret_nan_bb = llvm::BasicBlock::Create(ctx, "ret.nan", fn);
    b.CreateCondBr(is_nan_out, ret_nan_bb, check_lt_bb);

    b.SetInsertPoint(ret_nan_bb);
    b.CreateRet(b.CreateBitCast(c32(0x7fc00000u), f32));

    b.SetInsertPoint(check_lt_bb);
    auto *ret_x_bb = llvm::BasicBlock::Create(ctx, "ret.x", fn);
    b.CreateCondBr(b.CreateICmpULT(x_mag, y_mag), ret_x_bb, check_eq_bb);

    b.SetInsertPoint(ret_x_bb);
    b.CreateRet(x);

    b.SetInsertPoint(check_eq_bb);
    b.CreateCondBr(b.CreateICmpEQ(x_mag, y_mag), ret_zero_bb, try_native_bb);

    b.SetInsertPoint(ret_zero_bb);
    b.CreateRet(b.CreateBitCast(sign, f32));

    b.SetInsertPoint(try_native_bb);
    auto *native = b.CreateFRem(x, y);
    auto *native_mag = b.CreateAnd(b.CreateBitCast(native, i32), c32(0x7fffffffu));
    auto *exp_span_ok = b.CreateICmpULT(
        b.CreateSub(x_mag, y_mag), c32(23u << 23u));
    auto *native_ok = b.CreateAnd(
        exp_span_ok,
        b.CreateAnd(
            b.CreateICmpNE(native_mag, c32(0u)),
            b.CreateICmpULT(native_mag, y_mag)));
    auto *ret_native_bb = llvm::BasicBlock::Create(ctx, "ret.native", fn);
    b.CreateCondBr(native_ok, ret_native_bb, slow_bb);

    b.SetInsertPoint(ret_native_bb);
    b.CreateRet(native);

    b.SetInsertPoint(slow_bb);
    auto normalize_mag = [&](llvm::Value *mag) -> std::pair<llvm::Value *, llvm::Value *> {
        auto *exp_raw = b.CreateLShr(mag, c32(23u));
        auto *is_sub = b.CreateICmpEQ(exp_raw, c32(0u));
        auto *lz = b.CreateBinaryIntrinsic(
            llvm::Intrinsic::ctlz, mag, llvm::ConstantInt::get(i1, 1u));
        auto *sub_shift = b.CreateSub(lz, c32(8u));
        auto *sub_exp = b.CreateSub(c32(1u), sub_shift);
        auto *sub_mant = b.CreateShl(mag, sub_shift);
        auto *norm_mant = b.CreateOr(b.CreateAnd(mag, c32(0x007fffffu)), c32(0x00800000u));
        return {
            b.CreateSelect(is_sub, sub_exp, exp_raw),
            b.CreateSelect(is_sub, sub_mant, norm_mant)};
    };
    auto [x_exp0, x_mant0] = normalize_mag(x_mag);
    auto [y_exp0, y_mant0] = normalize_mag(y_mag);
    b.CreateBr(loop_cond_bb);

    b.SetInsertPoint(loop_cond_bb);
    auto *xe_phi = b.CreatePHI(i32, 2u);
    auto *xm_phi = b.CreatePHI(i32, 2u);
    xe_phi->addIncoming(x_exp0, slow_bb);
    xm_phi->addIncoming(x_mant0, slow_bb);
    b.CreateCondBr(b.CreateICmpSGT(xe_phi, y_exp0), loop_body_bb, after_loop_bb);

    b.SetInsertPoint(loop_body_bb);
    auto *diff_loop = b.CreateSub(xm_phi, y_mant0);
    b.CreateCondBr(b.CreateICmpEQ(diff_loop, c32(0u)), ret_zero_bb, loop_step_bb);

    b.SetInsertPoint(loop_step_bb);
    auto *ge_loop = b.CreateICmpUGE(xm_phi, y_mant0);
    auto *xm_reduced = b.CreateSelect(ge_loop, diff_loop, xm_phi);
    auto *xm_next = b.CreateShl(xm_reduced, c32(1u));
    auto *xe_next = b.CreateSub(xe_phi, c32(1u));
    xe_phi->addIncoming(xe_next, loop_step_bb);
    xm_phi->addIncoming(xm_next, loop_step_bb);
    b.CreateBr(loop_cond_bb);

    b.SetInsertPoint(after_loop_bb);
    auto *diff_end = b.CreateSub(xm_phi, y_mant0);
    b.CreateCondBr(b.CreateICmpEQ(diff_end, c32(0u)), ret_zero_bb, renorm_bb);

    b.SetInsertPoint(renorm_bb);
    auto *ge_end = b.CreateICmpUGE(xm_phi, y_mant0);
    auto *xm_rem = b.CreateSelect(ge_end, diff_end, xm_phi);
    auto *lz_rem = b.CreateBinaryIntrinsic(
        llvm::Intrinsic::ctlz, xm_rem, llvm::ConstantInt::get(i1, 1u));
    auto *renorm_shift = b.CreateSub(lz_rem, c32(8u));
    auto *xm_norm = b.CreateShl(xm_rem, renorm_shift);
    auto *xe_norm = b.CreateSub(xe_phi, renorm_shift);
    auto *is_norm_out = b.CreateICmpSGT(xe_norm, c32(0u));
    auto *norm_packed = b.CreateOr(
        b.CreateSub(xm_norm, c32(0x00800000u)),
        b.CreateShl(xe_norm, c32(23u)));
    auto *denorm_shift = b.CreateSub(c32(1u), xe_norm);
    auto *denorm_packed = b.CreateLShr(xm_norm, denorm_shift);
    auto *mag_packed = b.CreateSelect(is_norm_out, norm_packed, denorm_packed);
    b.CreateRet(b.CreateBitCast(b.CreateOr(sign, mag_packed), f32));

    return fn;
}

}// namespace

llvm::Value *MetalCodegenLLVMImpl::_translate_arithmetic(IB &builder, FunctionContext &function, const xir::ArithmeticInst *inst) noexcept {
    auto unary = [&](auto operation) noexcept -> llvm::Value * {
        LUISA_ASSERT(inst->operand_count() == 1u, "Invalid unary XIR arithmetic operand count.");
        return operation(_value(builder, function, inst->operand(0u)));
    };
    auto binary = [&](auto operation) noexcept -> llvm::Value * {
        LUISA_ASSERT(inst->operand_count() == 2u, "Invalid binary XIR arithmetic operand count.");
        return operation(_value(builder, function, inst->operand(0u)),
                         _value(builder, function, inst->operand(1u)));
    };
    auto ternary = [&](auto operation) noexcept -> llvm::Value * {
        LUISA_ASSERT(inst->operand_count() == 3u, "Invalid ternary XIR arithmetic operand count.");
        return operation(_value(builder, function, inst->operand(0u)),
                         _value(builder, function, inst->operand(1u)),
                         _value(builder, function, inst->operand(2u)));
    };
    auto comparison = [&](auto signed_op, auto unsigned_op, auto float_op) noexcept -> llvm::Value * {
        return binary([&](llvm::Value *lhs, llvm::Value *rhs) noexcept {
            auto operand_type = inst->operand(0u)->type();
            return operand_type->is_int_or_int_vector()                                             ? signed_op(lhs, rhs) :
                   operand_type->is_uint_or_uint_vector() || operand_type->is_bool_or_bool_vector() ? unsigned_op(lhs, rhs) :
                                                                                                      float_op(lhs, rhs);
        });
    };
    auto component_binary = [&](auto operation) noexcept -> llvm::Value * {
        auto lhs = _value(builder, function, inst->operand(0u));
        auto rhs = _value(builder, function, inst->operand(1u));
        if (!inst->type()->is_matrix()) { return operation(lhs, rhs); }
        auto dimension = inst->type()->dimension();
        auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(_type(inst->type())->reg_type));
        if (!lhs->getType()->isArrayTy()) { lhs = builder.CreateVectorSplat(dimension, lhs); }
        if (!rhs->getType()->isArrayTy()) { rhs = builder.CreateVectorSplat(dimension, rhs); }
        for (auto i = 0u; i < dimension; i++) {
            auto lhs_column = lhs->getType()->isArrayTy() ? builder.CreateExtractValue(lhs, i) : lhs;
            auto rhs_column = rhs->getType()->isArrayTy() ? builder.CreateExtractValue(rhs, i) : rhs;
            result = builder.CreateInsertValue(result, operation(lhs_column, rhs_column), i);
        }
        return result;
    };
    auto reduce = [&](llvm::Value *value, llvm::Value *identity, auto operation) noexcept -> llvm::Value * {
        if (!value->getType()->isVectorTy()) { return value; }
        auto vector_type = llvm::cast<llvm::FixedVectorType>(value->getType());
        auto result = identity;
        for (auto i = 0u; i < vector_type->getNumElements(); i++) {
            result = operation(result, builder.CreateExtractElement(value, i));
        }
        return result;
    };
    switch (inst->op()) {
        case xir::ArithmeticOp::UNARY_MINUS: return unary([&](auto value) noexcept {
            return value->getType()->isIntOrIntVectorTy() ? builder.CreateNeg(value) : builder.CreateFNeg(value);
        });
        case xir::ArithmeticOp::UNARY_BIT_NOT: return unary([&](auto value) noexcept { return builder.CreateNot(value); });
        case xir::ArithmeticOp::BINARY_ADD: return component_binary([&](auto lhs, auto rhs) noexcept {
            return lhs->getType()->isIntOrIntVectorTy() ? builder.CreateAdd(lhs, rhs) : builder.CreateFAdd(lhs, rhs);
        });
        case xir::ArithmeticOp::BINARY_SUB: return component_binary([&](auto lhs, auto rhs) noexcept {
            return lhs->getType()->isIntOrIntVectorTy() ? builder.CreateSub(lhs, rhs) : builder.CreateFSub(lhs, rhs);
        });
        case xir::ArithmeticOp::BINARY_MUL: return component_binary([&](auto lhs, auto rhs) noexcept {
            return lhs->getType()->isIntOrIntVectorTy() ? builder.CreateMul(lhs, rhs) : builder.CreateFMul(lhs, rhs);
        });
        case xir::ArithmeticOp::BINARY_DIV: return component_binary([&](auto lhs, auto rhs) noexcept {
            return inst->type()->is_int_or_int_vector()   ? builder.CreateSDiv(lhs, rhs) :
                   inst->type()->is_uint_or_uint_vector() ? builder.CreateUDiv(lhs, rhs) :
                                                            builder.CreateFDiv(lhs, rhs);
        });
        case xir::ArithmeticOp::BINARY_MOD: return binary([&](auto lhs, auto rhs) noexcept -> llvm::Value * {
            if (inst->type()->is_int_or_int_vector()) { return builder.CreateSRem(lhs, rhs); }
            if (inst->type()->is_uint_or_uint_vector()) { return builder.CreateURem(lhs, rhs); }
            if (!_config.enable_fast_math && lhs->getType()->getScalarType()->isFloatTy()) {
                auto *fmod_fn = get_or_create_strict_fmod_f32(_module);
                if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(lhs->getType())) {
                    llvm::Value *res = llvm::PoisonValue::get(vt);
                    for (unsigned i = 0u; i < vt->getNumElements(); ++i) {
                        auto *lx = builder.CreateExtractElement(lhs, static_cast<uint64_t>(i));
                        auto *ry = builder.CreateExtractElement(rhs, static_cast<uint64_t>(i));
                        res = builder.CreateInsertElement(
                            res, builder.CreateCall(fmod_fn, {lx, ry}), static_cast<uint64_t>(i));
                    }
                    return res;
                }
                return builder.CreateCall(fmod_fn, {lhs, rhs});
            }
            return builder.CreateFRem(lhs, rhs);
        });
        case xir::ArithmeticOp::BINARY_BIT_AND: return binary([&](auto lhs, auto rhs) noexcept { return builder.CreateAnd(lhs, rhs); });
        case xir::ArithmeticOp::BINARY_BIT_OR: return binary([&](auto lhs, auto rhs) noexcept { return builder.CreateOr(lhs, rhs); });
        case xir::ArithmeticOp::BINARY_BIT_XOR: return binary([&](auto lhs, auto rhs) noexcept { return builder.CreateXor(lhs, rhs); });
        case xir::ArithmeticOp::BINARY_SHIFT_LEFT: return binary([&](auto lhs, auto rhs) noexcept { return builder.CreateShl(lhs, rhs); });
        case xir::ArithmeticOp::BINARY_SHIFT_RIGHT: return binary([&](auto lhs, auto rhs) noexcept {
            return inst->type()->is_int_or_int_vector() ? builder.CreateAShr(lhs, rhs) : builder.CreateLShr(lhs, rhs);
        });
        case xir::ArithmeticOp::BINARY_ROTATE_LEFT: return binary([&](auto lhs, auto rhs) noexcept {
            auto width = lhs->getType()->getScalarSizeInBits();
            auto mask = llvm::ConstantInt::get(rhs->getType(), width - 1u);
            auto shift = builder.CreateAnd(rhs, mask);
            auto inverse = builder.CreateAnd(builder.CreateSub(llvm::ConstantInt::get(rhs->getType(), width), shift), mask);
            return builder.CreateOr(builder.CreateShl(lhs, shift), builder.CreateLShr(lhs, inverse));
        });
        case xir::ArithmeticOp::BINARY_ROTATE_RIGHT: return binary([&](auto lhs, auto rhs) noexcept {
            auto width = lhs->getType()->getScalarSizeInBits();
            auto mask = llvm::ConstantInt::get(rhs->getType(), width - 1u);
            auto shift = builder.CreateAnd(rhs, mask);
            auto inverse = builder.CreateAnd(builder.CreateSub(llvm::ConstantInt::get(rhs->getType(), width), shift), mask);
            return builder.CreateOr(builder.CreateLShr(lhs, shift), builder.CreateShl(lhs, inverse));
        });
        case xir::ArithmeticOp::BINARY_LESS: return comparison(
            [&](auto lhs, auto rhs) { return builder.CreateICmpSLT(lhs, rhs); },
            [&](auto lhs, auto rhs) { return builder.CreateICmpULT(lhs, rhs); },
            [&](auto lhs, auto rhs) { return builder.CreateFCmpOLT(lhs, rhs); });
        case xir::ArithmeticOp::BINARY_GREATER: return comparison(
            [&](auto lhs, auto rhs) { return builder.CreateICmpSGT(lhs, rhs); },
            [&](auto lhs, auto rhs) { return builder.CreateICmpUGT(lhs, rhs); },
            [&](auto lhs, auto rhs) { return builder.CreateFCmpOGT(lhs, rhs); });
        case xir::ArithmeticOp::BINARY_LESS_EQUAL: return comparison(
            [&](auto lhs, auto rhs) { return builder.CreateICmpSLE(lhs, rhs); },
            [&](auto lhs, auto rhs) { return builder.CreateICmpULE(lhs, rhs); },
            [&](auto lhs, auto rhs) { return builder.CreateFCmpOLE(lhs, rhs); });
        case xir::ArithmeticOp::BINARY_GREATER_EQUAL: return comparison(
            [&](auto lhs, auto rhs) { return builder.CreateICmpSGE(lhs, rhs); },
            [&](auto lhs, auto rhs) { return builder.CreateICmpUGE(lhs, rhs); },
            [&](auto lhs, auto rhs) { return builder.CreateFCmpOGE(lhs, rhs); });
        case xir::ArithmeticOp::BINARY_EQUAL: return comparison(
            [&](auto lhs, auto rhs) { return builder.CreateICmpEQ(lhs, rhs); },
            [&](auto lhs, auto rhs) { return builder.CreateICmpEQ(lhs, rhs); },
            [&](auto lhs, auto rhs) { return builder.CreateFCmpOEQ(lhs, rhs); });
        case xir::ArithmeticOp::BINARY_NOT_EQUAL: return comparison(
            [&](auto lhs, auto rhs) { return builder.CreateICmpNE(lhs, rhs); },
            [&](auto lhs, auto rhs) { return builder.CreateICmpNE(lhs, rhs); },
            [&](auto lhs, auto rhs) { return builder.CreateFCmpUNE(lhs, rhs); });
        case xir::ArithmeticOp::ALL:
        case xir::ArithmeticOp::ANY: return unary([&](auto value) noexcept -> llvm::Value * {
            if (!value->getType()->isVectorTy()) { return value; }
            // LLVM can fold scalar boolean reductions into i3 bitcasts, which
            // AGX cannot legalize. Match metalfe's native AIR reduction ABI.
            auto vector_type = llvm::cast<llvm::FixedVectorType>(value->getType());
            auto name = luisa::format("air.{}.v{}i1",
                                      inst->op() == xir::ArithmeticOp::ALL ? "all" : "any",
                                      vector_type->getNumElements());
            auto function_type = llvm::FunctionType::get(
                builder.getInt1Ty(), {vector_type}, false);
            auto air_function = llvm::cast<llvm::Function>(
                _module.getOrInsertFunction(
                    llvm::StringRef{name.data(), name.size()}, function_type).getCallee());
            air_function->setDoesNotAccessMemory();
            air_function->setDoesNotFreeMemory();
            air_function->setDoesNotThrow();
            air_function->setNoSync();
            air_function->setWillReturn();
            return builder.CreateCall(air_function, {value});
        });
        case xir::ArithmeticOp::SELECT: {
            auto false_value = _value(builder, function, inst->operand(0u));
            auto true_value = _value(builder, function, inst->operand(1u));
            auto condition = _value(builder, function, inst->operand(2u));
            return builder.CreateSelect(condition, true_value, false_value);
        }
        case xir::ArithmeticOp::CLAMP: return ternary([&](auto value, auto low, auto high) noexcept {
            if (inst->type()->is_float_or_float_vector()) {
                return _air_ternary(builder, "clamp", value, low, high);
            }
            auto is_signed = inst->type()->is_int_or_int_vector();
            auto below = is_signed ? builder.CreateICmpSLT(value, low) : builder.CreateICmpULT(value, low);
            auto low_clamped = builder.CreateSelect(below, low, value);
            auto above = is_signed ? builder.CreateICmpSGT(low_clamped, high) : builder.CreateICmpUGT(low_clamped, high);
            return builder.CreateSelect(above, high, low_clamped);
        });
        case xir::ArithmeticOp::SATURATE: return unary([&](auto value) noexcept {
            return _air_unary(builder, "saturate", value);
        });
        case xir::ArithmeticOp::LERP: return ternary([&](auto x, auto y, auto t) noexcept {
            return builder.CreateFAdd(x, builder.CreateFMul(t, builder.CreateFSub(y, x)));
        });
        case xir::ArithmeticOp::SMOOTHSTEP: return ternary([&](auto edge0, auto edge1, auto x) noexcept {
            auto zero = llvm::Constant::getNullValue(x->getType());
            auto one = llvm::ConstantFP::get(x->getType(), 1.0);
            auto t = builder.CreateFDiv(builder.CreateFSub(x, edge0), builder.CreateFSub(edge1, edge0));
            t = _air_ternary(builder, "clamp", t, zero, one);
            auto three = llvm::ConstantFP::get(x->getType(), 3.0);
            auto two = llvm::ConstantFP::get(x->getType(), 2.0);
            return builder.CreateFMul(builder.CreateFMul(t, t), builder.CreateFSub(three, builder.CreateFMul(two, t)));
        });
        case xir::ArithmeticOp::STEP: return binary([&](auto edge, auto x) noexcept {
            auto zero = llvm::Constant::getNullValue(x->getType());
            auto one = llvm::ConstantFP::get(x->getType(), 1.0);
            return builder.CreateSelect(builder.CreateFCmpOGE(x, edge), one, zero);
        });
        case xir::ArithmeticOp::ABS: return unary([&](auto value) noexcept {
            if (inst->type()->is_uint_or_uint_vector()) { return value; }
            if (inst->type()->is_float_or_float_vector()) { return _air_unary(builder, "fabs", value); }
            auto zero = llvm::Constant::getNullValue(value->getType());
            return builder.CreateSelect(builder.CreateICmpSLT(value, zero), builder.CreateNeg(value), value);
        });
        case xir::ArithmeticOp::MIN: return binary([&](auto lhs, auto rhs) noexcept {
            if (inst->type()->is_float_or_float_vector()) {
                return _air_binary(builder, "fmin", lhs, rhs);
            }
            auto condition = inst->type()->is_int_or_int_vector() ?
                                 builder.CreateICmpSLT(lhs, rhs) :
                                 builder.CreateICmpULT(lhs, rhs);
            return builder.CreateSelect(condition, lhs, rhs);
        });
        case xir::ArithmeticOp::MAX: return binary([&](auto lhs, auto rhs) noexcept {
            if (inst->type()->is_float_or_float_vector()) {
                return _air_binary(builder, "fmax", lhs, rhs);
            }
            auto condition = inst->type()->is_int_or_int_vector() ?
                                 builder.CreateICmpSGT(lhs, rhs) :
                                 builder.CreateICmpUGT(lhs, rhs);
            return builder.CreateSelect(condition, lhs, rhs);
        });
        case xir::ArithmeticOp::CLZ:
            return unary([&](auto value) noexcept { return _air_integer_call(builder, "clz", value, false); });
        case xir::ArithmeticOp::CTZ:
            return unary([&](auto value) noexcept { return _air_integer_call(builder, "ctz", value, false); });
        case xir::ArithmeticOp::POPCOUNT:
            return unary([&](auto value) noexcept { return _air_integer_call(builder, "popcount", value, false); });
        case xir::ArithmeticOp::REVERSE:
            return unary([&](auto value) noexcept { return _air_integer_call(builder, "reverse_bits", value, false); });
        case xir::ArithmeticOp::ISINF: return unary([&](auto value) noexcept {
            auto scalar_bits = value->getType()->getScalarSizeInBits();
            auto integer = static_cast<llvm::Type *>(llvm::Type::getIntNTy(_context, scalar_bits));
            if (value->getType()->isVectorTy()) {
                integer = llvm::FixedVectorType::get(integer, llvm::cast<llvm::FixedVectorType>(value->getType())->getNumElements());
            }
            auto mask_value = scalar_bits == 16u ? 0x7fffu : 0x7fffffffu;
            auto test_value = scalar_bits == 16u ? 0x7c00u : 0x7f800000u;
            auto mask = llvm::ConstantInt::get(integer, mask_value);
            auto test = llvm::ConstantInt::get(integer, test_value);
            return builder.CreateICmpEQ(builder.CreateAnd(builder.CreateBitCast(value, integer), mask), test);
        });
        case xir::ArithmeticOp::ISNAN: return unary([&](auto value) noexcept {
            auto scalar_bits = value->getType()->getScalarSizeInBits();
            auto integer = static_cast<llvm::Type *>(llvm::Type::getIntNTy(_context, scalar_bits));
            if (value->getType()->isVectorTy()) {
                integer = llvm::FixedVectorType::get(integer, llvm::cast<llvm::FixedVectorType>(value->getType())->getNumElements());
            }
            auto mask_value = scalar_bits == 16u ? 0x7fffu : 0x7fffffffu;
            auto test_value = scalar_bits == 16u ? 0x7c00u : 0x7f800000u;
            auto mask = llvm::ConstantInt::get(integer, mask_value);
            auto test = llvm::ConstantInt::get(integer, test_value);
            return builder.CreateICmpUGT(builder.CreateAnd(builder.CreateBitCast(value, integer), mask), test);
        });
        case xir::ArithmeticOp::ACOS: return unary([&](auto value) { return _air_unary(builder, "acos", value); });
        case xir::ArithmeticOp::ACOSH: return unary([&](auto value) { return _air_unary(builder, "acosh", value); });
        case xir::ArithmeticOp::ASIN: return unary([&](auto value) { return _air_unary(builder, "asin", value); });
        case xir::ArithmeticOp::ASINH: return unary([&](auto value) { return _air_unary(builder, "asinh", value); });
        case xir::ArithmeticOp::ATAN: return unary([&](auto value) { return _air_unary(builder, "atan", value); });
        case xir::ArithmeticOp::ATAN2: return binary([&](auto lhs, auto rhs) { return _air_binary(builder, "atan2", lhs, rhs); });
        case xir::ArithmeticOp::ATANH: return unary([&](auto value) { return _air_unary(builder, "atanh", value); });
        case xir::ArithmeticOp::COS: return unary([&](auto value) { return _air_unary(builder, "cos", value); });
        case xir::ArithmeticOp::COSH: return unary([&](auto value) { return _air_unary(builder, "cosh", value); });
        case xir::ArithmeticOp::SIN: return unary([&](auto value) { return _air_unary(builder, "sin", value); });
        case xir::ArithmeticOp::SINH: return unary([&](auto value) { return _air_unary(builder, "sinh", value); });
        case xir::ArithmeticOp::TAN: return unary([&](auto value) { return _air_unary(builder, "tan", value); });
        case xir::ArithmeticOp::TANH: return unary([&](auto value) { return _air_unary(builder, "tanh", value); });
        case xir::ArithmeticOp::EXP: return unary([&](auto value) { return _air_unary(builder, "exp", value); });
        case xir::ArithmeticOp::EXP2: return unary([&](auto value) { return _air_unary(builder, "exp2", value); });
        case xir::ArithmeticOp::EXP10: return unary([&](auto value) { return _air_unary(builder, "exp10", value); });
        case xir::ArithmeticOp::ERF: return unary([&](auto value) { return _air_unary(builder, "erf", value); });
        case xir::ArithmeticOp::ERFC: return unary([&](auto value) { return _air_unary(builder, "erfc", value); });
        case xir::ArithmeticOp::EXPM1: return unary([&](auto value) { return _air_unary(builder, "expm1", value); });
        case xir::ArithmeticOp::LOG: return unary([&](auto value) { return _air_unary(builder, "log", value); });
        case xir::ArithmeticOp::LOG2: return unary([&](auto value) { return _air_unary(builder, "log2", value); });
        case xir::ArithmeticOp::LOG10: return unary([&](auto value) { return _air_unary(builder, "log10", value); });
        case xir::ArithmeticOp::POW: return binary([&](auto lhs, auto rhs) { return _air_binary(builder, "pow", lhs, rhs); });
        case xir::ArithmeticOp::POW_INT: {
            auto base = _value(builder, function, inst->operand(0u));
            auto exponent = _value(builder, function, inst->operand(1u));
            LUISA_DEBUG_ASSERT(base->getType()->isFPOrFPVectorTy() &&
                               exponent->getType()->isIntOrIntVectorTy());
            auto exponent_as_float = builder.CreateSIToFP(exponent, base->getType());
            return _air_binary(builder, "pow", base, exponent_as_float);
        }
        case xir::ArithmeticOp::LDEXP: {
            auto base = _value(builder, function, inst->operand(0u));
            auto exponent = _value(builder, function, inst->operand(1u));
            LUISA_DEBUG_ASSERT(base->getType()->isFPOrFPVectorTy() &&
                               exponent->getType()->isIntOrIntVectorTy());
            // air.ldexp is `float (float, i32)` / `half (half, i32)`: its exponent
            // parameter does *not* have the mantissa type, so this call must not go
            // through _air_binary/_air_scalar_call. Those declare every parameter
            // with the type of the first argument, which would produce
            // `float (float, float)` for a `{float, i32}` call site and hence
            // invalid IR (CallInst assertions / verifier rejection). Build the
            // declaration from the real signature instead, per lane; there is no
            // fast-math AIR variant of ldexp, so the plain symbol is used.
            //
            // Signedness contract: LLVM IR drops the XIR signedness, so it is
            // recovered from the operand type. A signed exponent is sign-extended
            // and clamped to [INT32_MIN, INT32_MAX]; an unsigned exponent is
            // zero-extended and clamped to INT32_MAX only, because an unsigned
            // exponent above INT32_MAX already saturates the mantissa to zero or
            // +inf, so keeping its value stays exact while sign-extending it would
            // turn a huge exponent into a hugely negative one.
            auto exponent_xir_type = inst->operand(1u)->type();
            auto exponent_element_type = exponent_xir_type->is_vector() ?
                                             exponent_xir_type->element() :
                                             exponent_xir_type;
            auto exponent_is_signed = exponent_element_type->is_int();
            auto *i32 = builder.getInt32Ty();
            auto narrow_exponent = [&](llvm::Value *n) noexcept -> llvm::Value * {
                    auto *source_type = n->getType();
                    auto source_bits = source_type->getIntegerBitWidth();
                    if (source_bits < 32u) {
                        return exponent_is_signed ? builder.CreateSExt(n, i32) :
                                                    builder.CreateZExt(n, i32);
                    }
                    if (exponent_is_signed && source_bits == 32u) { return n; }
                    // Clamp in the source type *before* truncating: truncation alone
                    // would wrap an out-of-range exponent back into the i32 range.
                    auto *high = llvm::ConstantInt::get(source_type, 0x7fffffffull);
                    n = builder.CreateSelect(exponent_is_signed ?
                                                 builder.CreateICmpSGT(n, high) :
                                                 builder.CreateICmpUGT(n, high),
                                             high, n);
                    if (exponent_is_signed) {
                        auto *low = llvm::ConstantInt::getSigned(source_type, -2147483648ll);
                        n = builder.CreateSelect(builder.CreateICmpSLT(n, low), low, n);
                    }
                    return source_bits == 32u ? n : builder.CreateTrunc(n, i32);
                };
            auto call_scalar = [&](llvm::Value *x, llvm::Value *n) noexcept -> llvm::Value * {
                auto *scalar_type = x->getType();
                LUISA_DEBUG_ASSERT(scalar_type->isHalfTy() || scalar_type->isFloatTy(),
                                   "air.ldexp requires an f16 or f32 mantissa.");
                auto *function_type = llvm::FunctionType::get(
                    scalar_type, {scalar_type, i32}, false);
                auto callee = _module.getOrInsertFunction(
                    scalar_type->isHalfTy() ? "air.ldexp.f16" : "air.ldexp.f32",
                    function_type);
                if (auto *function = llvm::dyn_cast<llvm::Function>(callee.getCallee())) {
                    function->setMustProgress();
                    function->setDoesNotFreeMemory();
                    function->setNoSync();
                    function->setDoesNotThrow();
                    function->setWillReturn();
                    function->setDoesNotAccessMemory();
                    function->setSpeculatable();
                }
                return builder.CreateCall(callee, {x, narrow_exponent(n)});
            };
            if (auto *vector_type = llvm::dyn_cast<llvm::FixedVectorType>(
                    base->getType())) {
                auto exponent_is_vector = exponent->getType()->isVectorTy();
                auto *result = static_cast<llvm::Value *>(
                    llvm::PoisonValue::get(vector_type));
                for (auto i = 0u; i < vector_type->getNumElements(); i++) {
                    auto *lane_exponent = exponent_is_vector ?
                                              builder.CreateExtractElement(exponent, i) :
                                              exponent;
                    result = builder.CreateInsertElement(
                        result,
                        call_scalar(builder.CreateExtractElement(base, i), lane_exponent),
                        i);
                }
                return result;
            }
            return call_scalar(base, exponent);
        }
        case xir::ArithmeticOp::SQRT: return unary([&](auto value) { return _air_unary(builder, "sqrt", value); });
        case xir::ArithmeticOp::RSQRT: return unary([&](auto value) { return _air_unary(builder, "rsqrt", value); });
        case xir::ArithmeticOp::CEIL: return unary([&](auto value) { return _air_unary(builder, "ceil", value); });
        case xir::ArithmeticOp::FLOOR: return unary([&](auto value) { return _air_unary(builder, "floor", value); });
        case xir::ArithmeticOp::FRACT: return unary([&](auto value) { return _air_unary(builder, "fract", value); });
        case xir::ArithmeticOp::TRUNC: return unary([&](auto value) { return _air_unary(builder, "trunc", value); });
        case xir::ArithmeticOp::ROUND: return unary([&](auto value) { return _air_unary(builder, "round", value); });
        case xir::ArithmeticOp::RINT: return unary([&](auto value) { return _air_unary(builder, "rint", value); });
        case xir::ArithmeticOp::FMA: return ternary([&](auto a, auto b, auto c) { return _air_ternary(builder, "fma", a, b, c); });
        case xir::ArithmeticOp::COPYSIGN: return binary([&](auto lhs, auto rhs) noexcept {
            auto scalar_bits = lhs->getType()->getScalarSizeInBits();
            auto integer = static_cast<llvm::Type *>(llvm::Type::getIntNTy(_context, scalar_bits));
            if (lhs->getType()->isVectorTy()) {
                integer = llvm::FixedVectorType::get(integer, llvm::cast<llvm::FixedVectorType>(lhs->getType())->getNumElements());
            }
            auto sign_value = scalar_bits == 16u ? uint64_t{0x8000u} : uint64_t{0x80000000u};
            auto magnitude_value = scalar_bits == 16u ? uint64_t{0x7fffu} : uint64_t{0x7fffffffu};
            auto sign_mask = llvm::ConstantInt::get(integer, sign_value);
            auto magnitude_mask = llvm::ConstantInt::get(integer, magnitude_value);
            auto lhs_bits = builder.CreateBitCast(lhs, integer);
            auto rhs_bits = builder.CreateBitCast(rhs, integer);
            auto bits = builder.CreateOr(builder.CreateAnd(lhs_bits, magnitude_mask), builder.CreateAnd(rhs_bits, sign_mask));
            return builder.CreateBitCast(bits, lhs->getType());
        });
        case xir::ArithmeticOp::CROSS: return binary([&](auto lhs, auto rhs) noexcept {
            LUISA_ASSERT(lhs->getType()->isVectorTy() &&
                             llvm::cast<llvm::FixedVectorType>(lhs->getType())->getNumElements() == 3u,
                         "Cross product requires three-component vectors.");
            auto poison = llvm::PoisonValue::get(lhs->getType());
            auto lhs_yzx = builder.CreateShuffleVector(lhs, poison, {1, 2, 0});
            auto rhs_zxy = builder.CreateShuffleVector(rhs, poison, {2, 0, 1});
            auto rhs_yzx = builder.CreateShuffleVector(rhs, poison, {1, 2, 0});
            auto lhs_zxy = builder.CreateShuffleVector(lhs, poison, {2, 0, 1});
            return builder.CreateFSub(builder.CreateFMul(lhs_yzx, rhs_zxy),
                                      builder.CreateFMul(rhs_yzx, lhs_zxy));
        });
        case xir::ArithmeticOp::DOT: return binary([&](auto lhs, auto rhs) noexcept {
            auto product = builder.CreateFMul(lhs, rhs);
            auto zero = llvm::ConstantFP::get(product->getType()->getScalarType(), 0.0);
            return reduce(product, zero, [&](auto a, auto b) { return builder.CreateFAdd(a, b); });
        });
        case xir::ArithmeticOp::LENGTH_SQUARED: return unary([&](auto value) noexcept {
            auto product = builder.CreateFMul(value, value);
            auto zero = llvm::ConstantFP::get(product->getType()->getScalarType(), 0.0);
            return reduce(product, zero, [&](auto a, auto b) { return builder.CreateFAdd(a, b); });
        });
        case xir::ArithmeticOp::LENGTH: return unary([&](auto value) noexcept {
            auto product = builder.CreateFMul(value, value);
            auto zero = llvm::ConstantFP::get(product->getType()->getScalarType(), 0.0);
            auto squared = reduce(product, zero, [&](auto a, auto b) { return builder.CreateFAdd(a, b); });
            return _air_unary(builder, "sqrt", squared);
        });
        case xir::ArithmeticOp::NORMALIZE: return unary([&](auto value) noexcept {
            auto product = builder.CreateFMul(value, value);
            auto zero = llvm::ConstantFP::get(product->getType()->getScalarType(), 0.0);
            auto squared = reduce(product, zero, [&](auto a, auto b) { return builder.CreateFAdd(a, b); });
            auto inverse_length = _air_unary(builder, "rsqrt", squared);
            return builder.CreateFMul(value, builder.CreateVectorSplat(inst->type()->dimension(), inverse_length));
        });
        case xir::ArithmeticOp::FACEFORWARD: return ternary([&](auto n, auto i, auto n_ref) noexcept {
            auto product = builder.CreateFMul(i, n_ref);
            auto dot = reduce(product, llvm::ConstantFP::get(product->getType()->getScalarType(), 0.0),
                              [&](auto a, auto b) { return builder.CreateFAdd(a, b); });
            auto negative = builder.CreateFCmpOLT(dot, llvm::ConstantFP::get(dot->getType(), 0.0));
            return builder.CreateSelect(negative, n, builder.CreateFNeg(n));
        });
        case xir::ArithmeticOp::REFLECT: return binary([&](auto incident, auto normal) noexcept {
            auto product = builder.CreateFMul(normal, incident);
            auto dot = reduce(product, llvm::ConstantFP::get(product->getType()->getScalarType(), 0.0),
                              [&](auto a, auto b) { return builder.CreateFAdd(a, b); });
            auto scale = builder.CreateFMul(llvm::ConstantFP::get(dot->getType(), 2.0), dot);
            return builder.CreateFSub(incident,
                                      builder.CreateFMul(normal, builder.CreateVectorSplat(inst->type()->dimension(), scale)));
        });
        case xir::ArithmeticOp::REDUCE_SUM: return unary([&](auto value) noexcept {
            auto scalar_type = value->getType()->getScalarType();
            if (scalar_type->isIntegerTy()) {
                return reduce(value, llvm::ConstantInt::get(scalar_type, 0u),
                              [&](auto a, auto b) { return builder.CreateAdd(a, b); });
            }
            return reduce(value, llvm::ConstantFP::get(scalar_type, 0.0),
                          [&](auto a, auto b) { return builder.CreateFAdd(a, b); });
        });
        case xir::ArithmeticOp::REDUCE_PRODUCT: return unary([&](auto value) noexcept {
            auto scalar_type = value->getType()->getScalarType();
            if (scalar_type->isIntegerTy()) {
                return reduce(value, llvm::ConstantInt::get(scalar_type, 1u),
                              [&](auto a, auto b) { return builder.CreateMul(a, b); });
            }
            return reduce(value, llvm::ConstantFP::get(scalar_type, 1.0),
                          [&](auto a, auto b) { return builder.CreateFMul(a, b); });
        });
        case xir::ArithmeticOp::REDUCE_MIN: return unary([&](auto value) noexcept {
            auto result = static_cast<llvm::Value *>(builder.CreateExtractElement(value, uint64_t{0u}));
            auto dimension = llvm::cast<llvm::FixedVectorType>(value->getType())->getNumElements();
            auto element_type = inst->operand(0u)->type()->element();
            for (auto i = 1u; i < dimension; i++) {
                auto element = builder.CreateExtractElement(value, i);
                if (element_type->is_float()) {
                    result = _air_binary(builder, "fmin", result, element);
                } else {
                    auto condition = element_type->is_int() ?
                                         builder.CreateICmpSLT(element, result) :
                                         builder.CreateICmpULT(element, result);
                    result = builder.CreateSelect(condition, element, result);
                }
            }
            return result;
        });
        case xir::ArithmeticOp::REDUCE_MAX: return unary([&](auto value) noexcept {
            auto result = static_cast<llvm::Value *>(builder.CreateExtractElement(value, uint64_t{0u}));
            auto dimension = llvm::cast<llvm::FixedVectorType>(value->getType())->getNumElements();
            auto element_type = inst->operand(0u)->type()->element();
            for (auto i = 1u; i < dimension; i++) {
                auto element = builder.CreateExtractElement(value, i);
                if (element_type->is_float()) {
                    result = _air_binary(builder, "fmax", result, element);
                } else {
                    auto condition = element_type->is_int() ?
                                         builder.CreateICmpSGT(element, result) :
                                         builder.CreateICmpUGT(element, result);
                    result = builder.CreateSelect(condition, element, result);
                }
            }
            return result;
        });
        case xir::ArithmeticOp::OUTER_PRODUCT: return binary([&](auto lhs, auto rhs) noexcept {
            auto dimension = inst->type()->dimension();
            if (lhs->getType()->isVectorTy()) {
                auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(_type(inst->type())->reg_type));
                for (auto i = 0u; i < dimension; i++) {
                    auto scale = builder.CreateVectorSplat(dimension, builder.CreateExtractElement(rhs, i));
                    result = builder.CreateInsertValue(result, builder.CreateFMul(lhs, scale), i);
                }
                return result;
            }
            auto rhs_transposed = static_cast<llvm::Value *>(llvm::PoisonValue::get(rhs->getType()));
            for (auto i = 0u; i < dimension; i++) {
                auto column = static_cast<llvm::Value *>(llvm::PoisonValue::get(rhs->getType()->getArrayElementType()));
                for (auto j = 0u; j < dimension; j++) {
                    column = builder.CreateInsertElement(
                        column, builder.CreateExtractElement(builder.CreateExtractValue(rhs, j), i), j);
                }
                rhs_transposed = builder.CreateInsertValue(rhs_transposed, column, i);
            }
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(lhs->getType()));
            for (auto column_index = 0u; column_index < dimension; column_index++) {
                auto rhs_column = builder.CreateExtractValue(rhs_transposed, column_index);
                auto result_column = static_cast<llvm::Value *>(
                    llvm::Constant::getNullValue(rhs_column->getType()));
                for (auto i = 0u; i < dimension; i++) {
                    auto lhs_column = builder.CreateExtractValue(lhs, i);
                    auto scale = builder.CreateVectorSplat(dimension, builder.CreateExtractElement(rhs_column, i));
                    result_column = builder.CreateFAdd(result_column, builder.CreateFMul(lhs_column, scale));
                }
                result = builder.CreateInsertValue(result, result_column, column_index);
            }
            return result;
        });
        case xir::ArithmeticOp::AGGREGATE: {
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(_type(inst->type())->reg_type));
            for (auto i = 0u; i < inst->operand_count(); i++) {
                auto element = _value(builder, function, inst->operand(i));
                result = inst->type()->is_vector() ? builder.CreateInsertElement(result, element, i) : builder.CreateInsertValue(result, element, i);
            }
            return result;
        }
        case xir::ArithmeticOp::SHUFFLE: {
            auto source = _value(builder, function, inst->operand(0u));
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(_type(inst->type())->reg_type));
            for (auto i = 1u; i < inst->operand_count(); i++) {
                auto index = _value(builder, function, inst->operand(i));
                result = builder.CreateInsertElement(result, builder.CreateExtractElement(source, index), i - 1u);
            }
            return result;
        }
        case xir::ArithmeticOp::EXTRACT: {
            auto source = _value(builder, function, inst->operand(0u));
            auto indices = inst->operand_uses().subspan(1u);
            if (source->getType()->isVectorTy()) {
                return builder.CreateExtractElement(source, _value(builder, function, indices.front()->value()));
            }
            if (indices.size() == 1u && indices.front()->value()->isa<xir::Constant>()) {
                auto index = metal_constant_index(static_cast<const xir::Constant *>(indices.front()->value()));
                return builder.CreateExtractValue(source, static_cast<unsigned>(index));
            }
            auto memory = _reg_to_mem(builder, source, inst->operand(0u)->type());
            auto temporary = _temporary(function, memory->getType(), inst->operand(0u)->type()->alignment());
            builder.CreateStore(memory, temporary);
            auto [pointer, type] = _access_chain(builder, function, temporary, inst->operand(0u)->type(), indices);
            LUISA_ASSERT(type == inst->type(), "XIR extract type mismatch.");
            return _load(builder, pointer, type);
        }
        case xir::ArithmeticOp::INSERT: {
            auto source = _value(builder, function, inst->operand(0u));
            auto value = _value(builder, function, inst->operand(1u));
            auto indices = inst->operand_uses().subspan(2u);
            if (source->getType()->isVectorTy()) {
                return builder.CreateInsertElement(source, value, _value(builder, function, indices.front()->value()));
            }
            if (indices.size() == 1u && indices.front()->value()->isa<xir::Constant>()) {
                auto index = metal_constant_index(static_cast<const xir::Constant *>(indices.front()->value()));
                return builder.CreateInsertValue(source, value, static_cast<unsigned>(index));
            }
            auto memory = _reg_to_mem(builder, source, inst->type());
            auto temporary = _temporary(function, memory->getType(), inst->type()->alignment());
            builder.CreateStore(memory, temporary);
            auto [pointer, type] = _access_chain(builder, function, temporary, inst->type(), indices);
            _store(builder, pointer, value, type);
            return _load(builder, temporary, inst->type());
        }
        case xir::ArithmeticOp::MATRIX_COMP_NEG: return unary([&](auto matrix) noexcept {
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(matrix->getType()));
            for (auto i = 0u; i < inst->type()->dimension(); i++) { result = builder.CreateInsertValue(result, builder.CreateFNeg(builder.CreateExtractValue(matrix, i)), i); }
            return result;
        });
        case xir::ArithmeticOp::MATRIX_COMP_ADD: return component_binary([&](auto lhs, auto rhs) { return builder.CreateFAdd(lhs, rhs); });
        case xir::ArithmeticOp::MATRIX_COMP_SUB: return component_binary([&](auto lhs, auto rhs) { return builder.CreateFSub(lhs, rhs); });
        case xir::ArithmeticOp::MATRIX_COMP_MUL: return component_binary([&](auto lhs, auto rhs) { return builder.CreateFMul(lhs, rhs); });
        case xir::ArithmeticOp::MATRIX_COMP_DIV: return component_binary([&](auto lhs, auto rhs) { return builder.CreateFDiv(lhs, rhs); });
        case xir::ArithmeticOp::MATRIX_LINALG_MUL: return binary([&](auto lhs, auto rhs) noexcept {
            auto dimension = inst->operand(0u)->type()->dimension();
            auto multiply_vector = [&](llvm::Value *vector) noexcept {
                auto result = static_cast<llvm::Value *>(llvm::Constant::getNullValue(vector->getType()));
                for (auto i = 0u; i < dimension; i++) {
                    auto lhs_column = builder.CreateExtractValue(lhs, i);
                    auto scale = builder.CreateVectorSplat(dimension, builder.CreateExtractElement(vector, i));
                    result = builder.CreateFAdd(result, builder.CreateFMul(lhs_column, scale));
                }
                return result;
            };
            if (rhs->getType()->isVectorTy()) { return multiply_vector(rhs); }
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(lhs->getType()));
            for (auto i = 0u; i < dimension; i++) {
                result = builder.CreateInsertValue(result, multiply_vector(builder.CreateExtractValue(rhs, i)), i);
            }
            return result;
        });
        case xir::ArithmeticOp::MATRIX_DETERMINANT: return unary([&](auto matrix) noexcept {
            return metal_matrix_determinant(builder, matrix);
        });
        case xir::ArithmeticOp::MATRIX_TRANSPOSE: return unary([&](auto matrix) noexcept {
            auto dimension = inst->type()->dimension();
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(matrix->getType()));
            for (auto i = 0u; i < dimension; i++) {
                auto column = static_cast<llvm::Value *>(llvm::PoisonValue::get(matrix->getType()->getArrayElementType()));
                for (auto j = 0u; j < dimension; j++) {
                    column = builder.CreateInsertElement(
                        column, builder.CreateExtractElement(builder.CreateExtractValue(matrix, j), i), j);
                }
                result = builder.CreateInsertValue(result, column, i);
            }
            return result;
        });
        case xir::ArithmeticOp::MATRIX_INVERSE: return unary([&](auto matrix) noexcept {
            auto dimension = inst->type()->dimension();
            auto determinant = metal_matrix_determinant(builder, matrix);
            auto inverse_determinant = builder.CreateFDiv(
                llvm::ConstantFP::get(determinant->getType(), 1.0), determinant);
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(matrix->getType()));
            for (auto column_index = 0u; column_index < dimension; column_index++) {
                auto column = static_cast<llvm::Value *>(
                    llvm::PoisonValue::get(matrix->getType()->getArrayElementType()));
                for (auto row_index = 0u; row_index < dimension; row_index++) {
                    auto cofactor = metal_matrix_determinant(
                        builder, matrix, static_cast<int>(column_index), static_cast<int>(row_index));
                    if ((column_index + row_index) % 2u != 0u) {
                        cofactor = builder.CreateFNeg(cofactor);
                    }
                    column = builder.CreateInsertElement(
                        column, builder.CreateFMul(cofactor, inverse_determinant), row_index);
                }
                result = builder.CreateInsertValue(result, column, column_index);
            }
            return result;
        });
        default: _unsupported_instruction(inst);
    }
}

void MetalCodegenLLVMImpl::_translate_print(
    IB &builder, FunctionContext &function,
    const xir::PrintInst *inst) noexcept {
    auto token_iter = _print_tokens.find(inst);
    LUISA_ASSERT(token_iter != _print_tokens.end() &&
                     token_iter->second < _print_formats.size(),
                 "Metal shader-log instruction has no format token.");
    auto token = token_iter->second;
    auto &format = _print_formats[token];
    if (format.native_format_pointer == nullptr) {
        auto name = luisa::format("luisa.shader.log.format.{}", token);
        format.native_format_pointer = _constant_string(
            format.native_format, name);
    }
    llvm::SmallVector<llvm::Value *> arguments;
    arguments.reserve(inst->operand_count() + 2u);
    arguments.emplace_back(format.native_format_pointer);
    arguments.emplace_back(nullptr);
    auto argument_size = static_cast<size_t>(0u);
    for (auto i = 0u; i < inst->operand_count(); i++) {
        auto operand = inst->operand(i);
        _append_shader_log_arguments(
            builder, _value(builder, function, operand), operand->type(),
            arguments, argument_size);
    }
    arguments[1u] = builder.getInt64(argument_size);
    auto call = builder.CreateCall(_shader_log(), arguments);
    call->setConvergent();
}

}// namespace luisa::compute::metal::detail
