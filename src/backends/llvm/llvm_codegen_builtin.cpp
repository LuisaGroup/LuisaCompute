//
// Created by Mike Smith on 2022/5/23.
//

#include <dsl/sugar.h>
#include <rtx/ray.h>
#include <rtx/hit.h>
#include <backends/llvm/llvm_codegen.h>

namespace luisa::compute::llvm {

[[nodiscard]] static Function _float2x2_inverse() noexcept {
    static Callable inverse = [](Float2x2 m) noexcept {
        auto inv_det = 1.0f / (m[0][0] * m[1][1] - m[1][0] * m[0][1]);
        return inv_det * make_float2x2(m[1][1], -m[0][1], -m[1][0], +m[0][0]);
    };
    return inverse.function();
}

[[nodiscard]] static Function _float3x3_inverse() noexcept {
    static Callable inverse = [](Float3x3 m) noexcept {
        auto inv_det = 1.0f /
                       (m[0].x * (m[1].y * m[2].z - m[2].y * m[1].z) -
                        m[1].x * (m[0].y * m[2].z - m[2].y * m[0].z) +
                        m[2].x * (m[0].y * m[1].z - m[1].y * m[0].z));
        auto mm = make_float3x3(
            m[1].y * m[2].z - m[2].y * m[1].z,
            m[2].y * m[0].z - m[0].y * m[2].z,
            m[0].y * m[1].z - m[1].y * m[0].z,
            m[2].x * m[1].z - m[1].x * m[2].z,
            m[0].x * m[2].z - m[2].x * m[0].z,
            m[1].x * m[0].z - m[0].x * m[1].z,
            m[1].x * m[2].y - m[2].x * m[1].y,
            m[2].x * m[0].y - m[0].x * m[2].y,
            m[0].x * m[1].y - m[1].x * m[0].y);
        return inv_det * mm;
    };
    return inverse.function();
}

[[nodiscard]] static Function _float4x4_inverse() noexcept {
    static Callable inverse = [](Float4x4 m) noexcept {
        auto coef00 = m[2].z * m[3].w - m[3].z * m[2].w;
        auto coef02 = m[1].z * m[3].w - m[3].z * m[1].w;
        auto coef03 = m[1].z * m[2].w - m[2].z * m[1].w;
        auto coef04 = m[2].y * m[3].w - m[3].y * m[2].w;
        auto coef06 = m[1].y * m[3].w - m[3].y * m[1].w;
        auto coef07 = m[1].y * m[2].w - m[2].y * m[1].w;
        auto coef08 = m[2].y * m[3].z - m[3].y * m[2].z;
        auto coef10 = m[1].y * m[3].z - m[3].y * m[1].z;
        auto coef11 = m[1].y * m[2].z - m[2].y * m[1].z;
        auto coef12 = m[2].x * m[3].w - m[3].x * m[2].w;
        auto coef14 = m[1].x * m[3].w - m[3].x * m[1].w;
        auto coef15 = m[1].x * m[2].w - m[2].x * m[1].w;
        auto coef16 = m[2].x * m[3].z - m[3].x * m[2].z;
        auto coef18 = m[1].x * m[3].z - m[3].x * m[1].z;
        auto coef19 = m[1].x * m[2].z - m[2].x * m[1].z;
        auto coef20 = m[2].x * m[3].y - m[3].x * m[2].y;
        auto coef22 = m[1].x * m[3].y - m[3].x * m[1].y;
        auto coef23 = m[1].x * m[2].y - m[2].x * m[1].y;
        auto fac0 = make_float4(coef00, coef00, coef02, coef03);
        auto fac1 = make_float4(coef04, coef04, coef06, coef07);
        auto fac2 = make_float4(coef08, coef08, coef10, coef11);
        auto fac3 = make_float4(coef12, coef12, coef14, coef15);
        auto fac4 = make_float4(coef16, coef16, coef18, coef19);
        auto fac5 = make_float4(coef20, coef20, coef22, coef23);
        auto Vec0 = make_float4(m[1].x, m[0].x, m[0].x, m[0].x);
        auto Vec1 = make_float4(m[1].y, m[0].y, m[0].y, m[0].y);
        auto Vec2 = make_float4(m[1].z, m[0].z, m[0].z, m[0].z);
        auto Vec3 = make_float4(m[1].w, m[0].w, m[0].w, m[0].w);
        auto inv0 = Vec1 * fac0 - Vec2 * fac1 + Vec3 * fac2;
        auto inv1 = Vec0 * fac0 - Vec2 * fac3 + Vec3 * fac4;
        auto inv2 = Vec0 * fac1 - Vec1 * fac3 + Vec3 * fac5;
        auto inv3 = Vec0 * fac2 - Vec1 * fac4 + Vec2 * fac5;
        auto sign_a = make_float4(+1.0f, -1.0f, +1.0f, -1.0f);
        auto sign_b = make_float4(-1.0f, +1.0f, -1.0f, +1.0f);
        auto inv_0 = inv0 * sign_a;
        auto inv_1 = inv1 * sign_b;
        auto inv_2 = inv2 * sign_a;
        auto inv_3 = inv3 * sign_b;
        auto dot0 = m[0] * make_float4(inv_0.x, inv_1.x, inv_2.x, inv_3.x);
        auto dot1 = dot0.x + dot0.y + dot0.z + dot0.w;
        auto inv_det = 1.0f / dot1;
        return inv_det * make_float4x4(inv_0, inv_1, inv_2, inv_3);
    };
    return inverse.function();
}

[[nodiscard]] static Function _float2x2_det() noexcept {
    static Callable inverse = [](Float2x2 m) noexcept {
        return m[0][0] * m[1][1] - m[1][0] * m[0][1];
    };
    return inverse.function();
}

[[nodiscard]] static Function _float3x3_det() noexcept {
    static Callable inverse = [](Float3x3 m) noexcept {
        return m[0].x * (m[1].y * m[2].z - m[2].y * m[1].z) -
               m[1].x * (m[0].y * m[2].z - m[2].y * m[0].z) +
               m[2].x * (m[0].y * m[1].z - m[1].y * m[0].z);
    };
    return inverse.function();
}

[[nodiscard]] static Function _float4x4_det() noexcept {
    static Callable inverse = [](Float4x4 m) noexcept {
        auto coef00 = m[2].z * m[3].w - m[3].z * m[2].w;
        auto coef02 = m[1].z * m[3].w - m[3].z * m[1].w;
        auto coef03 = m[1].z * m[2].w - m[2].z * m[1].w;
        auto coef04 = m[2].y * m[3].w - m[3].y * m[2].w;
        auto coef06 = m[1].y * m[3].w - m[3].y * m[1].w;
        auto coef07 = m[1].y * m[2].w - m[2].y * m[1].w;
        auto coef08 = m[2].y * m[3].z - m[3].y * m[2].z;
        auto coef10 = m[1].y * m[3].z - m[3].y * m[1].z;
        auto coef11 = m[1].y * m[2].z - m[2].y * m[1].z;
        auto coef12 = m[2].x * m[3].w - m[3].x * m[2].w;
        auto coef14 = m[1].x * m[3].w - m[3].x * m[1].w;
        auto coef15 = m[1].x * m[2].w - m[2].x * m[1].w;
        auto coef16 = m[2].x * m[3].z - m[3].x * m[2].z;
        auto coef18 = m[1].x * m[3].z - m[3].x * m[1].z;
        auto coef19 = m[1].x * m[2].z - m[2].x * m[1].z;
        auto coef20 = m[2].x * m[3].y - m[3].x * m[2].y;
        auto coef22 = m[1].x * m[3].y - m[3].x * m[1].y;
        auto coef23 = m[1].x * m[2].y - m[2].x * m[1].y;
        auto fac0 = make_float4(coef00, coef00, coef02, coef03);
        auto fac1 = make_float4(coef04, coef04, coef06, coef07);
        auto fac2 = make_float4(coef08, coef08, coef10, coef11);
        auto fac3 = make_float4(coef12, coef12, coef14, coef15);
        auto fac4 = make_float4(coef16, coef16, coef18, coef19);
        auto fac5 = make_float4(coef20, coef20, coef22, coef23);
        auto Vec0 = make_float4(m[1].x, m[0].x, m[0].x, m[0].x);
        auto Vec1 = make_float4(m[1].y, m[0].y, m[0].y, m[0].y);
        auto Vec2 = make_float4(m[1].z, m[0].z, m[0].z, m[0].z);
        auto Vec3 = make_float4(m[1].w, m[0].w, m[0].w, m[0].w);
        auto inv0 = Vec1 * fac0 - Vec2 * fac1 + Vec3 * fac2;
        auto inv1 = Vec0 * fac0 - Vec2 * fac3 + Vec3 * fac4;
        auto inv2 = Vec0 * fac1 - Vec1 * fac3 + Vec3 * fac5;
        auto inv3 = Vec0 * fac2 - Vec1 * fac4 + Vec2 * fac5;
        auto sign_a = make_float4(+1.0f, -1.0f, +1.0f, -1.0f);
        auto sign_b = make_float4(-1.0f, +1.0f, -1.0f, +1.0f);
        auto inv_0 = inv0 * sign_a;
        auto inv_1 = inv1 * sign_b;
        auto inv_2 = inv2 * sign_a;
        auto inv_3 = inv3 * sign_b;
        auto dot0 = m[0] * make_float4(inv_0.x, inv_1.x, inv_2.x, inv_3.x);
        return dot0.x + dot0.y + dot0.z + dot0.w;
    };
    return inverse.function();
}

[[nodiscard]] static Function _float2x2_transpose() noexcept {
    static Callable inverse = [](Float2x2 m) noexcept {
        return make_float2x2(m[0].x, m[1].x, m[0].y, m[1].y);
    };
    return inverse.function();
}

[[nodiscard]] static Function _float3x3_transpose() noexcept {
    static Callable inverse = [](Float3x3 m) noexcept {
        return make_float3x3(
            m[0].x, m[1].x, m[2].x,
            m[0].y, m[1].y, m[2].y,
            m[0].z, m[1].z, m[2].z);
    };
    return inverse.function();
}

[[nodiscard]] static Function _float4x4_transpose() noexcept {
    static Callable inverse = [](Float4x4 m) noexcept {
        return make_float4x4(
            m[0].x, m[1].x, m[2].x, m[3].x,
            m[0].y, m[1].y, m[2].y, m[3].y,
            m[0].z, m[1].z, m[2].z, m[3].z,
            m[0].w, m[1].w, m[2].w, m[3].w);
    };
    return inverse.function();
}

::llvm::Value *LLVMCodegen::_builtin_inverse(const Type *t, ::llvm::Value *pm) noexcept {
    LUISA_ASSERT(t->is_matrix(), "Expected matrix type.");
    auto ast = [t] {
        if (t->dimension() == 2u) { return _float2x2_inverse(); }
        if (t->dimension() == 3u) { return _float3x3_inverse(); }
        if (t->dimension() == 4u) { return _float4x4_inverse(); }
        LUISA_ERROR_WITH_LOCATION("Invalid matrix dimension {}.", t->dimension());
    }();
    auto func = _create_function(ast);
    auto b = _current_context()->builder.get();
    auto m = b->CreateLoad(_create_type(t), pm, "inverse.m");
    auto ret = b->CreateCall(func->getFunctionType(), func, m, "inverse.ret");
    return _create_stack_variable(ret, "inverse.ret.addr");
}

::llvm::Value *LLVMCodegen::_builtin_determinant(const Type *t, ::llvm::Value *pm) noexcept {
    LUISA_ASSERT(t->is_matrix(), "Expected matrix type.");
    auto ast = [t] {
        if (t->dimension() == 2u) { return _float2x2_det(); }
        if (t->dimension() == 3u) { return _float3x3_det(); }
        if (t->dimension() == 4u) { return _float4x4_det(); }
        LUISA_ERROR_WITH_LOCATION("Invalid matrix dimension {}.", t->dimension());
    }();
    auto func = _create_function(ast);
    auto b = _current_context()->builder.get();
    auto m = b->CreateLoad(_create_type(t), pm, "determinant.m");
    auto ret = b->CreateCall(func->getFunctionType(), func, m, "determinant.ret");
    return _create_stack_variable(ret, "determinant.ret.addr");
}

::llvm::Value *LLVMCodegen::_builtin_transpose(const Type *t, ::llvm::Value *pm) noexcept {
    LUISA_ASSERT(t->is_matrix(), "Expected matrix type.");
    auto ast = [t] {
        if (t->dimension() == 2u) { return _float2x2_transpose(); }
        if (t->dimension() == 3u) { return _float3x3_transpose(); }
        if (t->dimension() == 4u) { return _float4x4_transpose(); }
        LUISA_ERROR_WITH_LOCATION("Invalid matrix dimension {}.", t->dimension());
    }();
    auto func = _create_function(ast);
    auto b = _current_context()->builder.get();
    auto m = b->CreateLoad(_create_type(t), pm, "transpose.m");
    auto ret = b->CreateCall(func->getFunctionType(), func, m, "transpose.ret");
    return _create_stack_variable(ret, "transpose.ret.addr");
}

[[nodiscard]] ::llvm::Function *_declare_external_math_function(
    ::llvm::Module *module, luisa::string_view name, size_t n_args) noexcept {
    auto func_name = luisa::format("{}f", name);
    auto f = module->getFunction(luisa::string_view{func_name});
    auto ir_type = ::llvm::Type::getFloatTy(module->getContext());
    if (f == nullptr) {
        ::llvm::SmallVector<::llvm::Type *, 2u> arg_types(n_args, ir_type);
        f = ::llvm::Function::Create(
            ::llvm::FunctionType::get(ir_type, arg_types, false),
            ::llvm::Function::ExternalLinkage,
            luisa::string_view{func_name},
            module);
        f->setNoSync();
        f->setWillReturn();
        f->setDoesNotThrow();
        f->setMustProgress();
        f->setSpeculatable();
        f->setDoesNotAccessMemory();
        f->setDoesNotFreeMemory();
    }
    return f;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-static-cast-downcast"
[[nodiscard]] ::llvm::Value *_call_external_math_function(
    ::llvm::Module *module, ::llvm::IRBuilder<> *builder, const Type *t,
    luisa::string_view name, ::llvm::SmallVector<::llvm::Value *, 2u> p_args) noexcept {
    auto f = _declare_external_math_function(module, name, p_args.size());
    ::llvm::SmallVector<::llvm::Value *, 2u> args;
    for (auto i = 0u; i < p_args.size(); i++) {
        args.emplace_back(builder->CreateLoad(
            p_args[i]->getType()->getPointerElementType(), p_args[i],
            luisa::string_view{luisa::format("{}.arg{}", name, i)}));
    }
    // TODO: vectorize...
    if (t->is_vector()) {
        ::llvm::SmallVector<::llvm::Value *, 4u> v;
        for (auto i = 0u; i < t->dimension(); i++) {
            ::llvm::SmallVector<::llvm::Value *, 2u> args_i;
            for (auto a = 0u; a < args.size(); a++) {
                args_i.emplace_back(builder->CreateExtractElement(args[a], i));
            }
            v.emplace_back(builder->CreateCall(f, args_i));
        }
        auto dim = t->dimension() == 3u ? 4u : t->dimension();
        auto vec_type = ::llvm::VectorType::get(
            ::llvm::Type::getFloatTy(module->getContext()), dim, false);
        auto vec = static_cast<::llvm::Value *>(::llvm::UndefValue::get(vec_type));
        for (auto i = 0u; i < t->dimension(); i++) {
            vec = builder->CreateInsertElement(vec, v[i], i);
        }
        auto p_vec = builder->CreateAlloca(vec_type);
        builder->CreateStore(vec, p_vec);
        return p_vec;
    }
    // scalar
    auto y = builder->CreateCall(f, args, luisa::string_view{luisa::format("{}.call", name)});
    auto py = builder->CreateAlloca(p_args.front()->getType()->getPointerElementType(),
                                    nullptr, luisa::string_view{luisa::format("{}.addr", name)});
    builder->CreateStore(y, py);
    return py;
}
#pragma clang diagnostic pop

::llvm::Value *LLVMCodegen::_create_builtin_call_expr(const Type *ret_type, CallOp op, luisa::span<const Expression *const> args) noexcept {
    auto builder = _current_context()->builder.get();
    switch (op) {
        case CallOp::ALL: return _builtin_all(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::ANY: return _builtin_any(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::SELECT: return _builtin_select(
            args[2]->type(), args[0]->type(), _create_expr(args[2]),
            _create_expr(args[1]), _create_expr(args[0]));
        case CallOp::CLAMP: return _builtin_clamp(
            args[0]->type(), _create_expr(args[0]),
            _create_expr(args[1]), _create_expr(args[2]));
        case CallOp::LERP: return _builtin_lerp(
            args[0]->type(), _create_expr(args[0]),
            _create_expr(args[1]), _create_expr(args[2]));
        case CallOp::STEP: return _builtin_step(
            args[0]->type(), _create_expr(args[0]), _create_expr(args[1]));
        case CallOp::ABS: return _builtin_abs(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::MIN: return _builtin_min(
            args[0]->type(), _create_expr(args[0]), _create_expr(args[1]));
        case CallOp::MAX: return _builtin_max(
            args[0]->type(), _create_expr(args[0]), _create_expr(args[1]));
        case CallOp::CLZ: return _builtin_clz(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::CTZ: return _builtin_ctz(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::POPCOUNT: return _builtin_popcount(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::REVERSE: return _builtin_reverse(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::ISINF: return _builtin_isinf(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::ISNAN: return _builtin_isnan(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::ACOS: return _call_external_math_function(
            _module, builder, args[0]->type(), "acos", {_create_expr(args[0])});
        case CallOp::ACOSH: return _call_external_math_function(
            _module, builder, args[0]->type(), "acosh", {_create_expr(args[0])});
        case CallOp::ASIN: return _call_external_math_function(
            _module, builder, args[0]->type(), "asin", {_create_expr(args[0])});
        case CallOp::ASINH: return _call_external_math_function(
            _module, builder, args[0]->type(), "asinh", {_create_expr(args[0])});
        case CallOp::ATAN: return _call_external_math_function(
            _module, builder, args[0]->type(), "atan", {_create_expr(args[0])});
        case CallOp::ATAN2: return _call_external_math_function(
            _module, builder, args[0]->type(), "atan2",
            {_create_expr(args[0]), _create_expr(args[1])});
        case CallOp::ATANH: return _call_external_math_function(
            _module, builder, args[0]->type(), "atanh", {_create_expr(args[0])});
        case CallOp::COS: return _builtin_cos(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::COSH: return _call_external_math_function(
            _module, builder, args[0]->type(), "cosh", {_create_expr(args[0])});
        case CallOp::SIN: return _builtin_sin(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::SINH: return _call_external_math_function(
            _module, builder, args[0]->type(), "sinh", {_create_expr(args[0])});
        case CallOp::TAN: return _call_external_math_function(
            _module, builder, args[0]->type(), "tan", {_create_expr(args[0])});
        case CallOp::TANH: return _call_external_math_function(
            _module, builder, args[0]->type(), "tanh", {_create_expr(args[0])});
        case CallOp::EXP: return _builtin_exp(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::EXP2: return _builtin_exp2(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::EXP10: return _builtin_exp10(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::LOG: return _builtin_log(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::LOG2: return _builtin_log2(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::LOG10: return _builtin_log10(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::POW: return _builtin_pow(
            args[0]->type(), _create_expr(args[0]), _create_expr(args[1]));
        case CallOp::SQRT: return _builtin_sqrt(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::RSQRT: return _builtin_rsqrt(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::CEIL: return _builtin_ceil(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::FLOOR: return _builtin_floor(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::FRACT: return _builtin_fract(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::TRUNC: return _builtin_trunc(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::ROUND: return _builtin_round(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::FMA: return _builtin_fma(
            args[0]->type(), _create_expr(args[0]),
            _create_expr(args[1]), _create_expr(args[2]));
        case CallOp::COPYSIGN: return _builtin_copysign(
            args[0]->type(), _create_expr(args[0]), _create_expr(args[1]));
        case CallOp::CROSS: return _builtin_cross(
            args[0]->type(), _create_expr(args[0]), _create_expr(args[1]));
        case CallOp::DOT: return _builtin_dot(
            args[0]->type(), _create_expr(args[0]), _create_expr(args[1]));
        case CallOp::LENGTH: return _builtin_sqrt(
            Type::of<float>(), _builtin_length_squared(args[0]->type(), _create_expr(args[0])));
        case CallOp::LENGTH_SQUARED: return _builtin_length_squared(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::NORMALIZE: return _builtin_normalize(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::FACEFORWARD: return _builtin_faceforward(
            args[0]->type(), _create_expr(args[0]),
            _create_expr(args[1]), _create_expr(args[2]));
        case CallOp::DETERMINANT: return _builtin_determinant(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::TRANSPOSE: return _builtin_transpose(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::INVERSE: return _builtin_inverse(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::SYNCHRONIZE_BLOCK:
            LUISA_WARNING_WITH_LOCATION(
                "Block synchronization is not "
                "supported on LLVM backend.");
            return nullptr;
        case CallOp::ATOMIC_EXCHANGE: return _builtin_atomic_exchange(
            args[0]->type(), _create_expr(args[0]), _create_expr(args[1]));
        case CallOp::ATOMIC_COMPARE_EXCHANGE: return _builtin_atomic_compare_exchange(
            args[0]->type(), _create_expr(args[0]), _create_expr(args[1]), _create_expr(args[2]));
        case CallOp::ATOMIC_FETCH_ADD: return _builtin_atomic_fetch_add(
            args[0]->type(), _create_expr(args[0]), _create_expr(args[1]));
        case CallOp::ATOMIC_FETCH_SUB: return _builtin_atomic_fetch_sub(
            args[0]->type(), _create_expr(args[0]), _create_expr(args[1]));
        case CallOp::ATOMIC_FETCH_AND: return _builtin_atomic_fetch_and(
            args[0]->type(), _create_expr(args[0]), _create_expr(args[1]));
        case CallOp::ATOMIC_FETCH_OR: return _builtin_atomic_fetch_or(
            args[0]->type(), _create_expr(args[0]), _create_expr(args[1]));
        case CallOp::ATOMIC_FETCH_XOR: return _builtin_atomic_fetch_xor(
            args[0]->type(), _create_expr(args[0]), _create_expr(args[1]));
        case CallOp::ATOMIC_FETCH_MIN: return _builtin_atomic_fetch_min(
            args[0]->type(), _create_expr(args[0]), _create_expr(args[1]));
        case CallOp::ATOMIC_FETCH_MAX: return _builtin_atomic_fetch_max(
            args[0]->type(), _create_expr(args[0]), _create_expr(args[1]));
        case CallOp::BUFFER_READ: return _builtin_buffer_read(
            args[0]->type()->element(), _create_expr(args[0]), _create_expr(args[1]));
        case CallOp::BUFFER_WRITE:
            _builtin_buffer_write(
                args[0]->type()->element(), _create_expr(args[0]), _create_expr(args[1]),
                _builtin_static_cast(args[0]->type()->element(), args[2]->type(), _create_expr(args[2])));
            return nullptr;
        case CallOp::TEXTURE_READ: return _builtin_texture_read(
            ret_type, _create_expr(args[0]), _create_expr(args[1]));
        case CallOp::TEXTURE_WRITE:
            _builtin_texture_write(
                args[2]->type(), _create_expr(args[0]),
                _create_expr(args[1]), _create_expr(args[2]));
            return nullptr;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE: LUISA_WARNING_WITH_LOCATION("Not implemented."); break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: LUISA_WARNING_WITH_LOCATION("Not implemented."); break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: LUISA_WARNING_WITH_LOCATION("Not implemented."); break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE: LUISA_WARNING_WITH_LOCATION("Not implemented."); break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: LUISA_WARNING_WITH_LOCATION("Not implemented."); break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: LUISA_WARNING_WITH_LOCATION("Not implemented."); break;
        case CallOp::BINDLESS_TEXTURE2D_READ: LUISA_WARNING_WITH_LOCATION("Not implemented."); break;
        case CallOp::BINDLESS_TEXTURE3D_READ: LUISA_WARNING_WITH_LOCATION("Not implemented."); break;
        case CallOp::BINDLESS_TEXTURE2D_READ_LEVEL: LUISA_WARNING_WITH_LOCATION("Not implemented."); break;
        case CallOp::BINDLESS_TEXTURE3D_READ_LEVEL: LUISA_WARNING_WITH_LOCATION("Not implemented."); break;
        case CallOp::BINDLESS_TEXTURE2D_SIZE: LUISA_WARNING_WITH_LOCATION("Not implemented."); break;
        case CallOp::BINDLESS_TEXTURE3D_SIZE: LUISA_WARNING_WITH_LOCATION("Not implemented."); break;
        case CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL: LUISA_WARNING_WITH_LOCATION("Not implemented."); break;
        case CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: LUISA_WARNING_WITH_LOCATION("Not implemented."); break;
        case CallOp::BINDLESS_BUFFER_READ: LUISA_WARNING_WITH_LOCATION("Not implemented."); break;
        case CallOp::MAKE_BOOL2: return _builtin_make_vector2_overloaded(ret_type, args);
        case CallOp::MAKE_BOOL3: return _builtin_make_vector3_overloaded(ret_type, args);
        case CallOp::MAKE_BOOL4: return _builtin_make_vector4_overloaded(ret_type, args);
        case CallOp::MAKE_INT2: return _builtin_make_vector2_overloaded(ret_type, args);
        case CallOp::MAKE_INT3: return _builtin_make_vector3_overloaded(ret_type, args);
        case CallOp::MAKE_INT4: return _builtin_make_vector4_overloaded(ret_type, args);
        case CallOp::MAKE_UINT2: return _builtin_make_vector2_overloaded(ret_type, args);
        case CallOp::MAKE_UINT3: return _builtin_make_vector3_overloaded(ret_type, args);
        case CallOp::MAKE_UINT4: return _builtin_make_vector4_overloaded(ret_type, args);
        case CallOp::MAKE_FLOAT2: return _builtin_make_vector2_overloaded(ret_type, args);
        case CallOp::MAKE_FLOAT3: return _builtin_make_vector3_overloaded(ret_type, args);
        case CallOp::MAKE_FLOAT4: return _builtin_make_vector4_overloaded(ret_type, args);
        case CallOp::MAKE_FLOAT2X2: return _builtin_make_matrix2_overloaded(args);
        case CallOp::MAKE_FLOAT3X3: return _builtin_make_matrix3_overloaded(args);
        case CallOp::MAKE_FLOAT4X4: return _builtin_make_matrix4_overloaded(args);
        case CallOp::ASSUME:
            _builtin_assume(_create_expr(args[0]));
            return nullptr;
        case CallOp::UNREACHABLE:
            _builtin_unreachable();
            return nullptr;
        case CallOp::INSTANCE_TO_WORLD_MATRIX: LUISA_WARNING_WITH_LOCATION("Not implemented."); break;
        case CallOp::SET_INSTANCE_TRANSFORM: LUISA_WARNING_WITH_LOCATION("Not implemented."); break;
        case CallOp::SET_INSTANCE_VISIBILITY: LUISA_WARNING_WITH_LOCATION("Not implemented."); break;
        case CallOp::TRACE_CLOSEST: return _builtin_trace_closest(
            _create_expr(args[0]), _create_expr(args[1]));
        case CallOp::TRACE_ANY: return _builtin_trace_any(
            _create_expr(args[0]), _create_expr(args[1]));
        default: break;
    }
    // TODO: implement
    auto ctx = _current_context();
    if (ret_type == nullptr) { return nullptr; }
    return ctx->builder->CreateAlloca(_create_type(ret_type), nullptr, "tmp.addr");
}

[[nodiscard]] inline auto is_scalar_or_vector(const Type *t, Type::Tag tag) noexcept {
    return t->tag() == tag ||
           (t->is_vector() && t->element()->tag() == tag);
}

::llvm::Value *LLVMCodegen::_builtin_all(const Type *t, ::llvm::Value *v) noexcept {
    auto ctx = _current_context();
    auto result = static_cast<::llvm::Value *>(ctx->builder->getInt1(true));
    auto bv = ctx->builder->CreateLoad(_create_type(t), v, "load");
    static constexpr std::array elem_names{"v.x", "v.y", "v.z", "v.w"};
    static constexpr std::array cmp_names{"cmp.x", "cmp.y", "cmp.z", "cmp.w"};
    for (auto i = 0u; i < t->dimension(); i++) {
        auto elem = ctx->builder->CreateExtractElement(v, i, elem_names[i]);
        auto elem_not_zero = ctx->builder->CreateICmpNE(elem, ctx->builder->getInt8(0), cmp_names[i]);
        result = ctx->builder->CreateLogicalAnd(result, elem, "and");
    }
    return _create_stack_variable(result, "all.addr");
}

::llvm::Value *LLVMCodegen::_builtin_any(const Type *t, ::llvm::Value *v) noexcept {
    auto ctx = _current_context();
    auto result = static_cast<::llvm::Value *>(ctx->builder->getInt1(false));
    auto bv = ctx->builder->CreateLoad(_create_type(t), v, "load");
    static constexpr std::array elem_names{"v.x", "v.y", "v.z", "v.w"};
    static constexpr std::array cmp_names{"cmp.x", "cmp.y", "cmp.z", "cmp.w"};
    for (auto i = 0u; i < t->dimension(); i++) {
        auto elem = ctx->builder->CreateExtractElement(v, i, elem_names[i]);
        auto elem_not_zero = ctx->builder->CreateICmpNE(elem, ctx->builder->getInt8(0), cmp_names[i]);
        result = ctx->builder->CreateLogicalOr(result, elem_not_zero, "or");
    }
    return _create_stack_variable(result, "any.addr");
}

::llvm::Value *LLVMCodegen::_builtin_select(const Type *t_pred, const Type *t_value,
                                            ::llvm::Value *pred, ::llvm::Value *v_true, ::llvm::Value *v_false) noexcept {
    auto ctx = _current_context();
    auto pred_load = ctx->builder->CreateLoad(_create_type(t_pred), pred, "sel.pred");
    auto bv = [&] {
        auto zero = ctx->builder->CreateLoad(
            _create_type(t_pred),
            _builtin_static_cast(
                t_pred, Type::of<bool>(),
                _create_stack_variable(_literal(false), "sel.zero.addr")),
            "sel.zero");
        return ctx->builder->CreateICmpNE(pred_load, zero, "sel.cmp");
    }();
    auto v_true_load = ctx->builder->CreateLoad(_create_type(t_value), v_true, "sel.true");
    auto v_false_load = ctx->builder->CreateLoad(_create_type(t_value), v_false, "sel.false");
    auto result = ctx->builder->CreateSelect(bv, v_true_load, v_false_load, "sel");
    return _create_stack_variable(result, "sel.addr");
}

::llvm::Value *LLVMCodegen::_builtin_clamp(const Type *t, ::llvm::Value *v, ::llvm::Value *lo, ::llvm::Value *hi) noexcept {
    return _builtin_min(t, _builtin_max(t, v, lo), hi);
}

::llvm::Value *LLVMCodegen::_builtin_lerp(const Type *t, ::llvm::Value *a, ::llvm::Value *b, ::llvm::Value *x) noexcept {
    auto s = _builtin_sub(t, b, a);
    return _builtin_fma(t, x, s, a);// lerp(a, b, x) == x * (b - a) + a == fma(x, (b - a), a)
}

::llvm::Value *LLVMCodegen::_builtin_step(const Type *t, ::llvm::Value *edge, ::llvm::Value *x) noexcept {
    auto b = _current_context()->builder.get();
    auto zero = _builtin_static_cast(
        t, Type::of<float>(), _create_stack_variable(_literal(0.0f), "step.zero.addr"));
    auto one = _builtin_static_cast(
        t, Type::of<float>(), _create_stack_variable(_literal(1.0f), "step.one.addr"));
    if (t->is_scalar()) { return _builtin_select(Type::of<bool>(), t, _builtin_lt(t, x, edge), zero, one); }
    if (t->dimension() == 2u) { return _builtin_select(Type::of<bool2>(), t, _builtin_lt(t, x, edge), zero, one); }
    if (t->dimension() == 3u) { return _builtin_select(Type::of<bool3>(), t, _builtin_lt(t, x, edge), zero, one); }
    if (t->dimension() == 4u) { return _builtin_select(Type::of<bool4>(), t, _builtin_lt(t, x, edge), zero, one); }
    LUISA_ERROR_WITH_LOCATION("Invalid type '{}' for step.", t->description());
}

::llvm::Value *LLVMCodegen::_builtin_abs(const Type *t, ::llvm::Value *x) noexcept {
    auto ctx = _current_context();
    if (is_scalar_or_vector(t, Type::Tag::UINT)) { return x; }
    auto ir_type = _create_type(t);
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        auto m = ctx->builder->CreateIntrinsic(
            ::llvm::Intrinsic::abs, {ir_type},
            {ctx->builder->CreateLoad(ir_type, x, "iabs.x")},
            nullptr, "iabs");
        return _create_stack_variable(m, "iabs.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        auto m = ctx->builder->CreateIntrinsic(
            ::llvm::Intrinsic::fabs, {ir_type},
            {ctx->builder->CreateLoad(ir_type, x, "fabs.x")},
            nullptr, "fabs");
        return _create_stack_variable(m, "fabs.addr");
    }
    LUISA_ERROR_WITH_LOCATION("Invalid type '{}' for abs", t->description());
}

::llvm::Value *LLVMCodegen::_builtin_min(const Type *t, ::llvm::Value *x, ::llvm::Value *y) noexcept {
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        auto m = ctx->builder->CreateIntrinsic(
            ::llvm::Intrinsic::minimum, {ir_type},
            {ctx->builder->CreateLoad(ir_type, x, "fmin.x"),
             ctx->builder->CreateLoad(ir_type, y, "fmin.y")},
            nullptr, "fmin");
        return _create_stack_variable(m, "fmin.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        auto m = ctx->builder->CreateIntrinsic(
            ::llvm::Intrinsic::smin, {ir_type},
            {ctx->builder->CreateLoad(ir_type, x, "imin.x"),
             ctx->builder->CreateLoad(ir_type, y, "imin.y")},
            nullptr, "imin");
        return _create_stack_variable(m, "imin.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        auto m = ctx->builder->CreateIntrinsic(
            ::llvm::Intrinsic::umin, {ir_type},
            {ctx->builder->CreateLoad(ir_type, x, "umin.x"),
             ctx->builder->CreateLoad(ir_type, y, "umin.y")},
            nullptr, "umin");
        return _create_stack_variable(m, "umin.addr");
    }
    LUISA_ERROR_WITH_LOCATION("Invalid type '{}' for min.", t->description());
}

::llvm::Value *LLVMCodegen::_builtin_max(const Type *t, ::llvm::Value *x, ::llvm::Value *y) noexcept {
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        auto m = ctx->builder->CreateIntrinsic(
            ::llvm::Intrinsic::maximum, {ir_type},
            {ctx->builder->CreateLoad(ir_type, x, "fmax.x"),
             ctx->builder->CreateLoad(ir_type, y, "fmax.y")},
            nullptr, "fmax");
        return _create_stack_variable(m, "fmax.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        auto m = ctx->builder->CreateIntrinsic(
            ::llvm::Intrinsic::smax, {ir_type},
            {ctx->builder->CreateLoad(ir_type, x, "imax.x"),
             ctx->builder->CreateLoad(ir_type, y, "imax.y")},
            nullptr, "imax");
        return _create_stack_variable(m, "imax.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        auto m = ctx->builder->CreateIntrinsic(
            ::llvm::Intrinsic::umax, {ir_type},
            {ctx->builder->CreateLoad(ir_type, x, "umax.x"),
             ctx->builder->CreateLoad(ir_type, y, "umax.y")},
            nullptr, "umax");
        return _create_stack_variable(m, "umax.addr");
    }
    LUISA_ERROR_WITH_LOCATION("Invalid type '{}' for max.", t->description());
}

::llvm::Value *LLVMCodegen::_builtin_clz(const Type *t, ::llvm::Value *p) noexcept {
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    auto x = ctx->builder->CreateIntrinsic(
        ::llvm::Intrinsic::ctlz, {ir_type},
        {ctx->builder->CreateLoad(ir_type, p, "clz.x")},
        nullptr, "clz");
    return _create_stack_variable(x, "clz.addr");
}

::llvm::Value *LLVMCodegen::_builtin_ctz(const Type *t, ::llvm::Value *p) noexcept {
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    auto x = ctx->builder->CreateIntrinsic(
        ::llvm::Intrinsic::cttz, {ir_type},
        {ctx->builder->CreateLoad(ir_type, p, "ctz.x")},
        nullptr, "ctz");
    return _create_stack_variable(x, "ctz.addr");
}

::llvm::Value *LLVMCodegen::_builtin_popcount(const Type *t, ::llvm::Value *p) noexcept {
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    auto x = ctx->builder->CreateIntrinsic(
        ::llvm::Intrinsic::ctpop, {ir_type},
        {ctx->builder->CreateLoad(ir_type, p, "popcount.x")},
        nullptr, "popcount");
    return _create_stack_variable(x, "popcount.addr");
}

::llvm::Value *LLVMCodegen::_builtin_reverse(const Type *t, ::llvm::Value *p) noexcept {
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    auto x = ctx->builder->CreateIntrinsic(
        ::llvm::Intrinsic::bitreverse, {ir_type},
        {ctx->builder->CreateLoad(ir_type, p, "reverse.x")},
        nullptr, "reverse");
    return _create_stack_variable(x, "reverse.addr");
}

::llvm::Value *LLVMCodegen::_builtin_fma(const Type *t, ::llvm::Value *a, ::llvm::Value *b, ::llvm::Value *c) noexcept {
    auto ir_type = _create_type(t);
    auto ctx = _current_context();
    auto m = ctx->builder->CreateIntrinsic(
        ::llvm::Intrinsic::fma, {ir_type},
        {ctx->builder->CreateLoad(ir_type, a, "fma.a"),
         ctx->builder->CreateLoad(ir_type, b, "fma.b"),
         ctx->builder->CreateLoad(ir_type, c, "fma.c")},
        nullptr, "fma");
    return _create_stack_variable(m, "fma.addr");
}

::llvm::Value *LLVMCodegen::_builtin_exp(const Type *t, ::llvm::Value *v) noexcept {
    auto ir_type = _create_type(t);
    auto ctx = _current_context();
    auto m = ctx->builder->CreateIntrinsic(
        ::llvm::Intrinsic::exp, {ir_type},
        {ctx->builder->CreateLoad(ir_type, v, "exp.x")},
        nullptr, "exp");
    return _create_stack_variable(m, "exp.addr");
}

::llvm::Value *LLVMCodegen::_builtin_exp2(const Type *t, ::llvm::Value *v) noexcept {
    auto ir_type = _create_type(t);
    auto ctx = _current_context();
    auto m = ctx->builder->CreateIntrinsic(
        ::llvm::Intrinsic::exp2, {ir_type},
        {ctx->builder->CreateLoad(ir_type, v, "exp2.x")},
        nullptr, "exp2");
    return _create_stack_variable(m, "exp2.addr");
}

::llvm::Value *LLVMCodegen::_builtin_exp10(const Type *t, ::llvm::Value *v) noexcept {
    auto ten = _builtin_static_cast(
        t, Type::of<float>(),
        _create_stack_variable(_literal(10.f), "ten"));
    return _builtin_pow(t, ten, v);
}

::llvm::Value *LLVMCodegen::_builtin_log(const Type *t, ::llvm::Value *v) noexcept {
    auto ir_type = _create_type(t);
    auto ctx = _current_context();
    auto m = ctx->builder->CreateIntrinsic(
        ::llvm::Intrinsic::log, {ir_type},
        {ctx->builder->CreateLoad(ir_type, v, "log.x")},
        nullptr, "log");
    return _create_stack_variable(m, "log.addr");
}

::llvm::Value *LLVMCodegen::_builtin_log2(const Type *t, ::llvm::Value *v) noexcept {
    auto ir_type = _create_type(t);
    auto ctx = _current_context();
    auto m = ctx->builder->CreateIntrinsic(
        ::llvm::Intrinsic::log2, {ir_type},
        {ctx->builder->CreateLoad(ir_type, v, "log2.x")},
        nullptr, "log2");
    return _create_stack_variable(m, "log2.addr");
}

::llvm::Value *LLVMCodegen::_builtin_log10(const Type *t, ::llvm::Value *v) noexcept {
    auto ir_type = _create_type(t);
    auto ctx = _current_context();
    auto m = ctx->builder->CreateIntrinsic(
        ::llvm::Intrinsic::log10, {ir_type},
        {ctx->builder->CreateLoad(ir_type, v, "log10.x")},
        nullptr, "log10");
    return _create_stack_variable(m, "log10.addr");
}

::llvm::Value *LLVMCodegen::_builtin_pow(const Type *t, ::llvm::Value *x, ::llvm::Value *y) noexcept {
    auto ir_type = _create_type(t);
    auto ctx = _current_context();
    auto m = ctx->builder->CreateIntrinsic(
        ::llvm::Intrinsic::pow, {ir_type},
        {ctx->builder->CreateLoad(ir_type, x, "pow.x"),
         ctx->builder->CreateLoad(ir_type, y, "pow.y")},
        nullptr, "pow");
    return _create_stack_variable(m, "pow.addr");
}

::llvm::Value *LLVMCodegen::_builtin_copysign(const Type *t, ::llvm::Value *x, ::llvm::Value *y) noexcept {
    auto ir_type = _create_type(t);
    auto ctx = _current_context();
    auto m = ctx->builder->CreateIntrinsic(
        ::llvm::Intrinsic::copysign, {ir_type},
        {ctx->builder->CreateLoad(ir_type, x, "copysign.x"),
         ctx->builder->CreateLoad(ir_type, y, "copysign.y")},
        nullptr, "copysign");
    return _create_stack_variable(m, "copysign.addr");
}

::llvm::Value *LLVMCodegen::_builtin_faceforward(const Type *t, ::llvm::Value *n, ::llvm::Value *i, ::llvm::Value *nref) noexcept {
    auto ir_type = _create_type(t);
    auto ctx = _current_context();
    auto dot = ctx->builder->CreateLoad(
        _create_type(Type::of<float>()),
        _builtin_dot(t, nref, i), "faceforward.dot");
    auto pos_n = ctx->builder->CreateLoad(
        ir_type, n, "faceforward.pos_n");
    auto neg_n = ctx->builder->CreateFNeg(
        pos_n, "faceforward.neg_n");
    auto m = ctx->builder->CreateSelect(
        ctx->builder->CreateFCmpOLT(dot, _literal(0.f)),
        pos_n, neg_n, "faceforward.select");
    return _create_stack_variable(m, "faceforward.addr");
}

::llvm::Value *LLVMCodegen::_builtin_sin(const Type *t, ::llvm::Value *v) noexcept {
    auto ir_type = _create_type(t);
    auto ctx = _current_context();
    auto m = ctx->builder->CreateIntrinsic(
        ::llvm::Intrinsic::sin, {ir_type},
        {ctx->builder->CreateLoad(ir_type, v, "sin.x")},
        nullptr, "sin");
    return _create_stack_variable(m, "sin.addr");
}

::llvm::Value *LLVMCodegen::_builtin_cos(const Type *t, ::llvm::Value *v) noexcept {
    auto ir_type = _create_type(t);
    auto ctx = _current_context();
    auto m = ctx->builder->CreateIntrinsic(
        ::llvm::Intrinsic::cos, {ir_type},
        {ctx->builder->CreateLoad(ir_type, v, "cos.x")},
        nullptr, "cos");
    return _create_stack_variable(m, "cos.addr");
}

::llvm::Value *LLVMCodegen::_builtin_tan(const Type *t, ::llvm::Value *v) noexcept {
    return _builtin_div(t, _builtin_sin(t, v), _builtin_cos(t, v));
}

::llvm::Value *LLVMCodegen::_builtin_sqrt(const Type *t, ::llvm::Value *x) noexcept {
    auto ir_type = _create_type(t);
    auto ctx = _current_context();
    auto m = ctx->builder->CreateIntrinsic(
        ::llvm::Intrinsic::sqrt, {ir_type},
        {ctx->builder->CreateLoad(ir_type, x, "sqrt.x")},
        nullptr, "sqrt");
    return _create_stack_variable(m, "sqrt.addr");
}

::llvm::Value *LLVMCodegen::_builtin_fract(const Type *t, ::llvm::Value *v) noexcept {
    return _builtin_sub(t, v, _builtin_floor(t, v));
}

::llvm::Value *LLVMCodegen::_builtin_floor(const Type *t, ::llvm::Value *v) noexcept {
    auto ir_type = _create_type(t);
    auto ctx = _current_context();
    auto m = ctx->builder->CreateIntrinsic(
        ::llvm::Intrinsic::floor, {ir_type},
        {ctx->builder->CreateLoad(ir_type, v, "floor.x")},
        nullptr, "floor");
    return _create_stack_variable(m, "floor.addr");
}

::llvm::Value *LLVMCodegen::_builtin_ceil(const Type *t, ::llvm::Value *v) noexcept {
    auto ir_type = _create_type(t);
    auto ctx = _current_context();
    auto m = ctx->builder->CreateIntrinsic(
        ::llvm::Intrinsic::ceil, {ir_type},
        {ctx->builder->CreateLoad(ir_type, v, "ceil.x")},
        nullptr, "ceil");
    return _create_stack_variable(m, "ceil.addr");
}

::llvm::Value *LLVMCodegen::_builtin_trunc(const Type *t, ::llvm::Value *v) noexcept {
    auto ir_type = _create_type(t);
    auto ctx = _current_context();
    auto m = ctx->builder->CreateIntrinsic(
        ::llvm::Intrinsic::trunc, {ir_type},
        {ctx->builder->CreateLoad(ir_type, v, "trunc.x")},
        nullptr, "trunc");
    return _create_stack_variable(m, "trunc.addr");
}

::llvm::Value *LLVMCodegen::_builtin_round(const Type *t, ::llvm::Value *v) noexcept {
    auto ir_type = _create_type(t);
    auto ctx = _current_context();
    auto m = ctx->builder->CreateIntrinsic(
        ::llvm::Intrinsic::round, {ir_type},
        {ctx->builder->CreateLoad(ir_type, v, "round.x")},
        nullptr, "round");
    return _create_stack_variable(m, "round.addr");
}

::llvm::Value *LLVMCodegen::_builtin_rsqrt(const Type *t, ::llvm::Value *x) noexcept {
    //    auto ctx = _current_context();
    //    auto v = static_cast<::llvm::Value *>(
    //        ctx->builder->CreateLoad(_create_type(t), x, "rsqrt.x"));
    //    auto vectorize = [ctx, t](auto s) noexcept {
    //        if (t->is_scalar()) { return s; }
    //        auto dim = t->dimension() == 3u ? 4u : t->dimension();
    //        return ctx->builder->CreateVectorSplat(dim, s);
    //    };
    //    auto half_f32 = vectorize(_literal(0.5f));
    //    auto one_i32 = vectorize(_literal(1));
    //    auto magic_i32 = vectorize(_literal(0x5f375a86u));
    //    auto three_halves_f32 = vectorize(_literal(1.5f));
    //    auto x_half = ctx->builder->CreateFMul(v, half_f32, "rsqrt.x.half");
    //    auto i = ctx->builder->CreateBitCast(v, one_i32->getType(), "rsqrt.vi");
    //    i = ctx->builder->CreateLShr(i, one_i32);
    //    i = ctx->builder->CreateSub(magic_i32, i);
    //    x = ctx->builder->CreateBitCast(i, v->getType(), "rsqrt.vf");
    //    auto xx = ctx->builder->CreateFMul(x, x);
    //    auto xx_x_half = ctx->builder->CreateFMul(xx, x_half);
    //    auto sub = ctx->builder->CreateFSub(three_halves_f32, xx_x_half);
    //    v = ctx->builder->CreateFMul(x, sub);
    //    return _create_stack_variable(v, "rsqrt.addr");
    auto s = _builtin_sqrt(t, x);
    auto one = _builtin_static_cast(
        t, Type::of<float>(),
        _create_stack_variable(_literal(1.f), "rsqrt.one"));
    return _builtin_div(t, one, s);
}

static constexpr auto atomic_operation_order = ::llvm::AtomicOrdering::Monotonic;

::llvm::Value *LLVMCodegen::_builtin_atomic_exchange(const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_desired) noexcept {
    auto ctx = _current_context();
    auto desired = ctx->builder->CreateLoad(
        _create_type(t), p_desired, "atomic.exchange.desired");
    auto old = ctx->builder->CreateAtomicRMW(
        ::llvm::AtomicRMWInst::Xchg, p_atomic,
        desired, {}, atomic_operation_order);
    old->setName("atomic.exchange.old");
    return _create_stack_variable(old, "atomic.exchange.addr");
}

::llvm::Value *LLVMCodegen::_builtin_atomic_compare_exchange(const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_expected, ::llvm::Value *p_desired) noexcept {
    auto ctx = _current_context();
    auto expected = static_cast<::llvm::Value *>(ctx->builder->CreateLoad(
        _create_type(t), p_expected, "atomic.compare.exchange.expected"));
    auto desired = static_cast<::llvm::Value *>(ctx->builder->CreateLoad(
        _create_type(t), p_desired, "atomic.compare.exchange.desired"));
    if (t->tag() == Type::Tag::FLOAT) {
        expected = ctx->builder->CreateBitCast(
            expected, ctx->builder->getInt32Ty(),
            "atomic.compare.exchange.expected.int");
        desired = ctx->builder->CreateBitCast(
            desired, ctx->builder->getInt32Ty(),
            "atomic.compare.exchange.desired.int");
        p_atomic = ctx->builder->CreateBitOrPointerCast(
            p_atomic, ::llvm::PointerType::get(ctx->builder->getInt32Ty(), 0),
            "atomic.compare.exchange.atomic.int");
    }
    auto old_and_success = ctx->builder->CreateAtomicCmpXchg(
        p_atomic, expected, desired, {},
        atomic_operation_order,
        atomic_operation_order);
    old_and_success->setName("atomic.compare.exchange.old_and_success");
    auto old = ctx->builder->CreateExtractValue(
        old_and_success, 0, "atomic.compare.exchange.old");
    if (t->tag() == Type::Tag::FLOAT) {
        old = ctx->builder->CreateBitCast(
            old, ctx->builder->getFloatTy(),
            "atomic.compare.exchange.old.float");
    }
    return _create_stack_variable(old, "atomic.compare.exchange.addr");
}

// TODO: atomic_fetch_add for float seems not correct, hence it is manually implemented with atomic_compare_exchange
[[nodiscard]] inline ::llvm::Value *_atomic_fetch_add_float(::llvm::IRBuilder<> *builder, ::llvm::Value *ptr, ::llvm::Value *v_float) noexcept {
#define LUISA_COMPUTE_LLVM_USE_ATOMIC_FADD 0
#if LUISA_COMPUTE_LLVM_USE_ATOMIC_FADD
    auto old = builder->CreateAtomicRMW(
        ::llvm::AtomicRMWInst::FAdd, ptr,
        v_float, {}, atomic_operation_order);
    old->setName("atomic.fetch.add.old");
    auto p_old = builder->CreateAlloca(
        builder->getFloatTy(), nullptr,
        "atomic.fetch.add.old.addr");
    builder->CreateStore(old, p_old);
    return p_old;
#else
    auto p_old = builder->CreateAlloca(builder->getFloatTy(), nullptr, "atomic.fetch.add.old.addr");
    auto v_int = builder->CreateBitCast(v_float, builder->getInt32Ty(), "atomic.fetch.add.value.int");
    auto ptr_int = builder->CreateBitOrPointerCast(ptr, ::llvm::PointerType::get(builder->getInt32Ty(), 0), "atomic.fetch.add.ptr.int");
    auto func = builder->GetInsertBlock()->getParent();
    auto loop = ::llvm::BasicBlock::Create(builder->getContext(), "atomic.fetch.add.loop", func);
    auto loop_out = ::llvm::BasicBlock::Create(builder->getContext(), "atomic.fetch.add.loop.out", func);
    loop->moveAfter(builder->GetInsertBlock());
    builder->CreateBr(loop);
    builder->SetInsertPoint(loop);
    auto expected = builder->CreateLoad(builder->getFloatTy(), ptr, "atomic.fetch.add.expected");
    builder->CreateStore(expected, p_old);
    auto desired = builder->CreateFAdd(expected, v_float, "atomic.fetch.add.desired");
    auto desired_int = builder->CreateBitCast(desired, builder->getInt32Ty(), "atomic.fetch.add.desired.int");
    auto expected_int = builder->CreateBitCast(expected, builder->getInt32Ty(), "atomic.fetch.add.expected.int");
    auto old_and_success = builder->CreateAtomicCmpXchg(
        ptr_int, expected_int, desired_int, {}, atomic_operation_order, atomic_operation_order);
    old_and_success->setName("atomic.fetch.add.old_and_success");
    auto success = builder->CreateExtractValue(old_and_success, 1, "atomic.fetch.add.success");
    builder->CreateCondBr(success, loop_out, loop);
    loop_out->moveAfter(builder->GetInsertBlock());
    builder->SetInsertPoint(loop_out);
    return p_old;
#endif
#undef LUISA_COMPUTE_LLVM_USE_ATOMIC_FADD
}

::llvm::Value *LLVMCodegen::_builtin_atomic_fetch_add(const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_value) noexcept {
    auto ctx = _current_context();
    auto value = ctx->builder->CreateLoad(
        _create_type(t), p_value, "atomic.fetch.add.value");
    if (t->tag() == Type::Tag::FLOAT) {
        return _atomic_fetch_add_float(
            ctx->builder.get(), p_atomic, value);
    }
    auto old = ctx->builder->CreateAtomicRMW(
        ::llvm::AtomicRMWInst::Add, p_atomic,
        value, {}, atomic_operation_order);
    old->setName("atomic.fetch.add.old");
    return _create_stack_variable(old, "atomic.fetch.add.addr");
}

::llvm::Value *LLVMCodegen::_builtin_atomic_fetch_sub(const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_value) noexcept {
    auto ctx = _current_context();
    if (t->tag() == Type::Tag::FLOAT) {
        return _builtin_atomic_fetch_add(
            t, p_atomic, _builtin_unary_minus(t, p_value));
    }
    auto value = ctx->builder->CreateLoad(
        _create_type(t), p_value, "atomic.fetch.sub.value");
    auto old = ctx->builder->CreateAtomicRMW(
        ::llvm::AtomicRMWInst::Sub, p_atomic,
        value, {}, atomic_operation_order);
    old->setName("atomic.fetch.sub.old");
    return _create_stack_variable(old, "atomic.fetch.sub.addr");
}

::llvm::Value *LLVMCodegen::_builtin_atomic_fetch_and(const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_value) noexcept {
    auto ctx = _current_context();
    auto value = ctx->builder->CreateLoad(
        _create_type(t), p_value, "atomic.fetch.and.value");
    auto old = ctx->builder->CreateAtomicRMW(
        ::llvm::AtomicRMWInst::And, p_atomic,
        value, {}, atomic_operation_order);
    old->setName("atomic.fetch.and.old");
    return _create_stack_variable(old, "atomic.fetch.and.addr");
}

::llvm::Value *LLVMCodegen::_builtin_atomic_fetch_or(const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_value) noexcept {
    auto ctx = _current_context();
    auto value = ctx->builder->CreateLoad(
        _create_type(t), p_value, "atomic.fetch.or.value");
    auto old = ctx->builder->CreateAtomicRMW(
        ::llvm::AtomicRMWInst::Or, p_atomic,
        value, {}, atomic_operation_order);
    old->setName("atomic.fetch.or.old");
    return _create_stack_variable(old, "atomic.fetch.or.addr");
}

::llvm::Value *LLVMCodegen::_builtin_atomic_fetch_xor(const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_value) noexcept {
    auto ctx = _current_context();
    auto value = ctx->builder->CreateLoad(
        _create_type(t), p_value, "atomic.fetch.xor.value");
    auto old = ctx->builder->CreateAtomicRMW(
        ::llvm::AtomicRMWInst::Xor, p_atomic,
        value, {}, atomic_operation_order);
    old->setName("atomic.fetch.xor.old");
    return _create_stack_variable(old, "atomic.fetch.xor.addr");
}

::llvm::Value *LLVMCodegen::_builtin_atomic_fetch_min(const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_value) noexcept {
    auto ctx = _current_context();
    auto value = static_cast<::llvm::Value *>(ctx->builder->CreateLoad(
        _create_type(t), p_value, "atomic.fetch.min.value"));
    if (t->tag() == Type::Tag::UINT) {
        auto old = ctx->builder->CreateAtomicRMW(
            ::llvm::AtomicRMWInst::UMin, p_atomic,
            value, {}, atomic_operation_order);
        old->setName("atomic.fetch.min.old");
        return _create_stack_variable(old, "atomic.fetch.min.addr");
    }
    if (t->tag() == Type::Tag::FLOAT) {
        auto elem_type = ctx->builder->getInt32Ty();
        value = ctx->builder->CreateBitCast(
            value, elem_type, "atomic.fetch.min.value.int");
        p_atomic = ctx->builder->CreateBitOrPointerCast(
            p_atomic, ::llvm::PointerType::get(elem_type, 0),
            "atomic.fetch.min.addr.int");
        auto old = static_cast<::llvm::Value *>(
            ctx->builder->CreateAtomicRMW(
                ::llvm::AtomicRMWInst::Min, p_atomic,
                value, {}, atomic_operation_order));
        old->setName("atomic.fetch.min.old.int");
        old = ctx->builder->CreateBitCast(
            old, ctx->builder->getFloatTy(), "atomic.fetch.min.old");
        return _create_stack_variable(old, "atomic.fetch.min.addr");
    }
    auto old = ctx->builder->CreateAtomicRMW(
        ::llvm::AtomicRMWInst::Min, p_atomic,
        value, {}, atomic_operation_order);
    old->setName("atomic.fetch.min.old");
    return _create_stack_variable(old, "atomic.fetch.min.addr");
}

::llvm::Value *LLVMCodegen::_builtin_atomic_fetch_max(const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_value) noexcept {
    auto ctx = _current_context();
    auto value = static_cast<::llvm::Value *>(ctx->builder->CreateLoad(
        _create_type(t), p_value, "atomic.fetch.max.value"));
    if (t->tag() == Type::Tag::UINT) {
        auto old = ctx->builder->CreateAtomicRMW(
            ::llvm::AtomicRMWInst::UMax, p_atomic,
            value, {}, atomic_operation_order);
        old->setName("atomic.fetch.max.old");
        return _create_stack_variable(old, "atomic.fetch.max.addr");
    }
    if (t->tag() == Type::Tag::FLOAT) {
        auto elem_type = ctx->builder->getInt32Ty();
        value = ctx->builder->CreateBitCast(
            value, elem_type, "atomic.fetch.max.value.int");
        p_atomic = ctx->builder->CreateBitOrPointerCast(
            p_atomic, ::llvm::PointerType::get(elem_type, 0),
            "atomic.fetch.max.addr.int");
        auto old = static_cast<::llvm::Value *>(
            ctx->builder->CreateAtomicRMW(
                ::llvm::AtomicRMWInst::Max, p_atomic,
                value, {}, atomic_operation_order));
        old->setName("atomic.fetch.max.old.int");
        old = ctx->builder->CreateBitCast(
            old, ctx->builder->getFloatTy(), "atomic.fetch.max.old");
        return _create_stack_variable(old, "atomic.fetch.max.addr");
    }
    auto old = ctx->builder->CreateAtomicRMW(
        ::llvm::AtomicRMWInst::Max, p_atomic,
        value, {}, atomic_operation_order);
    old->setName("atomic.fetch.max.old");
    return _create_stack_variable(old, "atomic.fetch.max.addr");
}

::llvm::Value *LLVMCodegen::_builtin_normalize(const Type *t, ::llvm::Value *v) noexcept {
    auto norm = _builtin_rsqrt(Type::of<float>(), _builtin_dot(t, v, v));
    auto norm_v = _builtin_static_cast(t, Type::of<float>(), norm);
    return _builtin_mul(t, v, norm_v);
}

::llvm::Value *LLVMCodegen::_builtin_dot(const Type *t, ::llvm::Value *a, ::llvm::Value *b) noexcept {
    auto ctx = _current_context();
    auto mul = ctx->builder->CreateLoad(
        _create_type(t), _builtin_mul(t, a, b), "dot.mul");
    std::array<::llvm::Value *, 4> v{};
    for (auto i = 0; i < t->dimension(); i++) {
        auto elem_name = luisa::format("dot.sum.{}", i);
        auto var_name = luisa::format("dot.sum.{}.addr", i);
        v[i] = _create_stack_variable(
            ctx->builder->CreateExtractElement(
                mul, i, luisa::string_view{elem_name}),
            luisa::string_view{var_name});
    }
    if (t->dimension() == 2u) {
        return _builtin_add(t->element(), v[0], v[1]);
    }
    if (t->dimension() == 3u) {
        return _builtin_add(t->element(), v[0],
                            _builtin_add(t->element(), v[1], v[2]));
    }
    if (t->dimension() == 4u) {
        return _builtin_add(t->element(), _builtin_add(t->element(), v[0], v[1]),
                            _builtin_add(t->element(), v[2], v[3]));
    }
    LUISA_ERROR_WITH_LOCATION("Invalid dimension {} for dot.", t->dimension());
}

::llvm::Value *LLVMCodegen::_builtin_add(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    auto lhs_v = ctx->builder->CreateLoad(ir_type, lhs, "add.lhs");
    auto rhs_v = ctx->builder->CreateLoad(ir_type, rhs, "add.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            ctx->builder->CreateNSWAdd(lhs_v, rhs_v, "add"),
            "add.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            ctx->builder->CreateAdd(lhs_v, rhs_v, "add"),
            "add.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        return _create_stack_variable(
            ctx->builder->CreateFAdd(lhs_v, rhs_v, "add"),
            "add.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for add.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_sub(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    auto lhs_v = ctx->builder->CreateLoad(ir_type, lhs, "sub.lhs");
    auto rhs_v = ctx->builder->CreateLoad(ir_type, rhs, "sub.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            ctx->builder->CreateNSWSub(lhs_v, rhs_v, "sub"),
            "sub.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            ctx->builder->CreateSub(lhs_v, rhs_v, "sub"),
            "sub.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        return _create_stack_variable(
            ctx->builder->CreateFSub(lhs_v, rhs_v, "sub"),
            "sub.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for sub.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_mul(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    auto lhs_v = ctx->builder->CreateLoad(ir_type, lhs, "mul.lhs");
    auto rhs_v = ctx->builder->CreateLoad(ir_type, rhs, "mul.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            ctx->builder->CreateNSWMul(lhs_v, rhs_v, "mul"),
            "mul.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            ctx->builder->CreateMul(lhs_v, rhs_v, "mul"),
            "mul.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        return _create_stack_variable(
            ctx->builder->CreateFMul(lhs_v, rhs_v, "mul"),
            "mul.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for mul.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_div(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    auto lhs_v = ctx->builder->CreateLoad(ir_type, lhs, "div.lhs");
    auto rhs_v = ctx->builder->CreateLoad(ir_type, rhs, "div.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            ctx->builder->CreateSDiv(lhs_v, rhs_v, "div"),
            "div.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            ctx->builder->CreateUDiv(lhs_v, rhs_v, "div"),
            "div.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        return _create_stack_variable(
            ctx->builder->CreateFDiv(lhs_v, rhs_v, "div"),
            "div.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for div.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_mod(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    auto lhs_v = ctx->builder->CreateLoad(ir_type, lhs, "mod.lhs");
    auto rhs_v = ctx->builder->CreateLoad(ir_type, rhs, "mod.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            ctx->builder->CreateSRem(lhs_v, rhs_v, "mod"),
            "mod.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            ctx->builder->CreateURem(lhs_v, rhs_v, "mod"),
            "mod.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for mod.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_and(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    LUISA_ASSERT(is_scalar_or_vector(t, Type::Tag::INT) ||
                     is_scalar_or_vector(t, Type::Tag::UINT) ||
                     is_scalar_or_vector(t, Type::Tag::BOOL),
                 "Invalid type '{}' for and.", t->description());
    auto ctx = _current_context();
    auto result = ctx->builder->CreateAnd(
        ctx->builder->CreateLoad(_create_type(t), lhs, "and.lhs"),
        ctx->builder->CreateLoad(_create_type(t), rhs, "and.rhs"), "and");
    return _create_stack_variable(result, "and.addr");
}

::llvm::Value *LLVMCodegen::_builtin_or(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    LUISA_ASSERT(is_scalar_or_vector(t, Type::Tag::INT) ||
                     is_scalar_or_vector(t, Type::Tag::UINT) ||
                     is_scalar_or_vector(t, Type::Tag::BOOL),
                 "Invalid type '{}' for or.", t->description());
    auto ctx = _current_context();
    auto result = ctx->builder->CreateOr(
        ctx->builder->CreateLoad(_create_type(t), lhs, "or.lhs"),
        ctx->builder->CreateLoad(_create_type(t), rhs, "or.rhs"), "or");
    return _create_stack_variable(result, "or.addr");
}

::llvm::Value *LLVMCodegen::_builtin_xor(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    LUISA_ASSERT(is_scalar_or_vector(t, Type::Tag::INT) ||
                     is_scalar_or_vector(t, Type::Tag::UINT) ||
                     is_scalar_or_vector(t, Type::Tag::BOOL),
                 "Invalid type '{}' for xor.", t->description());
    auto ctx = _current_context();
    auto result = ctx->builder->CreateXor(
        ctx->builder->CreateLoad(_create_type(t), lhs, "xor.lhs"),
        ctx->builder->CreateLoad(_create_type(t), rhs, "xor.rhs"), "xor");
    return _create_stack_variable(result, "or.addr");
}

::llvm::Value *LLVMCodegen::_builtin_lt(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    auto lhs_v = ctx->builder->CreateLoad(ir_type, lhs, "lt.lhs");
    auto rhs_v = ctx->builder->CreateLoad(ir_type, rhs, "lt.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            ctx->builder->CreateICmpSLT(lhs_v, rhs_v, "lt"),
            "lt.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            ctx->builder->CreateICmpULT(lhs_v, rhs_v, "lt"),
            "lt.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        return _create_stack_variable(
            ctx->builder->CreateFCmpOLT(lhs_v, rhs_v, "lt"),
            "lt.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::BOOL)) {
        return _create_stack_variable(
            ctx->builder->CreateICmpULT(lhs_v, rhs_v, "lt"),
            "lt.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for lt.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_le(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    auto lhs_v = ctx->builder->CreateLoad(ir_type, lhs, "le.lhs");
    auto rhs_v = ctx->builder->CreateLoad(ir_type, rhs, "le.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            ctx->builder->CreateICmpSLE(lhs_v, rhs_v, "le"),
            "le.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            ctx->builder->CreateICmpULE(lhs_v, rhs_v, "le"),
            "le.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        return _create_stack_variable(
            ctx->builder->CreateFCmpOLE(lhs_v, rhs_v, "le"),
            "le.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::BOOL)) {
        return _create_stack_variable(
            ctx->builder->CreateICmpULE(lhs_v, rhs_v, "le"),
            "le.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for le.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_gt(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    auto lhs_v = ctx->builder->CreateLoad(ir_type, lhs, "gt.lhs");
    auto rhs_v = ctx->builder->CreateLoad(ir_type, rhs, "gt.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            ctx->builder->CreateICmpSGT(lhs_v, rhs_v, "gt"),
            "gt.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            ctx->builder->CreateICmpUGT(lhs_v, rhs_v, "gt"),
            "gt.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        return _create_stack_variable(
            ctx->builder->CreateFCmpOGT(lhs_v, rhs_v, "gt"),
            "gt.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::BOOL)) {
        return _create_stack_variable(
            ctx->builder->CreateICmpUGT(lhs_v, rhs_v, "gt"),
            "gt.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for gt.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_ge(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    auto lhs_v = ctx->builder->CreateLoad(ir_type, lhs, "ge.lhs");
    auto rhs_v = ctx->builder->CreateLoad(ir_type, rhs, "ge.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            ctx->builder->CreateICmpSGE(lhs_v, rhs_v, "ge"),
            "ge.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            ctx->builder->CreateICmpUGE(lhs_v, rhs_v, "ge"),
            "ge.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        return _create_stack_variable(
            ctx->builder->CreateFCmpOGE(lhs_v, rhs_v, "ge"),
            "ge.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::BOOL)) {
        return _create_stack_variable(
            ctx->builder->CreateICmpUGE(lhs_v, rhs_v, "ge"),
            "ge.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for ge.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_eq(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    auto lhs_v = ctx->builder->CreateLoad(ir_type, lhs, "eq.lhs");
    auto rhs_v = ctx->builder->CreateLoad(ir_type, rhs, "eq.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            ctx->builder->CreateICmpEQ(lhs_v, rhs_v, "eq"),
            "eq.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            ctx->builder->CreateICmpEQ(lhs_v, rhs_v, "eq"),
            "eq.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        return _create_stack_variable(
            ctx->builder->CreateFCmpOEQ(lhs_v, rhs_v, "eq"),
            "eq.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::BOOL)) {
        return _create_stack_variable(
            ctx->builder->CreateICmpEQ(lhs_v, rhs_v, "eq"),
            "eq.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for eq.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_ne(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    auto lhs_v = ctx->builder->CreateLoad(ir_type, lhs, "neq.lhs");
    auto rhs_v = ctx->builder->CreateLoad(ir_type, rhs, "neq.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            ctx->builder->CreateICmpNE(lhs_v, rhs_v, "neq"),
            "neq.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            ctx->builder->CreateICmpNE(lhs_v, rhs_v, "neq"),
            "neq.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        return _create_stack_variable(
            ctx->builder->CreateFCmpONE(lhs_v, rhs_v, "neq"),
            "neq.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::BOOL)) {
        return _create_stack_variable(
            ctx->builder->CreateICmpNE(lhs_v, rhs_v, "neq"),
            "neq.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for neq.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_shl(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    auto lhs_v = ctx->builder->CreateLoad(ir_type, lhs, "shl.lhs");
    auto rhs_v = ctx->builder->CreateLoad(ir_type, rhs, "shl.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT) ||
        is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            ctx->builder->CreateShl(lhs_v, rhs_v, "shl"),
            "shl.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for shl.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_shr(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    auto lhs_v = ctx->builder->CreateLoad(ir_type, lhs, "shr.lhs");
    auto rhs_v = ctx->builder->CreateLoad(ir_type, rhs, "shr.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            ctx->builder->CreateAShr(lhs_v, rhs_v, "shr"),
            "shr.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            ctx->builder->CreateLShr(lhs_v, rhs_v, "shr"),
            "shr.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for shr.",
        t->description());
}

void LLVMCodegen::_builtin_assume(::llvm::Value *p) noexcept {
    auto ctx = _current_context();
    auto pred = ctx->builder->CreateICmpNE(
        ctx->builder->CreateLoad(
            _create_type(Type::of<bool>()), p, "assume.load"),
        _literal(false), "assume.pred");
    ctx->builder->CreateAssumption(pred);
}

void LLVMCodegen::_builtin_unreachable() noexcept {
    _current_context()->builder->CreateUnreachable();
}

::llvm::Value *LLVMCodegen::_builtin_isinf(const Type *t, ::llvm::Value *p) noexcept {
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    if (t->is_scalar()) {
        auto bits = ctx->builder->CreateLoad(
            ctx->builder->getInt32Ty(),
            _builtin_bitwise_cast(Type::of<uint>(), t, p),
            "isinf.bits");
        auto is_inf = ctx->builder->CreateLogicalOr(
            ctx->builder->CreateICmpEQ(bits, _literal(0x7f800000u), "isinf.pos"),
            ctx->builder->CreateICmpEQ(bits, _literal(0xff800000u), "isinf.neg"),
            "isinf.pred");
        return _create_stack_variable(is_inf, "isinf.addr");
    }
    switch (t->dimension()) {
        case 2u: {
            auto bits = ctx->builder->CreateLoad(
                _create_type(Type::of<uint2>()),
                _builtin_bitwise_cast(Type::of<uint2>(), t, p),
                "isinf.bits");
            auto is_inf = ctx->builder->CreateLogicalOr(
                ctx->builder->CreateICmpEQ(bits, _literal(make_uint2(0x7f800000u)), "isinf.pos"),
                ctx->builder->CreateICmpEQ(bits, _literal(make_uint2(0xff800000u)), "isinf.neg"),
                "isinf.pred");
            return _create_stack_variable(is_inf, "isinf.addr");
        }
        case 3u: {
            auto bits = ctx->builder->CreateLoad(
                _create_type(Type::of<uint3>()),
                _builtin_bitwise_cast(Type::of<uint3>(), t, p),
                "isinf.bits");
            auto is_inf = ctx->builder->CreateLogicalOr(
                ctx->builder->CreateICmpEQ(bits, _literal(make_uint3(0x7f800000u)), "isinf.pos"),
                ctx->builder->CreateICmpEQ(bits, _literal(make_uint3(0xff800000u)), "isinf.neg"),
                "isinf.pred");
            return _create_stack_variable(is_inf, "isinf.addr");
        }
        case 4u: {
            auto bits = ctx->builder->CreateLoad(
                _create_type(Type::of<uint4>()),
                _builtin_bitwise_cast(Type::of<uint4>(), t, p),
                "isinf.bits");
            auto is_inf = ctx->builder->CreateLogicalOr(
                ctx->builder->CreateICmpEQ(bits, _literal(make_uint4(0x7f800000u)), "isinf.pos"),
                ctx->builder->CreateICmpEQ(bits, _literal(make_uint4(0xff800000u)), "isinf.neg"),
                "isinf.pred");
            return _create_stack_variable(is_inf, "isinf.addr");
        }
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid argument type '{}' for isinf.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_isnan(const Type *t, ::llvm::Value *p) noexcept {
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    if (t->is_scalar()) {
        auto bits = ctx->builder->CreateLoad(
            ctx->builder->getInt32Ty(),
            _builtin_bitwise_cast(Type::of<uint>(), t, p),
            "isnan.bits");
        auto is_nan = ctx->builder->CreateLogicalAnd(
            ctx->builder->CreateICmpEQ(
                ctx->builder->CreateAnd(bits, _literal(0x7ff00000u), "isnan.exp"),
                _literal(0x7ff00000u), "isnan.exp.cmp"),
            ctx->builder->CreateICmpNE(
                ctx->builder->CreateAnd(bits, _literal(0x7fffffu), "isnan.mant"),
                _literal(0u), "isnan.mant.cmp"),
            "isnan.pred");
        return _create_stack_variable(is_nan, "isnan.addr");
    }
    switch (t->dimension()) {
        case 2u: {
            auto bits = ctx->builder->CreateLoad(
                _create_type(Type::of<uint2>()),
                _builtin_bitwise_cast(Type::of<uint2>(), t, p),
                "isnan.bits");
            auto is_nan = ctx->builder->CreateLogicalAnd(
                ctx->builder->CreateICmpEQ(
                    ctx->builder->CreateAnd(bits, _literal(make_uint2(0x7ff00000u)), "isnan.exp"),
                    _literal(make_uint2(0x7ff00000u)), "isnan.exp.cmp"),
                ctx->builder->CreateICmpNE(
                    ctx->builder->CreateAnd(bits, _literal(make_uint2(0x7fffffu)), "isnan.mant"),
                    _literal(make_uint2(0u)), "isnan.mant.cmp"),
                "isnan.pred");
            return _create_stack_variable(is_nan, "isnan.addr");
        }
        case 3u: {
            auto bits = ctx->builder->CreateLoad(
                _create_type(Type::of<uint3>()),
                _builtin_bitwise_cast(Type::of<uint3>(), t, p),
                "isnan.bits");
            auto is_nan = ctx->builder->CreateLogicalAnd(
                ctx->builder->CreateICmpEQ(
                    ctx->builder->CreateAnd(bits, _literal(make_uint3(0x7ff00000u)), "isnan.exp"),
                    _literal(make_uint3(0x7ff00000u)), "isnan.exp.cmp"),
                ctx->builder->CreateICmpNE(
                    ctx->builder->CreateAnd(bits, _literal(make_uint3(0x7fffffu)), "isnan.mant"),
                    _literal(make_uint3(0u)), "isnan.mant.cmp"),
                "isnan.pred");
            return _create_stack_variable(is_nan, "isnan.addr");
        }
        case 4u: {
            auto bits = ctx->builder->CreateLoad(
                _create_type(Type::of<uint4>()),
                _builtin_bitwise_cast(Type::of<uint4>(), t, p),
                "isnan.bits");
            auto is_nan = ctx->builder->CreateLogicalAnd(
                ctx->builder->CreateICmpEQ(
                    ctx->builder->CreateAnd(bits, _literal(make_uint4(0x7ff00000u)), "isnan.exp"),
                    _literal(make_uint4(0x7ff00000u)), "isnan.exp.cmp"),
                ctx->builder->CreateICmpNE(
                    ctx->builder->CreateAnd(bits, _literal(make_uint4(0x7fffffu)), "isnan.mant"),
                    _literal(make_uint4(0u)), "isnan.mant.cmp"),
                "isnan.pred");
            return _create_stack_variable(is_nan, "isnan.addr");
        }
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid argument type '{}' for isnan.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_static_cast(const Type *t_dst, const Type *t_src, ::llvm::Value *p) noexcept {
    switch (t_dst->tag()) {
        case Type::Tag::BOOL: return _scalar_to_bool(t_src, p);
        case Type::Tag::FLOAT: return _scalar_to_float(t_src, p);
        case Type::Tag::INT: return _scalar_to_int(t_src, p);
        case Type::Tag::UINT: return _scalar_to_uint(t_src, p);
        case Type::Tag::VECTOR:
            return t_src->is_vector() ?
                       _vector_to_vector(t_dst, t_src, p) :
                       _scalar_to_vector(t_dst, t_src, p);
        case Type::Tag::MATRIX:
            return t_src->is_matrix() ?
                       _matrix_to_matrix(t_dst, t_src, p) :
                       _scalar_to_matrix(t_dst, t_src, p);
        default: break;
    }
    LUISA_ASSERT(*t_dst == *t_src, "Cannot convert '{}' to '{}'.",
                 t_src->description(), t_dst->description());
    return p;
}

::llvm::Value *LLVMCodegen::_builtin_bitwise_cast(const Type *t_dst, const Type *t_src, ::llvm::Value *p) noexcept {
    LUISA_ASSERT(t_dst->size() == t_src->size(),
                 "Invalid bitwise cast: {} to {}.",
                 t_src->description(), t_dst->description());
    auto ctx = _current_context();
    auto src_ir_type = _create_type(t_src);
    auto dst_ir_type = _create_type(t_dst);
    auto p_dst = ctx->builder->CreateBitOrPointerCast(p, dst_ir_type->getPointerTo(), "bitcast.ptr");
    auto dst = ctx->builder->CreateLoad(dst_ir_type, p_dst, "bitcast.dst");
    return _create_stack_variable(dst, "bitcast.addr");
}

::llvm::Value *LLVMCodegen::_builtin_unary_plus(const Type *t, ::llvm::Value *p) noexcept {
    return p;
}

::llvm::Value *LLVMCodegen::_builtin_unary_minus(const Type *t, ::llvm::Value *p) noexcept {
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    if (t->is_matrix()) {
        std::array<::llvm::Value *, 4u> m{};
        for (auto i = 0u; i < t->dimension(); ++i) {
            auto name = fmt::format("unary.minus.m{}.addr", i);
            m[i] = ctx->builder->CreateStructGEP(ir_type, p, i, luisa::string_view{name});
        }
        if (t->dimension() == 2u) {
            return _make_float2x2(
                ctx->builder->CreateFNeg(m[0]),
                ctx->builder->CreateFNeg(m[1]));
        }
        if (t->dimension() == 3u) {
            return _make_float3x3(
                ctx->builder->CreateFNeg(m[0]),
                ctx->builder->CreateFNeg(m[1]),
                ctx->builder->CreateFNeg(m[2]));
        }
        if (t->dimension() == 4u) {
            return _make_float4x4(
                ctx->builder->CreateFNeg(m[0]),
                ctx->builder->CreateFNeg(m[1]),
                ctx->builder->CreateFNeg(m[2]),
                ctx->builder->CreateFNeg(m[3]));
        }
        LUISA_ERROR_WITH_LOCATION(
            "Invalid matrix dimension '{}' for unary minus.",
            t->dimension());
    }
    auto x = ctx->builder->CreateLoad(ir_type, p, "unary.minus.load");
    switch (auto tag = t->is_scalar() ? t->tag() : t->element()->tag()) {
        case Type::Tag::BOOL: return _create_stack_variable(
            ctx->builder->CreateNot(x, "unary.minus"),
            "unary.minus.addr");
        case Type::Tag::FLOAT: return _create_stack_variable(
            ctx->builder->CreateFNeg(x, "unary.minus"),
            "unary.minus.addr");
        case Type::Tag::INT:
        case Type::Tag::UINT: return _create_stack_variable(
            ctx->builder->CreateNeg(x, "unary.minus"),
            "unary.minus.addr");
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid argument type '{}' for unary minus.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_unary_not(const Type *t, ::llvm::Value *p) noexcept {
    auto ctx = _current_context();
    auto pred = t->is_scalar() ? _scalar_to_bool(t, p) : _vector_to_bool_vector(t, p);
    auto pred_type = pred->getType()->getPointerElementType();
    auto b = ctx->builder->CreateLoad(pred_type, pred, "unary.not.load");
    auto zero = static_cast<::llvm::Value *>(_literal(false));
    if (pred->getType()->isVectorTy()) {
        auto dim = static_cast<::llvm::FixedVectorType *>(pred_type)->getNumElements();
        zero = ctx->builder->CreateVectorSplat(dim, zero, "unary.not.zero");
    }
    return _create_stack_variable(ctx->builder->CreateICmpEQ(b, zero, "unary.not.cmp"), "unary.not.addr");
}

::llvm::Value *LLVMCodegen::_builtin_unary_bit_not(const Type *t, ::llvm::Value *p) noexcept {
    LUISA_ASSERT(t->tag() == Type::Tag::INT || t->tag() == Type::Tag::UINT ||
                     (t->is_vector() && t->element()->tag() == Type::Tag::INT) ||
                     (t->is_vector() && t->element()->tag() == Type::Tag::UINT),
                 "Invalid argument type '{}' for bitwise not.",
                 t->description());
    auto ctx = _current_context();
    auto ir_type = _create_type(t);
    auto x = ctx->builder->CreateLoad(ir_type, p, "unary.bitnot.load");
    return _create_stack_variable(ctx->builder->CreateNot(x, "unary.bitnot"), "unary.bitnot.addr");
}

::llvm::Value *LLVMCodegen::_builtin_add_matrix_scalar(const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept {
    LUISA_ASSERT(t_lhs->is_matrix() && t_rhs->is_scalar(),
                 "Invalid argument types '{}' and '{}' for matrix-scalar addition.",
                 t_lhs->description(), t_rhs->description());
    auto ctx = _current_context();
    auto lhs_type = _create_type(t_lhs);
    auto col_type = Type::from(luisa::format("vector<float,{}>", t_lhs->dimension()));
    auto rhs = _scalar_to_vector(col_type, t_rhs, p_rhs);
    std::array<::llvm::Value *, 4u> m{};
    for (auto i = 0u; i < t_lhs->dimension(); ++i) {
        auto name = fmt::format("add.lhs.m{}.addr", i);
        auto col = ctx->builder->CreateStructGEP(
            lhs_type, p_lhs, i, luisa::string_view{name});
        m[i] = _builtin_add(col_type, col, rhs);
    }
    if (t_lhs->dimension() == 2u) { return _make_float2x2(m[0], m[1]); }
    if (t_lhs->dimension() == 3u) { return _make_float3x3(m[0], m[1], m[2]); }
    if (t_lhs->dimension() == 4u) { return _make_float4x4(m[0], m[1], m[2], m[3]); }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid matrix dimension '{}' for matrix-scalar addition.",
        t_lhs->dimension());
}

::llvm::Value *LLVMCodegen::_builtin_add_scalar_matrix(const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept {
    return _builtin_add_matrix_scalar(t_rhs, t_lhs, p_rhs, p_lhs);
}

::llvm::Value *LLVMCodegen::_builtin_add_matrix_matrix(const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept {
    LUISA_ASSERT(t_lhs->is_matrix() && t_rhs->is_matrix() &&
                     t_lhs->dimension() == t_rhs->dimension(),
                 "Invalid argument types '{}' and '{}' for matrix-matrix addition.",
                 t_lhs->description(), t_rhs->description());
    auto ctx = _current_context();
    auto matrix_type = _create_type(t_lhs);
    auto col_type = Type::from(luisa::format("vector<float,{}>", t_lhs->dimension()));
    std::array<::llvm::Value *, 4u> m{};
    for (auto i = 0u; i < t_lhs->dimension(); ++i) {
        auto lhs_name = fmt::format("add.lhs.m{}.addr", i);
        auto rhs_name = fmt::format("add.rhs.m{}.addr", i);
        auto lhs = ctx->builder->CreateStructGEP(
            matrix_type, p_lhs, i, luisa::string_view{lhs_name});
        auto rhs = ctx->builder->CreateStructGEP(
            matrix_type, p_rhs, i, luisa::string_view{rhs_name});
        m[i] = _builtin_add(col_type, lhs, rhs);
    }
    if (t_lhs->dimension() == 2u) { return _make_float2x2(m[0], m[1]); }
    if (t_lhs->dimension() == 3u) { return _make_float3x3(m[0], m[1], m[2]); }
    if (t_lhs->dimension() == 4u) { return _make_float4x4(m[0], m[1], m[2], m[3]); }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid matrix dimension '{}' for matrix-matrix addition.",
        t_lhs->dimension());
}

::llvm::Value *LLVMCodegen::_builtin_sub_matrix_scalar(const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept {
    LUISA_ASSERT(t_lhs->is_matrix() && t_rhs->is_scalar(),
                 "Invalid argument types '{}' and '{}' for matrix-scalar subtraction.",
                 t_lhs->description(), t_rhs->description());
    auto ctx = _current_context();
    auto lhs_type = _create_type(t_lhs);
    auto col_type = Type::from(luisa::format("vector<float,{}>", t_lhs->dimension()));
    auto rhs = _scalar_to_vector(col_type, t_rhs, p_rhs);
    std::array<::llvm::Value *, 4u> m{};
    for (auto i = 0u; i < t_lhs->dimension(); ++i) {
        auto name = fmt::format("sub.lhs.m{}.addr", i);
        auto col = ctx->builder->CreateStructGEP(
            lhs_type, p_lhs, i, luisa::string_view{name});
        m[i] = _builtin_sub(col_type, col, rhs);
    }
    if (t_lhs->dimension() == 2u) { return _make_float2x2(m[0], m[1]); }
    if (t_lhs->dimension() == 3u) { return _make_float3x3(m[0], m[1], m[2]); }
    if (t_lhs->dimension() == 4u) { return _make_float4x4(m[0], m[1], m[2], m[3]); }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid matrix dimension '{}' for matrix-scalar subtraction.",
        t_lhs->dimension());
}

::llvm::Value *LLVMCodegen::_builtin_sub_scalar_matrix(const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept {
    LUISA_ASSERT(t_lhs->is_scalar() && t_rhs->is_matrix(),
                 "Invalid argument types '{}' and '{}' for matrix-scalar subtraction.",
                 t_lhs->description(), t_rhs->description());
    auto ctx = _current_context();
    auto matrix_type = _create_type(t_rhs);
    auto col_type = Type::from(luisa::format("vector<float,{}>", t_lhs->dimension()));
    auto lhs = _scalar_to_vector(col_type, t_lhs, p_lhs);
    std::array<::llvm::Value *, 4u> m{};
    for (auto i = 0u; i < t_lhs->dimension(); ++i) {
        auto name = fmt::format("add.rhs.m{}.addr", i);
        auto rhs_col = ctx->builder->CreateStructGEP(
            matrix_type, p_rhs, i, luisa::string_view{name});
        m[i] = _builtin_sub(col_type, lhs, rhs_col);
    }
    if (t_lhs->dimension() == 2u) { return _make_float2x2(m[0], m[1]); }
    if (t_lhs->dimension() == 3u) { return _make_float3x3(m[0], m[1], m[2]); }
    if (t_lhs->dimension() == 4u) { return _make_float4x4(m[0], m[1], m[2], m[3]); }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid matrix dimension '{}' for matrix-scalar subtraction.",
        t_lhs->dimension());
}

::llvm::Value *LLVMCodegen::_builtin_sub_matrix_matrix(const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept {
    LUISA_ASSERT(t_lhs->is_matrix() && t_rhs->is_matrix() &&
                     t_lhs->dimension() == t_rhs->dimension(),
                 "Invalid argument types '{}' and '{}' for matrix-matrix subtraction.",
                 t_lhs->description(), t_rhs->description());
    auto ctx = _current_context();
    auto matrix_type = _create_type(t_lhs);
    auto col_type = Type::from(luisa::format("vector<float,{}>", t_lhs->dimension()));
    std::array<::llvm::Value *, 4u> m{};
    for (auto i = 0u; i < t_lhs->dimension(); ++i) {
        auto lhs_name = fmt::format("sub.lhs.m{}.addr", i);
        auto rhs_name = fmt::format("sub.rhs.m{}.addr", i);
        auto lhs = ctx->builder->CreateStructGEP(
            matrix_type, p_lhs, i, luisa::string_view{lhs_name});
        auto rhs = ctx->builder->CreateStructGEP(
            matrix_type, p_rhs, i, luisa::string_view{rhs_name});
        m[i] = _builtin_sub(col_type, lhs, rhs);
    }
    if (t_lhs->dimension() == 2u) { return _make_float2x2(m[0], m[1]); }
    if (t_lhs->dimension() == 3u) { return _make_float3x3(m[0], m[1], m[2]); }
    if (t_lhs->dimension() == 4u) { return _make_float4x4(m[0], m[1], m[2], m[3]); }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid matrix dimension '{}' for matrix-matrix subtraction.",
        t_lhs->dimension());
}

::llvm::Value *LLVMCodegen::_builtin_mul_matrix_scalar(const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept {
    LUISA_ASSERT(t_lhs->is_matrix() && t_rhs->is_scalar(),
                 "Invalid argument types '{}' and '{}' for matrix-scalar multiplication.",
                 t_lhs->description(), t_rhs->description());
    auto ctx = _current_context();
    auto lhs_type = _create_type(t_lhs);
    auto col_type = Type::from(luisa::format("vector<float,{}>", t_lhs->dimension()));
    auto rhs = _scalar_to_vector(col_type, t_rhs, p_rhs);
    std::array<::llvm::Value *, 4u> m{};
    for (auto i = 0u; i < t_lhs->dimension(); ++i) {
        auto name = fmt::format("mul.lhs.m{}.addr", i);
        auto col = ctx->builder->CreateStructGEP(
            lhs_type, p_lhs, i, luisa::string_view{name});
        m[i] = _builtin_mul(col_type, col, rhs);
    }
    if (t_lhs->dimension() == 2u) { return _make_float2x2(m[0], m[1]); }
    if (t_lhs->dimension() == 3u) { return _make_float3x3(m[0], m[1], m[2]); }
    if (t_lhs->dimension() == 4u) { return _make_float4x4(m[0], m[1], m[2], m[3]); }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid matrix dimension '{}' for matrix-scalar multiplication.",
        t_lhs->dimension());
}

::llvm::Value *LLVMCodegen::_builtin_mul_scalar_matrix(const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept {
    return _builtin_mul_matrix_scalar(t_rhs, t_lhs, p_rhs, p_lhs);
}

::llvm::Value *LLVMCodegen::_builtin_mul_matrix_matrix(const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept {
    LUISA_ASSERT(t_lhs->is_matrix() && t_rhs->is_matrix() &&
                     t_lhs->dimension() == t_rhs->dimension(),
                 "Invalid argument types '{}' and '{}' for matrix-matrix multiplication.",
                 t_lhs->description(), t_rhs->description());
    std::array<::llvm::Value *, 4u> m{};
    auto matrix_type = _create_type(t_lhs);
    auto col_type = Type::from(luisa::format("vector<float,{}>", t_lhs->dimension()));
    auto ctx = _current_context();
    for (auto i = 0u; i < t_lhs->dimension(); ++i) {
        auto rhs_name = fmt::format("mul.rhs.m{}.addr", i);
        auto rhs_col = ctx->builder->CreateStructGEP(
            matrix_type, p_rhs, i, luisa::string_view{rhs_name});
        m[i] = _builtin_mul_matrix_vector(t_lhs, col_type, p_lhs, rhs_col);
    }
    if (t_lhs->dimension() == 2u) { return _make_float2x2(m[0], m[1]); }
    if (t_lhs->dimension() == 3u) { return _make_float3x3(m[0], m[1], m[2]); }
    if (t_lhs->dimension() == 4u) { return _make_float4x4(m[0], m[1], m[2], m[3]); }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid matrix dimension '{}' for matrix-matrix multiplication.",
        t_lhs->dimension());
}

::llvm::Value *LLVMCodegen::_builtin_mul_matrix_vector(const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept {
    LUISA_ASSERT(t_lhs->is_matrix() && t_rhs->is_vector() &&
                     t_lhs->dimension() == t_rhs->dimension(),
                 "Invalid argument types '{}' and '{}' for matrix-vector multiplication.",
                 t_lhs->description(), t_rhs->description());
    std::array<::llvm::Value *, 4u> m{};
    auto ctx = _current_context();
    auto matrix_type = _create_type(t_lhs);
    auto col_type = Type::from(luisa::format("vector<float,{}>", t_lhs->dimension()));
    auto rhs = ctx->builder->CreateLoad(_create_type(t_rhs), p_rhs, "mul.rhs");
    for (auto i = 0u; i < t_lhs->dimension(); ++i) {
        auto col_name = fmt::format("mul.lhs.m{}.addr", i);
        auto col = ctx->builder->CreateStructGEP(matrix_type, p_lhs, i, luisa::string_view{col_name});
        auto v_name = fmt::format("mul.rhs.v{}", i);
        auto dim = t_rhs->dimension() == 3u ? 4u : t_rhs->dimension();
        ::llvm::SmallVector<int, 4u> masks(dim, static_cast<int>(i));
        auto v = ctx->builder->CreateShuffleVector(rhs, masks, luisa::string_view{v_name});
        auto pv_name = fmt::format("mul.rhs.v{}.addr", i);
        m[i] = _builtin_mul(col_type, col, _create_stack_variable(v, luisa::string_view{pv_name}));
    }
    if (t_lhs->dimension() == 2u) { return _builtin_add(col_type, m[0], m[1]); }
    if (t_lhs->dimension() == 3u) { return _builtin_add(col_type, _builtin_add(col_type, m[0], m[1]), m[2]); }
    if (t_lhs->dimension() == 4u) { return _builtin_add(col_type, _builtin_add(col_type, m[0], m[1]), _builtin_add(col_type, m[2], m[3])); }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid matrix dimension '{}' for matrix-vector multiplication.",
        t_lhs->dimension());
}

::llvm::Value *LLVMCodegen::_builtin_div_matrix_scalar(const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept {
    LUISA_ASSERT(t_lhs->is_matrix() && t_rhs->is_scalar(),
                 "Invalid argument types '{}' and '{}' for matrix-scalar division.",
                 t_lhs->description(), t_rhs->description());
    auto ctx = _current_context();
    auto lhs_type = _create_type(t_lhs);
    auto col_type = Type::from(luisa::format("vector<float,{}>", t_lhs->dimension()));
    auto rhs = _scalar_to_vector(col_type, t_rhs, p_rhs);
    std::array<::llvm::Value *, 4u> m{};
    for (auto i = 0u; i < t_lhs->dimension(); ++i) {
        auto name = fmt::format("div.lhs.m{}.addr", i);
        auto col = ctx->builder->CreateStructGEP(
            lhs_type, p_lhs, i, luisa::string_view{name});
        m[i] = _builtin_div(col_type, col, rhs);
    }
    if (t_lhs->dimension() == 2u) { return _make_float2x2(m[0], m[1]); }
    if (t_lhs->dimension() == 3u) { return _make_float3x3(m[0], m[1], m[2]); }
    if (t_lhs->dimension() == 4u) { return _make_float4x4(m[0], m[1], m[2], m[3]); }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid matrix dimension '{}' for matrix-scalar division.",
        t_lhs->dimension());
}

::llvm::Value *LLVMCodegen::_builtin_div_scalar_matrix(const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept {
    LUISA_ASSERT(t_lhs->is_scalar() && t_rhs->is_matrix(),
                 "Invalid argument types '{}' and '{}' for matrix-scalar division.",
                 t_lhs->description(), t_rhs->description());
    auto ctx = _current_context();
    auto matrix_type = _create_type(t_rhs);
    auto col_type = Type::from(luisa::format("vector<float,{}>", t_lhs->dimension()));
    auto lhs = _scalar_to_vector(col_type, t_lhs, p_lhs);
    std::array<::llvm::Value *, 4u> m{};
    for (auto i = 0u; i < t_lhs->dimension(); ++i) {
        auto name = fmt::format("add.rhs.m{}.addr", i);
        auto rhs_col = ctx->builder->CreateStructGEP(
            matrix_type, p_rhs, i, luisa::string_view{name});
        m[i] = _builtin_div(col_type, lhs, rhs_col);
    }
    if (t_lhs->dimension() == 2u) { return _make_float2x2(m[0], m[1]); }
    if (t_lhs->dimension() == 3u) { return _make_float3x3(m[0], m[1], m[2]); }
    if (t_lhs->dimension() == 4u) { return _make_float4x4(m[0], m[1], m[2], m[3]); }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid matrix dimension '{}' for matrix-scalar division.",
        t_lhs->dimension());
}

::llvm::Value *LLVMCodegen::_builtin_buffer_read(const Type *t_value, ::llvm::Value *buffer, ::llvm::Value *p_index) noexcept {
    auto ctx = _current_context();
    auto value_type = _create_type(t_value);
    auto index = ctx->builder->CreateLoad(_create_type(Type::of<uint>()), p_index, "buffer.read.index");
    auto ptr = ctx->builder->CreateInBoundsGEP(value_type, buffer, index, "buffer.read.ptr");
    auto value = ctx->builder->CreateLoad(value_type, ptr, "buffer.read");
    return _create_stack_variable(value, "buffer.read.addr");
}

void LLVMCodegen::_builtin_buffer_write(const Type *t_value, ::llvm::Value *buffer, ::llvm::Value *p_index, ::llvm::Value *p_value) noexcept {
    auto ctx = _current_context();
    auto value_type = _create_type(t_value);
    auto index = ctx->builder->CreateLoad(_create_type(Type::of<uint>()), p_index, "buffer.write.index");
    auto ptr = ctx->builder->CreateInBoundsGEP(value_type, buffer, index, "buffer.write.ptr");
    _create_assignment(t_value, t_value, ptr, p_value);
}

::llvm::Value *LLVMCodegen::_builtin_texture_read(const Type *t, ::llvm::Value *texture, ::llvm::Value *p_coord) noexcept {
    LUISA_ASSERT(t->is_vector() && t->dimension() == 4u,
                 "Invalid type '{}' for texture-read.",
                 t->description());
    auto value_type = static_cast<::llvm::FixedVectorType *>(_create_type(t));
    auto elem_type = value_type->getElementType();
    auto coord_type = static_cast<::llvm::FixedVectorType *>(p_coord->getType()->getPointerElementType());
    auto dim = coord_type->getNumElements() == 2u ? 2u : 3u;
    auto func_name = luisa::format("texture.read.{}d.{}", dim, t->element()->description());
    auto func = _module->getFunction(luisa::string_view{func_name});
    auto coord_struct_type = dim == 2u ? ::llvm::Type::getInt64Ty(_context) :
                                         ::llvm::Type::getInt128Ty(_context);
    auto value_struct_type = elem_type->isFloatTy() ?
                                 ::llvm::StructType::get(elem_type, elem_type, elem_type, elem_type) :
                                 static_cast<::llvm::Type *>(::llvm::Type::getInt128Ty(_context));
    if (func == nullptr) {
        func = ::llvm::Function::Create(
            ::llvm::FunctionType::get(value_struct_type, {texture->getType(), coord_struct_type}, false),
            ::llvm::Function::ExternalLinkage, luisa::string_view{func_name}, _module);
        func->setNoSync();
        func->setWillReturn();
        func->setDoesNotThrow();
        func->setMustProgress();
        func->setSpeculatable();
        func->setOnlyReadsMemory();
        func->setDoesNotFreeMemory();
        func->setOnlyAccessesInaccessibleMemory();
    }
    auto ctx = _current_context();
    auto coord_ptr = ctx->builder->CreateBitOrPointerCast(
        p_coord, coord_struct_type->getPointerTo(0), "texture.read.coord.addr");
    auto coord = ctx->builder->CreateLoad(coord_struct_type, coord_ptr, "texture.write.coord");
    auto value = ctx->builder->CreateCall(func->getFunctionType(), func, {texture, coord}, "texture.read.struct");
    auto value_vector = static_cast<::llvm::Value *>(::llvm::UndefValue::get(value_type));
    for (auto i = 0; i < value_type->getNumElements(); i++) {
        auto elem = ctx->builder->CreateExtractValue(value, i, "texture.read.struct.extract");
        value_vector = ctx->builder->CreateInsertElement(value_vector, elem, i, "texture.read.struct.insert");
    }
    return _create_stack_variable(value_vector, "texture.read.addr");
}

void LLVMCodegen::_builtin_texture_write(const Type *t, ::llvm::Value *texture, ::llvm::Value *p_coord, ::llvm::Value *p_value) noexcept {
    LUISA_ASSERT(t->is_vector() && t->dimension() == 4u,
                 "Invalid type '{}' for texture-read.",
                 t->description());
    auto value_type = static_cast<::llvm::FixedVectorType *>(_create_type(t));
    auto elem_type = value_type->getElementType();
    auto coord_type = static_cast<::llvm::FixedVectorType *>(p_coord->getType()->getPointerElementType());
    auto dim = coord_type->getNumElements() == 2u ? 2u : 3u;
    auto func_name = luisa::format("texture.write.{}d.{}", dim, t->element()->description());
    auto func = _module->getFunction(luisa::string_view{func_name});
    auto coord_struct_type = dim == 2u ? ::llvm::Type::getInt64Ty(_context) : ::llvm::Type::getInt128Ty(_context);
    auto value_struct_type = elem_type->isFloatTy() ?
                                 static_cast<::llvm::Type *>(::llvm::ArrayType::get(elem_type, 4u)) :
                                 ::llvm::Type::getInt128Ty(_context);
    if (func == nullptr) {
        func = ::llvm::Function::Create(
            ::llvm::FunctionType::get(
                ::llvm::Type::getVoidTy(_context),
                {texture->getType(), coord_struct_type, value_struct_type}, false),
            ::llvm::Function::ExternalLinkage, luisa::string_view{func_name}, _module);
        func->setNoSync();
        func->setWillReturn();
        func->setDoesNotThrow();
        func->setMustProgress();
        func->setSpeculatable();
        func->setDoesNotReadMemory();
        func->setDoesNotFreeMemory();
        func->setOnlyAccessesInaccessibleMemory();
    }
    auto ctx = _current_context();
    auto coord_ptr = ctx->builder->CreateBitOrPointerCast(
        p_coord, coord_struct_type->getPointerTo(0), "texture.write.coord.addr");
    auto value_ptr = ctx->builder->CreateBitOrPointerCast(
        p_value, value_struct_type->getPointerTo(0), "texture.write.value.addr");
    auto coord = ctx->builder->CreateLoad(coord_struct_type, coord_ptr, "texture.write.coord");
    auto value = ctx->builder->CreateLoad(value_struct_type, value_ptr, "texture.write.value");
    ctx->builder->CreateCall(func->getFunctionType(), func, {texture, coord, value});
}

::llvm::Value *LLVMCodegen::_builtin_trace_closest(::llvm::Value *accel, ::llvm::Value *p_ray) noexcept {
    auto func_name = "accel.trace.closest";
    auto func = _module->getFunction(func_name);
    auto hit_struct_type = ::llvm::Type::getInt128Ty(_context);
    if (func == nullptr) {
        func = ::llvm::Function::Create(
            ::llvm::FunctionType::get(hit_struct_type, {accel->getType(), p_ray->getType()}, false),
            ::llvm::Function::ExternalLinkage, luisa::string_view{func_name}, _module);
        func->setNoSync();
        func->setWillReturn();
        func->setDoesNotThrow();
        func->setMustProgress();
        func->setSpeculatable();
        func->setDoesNotFreeMemory();
        func->setOnlyReadsMemory();
        func->setOnlyAccessesInaccessibleMemOrArgMem();
    }
    auto ctx = _current_context();
    auto hit_struct = ctx->builder->CreateCall(
        func->getFunctionType(), func, {accel, p_ray}, "accel.trace.closest.struct");
    auto hit_struct_ptr = _create_stack_variable(hit_struct, "accel.trace.closest.struct.addr");
    return ctx->builder->CreateBitOrPointerCast(
        hit_struct_ptr, _create_type(Type::of<Hit>())->getPointerTo(0), "accel.trace.closest.addr");
}

::llvm::Value *LLVMCodegen::_builtin_trace_any(::llvm::Value *accel, ::llvm::Value *p_ray) noexcept {
    auto func_name = "accel.trace.any";
    auto func = _module->getFunction(func_name);
    if (func == nullptr) {
        func = ::llvm::Function::Create(
            ::llvm::FunctionType::get(::llvm::Type::getInt8Ty(_context),
                                      {accel->getType(), p_ray->getType()}, false),
            ::llvm::Function::ExternalLinkage, luisa::string_view{func_name}, _module);
        func->setNoSync();
        func->setWillReturn();
        func->setDoesNotThrow();
        func->setMustProgress();
        func->setSpeculatable();
        func->setOnlyReadsMemory();
        func->setDoesNotFreeMemory();
        func->setOnlyAccessesInaccessibleMemOrArgMem();
    }
    auto ctx = _current_context();
    auto hit = ctx->builder->CreateCall(
        func->getFunctionType(), func, {accel, p_ray}, "accel.trace.any.hit");
    return _create_stack_variable(hit, "accel.trace.any.hit.addr");
}

::llvm::Value *LLVMCodegen::_builtin_length_squared(const Type *t, ::llvm::Value *v) noexcept {
    return _builtin_dot(t, v, v);
}

::llvm::Value *LLVMCodegen::_builtin_cross(const Type *t, ::llvm::Value *a, ::llvm::Value *b) noexcept {
    LUISA_ASSERT(t->is_vector() && t->dimension() == 3u,
                 "Invalid argument types '{}' and '{}' for cross product.",
                 t->description(), t->description());
    auto ctx = _current_context();
    auto type = _create_type(t);
    a = ctx->builder->CreateLoad(type, a, "cross.a");
    b = ctx->builder->CreateLoad(type, b, "cross.b");
    auto a_x = ctx->builder->CreateExtractElement(a, static_cast<uint64_t>(0u), "cross.a.x");
    auto a_y = ctx->builder->CreateExtractElement(a, static_cast<uint64_t>(1u), "cross.a.y");
    auto a_z = ctx->builder->CreateExtractElement(a, static_cast<uint64_t>(2u), "cross.a.z");
    auto b_x = ctx->builder->CreateExtractElement(b, static_cast<uint64_t>(0u), "cross.b.x");
    auto b_y = ctx->builder->CreateExtractElement(b, static_cast<uint64_t>(1u), "cross.b.y");
    auto b_z = ctx->builder->CreateExtractElement(b, static_cast<uint64_t>(2u), "cross.b.z");
    auto x = ctx->builder->CreateFSub(
        ctx->builder->CreateFMul(a_y, b_z),
        ctx->builder->CreateFMul(a_z, b_y),
        "cross.x");
    auto y = ctx->builder->CreateFSub(
        ctx->builder->CreateFMul(a_z, b_x),
        ctx->builder->CreateFMul(a_x, b_z),
        "cross.y");
    auto z = ctx->builder->CreateFSub(
        ctx->builder->CreateFMul(a_x, b_y),
        ctx->builder->CreateFMul(a_y, b_x),
        "cross.z");
    return _make_float3(
        _create_stack_variable(x, "cross.x.addr"),
        _create_stack_variable(y, "cross.y.addr"),
        _create_stack_variable(z, "cross.z.addr"));
}

::llvm::Value *LLVMCodegen::_builtin_make_vector2_overloaded(const Type *t_vec, luisa::span<const Expression *const> args) noexcept {
    if (args.size() == 1u) {
        return _builtin_static_cast(t_vec, args[0]->type(), _create_expr(args[0]));
    }
    if (args.size() == 2u) {
        LUISA_ASSERT(args[0]->type()->is_scalar() && args[1]->type()->is_scalar(),
                     "Invalid argument types '{}' and '{}' for make-vector2.",
                     args[0]->type()->description(), args[1]->type()->description());
        return _make_float2(_create_expr(args[0]), _create_expr(args[1]));
    }
    LUISA_ERROR_WITH_LOCATION("Invalid number of arguments '{}' for make-vector2.", args.size());
}

::llvm::Value *LLVMCodegen::_builtin_make_vector3_overloaded(const Type *t_vec, luisa::span<const Expression *const> args) noexcept {
    if (args.size() == 1u) {
        return _builtin_static_cast(t_vec, args[0]->type(), _create_expr(args[0]));
    }
    auto b = _current_context()->builder.get();
    if (args.size() == 2u) {
        if (args[0]->type()->is_scalar()) {
            LUISA_ASSERT(args[1]->type()->is_vector() && args[1]->type()->dimension() == 2u,
                         "Invalid argument types ('{}', '{}') to make {}.",
                         args[0]->type()->description(), args[1]->type()->description(),
                         t_vec->description());
            auto yz = b->CreateLoad(
                _create_type(args[1]->type()), _create_expr(args[1]), "make.vector3.yz");
            auto y = _create_stack_variable(
                b->CreateExtractElement(yz, static_cast<uint64_t>(0u), "make.vector3.x"),
                "make.vector3.y.addr");
            auto z = _create_stack_variable(
                b->CreateExtractElement(yz, static_cast<uint64_t>(1u), "make.vector3.y"),
                "make.vector3.z.addr");
            return _make_float3(_create_expr(args[0]), y, z);
        }
        LUISA_ASSERT(args[0]->type()->is_vector() && args[0]->type()->dimension() == 2u &&
                         args[1]->type()->is_scalar(),
                     "Invalid argument types ('{}', '{}') to make {}.",
                     args[0]->type()->description(), args[1]->type()->description(),
                     t_vec->description());
        auto xy = b->CreateLoad(
            _create_type(args[0]->type()), _create_expr(args[0]), "make.vector3.xy");
        auto x = _create_stack_variable(
            b->CreateExtractElement(xy, static_cast<uint64_t>(0u), "make.vector3.x"),
            "make.vector3.x.addr");
        auto y = _create_stack_variable(
            b->CreateExtractElement(xy, static_cast<uint64_t>(1u), "make.vector3.y"),
            "make.vector3.y.addr");
        return _make_float3(x, y, _create_expr(args[1]));
    }
    if (args.size() == 3u) {
        LUISA_ASSERT(args[0]->type()->is_scalar() && args[1]->type()->is_scalar() &&
                         args[2]->type()->is_scalar(),
                     "Invalid argument types ('{}', '{}', '{}') for make-vector3.",
                     args[0]->type()->description(), args[1]->type()->description(),
                     args[2]->type()->description());
        return _make_float3(_create_expr(args[0]), _create_expr(args[1]), _create_expr(args[2]));
    }
    LUISA_ERROR_WITH_LOCATION("Invalid number of arguments '{}' for make-vector3.", args.size());
}

::llvm::Value *LLVMCodegen::_builtin_make_vector4_overloaded(const Type *t_vec, luisa::span<const Expression *const> args) noexcept {
    if (args.size() == 1u) {
        return _builtin_static_cast(t_vec, args[0]->type(), _create_expr(args[0]));
    }
    auto b = _current_context()->builder.get();
    if (args.size() == 2u) {
        // (x, yzw)
        if (args[0]->type()->is_scalar()) {
            LUISA_ASSERT(args[1]->type()->is_vector() && args[1]->type()->dimension() == 3u,
                         "Invalid argument types ('{}', '{}') to make {}.",
                         args[0]->type()->description(), args[1]->type()->description(),
                         t_vec->description());
            auto yzw = b->CreateLoad(
                _create_type(args[1]->type()), _create_expr(args[1]), "make.vector4.yzw");
            auto y = _create_stack_variable(
                b->CreateExtractElement(yzw, static_cast<uint64_t>(0u), "make.vector4.x"),
                "make.vector4.y.addr");
            auto z = _create_stack_variable(
                b->CreateExtractElement(yzw, static_cast<uint64_t>(1u), "make.vector4.y"),
                "make.vector4.z.addr");
            auto w = _create_stack_variable(
                b->CreateExtractElement(yzw, static_cast<uint64_t>(2u), "make.vector4.z"),
                "make.vector4.w.addr");
            return _make_float4(_create_expr(args[0]), y, z, w);
        }
        LUISA_ASSERT(args[0]->type()->is_vector(),
                     "Invalid argument types ('{}', '{}') to make {}.",
                     args[0]->type()->description(), args[1]->type()->description(),
                     t_vec->description());
        // (xyz, w)
        if (args[0]->type()->dimension() == 3u) {
            LUISA_ASSERT(args[1]->type()->is_scalar(),
                         "Invalid argument types ('{}', '{}') to make {}.",
                         args[0]->type()->description(), args[1]->type()->description(),
                         t_vec->description());
            auto xyz = b->CreateLoad(
                _create_type(args[0]->type()), _create_expr(args[0]), "make.vector4.xyz");
            auto x = _create_stack_variable(
                b->CreateExtractElement(xyz, static_cast<uint64_t>(0u), "make.vector4.x"),
                "make.vector4.x.addr");
            auto y = _create_stack_variable(
                b->CreateExtractElement(xyz, static_cast<uint64_t>(1u), "make.vector4.y"),
                "make.vector4.y.addr");
            auto z = _create_stack_variable(
                b->CreateExtractElement(xyz, static_cast<uint64_t>(2u), "make.vector4.z"),
                "make.vector4.z.addr");
            return _make_float4(x, y, z, _create_expr(args[1]));
        }
        // (xy, zw)
        LUISA_ASSERT(args[0]->type()->dimension() == 2u &&
                         args[1]->type()->is_vector() && args[1]->type()->dimension() == 2u,
                     "Invalid argument types ('{}', '{}') to make {}.",
                     args[0]->type()->description(), args[1]->type()->description(),
                     t_vec->description());
        auto xy = b->CreateLoad(
            _create_type(args[0]->type()), _create_expr(args[0]), "make.vector4.xy");
        auto zw = b->CreateLoad(
            _create_type(args[1]->type()), _create_expr(args[1]), "make.vector4.zw");
        auto x = _create_stack_variable(
            b->CreateExtractElement(xy, static_cast<uint64_t>(0u), "make.vector4.x"),
            "make.vector4.x.addr");
        auto y = _create_stack_variable(
            b->CreateExtractElement(xy, static_cast<uint64_t>(1u), "make.vector4.y"),
            "make.vector4.y.addr");
        auto z = _create_stack_variable(
            b->CreateExtractElement(zw, static_cast<uint64_t>(0u), "make.vector4.z"),
            "make.vector4.z.addr");
        auto w = _create_stack_variable(
            b->CreateExtractElement(zw, static_cast<uint64_t>(1u), "make.vector4.w"),
            "make.vector4.w.addr");
        return _make_float4(x, y, z, w);
    }
    if (args.size() == 3u) {
        ::llvm::SmallVector<::llvm::Value *, 4u> v;
        for (auto arg : args) {
            if (arg->type()->is_scalar()) {
                v.emplace_back(_create_expr(arg));
            } else {
                LUISA_ASSERT(arg->type()->is_vector() && arg->type()->dimension() == 2u,
                             "Invalid argument types ('{}', '{}', '{}') to make {}.",
                             args[0]->type()->description(), args[1]->type()->description(),
                             args[2]->type()->description(), t_vec->description());
                auto vec = b->CreateLoad(
                    _create_type(arg->type()), _create_expr(arg), "make.vector4.v");
                v.emplace_back(_create_stack_variable(
                    b->CreateExtractElement(vec, static_cast<uint64_t>(0u), "make.vector4.v.x"),
                    "make.vector4.v.x.addr"));
                v.emplace_back(_create_stack_variable(
                    b->CreateExtractElement(vec, static_cast<uint64_t>(1u), "make.vector4.v.y"),
                    "make.vector4.v.y.addr"));
            }
        }
        LUISA_ASSERT(v.size() == 4u, "Invalid argument types ('{}', '{}', '{}') to make {}.",
                     args[0]->type()->description(), args[1]->type()->description(),
                     args[2]->type()->description(), t_vec->description());
        return _make_float4(v[0], v[1], v[2], v[3]);
    }
    LUISA_ASSERT(args.size() == 4u &&
                     args[0]->type()->is_scalar() && args[1]->type()->is_scalar() &&
                     args[2]->type()->is_scalar() && args[3]->type()->is_scalar(),
                 "Invalid argument types ('{}', '{}', '{}', '{}') to make {}.",
                 args[0]->type()->description(), args[1]->type()->description(),
                 args[2]->type()->description(), args[3]->type()->description(),
                 t_vec->description());
    return _make_float4(_create_expr(args[0]), _create_expr(args[1]),
                        _create_expr(args[2]), _create_expr(args[3]));
}

::llvm::Value *LLVMCodegen::_builtin_make_matrix2_overloaded(luisa::span<const Expression *const> args) noexcept {
    if (args.size() == 1u) {
        return _builtin_static_cast(Type::of<float2x2>(), args[0]->type(), _create_expr(args[0]));
    }
    if (args.size() == 2u) {
        LUISA_ASSERT(args[0]->type()->is_vector() && args[0]->type()->dimension() == 2u &&
                         args[1]->type()->is_vector() && args[1]->type()->dimension() == 2u,
                     "Invalid argument types '{}' and '{}' for float2x2 constructor.",
                     args[0]->type()->description(), args[1]->type()->description());
        return _make_float2x2(_create_expr(args[0]), _create_expr(args[1]));
    }
    LUISA_ASSERT(args.size() == 4u,
                 "Invalid number of arguments '{}' for float2x2 constructor.",
                 args.size());
    LUISA_ASSERT(args[0]->type()->is_scalar() && args[1]->type()->is_scalar() &&
                     args[2]->type()->is_scalar() && args[3]->type()->is_scalar(),
                 "Invalid argument types ('{}', '{}', '{}', '{}') for float2x2 constructor.",
                 args[0]->type()->description(), args[1]->type()->description(),
                 args[2]->type()->description(), args[3]->type()->description());
    auto c0 = _make_float2(_create_expr(args[0]), _create_expr(args[1]));
    auto c1 = _make_float2(_create_expr(args[2]), _create_expr(args[3]));
    return _make_float2x2(c0, c1);
}

::llvm::Value *LLVMCodegen::_builtin_make_matrix3_overloaded(luisa::span<const Expression *const> args) noexcept {
    if (args.size() == 1u) {
        return _builtin_static_cast(Type::of<float3x3>(), args[0]->type(), _create_expr(args[0]));
    }
    if (args.size() == 3u) {
        LUISA_ASSERT(args[0]->type()->is_vector() && args[0]->type()->dimension() == 3u &&
                         args[1]->type()->is_vector() && args[1]->type()->dimension() == 3u &&
                         args[2]->type()->is_vector() && args[2]->type()->dimension() == 3u,
                     "Invalid argument types '{}' and '{}' for float3x3 constructor.",
                     args[0]->type()->description(), args[1]->type()->description(),
                     args[2]->type()->description());
        return _make_float3x3(_create_expr(args[0]), _create_expr(args[1]), _create_expr(args[2]));
    }
    LUISA_ASSERT(args.size() == 9u,
                 "Invalid number of arguments '{}' for float3x3 constructor.",
                 args.size());
    LUISA_ASSERT(args[0]->type()->is_scalar() && args[1]->type()->is_scalar() &&
                     args[2]->type()->is_scalar() && args[3]->type()->is_scalar() &&
                     args[4]->type()->is_scalar() && args[5]->type()->is_scalar() &&
                     args[6]->type()->is_scalar() && args[7]->type()->is_scalar() &&
                     args[8]->type()->is_scalar(),
                 "Invalid argument types ('{}', '{}', '{}', '{}', '{}', "
                 "'{}', '{}', '{}', '{}') for float3x3 constructor.",
                 args[0]->type()->description(), args[1]->type()->description(),
                 args[2]->type()->description(), args[3]->type()->description(),
                 args[4]->type()->description(), args[5]->type()->description(),
                 args[6]->type()->description(), args[7]->type()->description(),
                 args[8]->type()->description());
    auto c0 = _make_float3(_create_expr(args[0]), _create_expr(args[1]), _create_expr(args[2]));
    auto c1 = _make_float3(_create_expr(args[3]), _create_expr(args[4]), _create_expr(args[5]));
    auto c2 = _make_float3(_create_expr(args[6]), _create_expr(args[7]), _create_expr(args[8]));
    return _make_float3x3(c0, c1, c2);
}

::llvm::Value *LLVMCodegen::_builtin_make_matrix4_overloaded(luisa::span<const Expression *const> args) noexcept {
    if (args.size() == 1u) {
        return _builtin_static_cast(Type::of<float4x4>(), args[0]->type(), _create_expr(args[0]));
    }
    if (args.size() == 4u) {
        LUISA_ASSERT(args[0]->type()->is_vector() && args[0]->type()->dimension() == 4u &&
                         args[1]->type()->is_vector() && args[1]->type()->dimension() == 4u &&
                         args[2]->type()->is_vector() && args[2]->type()->dimension() == 4u &&
                         args[3]->type()->is_vector() && args[3]->type()->dimension() == 4u,
                     "Invalid argument types '{}' and '{}' for float4x4 constructor.",
                     args[0]->type()->description(), args[1]->type()->description(),
                     args[2]->type()->description(), args[3]->type()->description());
        return _make_float4x4(_create_expr(args[0]), _create_expr(args[1]),
                              _create_expr(args[2]), _create_expr(args[3]));
    }
    LUISA_ASSERT(args.size() == 16u,
                 "Invalid number of arguments '{}' for float4x4 constructor.",
                 args.size());
    LUISA_ASSERT(args[0]->type()->is_scalar() && args[1]->type()->is_scalar() &&
                     args[2]->type()->is_scalar() && args[3]->type()->is_scalar() &&
                     args[4]->type()->is_scalar() && args[5]->type()->is_scalar() &&
                     args[6]->type()->is_scalar() && args[7]->type()->is_scalar() &&
                     args[8]->type()->is_scalar() && args[9]->type()->is_scalar() &&
                     args[10]->type()->is_scalar() && args[11]->type()->is_scalar() &&
                     args[12]->type()->is_scalar() && args[13]->type()->is_scalar() &&
                     args[14]->type()->is_scalar() && args[15]->type()->is_scalar(),
                 "Invalid argument types ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', "
                 "'{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}') for float4x4 constructor.",
                 args[0]->type()->description(), args[1]->type()->description(),
                 args[2]->type()->description(), args[3]->type()->description(),
                 args[4]->type()->description(), args[5]->type()->description(),
                 args[6]->type()->description(), args[7]->type()->description(),
                 args[8]->type()->description(), args[9]->type()->description(),
                 args[10]->type()->description(), args[11]->type()->description(),
                 args[12]->type()->description(), args[13]->type()->description(),
                 args[14]->type()->description(), args[15]->type()->description());
    auto c0 = _make_float4(_create_expr(args[0]), _create_expr(args[1]),
                           _create_expr(args[2]), _create_expr(args[3]));
    auto c1 = _make_float4(_create_expr(args[4]), _create_expr(args[5]),
                           _create_expr(args[6]), _create_expr(args[7]));
    auto c2 = _make_float4(_create_expr(args[8]), _create_expr(args[9]),
                           _create_expr(args[10]), _create_expr(args[11]));
    auto c3 = _make_float4(_create_expr(args[12]), _create_expr(args[13]),
                           _create_expr(args[14]), _create_expr(args[15]));
    return _make_float4x4(c0, c1, c2, c3);
}

}// namespace luisa::compute::llvm
