//
// Created by Mike Smith on 2022/5/23.
//

#include <numeric>

#include <dsl/sugar.h>
#include <rtx/hit.h>
#include <backends/llvm/llvm_codegen.h>

namespace luisa::compute::llvm {

[[nodiscard]] static Function float2x2_inverse() noexcept {
    static Callable inverse = [](Float2x2 m) noexcept {
        auto inv_det = 1.0f / (m[0][0] * m[1][1] - m[1][0] * m[0][1]);
        return inv_det * make_float2x2(m[1][1], -m[0][1], -m[1][0], +m[0][0]);
    };
    return inverse.function();
}

[[nodiscard]] static Function float3x3_inverse() noexcept {
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

[[nodiscard]] static Function float4x4_inverse() noexcept {
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

[[nodiscard]] static Function float2x2_det() noexcept {
    static Callable inverse = [](Float2x2 m) noexcept {
        return m[0][0] * m[1][1] - m[1][0] * m[0][1];
    };
    return inverse.function();
}

[[nodiscard]] static Function float3x3_det() noexcept {
    static Callable inverse = [](Float3x3 m) noexcept {
        return m[0].x * (m[1].y * m[2].z - m[2].y * m[1].z) -
               m[1].x * (m[0].y * m[2].z - m[2].y * m[0].z) +
               m[2].x * (m[0].y * m[1].z - m[1].y * m[0].z);
    };
    return inverse.function();
}

[[nodiscard]] static Function float4x4_det() noexcept {
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

[[nodiscard]] static Function float2x2_transpose() noexcept {
    static Callable inverse = [](Float2x2 m) noexcept {
        return make_float2x2(m[0].x, m[1].x, m[0].y, m[1].y);
    };
    return inverse.function();
}

[[nodiscard]] static Function float3x3_transpose() noexcept {
    static Callable inverse = [](Float3x3 m) noexcept {
        return make_float3x3(
            m[0].x, m[1].x, m[2].x,
            m[0].y, m[1].y, m[2].y,
            m[0].z, m[1].z, m[2].z);
    };
    return inverse.function();
}

[[nodiscard]] static Function float4x4_transpose() noexcept {
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
        if (t->dimension() == 2u) { return float2x2_inverse(); }
        if (t->dimension() == 3u) { return float3x3_inverse(); }
        if (t->dimension() == 4u) { return float4x4_inverse(); }
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
        if (t->dimension() == 2u) { return float2x2_det(); }
        if (t->dimension() == 3u) { return float3x3_det(); }
        if (t->dimension() == 4u) { return float4x4_det(); }
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
        if (t->dimension() == 2u) { return float2x2_transpose(); }
        if (t->dimension() == 3u) { return float3x3_transpose(); }
        if (t->dimension() == 4u) { return float4x4_transpose(); }
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
    auto f = module->getFunction(::llvm::StringRef{func_name.data(), func_name.size()});
    auto ir_type = ::llvm::Type::getFloatTy(module->getContext());
    if (f == nullptr) {
        ::llvm::SmallVector<::llvm::Type *, 2u> arg_types(n_args, ir_type);
        f = ::llvm::Function::Create(
            ::llvm::FunctionType::get(ir_type, arg_types, false),
            ::llvm::Function::ExternalLinkage,
            ::llvm::StringRef{func_name.data(), func_name.size()},
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
    ::llvm::Module *module, ::llvm::IRBuilder<> *builder, const Type *t, luisa::string_view name,
    ::llvm::SmallVector<::llvm::Type *, 2u> t_args, ::llvm::SmallVector<::llvm::Value *, 2u> p_args) noexcept {
    auto f = _declare_external_math_function(module, name, p_args.size());
    ::llvm::SmallVector<::llvm::Value *, 2u> args;
    for (auto i = 0u; i < p_args.size(); i++) {
        auto value_name = luisa::format("{}.arg{}", name, i);
        args.emplace_back(builder->CreateLoad(
            t_args[i], p_args[i],
            ::llvm::StringRef{value_name.data(), value_name.size()}));
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
        auto vec_type = ::llvm::VectorType::get(
            ::llvm::Type::getFloatTy(module->getContext()), t->dimension(), false);
        auto vec = static_cast<::llvm::Value *>(::llvm::UndefValue::get(vec_type));
        for (auto i = 0u; i < t->dimension(); i++) {
            vec = builder->CreateInsertElement(vec, v[i], i);
        }
        auto p_vec = builder->CreateAlloca(vec_type);
        p_vec->setAlignment(::llvm::Align{16});
        builder->CreateStore(vec, p_vec);
        return p_vec;
    }
    // scalar
    auto y_name = luisa::format("{}.call", name);
    auto y = builder->CreateCall(f, args, ::llvm::StringRef{y_name.data(), y_name.size()});
    auto py_name = luisa::format("{}.call.addr", name);
    auto py = builder->CreateAlloca(
        t_args.front(), nullptr, ::llvm::StringRef{py_name.data(), py_name.size()});
    py->setAlignment(::llvm::Align{16});
    builder->CreateStore(y, py);
    return py;
}
#pragma clang diagnostic pop

::llvm::Value *LLVMCodegen::_builtin_instance_transform(::llvm::Value *accel, ::llvm::Value *p_index) noexcept {
    static Callable impl = [](BufferVar<LLVMAccelInstance> instances, UInt index) noexcept {
        auto m = instances.read(index).affine;
        return make_float4x4(
            m[0], m[4], m[8], 0.f,
            m[1], m[5], m[9], 0.f,
            m[2], m[6], m[10], 0.f,
            m[3], m[7], m[11], 1.f);
    };
    auto func = _create_function(impl.function());
    auto b = _current_context()->builder.get();
    auto instances = b->CreateExtractValue(accel, 1, "accel.instance.transform.instances");
    auto index = b->CreateLoad(b->getInt32Ty(), p_index, "accel.instance.transform.index");
    auto ret = b->CreateCall(func->getFunctionType(), func, {instances, index}, "accel.instance.transform.ret");
    return _create_stack_variable(ret, "accel.instance.transform.ret.addr");
}

void LLVMCodegen::_builtin_set_instance_transform(::llvm::Value *accel, ::llvm::Value *p_index, ::llvm::Value *p_mat) noexcept {
    static Callable impl = [](BufferVar<LLVMAccelInstance> instances, UInt index, Float4x4 m) noexcept {
        auto inst = instances.read(index);
        inst.dirty = true;
        inst.affine[0] = m[0].x;
        inst.affine[1] = m[1].x;
        inst.affine[2] = m[2].x;
        inst.affine[3] = m[3].x;
        inst.affine[4] = m[0].y;
        inst.affine[5] = m[1].y;
        inst.affine[6] = m[2].y;
        inst.affine[7] = m[3].y;
        inst.affine[8] = m[0].z;
        inst.affine[9] = m[1].z;
        inst.affine[10] = m[2].z;
        inst.affine[11] = m[3].z;
        instances.write(index, inst);
    };
    auto func = _create_function(impl.function());
    auto b = _current_context()->builder.get();
    auto instances = b->CreateExtractValue(accel, 1, "accel.instance.transform.instances");
    auto index = b->CreateLoad(b->getInt32Ty(), p_index, "accel.instance.transform.index");
    auto mat = b->CreateLoad(b->getFloatTy()->getPointerTo(), p_mat, "accel.instance.transform.mat");
    b->CreateCall(func->getFunctionType(), func, {instances, index, mat});
}

void LLVMCodegen::_builtin_set_instance_visibility(::llvm::Value *accel, ::llvm::Value *p_index, ::llvm::Value *p_vis) noexcept {
    auto b = _current_context()->builder.get();
    auto ptr = b->CreateExtractValue(accel, 1, "accel.instance.visibility.instances");
    auto index = b->CreateLoad(b->getInt32Ty(), p_index, "accel.instance.visibility.index");
    auto t_instance = _create_type(Type::of<LLVMAccelInstance>());
    auto ptr_vis = b->CreateInBoundsGEP(t_instance, ptr, {index, _literal(1)}, "accel.instance.visibility.vis.ptr");
    auto ptr_dirty = b->CreateInBoundsGEP(t_instance, ptr, {index, _literal(2)}, "accel.instance.visibility.dirty.ptr");
    auto vis = b->CreateLoad(b->getInt32Ty(), p_vis, "accel.instance.visibility.vis");
    b->CreateStore(vis, ptr_vis);
    b->CreateStore(_literal(true), ptr_dirty);
}

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
            _module, builder, args[0]->type(), "acos",
            {_create_type(args[0]->type())}, {_create_expr(args[0])});
        case CallOp::ACOSH: return _builtin_acosh(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::ASIN: return _call_external_math_function(
            _module, builder, args[0]->type(), "asin",
            {_create_type(args[0]->type())}, {_create_expr(args[0])});
        case CallOp::ASINH: return _builtin_asinh(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::ATAN: return _call_external_math_function(
            _module, builder, args[0]->type(), "atan",
            {_create_type(args[0]->type())}, {_create_expr(args[0])});
        case CallOp::ATAN2: return _call_external_math_function(
            _module, builder, args[0]->type(), "atan2",
            {_create_type(args[0]->type()), _create_type(args[1]->type())},
            {_create_expr(args[0]), _create_expr(args[1])});
        case CallOp::ATANH: return _builtin_atanh(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::COS: return _builtin_cos(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::COSH: return _builtin_cosh(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::SIN: return _builtin_sin(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::SINH: return _builtin_sinh(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::TAN: return _builtin_tan(
            args[0]->type(), _create_expr(args[0]));
        case CallOp::TANH: return _builtin_tanh(
            args[0]->type(), _create_expr(args[0]));
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
            _builtin_synchronize_block();
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
            ret_type, _create_expr(args[0]), args[1]->type(), _create_expr(args[1]));
        case CallOp::TEXTURE_WRITE:
            _builtin_texture_write(
                args[2]->type(), _create_expr(args[0]), args[1]->type(),
                _create_expr(args[1]), _create_expr(args[2]));
            return nullptr;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE: return _builtin_bindless_texture_sample2d(
            _create_expr(args[0]), _create_expr(args[1]), _create_expr(args[2]));
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: return _builtin_bindless_texture_sample2d_level(
            _create_expr(args[0]), _create_expr(args[1]), _create_expr(args[2]), _create_expr(args[3]));
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: return _builtin_bindless_texture_sample2d_grad(
            _create_expr(args[0]), _create_expr(args[1]), _create_expr(args[2]),
            _create_expr(args[3]), _create_expr(args[4]));
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE: return _builtin_bindless_texture_sample3d(
            _create_expr(args[0]), _create_expr(args[1]), _create_expr(args[2]));
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: return _builtin_bindless_texture_sample3d_level(
            _create_expr(args[0]), _create_expr(args[1]), _create_expr(args[2]), _create_expr(args[3]));
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: return _builtin_bindless_texture_sample3d_grad(
            _create_expr(args[0]), _create_expr(args[1]), _create_expr(args[2]),
            _create_expr(args[3]), _create_expr(args[4]));
        case CallOp::BINDLESS_TEXTURE2D_READ: return _builtin_bindless_texture_read2d(
            _create_expr(args[0]), _create_expr(args[1]), nullptr, _create_expr(args[2]));
        case CallOp::BINDLESS_TEXTURE3D_READ: return _builtin_bindless_texture_read3d(
            _create_expr(args[0]), _create_expr(args[1]), nullptr, _create_expr(args[2]));
        case CallOp::BINDLESS_TEXTURE2D_READ_LEVEL: return _builtin_bindless_texture_read2d(
            _create_expr(args[0]), _create_expr(args[1]), _create_expr(args[3]), _create_expr(args[2]));
        case CallOp::BINDLESS_TEXTURE3D_READ_LEVEL: return _builtin_bindless_texture_read3d(
            _create_expr(args[0]), _create_expr(args[1]), _create_expr(args[3]), _create_expr(args[2]));
        case CallOp::BINDLESS_TEXTURE2D_SIZE: return _builtin_bindless_texture_size2d(
            _create_expr(args[0]), _create_expr(args[1]), nullptr);
        case CallOp::BINDLESS_TEXTURE3D_SIZE: return _builtin_bindless_texture_size3d(
            _create_expr(args[0]), _create_expr(args[1]), nullptr);
        case CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL: return _builtin_bindless_texture_size2d(
            _create_expr(args[0]), _create_expr(args[1]), _create_expr(args[2]));
        case CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: return _builtin_bindless_texture_size3d(
            _create_expr(args[0]), _create_expr(args[1]), _create_expr(args[2]));
        case CallOp::BINDLESS_BUFFER_READ: return _builtin_bindless_buffer_read(
            ret_type, _create_expr(args[0]), _create_expr(args[1]), _create_expr(args[2]));
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
        case CallOp::INSTANCE_TO_WORLD_MATRIX: return _builtin_instance_transform(
            _create_expr(args[0]), _create_expr(args[1]));
        case CallOp::SET_INSTANCE_TRANSFORM:
            _builtin_set_instance_transform(
                _create_expr(args[0]), _create_expr(args[1]), _create_expr(args[2]));
            return nullptr;
        case CallOp::SET_INSTANCE_VISIBILITY:
            _builtin_set_instance_visibility(
                _create_expr(args[0]), _create_expr(args[1]), _create_expr(args[2]));
            return nullptr;
        case CallOp::TRACE_CLOSEST: return _builtin_trace_closest(
            _create_expr(args[0]), _create_expr(args[1]));
        case CallOp::TRACE_ANY: return _builtin_trace_any(
            _create_expr(args[0]), _create_expr(args[1]));
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid built-in call.");
}

void LLVMCodegen::_builtin_synchronize_block() noexcept {
    auto ctx = _current_context();
    auto b = ctx->builder.get();
    auto i8_type = ::llvm::Type::getInt8Ty(_context);
    auto i1_type = ::llvm::Type::getInt1Ty(_context);
    auto coro_state = b->CreateIntrinsic(
        ::llvm::Intrinsic::coro_suspend, {},
        {::llvm::ConstantTokenNone::get(_context),
         ::llvm::ConstantInt::get(i1_type, false)},
        nullptr, "synchronize.block.state");
    auto resume = ::llvm::BasicBlock::Create(
        _context, "synchronize.block.resume", ctx->ir);
    resume->moveAfter(b->GetInsertBlock());
    auto coro_switch = b->CreateSwitch(
        coro_state, ctx->coro_suspend, 2u);
    coro_switch->addCase(
        ::llvm::ConstantInt::get(i8_type, 0), resume);
    coro_switch->addCase(
        ::llvm::ConstantInt::get(i8_type, 1), ctx->coro_cleanup);
    b->SetInsertPoint(resume);
}

[[nodiscard]] inline auto is_scalar_or_vector(const Type *t, Type::Tag tag) noexcept {
    return t->tag() == tag ||
           (t->is_vector() && t->element()->tag() == tag);
}

::llvm::Value *LLVMCodegen::_builtin_all(const Type *t, ::llvm::Value *v) noexcept {
    auto b = _current_context()->builder.get();
    auto pred_type = ::llvm::FixedVectorType::get(b->getInt1Ty(), t->dimension());
    v = b->CreateLoad(_create_type(t), v, "all.load");
    v = b->CreateTrunc(v, pred_type, "all.pred");
    return _create_stack_variable(b->CreateAndReduce(v), "all.addr");
}

::llvm::Value *LLVMCodegen::_builtin_any(const Type *t, ::llvm::Value *v) noexcept {
    auto b = _current_context()->builder.get();
    auto pred_type = ::llvm::FixedVectorType::get(b->getInt1Ty(), t->dimension());
    v = b->CreateLoad(_create_type(t), v, "any.load");
    v = b->CreateTrunc(v, pred_type, "any.pred");
    return _create_stack_variable(b->CreateOrReduce(v), "any.addr");
}

::llvm::Value *LLVMCodegen::_builtin_select(const Type *t_pred, const Type *t_value,
                                            ::llvm::Value *pred, ::llvm::Value *v_true, ::llvm::Value *v_false) noexcept {
    auto b = _current_context()->builder.get();
    auto pred_type = static_cast<::llvm::Type *>(b->getInt1Ty());
    if (t_pred->is_vector()) { pred_type = ::llvm::FixedVectorType::get(pred_type, t_pred->dimension()); }
    auto pred_load = b->CreateLoad(_create_type(t_pred), pred, "sel.pred.load");
    auto bv = b->CreateTrunc(pred_load, pred_type, "sel.pred.bv");
    auto v_true_load = b->CreateLoad(_create_type(t_value), v_true, "sel.true");
    auto v_false_load = b->CreateLoad(_create_type(t_value), v_false, "sel.false");
    auto result = b->CreateSelect(bv, v_true_load, v_false_load, "sel");
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
    auto b = _current_context()->builder.get();
    if (is_scalar_or_vector(t, Type::Tag::UINT)) { return x; }
    auto ir_type = _create_type(t);
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        auto m = b->CreateIntrinsic(
            ::llvm::Intrinsic::abs, {ir_type},
            {b->CreateLoad(ir_type, x, "iabs.x")},
            nullptr, "iabs");
        return _create_stack_variable(m, "iabs.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        auto m = b->CreateIntrinsic(
            ::llvm::Intrinsic::fabs, {ir_type},
            {b->CreateLoad(ir_type, x, "fabs.x")},
            nullptr, "fabs");
        return _create_stack_variable(m, "fabs.addr");
    }
    LUISA_ERROR_WITH_LOCATION("Invalid type '{}' for abs", t->description());
}

::llvm::Value *LLVMCodegen::_builtin_min(const Type *t, ::llvm::Value *x, ::llvm::Value *y) noexcept {
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        auto m = b->CreateIntrinsic(
            ::llvm::Intrinsic::minnum, {ir_type},
            {b->CreateLoad(ir_type, x, "fmin.x"),
             b->CreateLoad(ir_type, y, "fmin.y")},
            nullptr, "fmin");
        return _create_stack_variable(m, "fmin.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        auto m = b->CreateIntrinsic(
            ::llvm::Intrinsic::smin, {ir_type},
            {b->CreateLoad(ir_type, x, "imin.x"),
             b->CreateLoad(ir_type, y, "imin.y")},
            nullptr, "imin");
        return _create_stack_variable(m, "imin.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        auto m = b->CreateIntrinsic(
            ::llvm::Intrinsic::umin, {ir_type},
            {b->CreateLoad(ir_type, x, "umin.x"),
             b->CreateLoad(ir_type, y, "umin.y")},
            nullptr, "umin");
        return _create_stack_variable(m, "umin.addr");
    }
    LUISA_ERROR_WITH_LOCATION("Invalid type '{}' for min.", t->description());
}

::llvm::Value *LLVMCodegen::_builtin_max(const Type *t, ::llvm::Value *x, ::llvm::Value *y) noexcept {
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        auto m = b->CreateIntrinsic(
            ::llvm::Intrinsic::maxnum, {ir_type},
            {b->CreateLoad(ir_type, x, "fmax.x"),
             b->CreateLoad(ir_type, y, "fmax.y")},
            nullptr, "fmax");
        return _create_stack_variable(m, "fmax.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        auto m = b->CreateIntrinsic(
            ::llvm::Intrinsic::smax, {ir_type},
            {b->CreateLoad(ir_type, x, "imax.x"),
             b->CreateLoad(ir_type, y, "imax.y")},
            nullptr, "imax");
        return _create_stack_variable(m, "imax.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        auto m = b->CreateIntrinsic(
            ::llvm::Intrinsic::umax, {ir_type},
            {b->CreateLoad(ir_type, x, "umax.x"),
             b->CreateLoad(ir_type, y, "umax.y")},
            nullptr, "umax");
        return _create_stack_variable(m, "umax.addr");
    }
    LUISA_ERROR_WITH_LOCATION("Invalid type '{}' for max.", t->description());
}

::llvm::Value *LLVMCodegen::_builtin_clz(const Type *t, ::llvm::Value *p) noexcept {
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    auto x = b->CreateIntrinsic(
        ::llvm::Intrinsic::ctlz, {ir_type},
        {b->CreateLoad(ir_type, p, "clz.x")},
        nullptr, "clz");
    return _create_stack_variable(x, "clz.addr");
}

::llvm::Value *LLVMCodegen::_builtin_ctz(const Type *t, ::llvm::Value *p) noexcept {
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    auto x = b->CreateIntrinsic(
        ::llvm::Intrinsic::cttz, {ir_type},
        {b->CreateLoad(ir_type, p, "ctz.x")},
        nullptr, "ctz");
    return _create_stack_variable(x, "ctz.addr");
}

::llvm::Value *LLVMCodegen::_builtin_popcount(const Type *t, ::llvm::Value *p) noexcept {
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    auto x = b->CreateIntrinsic(
        ::llvm::Intrinsic::ctpop, {ir_type},
        {b->CreateLoad(ir_type, p, "popcount.x")},
        nullptr, "popcount");
    return _create_stack_variable(x, "popcount.addr");
}

::llvm::Value *LLVMCodegen::_builtin_reverse(const Type *t, ::llvm::Value *p) noexcept {
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    auto x = b->CreateIntrinsic(
        ::llvm::Intrinsic::bitreverse, {ir_type},
        {b->CreateLoad(ir_type, p, "reverse.x")},
        nullptr, "reverse");
    return _create_stack_variable(x, "reverse.addr");
}

::llvm::Value *LLVMCodegen::_builtin_fma(const Type *t, ::llvm::Value *pa, ::llvm::Value *pb, ::llvm::Value *pc) noexcept {
    auto ir_type = _create_type(t);
    auto b = _current_context()->builder.get();
    auto m = b->CreateIntrinsic(
        ::llvm::Intrinsic::fma, {ir_type},
        {b->CreateLoad(ir_type, pa, "fma.a"),
         b->CreateLoad(ir_type, pb, "fma.b"),
         b->CreateLoad(ir_type, pc, "fma.c")},
        nullptr, "fma");
    return _create_stack_variable(m, "fma.addr");
}

::llvm::Value *LLVMCodegen::_builtin_exp(const Type *t, ::llvm::Value *v) noexcept {
    auto ir_type = _create_type(t);
    auto b = _current_context()->builder.get();
    auto m = b->CreateIntrinsic(
        ::llvm::Intrinsic::exp, {ir_type},
        {b->CreateLoad(ir_type, v, "exp.x")},
        nullptr, "exp");
    return _create_stack_variable(m, "exp.addr");
}

::llvm::Value *LLVMCodegen::_builtin_exp2(const Type *t, ::llvm::Value *v) noexcept {
    auto ir_type = _create_type(t);
    auto b = _current_context()->builder.get();
    auto m = b->CreateIntrinsic(
        ::llvm::Intrinsic::exp2, {ir_type},
        {b->CreateLoad(ir_type, v, "exp2.x")},
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
    auto b = _current_context()->builder.get();
    auto m = b->CreateIntrinsic(
        ::llvm::Intrinsic::log, {ir_type},
        {b->CreateLoad(ir_type, v, "log.x")},
        nullptr, "log");
    return _create_stack_variable(m, "log.addr");
}

::llvm::Value *LLVMCodegen::_builtin_log2(const Type *t, ::llvm::Value *v) noexcept {
    auto ir_type = _create_type(t);
    auto b = _current_context()->builder.get();
    auto m = b->CreateIntrinsic(
        ::llvm::Intrinsic::log2, {ir_type},
        {b->CreateLoad(ir_type, v, "log2.x")},
        nullptr, "log2");
    return _create_stack_variable(m, "log2.addr");
}

::llvm::Value *LLVMCodegen::_builtin_log10(const Type *t, ::llvm::Value *v) noexcept {
    auto ir_type = _create_type(t);
    auto b = _current_context()->builder.get();
    auto m = b->CreateIntrinsic(
        ::llvm::Intrinsic::log10, {ir_type},
        {b->CreateLoad(ir_type, v, "log10.x")},
        nullptr, "log10");
    return _create_stack_variable(m, "log10.addr");
}

::llvm::Value *LLVMCodegen::_builtin_pow(const Type *t, ::llvm::Value *x, ::llvm::Value *y) noexcept {
    auto ir_type = _create_type(t);
    auto b = _current_context()->builder.get();
    auto m = b->CreateIntrinsic(
        ::llvm::Intrinsic::pow, {ir_type},
        {b->CreateLoad(ir_type, x, "pow.x"),
         b->CreateLoad(ir_type, y, "pow.y")},
        nullptr, "pow.ret");
    return _create_stack_variable(m, "pow.addr");
}

::llvm::Value *LLVMCodegen::_builtin_copysign(const Type *t, ::llvm::Value *x, ::llvm::Value *y) noexcept {
    auto ir_type = _create_type(t);
    auto b = _current_context()->builder.get();
    auto m = b->CreateIntrinsic(
        ::llvm::Intrinsic::copysign, {ir_type},
        {b->CreateLoad(ir_type, x, "copysign.x"),
         b->CreateLoad(ir_type, y, "copysign.y")},
        nullptr, "copysign");
    return _create_stack_variable(m, "copysign.addr");
}

::llvm::Value *LLVMCodegen::_builtin_faceforward(const Type *t, ::llvm::Value *n, ::llvm::Value *i, ::llvm::Value *nref) noexcept {
    auto ir_type = _create_type(t);
    auto b = _current_context()->builder.get();
    auto dot = b->CreateLoad(
        _create_type(Type::of<float>()),
        _builtin_dot(t, nref, i), "faceforward.dot");
    auto pos_n = b->CreateLoad(
        ir_type, n, "faceforward.pos_n");
    auto neg_n = b->CreateFNeg(
        pos_n, "faceforward.neg_n");
    auto m = b->CreateSelect(
        b->CreateFCmpOLT(dot, _literal(0.f)),
        pos_n, neg_n, "faceforward.select");
    return _create_stack_variable(m, "faceforward.addr");
}

::llvm::Value *LLVMCodegen::_builtin_sin(const Type *t, ::llvm::Value *v) noexcept {
    auto ir_type = _create_type(t);
    auto b = _current_context()->builder.get();
    auto m = b->CreateIntrinsic(
        ::llvm::Intrinsic::sin, {ir_type},
        {b->CreateLoad(ir_type, v, "sin.x")},
        nullptr, "sin");
    return _create_stack_variable(m, "sin.addr");
}

::llvm::Value *LLVMCodegen::_builtin_cos(const Type *t, ::llvm::Value *v) noexcept {
    auto ir_type = _create_type(t);
    auto b = _current_context()->builder.get();
    auto m = b->CreateIntrinsic(
        ::llvm::Intrinsic::cos, {ir_type},
        {b->CreateLoad(ir_type, v, "cos.x")},
        nullptr, "cos");
    return _create_stack_variable(m, "cos.addr");
}

::llvm::Value *LLVMCodegen::_builtin_tan(const Type *t, ::llvm::Value *v) noexcept {
    auto one = _create_stack_variable(_literal(1.f), "tan.one");
    if (t->is_vector()) { one = _builtin_static_cast(t, t->element(), one); }
    return _builtin_mul(t, _builtin_sin(t, v), _builtin_div(t, one, _builtin_cos(t, v)));
}

::llvm::Value *LLVMCodegen::_builtin_sqrt(const Type *t, ::llvm::Value *x) noexcept {
    auto ir_type = _create_type(t);
    auto b = _current_context()->builder.get();
    auto m = b->CreateIntrinsic(
        ::llvm::Intrinsic::sqrt, {ir_type},
        {b->CreateLoad(ir_type, x, "sqrt.x")},
        nullptr, "sqrt");
    return _create_stack_variable(m, "sqrt.addr");
}

::llvm::Value *LLVMCodegen::_builtin_fract(const Type *t, ::llvm::Value *v) noexcept {
    return _builtin_sub(t, v, _builtin_floor(t, v));
}

::llvm::Value *LLVMCodegen::_builtin_floor(const Type *t, ::llvm::Value *v) noexcept {
    auto ir_type = _create_type(t);
    auto b = _current_context()->builder.get();
    auto m = b->CreateIntrinsic(
        ::llvm::Intrinsic::floor, {ir_type},
        {b->CreateLoad(ir_type, v, "floor.x")},
        nullptr, "floor");
    return _create_stack_variable(m, "floor.addr");
}

::llvm::Value *LLVMCodegen::_builtin_ceil(const Type *t, ::llvm::Value *v) noexcept {
    auto ir_type = _create_type(t);
    auto b = _current_context()->builder.get();
    auto m = b->CreateIntrinsic(
        ::llvm::Intrinsic::ceil, {ir_type},
        {b->CreateLoad(ir_type, v, "ceil.x")},
        nullptr, "ceil");
    return _create_stack_variable(m, "ceil.addr");
}

::llvm::Value *LLVMCodegen::_builtin_trunc(const Type *t, ::llvm::Value *v) noexcept {
    auto ir_type = _create_type(t);
    auto b = _current_context()->builder.get();
    auto m = b->CreateIntrinsic(
        ::llvm::Intrinsic::trunc, {ir_type},
        {b->CreateLoad(ir_type, v, "trunc.x")},
        nullptr, "trunc");
    return _create_stack_variable(m, "trunc.addr");
}

::llvm::Value *LLVMCodegen::_builtin_round(const Type *t, ::llvm::Value *v) noexcept {
    auto ir_type = _create_type(t);
    auto b = _current_context()->builder.get();
    auto m = b->CreateIntrinsic(
        ::llvm::Intrinsic::round, {ir_type},
        {b->CreateLoad(ir_type, v, "round.x")},
        nullptr, "round");
    return _create_stack_variable(m, "round.addr");
}

::llvm::Value *LLVMCodegen::_builtin_rsqrt(const Type *t, ::llvm::Value *x) noexcept {
    auto s = _builtin_sqrt(t, x);
    auto one = _builtin_static_cast(
        t, Type::of<float>(),
        _create_stack_variable(_literal(1.f), "rsqrt.one"));
    return _builtin_div(t, one, s);
}

static constexpr auto atomic_operation_order = ::llvm::AtomicOrdering::Monotonic;

::llvm::Value *LLVMCodegen::_builtin_atomic_exchange(const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_desired) noexcept {
    auto b = _current_context()->builder.get();
    auto desired = b->CreateLoad(
        _create_type(t), p_desired, "atomic.exchange.desired");
    auto old = b->CreateAtomicRMW(
        ::llvm::AtomicRMWInst::Xchg, p_atomic,
        desired, {}, atomic_operation_order);
    old->setName("atomic.exchange.old");
    return _create_stack_variable(old, "atomic.exchange.addr");
}

::llvm::Value *LLVMCodegen::_builtin_atomic_compare_exchange(const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_expected, ::llvm::Value *p_desired) noexcept {
    auto b = _current_context()->builder.get();
    auto expected = static_cast<::llvm::Value *>(b->CreateLoad(
        _create_type(t), p_expected, "atomic.compare.exchange.expected"));
    auto desired = static_cast<::llvm::Value *>(b->CreateLoad(
        _create_type(t), p_desired, "atomic.compare.exchange.desired"));
    if (t->tag() == Type::Tag::FLOAT) {
        expected = b->CreateBitCast(
            expected, b->getInt32Ty(),
            "atomic.compare.exchange.expected.int");
        desired = b->CreateBitCast(
            desired, b->getInt32Ty(),
            "atomic.compare.exchange.desired.int");
        p_atomic = b->CreateBitOrPointerCast(
            p_atomic, ::llvm::PointerType::get(b->getInt32Ty(), 0),
            "atomic.compare.exchange.atomic.int");
    }
    auto old_and_success = b->CreateAtomicCmpXchg(
        p_atomic, expected, desired, {},
        atomic_operation_order,
        atomic_operation_order);
    old_and_success->setName("atomic.compare.exchange.old_and_success");
    auto old = b->CreateExtractValue(
        old_and_success, 0, "atomic.compare.exchange.old");
    if (t->tag() == Type::Tag::FLOAT) {
        old = b->CreateBitCast(
            old, b->getFloatTy(),
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
    p_old->setAlignment(::llvm::Align{16});
    builder->CreateStore(old, p_old);
    return p_old;
#else
    auto p_old = builder->CreateAlloca(builder->getFloatTy(), nullptr, "atomic.fetch.add.old.addr");
    p_old->setAlignment(::llvm::Align{16});
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
    auto b = _current_context()->builder.get();
    auto value = b->CreateLoad(
        _create_type(t), p_value, "atomic.fetch.add.value");
    if (t->tag() == Type::Tag::FLOAT) {
        return _atomic_fetch_add_float(b, p_atomic, value);
    }
    auto old = b->CreateAtomicRMW(
        ::llvm::AtomicRMWInst::Add, p_atomic,
        value, {}, atomic_operation_order);
    old->setName("atomic.fetch.add.old");
    return _create_stack_variable(old, "atomic.fetch.add.addr");
}

::llvm::Value *LLVMCodegen::_builtin_atomic_fetch_sub(const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_value) noexcept {
    auto b = _current_context()->builder.get();
    if (t->tag() == Type::Tag::FLOAT) {
        return _builtin_atomic_fetch_add(
            t, p_atomic, _builtin_unary_minus(t, p_value));
    }
    auto value = b->CreateLoad(
        _create_type(t), p_value, "atomic.fetch.sub.value");
    auto old = b->CreateAtomicRMW(
        ::llvm::AtomicRMWInst::Sub, p_atomic,
        value, {}, atomic_operation_order);
    old->setName("atomic.fetch.sub.old");
    return _create_stack_variable(old, "atomic.fetch.sub.addr");
}

::llvm::Value *LLVMCodegen::_builtin_atomic_fetch_and(const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_value) noexcept {
    auto b = _current_context()->builder.get();
    auto value = b->CreateLoad(
        _create_type(t), p_value, "atomic.fetch.and.value");
    auto old = b->CreateAtomicRMW(
        ::llvm::AtomicRMWInst::And, p_atomic,
        value, {}, atomic_operation_order);
    old->setName("atomic.fetch.and.old");
    return _create_stack_variable(old, "atomic.fetch.and.addr");
}

::llvm::Value *LLVMCodegen::_builtin_atomic_fetch_or(const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_value) noexcept {
    auto b = _current_context()->builder.get();
    auto value = b->CreateLoad(
        _create_type(t), p_value, "atomic.fetch.or.value");
    auto old = b->CreateAtomicRMW(
        ::llvm::AtomicRMWInst::Or, p_atomic,
        value, {}, atomic_operation_order);
    old->setName("atomic.fetch.or.old");
    return _create_stack_variable(old, "atomic.fetch.or.addr");
}

::llvm::Value *LLVMCodegen::_builtin_atomic_fetch_xor(const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_value) noexcept {
    auto b = _current_context()->builder.get();
    auto value = b->CreateLoad(
        _create_type(t), p_value, "atomic.fetch.xor.value");
    auto old = b->CreateAtomicRMW(
        ::llvm::AtomicRMWInst::Xor, p_atomic,
        value, {}, atomic_operation_order);
    old->setName("atomic.fetch.xor.old");
    return _create_stack_variable(old, "atomic.fetch.xor.addr");
}

::llvm::Value *LLVMCodegen::_builtin_atomic_fetch_min(const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_value) noexcept {
    auto b = _current_context()->builder.get();
    auto value = static_cast<::llvm::Value *>(b->CreateLoad(
        _create_type(t), p_value, "atomic.fetch.min.value"));
    if (t->tag() == Type::Tag::UINT) {
        auto old = b->CreateAtomicRMW(
            ::llvm::AtomicRMWInst::UMin, p_atomic,
            value, {}, atomic_operation_order);
        old->setName("atomic.fetch.min.old");
        return _create_stack_variable(old, "atomic.fetch.min.addr");
    }
    if (t->tag() == Type::Tag::FLOAT) {
        auto elem_type = b->getInt32Ty();
        value = b->CreateBitCast(
            value, elem_type, "atomic.fetch.min.value.int");
        p_atomic = b->CreateBitOrPointerCast(
            p_atomic, ::llvm::PointerType::get(elem_type, 0),
            "atomic.fetch.min.addr.int");
        auto old = static_cast<::llvm::Value *>(
            b->CreateAtomicRMW(
                ::llvm::AtomicRMWInst::Min, p_atomic,
                value, {}, atomic_operation_order));
        old->setName("atomic.fetch.min.old.int");
        old = b->CreateBitCast(
            old, b->getFloatTy(), "atomic.fetch.min.old");
        return _create_stack_variable(old, "atomic.fetch.min.addr");
    }
    auto old = b->CreateAtomicRMW(
        ::llvm::AtomicRMWInst::Min, p_atomic,
        value, {}, atomic_operation_order);
    old->setName("atomic.fetch.min.old");
    return _create_stack_variable(old, "atomic.fetch.min.addr");
}

::llvm::Value *LLVMCodegen::_builtin_atomic_fetch_max(const Type *t, ::llvm::Value *p_atomic, ::llvm::Value *p_value) noexcept {
    auto b = _current_context()->builder.get();
    auto value = static_cast<::llvm::Value *>(b->CreateLoad(
        _create_type(t), p_value, "atomic.fetch.max.value"));
    if (t->tag() == Type::Tag::UINT) {
        auto old = b->CreateAtomicRMW(
            ::llvm::AtomicRMWInst::UMax, p_atomic,
            value, {}, atomic_operation_order);
        old->setName("atomic.fetch.max.old");
        return _create_stack_variable(old, "atomic.fetch.max.addr");
    }
    if (t->tag() == Type::Tag::FLOAT) {
        auto elem_type = b->getInt32Ty();
        value = b->CreateBitCast(
            value, elem_type, "atomic.fetch.max.value.int");
        p_atomic = b->CreateBitOrPointerCast(
            p_atomic, ::llvm::PointerType::get(elem_type, 0),
            "atomic.fetch.max.addr.int");
        auto old = static_cast<::llvm::Value *>(
            b->CreateAtomicRMW(
                ::llvm::AtomicRMWInst::Max, p_atomic,
                value, {}, atomic_operation_order));
        old->setName("atomic.fetch.max.old.int");
        old = b->CreateBitCast(
            old, b->getFloatTy(), "atomic.fetch.max.old");
        return _create_stack_variable(old, "atomic.fetch.max.addr");
    }
    auto old = b->CreateAtomicRMW(
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

::llvm::Value *LLVMCodegen::_builtin_dot(const Type *t, ::llvm::Value *va, ::llvm::Value *vb) noexcept {
    auto b = _current_context()->builder.get();
    auto type = _create_type(t);
    va = b->CreateLoad(type, va, "dot.a");
    vb = b->CreateLoad(type, vb, "dot.b");
    auto mul = b->CreateFMul(va, vb, "dot.mul");
    auto sum = b->CreateFAddReduce(_literal(0.f), mul);
    sum->setName("dot.sum");
    return _create_stack_variable(sum, "dot.addr");
}

::llvm::Value *LLVMCodegen::_builtin_add(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    auto lhs_v = b->CreateLoad(ir_type, lhs, "add.lhs");
    auto rhs_v = b->CreateLoad(ir_type, rhs, "add.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            b->CreateNSWAdd(lhs_v, rhs_v, "add"),
            "add.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            b->CreateAdd(lhs_v, rhs_v, "add"),
            "add.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        return _create_stack_variable(
            b->CreateFAdd(lhs_v, rhs_v, "add"),
            "add.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for add.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_sub(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    auto lhs_v = b->CreateLoad(ir_type, lhs, "sub.lhs");
    auto rhs_v = b->CreateLoad(ir_type, rhs, "sub.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            b->CreateNSWSub(lhs_v, rhs_v, "sub"),
            "sub.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            b->CreateSub(lhs_v, rhs_v, "sub"),
            "sub.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        return _create_stack_variable(
            b->CreateFSub(lhs_v, rhs_v, "sub"),
            "sub.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for sub.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_mul(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    auto lhs_v = b->CreateLoad(ir_type, lhs, "mul.lhs");
    auto rhs_v = b->CreateLoad(ir_type, rhs, "mul.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            b->CreateNSWMul(lhs_v, rhs_v, "mul"),
            "mul.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            b->CreateMul(lhs_v, rhs_v, "mul"),
            "mul.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        return _create_stack_variable(
            b->CreateFMul(lhs_v, rhs_v, "mul"),
            "mul.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for mul.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_div(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    auto lhs_v = b->CreateLoad(ir_type, lhs, "div.lhs");
    auto rhs_v = b->CreateLoad(ir_type, rhs, "div.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            b->CreateSDiv(lhs_v, rhs_v, "div"),
            "div.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            b->CreateUDiv(lhs_v, rhs_v, "div"),
            "div.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        return _create_stack_variable(
            b->CreateFDiv(lhs_v, rhs_v, "div"),
            "div.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for div.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_mod(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    auto lhs_v = b->CreateLoad(ir_type, lhs, "mod.lhs");
    auto rhs_v = b->CreateLoad(ir_type, rhs, "mod.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            b->CreateSRem(lhs_v, rhs_v, "mod"),
            "mod.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            b->CreateURem(lhs_v, rhs_v, "mod"),
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
    auto b = _current_context()->builder.get();
    auto result = b->CreateAnd(
        b->CreateLoad(_create_type(t), lhs, "and.lhs"),
        b->CreateLoad(_create_type(t), rhs, "and.rhs"), "and");
    return _create_stack_variable(result, "and.addr");
}

::llvm::Value *LLVMCodegen::_builtin_or(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    LUISA_ASSERT(is_scalar_or_vector(t, Type::Tag::INT) ||
                     is_scalar_or_vector(t, Type::Tag::UINT) ||
                     is_scalar_or_vector(t, Type::Tag::BOOL),
                 "Invalid type '{}' for or.", t->description());
    auto b = _current_context()->builder.get();
    auto result = b->CreateOr(
        b->CreateLoad(_create_type(t), lhs, "or.lhs"),
        b->CreateLoad(_create_type(t), rhs, "or.rhs"), "or");
    return _create_stack_variable(result, "or.addr");
}

::llvm::Value *LLVMCodegen::_builtin_xor(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    LUISA_ASSERT(is_scalar_or_vector(t, Type::Tag::INT) ||
                     is_scalar_or_vector(t, Type::Tag::UINT) ||
                     is_scalar_or_vector(t, Type::Tag::BOOL),
                 "Invalid type '{}' for xor.", t->description());
    auto b = _current_context()->builder.get();
    auto result = b->CreateXor(
        b->CreateLoad(_create_type(t), lhs, "xor.lhs"),
        b->CreateLoad(_create_type(t), rhs, "xor.rhs"), "xor");
    return _create_stack_variable(result, "xor.addr");
}

::llvm::Value *LLVMCodegen::_builtin_lt(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    auto lhs_v = b->CreateLoad(ir_type, lhs, "lt.lhs");
    auto rhs_v = b->CreateLoad(ir_type, rhs, "lt.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            b->CreateICmpSLT(lhs_v, rhs_v, "lt"),
            "lt.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            b->CreateICmpULT(lhs_v, rhs_v, "lt"),
            "lt.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        return _create_stack_variable(
            b->CreateFCmpOLT(lhs_v, rhs_v, "lt"),
            "lt.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::BOOL)) {
        return _create_stack_variable(
            b->CreateICmpULT(lhs_v, rhs_v, "lt"),
            "lt.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for lt.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_le(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    auto lhs_v = b->CreateLoad(ir_type, lhs, "le.lhs");
    auto rhs_v = b->CreateLoad(ir_type, rhs, "le.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            b->CreateICmpSLE(lhs_v, rhs_v, "le"),
            "le.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            b->CreateICmpULE(lhs_v, rhs_v, "le"),
            "le.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        return _create_stack_variable(
            b->CreateFCmpOLE(lhs_v, rhs_v, "le"),
            "le.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::BOOL)) {
        return _create_stack_variable(
            b->CreateICmpULE(lhs_v, rhs_v, "le"),
            "le.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for le.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_gt(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    auto lhs_v = b->CreateLoad(ir_type, lhs, "gt.lhs");
    auto rhs_v = b->CreateLoad(ir_type, rhs, "gt.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            b->CreateICmpSGT(lhs_v, rhs_v, "gt"),
            "gt.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            b->CreateICmpUGT(lhs_v, rhs_v, "gt"),
            "gt.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        return _create_stack_variable(
            b->CreateFCmpOGT(lhs_v, rhs_v, "gt"),
            "gt.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::BOOL)) {
        return _create_stack_variable(
            b->CreateICmpUGT(lhs_v, rhs_v, "gt"),
            "gt.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for gt.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_ge(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    auto lhs_v = b->CreateLoad(ir_type, lhs, "ge.lhs");
    auto rhs_v = b->CreateLoad(ir_type, rhs, "ge.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            b->CreateICmpSGE(lhs_v, rhs_v, "ge"),
            "ge.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            b->CreateICmpUGE(lhs_v, rhs_v, "ge"),
            "ge.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        return _create_stack_variable(
            b->CreateFCmpOGE(lhs_v, rhs_v, "ge"),
            "ge.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::BOOL)) {
        return _create_stack_variable(
            b->CreateICmpUGE(lhs_v, rhs_v, "ge"),
            "ge.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for ge.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_eq(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    auto lhs_v = b->CreateLoad(ir_type, lhs, "eq.lhs");
    auto rhs_v = b->CreateLoad(ir_type, rhs, "eq.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            b->CreateICmpEQ(lhs_v, rhs_v, "eq"),
            "eq.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            b->CreateICmpEQ(lhs_v, rhs_v, "eq"),
            "eq.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        return _create_stack_variable(
            b->CreateFCmpOEQ(lhs_v, rhs_v, "eq"),
            "eq.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::BOOL)) {
        return _create_stack_variable(
            b->CreateICmpEQ(lhs_v, rhs_v, "eq"),
            "eq.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for eq.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_ne(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    auto lhs_v = b->CreateLoad(ir_type, lhs, "neq.lhs");
    auto rhs_v = b->CreateLoad(ir_type, rhs, "neq.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            b->CreateICmpNE(lhs_v, rhs_v, "neq"),
            "neq.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            b->CreateICmpNE(lhs_v, rhs_v, "neq"),
            "neq.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::FLOAT)) {
        return _create_stack_variable(
            b->CreateFCmpONE(lhs_v, rhs_v, "neq"),
            "neq.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::BOOL)) {
        return _create_stack_variable(
            b->CreateICmpNE(lhs_v, rhs_v, "neq"),
            "neq.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for neq.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_shl(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    auto lhs_v = b->CreateLoad(ir_type, lhs, "shl.lhs");
    auto rhs_v = b->CreateLoad(ir_type, rhs, "shl.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT) ||
        is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            b->CreateShl(lhs_v, rhs_v, "shl"),
            "shl.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for shl.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_shr(const Type *t, ::llvm::Value *lhs, ::llvm::Value *rhs) noexcept {
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    auto lhs_v = b->CreateLoad(ir_type, lhs, "shr.lhs");
    auto rhs_v = b->CreateLoad(ir_type, rhs, "shr.rhs");
    if (is_scalar_or_vector(t, Type::Tag::INT)) {
        return _create_stack_variable(
            b->CreateAShr(lhs_v, rhs_v, "shr"),
            "shr.addr");
    }
    if (is_scalar_or_vector(t, Type::Tag::UINT)) {
        return _create_stack_variable(
            b->CreateLShr(lhs_v, rhs_v, "shr"),
            "shr.addr");
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid operand type '{}' for shr.",
        t->description());
}

void LLVMCodegen::_builtin_assume(::llvm::Value *p) noexcept {
    auto b = _current_context()->builder.get();
    auto pred = b->CreateICmpNE(
        b->CreateLoad(
            _create_type(Type::of<bool>()), p, "assume.load"),
        _literal(false), "assume.pred");
    b->CreateAssumption(pred);
}

void LLVMCodegen::_builtin_unreachable() noexcept {
    _current_context()->builder->CreateUnreachable();
}

::llvm::Value *LLVMCodegen::_builtin_isinf(const Type *t, ::llvm::Value *p) noexcept {
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    if (t->is_scalar()) {
        auto bits = b->CreateLoad(
            b->getInt32Ty(),
            _builtin_bitwise_cast(Type::of<uint>(), t, p),
            "isinf.bits");
        auto is_inf = b->CreateLogicalOr(
            b->CreateICmpEQ(bits, _literal(0x7f800000u), "isinf.pos"),
            b->CreateICmpEQ(bits, _literal(0xff800000u), "isinf.neg"),
            "isinf.pred");
        return _create_stack_variable(is_inf, "isinf.addr");
    }
    switch (t->dimension()) {
        case 2u: {
            auto bits = b->CreateLoad(
                _create_type(Type::of<uint2>()),
                _builtin_bitwise_cast(Type::of<uint2>(), t, p),
                "isinf.bits");
            auto is_inf = b->CreateLogicalOr(
                b->CreateICmpEQ(bits, _literal(make_uint2(0x7f800000u)), "isinf.pos"),
                b->CreateICmpEQ(bits, _literal(make_uint2(0xff800000u)), "isinf.neg"),
                "isinf.pred");
            return _create_stack_variable(is_inf, "isinf.addr");
        }
        case 3u: {
            auto bits = b->CreateLoad(
                _create_type(Type::of<uint3>()),
                _builtin_bitwise_cast(Type::of<uint3>(), t, p),
                "isinf.bits");
            auto is_inf = b->CreateLogicalOr(
                b->CreateICmpEQ(bits, _literal(make_uint3(0x7f800000u)), "isinf.pos"),
                b->CreateICmpEQ(bits, _literal(make_uint3(0xff800000u)), "isinf.neg"),
                "isinf.pred");
            return _create_stack_variable(is_inf, "isinf.addr");
        }
        case 4u: {
            auto bits = b->CreateLoad(
                _create_type(Type::of<uint4>()),
                _builtin_bitwise_cast(Type::of<uint4>(), t, p),
                "isinf.bits");
            auto is_inf = b->CreateLogicalOr(
                b->CreateICmpEQ(bits, _literal(make_uint4(0x7f800000u)), "isinf.pos"),
                b->CreateICmpEQ(bits, _literal(make_uint4(0xff800000u)), "isinf.neg"),
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
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    if (t->is_scalar()) {
        auto bits = b->CreateLoad(
            b->getInt32Ty(),
            _builtin_bitwise_cast(Type::of<uint>(), t, p),
            "isnan.bits");
        auto is_nan = b->CreateLogicalAnd(
            b->CreateICmpEQ(
                b->CreateAnd(bits, _literal(0x7f800000u), "isnan.exp"),
                _literal(0x7f800000u), "isnan.exp.cmp"),
            b->CreateICmpNE(
                b->CreateAnd(bits, _literal(0x7fffffu), "isnan.mant"),
                _literal(0u), "isnan.mant.cmp"),
            "isnan.pred");
        return _create_stack_variable(is_nan, "isnan.addr");
    }
    switch (t->dimension()) {
        case 2u: {
            auto bits = b->CreateLoad(
                _create_type(Type::of<uint2>()),
                _builtin_bitwise_cast(Type::of<uint2>(), t, p),
                "isnan.bits");
            auto is_nan = b->CreateLogicalAnd(
                b->CreateICmpEQ(
                    b->CreateAnd(bits, _literal(make_uint2(0x7f800000u)), "isnan.exp"),
                    _literal(make_uint2(0x7f800000u)), "isnan.exp.cmp"),
                b->CreateICmpNE(
                    b->CreateAnd(bits, _literal(make_uint2(0x7fffffu)), "isnan.mant"),
                    _literal(make_uint2(0u)), "isnan.mant.cmp"),
                "isnan.pred");
            return _create_stack_variable(is_nan, "isnan.addr");
        }
        case 3u: {
            auto bits = b->CreateLoad(
                _create_type(Type::of<uint3>()),
                _builtin_bitwise_cast(Type::of<uint3>(), t, p),
                "isnan.bits");
            auto is_nan = b->CreateLogicalAnd(
                b->CreateICmpEQ(
                    b->CreateAnd(bits, _literal(make_uint3(0x7f800000u)), "isnan.exp"),
                    _literal(make_uint3(0x7f800000u)), "isnan.exp.cmp"),
                b->CreateICmpNE(
                    b->CreateAnd(bits, _literal(make_uint3(0x7fffffu)), "isnan.mant"),
                    _literal(make_uint3(0u)), "isnan.mant.cmp"),
                "isnan.pred");
            return _create_stack_variable(is_nan, "isnan.addr");
        }
        case 4u: {
            auto bits = b->CreateLoad(
                _create_type(Type::of<uint4>()),
                _builtin_bitwise_cast(Type::of<uint4>(), t, p),
                "isnan.bits");
            auto is_nan = b->CreateLogicalAnd(
                b->CreateICmpEQ(
                    b->CreateAnd(bits, _literal(make_uint4(0x7f800000u)), "isnan.exp"),
                    _literal(make_uint4(0x7f800000u)), "isnan.exp.cmp"),
                b->CreateICmpNE(
                    b->CreateAnd(bits, _literal(make_uint4(0x7fffffu)), "isnan.mant"),
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
    auto b = _current_context()->builder.get();
    auto src_ir_type = _create_type(t_src);
    auto dst_ir_type = _create_type(t_dst);
    auto p_dst = b->CreateBitOrPointerCast(p, dst_ir_type->getPointerTo(), "bitcast.ptr");
    auto dst = b->CreateLoad(dst_ir_type, p_dst, "bitcast.dst");
    return _create_stack_variable(dst, "bitcast.addr");
}

::llvm::Value *LLVMCodegen::_builtin_unary_plus(const Type *t, ::llvm::Value *p) noexcept {
    return p;
}

::llvm::Value *LLVMCodegen::_builtin_unary_minus(const Type *t, ::llvm::Value *p) noexcept {
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    if (t->is_matrix()) {
        std::array<::llvm::Value *, 4u> m{};
        for (auto i = 0u; i < t->dimension(); ++i) {
            auto name = fmt::format("unary.minus.m{}.addr", i);
            m[i] = b->CreateStructGEP(
                ir_type, p, i,
                ::llvm::StringRef{name.data(), name.size()});
        }
        if (t->dimension() == 2u) {
            return _make_float2x2(
                b->CreateFNeg(m[0]),
                b->CreateFNeg(m[1]));
        }
        if (t->dimension() == 3u) {
            return _make_float3x3(
                b->CreateFNeg(m[0]),
                b->CreateFNeg(m[1]),
                b->CreateFNeg(m[2]));
        }
        if (t->dimension() == 4u) {
            return _make_float4x4(
                b->CreateFNeg(m[0]),
                b->CreateFNeg(m[1]),
                b->CreateFNeg(m[2]),
                b->CreateFNeg(m[3]));
        }
        LUISA_ERROR_WITH_LOCATION(
            "Invalid matrix dimension '{}' for unary minus.",
            t->dimension());
    }
    auto x = b->CreateLoad(ir_type, p, "unary.minus.load");
    switch (auto tag = t->is_scalar() ? t->tag() : t->element()->tag()) {
        case Type::Tag::BOOL: return _create_stack_variable(
            b->CreateNot(x, "unary.minus"),
            "unary.minus.addr");
        case Type::Tag::FLOAT: return _create_stack_variable(
            b->CreateFNeg(x, "unary.minus"),
            "unary.minus.addr");
        case Type::Tag::INT:
        case Type::Tag::UINT: return _create_stack_variable(
            b->CreateNeg(x, "unary.minus"),
            "unary.minus.addr");
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid argument type '{}' for unary minus.",
        t->description());
}

::llvm::Value *LLVMCodegen::_builtin_unary_not(const Type *t, ::llvm::Value *p) noexcept {
    auto b = _current_context()->builder.get();
    p = t->is_scalar() ? _scalar_to_bool(t, p) : _vector_to_bool_vector(t, p);
    auto i8_type = static_cast<::llvm::Type *>(::llvm::Type::getInt8Ty(_context));
    auto i8_vec_type = t->is_scalar() ? i8_type : ::llvm::FixedVectorType::get(i8_type, t->dimension());
    auto v = b->CreateLoad(i8_vec_type, p);
    auto type = static_cast<::llvm::Type *>(b->getInt1Ty());
    if (t->is_vector()) { type = ::llvm::FixedVectorType::get(type, t->dimension()); }
    auto nv = b->CreateNot(b->CreateTrunc(v, type), "unary.not");
    return _create_stack_variable(nv, "unary.not.addr");
}

::llvm::Value *LLVMCodegen::_builtin_unary_bit_not(const Type *t, ::llvm::Value *p) noexcept {
    LUISA_ASSERT(t->tag() == Type::Tag::INT || t->tag() == Type::Tag::UINT ||
                     (t->is_vector() && t->element()->tag() == Type::Tag::INT) ||
                     (t->is_vector() && t->element()->tag() == Type::Tag::UINT),
                 "Invalid argument type '{}' for bitwise not.",
                 t->description());
    auto b = _current_context()->builder.get();
    auto ir_type = _create_type(t);
    auto x = b->CreateLoad(ir_type, p, "unary.bitnot.load");
    return _create_stack_variable(b->CreateNot(x, "unary.bitnot"), "unary.bitnot.addr");
}

::llvm::Value *LLVMCodegen::_builtin_add_matrix_scalar(const Type *t_lhs, const Type *t_rhs, ::llvm::Value *p_lhs, ::llvm::Value *p_rhs) noexcept {
    LUISA_ASSERT(t_lhs->is_matrix() && t_rhs->is_scalar(),
                 "Invalid argument types '{}' and '{}' for matrix-scalar addition.",
                 t_lhs->description(), t_rhs->description());
    auto b = _current_context()->builder.get();
    auto lhs_type = _create_type(t_lhs);
    auto col_type = Type::from(luisa::format("vector<float,{}>", t_lhs->dimension()));
    auto rhs = _scalar_to_vector(col_type, t_rhs, p_rhs);
    std::array<::llvm::Value *, 4u> m{};
    for (auto i = 0u; i < t_lhs->dimension(); ++i) {
        auto name = fmt::format("add.lhs.m{}.addr", i);
        auto col = b->CreateStructGEP(
            lhs_type, p_lhs, i,
            ::llvm::StringRef{name.data(), name.size()});
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
    auto b = _current_context()->builder.get();
    auto matrix_type = _create_type(t_lhs);
    auto col_type = Type::from(luisa::format("vector<float,{}>", t_lhs->dimension()));
    std::array<::llvm::Value *, 4u> m{};
    for (auto i = 0u; i < t_lhs->dimension(); ++i) {
        auto lhs_name = fmt::format("add.lhs.m{}.addr", i);
        auto rhs_name = fmt::format("add.rhs.m{}.addr", i);
        auto lhs = b->CreateStructGEP(
            matrix_type, p_lhs, i,
            ::llvm::StringRef{lhs_name.data(), lhs_name.size()});
        auto rhs = b->CreateStructGEP(
            matrix_type, p_rhs, i,
            ::llvm::StringRef{rhs_name.data(), rhs_name.size()});
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
    auto b = _current_context()->builder.get();
    auto lhs_type = _create_type(t_lhs);
    auto col_type = Type::from(luisa::format("vector<float,{}>", t_lhs->dimension()));
    auto rhs = _scalar_to_vector(col_type, t_rhs, p_rhs);
    std::array<::llvm::Value *, 4u> m{};
    for (auto i = 0u; i < t_lhs->dimension(); ++i) {
        auto name = fmt::format("sub.lhs.m{}.addr", i);
        auto col = b->CreateStructGEP(
            lhs_type, p_lhs, i,
            ::llvm::StringRef{name.data(), name.size()});
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
    auto b = _current_context()->builder.get();
    auto matrix_type = _create_type(t_rhs);
    auto col_type = Type::from(luisa::format("vector<float,{}>", t_lhs->dimension()));
    auto lhs = _scalar_to_vector(col_type, t_lhs, p_lhs);
    std::array<::llvm::Value *, 4u> m{};
    for (auto i = 0u; i < t_lhs->dimension(); ++i) {
        auto name = fmt::format("add.rhs.m{}.addr", i);
        auto rhs_col = b->CreateStructGEP(
            matrix_type, p_rhs, i,
            ::llvm::StringRef{name.data(), name.size()});
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
    auto b = _current_context()->builder.get();
    auto matrix_type = _create_type(t_lhs);
    auto col_type = Type::from(luisa::format("vector<float,{}>", t_lhs->dimension()));
    std::array<::llvm::Value *, 4u> m{};
    for (auto i = 0u; i < t_lhs->dimension(); ++i) {
        auto lhs_name = fmt::format("sub.lhs.m{}.addr", i);
        auto rhs_name = fmt::format("sub.rhs.m{}.addr", i);
        auto lhs = b->CreateStructGEP(
            matrix_type, p_lhs, i,
            ::llvm::StringRef{lhs_name.data(), lhs_name.size()});
        auto rhs = b->CreateStructGEP(
            matrix_type, p_rhs, i,
            ::llvm::StringRef{rhs_name.data(), rhs_name.size()});
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
    auto b = _current_context()->builder.get();
    auto lhs_type = _create_type(t_lhs);
    auto col_type = Type::from(luisa::format("vector<float,{}>", t_lhs->dimension()));
    auto rhs = _scalar_to_vector(col_type, t_rhs, p_rhs);
    std::array<::llvm::Value *, 4u> m{};
    for (auto i = 0u; i < t_lhs->dimension(); ++i) {
        auto name = fmt::format("mul.lhs.m{}.addr", i);
        auto col = b->CreateStructGEP(
            lhs_type, p_lhs, i,
            ::llvm::StringRef{name.data(), name.size()});
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
    auto b = _current_context()->builder.get();
    for (auto i = 0u; i < t_lhs->dimension(); ++i) {
        auto rhs_name = fmt::format("mul.rhs.m{}.addr", i);
        auto rhs_col = b->CreateStructGEP(
            matrix_type, p_rhs, i,
            ::llvm::StringRef{rhs_name.data(), rhs_name.size()});
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
    auto b = _current_context()->builder.get();
    auto matrix_type = _create_type(t_lhs);
    auto col_type = Type::from(luisa::format("vector<float,{}>", t_lhs->dimension()));
    auto rhs = b->CreateLoad(_create_type(t_rhs), p_rhs, "mul.rhs");
    for (auto i = 0u; i < t_lhs->dimension(); ++i) {
        auto col_name = fmt::format("mul.lhs.m{}.addr", i);
        auto col = b->CreateStructGEP(
            matrix_type, p_lhs, i,
            ::llvm::StringRef{col_name.data(), col_name.size()});
        auto v_name = fmt::format("mul.rhs.v{}", i);
        ::llvm::SmallVector<int, 4u> masks(t_rhs->dimension(), static_cast<int>(i));
        auto v = b->CreateShuffleVector(rhs, masks, ::llvm::StringRef{v_name.data(), v_name.size()});
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
    auto b = _current_context()->builder.get();
    auto lhs_type = _create_type(t_lhs);
    auto col_type = Type::from(luisa::format("vector<float,{}>", t_lhs->dimension()));
    auto rhs = _scalar_to_vector(col_type, t_rhs, p_rhs);
    std::array<::llvm::Value *, 4u> m{};
    for (auto i = 0u; i < t_lhs->dimension(); ++i) {
        auto name = fmt::format("div.lhs.m{}.addr", i);
        auto col = b->CreateStructGEP(
            lhs_type, p_lhs, i,
            ::llvm::StringRef{name.data(), name.size()});
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
    auto b = _current_context()->builder.get();
    auto matrix_type = _create_type(t_rhs);
    auto col_type = Type::from(luisa::format("vector<float,{}>", t_lhs->dimension()));
    auto lhs = _scalar_to_vector(col_type, t_lhs, p_lhs);
    std::array<::llvm::Value *, 4u> m{};
    for (auto i = 0u; i < t_lhs->dimension(); ++i) {
        auto name = fmt::format("add.rhs.m{}.addr", i);
        auto rhs_col = b->CreateStructGEP(
            matrix_type, p_rhs, i,
            ::llvm::StringRef{name.data(), name.size()});
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
    auto b = _current_context()->builder.get();
    auto value_type = _create_type(t_value);
    auto index = b->CreateLoad(_create_type(Type::of<uint>()), p_index, "buffer.read.index");
    auto ptr = b->CreateInBoundsGEP(value_type, buffer, index, "buffer.read.ptr");
    auto value = b->CreateLoad(value_type, ptr, "buffer.read");
    return _create_stack_variable(value, "buffer.read.addr");
}

void LLVMCodegen::_builtin_buffer_write(const Type *t_value, ::llvm::Value *buffer, ::llvm::Value *p_index, ::llvm::Value *p_value) noexcept {
    auto b = _current_context()->builder.get();
    auto value_type = _create_type(t_value);
    auto index = b->CreateLoad(_create_type(Type::of<uint>()), p_index, "buffer.write.index");
    auto ptr = b->CreateInBoundsGEP(value_type, buffer, index, "buffer.write.ptr");
    _create_assignment(t_value, t_value, ptr, p_value);
}

::llvm::Value *LLVMCodegen::_builtin_texture_read(const Type *t, ::llvm::Value *texture,
                                                  const Type *t_coord, ::llvm::Value *p_coord) noexcept {
    LUISA_ASSERT(t->is_vector() && t->dimension() == 4u,
                 "Invalid type '{}' for texture-read.",
                 t->description());
    // <4 x float> texture.read.Nd.type(i64 t0, i64 t1, i64 c0, i64 c2)
    auto b = _current_context()->builder.get();
    auto coord = b->CreateLoad(_create_type(t_coord), p_coord, "texture.read.coord");
    auto coord_type = static_cast<::llvm::FixedVectorType *>(coord->getType());
    auto dim = coord_type->getNumElements() == 2u ? 2u : 3u;
    auto func_name = luisa::format("texture.read.{}d.{}", dim, t->element()->description());
    auto func = _module->getFunction(::llvm::StringRef{func_name.data(), func_name.size()});
    auto i64_type = ::llvm::Type::getInt64Ty(_context);
    auto f32v4_type = ::llvm::FixedVectorType::get(b->getFloatTy(), 4u);
    if (func == nullptr) {
        func = ::llvm::Function::Create(
            ::llvm::FunctionType::get(
                f32v4_type,
                {i64_type, i64_type, i64_type, i64_type},
                false),
            ::llvm::Function::ExternalLinkage,
            ::llvm::StringRef{func_name.data(), func_name.size()},
            _module);
        func->setNoSync();
        func->setWillReturn();
        func->setDoesNotThrow();
        func->setMustProgress();
        func->setDoesNotRecurse();
        func->setOnlyReadsMemory();
        func->setDoesNotFreeMemory();
        func->setOnlyAccessesInaccessibleMemOrArgMem();
    }
    auto t0 = b->CreateExtractValue(texture, 0u, "texture.read.texture.t0");
    auto t1 = b->CreateExtractValue(texture, 1u, "texture.read.texture.t1");
    std::array<int, 4> shuffle{0, 1, 0, 0};
    if (dim == 3u) { shuffle[2] = 2; }
    auto coord_vector = _create_stack_variable(
        b->CreateShuffleVector(
            coord, shuffle, "texture.read.coord.vector"),
        "texture.read.coord.vector.addr");
    auto i64v2_type = ::llvm::FixedVectorType::get(i64_type, 2u);
    p_coord = b->CreateBitOrPointerCast(
        coord_vector, i64v2_type->getPointerTo(0),
        "texture.read.coord.ulong2.addr");
    coord = b->CreateLoad(i64v2_type, p_coord, "texture.read.coord.ulong2");
    auto c0 = b->CreateExtractElement(coord, static_cast<uint64_t>(0u), "texture.read.coord.c0");
    auto c1 = b->CreateExtractElement(coord, static_cast<uint64_t>(1u), "texture.read.coord.c1");
    auto ret = static_cast<::llvm::Value *>(b->CreateCall(func, {t0, t1, c0, c1}, "texture.read.ret"));
    if (t->element()->tag() != Type::Tag::FLOAT) { ret = b->CreateBitCast(ret, _create_type(t), "texture.read.ret.cast"); }
    return _create_stack_variable(ret, "texture.read.addr");
}

void LLVMCodegen::_builtin_texture_write(const Type *t, ::llvm::Value *texture, const Type *t_coord,
                                         ::llvm::Value *p_coord, ::llvm::Value *p_value) noexcept {
    LUISA_ASSERT(t->is_vector() && t->dimension() == 4u,
                 "Invalid type '{}' for texture-write.",
                 t->description());
    // texture.write.Nd.type(i64 t0, i64 t1, i64 c0, i64 c1, i64 v0, i64 v1)
    auto b = _current_context()->builder.get();
    auto coord = b->CreateLoad(_create_type(t_coord), p_coord, "texture.write.coord");
    auto coord_type = static_cast<::llvm::FixedVectorType *>(coord->getType());
    auto dim = coord_type->getNumElements() == 2u ? 2u : 3u;
    auto func_name = luisa::format("texture.write.{}d.{}", dim, t->element()->description());
    auto func = _module->getFunction(::llvm::StringRef{func_name.data(), func_name.size()});
    auto i64_type = ::llvm::Type::getInt64Ty(_context);
    if (func == nullptr) {
        func = ::llvm::Function::Create(
            ::llvm::FunctionType::get(
                ::llvm::Type::getVoidTy(_context),
                {i64_type, i64_type, i64_type, i64_type, i64_type, i64_type},
                false),
            ::llvm::Function::ExternalLinkage,
            ::llvm::StringRef{func_name.data(), func_name.size()},
            _module);
        func->setNoSync();
        func->setWillReturn();
        func->setDoesNotThrow();
        func->setMustProgress();
        func->setDoesNotRecurse();
        func->setDoesNotFreeMemory();
        func->setOnlyAccessesInaccessibleMemOrArgMem();
    }
    auto t0 = b->CreateExtractValue(texture, 0u, "texture.write.texture.t0");
    auto t1 = b->CreateExtractValue(texture, 1u, "texture.write.texture.t1");
    std::array<int, 4> shuffle{0, 1, 0, 0};
    if (dim == 3u) { shuffle[2] = 2; }
    auto i64v2_type = ::llvm::FixedVectorType::get(i64_type, 2u);
    auto coord_vector = _create_stack_variable(
        b->CreateShuffleVector(
            coord, shuffle, "texture.write.coord.vector"),
        "texture.write.coord.vector.addr");
    p_coord = b->CreateBitOrPointerCast(
        coord_vector, i64v2_type->getPointerTo(0),
        "texture.write.coord.ulong2.addr");
    coord = b->CreateLoad(i64v2_type, p_coord, "texture.write.coord.ulong2");
    auto c0 = b->CreateExtractElement(coord, static_cast<uint64_t>(0u), "texture.write.coord.c0");
    auto c1 = b->CreateExtractElement(coord, static_cast<uint64_t>(1u), "texture.write.coord.c1");
    p_value = b->CreateBitOrPointerCast(
        p_value, i64v2_type->getPointerTo(0), "texture.write.value.ulong2.addr");
    auto value = b->CreateLoad(i64v2_type, p_value, "texture.write.value.ulong");
    auto v0 = b->CreateExtractElement(value, static_cast<uint64_t>(0u), "texture.write.value.v0");
    auto v1 = b->CreateExtractElement(value, static_cast<uint64_t>(1u), "texture.write.value.v1");
    b->CreateCall(func->getFunctionType(), func, {t0, t1, c0, c1, v0, v1});
}

::llvm::Value *LLVMCodegen::_builtin_trace_closest(::llvm::Value *accel, ::llvm::Value *p_ray) noexcept {
    // <4 x float> trace_closest(i64 accel, i64 r0, i64 r1, i64 r2, i64 r3)
    auto b = _current_context()->builder.get();
    auto i64_type = ::llvm::Type::getInt64Ty(_context);
    auto func_name = "accel.trace.closest";
    auto func = _module->getFunction(func_name);
    accel = b->CreateExtractValue(accel, 0u, "trace.closest.accel.handle");
    if (func == nullptr) {
        func = ::llvm::Function::Create(
            ::llvm::FunctionType::get(
                ::llvm::FixedVectorType::get(b->getFloatTy(), 4u),
                {accel->getType(), i64_type, i64_type, i64_type, i64_type},
                false),
            ::llvm::Function::ExternalLinkage, func_name, _module);
        func->setNoSync();
        func->setWillReturn();
        func->setDoesNotThrow();
        func->setMustProgress();
        func->setSpeculatable();
        func->setOnlyReadsMemory();
        func->setDoesNotFreeMemory();
        func->setOnlyAccessesInaccessibleMemOrArgMem();
    }
    auto ray_struct_type = ::llvm::FixedVectorType::get(i64_type, 4u);
    p_ray = b->CreateBitOrPointerCast(p_ray, ray_struct_type->getPointerTo(0),
                                      "trace.closest.ray.struct.addr");
    auto ray = b->CreateLoad(ray_struct_type, p_ray, "trace.closest.ray.struct");
    auto r0 = b->CreateExtractElement(ray, static_cast<uint64_t>(0u), "trace.closest.ray.r0");
    auto r1 = b->CreateExtractElement(ray, static_cast<uint64_t>(1u), "trace.closest.ray.r1");
    auto r2 = b->CreateExtractElement(ray, static_cast<uint64_t>(2u), "trace.closest.ray.r2");
    auto r3 = b->CreateExtractElement(ray, static_cast<uint64_t>(3u), "trace.closest.ray.r3");
    auto ret = b->CreateCall(
        func->getFunctionType(), func, {accel, r0, r1, r2, r3},
        "accel.trace.closest.struct");
    auto inst = b->CreateBitCast(b->CreateExtractElement(ret, static_cast<uint64_t>(0u)), b->getInt32Ty(), "trace.closest.hit.inst");
    auto prim = b->CreateBitCast(b->CreateExtractElement(ret, static_cast<uint64_t>(1u)), b->getInt32Ty(), "trace.closest.hit.prim");
    auto bary = b->CreateShuffleVector(ret, {2, 3}, "trace.closest.hit.bary");
    auto hit = static_cast<::llvm::Value *>(::llvm::UndefValue::get(_create_type(Type::of<Hit>())));
    hit = b->CreateInsertValue(hit, inst, 0u);
    hit = b->CreateInsertValue(hit, prim, 1u);
    hit = b->CreateInsertValue(hit, bary, 2u, "trace_closest.hit");
    return _create_stack_variable(hit, "trace_closest.hit.addr");
}

::llvm::Value *LLVMCodegen::_builtin_trace_any(::llvm::Value *accel, ::llvm::Value *p_ray) noexcept {
    // i8 trace_closest(i64 accel, i64 r0, i64 r1, i64 r2, i64 r3)
    auto b = _current_context()->builder.get();
    auto i64_type = ::llvm::Type::getInt64Ty(_context);
    auto func_name = "accel.trace.any";
    auto func = _module->getFunction(func_name);
    accel = b->CreateExtractValue(accel, 0u, "trace_closest.accel.handle");
    if (func == nullptr) {
        func = ::llvm::Function::Create(
            ::llvm::FunctionType::get(
                ::llvm::Type::getInt8Ty(_context),
                {accel->getType(), i64_type, i64_type, i64_type, i64_type},
                false),
            ::llvm::Function::ExternalLinkage, func_name, _module);
        func->setNoSync();
        func->setWillReturn();
        func->setDoesNotThrow();
        func->setMustProgress();
        func->setSpeculatable();
        func->setOnlyReadsMemory();
        func->setDoesNotFreeMemory();
        func->setOnlyAccessesInaccessibleMemOrArgMem();
    }
    auto ray_struct_type = ::llvm::FixedVectorType::get(i64_type, 4u);
    p_ray = b->CreateBitOrPointerCast(
        p_ray, ray_struct_type->getPointerTo(0),
        "trace_any.ray.struct.addr");
    auto ray = b->CreateLoad(ray_struct_type, p_ray, "trace_any.ray.struct");
    auto r0 = b->CreateExtractElement(ray, static_cast<uint64_t>(0u), "trace_any.ray.r0");
    auto r1 = b->CreateExtractElement(ray, static_cast<uint64_t>(1u), "trace_any.ray.r1");
    auto r2 = b->CreateExtractElement(ray, static_cast<uint64_t>(2u), "trace_any.ray.r2");
    auto r3 = b->CreateExtractElement(ray, static_cast<uint64_t>(3u), "trace_any.ray.r3");
    auto ret = b->CreateCall(
        func->getFunctionType(), func, {accel, r0, r1, r2, r3},
        "accel.trace.any.ret");
    auto hit = b->CreateTrunc(ret, b->getInt1Ty());
    hit->setName("accel.trace.any.hit");
    return _create_stack_variable(hit, "accel.trace.any.hit.addr");
}

::llvm::Value *LLVMCodegen::_builtin_length_squared(const Type *t, ::llvm::Value *v) noexcept {
    return _builtin_dot(t, v, v);
}

::llvm::Value *LLVMCodegen::_builtin_cross(const Type *t, ::llvm::Value *va, ::llvm::Value *vb) noexcept {
    LUISA_ASSERT(t->is_vector() && t->dimension() == 3u,
                 "Invalid argument types '{}' and '{}' for cross product.",
                 t->description(), t->description());
    auto b = _current_context()->builder.get();
    auto type = _create_type(t);
    va = b->CreateLoad(type, va, "cross.a");
    vb = b->CreateLoad(type, vb, "cross.b");
    auto a_x = b->CreateExtractElement(va, static_cast<uint64_t>(0u), "cross.a.x");
    auto a_y = b->CreateExtractElement(va, static_cast<uint64_t>(1u), "cross.a.y");
    auto a_z = b->CreateExtractElement(va, static_cast<uint64_t>(2u), "cross.a.z");
    auto b_x = b->CreateExtractElement(vb, static_cast<uint64_t>(0u), "cross.b.x");
    auto b_y = b->CreateExtractElement(vb, static_cast<uint64_t>(1u), "cross.b.y");
    auto b_z = b->CreateExtractElement(vb, static_cast<uint64_t>(2u), "cross.b.z");
    auto x = b->CreateFSub(b->CreateFMul(a_y, b_z), b->CreateFMul(a_z, b_y), "cross.x");
    auto y = b->CreateFSub(b->CreateFMul(a_z, b_x), b->CreateFMul(a_x, b_z), "cross.y");
    auto z = b->CreateFSub(b->CreateFMul(a_x, b_y), b->CreateFMul(a_y, b_x), "cross.z");
    return _make_float3(_create_stack_variable(x, "cross.x.addr"),
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
        return _make_float2(_builtin_static_cast(t_vec->element(), args[0]->type(), _create_expr(args[0])),
                            _builtin_static_cast(t_vec->element(), args[1]->type(), _create_expr(args[1])));
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
            auto x = _builtin_static_cast(t_vec->element(), args[0]->type(), _create_expr(args[0]));
            return _make_float3(x, y, z);
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
        auto z = _builtin_static_cast(t_vec->element(), args[1]->type(), _create_expr(args[1]));
        return _make_float3(x, y, z);
    }
    if (args.size() == 3u) {
        LUISA_ASSERT(args[0]->type()->is_scalar() && args[1]->type()->is_scalar() &&
                         args[2]->type()->is_scalar(),
                     "Invalid argument types ('{}', '{}', '{}') for make-vector3.",
                     args[0]->type()->description(), args[1]->type()->description(),
                     args[2]->type()->description());
        return _make_float3(_builtin_static_cast(t_vec->element(), args[0]->type(), _create_expr(args[0])),
                            _builtin_static_cast(t_vec->element(), args[1]->type(), _create_expr(args[1])),
                            _builtin_static_cast(t_vec->element(), args[2]->type(), _create_expr(args[2])));
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
            auto x = _builtin_static_cast(t_vec->element(), args[0]->type(), _create_expr(args[0]));
            return _make_float4(x, y, z, w);
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
            auto w = _builtin_static_cast(t_vec->element(), args[1]->type(), _create_expr(args[1]));
            return _make_float4(x, y, z, w);
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
                v.emplace_back(_builtin_static_cast(
                    t_vec->element(), arg->type(), _create_expr(arg)));
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
    return _make_float4(_builtin_static_cast(t_vec->element(), args[0]->type(), _create_expr(args[0])),
                        _builtin_static_cast(t_vec->element(), args[1]->type(), _create_expr(args[1])),
                        _builtin_static_cast(t_vec->element(), args[2]->type(), _create_expr(args[2])),
                        _builtin_static_cast(t_vec->element(), args[3]->type(), _create_expr(args[3])));
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
    auto c0 = _make_float2(_builtin_static_cast(Type::of<float>(), args[0]->type(), _create_expr(args[0])),
                           _builtin_static_cast(Type::of<float>(), args[1]->type(), _create_expr(args[1])));
    auto c1 = _make_float2(_builtin_static_cast(Type::of<float>(), args[2]->type(), _create_expr(args[2])),
                           _builtin_static_cast(Type::of<float>(), args[3]->type(), _create_expr(args[3])));
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
    auto c0 = _make_float3(_builtin_static_cast(Type::of<float>(), args[0]->type(), _create_expr(args[0])),
                           _builtin_static_cast(Type::of<float>(), args[1]->type(), _create_expr(args[1])),
                           _builtin_static_cast(Type::of<float>(), args[2]->type(), _create_expr(args[2])));
    auto c1 = _make_float3(_builtin_static_cast(Type::of<float>(), args[3]->type(), _create_expr(args[3])),
                           _builtin_static_cast(Type::of<float>(), args[4]->type(), _create_expr(args[4])),
                           _builtin_static_cast(Type::of<float>(), args[5]->type(), _create_expr(args[5])));
    auto c2 = _make_float3(_builtin_static_cast(Type::of<float>(), args[6]->type(), _create_expr(args[6])),
                           _builtin_static_cast(Type::of<float>(), args[7]->type(), _create_expr(args[7])),
                           _builtin_static_cast(Type::of<float>(), args[8]->type(), _create_expr(args[8])));
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
    auto c0 = _make_float4(_builtin_static_cast(Type::of<float>(), args[0]->type(), _create_expr(args[0])),
                           _builtin_static_cast(Type::of<float>(), args[1]->type(), _create_expr(args[1])),
                           _builtin_static_cast(Type::of<float>(), args[2]->type(), _create_expr(args[2])),
                           _builtin_static_cast(Type::of<float>(), args[3]->type(), _create_expr(args[3])));
    auto c1 = _make_float4(_builtin_static_cast(Type::of<float>(), args[4]->type(), _create_expr(args[4])),
                           _builtin_static_cast(Type::of<float>(), args[5]->type(), _create_expr(args[5])),
                           _builtin_static_cast(Type::of<float>(), args[6]->type(), _create_expr(args[6])),
                           _builtin_static_cast(Type::of<float>(), args[7]->type(), _create_expr(args[7])));
    auto c2 = _make_float4(_builtin_static_cast(Type::of<float>(), args[8]->type(), _create_expr(args[8])),
                           _builtin_static_cast(Type::of<float>(), args[9]->type(), _create_expr(args[9])),
                           _builtin_static_cast(Type::of<float>(), args[10]->type(), _create_expr(args[10])),
                           _builtin_static_cast(Type::of<float>(), args[11]->type(), _create_expr(args[11])));
    auto c3 = _make_float4(_builtin_static_cast(Type::of<float>(), args[12]->type(), _create_expr(args[12])),
                           _builtin_static_cast(Type::of<float>(), args[13]->type(), _create_expr(args[13])),
                           _builtin_static_cast(Type::of<float>(), args[14]->type(), _create_expr(args[14])),
                           _builtin_static_cast(Type::of<float>(), args[15]->type(), _create_expr(args[15])));
    return _make_float4x4(c0, c1, c2, c3);
}

::llvm::Value *LLVMCodegen::_builtin_bindless_texture_size2d(::llvm::Value *p_items, ::llvm::Value *p_index, ::llvm::Value *p_level) noexcept {
    auto b = _current_context()->builder.get();
    auto level = p_level == nullptr ? _literal(0u) : b->CreateLoad(b->getInt32Ty(), p_level, "bindless.texture.size.2d.level");
    auto index = b->CreateLoad(b->getInt32Ty(), p_index, "bindless.texture.size.2d.index");
    auto pp_texture = b->CreateInBoundsGEP(_bindless_item_type(), p_items, {index, _literal(1)}, "bindless.texture.size.2d.texture.ptr.ptr");
    auto p_texture = b->CreateLoad(_bindless_texture_type()->getPointerTo(), pp_texture, "bindless.texture.size.2d.texture.ptr");
    auto texture = b->CreateLoad(_bindless_texture_type(), p_texture, "bindless.texture.size.2d.texture");
    auto texture_size = b->CreateExtractValue(texture, 1u, "bindless.texture.size.2d.texture.size.i16");
    texture_size = b->CreateShuffleVector(texture_size, {0, 1}, "bindless.texture.size.2d.texture.i16.v2");
    texture_size = b->CreateZExt(texture_size, _create_type(Type::of<uint2>()), "bindless.texture.size.2d.texture.size.i32");
    auto shift = b->CreateVectorSplat(2u, level, "bindless.texture.size.2d.shift");
    texture_size = b->CreateLShr(texture_size, shift, "bindless.texture.size.2d.shifted");
    auto one = b->CreateVectorSplat(2u, _literal(1u), "bindless.texture.size.2d.one");
    texture_size = b->CreateMaxNum(texture_size, one, "bindless.texture.size.2d.max");
    return _create_stack_variable(texture_size, "bindless.texture.size.2d.size.addr");
}

::llvm::Value *LLVMCodegen::_builtin_bindless_texture_size3d(::llvm::Value *p_items, ::llvm::Value *p_index, ::llvm::Value *p_level) noexcept {
    auto b = _current_context()->builder.get();
    auto level = p_level == nullptr ? _literal(0u) : b->CreateLoad(b->getInt32Ty(), p_level, "bindless.texture.size.3d.level");
    auto index = b->CreateLoad(b->getInt32Ty(), p_index, "bindless.texture.size.3d.index");
    auto pp_texture = b->CreateInBoundsGEP(_bindless_item_type(), p_items, {index, _literal(2)}, "bindless.texture.size.3d.texture.ptr.ptr");
    auto p_texture = b->CreateLoad(_bindless_texture_type()->getPointerTo(), pp_texture, "bindless.texture.size.3d.texture.ptr");
    auto texture = b->CreateLoad(_bindless_texture_type(), p_texture, "bindless.texture.size.3d.texture");
    auto texture_size = b->CreateExtractValue(texture, 1u, "bindless.texture.size.3d.texture.size.i16");
    texture_size = b->CreateZExt(texture_size, _create_type(Type::of<uint3>()), "bindless.texture.size.3d.texture.size.i32");
    auto shift = b->CreateVectorSplat(4u, level, "bindless.texture.size.3d.shift");
    texture_size = b->CreateLShr(texture_size, shift, "bindless.texture.size.3d.shifted");
    auto one = b->CreateVectorSplat(4u, _literal(1u), "bindless.texture.size.3d.one");
    texture_size = b->CreateMaxNum(texture_size, one, "bindless.texture.size.3d.max");
    return _create_stack_variable(texture_size, "bindless.texture.size.3d.size.addr");
}

::llvm::Value *LLVMCodegen::_builtin_bindless_buffer_read(
    const Type *t, ::llvm::Value *p_items, ::llvm::Value *p_buffer_index, ::llvm::Value *p_elem_index) noexcept {
    auto elem_type = _create_type(t);
    auto b = _current_context()->builder.get();
    auto buffer_index = b->CreateLoad(b->getInt32Ty(), p_buffer_index, "bindless.buffer.read.buffer.index");
    auto buffer_ptr = b->CreateInBoundsGEP(_bindless_item_type(), p_items, {buffer_index, _literal(0u)}, "bindless.buffer.read.buffer.ptr");
    auto typeless_buffer = b->CreateLoad(b->getInt8PtrTy(), buffer_ptr, "bindless.buffer.read.buffer.typeless");
    auto buffer = b->CreateBitOrPointerCast(typeless_buffer, elem_type->getPointerTo(), "bindless.buffer.read.buffer");
    auto elem_index = b->CreateLoad(b->getInt32Ty(), p_elem_index, "bindless.buffer.read.elem.index");
    auto elem_ptr = b->CreateInBoundsGEP(elem_type, buffer, elem_index, "bindless.buffer.read.elem.ptr");
    auto elem = b->CreateLoad(elem_type, elem_ptr, "bindless.buffer.read.elem");
    return _create_stack_variable(elem, "bindless.buffer.read.elem.addr");
}

::llvm::Value *LLVMCodegen::_builtin_bindless_texture_read2d(
    ::llvm::Value *p_items, ::llvm::Value *p_index, ::llvm::Value *p_level, ::llvm::Value *p_uv) noexcept {
    auto b = _current_context()->builder.get();
    auto level = p_level == nullptr ? _literal(0u) : b->CreateLoad(b->getInt32Ty(), p_level, "bindless.texture.read.2d.level");
    auto index = b->CreateLoad(b->getInt32Ty(), p_index, "bindless.texture.read.2d.index");
    auto pp_texture = b->CreateInBoundsGEP(_bindless_item_type(), p_items, {index, _literal(1)}, "bindless.texture.read.2d.texture.ptr.ptr");
    auto p_texture = b->CreateLoad(_bindless_texture_type()->getPointerTo(), pp_texture, "bindless.texture.read.2d.texture.ptr");
    auto coord = b->CreateLoad(_create_type(Type::of<uint2>()), p_uv, "bindless.texture.read.2d.uv");
    auto coord_x = b->CreateExtractElement(coord, _literal(0u), "bindless.texture.read.2d.uv.x");
    auto coord_y = b->CreateExtractElement(coord, _literal(1u), "bindless.texture.read.2d.uv.y");
    auto func = _module->getFunction("bindless.texture.2d.read");
    if (func == nullptr) {
        func = ::llvm::Function::Create(
            ::llvm::FunctionType::get(
                ::llvm::FixedVectorType::get(b->getFloatTy(), 4u),
                {_bindless_texture_type()->getPointerTo(), b->getInt32Ty(), b->getInt32Ty(), b->getInt32Ty()}, false),
            ::llvm::Function::ExternalLinkage, "bindless.texture.2d.read", _module);
    }
    auto ret = b->CreateCall(func, {p_texture, level, coord_x, coord_y}, "bindless.texture.read.2d.ret");
    return _create_stack_variable(ret, "bindless.texture.read.2d.addr");
}

::llvm::Value *LLVMCodegen::_builtin_bindless_texture_read3d(
    ::llvm::Value *p_items, ::llvm::Value *p_index, ::llvm::Value *p_level, ::llvm::Value *p_uvw) noexcept {
    auto b = _current_context()->builder.get();
    auto level = p_level == nullptr ? _literal(0u) : b->CreateLoad(b->getInt32Ty(), p_level, "bindless.texture.read.3d.level");
    auto index = b->CreateLoad(b->getInt32Ty(), p_index, "bindless.texture.read.3d.index");
    auto pp_texture = b->CreateInBoundsGEP(_bindless_item_type(), p_items, {index, _literal(2)}, "bindless.texture.read.3d.texture.ptr.ptr");
    auto p_texture = b->CreateLoad(_bindless_texture_type()->getPointerTo(), pp_texture, "bindless.texture.read.3d.texture.ptr");
    auto coord = b->CreateLoad(_create_type(Type::of<uint3>()), p_uvw, "bindless.texture.read.3d.uvw");
    auto coord_x = b->CreateExtractElement(coord, _literal(0u), "bindless.texture.read.3d.uvw.x");
    auto coord_y = b->CreateExtractElement(coord, _literal(1u), "bindless.texture.read.3d.uvw.y");
    auto coord_z = b->CreateExtractElement(coord, _literal(2u), "bindless.texture.read.3d.uvw.z");
    auto func = _module->getFunction("bindless.texture.3d.read");
    if (func == nullptr) {
        func = ::llvm::Function::Create(
            ::llvm::FunctionType::get(
                ::llvm::FixedVectorType::get(b->getFloatTy(), 4u),
                {_bindless_texture_type()->getPointerTo(), b->getInt32Ty(), b->getInt32Ty(), b->getInt32Ty(), b->getInt32Ty()}, false),
            ::llvm::Function::ExternalLinkage, "bindless.texture.3d.read", _module);
    }
    auto ret = b->CreateCall(func, {p_texture, level, coord_x, coord_y, coord_z}, "bindless.texture.read.3d.ret");
    return _create_stack_variable(ret, "bindless.texture.read.3d.addr");
}

::llvm::Value *LLVMCodegen::_builtin_bindless_texture_sample2d(::llvm::Value *p_items, ::llvm::Value *p_index, ::llvm::Value *p_uv) noexcept {
    auto b = _current_context()->builder.get();
    auto index = b->CreateLoad(b->getInt32Ty(), p_index, "bindless.texture.sample.2d.index");
    auto pp_texture = b->CreateInBoundsGEP(_bindless_item_type(), p_items, {index, _literal(1)}, "bindless.texture.sample.2d.texture.ptr.ptr");
    auto p_texture = b->CreateLoad(_bindless_texture_type()->getPointerTo(), pp_texture, "bindless.texture.sample.2d.texture.ptr");
    auto p_sampler = b->CreateInBoundsGEP(_bindless_item_type(), p_items, {index, _literal(3)}, "bindless.texture.sample.2d.sampler.ptr");
    auto sampler = b->CreateLoad(b->getInt32Ty(), p_sampler, "bindless.texture.sample.2d.sampler");
    auto uv = b->CreateLoad(_create_type(Type::of<float2>()), p_uv, "bindless.texture.sample.2d.uv");
    auto uv_x = b->CreateExtractElement(uv, _literal(0u), "bindless.texture.sample.2d.uv.x");
    auto uv_y = b->CreateExtractElement(uv, _literal(1u), "bindless.texture.sample.2d.uv.y");
    auto func = _module->getFunction("bindless.texture.2d.sample");
    if (func == nullptr) {
        func = ::llvm::Function::Create(
            ::llvm::FunctionType::get(
                ::llvm::FixedVectorType::get(b->getFloatTy(), 4u),
                {_bindless_texture_type()->getPointerTo(), b->getInt32Ty(),
                 b->getFloatTy(), b->getFloatTy()},
                false),
            ::llvm::Function::ExternalLinkage, "bindless.texture.2d.sample", _module);
    }
    auto ret = b->CreateCall(func, {p_texture, sampler, uv_x, uv_y}, "bindless.texture.sample.2d.ret");
    return _create_stack_variable(ret, "bindless.texture.sample.2d.addr");
}

::llvm::Value *LLVMCodegen::_builtin_bindless_texture_sample3d(::llvm::Value *p_items, ::llvm::Value *p_index, ::llvm::Value *p_uvw) noexcept {
    auto b = _current_context()->builder.get();
    auto index = b->CreateLoad(b->getInt32Ty(), p_index, "bindless.texture.sample.3d.index");
    auto pp_texture = b->CreateInBoundsGEP(_bindless_item_type(), p_items, {index, _literal(2)}, "bindless.texture.sample.3d.texture.ptr.ptr");
    auto p_texture = b->CreateLoad(_bindless_texture_type()->getPointerTo(), pp_texture, "bindless.texture.sample.3d.texture.ptr");
    auto p_sampler = b->CreateInBoundsGEP(_bindless_item_type(), p_items, {index, _literal(4)}, "bindless.texture.sample.3d.sampler.ptr");
    auto sampler = b->CreateLoad(b->getInt32Ty(), p_sampler, "bindless.texture.sample.3d.sampler");
    auto uvw = b->CreateLoad(_create_type(Type::of<float3>()), p_uvw, "bindless.texture.sample.3d.uvw");
    auto uvw_x = b->CreateExtractElement(uvw, _literal(0u), "bindless.texture.sample.3d.uvw.x");
    auto uvw_y = b->CreateExtractElement(uvw, _literal(1u), "bindless.texture.sample.3d.uvw.y");
    auto uvw_z = b->CreateExtractElement(uvw, _literal(2u), "bindless.texture.sample.3d.uvw.z");
    auto func = _module->getFunction("bindless.texture.3d.sample");
    if (func == nullptr) {
        func = ::llvm::Function::Create(
            ::llvm::FunctionType::get(
                ::llvm::FixedVectorType::get(b->getFloatTy(), 4u),
                {_bindless_texture_type()->getPointerTo(), b->getInt32Ty(),
                 b->getFloatTy(), b->getFloatTy(), b->getFloatTy()},
                false),
            ::llvm::Function::ExternalLinkage, "bindless.texture.3d.sample", _module);
    }
    auto ret = b->CreateCall(func, {p_texture, sampler, uvw_x, uvw_y, uvw_z}, "bindless.texture.sample.3d.ret");
    return _create_stack_variable(ret, "bindless.texture.sample.3d.addr");
}

::llvm::Value *LLVMCodegen::_builtin_bindless_texture_sample2d_level(::llvm::Value *p_items, ::llvm::Value *p_index, ::llvm::Value *p_uv, ::llvm::Value *p_lod) noexcept {
    auto b = _current_context()->builder.get();
    auto index = b->CreateLoad(b->getInt32Ty(), p_index, "bindless.texture.sample.2d.level.index");
    auto pp_texture = b->CreateInBoundsGEP(_bindless_item_type(), p_items, {index, _literal(1)}, "bindless.texture.sample.2d.level.texture.ptr.ptr");
    auto p_texture = b->CreateLoad(_bindless_texture_type()->getPointerTo(), pp_texture, "bindless.texture.sample.2d.level.texture.ptr");
    auto p_sampler = b->CreateInBoundsGEP(_bindless_item_type(), p_items, {index, _literal(3)}, "bindless.texture.sample.2d.level.sampler.ptr");
    auto sampler = b->CreateLoad(b->getInt32Ty(), p_sampler, "bindless.texture.sample.2d.level.sampler");
    auto uv = b->CreateLoad(_create_type(Type::of<float2>()), p_uv, "bindless.texture.sample.2d.level.uv");
    auto uv_x = b->CreateExtractElement(uv, _literal(0u), "bindless.texture.sample.2d.level.uv.x");
    auto uv_y = b->CreateExtractElement(uv, _literal(1u), "bindless.texture.sample.2d.level.uv.y");
    auto lod = b->CreateLoad(b->getFloatTy(), p_lod, "bindless.texture.sample.2d.level.lod");
    auto func = _module->getFunction("bindless.texture.2d.sample.level");
    if (func == nullptr) {
        func = ::llvm::Function::Create(
            ::llvm::FunctionType::get(
                ::llvm::FixedVectorType::get(b->getFloatTy(), 4u),
                {_bindless_texture_type()->getPointerTo(), b->getInt32Ty(),
                 b->getFloatTy(), b->getFloatTy(), b->getFloatTy()},
                false),
            ::llvm::Function::ExternalLinkage, "bindless.texture.2d.sample.level", _module);
    }
    auto ret = b->CreateCall(func, {p_texture, sampler, uv_x, uv_y, lod}, "bindless.texture.sample.2d.level.ret");
    return _create_stack_variable(ret, "bindless.texture.sample.2d.level.addr");
}

::llvm::Value *LLVMCodegen::_builtin_bindless_texture_sample3d_level(::llvm::Value *p_items, ::llvm::Value *p_index, ::llvm::Value *p_uvw, ::llvm::Value *p_lod) noexcept {
    auto b = _current_context()->builder.get();
    auto index = b->CreateLoad(b->getInt32Ty(), p_index, "bindless.texture.sample.3d.level.index");
    auto pp_texture = b->CreateInBoundsGEP(_bindless_item_type(), p_items, {index, _literal(2)}, "bindless.texture.sample.3d.level.texture.ptr.ptr");
    auto p_texture = b->CreateLoad(_bindless_texture_type()->getPointerTo(), pp_texture, "bindless.texture.sample.3d.level.texture.ptr");
    auto p_sampler = b->CreateInBoundsGEP(_bindless_item_type(), p_items, {index, _literal(4)}, "bindless.texture.sample.3d.level.sampler.ptr");
    auto sampler = b->CreateLoad(b->getInt32Ty(), p_sampler, "bindless.texture.sample.3d.level.sampler");
    auto uvw = b->CreateLoad(_create_type(Type::of<float3>()), p_uvw, "bindless.texture.sample.3d.level.uvw");
    auto uvw_x = b->CreateExtractElement(uvw, _literal(0u), "bindless.texture.sample.3d.level.uvw.x");
    auto uvw_y = b->CreateExtractElement(uvw, _literal(1u), "bindless.texture.sample.3d.level.uvw.y");
    auto uvw_z = b->CreateExtractElement(uvw, _literal(2u), "bindless.texture.sample.3d.level.uvw.z");
    auto lod = b->CreateLoad(b->getFloatTy(), p_lod, "bindless.texture.sample.3d.level.lod");
    auto func = _module->getFunction("bindless.texture.3d.sample.level");
    if (func == nullptr) {
        func = ::llvm::Function::Create(
            ::llvm::FunctionType::get(
                ::llvm::FixedVectorType::get(b->getFloatTy(), 4u),
                {_bindless_texture_type()->getPointerTo(), b->getInt32Ty(),
                 b->getFloatTy(), b->getFloatTy(), b->getFloatTy(), b->getFloatTy()},
                false),
            ::llvm::Function::ExternalLinkage, "bindless.texture.3d.sample.level", _module);
    }
    auto ret = b->CreateCall(func, {p_texture, sampler, uvw_x, uvw_y, uvw_z, lod}, "bindless.texture.sample.3d.level.ret");
    return _create_stack_variable(ret, "bindless.texture.sample.3d.level.addr");
}

::llvm::Value *LLVMCodegen::_builtin_bindless_texture_sample2d_grad(::llvm::Value *p_items, ::llvm::Value *p_index, ::llvm::Value *p_uv, ::llvm::Value *p_dpdx, ::llvm::Value *p_dpdy) noexcept {
    auto b = _current_context()->builder.get();
    auto index = b->CreateLoad(b->getInt32Ty(), p_index, "bindless.texture.sample.2d.grad.index");
    auto pp_texture = b->CreateInBoundsGEP(_bindless_item_type(), p_items, {index, _literal(1)}, "bindless.texture.sample.2d.grad.texture.ptr.ptr");
    auto p_texture = b->CreateLoad(_bindless_texture_type()->getPointerTo(), pp_texture, "bindless.texture.sample.2d.grad.texture.ptr");
    auto p_sampler = b->CreateInBoundsGEP(_bindless_item_type(), p_items, {index, _literal(3)}, "bindless.texture.sample.2d.grad.sampler.ptr");
    auto sampler = b->CreateLoad(b->getInt32Ty(), p_sampler, "bindless.texture.sample.2d.grad.sampler");
    auto uv = b->CreateLoad(_create_type(Type::of<float2>()), p_uv, "bindless.texture.sample.2d.grad.uv");
    auto uv_x = b->CreateExtractElement(uv, _literal(0u), "bindless.texture.sample.2d.grad.uv.x");
    auto uv_y = b->CreateExtractElement(uv, _literal(1u), "bindless.texture.sample.2d.grad.uv.y");
    p_dpdx = b->CreateBitOrPointerCast(p_dpdx, b->getInt64Ty()->getPointerTo(), "bindless.texture.sample.2d.grad.dpdx.addr");
    p_dpdy = b->CreateBitOrPointerCast(p_dpdy, b->getInt64Ty()->getPointerTo(), "bindless.texture.sample.2d.grad.dpdy.addr");
    auto dpdx = b->CreateLoad(b->getInt64Ty(), p_dpdx, "bindless.texture.sample.2d.grad.dpdx");
    auto dpdy = b->CreateLoad(b->getInt64Ty(), p_dpdy, "bindless.texture.sample.2d.grad.dpdy");
    auto func = _module->getFunction("bindless.texture.2d.sample.grad");
    if (func == nullptr) {
        func = ::llvm::Function::Create(
            ::llvm::FunctionType::get(
                ::llvm::FixedVectorType::get(b->getFloatTy(), 4u),
                {_bindless_texture_type()->getPointerTo(), b->getInt32Ty(),
                 b->getFloatTy(), b->getFloatTy(), b->getInt64Ty(), b->getInt64Ty()},
                false),
            ::llvm::Function::ExternalLinkage, "bindless.texture.2d.sample.grad", _module);
    }
    auto ret = b->CreateCall(func, {p_texture, sampler, uv_x, uv_y, dpdx, dpdy}, "bindless.texture.sample.2d.grad.ret");
    return _create_stack_variable(ret, "bindless.texture.sample.2d.grad.addr");
}

::llvm::Value *LLVMCodegen::_builtin_bindless_texture_sample3d_grad(::llvm::Value *p_items, ::llvm::Value *p_index, ::llvm::Value *p_uvw, ::llvm::Value *p_dpdx, ::llvm::Value *p_dpdy) noexcept {
    auto b = _current_context()->builder.get();
    auto index = b->CreateLoad(b->getInt32Ty(), p_index, "bindless.texture.sample.3d.grad.index");
    auto pp_texture = b->CreateInBoundsGEP(_bindless_item_type(), p_items, {index, _literal(1)}, "bindless.texture.sample.3d.grad.texture.ptr.ptr");
    auto p_texture = b->CreateLoad(_bindless_texture_type()->getPointerTo(), pp_texture, "bindless.texture.sample.3d.grad.texture.ptr");
    auto p_sampler = b->CreateInBoundsGEP(_bindless_item_type(), p_items, {index, _literal(3)}, "bindless.texture.sample.3d.grad.sampler.ptr");
    p_uvw = _builtin_bitwise_cast(Type::of<uint3>(), Type::of<float3>(), p_uvw);
    p_dpdx = _builtin_bitwise_cast(Type::of<uint3>(), Type::of<float3>(), p_dpdx);
    p_dpdy = _builtin_bitwise_cast(Type::of<uint3>(), Type::of<float3>(), p_dpdy);
    auto uvw = b->CreateLoad(_create_type(Type::of<uint3>()), p_uvw, "bindless.texture.sample.3d.grad.uvw");
    auto dpdx = b->CreateLoad(_create_type(Type::of<uint3>()), p_dpdx, "bindless.texture.sample.3d.grad.dpdx");
    auto dpdy = b->CreateLoad(_create_type(Type::of<uint3>()), p_dpdy, "bindless.texture.sample.3d.grad.dpdy");
    auto p_u = _create_stack_variable(b->CreateExtractElement(uvw, _literal(0u), "bindless.texture.sample.3d.grad.uvw.x"), "bindless.texture.sample.3d.grad.uvw.x.addr");
    auto p_v = _create_stack_variable(b->CreateExtractElement(uvw, _literal(1u), "bindless.texture.sample.3d.grad.uvw.y"), "bindless.texture.sample.3d.grad.uvw.y.addr");
    auto p_w = _create_stack_variable(b->CreateExtractElement(uvw, _literal(2u), "bindless.texture.sample.3d.grad.uvw.z"), "bindless.texture.sample.3d.grad.uvw.z.addr");
    auto p_dudx = _create_stack_variable(b->CreateExtractElement(dpdx, _literal(0u), "bindless.texture.sample.3d.grad.dpdx.x"), "bindless.texture.sample.3d.grad.dpdx.x.addr");
    auto p_dvdx = _create_stack_variable(b->CreateExtractElement(dpdy, _literal(1u), "bindless.texture.sample.3d.grad.dpdy.y"), "bindless.texture.sample.3d.grad.dpdy.y.addr");
    auto p_dwdx = _create_stack_variable(b->CreateExtractElement(dpdy, _literal(2u), "bindless.texture.sample.3d.grad.dpdy.z"), "bindless.texture.sample.3d.grad.dpdy.z.addr");
    auto p_dudy = _create_stack_variable(b->CreateExtractElement(dpdx, _literal(1u), "bindless.texture.sample.3d.grad.dpdx.y"), "bindless.texture.sample.3d.grad.dpdx.y.addr");
    auto p_dvdy = _create_stack_variable(b->CreateExtractElement(dpdy, _literal(0u), "bindless.texture.sample.3d.grad.dpdy.x"), "bindless.texture.sample.3d.grad.dpdy.x.addr");
    auto p_dwdy = _create_stack_variable(b->CreateExtractElement(dpdy, _literal(2u), "bindless.texture.sample.3d.grad.dpdy.z"), "bindless.texture.sample.3d.grad.dpdy.z.addr");
    auto p_sampler_and_w = b->CreateBitOrPointerCast(_make_int2(p_sampler, p_w), b->getInt64Ty()->getPointerTo(), "bindless.texture.sample.3d.grad.sampler.w.addr");
    auto p_uv = b->CreateBitOrPointerCast(_make_int2(p_u, p_v), b->getInt64Ty()->getPointerTo(), "bindless.texture.sample.3d.grad.uv.addr");
    auto p_dudxy = b->CreateBitOrPointerCast(_make_int2(p_dudx, p_dudy), b->getInt64Ty()->getPointerTo(), "bindless.texture.sample.3d.grad.dudxy.addr");
    auto p_dvdxy = b->CreateBitOrPointerCast(_make_int2(p_dvdx, p_dvdy), b->getInt64Ty()->getPointerTo(), "bindless.texture.sample.3d.grad.dvdxy.addr");
    auto p_dwdxy = b->CreateBitOrPointerCast(_make_int2(p_dwdx, p_dwdy), b->getInt64Ty()->getPointerTo(), "bindless.texture.sample.3d.grad.dwdxy.addr");
    auto sampler_and_w = b->CreateLoad(b->getInt64Ty(), p_sampler_and_w, "bindless.texture.sample.3d.grad.sampler.and.w");
    auto uv = b->CreateLoad(b->getInt64Ty(), p_uv, "bindless.texture.sample.3d.grad.uv");
    auto dudxy = b->CreateLoad(b->getInt64Ty(), p_dudxy, "bindless.texture.sample.3d.grad.dudxy");
    auto dvdxy = b->CreateLoad(b->getInt64Ty(), p_dvdxy, "bindless.texture.sample.3d.grad.dvdxy");
    auto dwdxy = b->CreateLoad(b->getInt64Ty(), p_dwdxy, "bindless.texture.sample.3d.grad.dwdxy");
    auto func = _module->getFunction("bindless.texture.sample.3d.grad");
    if (func == nullptr) {
        func = ::llvm::Function::Create(
            ::llvm::FunctionType::get(
                ::llvm::FixedVectorType::get(b->getFloatTy(), 4u),
                {_bindless_texture_type()->getPointerTo(), b->getInt64Ty(),
                 b->getFloatTy(), b->getFloatTy(), b->getInt64Ty(), b->getInt64Ty()},
                false),
            ::llvm::Function::ExternalLinkage, "bindless.texture.3d.sample.grad", _module);
    }
    auto ret = b->CreateCall(func, {p_texture, sampler_and_w, uv, dudxy, dvdxy, dwdxy}, "bindless.texture.sample.3d.grad.ret");
    return _create_stack_variable(ret, "bindless.texture.sample.3d.grad.addr");
}

::llvm::Value *LLVMCodegen::_builtin_asinh(const Type *t, ::llvm::Value *v) noexcept {
    // log(x + sqrt(x * x + 1.0))
    auto one = _create_stack_variable(_literal(1.0f), "asinh.one");
    if (t->is_vector()) { one = _builtin_static_cast(t, t->element(), one); }
    return _builtin_log(t, _builtin_add(t, v, _builtin_sqrt(t, _builtin_add(t, _builtin_mul(t, v, v), one))));
}

::llvm::Value *LLVMCodegen::_builtin_acosh(const Type *t, ::llvm::Value *v) noexcept {
    // log(x + sqrt(x * x - 1.0))
    auto one = _create_stack_variable(_literal(1.0f), "cosh.one");
    if (t->is_vector()) { one = _builtin_static_cast(t, t->element(), one); }
    return _builtin_log(t, _builtin_add(t, v, _builtin_sqrt(t, _builtin_sub(t, _builtin_mul(t, v, v), one))));
}

::llvm::Value *LLVMCodegen::_builtin_atanh(const Type *t, ::llvm::Value *v) noexcept {
    // 0.5 * log((1.0 + x) / (1.0 - x))
    auto one = _create_stack_variable(_literal(1.0f), "tanh.one");
    auto half = _create_stack_variable(_literal(0.5f), "tanh.half");
    if (t->is_vector()) {
        one = _builtin_static_cast(t, t->element(), one);
        half = _builtin_static_cast(t, t->element(), half);
    }
    auto one_plus_x = _builtin_add(t, one, v);
    auto one_minus_x = _builtin_sub(t, one, v);
    auto one_plus_x_over_one_minus_x = _builtin_div(t, one_plus_x, one_minus_x);
    auto log_of_one_plus_x_over_one_minus_x = _builtin_log(t, one_plus_x_over_one_minus_x);
    return _builtin_mul(t, half, log_of_one_plus_x_over_one_minus_x);
}

::llvm::Value *LLVMCodegen::_builtin_cosh(const Type *t, ::llvm::Value *v) noexcept {
    // y = exp(x)
    // 0.5 * (y + 1 / y)
    auto half = _create_stack_variable(_literal(0.5f), "cosh.half");
    auto one = _create_stack_variable(_literal(1.0f), "cosh.one");
    if (t->is_vector()) {
        half = _builtin_static_cast(t, t->element(), half);
        one = _builtin_static_cast(t, t->element(), one);
    }
    auto exp_x = _builtin_exp(t, v);
    auto exp_minus_x = _builtin_div(t, one, exp_x);
    auto exp_x_plus_exp_minus_x = _builtin_add(t, exp_x, exp_minus_x);
    return _builtin_mul(t, half, exp_x_plus_exp_minus_x);
}

::llvm::Value *LLVMCodegen::_builtin_sinh(const Type *t, ::llvm::Value *v) noexcept {
    // y = exp(x)
    // 0.5 * (y – 1 / y)
    auto half = _create_stack_variable(_literal(0.5f), "sinh.half");
    auto one = _create_stack_variable(_literal(1.0f), "sinh.one");
    if (t->is_vector()) {
        half = _builtin_static_cast(t, t->element(), half);
        one = _builtin_static_cast(t, t->element(), one);
    }
    auto exp_x = _builtin_exp(t, v);
    auto exp_minus_x = _builtin_div(t, one, exp_x);
    auto exp_x_minus_exp_minus_x = _builtin_sub(t, exp_x, exp_minus_x);
    return _builtin_mul(t, half, exp_x_minus_exp_minus_x);
}

::llvm::Value *LLVMCodegen::_builtin_tanh(const Type *t, ::llvm::Value *v) noexcept {
    // y = exp(2.0 * x)
    // (y - 1.0) / (y + 1.0)
    auto one = _create_stack_variable(_literal(1.0f), "tanh.one");
    auto two = _create_stack_variable(_literal(2.0f), "tanh.two");
    if (t->is_vector()) {
        one = _builtin_static_cast(t, t->element(), one);
        two = _builtin_static_cast(t, t->element(), two);
    }
    auto y = _builtin_exp(t, _builtin_mul(t, two, v));
    auto y_minus_one = _builtin_sub(t, y, one);
    auto y_plus_one = _builtin_add(t, y, one);
    return _builtin_div(t, y_minus_one, y_plus_one);
}

}// namespace luisa::compute::llvm
