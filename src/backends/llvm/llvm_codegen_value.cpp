//
// Created by Mike Smith on 2022/5/23.
//

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-static-cast-downcast"

#include <backends/llvm/llvm_codegen.h>

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-static-cast-downcast"

namespace luisa::compute::llvm {

::llvm::Value *LLVMCodegen::_literal(int x) noexcept {
    return ::llvm::ConstantInt::get(::llvm::Type::getInt32Ty(_context), x);
}

::llvm::Value *LLVMCodegen::_literal(uint x) noexcept {
    return ::llvm::ConstantInt::get(::llvm::Type::getInt32Ty(_context), x);
}

::llvm::Value *LLVMCodegen::_literal(bool x) noexcept {
    return ::llvm::ConstantInt::get(::llvm::Type::getInt8Ty(_context),
                                    static_cast<uint8_t>(x));
}

::llvm::Value *LLVMCodegen::_literal(float x) noexcept {
    return ::llvm::ConstantFP::get(::llvm::Type::getFloatTy(_context), x);
}

::llvm::Value *LLVMCodegen::_literal(int2 x) noexcept {
    return _literal(make_uint2(x));
}

::llvm::Value *LLVMCodegen::_literal(uint2 x) noexcept {
    return ::llvm::ConstantVector::get({::llvm::ConstantInt::get(::llvm::Type::getInt32Ty(_context), x.x),
                                        ::llvm::ConstantInt::get(::llvm::Type::getInt32Ty(_context), x.y)});
}

::llvm::Value *LLVMCodegen::_literal(bool2 x) noexcept {
    return ::llvm::ConstantVector::get({::llvm::ConstantInt::get(::llvm::Type::getInt8Ty(_context),
                                                                 static_cast<uint8_t>(x.x)),
                                        ::llvm::ConstantInt::get(::llvm::Type::getInt8Ty(_context),
                                                                 static_cast<uint8_t>(x.y))});
}

::llvm::Value *LLVMCodegen::_literal(float2 x) noexcept {
    return ::llvm::ConstantVector::get({::llvm::ConstantFP::get(::llvm::Type::getFloatTy(_context), x.x),
                                        ::llvm::ConstantFP::get(::llvm::Type::getFloatTy(_context), x.y)});
}

::llvm::Value *LLVMCodegen::_literal(int3 x) noexcept {
    return _literal(make_uint3(x));
}

::llvm::Value *LLVMCodegen::_literal(uint3 x) noexcept {
    return _literal(make_uint4(x, 0u));
}

::llvm::Value *LLVMCodegen::_literal(bool3 x) noexcept {
    return _literal(make_bool4(x, false));
}

::llvm::Value *LLVMCodegen::_literal(float3 x) noexcept {
    return _literal(make_float4(x, 0.0f));
}

::llvm::Value *LLVMCodegen::_literal(int4 x) noexcept {
    return _literal(make_uint4(x));
}

::llvm::Value *LLVMCodegen::_literal(uint4 x) noexcept {
    return ::llvm::ConstantVector::get({::llvm::ConstantInt::get(::llvm::Type::getInt32Ty(_context), x.x),
                                        ::llvm::ConstantInt::get(::llvm::Type::getInt32Ty(_context), x.y),
                                        ::llvm::ConstantInt::get(::llvm::Type::getInt32Ty(_context), x.z),
                                        ::llvm::ConstantInt::get(::llvm::Type::getInt32Ty(_context), x.w)});
}

::llvm::Value *LLVMCodegen::_literal(bool4 x) noexcept {
    return ::llvm::ConstantVector::get({::llvm::ConstantInt::get(::llvm::Type::getInt8Ty(_context),
                                                                 static_cast<uint8_t>(x.x)),
                                        ::llvm::ConstantInt::get(::llvm::Type::getInt8Ty(_context),
                                                                 static_cast<uint8_t>(x.y)),
                                        ::llvm::ConstantInt::get(::llvm::Type::getInt8Ty(_context),
                                                                 static_cast<uint8_t>(x.z)),
                                        ::llvm::ConstantInt::get(::llvm::Type::getInt8Ty(_context),
                                                                 static_cast<uint8_t>(x.w))});
}

::llvm::Value *LLVMCodegen::_literal(float4 x) noexcept {
    return ::llvm::ConstantVector::get({::llvm::ConstantFP::get(::llvm::Type::getFloatTy(_context), x.x),
                                        ::llvm::ConstantFP::get(::llvm::Type::getFloatTy(_context), x.y),
                                        ::llvm::ConstantFP::get(::llvm::Type::getFloatTy(_context), x.z),
                                        ::llvm::ConstantFP::get(::llvm::Type::getFloatTy(_context), x.w)});
}

::llvm::Value *LLVMCodegen::_literal(float2x2 x) noexcept {
    return ::llvm::ConstantStruct::get(
        static_cast<::llvm::StructType *>(_create_type(Type::of<float2x2>())),
        static_cast<::llvm::Constant *>(_literal(x[0])),
        static_cast<::llvm::Constant *>(_literal(x[1])));
}

::llvm::Value *LLVMCodegen::_literal(float3x3 x) noexcept {
    return ::llvm::ConstantStruct::get(
        static_cast<::llvm::StructType *>(_create_type(Type::of<float3x3>())),
        static_cast<::llvm::Constant *>(_literal(x[0])),
        static_cast<::llvm::Constant *>(_literal(x[1])),
        static_cast<::llvm::Constant *>(_literal(x[2])));
}

::llvm::Value *LLVMCodegen::_literal(float4x4 x) noexcept {
    return ::llvm::ConstantStruct::get(
        static_cast<::llvm::StructType *>(_create_type(Type::of<float4x4>())),
        static_cast<::llvm::Constant *>(_literal(x[0])),
        static_cast<::llvm::Constant *>(_literal(x[1])),
        static_cast<::llvm::Constant *>(_literal(x[2])),
        static_cast<::llvm::Constant *>(_literal(x[3])));
}

::llvm::Value *LLVMCodegen::_create_stack_variable(::llvm::Value *x, luisa::string_view name) noexcept {
    auto builder = _current_context()->builder.get();
    auto t = x->getType();
    if (t->isIntegerTy(1)) {
        // special handling for int1
        return _create_stack_variable(
            builder->CreateZExt(x, builder->getInt8Ty(), "bit_to_bool"), name);
    }
    if (t->isVectorTy() && static_cast<::llvm::VectorType *>(t)->getElementType()->isIntegerTy(1)) {
        // special handling for int1 vector
        auto dim = static_cast<::llvm::VectorType *>(t)->getElementCount();
        auto dst_type = ::llvm::VectorType::get(builder->getInt8Ty(), dim);
        return _create_stack_variable(builder->CreateZExt(x, dst_type, "bit_to_bool"), name);
    }
    auto p = builder->CreateAlloca(x->getType(), nullptr, name);
    p->setAlignment(::llvm::Align{16u});
    builder->CreateStore(x, p);
    return p;
}

::llvm::Value *LLVMCodegen::_create_constant(ConstantData c) noexcept {
    auto key = c.hash();
    if (auto iter = _constants.find(key); iter != _constants.end()) {
        return iter->second;
    }
    auto value = luisa::visit(
        [this](auto s) noexcept {
            std::vector<::llvm::Constant *> elements;
            elements.reserve(s.size());
            for (auto x : s) { elements.push_back(static_cast<::llvm::Constant *>(_literal(x))); }
            using T = std::remove_cvref_t<decltype(s[0])>;
            auto array_type = ::llvm::ArrayType::get(
                _create_type(Type::of<T>()), static_cast<unsigned>(elements.size()));
            return ::llvm::ConstantArray::get(array_type, elements);
        },
        c.view());
    auto name = luisa::format("constant_{:016x}", key);
    _module->getOrInsertGlobal(luisa::string_view{name}, value->getType());
    auto global = _module->getNamedGlobal(luisa::string_view{name});
    global->setConstant(true);
    global->setLinkage(::llvm::GlobalValue::InternalLinkage);
    global->setInitializer(value);
    global->setUnnamedAddr(::llvm::GlobalValue::UnnamedAddr::Global);
    return _constants.emplace(key, global).first->second;
}

::llvm::Value *LLVMCodegen::_make_int2(::llvm::Value *px, ::llvm::Value *py) noexcept {
    auto b = _current_context()->builder.get();
    auto x = b->CreateLoad(b->getInt32Ty(), px, "v.x");
    auto y = b->CreateLoad(b->getInt32Ty(), py, "v.y");
    auto v = static_cast<::llvm::Value *>(::llvm::UndefValue::get(
        _create_type(Type::of<int2>())));
    v = b->CreateInsertElement(v, x, static_cast<uint64_t>(0u), "int2.x");
    v = b->CreateInsertElement(v, y, static_cast<uint64_t>(1u), "int2.xy");
    return _create_stack_variable(v, "int2.addr");
}

::llvm::Value *LLVMCodegen::_make_int3(::llvm::Value *px, ::llvm::Value *py, ::llvm::Value *pz) noexcept {
    auto b = _current_context()->builder.get();
    auto x = b->CreateLoad(b->getInt32Ty(), px, "v.x");
    auto y = b->CreateLoad(b->getInt32Ty(), py, "v.y");
    auto z = b->CreateLoad(b->getInt32Ty(), pz, "v.z");
    auto v = static_cast<::llvm::Value *>(::llvm::UndefValue::get(
        _create_type(Type::of<int3>())));
    v = b->CreateInsertElement(v, x, static_cast<uint64_t>(0u), "int3.x");
    v = b->CreateInsertElement(v, y, static_cast<uint64_t>(1u), "int3.xy");
    v = b->CreateInsertElement(v, z, static_cast<uint64_t>(2u), "int3.xyz");
    return _create_stack_variable(v, "int3.addr");
}

::llvm::Value *LLVMCodegen::_make_int4(::llvm::Value *px, ::llvm::Value *py, ::llvm::Value *pz, ::llvm::Value *pw) noexcept {
    auto b = _current_context()->builder.get();
    auto x = b->CreateLoad(b->getInt32Ty(), px, "v.x");
    auto y = b->CreateLoad(b->getInt32Ty(), py, "v.y");
    auto z = b->CreateLoad(b->getInt32Ty(), pz, "v.z");
    auto w = b->CreateLoad(b->getInt32Ty(), pw, "v.w");
    auto v = static_cast<::llvm::Value *>(::llvm::UndefValue::get(
        _create_type(Type::of<int4>())));
    v = b->CreateInsertElement(v, x, static_cast<uint64_t>(0u), "int4.x");
    v = b->CreateInsertElement(v, y, static_cast<uint64_t>(1u), "int4.xy");
    v = b->CreateInsertElement(v, z, static_cast<uint64_t>(2u), "int4.xyz");
    v = b->CreateInsertElement(v, w, static_cast<uint64_t>(3u), "int4.xyzw");
    return _create_stack_variable(v, "int4.addr");
}

::llvm::Value *LLVMCodegen::_make_bool2(::llvm::Value *px, ::llvm::Value *py) noexcept {
    auto b = _current_context()->builder.get();
    auto x = b->CreateLoad(b->getInt8Ty(), px, "v.x");
    auto y = b->CreateLoad(b->getInt8Ty(), py, "v.y");
    auto v = static_cast<::llvm::Value *>(::llvm::UndefValue::get(
        _create_type(Type::of<bool2>())));
    v = b->CreateInsertElement(v, x, static_cast<uint64_t>(0u), "bool2.x");
    v = b->CreateInsertElement(v, y, static_cast<uint64_t>(1u), "bool2.xy");
    return _create_stack_variable(v, "bool2.addr");
}

::llvm::Value *LLVMCodegen::_make_bool3(::llvm::Value *px, ::llvm::Value *py, ::llvm::Value *pz) noexcept {
    auto b = _current_context()->builder.get();
    auto x = b->CreateLoad(b->getInt8Ty(), px, "v.x");
    auto y = b->CreateLoad(b->getInt8Ty(), py, "v.y");
    auto z = b->CreateLoad(b->getInt8Ty(), pz, "v.z");
    auto v = static_cast<::llvm::Value *>(::llvm::UndefValue::get(
        _create_type(Type::of<bool3>())));
    v = b->CreateInsertElement(v, x, static_cast<uint64_t>(0u), "bool3.x");
    v = b->CreateInsertElement(v, y, static_cast<uint64_t>(1u), "bool3.xy");
    v = b->CreateInsertElement(v, z, static_cast<uint64_t>(2u), "bool3.xyz");
    return _create_stack_variable(v, "bool3.addr");
}

::llvm::Value *LLVMCodegen::_make_bool4(::llvm::Value *px, ::llvm::Value *py, ::llvm::Value *pz, ::llvm::Value *pw) noexcept {
    auto b = _current_context()->builder.get();
    auto x = b->CreateLoad(b->getInt8Ty(), px, "v.x");
    auto y = b->CreateLoad(b->getInt8Ty(), py, "v.y");
    auto z = b->CreateLoad(b->getInt8Ty(), pz, "v.z");
    auto w = b->CreateLoad(b->getInt8Ty(), pw, "v.w");
    auto v = static_cast<::llvm::Value *>(::llvm::UndefValue::get(
        _create_type(Type::of<bool4>())));
    v = b->CreateInsertElement(v, x, static_cast<uint64_t>(0u), "bool4.x");
    v = b->CreateInsertElement(v, y, static_cast<uint64_t>(1u), "bool4.xy");
    v = b->CreateInsertElement(v, z, static_cast<uint64_t>(2u), "bool4.xyz");
    v = b->CreateInsertElement(v, w, static_cast<uint64_t>(3u), "bool4.xyzw");
    return _create_stack_variable(v, "bool4.addr");
}

::llvm::Value *LLVMCodegen::_make_float2(::llvm::Value *px, ::llvm::Value *py) noexcept {
    auto b = _current_context()->builder.get();
    auto x = b->CreateLoad(b->getFloatTy(), px, "v.x");
    auto y = b->CreateLoad(b->getFloatTy(), py, "v.y");
    auto v = static_cast<::llvm::Value *>(::llvm::UndefValue::get(
        _create_type(Type::of<float2>())));
    v = b->CreateInsertElement(v, x, static_cast<uint64_t>(0u), "float2.x");
    v = b->CreateInsertElement(v, y, static_cast<uint64_t>(1u), "float2.xy");
    return _create_stack_variable(v, "float2.addr");
}

::llvm::Value *LLVMCodegen::_make_float3(::llvm::Value *px, ::llvm::Value *py, ::llvm::Value *pz) noexcept {
    auto b = _current_context()->builder.get();
    auto x = b->CreateLoad(b->getFloatTy(), px, "v.x");
    auto y = b->CreateLoad(b->getFloatTy(), py, "v.y");
    auto z = b->CreateLoad(b->getFloatTy(), pz, "v.z");
    auto v = static_cast<::llvm::Value *>(::llvm::UndefValue::get(
        _create_type(Type::of<float3>())));
    v = b->CreateInsertElement(v, x, static_cast<uint64_t>(0u), "float3.x");
    v = b->CreateInsertElement(v, y, static_cast<uint64_t>(1u), "float3.xy");
    v = b->CreateInsertElement(v, z, static_cast<uint64_t>(2u), "float3.xyz");
    return _create_stack_variable(v, "float3.addr");
}

::llvm::Value *LLVMCodegen::_make_float4(::llvm::Value *px, ::llvm::Value *py, ::llvm::Value *pz, ::llvm::Value *pw) noexcept {
    auto b = _current_context()->builder.get();
    auto x = b->CreateLoad(b->getFloatTy(), px, "v.x");
    auto y = b->CreateLoad(b->getFloatTy(), py, "v.y");
    auto z = b->CreateLoad(b->getFloatTy(), pz, "v.z");
    auto w = b->CreateLoad(b->getFloatTy(), pw, "v.w");
    auto v = static_cast<::llvm::Value *>(::llvm::UndefValue::get(
        _create_type(Type::of<float4>())));
    v = b->CreateInsertElement(v, x, static_cast<uint64_t>(0u), "float4.x");
    v = b->CreateInsertElement(v, y, static_cast<uint64_t>(1u), "float4.xy");
    v = b->CreateInsertElement(v, z, static_cast<uint64_t>(2u), "float4.xyz");
    v = b->CreateInsertElement(v, w, static_cast<uint64_t>(3u), "float4.xyzw");
    return _create_stack_variable(v, "float4.addr");
}

::llvm::Value *LLVMCodegen::_make_float2x2(::llvm::Value *p0, ::llvm::Value *p1) noexcept {
    auto b = _current_context()->builder.get();
    auto t = _create_type(Type::of<float2x2>());
    auto m = b->CreateAlloca(t, nullptr, "float2x2.addr");
    m->setAlignment(::llvm::Align{16});
    auto m0 = b->CreateStructGEP(t, m, 0u, "float2x2.a");
    auto m1 = b->CreateStructGEP(t, m, 1u, "float2x2.b");
    auto col_type = _create_type(Type::of<float2>());
    b->CreateStore(b->CreateLoad(col_type, p0, "m.a"), m0);
    b->CreateStore(b->CreateLoad(col_type, p1, "m.b"), m1);
    return m;
}

::llvm::Value *LLVMCodegen::_make_float3x3(::llvm::Value *p0, ::llvm::Value *p1, ::llvm::Value *p2) noexcept {
    auto b = _current_context()->builder.get();
    auto t = _create_type(Type::of<float3x3>());
    auto m = b->CreateAlloca(t, nullptr, "float3x3.addr");
    m->setAlignment(::llvm::Align{16});
    auto m0 = b->CreateStructGEP(t, m, 0u, "float3x3.a");
    auto m1 = b->CreateStructGEP(t, m, 1u, "float3x3.b");
    auto m2 = b->CreateStructGEP(t, m, 2u, "float3x3.c");
    auto col_type = _create_type(Type::of<float3>());
    b->CreateStore(b->CreateLoad(col_type, p0, "m.a"), m0);
    b->CreateStore(b->CreateLoad(col_type, p1, "m.b"), m1);
    b->CreateStore(b->CreateLoad(col_type, p2, "m.c"), m2);
    return m;
}

::llvm::Value *LLVMCodegen::_make_float4x4(::llvm::Value *p0, ::llvm::Value *p1, ::llvm::Value *p2, ::llvm::Value *p3) noexcept {
    auto b = _current_context()->builder.get();
    auto t = _create_type(Type::of<float4x4>());
    auto m = b->CreateAlloca(t, nullptr, "float4x4.addr");
    m->setAlignment(::llvm::Align{16});
    auto m0 = b->CreateStructGEP(t, m, 0u, "float4x4.a");
    auto m1 = b->CreateStructGEP(t, m, 1u, "float4x4.b");
    auto m2 = b->CreateStructGEP(t, m, 2u, "float4x4.c");
    auto m3 = b->CreateStructGEP(t, m, 3u, "float4x4.d");
    auto col_type = _create_type(Type::of<float4>());
    b->CreateStore(b->CreateLoad(col_type, p0, "m.a"), m0);
    b->CreateStore(b->CreateLoad(col_type, p1, "m.b"), m1);
    b->CreateStore(b->CreateLoad(col_type, p2, "m.c"), m2);
    b->CreateStore(b->CreateLoad(col_type, p3, "m.d"), m3);
    return m;
}

luisa::string LLVMCodegen::_variable_name(Variable v) const noexcept {
    switch (v.tag()) {
        case Variable::Tag::LOCAL: return luisa::format("v{}.local", v.uid());
        case Variable::Tag::SHARED: return luisa::format("v{}.shared", v.uid());
        case Variable::Tag::REFERENCE: return luisa::format("v{}.ref", v.uid());
        case Variable::Tag::BUFFER: return luisa::format("v{}.buffer", v.uid());
        case Variable::Tag::TEXTURE: return luisa::format("v{}.texture", v.uid());
        case Variable::Tag::BINDLESS_ARRAY: return luisa::format("v{}.bindless", v.uid());
        case Variable::Tag::ACCEL: return luisa::format("v{}.accel", v.uid());
        case Variable::Tag::THREAD_ID: return "tid";
        case Variable::Tag::BLOCK_ID: return "bid";
        case Variable::Tag::DISPATCH_ID: return "did";
        case Variable::Tag::DISPATCH_SIZE: return "ls";
    }
    LUISA_ERROR_WITH_LOCATION("Invalid variable.");
}

}// namespace luisa::compute::llvm

#pragma clang diagnostic pop
