//
// Created by Mike Smith on 2022/5/23.
//

#include <backends/llvm/llvm_codegen.h>

namespace luisa::compute::llvm {

::llvm::Value *LLVMCodegen::_scalar_to_bool(const Type *src_type, ::llvm::Value *p_src) noexcept {
    LUISA_ASSERT(src_type->is_scalar(), "Invalid source type: {}.", src_type->description());
    auto builder = _current_context()->builder.get();
    switch (src_type->tag()) {
        case Type::Tag::BOOL: return p_src;
        case Type::Tag::FLOAT: return _create_stack_variable(
            builder->CreateFCmpONE(
                builder->CreateLoad(_create_type(src_type), p_src, "cast.float_to_bool.src"),
                _literal(0.f), "cast.float_to_bool.cmp"),
            "cast.float_to_bool.addr");
        case Type::Tag::INT:
        case Type::Tag::UINT: return _create_stack_variable(
            builder->CreateICmpNE(
                builder->CreateLoad(_create_type(src_type), p_src, "cast.int_to_bool.src"),
                _literal(0u), "cast.int_to_bool.cmp"),
            "cast.int_to_bool.addr");
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid conversion: {} to bool.",
                              src_type->description());
}

::llvm::Value *LLVMCodegen::_scalar_to_float(const Type *src_type, ::llvm::Value *p_src) noexcept {
    LUISA_ASSERT(src_type->is_scalar(), "Invalid source type: {}.", src_type->description());
    auto builder = _current_context()->builder.get();
    switch (src_type->tag()) {
        case Type::Tag::BOOL: return _create_stack_variable(
            builder->CreateSelect(
                builder->CreateLoad(_create_type(src_type), p_src, "cast.bool_to_float.src"),
                _literal(1.f), _literal(0.f), "cast.bool_to_float.select"),
            "cast.bool_to_float.addr");
        case Type::Tag::FLOAT: return p_src;
        case Type::Tag::INT: return _create_stack_variable(
            builder->CreateSIToFP(
                builder->CreateLoad(_create_type(src_type), p_src, "cast.int_to_float.src"),
                builder->getFloatTy(), "cast.int_to_float.cast"),
            "cast.int_to_float.addr");
        case Type::Tag::UINT: return _create_stack_variable(
            builder->CreateUIToFP(
                builder->CreateLoad(_create_type(src_type), p_src, "cast.uint_to_float.src"),
                builder->getFloatTy(), "cast.uint_to_float.cast"),
            "cast.uint_to_float.addr");
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid conversion: {} to float.",
                              src_type->description());
}

::llvm::Value *LLVMCodegen::_scalar_to_int(const Type *src_type, ::llvm::Value *p_src) noexcept {
    LUISA_ASSERT(src_type->is_scalar(), "Invalid source type: {}.", src_type->description());
    auto builder = _current_context()->builder.get();
    switch (src_type->tag()) {
        case Type::Tag::BOOL: return _create_stack_variable(
            builder->CreateZExt(
                builder->CreateLoad(_create_type(src_type), p_src, "cast.bool_to_int.src"),
                builder->getInt32Ty(), "cast.bool_to_int.cast"),
            "cast.bool_to_int.addr");
        case Type::Tag::FLOAT: return _create_stack_variable(
            builder->CreateFPToSI(
                builder->CreateLoad(_create_type(src_type), p_src, "cast.float_to_int.src"),
                builder->getInt32Ty(), "cast.float_to_int.cast"),
            "cast.float_to_int.addr");
        case Type::Tag::INT:
        case Type::Tag::UINT: return p_src;
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid conversion: {} to int.",
                              src_type->description());
}

::llvm::Value *LLVMCodegen::_scalar_to_uint(const Type *src_type, ::llvm::Value *p_src) noexcept {
    LUISA_ASSERT(src_type->is_scalar(), "Invalid source type: {}.", src_type->description());
    auto builder = _current_context()->builder.get();
    switch (src_type->tag()) {
        case Type::Tag::BOOL: return _create_stack_variable(
            builder->CreateZExt(
                builder->CreateLoad(_create_type(src_type), p_src, "cast.bool_to_uint.src"),
                builder->getInt32Ty(), "cast.bool_to_uint.cast"),
            "cast.bool_to_uint.addr");
        case Type::Tag::FLOAT: return _create_stack_variable(
            builder->CreateFPToUI(
                builder->CreateLoad(_create_type(src_type), p_src, "cast.float_to_uint.src"),
                builder->getInt32Ty(), "cast.float_to_uint.cast"),
            "cast.float_to_uint.addr");
        case Type::Tag::INT:
        case Type::Tag::UINT: return p_src;
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid conversion: {} to uint.",
                              src_type->description());
}

::llvm::Value *LLVMCodegen::_vector_to_bool_vector(const Type *src_type, ::llvm::Value *p_src) noexcept {
    LUISA_ASSERT(src_type->is_vector(), "Invalid source type: {}.", src_type->description());
    auto builder = _current_context()->builder.get();
    switch (src_type->element()->tag()) {
        case Type::Tag::BOOL: return p_src;
        case Type::Tag::FLOAT: {
            switch (src_type->dimension()) {
                case 2u: return _create_stack_variable(
                    builder->CreateFCmpONE(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.float2_to_bool2.src"),
                        _literal(make_float2(0.f)), "cast.float2_to_bool2.cmp"),
                    "cast.float2_to_bool2.addr");
                case 3u: return _create_stack_variable(
                    builder->CreateFCmpONE(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.float3_to_bool3.src"),
                        _literal(make_float3(0.f)), "cast.float3_to_bool3.cmp"),
                    "cast.float3_to_bool3.addr");
                case 4u: return _create_stack_variable(
                    builder->CreateFCmpONE(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.float4_to_bool4.src"),
                        _literal(make_float4(0.f)), "cast.float4_to_bool4.cmp"),
                    "cast.float4_to_bool4.addr");
                default: break;
            }
            break;
        }
        case Type::Tag::INT:
        case Type::Tag::UINT: {
            switch (src_type->dimension()) {
                case 2u: return _create_stack_variable(
                    builder->CreateICmpNE(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.int2_to_bool2.src"),
                        _literal(make_uint2(0u)), "cast.int2_to_bool2.cmp"),
                    "cast.int2_to_bool2.addr");
                case 3u: return _create_stack_variable(
                    builder->CreateICmpNE(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.int3_to_bool3.src"),
                        _literal(make_uint3(0u)), "cast.int3_to_bool3.cmp"),
                    "cast.int3_to_bool3.addr");
                case 4u: return _create_stack_variable(
                    builder->CreateICmpNE(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.int4_to_bool4.src"),
                        _literal(make_uint4(0u)), "cast.int4_to_bool4.cmp"),
                    "cast.int4_to_bool4.addr");
                default: break;
            }
            break;
        }
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid conversion: {} to bool vector.",
                              src_type->description());
}

::llvm::Value *LLVMCodegen::_vector_to_float_vector(const Type *src_type, ::llvm::Value *p_src) noexcept {
    LUISA_ASSERT(src_type->is_vector(), "Invalid source type: {}.", src_type->description());
    auto builder = _current_context()->builder.get();
    switch (src_type->element()->tag()) {
        case Type::Tag::BOOL: {
            switch (src_type->dimension()) {
                case 2u: return _create_stack_variable(
                    builder->CreateSelect(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.bool2_to_float2.src"),
                        _literal(make_float2(1.f)), _literal(make_float2(0.f)), "cast.bool2_to_float2.select"),
                    "cast.bool2_to_float2.addr");
                case 3u: return _create_stack_variable(
                    builder->CreateSelect(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.bool3_to_float3.src"),
                        _literal(make_float3(1.f)), _literal(make_float3(0.f)), "cast.bool3_to_float3.select"),
                    "cast.bool3_to_float3.addr");
                case 4u: return _create_stack_variable(
                    builder->CreateSelect(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.bool4_to_float4.src"),
                        _literal(make_float4(1.f)), _literal(make_float4(0.f)), "cast.bool4_to_float4.select"),
                    "cast.bool4_to_float4.addr");
                default: break;
            }
            break;
        }
        case Type::Tag::FLOAT: return p_src;
        case Type::Tag::INT: {
            switch (src_type->dimension()) {
                case 2u: return _create_stack_variable(
                    builder->CreateSIToFP(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.int2_to_float2.src"),
                        _create_type(Type::of<float2>()), "cast.int2_to_float2.cast"),
                    "cast.int2_to_float2.addr");
                case 3u: return _create_stack_variable(
                    builder->CreateSIToFP(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.int3_to_float3.src"),
                        _create_type(Type::of<float3>()), "cast.int3_to_float3.cast"),
                    "cast.int3_to_float3.addr");
                case 4u: return _create_stack_variable(
                    builder->CreateSIToFP(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.int4_to_float4.src"),
                        _create_type(Type::of<float4>()), "cast.int4_to_float4.cast"),
                    "cast.int4_to_float4.addr");
                default: break;
            }
            break;
        }
        case Type::Tag::UINT: {
            switch (src_type->dimension()) {
                case 2u: return _create_stack_variable(
                    builder->CreateUIToFP(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.uint2_to_float2.src"),
                        _create_type(Type::of<float2>()), "cast.uint2_to_float2.cast"),
                    "cast.uint2_to_float2.addr");
                case 3u: return _create_stack_variable(
                    builder->CreateUIToFP(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.uint3_to_float3.src"),
                        _create_type(Type::of<float3>()), "cast.uint3_to_float3.cast"),
                    "cast.uint3_to_float3.addr");
                case 4u: return _create_stack_variable(
                    builder->CreateUIToFP(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.uint4_to_float4.src"),
                        _create_type(Type::of<float4>()), "cast.uint4_to_float4.cast"),
                    "cast.uint4_to_float4.addr");
                default: break;
            }
            break;
        }
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid conversion: {} to float vector.",
                              src_type->description());
}

::llvm::Value *LLVMCodegen::_vector_to_int_vector(const Type *src_type, ::llvm::Value *p_src) noexcept {
    LUISA_ASSERT(src_type->is_vector(), "Invalid source type: {}.", src_type->description());
    auto builder = _current_context()->builder.get();
    switch (src_type->element()->tag()) {
        case Type::Tag::BOOL: {
            switch (src_type->dimension()) {
                case 2u: return _create_stack_variable(
                    builder->CreateZExt(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.bool2_to_int2.src"),
                        _create_type(Type::of<int2>()), "cast.bool2_to_int2.cast"),
                    "cast.bool2_to_int2.addr");
                case 3u: return _create_stack_variable(
                    builder->CreateZExt(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.bool3_to_int3.src"),
                        _create_type(Type::of<int3>()), "cast.bool3_to_int3.cast"),
                    "cast.bool3_to_int3.addr");
                case 4u: return _create_stack_variable(
                    builder->CreateZExt(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.bool4_to_int4.src"),
                        _create_type(Type::of<int4>()), "cast.bool4_to_int4.cast"),
                    "cast.bool4_to_int4.addr");
                default: break;
            }
            break;
        }
        case Type::Tag::FLOAT: {
            switch (src_type->dimension()) {
                case 2u: return _create_stack_variable(
                    builder->CreateFPToSI(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.float2_to_int2.src"),
                        _create_type(Type::of<int2>()), "cast.float2_to_int2.cast"),
                    "cast.float2_to_int2.addr");
                case 3u: return _create_stack_variable(
                    builder->CreateFPToSI(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.float3_to_int3.src"),
                        _create_type(Type::of<int3>()), "cast.float3_to_int3.cast"),
                    "cast.float3_to_int3.addr");
                case 4u: return _create_stack_variable(
                    builder->CreateFPToSI(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.float4_to_int4.src"),
                        _create_type(Type::of<int4>()), "cast.float4_to_int4.cast"),
                    "cast.float4_to_int4.addr");
                default: break;
            }
            break;
        }
        case Type::Tag::INT:
        case Type::Tag::UINT: return p_src;
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid conversion: {} to int vector.",
                              src_type->description());
}

::llvm::Value *LLVMCodegen::_vector_to_uint_vector(const Type *src_type, ::llvm::Value *p_src) noexcept {
    LUISA_ASSERT(src_type->is_vector(), "Invalid source type: {}.", src_type->description());
    auto builder = _current_context()->builder.get();
    switch (src_type->element()->tag()) {
        case Type::Tag::BOOL: {
            switch (src_type->dimension()) {
                case 2u: return _create_stack_variable(
                    builder->CreateZExt(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.bool2_to_uint2.src"),
                        _create_type(Type::of<uint2>()), "cast.bool2_to_int2.cast"),
                    "cast.bool2_to_uint2.addr");
                case 3u: return _create_stack_variable(
                    builder->CreateZExt(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.bool3_to_uint3.src"),
                        _create_type(Type::of<uint3>()), "cast.bool3_to_uint3.cast"),
                    "cast.bool3_to_uint3.addr");
                case 4u: return _create_stack_variable(
                    builder->CreateZExt(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.bool4_to_uint4.src"),
                        _create_type(Type::of<uint4>()), "cast.bool4_to_uint4.cast"),
                    "cast.bool4_to_uint4.addr");
                default: break;
            }
            break;
        }
        case Type::Tag::FLOAT: {
            switch (src_type->dimension()) {
                case 2u: return _create_stack_variable(
                    builder->CreateFPToUI(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.float2_to_uint2.src"),
                        _create_type(Type::of<uint2>()), "cast.float2_to_uint2.cast"),
                    "cast.float2_to_uint2.addr");
                case 3u: return _create_stack_variable(
                    builder->CreateFPToUI(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.float3_to_uint3.src"),
                        _create_type(Type::of<uint3>()), "cast.float3_to_uint3.cast"),
                    "cast.float3_to_uint3.addr");
                case 4u: return _create_stack_variable(
                    builder->CreateFPToUI(
                        builder->CreateLoad(_create_type(src_type), p_src, "cast.float4_to_uint4.src"),
                        _create_type(Type::of<uint4>()), "cast.float4_to_uint4.cast"),
                    "cast.float4_to_uint4.addr");
                default: break;
            }
            break;
        }
        case Type::Tag::INT:
        case Type::Tag::UINT: return p_src;
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid conversion: {} to uint vector.",
                              src_type->description());
}

::llvm::Value *LLVMCodegen::_scalar_to_vector(const Type *dst_type, const Type *src_type, ::llvm::Value *p_src) noexcept {
    LUISA_ASSERT(dst_type->is_vector(), "Invalid destination type: {}.", dst_type->description());
    LUISA_ASSERT(src_type->is_scalar(), "Invalid source type: {}.", src_type->description());
    if (*dst_type->element() != *src_type) {
        return _scalar_to_vector(dst_type, dst_type->element(),
                                 _builtin_static_cast(dst_type->element(), src_type, p_src));
    }
    auto builder = _current_context()->builder.get();
    auto src = builder->CreateLoad(_create_type(src_type), p_src, "cast.scalar_to_vector.src");
    auto dim = dst_type->dimension() == 3u ? 4u : dst_type->dimension();
    switch (src_type->tag()) {
        case Type::Tag::BOOL:
        case Type::Tag::FLOAT:
        case Type::Tag::INT:
        case Type::Tag::UINT: return _create_stack_variable(
            builder->CreateVectorSplat(dim, src, "cast.scalar_to_vector.splat"),
            "cast.scalar_to_vector.addr");
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid conversion: {} to vector.",
                              src_type->description());
}

::llvm::Value *LLVMCodegen::_vector_to_vector(const Type *dst_type, const Type *src_type, ::llvm::Value *p_src) noexcept {
    LUISA_ASSERT(src_type->is_vector(), "Invalid source type: {}.", src_type->description());
    LUISA_ASSERT(dst_type->is_vector(), "Invalid destination type: {}.", dst_type->description());
    auto builder = _current_context()->builder.get();
    if (dst_type->dimension() == 2u && src_type->dimension() > 2u) {
        auto src = builder->CreateLoad(_create_type(src_type), p_src, "cast.vector_to_vector.src");
        auto shuffle = builder->CreateShuffleVector(src, {0, 1}, "cast.vector_to_vector.shuffle");
        return _vector_to_vector(
            dst_type, Type::from(luisa::format("vector<{},2>", src_type->element()->description())),
            _create_stack_variable(shuffle, "cast.vector_to_vector.addr"));
    }
    switch (dst_type->element()->tag()) {
        case Type::Tag::BOOL: return _vector_to_bool_vector(src_type, p_src);
        case Type::Tag::FLOAT: return _vector_to_float_vector(src_type, p_src);
        case Type::Tag::INT: return _vector_to_int_vector(src_type, p_src);
        case Type::Tag::UINT: return _vector_to_uint_vector(src_type, p_src);
        default: break;
    }
    return nullptr;
}

::llvm::Value *LLVMCodegen::_scalar_to_matrix(const Type *dst_type, const Type *src_type, ::llvm::Value *p_src) noexcept {
    LUISA_ASSERT(dst_type->is_matrix(), "Invalid destination type: {}.", dst_type->description());
    LUISA_ASSERT(src_type->is_scalar(), "Invalid source type: {}.", src_type->description());
    auto p_src_cvt = _scalar_to_float(src_type, p_src);
    auto zero = _create_stack_variable(_literal(0.f), "zero");
    switch (dst_type->dimension()) {
        case 2u: return _make_float2x2(
            _make_float2(p_src_cvt, zero),
            _make_float2(zero, p_src_cvt));
        case 3u: return _make_float3x3(
            _make_float3(p_src_cvt, zero, zero),
            _make_float3(zero, p_src_cvt, zero),
            _make_float3(zero, zero, p_src_cvt));
        case 4u: return _make_float4x4(
            _make_float4(p_src_cvt, zero, zero, zero),
            _make_float4(zero, p_src_cvt, zero, zero),
            _make_float4(zero, zero, p_src_cvt, zero),
            _make_float4(zero, zero, zero, p_src_cvt));
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid conversion: {} to matrix.",
                              src_type->description());
}

::llvm::Value *LLVMCodegen::_matrix_to_matrix(const Type *dst_type, const Type *src_type, ::llvm::Value *p_src) noexcept {
    LUISA_ASSERT(src_type->is_matrix(), "Invalid source type: {}.", src_type->description());
    LUISA_ASSERT(dst_type->is_matrix(), "Invalid destination type: {}.", dst_type->description());
    LUISA_ASSERT(src_type->dimension() == dst_type->dimension(), "Invalid conversion from '{}' to '{}'.",
                 src_type->description(), dst_type->description());
    auto builder = _current_context()->builder.get();
    auto matrix_type = _create_type(dst_type);
    auto zero = _create_stack_variable(_literal(0.f), "zero");
    auto one = _create_stack_variable(_literal(1.f), "one");
    switch (src_type->dimension()) {
        case 2u: {
            if (dst_type->dimension() == 2u) { return p_src; }
            auto col_type = _create_type(Type::of<float2>());
            auto m0 = builder->CreateLoad(
                col_type, builder->CreateStructGEP(matrix_type, p_src, 0, "cast.matrix_to_matrix.m0.addr"),
                "cast.matrix_to_matrix.m0");
            auto m1 = builder->CreateLoad(
                col_type, builder->CreateStructGEP(matrix_type, p_src, 1, "cast.matrix_to_matrix.m1.addr"),
                "cast.matrix_to_matrix.m1");
            auto m00 = _create_stack_variable(builder->CreateExtractElement(m0, 0ull, "cast.matrix_to_matrix.m00"), "cast.matrix_to_matrix.m00.addr");
            auto m01 = _create_stack_variable(builder->CreateExtractElement(m0, 1ull, "cast.matrix_to_matrix.m01"), "cast.matrix_to_matrix.m01.addr");
            auto m10 = _create_stack_variable(builder->CreateExtractElement(m1, 0ull, "cast.matrix_to_matrix.m10"), "cast.matrix_to_matrix.m10.addr");
            auto m11 = _create_stack_variable(builder->CreateExtractElement(m1, 1ull, "cast.matrix_to_matrix.m11"), "cast.matrix_to_matrix.m11.addr");
            if (dst_type->dimension() == 3u) {
                return _make_float3x3(
                    _make_float3(m00, m01, zero),
                    _make_float3(m10, m11, zero),
                    _make_float3(zero, zero, one));
            }
            if (dst_type->dimension() == 4u) {
                return _make_float4x4(
                    _make_float4(m00, m01, zero, zero),
                    _make_float4(m10, m11, zero, zero),
                    _make_float4(zero, zero, one, zero),
                    _make_float4(zero, zero, zero, one));
            }
            break;
        }
        case 3u: {
            if (dst_type->dimension() == 3u) { return p_src; }
            auto col_type = _create_type(Type::of<float3>());
            auto m0 = builder->CreateLoad(
                col_type, builder->CreateStructGEP(matrix_type, p_src, 0, "cast.matrix_to_matrix.m0.addr"),
                "cast.matrix_to_matrix.m0");
            auto m1 = builder->CreateLoad(
                col_type, builder->CreateStructGEP(matrix_type, p_src, 1, "cast.matrix_to_matrix.m1.addr"),
                "cast.matrix_to_matrix.m1");
            auto m2 = builder->CreateLoad(
                col_type, builder->CreateStructGEP(matrix_type, p_src, 2, "cast.matrix_to_matrix.m2.addr"),
                "cast.matrix_to_matrix.m2");
            auto m00 = _create_stack_variable(builder->CreateExtractElement(m0, 0ull, "cast.matrix_to_matrix.m00"), "cast.matrix_to_matrix.m00.addr");
            auto m01 = _create_stack_variable(builder->CreateExtractElement(m0, 1ull, "cast.matrix_to_matrix.m01"), "cast.matrix_to_matrix.m01.addr");
            auto m02 = _create_stack_variable(builder->CreateExtractElement(m0, 2ull, "cast.matrix_to_matrix.m02"), "cast.matrix_to_matrix.m02.addr");
            auto m10 = _create_stack_variable(builder->CreateExtractElement(m1, 0ull, "cast.matrix_to_matrix.m10"), "cast.matrix_to_matrix.m10.addr");
            auto m11 = _create_stack_variable(builder->CreateExtractElement(m1, 1ull, "cast.matrix_to_matrix.m11"), "cast.matrix_to_matrix.m11.addr");
            auto m12 = _create_stack_variable(builder->CreateExtractElement(m1, 2ull, "cast.matrix_to_matrix.m12"), "cast.matrix_to_matrix.m12.addr");
            auto m20 = _create_stack_variable(builder->CreateExtractElement(m2, 0ull, "cast.matrix_to_matrix.m20"), "cast.matrix_to_matrix.m20.addr");
            auto m21 = _create_stack_variable(builder->CreateExtractElement(m2, 1ull, "cast.matrix_to_matrix.m21"), "cast.matrix_to_matrix.m21.addr");
            auto m22 = _create_stack_variable(builder->CreateExtractElement(m2, 2ull, "cast.matrix_to_matrix.m22"), "cast.matrix_to_matrix.m22.addr");
            if (dst_type->dimension() == 2u) {
                return _make_float2x2(
                    _make_float2(m00, m01),
                    _make_float2(m10, m11));
            }
            if (dst_type->dimension() == 4u) {
                return _make_float4x4(
                    _make_float4(m00, m01, m02, zero),
                    _make_float4(m10, m11, m12, zero),
                    _make_float4(m20, m21, m22, zero),
                    _make_float4(zero, zero, zero, one));
            }
            break;
        }
        case 4u: {
            if (dst_type->dimension() == 4u) { return p_src; }
            auto col_type = _create_type(Type::of<float4>());
            auto m0 = builder->CreateLoad(
                col_type, builder->CreateStructGEP(matrix_type, p_src, 0, "cast.matrix_to_matrix.m0.addr"),
                "cast.matrix_to_matrix.m0");
            auto m1 = builder->CreateLoad(
                col_type, builder->CreateStructGEP(matrix_type, p_src, 1, "cast.matrix_to_matrix.m1.addr"),
                "cast.matrix_to_matrix.m1");
            auto m2 = builder->CreateLoad(
                col_type, builder->CreateStructGEP(matrix_type, p_src, 2, "cast.matrix_to_matrix.m2.addr"),
                "cast.matrix_to_matrix.m2");
            auto m3 = builder->CreateLoad(
                col_type, builder->CreateStructGEP(matrix_type, p_src, 3, "cast.matrix_to_matrix.m3.addr"),
                "cast.matrix_to_matrix.m3");
            auto m00 = _create_stack_variable(builder->CreateExtractElement(m0, 0ull, "cast.matrix_to_matrix.m00"), "cast.matrix_to_matrix.m00.addr");
            auto m01 = _create_stack_variable(builder->CreateExtractElement(m0, 1ull, "cast.matrix_to_matrix.m01"), "cast.matrix_to_matrix.m01.addr");
            auto m02 = _create_stack_variable(builder->CreateExtractElement(m0, 2ull, "cast.matrix_to_matrix.m02"), "cast.matrix_to_matrix.m02.addr");
            auto m03 = _create_stack_variable(builder->CreateExtractElement(m0, 3ull, "cast.matrix_to_matrix.m03"), "cast.matrix_to_matrix.m03.addr");
            auto m10 = _create_stack_variable(builder->CreateExtractElement(m1, 0ull, "cast.matrix_to_matrix.m10"), "cast.matrix_to_matrix.m10.addr");
            auto m11 = _create_stack_variable(builder->CreateExtractElement(m1, 1ull, "cast.matrix_to_matrix.m11"), "cast.matrix_to_matrix.m11.addr");
            auto m12 = _create_stack_variable(builder->CreateExtractElement(m1, 2ull, "cast.matrix_to_matrix.m12"), "cast.matrix_to_matrix.m12.addr");
            auto m13 = _create_stack_variable(builder->CreateExtractElement(m1, 3ull, "cast.matrix_to_matrix.m13"), "cast.matrix_to_matrix.m13.addr");
            auto m20 = _create_stack_variable(builder->CreateExtractElement(m2, 0ull, "cast.matrix_to_matrix.m20"), "cast.matrix_to_matrix.m20.addr");
            auto m21 = _create_stack_variable(builder->CreateExtractElement(m2, 1ull, "cast.matrix_to_matrix.m21"), "cast.matrix_to_matrix.m21.addr");
            auto m22 = _create_stack_variable(builder->CreateExtractElement(m2, 2ull, "cast.matrix_to_matrix.m22"), "cast.matrix_to_matrix.m22.addr");
            auto m23 = _create_stack_variable(builder->CreateExtractElement(m2, 3ull, "cast.matrix_to_matrix.m23"), "cast.matrix_to_matrix.m23.addr");
            auto m30 = _create_stack_variable(builder->CreateExtractElement(m3, 0ull, "cast.matrix_to_matrix.m30"), "cast.matrix_to_matrix.m30.addr");
            auto m31 = _create_stack_variable(builder->CreateExtractElement(m3, 1ull, "cast.matrix_to_matrix.m31"), "cast.matrix_to_matrix.m31.addr");
            auto m32 = _create_stack_variable(builder->CreateExtractElement(m3, 2ull, "cast.matrix_to_matrix.m32"), "cast.matrix_to_matrix.m32.addr");
            auto m33 = _create_stack_variable(builder->CreateExtractElement(m3, 3ull, "cast.matrix_to_matrix.m33"), "cast.matrix_to_matrix.m33.addr");
            if (dst_type->dimension() == 2u) {
                return _make_float2x2(
                    _make_float2(m00, m01),
                    _make_float2(m10, m11));
            }
            if (dst_type->dimension() == 3u) {
                return _make_float3x3(
                    _make_float3(m00, m01, m02),
                    _make_float3(m10, m11, m12),
                    _make_float3(m20, m21, m22));
            }
            break;
        }
    }
    LUISA_ERROR_WITH_LOCATION("Invalid conversion: {} to matrix.",
                              src_type->description());
}

::llvm::Type *LLVMCodegen::_create_type(const Type *t) noexcept {
    if (t == nullptr) { return ::llvm::Type::getVoidTy(_context); }
    switch (t->tag()) {
        case Type::Tag::BOOL: return ::llvm::Type::getInt8Ty(_context);
        case Type::Tag::FLOAT: return ::llvm::Type::getFloatTy(_context);
        case Type::Tag::INT: [[fallthrough]];
        case Type::Tag::UINT: return ::llvm::Type::getInt32Ty(_context);
        case Type::Tag::VECTOR: return ::llvm::VectorType::get(
            _create_type(t->element()),
            t->dimension() == 3u ? 4u : t->dimension(), false);
        case Type::Tag::MATRIX: return ::llvm::ArrayType::get(
            _create_type(Type::from(luisa::format(
                "vector<{},{}>", t->element()->description(), t->dimension()))),
            t->dimension());
        case Type::Tag::ARRAY: return ::llvm::ArrayType::get(
            _create_type(t->element()), t->dimension());
        case Type::Tag::STRUCTURE: {
            if (auto iter = _struct_types.find(t->hash()); iter != _struct_types.end()) {
                return iter->second.type;
            }
            auto member_index = 0u;
            luisa::vector<::llvm::Type *> field_types;
            luisa::vector<uint> field_indices;
            auto size = 0ul;
            for (auto &member : t->members()) {
                auto aligned_offset = luisa::align(size, member->alignment());
                if (aligned_offset > size) {
                    auto padding = ::llvm::ArrayType::get(
                        ::llvm::Type::getInt8Ty(_context), aligned_offset - size);
                    field_types.emplace_back(padding);
                    member_index++;
                }
                field_types.emplace_back(_create_type(member));
                field_indices.emplace_back(member_index++);
                size = aligned_offset + member->size();
            }
            if (t->size() > size) {// last padding
                auto padding = ::llvm::ArrayType::get(
                    ::llvm::Type::getInt8Ty(_context), t->size() - size);
                field_types.emplace_back(padding);
            }
            ::llvm::ArrayRef<::llvm::Type *> fields_ref{field_types.data(), field_types.size()};
            auto struct_type = ::llvm::StructType::get(_context, fields_ref);
            _struct_types.emplace(t->hash(), LLVMStruct{struct_type, std::move(field_indices)});
            return struct_type;
        }
        case Type::Tag::BUFFER: return ::llvm::PointerType::get(_create_type(t->element()), 0);
        case Type::Tag::TEXTURE: return ::llvm::Type::getInt128Ty(_context);
        case Type::Tag::BINDLESS_ARRAY: /* TODO: implement */ break;
        case Type::Tag::ACCEL: return ::llvm::Type::getInt64Ty(_context);
    }
    LUISA_ERROR_WITH_LOCATION("Invalid type: {}.", t->description());
}

}// namespace luisa::compute::llvm
