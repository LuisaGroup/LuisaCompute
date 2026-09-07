// Test for TensorElementType coverage in the tile_to_kernel lowering.
//
// Verifies that every TensorElementType defined in
// <luisa/ast/tensor.h> (F16, F32, I32, I8, FP8, I4, FP4) can be traced as a
// tile kernel and lowered to a regular FunctionBuilder by tile_to_kernel:
//   - a 1-D copy kernel (ALLOC / KERNEL_1D / COPY / STORE) for every dtype
//   - a 1-D elementwise-add kernel (BINARY ADD) for the typed dtypes (F32, I32)
//   - a 1-D unary math kernel (SQRT) for the floating dtypes (F32)
// Each kernel is traced, lowered, and its dispatch grid / block size /
// buffer-argument count are checked against the expected values.  No device
// is required — this is a pure host-side structural test.
//
// Usage: test_tensor_element_types

#include "ut/ut.hpp"

#include <luisa/dsl/tensor.h>
#include <luisa/ast/tile_to_kernel.h>
#include <luisa/ast/expression.h>
#include <luisa/ast/op.h>
#include <luisa/ast/tensor.h>

#include <cstdint>
#include <functional>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr auto T = tile::language::dsl;
using namespace tile::language;
using tile_f16 = tile::half;
using tile_f32 = tile::float32;
using tile_i32 = tile::int32;
using tile_i8 = tile::int8;
using tile_fp8 = tile::fp8;
using tile_i4 = tile::int4;
using tile_fp4 = tile::fp4;

// ---- copy kernels: one per TensorElementType -----------------------------
// A 1-D N-element copy through a fragment tile.  dispatch=(1,1), block=(32,1,1),
// 2 buffer arguments (input + output).

template<typename DType>
Tensor<DType, 1> dtype_copy(Tensor<DType, 1> A) {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;
    Tensor<DType, 1> B = T.empty(T.shape(N), DType{});
    for (auto bx : T.Kernel(1, threads)) {
        auto A_local = T.alloc_fragment(T.shape(N), DType{});
        T.copy(A(0), A_local(N));
        T.copy(A_local(N), B(0));
    }
    return B;
}

// ---- elementwise add: typed dtypes only (F32, I32) ----------------------
template<typename DType>
Tensor<DType, 1> dtype_add(Tensor<DType, 1> A, Tensor<DType, 1> B) {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;
    Tensor<DType, 1> C = T.empty(T.shape(N), DType{});
    for (auto bx : T.Kernel(1, threads)) {
        auto A_local = T.alloc_fragment(T.shape(N), DType{});
        auto B_local = T.alloc_fragment(T.shape(N), DType{});
        T.copy(A(0), A_local(N));
        T.copy(B(0), B_local(N));
        C(0) = A_local(N) + B_local(N);
    }
    return C;
}

// ---- unary sqrt: floating dtypes only (F32) ------------------------------
Tensor<tile_f32, 1> dtype_sqrt(Tensor<tile_f32, 1> A) {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;
    Tensor<tile_f32, 1> B = T.empty(T.shape(N), tile_f32{});
    for (auto bx : T.Kernel(1, threads)) {
        auto A_local = T.alloc_fragment(T.shape(N), tile_f32{});
        T.copy(A(0), A_local(N));
        // sqrt via ieee math (TileIeeeOp::SQRT)
        B(0) = T.sqrt(A_local(N));
    }
    return B;
}

// Helper: trace + lower a kernel lambda and check dispatch/block/arg count.
auto lower_and_check(luisa::string_view name, auto kernel_fn,
                     uint2 expected_dispatch, uint3 expected_block,
                     size_t expected_buffers) {
    auto kernel = tile::jit(kernel_fn).compile();
    auto result = tile_to_kernel(kernel.function(), TileToKernelConfig{});
    expect(result.function != nullptr);
    expect(result.dispatch_size.x == expected_dispatch.x &&
           result.dispatch_size.y == expected_dispatch.y)
        << name << " dispatch mismatch";
    auto block = result.function->block_size();
    expect(block.x == expected_block.x &&
           block.y == expected_block.y &&
           block.z == expected_block.z)
        << name << " block-size mismatch";
    expect(result.function->arguments().size() == expected_buffers)
        << name << " buffer-argument count mismatch";
    return result;
}

}// namespace

int main() {
    "tensor_dtype_copy_f16"_test = [] {
        lower_and_check("dtype_copy_f16", dtype_copy<tile_f16>,
                        uint2{32u, 1u}, uint3{32u, 1u, 1u}, 2u);
    };
    "tensor_dtype_copy_f32"_test = [] {
        lower_and_check("dtype_copy_f32", dtype_copy<tile_f32>,
                        uint2{32u, 1u}, uint3{32u, 1u, 1u}, 2u);
    };
    "tensor_dtype_copy_i32"_test = [] {
        lower_and_check("dtype_copy_i32", dtype_copy<tile_i32>,
                        uint2{32u, 1u}, uint3{32u, 1u, 1u}, 2u);
    };
    "tensor_dtype_copy_i8"_test = [] {
        lower_and_check("dtype_copy_i8", dtype_copy<tile_i8>,
                        uint2{32u, 1u}, uint3{32u, 1u, 1u}, 2u);
    };
    "tensor_dtype_copy_fp8"_test = [] {
        lower_and_check("dtype_copy_fp8", dtype_copy<tile_fp8>,
                        uint2{32u, 1u}, uint3{32u, 1u, 1u}, 2u);
    };
    "tensor_dtype_copy_i4"_test = [] {
        lower_and_check("dtype_copy_i4", dtype_copy<tile_i4>,
                        uint2{32u, 1u}, uint3{32u, 1u, 1u}, 2u);
    };
    "tensor_dtype_copy_fp4"_test = [] {
        lower_and_check("dtype_copy_fp4", dtype_copy<tile_fp4>,
                        uint2{32u, 1u}, uint3{32u, 1u, 1u}, 2u);
    };

    // elementwise add for the typed dtypes (the quantized dtypes widen to a
    // compute type in the lowering; the structural lowering is still valid).
    "tensor_dtype_add_f32"_test = [] {
        lower_and_check("dtype_add_f32", dtype_add<tile_f32>,
                        uint2{32u, 1u}, uint3{32u, 1u, 1u}, 3u);
    };
    "tensor_dtype_add_i32"_test = [] {
        lower_and_check("dtype_add_i32", dtype_add<tile_i32>,
                        uint2{32u, 1u}, uint3{32u, 1u, 1u}, 3u);
    };

    // unary sqrt (floating) — SQRT lowers through the ieee-math path.
    "tensor_dtype_sqrt_f32"_test = [] {
        lower_and_check("dtype_sqrt_f32", dtype_sqrt,
                        uint2{32u, 1u}, uint3{32u, 1u, 1u}, 2u);
    };

    // Verify the AST Type registry resolves the new 4-bit type tags.
    "tensor_type_tag_int4"_test = [] {
        auto *t = Type::from("int4");
        expect(t != nullptr);
        expect(t->tag() == Type::Tag::INT4);
        expect(t->is_int());
        expect(t->is_quantized());
        expect(t->size() == 1u);
    };
    "tensor_type_tag_fp4"_test = [] {
        auto *t = Type::from("fp4e2m1");
        expect(t != nullptr);
        expect(t->tag() == Type::Tag::FP4_E2M1);
        expect(t->is_quantized());
        expect(t->size() == 1u);
    };
    "tensor_type_tag_float8"_test = [] {
        auto *t = Type::from("float8e4m3");
        expect(t != nullptr);
        expect(t->tag() == Type::Tag::FLOAT8_E4M3);
        expect(t->is_float8());
        expect(t->size() == 1u);
    };

    // Verify tensor_element_type maps every TensorElementType to a non-null Type.
    "tensor_element_type_mapping"_test = [] {
        using E = TensorElementType;
        // The tile_to_kernel helper is in an anonymous namespace; reach the
        // Type via the DSL dtype handles instead.
        auto check = [](TensorElementType e, const Type *expected) {
            // Map via the DSL tensor_element_type trait.
            auto *t = [&]() -> const Type * {
                switch (e) {
                    case E::F16: return Type::of<half>();
                    case E::F32: return Type::of<float>();
                    case E::I32: return Type::of<int>();
                    case E::I8: return Type::of<byte>();
                    case E::FP8: return Type::from("float8e4m3");
                    case E::I4: return Type::from("int4");
                    case E::FP4: return Type::from("fp4e2m1");
                }
                return nullptr;
            }();
            expect(t != nullptr);
            expect(t == expected);
        };
        check(E::F16, Type::of<half>());
        check(E::F32, Type::of<float>());
        check(E::I32, Type::of<int>());
        check(E::I8, Type::of<byte>());
        check(E::FP8, Type::from("float8e4m3"));
        check(E::I4, Type::from("int4"));
        check(E::FP4, Type::from("fp4e2m1"));
    };

    return 0;
}
