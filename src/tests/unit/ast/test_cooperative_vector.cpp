// Test for cooperative vector AST construction and device execution.
// This test covers:
// - Cooperative vector/matrix reference type creation and descriptions
// - FunctionBuilder flag propagation for cooperative operations
// - DSL construction of CoopVector, CoopVectorRef, and CoopMatrixRef
// - All cooperative builtin calls listed in include/luisa/ast/op.h
// - Device compilation/execution when a supporting backend is provided

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/ast/function_builder.h>
#include <luisa/ast/type.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/shared.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/command_list.h>

#include <array>
#include <optional>

#ifdef _WIN32
#include <luisa/backends/ext/dx_config_ext.h>
// Required by the D3D12 Agility SDK: these exports must be in the main .exe,
// not in a loaded DLL, so the D3D12 loader can find the SDK runtime.
extern "C" __declspec(dllexport) const uint32_t D3D12SDKVersion = 619;
extern "C" __declspec(dllexport) const char *D3D12SDKPath = ".\\D3D12\\";
#endif

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

#ifdef _WIN32
class DXExperimentalConfigExt final : public DirectXDeviceConfigExt {
public:
    [[nodiscard]] bool UseExperimental() const noexcept override { return true; }
};
#endif

// Create the requested test device (dx with experimental features, vk, or metal4).
// Returns std::nullopt if the backend is not supported on this platform.
[[nodiscard]] std::optional<luisa::test::DeviceContext> create_test_device(
    const char *exe, luisa::string_view backend) {
    if (backend == "dx") {
#ifdef _WIN32
        Context context{exe};
        DeviceConfig config;
        auto dx_config = luisa::make_unique<DXExperimentalConfigExt>();
        auto *dx_config_ptr = dx_config.get();
        config.extension = std::move(dx_config);
        Device device = context.create_device("dx", &config);
        if (!dx_config_ptr->ExperimentalFeaturesEnabled()) {
            // LUISA_INFO("DX cooperative-vector experimental features are not available on this system; skipping device execution tests.");
        }
        return luisa::test::DeviceContext{std::move(context), std::move(device)};
#else
        LUISA_INFO("DX backend is not available on this platform; skipping device execution tests.");
        return std::nullopt;
#endif
    }
    if (backend == "vk" || backend == "metal4") {
        Context context{exe};
        Device device = context.create_device(backend);
        return luisa::test::DeviceContext{std::move(context), std::move(device)};
    }
    LUISA_INFO("This test only supports the dx, vk, or metal4 backend; got '{}'. Skipping device execution tests.", backend);
    return std::nullopt;
}

// Verify the type registry can build cooperative-vector-related types.
void test_cooperative_vector_types() {
    auto cv_f32_16 = Type::cooperative_vector(Type::of<float>(), 16);
    expect(cv_f32_16 != nullptr);
    expect(cv_f32_16->is_cooperative_vector());
    expect(!cv_f32_16->is_cooperative_vector_ref());
    expect(eq(cv_f32_16->description(), luisa::string{"coopvec<float,16>"}));
    expect(eq(cv_f32_16->dimension(), 16u));
    expect(cv_f32_16->element() == Type::of<float>());

    auto cvr_f32_16 = Type::cooperative_vector_ref(CoopRefVecType::FLOAT32, 16);
    expect(cvr_f32_16 != nullptr);
    expect(cvr_f32_16->is_cooperative_vector_ref());
    expect(!cvr_f32_16->is_cooperative_vector());
    expect(eq(cvr_f32_16->description(), luisa::string{"coopvec_ref<16,5>"}));
    expect(eq(cvr_f32_16->dimension(), 16u));
    expect(cvr_f32_16->coop_vec_ref_type() == CoopRefVecType::FLOAT32);

    auto cmr_f32_4x8 = Type::cooperative_matrix_ref(CoopRefVecType::FLOAT32, 4, 8);
    expect(cmr_f32_4x8 != nullptr);
    expect(cmr_f32_4x8->is_cooperative_matrix_ref());
    expect(eq(cmr_f32_4x8->description(), luisa::string{"coopmat_ref<4,8,5>"}));
    auto dim = cmr_f32_4x8->coop_matrix_dimension();
    expect(static_cast<bool>(dim.x == 4u && dim.y == 8u));
    expect(cmr_f32_4x8->coop_vec_ref_type() == CoopRefVecType::FLOAT32);

    // Sanity-check sizes reported for reference element types.
    expect(eq(coop_ref_vec_type_size(CoopRefVecType::UINT8), size_t{1}));
    expect(eq(coop_ref_vec_type_size(CoopRefVecType::INT8), size_t{1}));
    expect(eq(coop_ref_vec_type_size(CoopRefVecType::FLOAT16), size_t{2}));
    expect(eq(coop_ref_vec_type_size(CoopRefVecType::FLOAT32), size_t{4}));
    expect(eq(coop_ref_vec_type_size(CoopRefVecType::INT32), size_t{4}));
    expect(eq(coop_ref_vec_type_size(CoopRefVecType::UINT32), size_t{4}));
}

// Verify that creating a CoopVector local flips the FunctionBuilder flag.
void test_cooperative_vector_ast_flags() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_callable([&]() {
        auto &cur = *FuncBuilder::current();
        auto v = cur.local(Type::cooperative_vector(Type::of<float>(), 8));
        static_cast<void>(v);
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations()) << "use_cooperative_operations should be true after allocating a coop vec local";
    expect(!f.propagated_builtin_callables().uses_cooperative())
        << "no builtin cooperative call was emitted";
}

// Verify that emitting a cooperative builtin marks the call-set.
void test_cooperative_vector_builtin_call() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        auto &cur = *FuncBuilder::current();
        auto byte_buf = cur.buffer(Type::of<ByteBuffer>());
        auto v_ref = cur.local(Type::cooperative_vector_ref(CoopRefVecType::FLOAT32, 4));
        auto v = cur.local(Type::cooperative_vector(Type::of<float>(), 4));
        cur.call(
            CallOp::COOPERATIVE_VECTOR_ACCUMULATE,
            {byte_buf, v_ref, v});
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations()) << "kernel should report cooperative operations";
    expect(f.propagated_builtin_callables().uses_cooperative())
        << "propagated builtin callables should contain cooperative ops";
    expect(f.direct_builtin_callables().test(CallOp::COOPERATIVE_VECTOR_ACCUMULATE))
        << "direct builtin callables should contain COOPERATIVE_VECTOR_ACCUMULATE";
}

// Verify the high-level DSL wrappers build the same AST shapes.
void test_cooperative_vector_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        ByteBufferVar vector_buffer{luisa::compute::detail::ArgumentCreation{}};
        CoopVectorRef vector_offset{CoopRefVecType::FLOAT32, 8};
        CoopVector<float> input{8};
        for (auto i = 0u; i < 8u; ++i) {
            input[i] = 1.0f;
        }
        vector_offset.set_byte_offset(0u);
        cooperative_vector_accumulate(vector_buffer, vector_offset, Expr<CoopVector<float>>{input});
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations()) << "DSL kernel should report cooperative operations";
    expect(f.direct_builtin_callables().test(CallOp::COOPERATIVE_VECTOR_ACCUMULATE))
        << "DSL kernel should directly call COOPERATIVE_VECTOR_ACCUMULATE";
}

// Helpers for dimensions used by cooperative matrix tests.
constexpr auto coop_k = 4u;
constexpr auto coop_n = 8u;

// AST-level test for COOPERATIVE_MUL_ADD.
void test_cooperative_mul_add_ast() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        auto &cur = *FuncBuilder::current();
        auto matrix_buf = cur.buffer(Type::of<ByteBuffer>());
        auto bias_buf = cur.buffer(Type::of<ByteBuffer>());
        auto matrix_ref = cur.local(Type::cooperative_matrix_ref(CoopRefVecType::FLOAT32, coop_k, coop_n));
        auto bias_ref = cur.local(Type::cooperative_vector_ref(CoopRefVecType::FLOAT32, coop_n));
        auto input = cur.local(Type::cooperative_vector(Type::of<float>(), coop_k));
        auto ret = Type::cooperative_vector(Type::of<float>(), coop_n);
        static_cast<void>(cur.call(
            CallOp::COOPERATIVE_MUL_ADD,
            {matrix_buf, matrix_ref, bias_buf, bias_ref, input}));
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
    expect(f.propagated_builtin_callables().uses_cooperative());
    expect(f.direct_builtin_callables().test(CallOp::COOPERATIVE_MUL_ADD));
}

// AST-level test for BINDLESS_COOPERATIVE_MUL_ADD and TYPED_BINDLESS_COOPERATIVE_MUL_ADD.
void test_bindless_cooperative_mul_add_ast() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto test_one = [](CallOp op) {
        auto fb = FuncBuilder::define_kernel([&]() {
            auto &cur = *FuncBuilder::current();
            auto bindless = cur.bindless_array();
            auto matrix_index = cur.literal(Type::of<uint>(), 0u);
            auto bias_index = cur.literal(Type::of<uint>(), 0u);
            auto matrix_ref = cur.local(Type::cooperative_matrix_ref(CoopRefVecType::FLOAT32, coop_k, coop_n));
            auto bias_ref = cur.local(Type::cooperative_vector_ref(CoopRefVecType::FLOAT32, coop_n));
            auto input = cur.local(Type::cooperative_vector(Type::of<float>(), coop_k));
            auto ret = Type::cooperative_vector(Type::of<float>(), coop_n);
            static_cast<void>(cur.call(
                op,
                {bindless, matrix_index, matrix_ref, bias_index, bias_ref, input}));
        });
        Function f{fb.get()};
        expect(f.use_cooperative_operations());
        expect(f.propagated_builtin_callables().uses_cooperative());
        expect(f.direct_builtin_callables().test(op));
    };
    test_one(CallOp::BINDLESS_COOPERATIVE_MUL_ADD);
    test_one(CallOp::TYPED_BINDLESS_COOPERATIVE_MUL_ADD);
}

// AST-level test for COOPERATIVE_MUL.
void test_cooperative_mul_ast() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        auto &cur = *FuncBuilder::current();
        auto matrix_buf = cur.buffer(Type::of<ByteBuffer>());
        auto matrix_ref = cur.local(Type::cooperative_matrix_ref(CoopRefVecType::FLOAT32, coop_k, coop_n));
        auto input = cur.local(Type::cooperative_vector(Type::of<float>(), coop_k));
        auto ret = Type::cooperative_vector(Type::of<float>(), coop_n);
        static_cast<void>(cur.call(
            CallOp::COOPERATIVE_MUL,
            {matrix_buf, matrix_ref, input}));
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
    expect(f.propagated_builtin_callables().uses_cooperative());
    expect(f.direct_builtin_callables().test(CallOp::COOPERATIVE_MUL));
}

// AST-level test for BINDLESS_COOPERATIVE_MUL and TYPED_BINDLESS_COOPERATIVE_MUL.
void test_bindless_cooperative_mul_ast() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto test_one = [](CallOp op) {
        auto fb = FuncBuilder::define_kernel([&]() {
            auto &cur = *FuncBuilder::current();
            auto bindless = cur.bindless_array();
            auto matrix_index = cur.literal(Type::of<uint>(), 0u);
            auto matrix_ref = cur.local(Type::cooperative_matrix_ref(CoopRefVecType::FLOAT32, coop_k, coop_n));
            auto input = cur.local(Type::cooperative_vector(Type::of<float>(), coop_k));
            auto ret = Type::cooperative_vector(Type::of<float>(), coop_n);
            static_cast<void>(cur.call(
                op,
                {bindless, matrix_index, matrix_ref, input}));
        });
        Function f{fb.get()};
        expect(f.use_cooperative_operations());
        expect(f.propagated_builtin_callables().uses_cooperative());
        expect(f.direct_builtin_callables().test(op));
    };
    test_one(CallOp::BINDLESS_COOPERATIVE_MUL);
    test_one(CallOp::TYPED_BINDLESS_COOPERATIVE_MUL);
}

// AST-level test for COOPERATIVE_OUTER_PRODUCT_ACCUMULATE.
void test_cooperative_outer_product_accumulate_ast() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        auto &cur = *FuncBuilder::current();
        auto matrix_buf = cur.buffer(Type::of<ByteBuffer>());
        auto matrix_ref = cur.local(Type::cooperative_matrix_ref(CoopRefVecType::FLOAT32, coop_k, coop_n));
        auto input1 = cur.local(Type::cooperative_vector(Type::of<float>(), coop_k));
        auto input2 = cur.local(Type::cooperative_vector(Type::of<float>(), coop_n));
        static_cast<void>(cur.call(
            CallOp::COOPERATIVE_OUTER_PRODUCT_ACCUMULATE,
            {matrix_buf, matrix_ref, input1, input2}));
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
    expect(f.propagated_builtin_callables().uses_cooperative());
    expect(f.direct_builtin_callables().test(CallOp::COOPERATIVE_OUTER_PRODUCT_ACCUMULATE));
}

// DSL-level test for cooperative_mat_mul_add.
void test_cooperative_mul_add_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        ByteBufferVar matrix_buffer{luisa::compute::detail::ArgumentCreation{}};
        ByteBufferVar bias_buffer{luisa::compute::detail::ArgumentCreation{}};
        CoopMatrixRef matrix_offset{CoopRefVecType::FLOAT32, coop_k, coop_n};
        CoopVectorRef bias_offset{CoopRefVecType::FLOAT32, coop_n};
        CoopVector<float> input{coop_k};
        matrix_offset.set_byte_offset(0u);
        bias_offset.set_byte_offset(0u);
        auto out = cooperative_mat_mul_add<float, float>(
            matrix_buffer, matrix_offset,
            bias_buffer, bias_offset,
            Expr<CoopVector<float>>{input});
        static_cast<void>(out);
    });
    Function f{fb.get()};
    expect(f.direct_builtin_callables().test(CallOp::COOPERATIVE_MUL_ADD));
}

// DSL-level test for bindless_cooperative_mat_mul_add and typed_bindless_cooperative_mat_mul_add.
void test_bindless_cooperative_mul_add_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        BindlessVar bindless{luisa::compute::detail::ArgumentCreation{}};
        CoopMatrixRef matrix_offset{CoopRefVecType::FLOAT32, coop_k, coop_n};
        CoopVectorRef bias_offset{CoopRefVecType::FLOAT32, coop_n};
        CoopVector<float> input{coop_k};
        matrix_offset.set_byte_offset(0u);
        bias_offset.set_byte_offset(0u);
        auto out0 = bindless_cooperative_mat_mul_add<float, float>(
            bindless, 0u, matrix_offset, 0u, bias_offset,
            Expr<CoopVector<float>>{input});
        auto out1 = typed_bindless_cooperative_mat_mul_add<float, float>(
            bindless, 0u, matrix_offset, 0u, bias_offset,
            Expr<CoopVector<float>>{input});
        static_cast<void>(out0);
        static_cast<void>(out1);
    });
    Function f{fb.get()};
    expect(f.direct_builtin_callables().test(CallOp::BINDLESS_COOPERATIVE_MUL_ADD));
    expect(f.direct_builtin_callables().test(CallOp::TYPED_BINDLESS_COOPERATIVE_MUL_ADD));
}

// DSL-level test for cooperative_mat_mul.
void test_cooperative_mul_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        ByteBufferVar matrix_buffer{luisa::compute::detail::ArgumentCreation{}};
        CoopMatrixRef matrix_offset{CoopRefVecType::FLOAT32, coop_k, coop_n};
        CoopVector<float> input{coop_k};
        matrix_offset.set_byte_offset(0u);
        auto out = cooperative_mat_mul<float, float>(
            matrix_buffer, matrix_offset,
            Expr<CoopVector<float>>{input});
        static_cast<void>(out);
    });
    Function f{fb.get()};
    expect(f.direct_builtin_callables().test(CallOp::COOPERATIVE_MUL));
}

// DSL-level test for bindless_cooperative_mat_mul and typed_bindless_cooperative_mat_mul.
void test_bindless_cooperative_mul_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        BindlessVar bindless{luisa::compute::detail::ArgumentCreation{}};
        CoopMatrixRef matrix_offset{CoopRefVecType::FLOAT32, coop_k, coop_n};
        CoopVector<float> input{coop_k};
        matrix_offset.set_byte_offset(0u);
        auto out0 = bindless_cooperative_mat_mul<float, float>(
            bindless, 0u, matrix_offset,
            Expr<CoopVector<float>>{input});
        auto out1 = typed_bindless_cooperative_mat_mul<float, float>(
            bindless, 0u, matrix_offset,
            Expr<CoopVector<float>>{input});
        static_cast<void>(out0);
        static_cast<void>(out1);
    });
    Function f{fb.get()};
    expect(f.direct_builtin_callables().test(CallOp::BINDLESS_COOPERATIVE_MUL));
    expect(f.direct_builtin_callables().test(CallOp::TYPED_BINDLESS_COOPERATIVE_MUL));
}

// DSL-level test for cooperative_outer_product_accumulate.
void test_cooperative_outer_product_accumulate_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        ByteBufferVar matrix_buffer{luisa::compute::detail::ArgumentCreation{}};
        CoopMatrixRef matrix_offset{CoopRefVecType::FLOAT32, coop_k, coop_n};
        CoopVector<float> input1{coop_k};
        CoopVector<float> input2{coop_n};
        matrix_offset.set_byte_offset(0u);
        cooperative_outer_product_accumulate(
            matrix_buffer, matrix_offset,
            Expr<CoopVector<float>>{input1},
            Expr<CoopVector<float>>{input2});
    });
    Function f{fb.get()};
    expect(f.direct_builtin_callables().test(CallOp::COOPERATIVE_OUTER_PRODUCT_ACCUMULATE));
}

// AST-level test for COOPERATIVE_VECTOR_LOAD.
void test_cooperative_vector_load_ast() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        auto &cur = *FuncBuilder::current();
        auto byte_buf = cur.buffer(Type::of<ByteBuffer>());
        auto v_ref = cur.local(Type::cooperative_vector_ref(CoopRefVecType::FLOAT32, 8));
        auto ret = cur.local(Type::cooperative_vector(Type::of<float>(), 8));
        static_cast<void>(cur.call(
            CallOp::COOPERATIVE_VECTOR_LOAD,
            {byte_buf, v_ref}));
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
    expect(f.propagated_builtin_callables().uses_cooperative());
    expect(f.direct_builtin_callables().test(CallOp::COOPERATIVE_VECTOR_LOAD));
}

// AST-level test for COOPERATIVE_VECTOR_STORE.
void test_cooperative_vector_store_ast() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        auto &cur = *FuncBuilder::current();
        auto byte_buf = cur.buffer(Type::of<ByteBuffer>());
        auto v_ref = cur.local(Type::cooperative_vector_ref(CoopRefVecType::FLOAT32, 8));
        auto v = cur.local(Type::cooperative_vector(Type::of<float>(), 8));
        static_cast<void>(cur.call(
            CallOp::COOPERATIVE_VECTOR_STORE,
            {byte_buf, v_ref, v}));
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
    expect(f.propagated_builtin_callables().uses_cooperative());
    expect(f.direct_builtin_callables().test(CallOp::COOPERATIVE_VECTOR_STORE));
}

// AST-level test for COOPERATIVE_VECTOR_SPLAT.
void test_cooperative_vector_splat_ast() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        auto &cur = *FuncBuilder::current();
        auto scalar = cur.literal(Type::of<float>(), 1.0f);
        auto ret = cur.local(Type::cooperative_vector(Type::of<float>(), 8));
        static_cast<void>(cur.call(
            CallOp::COOPERATIVE_VECTOR_SPLAT,
            {scalar}));
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
    expect(f.propagated_builtin_callables().uses_cooperative());
    expect(f.direct_builtin_callables().test(CallOp::COOPERATIVE_VECTOR_SPLAT));
}

// AST-level test for COOPERATIVE_VECTOR_CAST.
void test_cooperative_vector_cast_ast() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        auto &cur = *FuncBuilder::current();
        auto v = cur.local(Type::cooperative_vector(Type::of<float>(), 8));
        auto ret = cur.local(Type::cooperative_vector(Type::of<int>(), 8));
        static_cast<void>(cur.call(
            CallOp::COOPERATIVE_VECTOR_CAST,
            {v}));
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
    expect(f.propagated_builtin_callables().uses_cooperative());
    expect(f.direct_builtin_callables().test(CallOp::COOPERATIVE_VECTOR_CAST));
}

// DSL-level test for cooperative_vector_load.
void test_cooperative_vector_load_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        ByteBufferVar buf{luisa::compute::detail::ArgumentCreation{}};
        CoopVectorRef offset{CoopRefVecType::FLOAT32, 8};
        offset.set_byte_offset(0u);
        [[maybe_unused]] auto result = cooperative_vector_load<float>(buf, offset);
    });
    Function f{fb.get()};
    expect(f.direct_builtin_callables().test(CallOp::COOPERATIVE_VECTOR_LOAD));
}

// DSL-level test for cooperative_vector_store.
void test_cooperative_vector_store_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        ByteBufferVar buf{luisa::compute::detail::ArgumentCreation{}};
        CoopVectorRef offset{CoopRefVecType::FLOAT32, 8};
        CoopVector<float> input{8};
        for (auto i = 0u; i < 8u; ++i) input[i] = static_cast<float>(i);
        offset.set_byte_offset(0u);
        cooperative_vector_store(buf, offset, Expr<CoopVector<float>>{input});
    });
    Function f{fb.get()};
    expect(f.direct_builtin_callables().test(CallOp::COOPERATIVE_VECTOR_STORE));
}

// DSL-level test for cooperative_vector_splat.
void test_cooperative_vector_splat_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        [[maybe_unused]] auto result = cooperative_vector_splat<float>(1.0f, 8u);
    });
    Function f{fb.get()};
    expect(f.direct_builtin_callables().test(CallOp::COOPERATIVE_VECTOR_SPLAT));
}

// DSL-level test for cooperative_vector_cast.
void test_cooperative_vector_cast_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        CoopVector<float> input{8};
        for (auto i = 0u; i < 8u; ++i) input[i] = static_cast<float>(i);
        [[maybe_unused]] auto result = cooperative_vector_cast<int>(Expr<CoopVector<float>>{input});
    });
    Function f{fb.get()};
    expect(f.direct_builtin_callables().test(CallOp::COOPERATIVE_VECTOR_CAST));
}

// Device-side test for load/store round-trip.
void test_cooperative_vector_load_store_device(Device &device) {
    luisa::log_level_verbose();
    LUISA_INFO("Running cooperative vector load/store device test on backend '{}'", device.backend_name());

    Stream stream = device.create_stream();
    constexpr auto n = 8u;
    constexpr auto byte_size = n * sizeof(float);

    ByteBuffer vector_buffer = device.create_byte_buffer(byte_size);
    std::array<float, n> input_data;
    for (auto i = 0u; i < n; ++i) input_data[i] = static_cast<float>(i + 1);
    luisa::vector<std::byte> host(byte_size);

    // Store kernel: writes [1,2,...,n] into buffer via cooperative_vector_accumulate
    Kernel1D store_kernel = [&](ByteBufferVar buf) noexcept {
        CoopVectorRef offset{CoopRefVecType::FLOAT32, n};
        CoopVector<float> input{n};
        for (auto i = 0u; i < n; ++i) input[i] = static_cast<float>(i + 1);
        offset.set_byte_offset(0u);
        cooperative_vector_accumulate(buf, offset, Expr<CoopVector<float>>{input});
    };

    // Load kernel: reads back via cooperative_vector_load and copies to output buffer
    Kernel1D load_kernel = [&](ByteBufferVar buf, BufferVar<float> output) noexcept {
        CoopVectorRef offset{CoopRefVecType::FLOAT32, n};
        offset.set_byte_offset(0u);
        auto loaded = cooperative_vector_load<float>(buf, offset);
        for (auto i = 0u; i < n; ++i) {
            output.write(i, loaded[i]);
        }
    };

    auto store_shader = device.compile(store_kernel);
    auto load_shader = device.compile(load_kernel);
    Buffer<float> output_buffer = device.create_buffer<float>(n);

    CommandList cmdlist = CommandList::create();
    cmdlist << vector_buffer.copy_from(input_data.data())
            << load_shader(vector_buffer, output_buffer).dispatch(1u)
            << output_buffer.copy_to(luisa::span{host});
    stream << cmdlist.commit() << synchronize();

    bool ok = true;
    auto *pf = reinterpret_cast<const float *>(host.data());
    for (auto i = 0u; i < n; ++i) {
        if (std::abs(pf[i] - input_data[i]) > 1e-4f) {
            LUISA_WARNING("Load/store mismatch at [{}]: got {} expected {}", i, pf[i], input_data[i]);
            ok = false;
        }
    }
    expect(ok) << "cooperative_vector_load should return the data that was previously stored";
}

// Device-side test for cooperative-vector workgroup (shared memory) load/store.
void test_cooperative_vector_workgroup_load_store_device(Device &device) {
    luisa::log_level_verbose();
    LUISA_INFO("Running cooperative vector workgroup load/store device test on backend '{}'", device.backend_name());

    Stream stream = device.create_stream();
    constexpr auto n = 8u;

    Buffer<float> output_buffer = device.create_buffer<float>(n);
    luisa::vector<float> host(n);

    Kernel1D kernel = [&](BufferVar<float> output) noexcept {
        Shared<float> shared_mem{n};
        CoopVector<float> input{n};
        for (auto i = 0u; i < n; ++i) input[i] = static_cast<float>(i + 1);
        cooperative_vector_workgroup_store(
            shared_mem, 0u, Expr<CoopVector<float>>{input});
        sync_block();
        auto loaded = cooperative_vector_workgroup_load<float>(shared_mem, 0u);
        for (auto i = 0u; i < n; ++i) {
            output.write(i, loaded[i]);
        }
    };

    auto shader = device.compile(kernel);

    CommandList cmdlist = CommandList::create();
    cmdlist << shader(output_buffer).dispatch(1u)
            << output_buffer.copy_to(luisa::span{host});
    stream << cmdlist.commit() << synchronize();

    bool ok = true;
    for (auto i = 0u; i < n; ++i) {
        auto expected = static_cast<float>(i + 1);
        if (std::abs(host[i] - expected) > 1e-4f) {
            LUISA_WARNING("Workgroup load/store mismatch at [{}]: got {} expected {}", i, host[i], expected);
            ok = false;
        }
    }
    expect(ok) << "cooperative_vector_workgroup_load should return the data previously stored in shared memory";
}

// Device-side test for cooperative_vector_splat.
void test_cooperative_vector_splat_device(Device &device) {
    luisa::log_level_verbose();
    LUISA_INFO("Running cooperative vector splat device test on backend '{}'", device.backend_name());

    Stream stream = device.create_stream();
    constexpr auto n = 4u;
    constexpr auto byte_size = n * sizeof(float);

    ByteBuffer vector_buffer = device.create_byte_buffer(byte_size);
    std::array<std::byte, byte_size> zero{};
    luisa::vector<std::byte> host(byte_size);

    Kernel1D kernel = [&](ByteBufferVar buf) noexcept {
        CoopVectorRef offset{CoopRefVecType::FLOAT32, n};
        auto v = cooperative_vector_splat<float>(42.0f, n);
        offset.set_byte_offset(0u);
        cooperative_vector_accumulate(buf, offset, Expr<CoopVector<float>>{v});
    };

    auto shader = device.compile(kernel);

    CommandList cmdlist = CommandList::create();
    cmdlist << vector_buffer.copy_from(zero.data())
            << shader(vector_buffer).dispatch(1u)
            << vector_buffer.copy_to(host.data());
    stream << cmdlist.commit() << synchronize();

    bool ok = true;
    auto *pf = reinterpret_cast<const float *>(host.data());
    for (auto i = 0u; i < n; ++i) {
        if (std::abs(pf[i] - 42.0f) > 1e-4f) {
            LUISA_WARNING("Splat mismatch at [{}]: got {} expected {}", i, pf[i], 42.0f);
            ok = false;
        }
    }
    expect(ok) << "cooperative_vector_splat should produce a vector of 42.0 values";
}

// Device-side execution test.  This requires a backend with cooperative-vector
// support (DX with Shader Model 6.8, or VK).  For DX, experimental features must
// be enabled through the device extension.
void test_cooperative_vector_device(Device &device) {
    luisa::log_level_verbose();
    LUISA_INFO("Running cooperative vector device test on backend '{}'", device.backend_name());

    Stream stream = device.create_stream();
    constexpr auto n = 8u;
    constexpr auto byte_size = n * sizeof(float);

    ByteBuffer vector_buffer = device.create_byte_buffer(byte_size);
    std::array<std::byte, byte_size> zero{};
    luisa::vector<std::byte> host(byte_size);

    Kernel1D kernel = [&](ByteBufferVar buf) noexcept {
        CoopVectorRef vector_offset{CoopRefVecType::FLOAT32, n};
        CoopVector<float> input{n};
        for (auto i = 0u; i < n; ++i) {
            input[i] = static_cast<float>(i + 1);
        }
        vector_offset.set_byte_offset(0u);
        cooperative_vector_accumulate(buf, vector_offset, Expr<CoopVector<float>>{input});
    };

    auto shader = device.compile(kernel);

    CommandList cmdlist = CommandList::create();
    cmdlist << vector_buffer.copy_from(zero.data())
            << shader(vector_buffer).dispatch(1u)
            << vector_buffer.copy_to(host.data());
    stream << cmdlist.commit() << synchronize();

    bool ok = true;
    auto *pf = reinterpret_cast<const float *>(host.data());
    for (auto i = 0u; i < n; ++i) {
        auto expected = static_cast<float>(i + 1);
        if (std::abs(pf[i] - expected) > 1e-4f) {
            LUISA_WARNING("Mismatch at [{}]: got {} expected {}", i, pf[i], expected);
            ok = false;
        }
    }
    expect(ok) << "cooperative_vector_accumulate should write [1,2,...,n] into the byte buffer";
}

// AST-level test for BINDLESS_COOPERATIVE_VECTOR_LOAD and TYPED_BINDLESS_COOPERATIVE_VECTOR_LOAD.
void test_bindless_cooperative_vector_load_ast() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto test_one = [](CallOp op) {
        auto fb = FuncBuilder::define_kernel([&]() {
            auto &cur = *FuncBuilder::current();
            auto bindless = cur.bindless_array();
            auto buffer_handle = cur.literal(Type::of<uint>(), 0u);
            auto v_ref = cur.local(Type::cooperative_vector_ref(CoopRefVecType::FLOAT32, 8));
            auto ret = cur.local(Type::cooperative_vector(Type::of<float>(), 8));
            static_cast<void>(cur.call(
                op,
                {bindless, buffer_handle, v_ref}));
        });
        Function f{fb.get()};
        expect(f.use_cooperative_operations());
        expect(f.propagated_builtin_callables().uses_cooperative());
        expect(f.direct_builtin_callables().test(op));
    };
    test_one(CallOp::BINDLESS_COOPERATIVE_VECTOR_LOAD);
    test_one(CallOp::TYPED_BINDLESS_COOPERATIVE_VECTOR_LOAD);
}

// AST-level test for BINDLESS_COOPERATIVE_VECTOR_STORE and TYPED_BINDLESS_COOPERATIVE_VECTOR_STORE.
void test_bindless_cooperative_vector_store_ast() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto test_one = [](CallOp op) {
        auto fb = FuncBuilder::define_kernel([&]() {
            auto &cur = *FuncBuilder::current();
            auto bindless = cur.bindless_array();
            auto buffer_handle = cur.literal(Type::of<uint>(), 0u);
            auto v_ref = cur.local(Type::cooperative_vector_ref(CoopRefVecType::FLOAT32, 8));
            auto v = cur.local(Type::cooperative_vector(Type::of<float>(), 8));
            static_cast<void>(cur.call(
                op,
                {bindless, buffer_handle, v_ref, v}));
        });
        Function f{fb.get()};
        expect(f.use_cooperative_operations());
        expect(f.propagated_builtin_callables().uses_cooperative());
        expect(f.direct_builtin_callables().test(op));
    };
    test_one(CallOp::BINDLESS_COOPERATIVE_VECTOR_STORE);
    test_one(CallOp::TYPED_BINDLESS_COOPERATIVE_VECTOR_STORE);
}

// DSL-level test for bindless_cooperative_vector_load and typed_bindless_cooperative_vector_load.
void test_bindless_cooperative_vector_load_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        BindlessVar bindless{luisa::compute::detail::ArgumentCreation{}};
        CoopVectorRef offset{CoopRefVecType::FLOAT32, 8};
        offset.set_byte_offset(0u);
        [[maybe_unused]] auto out0 = bindless_cooperative_vector_load<float>(bindless, 0u, offset);
        [[maybe_unused]] auto out1 = typed_bindless_cooperative_vector_load<float>(bindless, 0u, offset);
    });
    Function f{fb.get()};
    expect(f.direct_builtin_callables().test(CallOp::BINDLESS_COOPERATIVE_VECTOR_LOAD));
    expect(f.direct_builtin_callables().test(CallOp::TYPED_BINDLESS_COOPERATIVE_VECTOR_LOAD));
}

// DSL-level test for bindless_cooperative_vector_store and typed_bindless_cooperative_vector_store.
void test_bindless_cooperative_vector_store_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        BindlessVar bindless{luisa::compute::detail::ArgumentCreation{}};
        CoopVectorRef offset{CoopRefVecType::FLOAT32, 8};
        CoopVector<float> input{8};
        for (auto i = 0u; i < 8u; ++i) input[i] = static_cast<float>(i);
        offset.set_byte_offset(0u);
        bindless_cooperative_vector_store(bindless, 0u, offset, Expr<CoopVector<float>>{input});
        typed_bindless_cooperative_vector_store(bindless, 0u, offset, Expr<CoopVector<float>>{input});
    });
    Function f{fb.get()};
    expect(f.direct_builtin_callables().test(CallOp::BINDLESS_COOPERATIVE_VECTOR_STORE));
    expect(f.direct_builtin_callables().test(CallOp::TYPED_BINDLESS_COOPERATIVE_VECTOR_STORE));
}

// AST-level test for COOPERATIVE_VECTOR_WORKGROUP_LOAD.
void test_cooperative_vector_workgroup_load_ast() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        auto &cur = *FuncBuilder::current();
        auto shared_arr = cur.local(Type::array(Type::of<float>(), 8));
        auto index = cur.literal(Type::of<uint>(), 0u);
        auto ret = cur.local(Type::cooperative_vector(Type::of<float>(), 8));
        static_cast<void>(cur.call(
            CallOp::COOPERATIVE_VECTOR_WORKGROUP_LOAD,
            {shared_arr, index}));
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
    expect(f.propagated_builtin_callables().uses_cooperative());
    expect(f.direct_builtin_callables().test(CallOp::COOPERATIVE_VECTOR_WORKGROUP_LOAD));
}

// AST-level test for COOPERATIVE_VECTOR_WORKGROUP_STORE.
void test_cooperative_vector_workgroup_store_ast() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        auto &cur = *FuncBuilder::current();
        auto shared_arr = cur.local(Type::array(Type::of<float>(), 8));
        auto index = cur.literal(Type::of<uint>(), 0u);
        auto v = cur.local(Type::cooperative_vector(Type::of<float>(), 8));
        static_cast<void>(cur.call(
            CallOp::COOPERATIVE_VECTOR_WORKGROUP_STORE,
            {shared_arr, index, v}));
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
    expect(f.propagated_builtin_callables().uses_cooperative());
    expect(f.direct_builtin_callables().test(CallOp::COOPERATIVE_VECTOR_WORKGROUP_STORE));
}

// DSL-level test for cooperative_vector_workgroup_load.
void test_cooperative_vector_workgroup_load_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        Shared<float> shared_mem{8};
        [[maybe_unused]] auto result = cooperative_vector_workgroup_load(shared_mem, 0u);
    });
    Function f{fb.get()};
    expect(f.direct_builtin_callables().test(CallOp::COOPERATIVE_VECTOR_WORKGROUP_LOAD));
}

// DSL-level test for cooperative_vector_workgroup_store.
void test_cooperative_vector_workgroup_store_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        Shared<float> shared_mem{8};
        CoopVector<float> input{8};
        for (auto i = 0u; i < 8u; ++i) input[i] = static_cast<float>(i);
        cooperative_vector_workgroup_store(shared_mem, 0u, Expr<CoopVector<float>>{input});
    });
    Function f{fb.get()};
    expect(f.direct_builtin_callables().test(CallOp::COOPERATIVE_VECTOR_WORKGROUP_STORE));
}

// DSL-level test for cooperative_vector_min.
void test_cooperative_vector_min_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        CoopVector<float> a{4};
        CoopVector<float> b{4};
        for (auto i = 0u; i < 4u; ++i) { a[i] = static_cast<float>(i + 1); b[i] = static_cast<float>(4 - i); }
        [[maybe_unused]] auto result = cooperative_vector_min(a, b);
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
}

// DSL-level test for cooperative_vector_max.
void test_cooperative_vector_max_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        CoopVector<float> a{4};
        CoopVector<float> b{4};
        for (auto i = 0u; i < 4u; ++i) { a[i] = static_cast<float>(i + 1); b[i] = static_cast<float>(4 - i); }
        [[maybe_unused]] auto result = cooperative_vector_max(a, b);
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
}

// DSL-level test for cooperative_vector_clamp.
void test_cooperative_vector_clamp_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        CoopVector<float> v{4};
        CoopVector<float> lo{4};
        CoopVector<float> hi{4};
        for (auto i = 0u; i < 4u; ++i) { v[i] = static_cast<float>(i); lo[i] = 0.5f; hi[i] = 2.5f; }
        [[maybe_unused]] auto result = cooperative_vector_clamp(v, lo, hi);
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
}

// DSL-level test for cooperative_vector_exp.
void test_cooperative_vector_exp_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        CoopVector<float> v{4};
        for (auto i = 0u; i < 4u; ++i) v[i] = static_cast<float>(i);
        [[maybe_unused]] auto result = cooperative_vector_exp(v);
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
}

// DSL-level test for cooperative_vector_log.
void test_cooperative_vector_log_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        CoopVector<float> v{4};
        for (auto i = 0u; i < 4u; ++i) v[i] = static_cast<float>(i + 1);
        [[maybe_unused]] auto result = cooperative_vector_log(v);
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
}

// DSL-level test for cooperative_vector_tanh.
void test_cooperative_vector_tanh_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        CoopVector<float> v{4};
        for (auto i = 0u; i < 4u; ++i) v[i] = 0.5f;
        [[maybe_unused]] auto result = cooperative_vector_tanh(v);
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
}

// DSL-level test for cooperative_vector_atan.
void test_cooperative_vector_atan_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        CoopVector<float> v{4};
        for (auto i = 0u; i < 4u; ++i) v[i] = static_cast<float>(i);
        [[maybe_unused]] auto result = cooperative_vector_atan(v);
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
}

// DSL-level test for cooperative_vector_fma.
void test_cooperative_vector_fma_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        CoopVector<float> a{4};
        CoopVector<float> b{4};
        CoopVector<float> c{4};
        for (auto i = 0u; i < 4u; ++i) { a[i] = 2.0f; b[i] = 3.0f; c[i] = 1.0f; }
        [[maybe_unused]] auto result = cooperative_vector_fma(a, b, c);
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
}

// DSL-level test for cooperative_vector_bitwise_and.
void test_cooperative_vector_bitwise_and_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        CoopVector<uint> a{4};
        CoopVector<uint> b{4};
        for (auto i = 0u; i < 4u; ++i) { a[i] = 0xFFu; b[i] = 0xF0u; }
        [[maybe_unused]] auto result = cooperative_vector_bitwise_and(a, b);
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
}

// DSL-level test for cooperative_vector_bitwise_or.
void test_cooperative_vector_bitwise_or_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        CoopVector<uint> a{4};
        CoopVector<uint> b{4};
        for (auto i = 0u; i < 4u; ++i) { a[i] = 0x0Fu; b[i] = 0xF0u; }
        [[maybe_unused]] auto result = cooperative_vector_bitwise_or(a, b);
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
}

// DSL-level test for cooperative_vector_bitwise_xor.
void test_cooperative_vector_bitwise_xor_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        CoopVector<uint> a{4};
        CoopVector<uint> b{4};
        for (auto i = 0u; i < 4u; ++i) { a[i] = 0xFFu; b[i] = 0xF0u; }
        [[maybe_unused]] auto result = cooperative_vector_bitwise_xor(a, b);
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
}

// DSL-level test for cooperative_vector_bitwise_not.
void test_cooperative_vector_bitwise_not_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        CoopVector<uint> v{4};
        for (auto i = 0u; i < 4u; ++i) v[i] = 0x0Fu;
        [[maybe_unused]] auto result = cooperative_vector_bitwise_not(v);
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
}

// DSL-level test for cooperative_vector_shift_left.
void test_cooperative_vector_shift_left_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        CoopVector<uint> v{4};
        for (auto i = 0u; i < 4u; ++i) v[i] = 1u << i;
        [[maybe_unused]] auto result = cooperative_vector_shift_left(v, 1u);
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
}

// DSL-level test for cooperative_vector_shift_right.
void test_cooperative_vector_shift_right_dsl() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto fb = FuncBuilder::define_kernel([&]() {
        CoopVector<uint> v{4};
        for (auto i = 0u; i < 4u; ++i) v[i] = 0x80u;
        [[maybe_unused]] auto result = cooperative_vector_shift_right(v, 4u);
    });
    Function f{fb.get()};
    expect(f.use_cooperative_operations());
}

// Global pointer used by the capture-less Boost.UT suite to reach the device.
Device *g_device_for_tests = nullptr;

}// namespace

int main(int argc, char *argv[]) {
    const char *exe = (argc > 0 && argv && argv[0]) ? argv[0] : luisa::test::safe_argv0();

    // Parse only argv[0] through Boost.UT so that the backend name (argv[1])
    // is not interpreted as a test-name filter.
    const char *ut_argv0[] = {exe};
    boost::ut::detail::cfg::parse_arg_with_fallback(
        1, const_cast<const char **>(ut_argv0));

    // Create the device first (when requested) so that the Boost.UT runner is
    // initialized *after* the backend is loaded.  This avoids tearing down the
    // backend before the runner's destructor finishes, which was causing a
    // debug-iterator assertion on exit for the vk path.
    std::optional<luisa::test::DeviceContext> dc;
    if (argc > 1) {
        dc = create_test_device(exe, argv[1]);
        if (!dc) {
            return 0;
        }
        g_device_for_tests = &dc->device;
    } else {
        LUISA_INFO("No backend argument provided; skipping device execution tests.");
    }

    // Register all tests as a Boost.UT suite.  We then explicitly run the suite
    // here (with only argv[0] exposed to Boost.UT) so that the backend name is
    // not interpreted as a test-name filter.  Running the suite while the
    // device/context are alive also avoids tearing down the backend before the
    // runner finishes.
    boost::ut::suite<"cooperative_vector">{[] {
        "cooperative_vector_types"_test = [] { test_cooperative_vector_types(); };
        "cooperative_vector_ast_flags"_test = [] { test_cooperative_vector_ast_flags(); };
        "cooperative_vector_builtin_call"_test = [] { test_cooperative_vector_builtin_call(); };
        "cooperative_vector_dsl"_test = [] { test_cooperative_vector_dsl(); };
        "cooperative_mul_add_ast"_test = [] { test_cooperative_mul_add_ast(); };
        "bindless_cooperative_mul_add_ast"_test = [] { test_bindless_cooperative_mul_add_ast(); };
        "cooperative_mul_ast"_test = [] { test_cooperative_mul_ast(); };
        "bindless_cooperative_mul_ast"_test = [] { test_bindless_cooperative_mul_ast(); };
        "cooperative_outer_product_accumulate_ast"_test = [] { test_cooperative_outer_product_accumulate_ast(); };
        "cooperative_mul_add_dsl"_test = [] { test_cooperative_mul_add_dsl(); };
        "bindless_cooperative_mul_add_dsl"_test = [] { test_bindless_cooperative_mul_add_dsl(); };
        "cooperative_mul_dsl"_test = [] { test_cooperative_mul_dsl(); };
        "bindless_cooperative_mul_dsl"_test = [] { test_bindless_cooperative_mul_dsl(); };
        "cooperative_outer_product_accumulate_dsl"_test = [] { test_cooperative_outer_product_accumulate_dsl(); };
        "cooperative_vector_load_ast"_test = [] { test_cooperative_vector_load_ast(); };
        "cooperative_vector_store_ast"_test = [] { test_cooperative_vector_store_ast(); };
        "cooperative_vector_splat_ast"_test = [] { test_cooperative_vector_splat_ast(); };
        "cooperative_vector_cast_ast"_test = [] { test_cooperative_vector_cast_ast(); };
        "cooperative_vector_load_dsl"_test = [] { test_cooperative_vector_load_dsl(); };
        "cooperative_vector_store_dsl"_test = [] { test_cooperative_vector_store_dsl(); };
        "cooperative_vector_splat_dsl"_test = [] { test_cooperative_vector_splat_dsl(); };
        "cooperative_vector_cast_dsl"_test = [] { test_cooperative_vector_cast_dsl(); };
        "bindless_cooperative_vector_load_ast"_test = [] { test_bindless_cooperative_vector_load_ast(); };
        "bindless_cooperative_vector_store_ast"_test = [] { test_bindless_cooperative_vector_store_ast(); };
        "bindless_cooperative_vector_load_dsl"_test = [] { test_bindless_cooperative_vector_load_dsl(); };
        "bindless_cooperative_vector_store_dsl"_test = [] { test_bindless_cooperative_vector_store_dsl(); };
        "cooperative_vector_workgroup_load_ast"_test = [] { test_cooperative_vector_workgroup_load_ast(); };
        "cooperative_vector_workgroup_store_ast"_test = [] { test_cooperative_vector_workgroup_store_ast(); };
        "cooperative_vector_workgroup_load_dsl"_test = [] { test_cooperative_vector_workgroup_load_dsl(); };
        "cooperative_vector_workgroup_store_dsl"_test = [] { test_cooperative_vector_workgroup_store_dsl(); };
        "cooperative_vector_min_dsl"_test = [] { test_cooperative_vector_min_dsl(); };
        "cooperative_vector_max_dsl"_test = [] { test_cooperative_vector_max_dsl(); };
        "cooperative_vector_clamp_dsl"_test = [] { test_cooperative_vector_clamp_dsl(); };
        "cooperative_vector_exp_dsl"_test = [] { test_cooperative_vector_exp_dsl(); };
        "cooperative_vector_log_dsl"_test = [] { test_cooperative_vector_log_dsl(); };
        "cooperative_vector_tanh_dsl"_test = [] { test_cooperative_vector_tanh_dsl(); };
        "cooperative_vector_atan_dsl"_test = [] { test_cooperative_vector_atan_dsl(); };
        "cooperative_vector_fma_dsl"_test = [] { test_cooperative_vector_fma_dsl(); };
        "cooperative_vector_bitwise_and_dsl"_test = [] { test_cooperative_vector_bitwise_and_dsl(); };
        "cooperative_vector_bitwise_or_dsl"_test = [] { test_cooperative_vector_bitwise_or_dsl(); };
        "cooperative_vector_bitwise_xor_dsl"_test = [] { test_cooperative_vector_bitwise_xor_dsl(); };
        "cooperative_vector_bitwise_not_dsl"_test = [] { test_cooperative_vector_bitwise_not_dsl(); };
        "cooperative_vector_shift_left_dsl"_test = [] { test_cooperative_vector_shift_left_dsl(); };
        "cooperative_vector_shift_right_dsl"_test = [] { test_cooperative_vector_shift_right_dsl(); };
        if (g_device_for_tests) {
            "cooperative_vector_device"_test = [] { test_cooperative_vector_device(*g_device_for_tests); };
            "cooperative_vector_load_store_device"_test = [] { test_cooperative_vector_load_store_device(*g_device_for_tests); };
            "cooperative_vector_splat_device"_test = [] { test_cooperative_vector_splat_device(*g_device_for_tests); };
            "cooperative_vector_workgroup_load_store_device"_test = [] { test_cooperative_vector_workgroup_load_store_device(*g_device_for_tests); };
        }
    }};

    return boost::ut::cfg().run(
               boost::ut::run_cfg{.argc = 1,
                                  .argv = const_cast<const char **>(ut_argv0)})
               ? 1
               : 0;
}
