// Test for cooperative vector AST construction and device execution.
// This test covers:
// - Cooperative vector/matrix reference type creation and descriptions
// - FunctionBuilder flag propagation for cooperative operations
// - DSL construction of CoopVector, CoopVectorRef, and CoopMatrixRef
// - All cooperative builtin calls listed in include/luisa/ast/op.h
// - Device compilation/execution when a supporting backend is provided

#include "ut/ut.hpp"

#include <luisa/ast/function_builder.h>
#include <luisa/ast/type.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/command_list.h>

#include <array>

#ifdef _WIN32
#include <luisa/backends/ext/dx_config_ext.h>
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

}// namespace

static auto test_cooperative_vector_registration = [] {
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
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    auto ut_argc = boost::ut::detail::cfg::largc;
    auto ut_argv = boost::ut::detail::cfg::largv;
    if (ut_argc <= 1) {
        LUISA_INFO("No backend argument provided; skipping device execution tests.");
        return 0;
    }

    luisa::string backend = ut_argv[1];
    if (backend == "dx") {
#ifdef _WIN32
        Context context{ut_argv[0]};
        DeviceConfig config;
        auto dx_config = luisa::make_unique<DXExperimentalConfigExt>();
        auto *dx_config_ptr = dx_config.get();
        config.extension = std::move(dx_config);
        Device device = context.create_device("dx", &config);
        if (!dx_config_ptr->ExperimentalFeaturesEnabled()) {
            LUISA_INFO("DX cooperative-vector experimental features are not available on this system; skipping device execution tests.");
            return 0;
        }
        test_cooperative_vector_device(device);
#else
        LUISA_INFO("DX backend is not available on this platform; skipping device execution tests.");
#endif
    } else if (backend == "vk") {
        Context context{ut_argv[0]};
        Device device = context.create_device("vk");
        test_cooperative_vector_device(device);
    } else {
        LUISA_INFO("This test only supports the dx or vk backend; got '{}'. Skipping device execution tests.", backend);
        return 0;
    }
}
