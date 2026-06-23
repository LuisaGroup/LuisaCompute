// Test for cooperative vector AST construction and device execution.
// This test covers:
// - Cooperative vector/matrix reference type creation and descriptions
// - FunctionBuilder flag propagation for cooperative operations
// - DSL construction of CoopVector, CoopVectorRef, and CoopMatrixRef
// - The cooperative_vector_accumulate builtin call
// - Device compilation/execution when a supporting backend is provided

#include "ut/ut.hpp"

#include <luisa/ast/function_builder.h>
#include <luisa/ast/type.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

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

// Device-side execution test.  This requires the DX backend with Shader
// Model 6.8 cooperative-vector support; experimental features must be
// enabled through the device extension.
void test_cooperative_vector_device(Device &device) {
    luisa::log_level_verbose();
    LUISA_INFO("Running cooperative vector device test on backend '{}'", device.backend_name());

    Stream stream = device.create_stream();
    constexpr auto n = 8u;
    constexpr auto byte_size = n * sizeof(float);

    ByteBuffer vector_buffer = device.create_byte_buffer(byte_size);
    stream << vector_buffer.copy_from(std::array<std::byte, byte_size>{}.data()) << synchronize();

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
    stream << shader(vector_buffer).dispatch(1u)
           << synchronize();

    luisa::vector<std::byte> host(byte_size);
    stream << vector_buffer.copy_to(host.data())
           << synchronize();

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
    if (backend != "dx") {
        LUISA_INFO("This test only supports the dx backend; got '{}'. Skipping device execution tests.", backend);
        return 0;
    }

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
}
