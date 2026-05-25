#include <array>
#include <cctype>
#include <numeric>

#include <luisa/ast/function_builder.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/dsl/syntax.h>
#include <luisa/luisa-compute.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2ast.h>
#include <luisa/xir/translators/xir2text.h>

using namespace luisa;
using namespace luisa::compute;

struct Test {
    int3 something;
    float a;
};

LUISA_STRUCT(Test, something, a) {};

static Device create_test_device(Context &context, int argc, char *argv[]) {
    if (argc > 1) { return context.create_device(argv[1]); }
    auto device = context.create_default_device();
    LUISA_INFO("No backend specified. Using default backend '{}'.", device.backend_name());
    return device;
}

static luisa::string canonicalize_xir_text(luisa::string_view text) {
    luisa::unordered_map<luisa::string, luisa::string> id_map;
    luisa::string result;
    result.reserve(text.size());
    auto next_id = 0u;
    for (auto i = 0u; i < text.size();) {
        auto c = text[i];
        if (c == '%' &&
            i + 1u < text.size() &&
            std::isdigit(static_cast<unsigned char>(text[i + 1u])) != 0) {
            auto j = i + 1u;
            while (j < text.size() &&
                   std::isdigit(static_cast<unsigned char>(text[j])) != 0) {
                j++;
            }
            auto key = luisa::string{text.substr(i, j - i)};
            if (!id_map.contains(key)) {
                auto value = luisa::string{"%v"};
                value.append(std::to_string(next_id++));
                id_map.emplace(key, std::move(value));
            }
            result.append(id_map.at(key));
            i = j;
            continue;
        }
        if (c != '\r') { result.push_back(c); }
        i++;
    }
    return result;
}

static void smoke_dispatch_roundtrip_kernel(Device &device,
                                            Stream &stream,
                                            const luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder> &roundtrip_ast) {
    LUISA_ASSERT(roundtrip_ast != nullptr, "Round-tripped AST must not be null.");
    auto shader = device.create<Shader1D<Buffer<float>, uint>>(
        roundtrip_ast->function(),
        ShaderOption{.name = luisa::string{"xir2ast_roundtrip_smoke"}});
    std::array<float, 16u> host{};
    std::iota(host.begin(), host.end(), 0.0f);
    auto buffer = device.create_buffer<float>(host.size());
    stream << buffer.copy_from(host.data()) << synchronize();
    stream << shader(buffer, 0u).dispatch(1u) << synchronize();
    LUISA_INFO("Round-tripped AST kernel smoke dispatch finished.");
}

static bool effectful_roundtrip_runtime_test(Device &device, Stream &stream) {
    Kernel1D<Buffer<float>, uint> runtime_kernel = [&](BufferVar<float> buffer, Var<uint> count) noexcept -> void {
        auto index = dispatch_id().x;
        auto value = buffer.read(index);
        Var<float3> broadcast = make_float3(cast<float>(count) + value);
        buffer.write(index, broadcast.x + broadcast.y + broadcast.z);
    };

    auto runtime_xir = xir::ast_to_xir_translate(runtime_kernel.function()->function(), {});
    auto runtime_function = runtime_xir->function_list().front();
    LUISA_ASSERT(runtime_function->isa<xir::KernelFunction>(),
                 "Expected the runtime test function to be a kernel.");
    auto runtime_roundtrip_ast = xir::XIR2AST::build(
        static_cast<const xir::KernelFunction *>(runtime_function));

    auto original_shader = device.compile(runtime_kernel, ShaderOption{.name = luisa::string{"xir2ast_runtime_original"}});
    auto roundtrip_shader = device.create<Shader1D<Buffer<float>, uint>>(
        runtime_roundtrip_ast->function(),
        ShaderOption{.name = luisa::string{"xir2ast_runtime_roundtrip"}});

    std::array<float, 8u> input{};
    std::iota(input.begin(), input.end(), 0.5f);
    std::array<float, input.size()> original_output{};
    std::array<float, input.size()> roundtrip_output{};

    auto original_buffer = device.create_buffer<float>(input.size());
    auto roundtrip_buffer = device.create_buffer<float>(input.size());

    stream << original_buffer.copy_from(input.data())
           << roundtrip_buffer.copy_from(input.data())
           << synchronize();

    stream << original_shader(original_buffer, 2u).dispatch(static_cast<uint>(input.size()))
           << roundtrip_shader(roundtrip_buffer, 2u).dispatch(static_cast<uint>(input.size()))
           << synchronize();

    stream << original_buffer.copy_to(original_output.data())
           << roundtrip_buffer.copy_to(roundtrip_output.data())
           << synchronize();

    auto same = original_output == roundtrip_output;
    LUISA_INFO("Original runtime output: [{}, {}, {}, {}, {}, {}, {}, {}]",
               original_output[0], original_output[1], original_output[2], original_output[3],
               original_output[4], original_output[5], original_output[6], original_output[7]);
    LUISA_INFO("Round-trip runtime output: [{}, {}, {}, {}, {}, {}, {}, {}]",
               roundtrip_output[0], roundtrip_output[1], roundtrip_output[2], roundtrip_output[3],
               roundtrip_output[4], roundtrip_output[5], roundtrip_output[6], roundtrip_output[7]);
    return same;
}

int main(int argc, char *argv[]) {
    luisa::log_level_verbose();
    Context context{argv[0]};
    auto device = create_test_device(context, argc, argv);
    auto stream = device.create_stream();
    LUISA_INFO("Using backend '{}'.", device.backend_name());

    std::vector<int> const_vector(128u);
    std::iota(const_vector.begin(), const_vector.end(), 0);

    Clock clock;
    Constant float_consts = {1.0f, 2.0f};
    Constant int_consts = const_vector;

    Kernel1D<Buffer<float>, uint> kernel_def = [&](BufferVar<float> buffer_float, Var<uint> count) noexcept -> void {
        for (auto n = 0u; n < 1u; n++) {
            Shared<float4> shared_floats{16};

            count += 1u;

            Constant float_consts = {1.0f, 2.0f};
            auto ff = float_consts.read(0);

            Var mat = make_float2x2(1.0f, 2.0f, 3.0f, 4.0f);
            Var mat2 = make_float2x2(1.0f, 2.0f, 3.0f, 4.0f);
            Var mat3 = mat * mat2;
            Var mat4 = mat3 * make_float2(2.f);

            Var v_int = 10;
            Var t = make_int3(1, 2, 3);
            Var vv = ite(t == 10, 1, 2);

            Var vvv = min(vv, 10);
            Var xxx = make_uint4(5);
            Var vvvv = min(xxx, 1u);

            loop([] {
                if_(true, break_);
            });

            for (auto v : dynamic_range(v_int)) {
                v_int += v;
            }

            Var vv_int = int_consts[v_int];
            Var v_float = buffer_float.read(count + thread_id().x);
            Var vv_float = float_consts[0];

            Var v_float_copy = v_float;

            Var z = -1 + v_int * v_float + 1.0f;
            z += 1;
            Var v_vec = float3{1.0f};
            Var v2 = float3{2.0f} - v_vec * 2.0f;
            v2 *= 5.0f + v_float;

            Var<float2> w{cast<float>(v_int), v_float};
            w *= float2{1.2f};

            if_(v_int == v_int, [] {
                Var a = 0.0f;
            }).else_([] {
                Var c = 2.0f;
            });

            switch_(123)
                .case_(1, [] {})
                .case_(2, [] {})
                .default_([] {});

            Var x = w.x;

            Var<int3> s;
            Var<Test> vvt{s, v_float_copy};
            Var<Test> vt{vvt};

            Var vt_copy = vt;
            Var c = 0.5f + vt.a * 1.0f;
        }
    };
    LUISA_INFO("Kernel definition parsed in {} ms.", clock.toc());

    clock.tic();
    auto xir_module = xir::ast_to_xir_translate(kernel_def.function()->function(), {});
    LUISA_INFO("AST2XIR done in {} ms.", clock.toc());

    clock.tic();
    auto xir_function = xir_module->function_list().front();
    LUISA_ASSERT(xir_function->isa<xir::KernelFunction>(), "Expected the first XIR function to be a kernel.");
    auto xir_kernel = static_cast<const xir::KernelFunction *>(xir_function);
    auto xir_ast = xir::XIR2AST::build(xir_kernel);
    LUISA_INFO("XIR2AST done in {} ms.", clock.toc());

    auto original_xir_text = xir::xir_to_text_translate(xir_module.get(), false);
    auto roundtrip_xir_module = xir::ast_to_xir_translate(xir_ast->function(), {});
    auto roundtrip_xir_text = xir::xir_to_text_translate(roundtrip_xir_module.get(), false);
    auto canonical_original_xir = canonicalize_xir_text(original_xir_text);
    auto canonical_roundtrip_xir = canonicalize_xir_text(roundtrip_xir_text);

    auto raw_same = original_xir_text == roundtrip_xir_text;
    auto canonical_same = canonical_original_xir == canonical_roundtrip_xir;

    LUISA_INFO("AST -> XIR dump:\n{}", original_xir_text);
    LUISA_INFO("AST -> XIR -> AST -> XIR dump:\n{}", roundtrip_xir_text);
    LUISA_INFO("Raw text compare result: {}", raw_same ? "same" : "different");
    LUISA_INFO("Canonical logical compare result: {}", canonical_same ? "same" : "different");
    if (!canonical_same) {
        LUISA_INFO("Canonical AST -> XIR dump:\n{}", canonical_original_xir);
        LUISA_INFO("Canonical AST -> XIR -> AST -> XIR dump:\n{}", canonical_roundtrip_xir);
    }

    smoke_dispatch_roundtrip_kernel(device, stream, xir_ast);

    auto runtime_same = effectful_roundtrip_runtime_test(device, stream);
    LUISA_INFO("Effectful runtime compare result: {}", runtime_same ? "same" : "different");
    LUISA_ASSERT(runtime_same,
                 "Round-tripped AST kernel runtime output does not match the original AST kernel.");
    return 0;
}
