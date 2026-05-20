#include "ut/ut.hpp"
#include <luisa/luisa-compute.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2text.h>
#include <luisa/xir/translators/xir2json.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

// ---- AST to XIR translation ----

// ---- XIR to text translation ----

// ---- XIR to JSON translation ----

// ---- Direct XIR module to text/json ----

void reg_ast2xir() {

    "xir_ast_to_xir_simple_kernel"_test = [] {
        Kernel1D kernel = [](BufferFloat buf) {
            auto idx = dispatch_id().x;
            buf->write(idx, 42.0f);
        };
        auto module = ast_to_xir_translate(kernel.function()->function(), {});
        expect(module != nullptr) << "ast_to_xir should return non-null module";
        auto func_count = 0u;
        for ([[maybe_unused]] auto *f : module->function_list()) { func_count++; }
        expect(func_count >= 1u) << "translated module should have at least 1 function (the kernel)";
    };

    "xir_ast_to_xir_callable"_test = [] {
        Callable add_one = [](Float x) { return x + 1.0f; };
        Kernel1D kernel = [&add_one](BufferFloat buf) {
            auto idx = dispatch_id().x;
            auto val = buf->read(idx);
            buf->write(idx, add_one(val));
        };
        auto module = ast_to_xir_translate(kernel.function()->function(), {});
        expect(module != nullptr);
        auto func_count = 0u;
        for ([[maybe_unused]] auto *f : module->function_list()) { func_count++; }
        expect(func_count >= 2u) << "kernel + callable should produce at least 2 functions";
    };

    "xir_ast_to_xir_with_control_flow"_test = [] {
        Kernel1D kernel = [](BufferFloat buf) {
            auto idx = dispatch_id().x;
            auto val = buf->read(idx);
            Var<float> result = 0.0f;
            $if (val > 0.0f) {
                result = val * 2.0f;
            }
            $else {
                result = 0.0f;
            };
            buf->write(idx, result);
        };
        auto module = ast_to_xir_translate(kernel.function()->function(), {});
        expect(module != nullptr);
    };

    "xir_ast_to_xir_begin_add_finalize"_test = [] {
        Kernel1D kernel = [](BufferFloat buf) {
            auto idx = dispatch_id().x;
            buf->write(idx, 1.0f);
        };
        AST2XIRConfig config{};
        auto *ctx = ast_to_xir_translate_begin(config);
        expect(ctx != nullptr);
        ast_to_xir_translate_add_function(ctx, kernel.function()->function());
        auto module = ast_to_xir_translate_finalize(ctx);
        expect(module != nullptr);
        auto func_count = 0u;
        for ([[maybe_unused]] auto *f : module->function_list()) { func_count++; }
        expect(func_count >= 1u);
    };
}

void reg_xir2text() {

    "xir_to_text_basic"_test = [] {
        Kernel1D kernel = [](BufferFloat buf) {
            auto idx = dispatch_id().x;
            buf->write(idx, 42.0f);
        };
        auto module = ast_to_xir_translate(kernel.function()->function(), {});
        auto text = xir_to_text_translate(module.get(), false);
        expect(!text.empty()) << "text output should not be empty";
    };

    "xir_to_text_with_debug_info"_test = [] {
        Kernel1D kernel = [](BufferFloat buf) {
            auto idx = dispatch_id().x;
            buf->write(idx, 1.0f);
        };
        auto module = ast_to_xir_translate(kernel.function()->function(), {});
        auto text_no_debug = xir_to_text_translate(module.get(), false);
        auto text_debug = xir_to_text_translate(module.get(), true);
        expect(!text_no_debug.empty());
        expect(!text_debug.empty());
        expect(text_debug.size() >= text_no_debug.size()) << "debug info should add content";
    };

    "xir_to_flat_text_basic"_test = [] {
        Kernel1D kernel = [](BufferFloat buf) {
            auto idx = dispatch_id().x;
            buf->write(idx, 1.0f);
        };
        auto module = ast_to_xir_translate(kernel.function()->function(), {});
        auto text = xir_to_flat_text_translate(module.get(), true);
        expect(!text.empty());
        expect(text.find("define {") != luisa::string::npos);
    };
}

void reg_xir2json() {

    "xir_to_json_basic"_test = [] {
        Kernel1D kernel = [](BufferFloat buf) {
            auto idx = dispatch_id().x;
            buf->write(idx, 42.0f);
        };
        auto module = ast_to_xir_translate(kernel.function()->function(), {});
        auto json = xir_to_json_translate(module.get());
        expect(!json.empty()) << "JSON output should not be empty";
    };

    "xir_to_json_contains_functions"_test = [] {
        Kernel1D kernel = [](BufferFloat buf) {
            auto idx = dispatch_id().x;
            buf->write(idx, 1.0f);
        };
        auto module = ast_to_xir_translate(kernel.function()->function(), {});
        auto json = xir_to_json_translate(module.get());
        expect(!json.empty());
    };
}

void reg_direct_module() {

    "xir_text_translate_empty_module"_test = [] {
        Module module;
        auto text = xir_to_text_translate(&module, false);
        expect(!text.empty()) << "even empty module should produce some text output";
    };

    "xir_json_translate_empty_module"_test = [] {
        Module module;
        auto json = xir_to_json_translate(&module);
        expect(!json.empty()) << "even empty module should produce some JSON output";
    };

    "xir_text_translate_module_with_kernel"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        kernel->set_name("test_kernel");
        kernel->set_block_size(make_uint3(256u, 1u, 1u));
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.return_void();
        auto text = xir_to_text_translate(&module, true);
        expect(!text.empty());
        auto flat_text = xir_to_flat_text_translate(&module, true);
        expect(!flat_text.empty());
        expect(flat_text.find("define {") != luisa::string::npos);
    };
}

int main(int argc, char *argv[]) {

    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_ast2xir();
    reg_xir2text();
    reg_xir2json();
    reg_direct_module();
    return 0;
}
