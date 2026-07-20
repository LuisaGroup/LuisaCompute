// Test for AST-to-XIR and XIR debug translators.
// This test covers:
// - AST translation identity, callable, control-flow, and staged APIs
// - structured text and flat-text snapshots
// - parseable JSON schema, counts, payload, and null-module diagnostics

#include "ut/ut.hpp"
#include <luisa/luisa-compute.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2text.h>
#include <luisa/xir/translators/xir2json.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

// Lightweight string-based helpers to validate JSON without pulling in yyjson.
// The JSON output from xir_to_json_translate is pretty-printed (2-space indent)
// so field patterns like `"key": value` are reliable.
namespace {
[[nodiscard]] bool json_is_obj(const luisa::string &json) noexcept {
    auto p = json.find_first_not_of(" \t\r\n");
    return p != luisa::string::npos && json[p] == '{';
}
[[nodiscard]] bool json_get_bool(const luisa::string &json, const char *key) noexcept {
    return json.find(luisa::string("\"") + key + "\": true") != luisa::string::npos;
}
[[nodiscard]] uint64_t json_get_uint(const luisa::string &json, const char *key) noexcept {
    luisa::string pattern = luisa::string("\"") + key + "\": ";
    auto pos = json.find(pattern);
    if (pos == luisa::string::npos) { return static_cast<uint64_t>(-1); }
    pos += pattern.size();
    return static_cast<uint64_t>(std::stoull(json.c_str() + pos));
}
[[nodiscard]] bool json_str_equals(const luisa::string &json, const char *key, const char *expected) noexcept {
    luisa::string pattern = luisa::string("\"") + key + "\": \"" + expected + "\"";
    return json.find(pattern) != luisa::string::npos;
}
[[nodiscard]] bool json_has_str_field(const luisa::string &json, const char *key) noexcept {
    luisa::string pattern = luisa::string("\"") + key + "\": \"";
    return json.find(pattern) != luisa::string::npos;
}
}// namespace

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

    "xir_ast_to_xir_does_not_merge_distinct_functions_by_hash"_test = [] {
        Callable a = [](Float x) { return x + 1.0f; };
        Callable b = [](Float x) { return x + 1.0f; };
        expect(a.function().hash() == b.function().hash());
        AST2XIRConfig config{};
        auto *ctx = ast_to_xir_translate_begin(config);
        expect(ctx != nullptr);
        ast_to_xir_translate_add_function(ctx, a.function());
        ast_to_xir_translate_add_function(ctx, b.function());
        auto module = ast_to_xir_translate_finalize(ctx);
        expect(module != nullptr);
        auto callable_count = 0u;
        for (auto *f : module->function_list()) {
            if (f->derived_function_tag() == DerivedFunctionTag::CALLABLE) {
                callable_count++;
            }
        }
        expect(callable_count == 2u) << "AST2XIR must key generated functions by builder identity, not only Function::hash()";
    };

    "xir_ast_to_xir_normalizes_promoted_unary_operands"_test = [] {
        Kernel1D kernel = [](BufferVar<uint8_t> input,
                             BufferVar<int8_t> signed_input,
                             BufferUInt output) {
            auto index = dispatch_id().x;
            auto value = input.read(index);
            auto base = index * 5u;
            output.write(base, clz(value));
            output.write(base + 1u, ctz(value));
            output.write(base + 2u, popcount(value));
            output.write(base + 3u, reverse(value));
            output.write(base + 4u, cast<uint32_t>(abs(signed_input.read(index))));
        };
        auto module = ast_to_xir_translate(kernel.function()->function(), {});
        expect(module != nullptr);
        auto bit_count = 0u;
        auto abs_count = 0u;
        for (auto *function : module->function_list()) {
            if (auto *definition = function->definition()) {
                definition->traverse_instructions([&](Instruction *instruction) noexcept {
                    if (!instruction->isa<ArithmeticInst>()) { return; }
                    auto *arithmetic = static_cast<ArithmeticInst *>(instruction);
                    switch (arithmetic->op()) {
                        case ArithmeticOp::CLZ:
                        case ArithmeticOp::CTZ:
                        case ArithmeticOp::POPCOUNT:
                        case ArithmeticOp::REVERSE:
                            bit_count++;
                            expect(arithmetic->type() == Type::of<uint32_t>());
                            expect(arithmetic->operand_count() == 1u);
                            if (arithmetic->operand_count() == 1u) {
                                expect(arithmetic->operand(0u)->type() == Type::of<uint32_t>());
                            }
                            break;
                        case ArithmeticOp::ABS:
                            abs_count++;
                            expect(arithmetic->type() == Type::of<int32_t>());
                            expect(arithmetic->operand_count() == 1u);
                            if (arithmetic->operand_count() == 1u) {
                                expect(arithmetic->operand(0u)->type() == Type::of<int32_t>());
                            }
                            break;
                        default: break;
                    }
                });
            }
        }
        expect(bit_count == 4u);
        expect(abs_count == 1u);
        expect(xir_verify_module(module.get()).succeeded());
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
        expect(json_is_obj(json));
        if (!json_is_obj(json)) { return; }
        expect(json_str_equals(json, "schema", "luisa.xir.debug"));
        expect(json_get_uint(json, "version") == 1u);
        expect(json_get_bool(json, "ok"));
        expect(json_get_uint(json, "function_count") >= 1u);
        expect(json_get_uint(json, "instruction_count") >= 1u);
        expect(json_has_str_field(json, "text"));
        expect(json.find("define {") != luisa::string::npos);
    };

    "xir_to_json_contains_functions"_test = [] {
        Kernel1D kernel = [](BufferFloat buf) {
            auto idx = dispatch_id().x;
            buf->write(idx, 1.0f);
        };
        auto module = ast_to_xir_translate(kernel.function()->function(), {});
        auto json = xir_to_json_translate(module.get());
        expect(json_is_obj(json));
        if (!json_is_obj(json)) { return; }
        expect(json_get_bool(json, "ok"));
        expect(json_get_uint(json, "function_count") == 1u);
        expect(json_get_uint(json, "block_count") >= 1u);
        expect(json_get_uint(json, "constant_count") >= 1u);
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
        expect(json_is_obj(json));
        if (!json_is_obj(json)) { return; }
        expect(json_get_bool(json, "ok"));
        expect(json_get_uint(json, "function_count") == 0u);
        expect(json_get_uint(json, "block_count") == 0u);
        expect(json_has_str_field(json, "text"));
    };

    "xir_json_translate_null_module_reports_error"_test = [] {
        auto json = xir_to_json_translate(nullptr);
        expect(json_is_obj(json));
        if (!json_is_obj(json)) { return; }
        expect(!json_get_bool(json, "ok"));
        expect(json_str_equals(json, "error", "null XIR module"));
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
