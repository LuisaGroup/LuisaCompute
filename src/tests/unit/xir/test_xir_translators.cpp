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
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/metadata/reg2mem_spill.h>
#include <luisa/xir/metadata/signature_constraint.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2text.h>
#include <luisa/xir/translators/xir2json.h>
#include <luisa/xir/verifier.h>
#include <yyjson.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

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

    "xir_to_text_preserves_u64_switch_case_bits"_test = [] {
        Module module;
        auto *callable = module.create_callable(nullptr);
        auto *selector =
            callable->create_value_argument(Type::of<uint64_t>());
        auto *body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *switch_inst = builder.switch_(selector);
        auto *low_word_block =
            switch_inst->create_case_block(0x00000000ffffffffull);
        auto *all_ones_block =
            switch_inst->create_case_block(0xffffffffffffffffull);
        auto *default_block = switch_inst->create_default_block();
        auto *merge_block = switch_inst->create_merge_block();
        for (auto *block :
             {low_word_block, all_ones_block, default_block}) {
            builder.set_insertion_point(block);
            builder.br(merge_block);
        }
        builder.set_insertion_point(merge_block);
        builder.return_void();

        auto verify = [](luisa::string_view text) noexcept {
            expect(text.find("case 4294967295 ") !=
                   luisa::string_view::npos);
            expect(text.find("case 18446744073709551615 ") !=
                   luisa::string_view::npos);
        };
        verify(xir_to_text_translate(&module, false));
        verify(xir_to_flat_text_translate(&module, false));
    };

    "xir_to_text_emits_all_marker_metadata"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        static_cast<void>(kernel->create_metadata<SignatureConstraintMD>());
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *spill = builder.alloca_local(Type::of<uint32_t>());
        spill->create_metadata<Reg2MemSpillMD>()->set_kind(
            Reg2MemSpillKind::CROSS_BLOCK);
        builder.return_void();

        auto text = xir_to_text_translate(&module, true);
        expect(text.find("signature_constraint") != luisa::string::npos);
        expect(text.find("reg2mem_spill = cross_block") !=
               luisa::string::npos);
        auto flat_text = xir_to_flat_text_translate(&module, true);
        expect(flat_text.find("signature_constraint") !=
               luisa::string::npos);
        expect(flat_text.find("reg2mem_spill = cross_block") !=
               luisa::string::npos);
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
        auto *doc = yyjson_read(json.data(), json.size(), YYJSON_READ_NOFLAG);
        expect(doc != nullptr);
        if (doc == nullptr) { return; }
        auto *root = yyjson_doc_get_root(doc);
        expect(yyjson_is_obj(root));
        if (!yyjson_is_obj(root)) {
            yyjson_doc_free(doc);
            return;
        }
        expect(yyjson_equals_str(yyjson_obj_get(root, "schema"), "luisa.xir.debug"));
        expect(yyjson_get_uint(yyjson_obj_get(root, "version")) == 1u);
        expect(yyjson_get_bool(yyjson_obj_get(root, "ok")));
        expect(yyjson_get_uint(yyjson_obj_get(root, "function_count")) >= 1u);
        expect(yyjson_get_uint(yyjson_obj_get(root, "instruction_count")) >= 1u);
        auto *text = yyjson_obj_get(root, "text");
        expect(yyjson_is_str(text));
        if (yyjson_is_str(text)) {
            expect(luisa::string_view{yyjson_get_str(text), yyjson_get_len(text)}.find("define {") != luisa::string_view::npos);
        }
        yyjson_doc_free(doc);
    };

    "xir_to_json_contains_functions"_test = [] {
        Kernel1D kernel = [](BufferFloat buf) {
            auto idx = dispatch_id().x;
            buf->write(idx, 1.0f);
        };
        auto module = ast_to_xir_translate(kernel.function()->function(), {});
        auto json = xir_to_json_translate(module.get());
        auto *doc = yyjson_read(json.data(), json.size(), YYJSON_READ_NOFLAG);
        expect(doc != nullptr);
        if (doc == nullptr) { return; }
        auto *root = yyjson_doc_get_root(doc);
        expect(yyjson_is_obj(root));
        if (!yyjson_is_obj(root)) {
            yyjson_doc_free(doc);
            return;
        }
        expect(yyjson_get_bool(yyjson_obj_get(root, "ok")));
        expect(yyjson_get_uint(yyjson_obj_get(root, "function_count")) == 1u);
        expect(yyjson_get_uint(yyjson_obj_get(root, "block_count")) >= 1u);
        expect(yyjson_get_uint(yyjson_obj_get(root, "constant_count")) >= 1u);
        yyjson_doc_free(doc);
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
        auto *doc = yyjson_read(json.data(), json.size(), YYJSON_READ_NOFLAG);
        expect(doc != nullptr);
        if (doc == nullptr) { return; }
        auto *root = yyjson_doc_get_root(doc);
        expect(yyjson_is_obj(root));
        if (!yyjson_is_obj(root)) {
            yyjson_doc_free(doc);
            return;
        }
        expect(yyjson_get_bool(yyjson_obj_get(root, "ok")));
        expect(yyjson_get_uint(yyjson_obj_get(root, "function_count")) == 0u);
        expect(yyjson_get_uint(yyjson_obj_get(root, "block_count")) == 0u);
        expect(yyjson_is_str(yyjson_obj_get(root, "text")));
        yyjson_doc_free(doc);
    };

    "xir_json_translate_null_module_reports_error"_test = [] {
        auto json = xir_to_json_translate(nullptr);
        auto *doc = yyjson_read(json.data(), json.size(), YYJSON_READ_NOFLAG);
        expect(doc != nullptr);
        if (doc == nullptr) { return; }
        auto *root = yyjson_doc_get_root(doc);
        expect(yyjson_is_obj(root));
        if (!yyjson_is_obj(root)) {
            yyjson_doc_free(doc);
            return;
        }
        expect(!yyjson_get_bool(yyjson_obj_get(root, "ok")));
        expect(yyjson_equals_str(yyjson_obj_get(root, "error"), "null XIR module"));
        yyjson_doc_free(doc);
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
