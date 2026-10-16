#include "ut/ut.hpp"

#include <luisa/luisa-compute.h>
#include <luisa/ast/external_function.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/translators/ast2xir.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "external_call_preserves_xir_argument_semantics_and_ids"_test = [] {
        auto external = luisa::make_shared<compute::ExternalFunction>(
            "lc_test_external", Type::of<void>(),
            luisa::vector<const Type *>{
                Type::of<uint32_t>(), Type::of<uint32_t>(),
                Type::of<Buffer<uint32_t>>(), Type::of<uint64_t>(),
                Type::of<uint64_t>()},
            luisa::vector<Usage>{
                Usage::READ, Usage::WRITE, Usage::READ,
                Usage::READ, Usage::READ});

        Kernel1D kernel = [external](UInt input, BufferUInt output) noexcept {
            Var<uint32_t> written;
            auto fb = compute::detail::FunctionBuilder::current();
            auto type_id = fb->type_id(Type::of<float3>());
            auto string_id = fb->string_id("external-xir-lowering");
            const Expression *args[]{
                input.expression(), written.expression(), output.expression(),
                type_id, string_id};
            fb->call(external, args);
            output.write(0u, written);
        };

        compute::Function ast_function{kernel.function().get()};
        auto module = xir::ast_to_xir_translate(ast_function, {});
        xir::ExternalFunction *xir_external = nullptr;
        xir::CallInst *xir_call = nullptr;
        for (auto function : module->function_list()) {
            if (function->derived_function_tag() == xir::DerivedFunctionTag::EXTERNAL) {
                xir_external = static_cast<xir::ExternalFunction *>(function);
            }
            if (auto definition = function->definition()) {
                definition->traverse_instructions([&](xir::Instruction *inst) noexcept {
                    if (inst->isa<xir::CallInst>()) {
                        auto call = static_cast<xir::CallInst *>(inst);
                        if (call->callee()->derived_function_tag() ==
                            xir::DerivedFunctionTag::EXTERNAL) {
                            xir_call = call;
                        }
                    }
                });
            }
        }

        expect(xir_external != nullptr);
        expect(xir_call != nullptr);
        if (xir_external == nullptr || xir_call == nullptr) { return; }
        auto external_name = xir_external->name();
        expect(external_name.has_value());
        if (external_name.has_value()) {
            expect(external_name.value() == "lc_test_external");
        }
        expect(xir_external->type() == Type::of<void>());
        luisa::vector<xir::Argument *> external_arguments;
        for (auto argument : xir_external->arguments()) {
            external_arguments.emplace_back(argument);
        }
        expect(external_arguments.size() == 5u);
        if (external_arguments.size() != 5u) { return; }
        expect(external_arguments[0u]->is_value());
        expect(external_arguments[1u]->is_reference());
        expect(external_arguments[2u]->is_resource());
        expect(external_arguments[3u]->is_value());
        expect(external_arguments[4u]->is_value());

        expect(xir_call->callee() == xir_external);
        expect(xir_call->argument_count() == 5u);
        expect(!xir_call->argument(0u)->is_lvalue());
        expect(xir_call->argument(1u)->is_lvalue());
        expect(xir_call->argument(2u)->type()->is_buffer());
        auto type_id_value = xir_call->argument(3u);
        auto string_id_value = xir_call->argument(4u);
        expect(type_id_value->isa<xir::Constant>());
        expect(string_id_value->isa<xir::Constant>());
        if (type_id_value->isa<xir::Constant>() &&
            string_id_value->isa<xir::Constant>()) {
            auto type_id = static_cast<xir::Constant *>(type_id_value);
            auto string_id = static_cast<xir::Constant *>(string_id_value);
            expect(type_id->as<uint64_t>() == 0u);
            expect(string_id->as<uint64_t>() ==
                   luisa::hash_value(luisa::string_view{"external-xir-lowering"}));
        }
    };

    return 0;
}
