#include "ut/ut.hpp"

#include <luisa/luisa-compute.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    "callable_swizzle_reference_is_addressable_and_written_back_before_result"_test = [] {
        for (auto variant = 0u; variant < 4u; variant++) {
            auto nested = (variant & 1u) != 0u;
            auto writes = (variant & 2u) == 0u;
            auto mutate = Callable<float(float2 &)>{[writes](Float2 &value) noexcept {
                if (writes) { value += make_float2(1.0f); }
                return value.x;
            }};
            Kernel1D kernel = [&](BufferFloat4 output, UInt index) noexcept {
                ArrayFloat4<2u> values;
                values[0u] = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
                values[1u] = make_float4(5.0f, 6.0f, 7.0f, 8.0f);
                auto &builder = *compute::detail::FunctionBuilder::current();
                const Expression *base = values[index & 1u].expression();
                if (nested) {
                    base = builder.swizzle(Type::of<float3>(), base, 3u, 0x012u);
                }
                auto swizzle = compute::detail::Ref<float2>{
                    builder.swizzle(Type::of<float2>(), base, 2u, 0x01u)};
                values[index & 1u].x = mutate(swizzle);
                output.write(0u, values[index & 1u]);
            };
            auto module = xir::ast_to_xir_translate(kernel.function()->function(), {});
            expect(module != nullptr);
            if (!module) { continue; }
            expect(xir::xir_verify_module(module.get()).succeeded());
            xir::CallInst *call = nullptr;
            for (auto function : module->function_list()) {
                if (function->derived_function_tag() != xir::DerivedFunctionTag::KERNEL) { continue; }
                function->definition()->traverse_instructions([&](xir::Instruction *inst) noexcept {
                    if (inst->isa<xir::CallInst>()) { call = static_cast<xir::CallInst *>(inst); }
                });
            }
            expect(call != nullptr);
            if (!call) { continue; }
            expect(call->argument_count() == 1u);
            expect(call->argument(0u)->is_lvalue());
            expect(call->argument(0u)->type() == Type::of<float2>());
            auto stores_before_result = 0u;
            auto result_assigned = false;
            for (auto inst = call->next(); inst != nullptr && !inst->is_sentinel(); inst = inst->next()) {
                if (inst->isa<xir::StoreInst>()) {
                    auto store = static_cast<xir::StoreInst *>(inst);
                    if (store->value() == call) {
                        result_assigned = true;
                        break;
                    }
                    expect(store->variable()->isa<xir::GEPInst>());
                    if (store->variable()->isa<xir::GEPInst>()) {
                        auto pointer = static_cast<xir::GEPInst *>(store->variable());
                        expect(pointer->index_count() == 1u);
                        expect(pointer->index(0u)->isa<xir::Constant>());
                        if (pointer->index(0u)->isa<xir::Constant>()) {
                            auto index = static_cast<xir::Constant *>(pointer->index(0u))->as<uint32_t>();
                            auto expected = stores_before_result == 0u ? 1u : (nested ? 2u : 0u);
                            expect(index == expected);
                        }
                    }
                    stores_before_result++;
                }
            }
            expect(result_assigned);
            expect(stores_before_result == (writes ? 2u : 0u))
                << "copy-out must precede an overlapping result assignment; readonly references must not write back";
        }
    };
    return 0;
}
