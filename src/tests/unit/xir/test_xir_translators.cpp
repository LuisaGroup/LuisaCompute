// Test for AST-to-XIR and XIR debug translators.
// This test covers:
// - AST translation identity, callable, control-flow, and staged APIs
// - structured text and flat-text snapshots
// - parseable JSON schema, counts, payload, and null-module diagnostics

#include "ut/ut.hpp"
#include <array>
#include <luisa/luisa-compute.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/resource.h>
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

namespace {

void debug_break_wrapper_a(void *, DebugBreakStmt::Evaluator *) {}
void debug_break_wrapper_b(void *, DebugBreakStmt::Evaluator *) {}
void cpu_custom_callback_a(void *, void *) {}
void cpu_custom_callback_b(void *, void *) {}
void cpu_custom_destructor(void *) {}

template<typename Pred>
[[nodiscard]] size_t count_functions(
    const Module *module, Pred predicate) noexcept {
    auto count = 0u;
    for (auto *function : module->function_list()) {
        if (predicate(function)) { count++; }
    }
    return count;
}

[[nodiscard]] const CallableFunction *
find_only_callable(const Module *module) noexcept {
    const CallableFunction *result = nullptr;
    for (auto *function : module->function_list()) {
        if (function->isa<CallableFunction>()) {
            result = static_cast<const CallableFunction *>(function);
        }
    }
    return result;
}

[[nodiscard]] const FunctionDefinition *
find_kernel_definition(const Module *module) noexcept {
    for (auto *function : module->function_list()) {
        if (function->isa<KernelFunction>()) {
            return function->definition();
        }
    }
    return nullptr;
}

[[nodiscard]] size_t count_calls_to(
    const FunctionDefinition *definition,
    const CallableFunction *callee) noexcept {
    auto count = 0u;
    if (definition == nullptr) { return count; }
    definition->traverse_instructions(
        [&](const Instruction *instruction) noexcept {
            if (!instruction->isa<CallInst>()) { return; }
            auto *call = static_cast<const CallInst *>(instruction);
            if (call->callee() == callee) { count++; }
        });
    return count;
}

}// namespace

void reg_ast2xir() {

    "xir_ast_to_xir_preserves_undefined_aggregate"_test = [] {
        using Bank = std::array<float4, 3u>;
        expect(luisa::to_string(CallOp::UNDEFINED) == "UNDEFINED")
            << "the appended operation must remain discoverable by CallOp users";
        Kernel1D kernel = [](BufferVar<Bank> output) {
            output.write(dispatch_id().x, undefined<Bank>());
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        expect(module != nullptr);
        expect(xir_verify_module(module.get()).succeeded());

        auto undefined_operand_count = 0u;
        auto *definition = find_kernel_definition(module.get());
        expect(definition != nullptr);
        definition->traverse_instructions(
            [&](const Instruction *instruction) noexcept {
                for (auto i = 0u; i < instruction->operand_count(); ++i) {
                    auto *operand = instruction->operand(i);
                    if (operand != nullptr &&
                        operand->derived_value_tag() ==
                        DerivedValueTag::UNDEFINED) {
                        expect(operand->type() == Type::of<Bank>());
                        undefined_operand_count++;
                    }
                }
            });
        expect(undefined_operand_count == 1u)
            << "undefined must remain a value, not become a zero constant";
    };

    "xir_ast_to_xir_simple_kernel"_test = [] {
        Kernel1D kernel = [](BufferFloat buf) {
            auto idx = dispatch_id().x;
            buf->write(idx, 42.0f);
        };
        auto module = ast_to_xir_translate(kernel.function()->function(), {});
        expect(module != nullptr) << "ast_to_xir should return non-null module";
        expect(count_functions(module.get(), [](auto *) { return true; }) >= 1u)
            << "translated module should have at least 1 function (the kernel)";
    };

    "xir_ast_to_xir_preserves_bindless_access_axes"_test = [] {
        Kernel1D kernel = [](BindlessVar bindless, BufferUInt output) {
            auto lane = dispatch_id().x;
            auto typed_uniform =
                bindless.buffer<uint32_t>(lane, true, true);
            auto typed_divergent =
                bindless.buffer<uint32_t>(lane, true, false);
            auto ordinary_uniform =
                bindless.buffer<uint32_t>(lane, false, true);
            output.write(0u, typed_uniform.read(0u));
            typed_divergent.write(1u, 42u);
            output.write(1u, ordinary_uniform.size());
        };
        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        expect(module != nullptr);
        expect(xir_verify_module(module.get()).succeeded());

        auto saw_typed_uniform_read = false;
        auto saw_typed_divergent_write = false;
        auto saw_ordinary_uniform_query = false;
        auto *definition = find_kernel_definition(module.get());
        expect(definition != nullptr);
        definition->traverse_instructions(
            [&](const Instruction *instruction) noexcept {
                if (instruction->isa<ResourceReadInst>()) {
                    auto read = static_cast<const ResourceReadInst *>(
                        instruction);
                    if (read->op() ==
                            ResourceReadOp::BINDLESS_BUFFER_READ &&
                        read->bindless_access() ==
                            BindlessResourceAccess{
                                .typed = true, .uniform = true}) {
                        saw_typed_uniform_read = true;
                    }
                } else if (instruction->isa<ResourceWriteInst>()) {
                    auto write = static_cast<const ResourceWriteInst *>(
                        instruction);
                    if (write->op() ==
                            ResourceWriteOp::BINDLESS_BUFFER_WRITE &&
                        write->bindless_access() ==
                            BindlessResourceAccess{
                                .typed = true, .uniform = false}) {
                        saw_typed_divergent_write = true;
                    }
                } else if (instruction->isa<ResourceQueryInst>()) {
                    auto query = static_cast<const ResourceQueryInst *>(
                        instruction);
                    if (query->op() ==
                            ResourceQueryOp::BINDLESS_BUFFER_SIZE &&
                        query->bindless_access() ==
                            BindlessResourceAccess{
                                .typed = false, .uniform = true}) {
                        saw_ordinary_uniform_query = true;
                    }
                }
            });
        expect(saw_typed_uniform_read);
        expect(saw_typed_divergent_write);
        expect(saw_ordinary_uniform_query);
    };

    "ast_bindless_resource_call_axes_are_bijective"_test = [] {
        constexpr auto all_variants_round_trip = []() noexcept {
            constexpr auto begin = luisa::to_underlying(
                CallOp::BINDLESS_TEXTURE2D_SAMPLE);
            constexpr auto end = luisa::to_underlying(
                CallOp::BINDLESS_BUFFER_ADDRESS);
            for (auto value = begin; value <= end; ++value) {
                auto base = static_cast<CallOp>(value);
                for (auto typed : {false, true}) {
                    for (auto uniform : {false, true}) {
                        auto specialized = specialize_bindless_resource_call(
                            base, typed, uniform);
                        if (canonical_bindless_resource_call(specialized) !=
                                base ||
                            is_typed_bindless_resource_call(specialized) !=
                                typed ||
                            is_uniform_bindless_resource_call(specialized) !=
                                uniform) {
                            return false;
                        }
                    }
                }
            }
            return true;
        }();
        static_assert(all_variants_round_trip);
        expect(all_variants_round_trip);
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
        expect(count_functions(module.get(), [](auto *) { return true; }) >= 2u)
            << "kernel + callable should produce at least 2 functions";
    };

    "xir_ast_to_xir_dense_variable_uids_are_function_local"_test = [] {
        // FunctionBuilder assigns UIDs independently per builder. The callee
        // therefore deliberately overlaps the kernel's argument/local UIDs,
        // while kernel builtins leave unmapped holes before its local values.
        Callable add_one = [](Float x) {
            Var<float> local = x + 1.0f;
            return local;
        };
        Kernel1D kernel = [&add_one](BufferFloat output) {
            auto thread = thread_id().x;
            auto block = block_id().x;
            auto dispatch = dispatch_id().x;
            auto kernel_index = kernel_id();
            auto lane_count = warp_lane_count();
            auto lane = warp_lane_id();
            Var<uint> mixed = thread + block + dispatch + kernel_index +
                              lane_count + lane;
            Var<float> local = add_one(cast<float>(mixed));
            output.write(dispatch, local);
        };

        auto module = ast_to_xir_translate(
            kernel.function()->function(), {});
        expect(module != nullptr);
        expect(xir_verify_module(module.get()).succeeded());
        expect(module->special_register_list().count_size() == 6u)
            << "builtin UID holes must resolve to their six exact registers";
        auto *callable = find_only_callable(module.get());
        auto *kernel_definition = find_kernel_definition(module.get());
        expect(callable != nullptr);
        expect(kernel_definition != nullptr);
        expect(count_calls_to(kernel_definition, callable) == 1u)
            << "overlapping caller/callee UIDs must remain frame-local";
    };

    "xir_ast_to_xir_merges_equivalent_callable_definitions"_test = [] {
        // One callable used from many call sites must produce a single
        // definition that every call site references.
        Callable add_one = [](Float x) { return x + 1.0f; };
        Kernel1D kernel = [&add_one](BufferFloat buffer) {
            auto index = dispatch_id().x;
            Float value = buffer.read(index);
            for (auto i = 0u; i < 16u; i++) { value = add_one(value); }
            buffer.write(index, value);
        };
        auto module = ast_to_xir_translate(kernel.function()->function(), {});
        expect(module != nullptr);
        auto *callable = find_only_callable(module.get());
        auto *kernel_definition = find_kernel_definition(module.get());
        expect(callable != nullptr);
        expect(kernel_definition != nullptr);
        expect(count_calls_to(kernel_definition, callable) == 16u)
            << "all call sites must reference the one translated Callable";
        expect(xir_verify_module(module.get()).succeeded());

        // Two independently constructed callables with one structural hash
        // must share one definition through the staged translation API.
        Callable twin = [](Float x) { return x + 1.0f; };
        expect(add_one.function().hash() == twin.function().hash());
        AST2XIRConfig config{};
        auto *ctx = ast_to_xir_translate_begin(config);
        expect(ctx != nullptr);
        ast_to_xir_translate_add_function(ctx, add_one.function());
        ast_to_xir_translate_add_function(ctx, twin.function());
        auto merged = ast_to_xir_translate_finalize(ctx);
        expect(merged != nullptr);
        expect(count_functions(
                   merged.get(),
                   [](auto *f) {
                       return f->derived_function_tag() ==
                              DerivedFunctionTag::CALLABLE;
                   }) == 1u)
            << "independently constructed callables with one structural hash "
               "must share one XIR definition";

        // The completed AST call graph retains one canonical builder per hash.
        Kernel1D dual_call_kernel = [&add_one, &twin](BufferFloat buffer) {
            auto index = dispatch_id().x;
            buffer.write(index,
                         add_one(buffer.read(index)) +
                             twin(buffer.read(index)));
        };
        expect(dual_call_kernel.function()->function().custom_callables().size() ==
               1u)
            << "the completed AST call graph must retain one canonical builder "
               "per structural hash";
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
        expect(count_functions(module.get(), [](auto *) { return true; }) >= 1u);
    };

    "ast_function_hash_covers_declared_variables"_test = [] {
        auto plain = compute::detail::FunctionBuilder::define_callable([] {
            auto *builder = compute::detail::FunctionBuilder::current();
            auto *argument = builder->argument(Type::of<float>());
            builder->return_(argument);
        });
        auto with_unused_local = compute::detail::FunctionBuilder::define_callable([] {
            auto *builder = compute::detail::FunctionBuilder::current();
            auto *argument = builder->argument(Type::of<float>());
            static_cast<void>(builder->local(Type::of<float>()));
            builder->return_(argument);
        });
        expect(plain->body()->hash() == with_unused_local->body()->hash());
        expect(plain->hash() != with_unused_local->hash())
            << "local declarations emitted by AST2XIR must participate in Function::hash()";

        auto plain_kernel = compute::detail::FunctionBuilder::define_kernel([] {});
        auto with_unused_shared = compute::detail::FunctionBuilder::define_kernel([] {
            auto *builder = compute::detail::FunctionBuilder::current();
            static_cast<void>(builder->shared(Type::array(Type::of<float>(), 8u)));
        });
        expect(plain_kernel->body()->hash() == with_unused_shared->body()->hash());
        expect(plain_kernel->hash() != with_unused_shared->hash())
            << "shared-memory declarations emitted by AST2XIR must participate in Function::hash()";
    };

    "ast_function_hash_covers_argument_layout"_test = [] {
        auto with_runtime_buffer = compute::detail::FunctionBuilder::define_kernel([] {
            auto *builder = compute::detail::FunctionBuilder::current();
            static_cast<void>(builder->buffer(Type::of<Buffer<float>>()));
        });
        auto with_bound_buffer = compute::detail::FunctionBuilder::define_kernel([] {
            auto *builder = compute::detail::FunctionBuilder::current();
            static_cast<void>(builder->buffer_binding(
                Type::of<Buffer<float>>(), 0x1234u, 0u, 64u));
        });
        expect(with_runtime_buffer->body()->hash() == with_bound_buffer->body()->hash());
        expect(with_runtime_buffer->hash() != with_bound_buffer->hash())
            << "captured-versus-runtime argument layout must participate without hashing resource handles";
    };

    "ast_value_hash_covers_codegen_semantics"_test = [] {
        DebugBreakStmt debug_a{debug_break_wrapper_a, {}};
        DebugBreakStmt debug_b{debug_break_wrapper_b, {}};
        expect(debug_a.hash() != debug_b.hash())
            << "debug-break wrappers change the emitted XIR instruction";

        auto binding_a = compute::Function::BufferBinding{0x1234u, 16u, 64u};
        auto binding_b = compute::Function::BufferBinding{0x1234u, 16u, 128u};
        expect(binding_a.hash() != binding_b.hash())
            << "buffer binding size must participate in its complete value hash";

        auto half_a = luisa::bit_cast<half>(static_cast<ushort>(0x7e01u));
        auto half_b = luisa::bit_cast<half>(static_cast<ushort>(0x7e02u));
        auto constant_a = ConstantData::create(Type::of<half>(), &half_a, sizeof(half_a));
        auto constant_b = ConstantData::create(Type::of<half>(), &half_b, sizeof(half_b));
        expect(constant_a.hash() != constant_b.hash())
            << "half constants must preserve exact payload bits, including NaN payloads";

        uint64_t cpu_hash_a = 0u;
        uint64_t cpu_hash_b = 0u;
        uint64_t gpu_hash_a = 0u;
        uint64_t gpu_hash_b = 0u;
        auto custom_hash_scope = compute::detail::FunctionBuilder::define_callable([&] {
            auto *builder = compute::detail::FunctionBuilder::current();
            auto *argument = builder->argument(Type::of<float>());
            CpuCustomOpExpr cpu_a{Type::of<float>(), cpu_custom_callback_a,
                                  cpu_custom_destructor, nullptr, argument};
            CpuCustomOpExpr cpu_b{Type::of<float>(), cpu_custom_callback_b,
                                  cpu_custom_destructor, nullptr, argument};
            GpuCustomOpExpr gpu_a{Type::of<float>(), "source_a", argument};
            GpuCustomOpExpr gpu_b{Type::of<float>(), "source_b", argument};
            cpu_hash_a = cpu_a.hash();
            cpu_hash_b = cpu_b.hash();
            gpu_hash_a = gpu_a.hash();
            gpu_hash_b = gpu_b.hash();
            builder->return_(argument);
        });
        static_cast<void>(custom_hash_scope);
        expect(cpu_hash_a != cpu_hash_b)
            << "CPU custom callback identity must participate in expression hashing";
        expect(gpu_hash_a != gpu_hash_b)
            << "GPU custom source must participate in expression hashing";
    };

    "xir_ast_to_xir_does_not_merge_different_callable_hashes"_test = [] {
        Callable add_one = [](Float x) { return x + 1.0f; };
        Callable add_two = [](Float x) { return x + 2.0f; };
        expect(add_one.function().hash() != add_two.function().hash());
        AST2XIRConfig config{};
        auto *ctx = ast_to_xir_translate_begin(config);
        ast_to_xir_translate_add_function(ctx, add_one.function());
        ast_to_xir_translate_add_function(ctx, add_two.function());
        auto module = ast_to_xir_translate_finalize(ctx);
        expect(count_functions(
                   module.get(),
                   [](auto *f) { return f->template isa<CallableFunction>(); }) == 2u);
    };

    "xir_ast_to_xir_preserves_distinct_kernel_entries"_test = [] {
        Kernel1D a = [](BufferFloat buffer) {
            buffer.write(dispatch_id().x, 1.0f);
        };
        Kernel1D b = [](BufferFloat buffer) {
            buffer.write(dispatch_id().x, 1.0f);
        };
        expect(a.function()->function().hash() == b.function()->function().hash());
        AST2XIRConfig config{};
        auto *ctx = ast_to_xir_translate_begin(config);
        ast_to_xir_translate_add_function(ctx, a.function()->function());
        ast_to_xir_translate_add_function(ctx, b.function()->function());
        auto module = ast_to_xir_translate_finalize(ctx);
        expect(count_functions(
                   module.get(),
                   [](auto *f) { return f->template isa<KernelFunction>(); }) == 2u)
            << "hash canonicalization must not collapse independently "
               "addressable entry points";
    };

    "xir_ast_to_xir_keeps_distinct_capture_operands_when_merging_callables"_test = [] {
        Kernel1D kernel = [](BufferFloat lhs, BufferFloat rhs, BufferFloat output) {
            auto index = dispatch_id().x;
            Callable read_lhs = [&lhs](UInt i) { return lhs.read(i); };
            Callable read_rhs = [&rhs](UInt i) { return rhs.read(i); };
            output.write(index, read_lhs(index) + read_rhs(index));
        };
        auto module = ast_to_xir_translate(kernel.function()->function(), {});
        expect(module != nullptr);
        auto *translated_callable = find_only_callable(module.get());
        auto *translated_kernel = find_kernel_definition(module.get());
        expect(translated_callable != nullptr);
        expect(translated_kernel != nullptr);
        expect(count_calls_to(translated_kernel, translated_callable) == 2u);

        auto resource_argument_index = static_cast<size_t>(-1);
        if (translated_callable != nullptr) {
            auto index = 0u;
            for (auto *argument : translated_callable->arguments()) {
                if (argument->is_resource()) {
                    resource_argument_index = index;
                    break;
                }
                index++;
            }
        }
        expect(resource_argument_index != static_cast<size_t>(-1));

        luisa::vector<const CallInst *> calls;
        if (translated_kernel != nullptr) {
            translated_kernel->traverse_instructions(
                [&](const Instruction *instruction) noexcept {
                    if (!instruction->isa<CallInst>()) { return; }
                    auto *call = static_cast<const CallInst *>(instruction);
                    if (call->callee() == translated_callable) {
                        calls.emplace_back(call);
                    }
                });
        }
        expect(calls.size() == 2u);
        if (calls.size() == 2u &&
            resource_argument_index != static_cast<size_t>(-1)) {
            expect(calls[0]->argument(resource_argument_index) !=
                   calls[1]->argument(resource_argument_index))
                << "definition deduplication must retain each call site's "
                   "captured resource";
        }
        expect(xir_verify_module(module.get()).succeeded());
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

    "xir_ast_to_xir_preserves_complete_suspend_extension"_test = [] {
        Coroutine c = [](Var<uint> key) {
            $suspend("shade_surface", coro_sort_by(key, 1024u));
        };
        auto module = ast_to_xir_translate(
            c.function_builder()->function(), {});
        expect(module != nullptr);
        expect(xir_verify_module(module.get()).succeeded());
        const CoroSuspendInst *suspend = nullptr;
        for (auto *function : module->function_list()) {
            if (auto *definition = function->definition()) {
                definition->traverse_instructions(
                    [&](const Instruction *instruction) noexcept {
                        if (instruction->isa<CoroSuspendInst>()) {
                            suspend = static_cast<const CoroSuspendInst *>(
                                instruction);
                        }
                    });
            }
        }
        expect(suspend != nullptr);
        if (suspend != nullptr) {
            expect(suspend->extensions().size() == 1u);
            expect(suspend->extension_binding_value_count() == 1u);
            if (suspend->extensions().size() == 1u) {
                auto &&extension = suspend->extensions().front();
                expect(extension->schema() ==
                       "luisa.coro.schedule.sort");
                expect(extension->version() == 1u);
                expect(extension->is_annotation());
                expect(extension->fallback() ==
                       CoroSuspendFallback::ignore);
                expect(extension->bindings().size() == 1u);
                expect(extension->attributes().size() == 1u);
                expect(extension->bindings().front().name == "key");
                expect(extension->bindings().front().index == 0u);
                expect(suspend->extension_binding_value(0u)->type() ==
                       Type::of<uint>());
                expect(luisa::get<uint64_t>(
                           extension->attributes().front().value) ==
                       1024u);
            }
        }
        auto text = xir_to_text_translate(module.get(), true);
        expect(text.find("luisa.coro.schedule.sort") !=
               luisa::string::npos);
        expect(text.find("attribute \"range\"") !=
               luisa::string::npos);
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
