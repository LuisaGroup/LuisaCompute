#include "ut/ut.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <limits>

#if __has_include(<unistd.h>) && __has_include(<sys/wait.h>)
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <luisa/luisa-compute.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/mem2reg.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2ast.h>
#include <luisa/xir/translators/xir2text.h>
#include <luisa/xir/verifier.h>

#include "../../../backends/common/xir_autodiff.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct StatementCounter final : StmtVisitor {
    uint stores = 0u;
    uint returns = 0u;
    uint breaks = 0u;
    uint continues = 0u;
    uint ifs = 0u;
    uint loops = 0u;
    uint fors = 0u;
    uint switches = 0u;
    uint exprs = 0u;
    uint prints = 0u;

    void visit(const BreakStmt *) override { breaks++; }
    void visit(const ContinueStmt *) override { continues++; }
    void visit(const ReturnStmt *) override { returns++; }
    void visit(const ScopeStmt *stmt) override {
        for (auto s : stmt->statements()) { s->accept(*this); }
    }
    void visit(const IfStmt *stmt) override {
        ifs++;
        stmt->true_branch()->accept(*this);
        stmt->false_branch()->accept(*this);
    }
    void visit(const LoopStmt *stmt) override {
        loops++;
        stmt->body()->accept(*this);
    }
    void visit(const ExprStmt *) override { exprs++; }
    void visit(const SwitchStmt *stmt) override {
        switches++;
        stmt->body()->accept(*this);
    }
    void visit(const SwitchCaseStmt *stmt) override { stmt->body()->accept(*this); }
    void visit(const SwitchDefaultStmt *stmt) override { stmt->body()->accept(*this); }
    void visit(const AssignStmt *) override { stores++; }
    void visit(const ForStmt *stmt) override {
        fors++;
        stmt->body()->accept(*this);
    }
    void visit(const CommentStmt *) override {}
    void visit(const RayQueryStmt *stmt) override {
        stmt->on_triangle_candidate()->accept(*this);
        stmt->on_procedural_candidate()->accept(*this);
    }
    void visit(const SuspendStmt *) override {}
    void visit(const AutoDiffStmt *stmt) override { stmt->body()->accept(*this); }
    void visit(const PrintStmt *) override { prints++; }
    void visit(const DebugBreakStmt *) override {}
};

[[nodiscard]] auto first_definition(Module *module) noexcept {
    for (auto *f : module->function_list()) {
        if (f->is_definition()) { return static_cast<FunctionDefinition *>(f); }
    }
    return static_cast<FunctionDefinition *>(nullptr);
}

[[nodiscard]] auto first_kernel_definition(Module *module) noexcept {
    for (auto *f : module->function_list()) {
        if (f->derived_function_tag() == DerivedFunctionTag::KERNEL) { return static_cast<FunctionDefinition *>(f); }
    }
    return static_cast<FunctionDefinition *>(nullptr);
}

#if __has_include(<unistd.h>) && __has_include(<sys/wait.h>)
template<typename F>
[[nodiscard]] bool terminates_with_abort(F &&f) noexcept {
    auto pid = fork();
    if (pid < 0) { return false; }
    if (pid == 0) {
        f();
        _exit(0);
    }
    auto status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) { return false; }
    }
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}
#endif

}// namespace

void reg_xir2ast_direct() {

    "xir_to_ast_roundtrips_canonical_low_level_ray_query_state"_test = [] {
        Module module;
        auto *callable = module.create_callable(nullptr);
        auto *query = callable->create_reference_argument(
            Type::of<RayQueryAll>());
        auto *distance = callable->create_value_argument(Type::of<float>());
        auto *body = callable->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);

        static_cast<void>(b.call(
            Type::of<Ray>(),
            RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY,
            {query}));
        static_cast<void>(b.call(
            Type::of<ProceduralHit>(),
            RayQueryObjectReadOp::
                RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT,
            {query}));
        static_cast<void>(b.call(
            Type::of<TriangleHit>(),
            RayQueryObjectReadOp::
                RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT,
            {query}));
        static_cast<void>(b.call(
            Type::of<CommittedHit>(),
            RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT,
            {query}));
        static_cast<void>(b.call(
            Type::of<bool>(),
            RayQueryObjectReadOp::
                RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE,
            {query}));
        static_cast<void>(b.call(
            Type::of<bool>(),
            RayQueryObjectReadOp::
                RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE,
            {query}));
        b.call(
            RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE,
            {query});
        b.call(
            RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL,
            {query, distance});
        b.call(
            RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE,
            {query});

        // XIR splits AST proceed() into an adjacent state transition and a
        // read of the complementary termination bit. This exact pair is the
        // representable inverse boundary for XIR-to-AST.
        b.call(RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED,
               {query});
        auto *terminated = b.call(
            Type::of<bool>(),
            RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED,
            {query});
        auto *active = b.call(
            Type::of<bool>(), ArithmeticOp::UNARY_BIT_NOT,
            {terminated});
        b.assume_(active, "canonical proceed pair remains active");
        b.return_void();

        auto ast = xir_to_ast_translate(*callable, {});
        expect(ast != nullptr);
        auto rebuilt = ast_to_xir_translate(ast->function(), {});
        expect(rebuilt != nullptr);
        expect(xir_verify_module(rebuilt.get()).succeeded());
        // XIR-to-AST materializes stateful calls into AST locals. Promote those
        // transport-only temporaries so the semantic polarity can be inspected
        // on the rebuilt SSA value rather than through incidental loads/stores.
        static_cast<void>(mem2reg_pass_run_on_module(rebuilt.get()));

        std::array<size_t, 7u> read_counts{};
        std::array<size_t, 4u> write_counts{};
        const AssumeInst *rebuilt_assume = nullptr;
        for (auto *function : rebuilt->function_list()) {
            auto *definition = function->definition();
            if (definition == nullptr) { continue; }
            definition->traverse_instructions(
                [&](Instruction *instruction) noexcept {
                    if (instruction->isa<RayQueryObjectReadInst>()) {
                        auto op = static_cast<RayQueryObjectReadInst *>(
                                      instruction)
                                      ->op();
                        read_counts[luisa::to_underlying(op)]++;
                    } else if (instruction
                                   ->isa<RayQueryObjectWriteInst>()) {
                        auto op = static_cast<RayQueryObjectWriteInst *>(
                                      instruction)
                                      ->op();
                        write_counts[luisa::to_underlying(op)]++;
                    } else if (instruction->isa<AssumeInst>()) {
                        rebuilt_assume = static_cast<const AssumeInst *>(
                            instruction);
                    }
                });
        }
        for (auto count : read_counts) { expect(count == 1u); }
        for (auto count : write_counts) { expect(count == 1u); }
        // The assumed value was `active = !terminated`. Follow the rebuilt
        // unary chain back to the termination read and verify odd parity. This
        // catches the tempting but incorrect mapping of AST proceed() directly
        // to XIR is_terminated, even though both shapes contain the same number
        // of ray-query reads and writes.
        expect(rebuilt_assume != nullptr);
        auto *value = rebuilt_assume->condition();
        auto negation_count = 0u;
        while (value->isa<ArithmeticInst>()) {
            auto *arithmetic = static_cast<const ArithmeticInst *>(value);
            if (arithmetic->op() != ArithmeticOp::UNARY_BIT_NOT) { break; }
            negation_count++;
            value = arithmetic->operand(0u);
        }
        expect((negation_count & 1u) == 1u);
        expect(value->isa<RayQueryObjectReadInst>());
        if (value->isa<RayQueryObjectReadInst>()) {
            expect(static_cast<const RayQueryObjectReadInst *>(value)->op() ==
                   RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED);
        }
    };

    "xir_to_ast_direct_memory_and_resource_kernel"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        kernel->set_block_size(make_uint3(64u, 1u, 1u));
        auto *buffer = kernel->create_resource_argument(Type::buffer(Type::of<float>()));
        auto *body = kernel->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *local = b.alloca_local(Type::of<float>());
        auto *idx = module.create_dispatch_id();
        auto *zero = module.create_constant_zero(Type::of<uint>());
        auto *x = b.call(Type::of<uint>(), ArithmeticOp::EXTRACT, {idx, zero});
        auto *read = b.call(Type::of<float>(), ResourceReadOp::BUFFER_READ, {buffer, x});
        float two = 2.0f;
        auto *scale = module.create_constant(Type::of<float>(), &two);
        auto *mul = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {read, scale});
        b.store(local, mul);
        auto *load = b.load(Type::of<float>(), local);
        b.call(ResourceWriteOp::BUFFER_WRITE, {buffer, x, load});
        b.return_void();

        auto ast = xir_to_ast_translate(*kernel, {});
        expect(ast != nullptr);
        auto block_size = ast->block_size();
        expect(block_size.x == 64u);
        expect(block_size.y == 1u);
        expect(block_size.z == 1u);
        expect(ast->arguments().size() == 1u);
        expect(ast->local_variables().size() == 3u);
        StatementCounter counter;
        ast->body()->accept(counter);
        expect(counter.stores >= 1u);
        expect(counter.exprs >= 1u);
        expect(counter.returns == 1u);

        auto roundtrip = ast_to_xir_translate(ast->function(), {});
        expect(roundtrip != nullptr);
        auto text = xir_to_text_translate(roundtrip.get(), false);
        expect(text.find("resource_write buffer_write") != string::npos);
    };

    "xir_to_ast_direct_preserves_unused_atomic"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *buffer = kernel->create_resource_argument(Type::buffer(Type::of<int>()));
        auto *body = kernel->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        uint32_t index_value = 0u;
        int increment_value = 1;
        auto *index = module.create_constant(Type::of<uint32_t>(), &index_value);
        auto *increment = module.create_constant(Type::of<int>(), &increment_value);
        std::array<Value *, 1u> indices{index};
        b.atomic_fetch_add(Type::of<int>(), buffer, luisa::span<Value *const>{indices}, increment);
        b.return_void();

        auto ast = xir_to_ast_translate(*kernel, {});
        expect(ast != nullptr);
        expect(ast->arguments().size() == 1u);
        expect(ast->variable_usage(ast->arguments().front().uid()) != Usage::NONE);
        StatementCounter counter;
        ast->body()->accept(counter);
        expect(counter.stores == 1u);

        auto roundtrip = ast_to_xir_translate(ast->function(), {});
        expect(roundtrip != nullptr);
        expect(xir_verify_module(roundtrip.get()).succeeded());
        auto *roundtrip_kernel = first_kernel_definition(roundtrip.get());
        expect(roundtrip_kernel != nullptr);
        auto atomic_count = 0u;
        for (auto *block : roundtrip_kernel->basic_blocks()) {
            for (auto *inst : block->instructions()) {
                atomic_count += inst->isa<AtomicInst>() && static_cast<AtomicInst *>(inst)->op() == AtomicOp::FETCH_ADD;
            }
        }
        expect(atomic_count == 1u);
    };

    "xir_to_ast_preserves_bindless_byte_size_units"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *bindless =
            kernel->create_resource_argument(Type::of<BindlessArray>());
        auto *output = kernel->create_resource_argument(
            Type::buffer(Type::of<uint64_t>()));
        auto *body = kernel->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        uint32_t slot_value = 7u;
        uint32_t output_index_value = 0u;
        auto *slot = module.create_constant(
            Type::of<uint32_t>(), &slot_value);
        auto *output_index = module.create_constant(
            Type::of<uint32_t>(), &output_index_value);
        auto *size = b.call(
            Type::of<uint64_t>(),
            ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE,
            {bindless, slot});
        b.call(ResourceWriteOp::BUFFER_WRITE,
               {output, output_index, size});
        b.return_void();

        auto ast = xir_to_ast_translate(*kernel, {});
        expect(ast != nullptr);
        auto roundtrip = ast_to_xir_translate(ast->function(), {});
        expect(roundtrip != nullptr);
        auto *roundtrip_kernel = first_kernel_definition(roundtrip.get());
        expect(roundtrip_kernel != nullptr);

        auto size_query_count = 0u;
        for (auto *block : roundtrip_kernel->basic_blocks()) {
            for (auto *inst : block->instructions()) {
                if (!inst->isa<ResourceQueryInst>()) { continue; }
                auto *query = static_cast<ResourceQueryInst *>(inst);
                if (query->op() != ResourceQueryOp::BINDLESS_BUFFER_SIZE) {
                    continue;
                }
                size_query_count++;
                expect(query->operand_count() == 3u);
                if (query->operand_count() == 3u) {
                    auto *stride = query->operand(2u);
                    expect(stride->isa<xir::Constant>());
                    if (stride->isa<xir::Constant>()) {
                        auto *constant =
                            static_cast<xir::Constant *>(stride);
                        expect(constant->type()->is_uint32());
                        expect(constant->as<uint32_t>() == 1u)
                            << "bindless byte size must round-trip with a one-byte stride";
                    }
                }
            }
        }
        expect(size_query_count == 1u);
    };

    "xir_to_ast_preserves_bindless_access_axes"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *bindless =
            kernel->create_resource_argument(Type::of<BindlessArray>());
        auto *output = kernel->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        auto *body = kernel->create_body_block();
        auto *slot = module.create_constant_zero(Type::of<uint32_t>());
        auto *stride = module.create_constant_one(Type::of<uint32_t>());
        XIRBuilder b;
        b.set_insertion_point(body);
        std::array accesses{
            BindlessResourceAccess{.typed = true, .uniform = true},
            BindlessResourceAccess{.typed = true, .uniform = false},
            BindlessResourceAccess{.typed = false, .uniform = true}};
        for (auto i = 0u; i < accesses.size(); ++i) {
            auto *size = b.call(
                Type::of<uint32_t>(),
                ResourceQueryOp::BINDLESS_BUFFER_SIZE,
                {bindless, slot, stride}, accesses[i]);
            auto *index = module.create_constant(
                Type::of<uint32_t>(), &i);
            b.call(ResourceWriteOp::BUFFER_WRITE,
                   {output, index, size});
        }
        b.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto ast = xir_to_ast_translate(*kernel, {});
        expect(ast != nullptr);
        auto roundtrip = ast_to_xir_translate(ast->function(), {});
        expect(roundtrip != nullptr);
        expect(xir_verify_module(roundtrip.get()).succeeded());

        std::array found{false, false, false};
        auto *roundtrip_kernel = first_kernel_definition(roundtrip.get());
        expect(roundtrip_kernel != nullptr);
        roundtrip_kernel->traverse_instructions(
            [&](const Instruction *instruction) noexcept {
                if (!instruction->isa<ResourceQueryInst>()) { return; }
                auto *query = static_cast<const ResourceQueryInst *>(
                    instruction);
                if (query->op() !=
                    ResourceQueryOp::BINDLESS_BUFFER_SIZE) {
                    return;
                }
                for (auto i = 0u; i < accesses.size(); ++i) {
                    found[i] = found[i] ||
                               query->bindless_access() == accesses[i];
                }
            });
        expect(found[0] && found[1] && found[2]);
    };

#if __has_include(<unistd.h>) && __has_include(<sys/wait.h>)
    "xir_to_ast_rejects_unrepresentable_bindless_byte_write"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *bindless =
            kernel->create_resource_argument(Type::of<BindlessArray>());
        auto *body = kernel->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        uint32_t slot_value = 2u;
        uint64_t byte_offset_value = 3u;
        uint32_t payload_value = 0x12345678u;
        auto *slot = module.create_constant(
            Type::of<uint32_t>(), &slot_value);
        auto *byte_offset = module.create_constant(
            Type::of<uint64_t>(), &byte_offset_value);
        auto *payload = module.create_constant(
            Type::of<uint32_t>(), &payload_value);
        b.call(ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE,
               {bindless, slot, byte_offset, payload});
        b.return_void();

        expect(terminates_with_abort([&] {
            static_cast<void>(xir_to_ast_translate(*kernel, {}));
        })) << "XIR-to-AST must reject byte-addressed bindless writes "
               "instead of reinterpreting byte offsets as element indices";
    };
#endif

    "xir_to_ast_direct_materializes_dynamic_values_once"_test = [] {
        Module module;
        auto *int_type = Type::of<int>();
        auto *uint_type = Type::of<uint32_t>();
        auto *ulong_type = Type::of<uint64_t>();
        uint32_t index_value = 0u;
        int one_value = 1;
        uint64_t zero_ulong_value = 0u;
        auto *index = module.create_constant(uint_type, &index_value);
        auto *one = module.create_constant(int_type, &one_value);
        auto *zero_ulong = module.create_constant(ulong_type, &zero_ulong_value);

        auto *callee = module.create_callable(int_type);
        auto *callee_buffer = callee->create_resource_argument(Type::buffer(int_type));
        auto *callee_body = callee->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        b.call(ResourceWriteOp::BUFFER_WRITE, {callee_buffer, index, one});
        b.return_(one);

        auto *kernel = module.create_kernel();
        auto *buffer = kernel->create_resource_argument(Type::buffer(int_type));
        auto *clock_buffer = kernel->create_resource_argument(Type::buffer(ulong_type));
        auto *body = kernel->create_body_block();
        b.set_insertion_point(body);
        auto *read = b.call(int_type, ResourceReadOp::BUFFER_READ, {buffer, index});
        b.call(ResourceWriteOp::BUFFER_WRITE, {buffer, index, one});
        auto *read_twice = b.call(int_type, ArithmeticOp::BINARY_ADD, {read, read});
        b.call(ResourceWriteOp::BUFFER_WRITE, {buffer, index, read_twice});
        static_cast<void>(b.call(int_type, ResourceReadOp::BUFFER_VOLATILE_READ, {buffer, index}));
        std::array<Value *, 1u> indices{index};
        auto *atomic = b.atomic_fetch_add(int_type, buffer, luisa::span<Value *const>{indices}, one);
        auto *atomic_twice = b.call(int_type, ArithmeticOp::BINARY_ADD, {atomic, atomic});
        b.call(ResourceWriteOp::BUFFER_WRITE, {buffer, index, atomic_twice});
        auto *warp = b.call(int_type, ThreadGroupOp::WARP_ACTIVE_SUM, {one});
        auto *warp_twice = b.call(int_type, ArithmeticOp::BINARY_ADD, {warp, warp});
        b.call(ResourceWriteOp::BUFFER_WRITE, {buffer, index, warp_twice});
        static_cast<void>(b.call(int_type, ThreadGroupOp::WARP_ACTIVE_MAX, {one}));
        auto *clock = b.clock();
        b.call(ResourceWriteOp::BUFFER_VOLATILE_WRITE, {clock_buffer, index, zero_ulong});
        auto *clock_twice = b.call(ulong_type, ArithmeticOp::BINARY_ADD, {clock, clock});
        b.call(ResourceWriteOp::BUFFER_WRITE, {clock_buffer, index, clock_twice});
        auto *call = b.call(int_type, callee, {buffer});
        auto *call_twice = b.call(int_type, ArithmeticOp::BINARY_ADD, {call, call});
        b.call(ResourceWriteOp::BUFFER_WRITE, {buffer, index, call_twice});
        static_cast<void>(b.call(int_type, callee, {buffer}));
        b.return_void();

        auto ast = xir_to_ast_translate(*kernel, {});
        expect(ast != nullptr);
        auto roundtrip = ast_to_xir_translate(ast->function(), {});
        expect(roundtrip != nullptr);
        expect(xir_verify_module(roundtrip.get()).succeeded());
        auto *roundtrip_kernel = first_kernel_definition(roundtrip.get());
        expect(roundtrip_kernel != nullptr);
        auto buffer_read_count = 0u;
        auto volatile_read_count = 0u;
        auto atomic_count = 0u;
        auto warp_sum_count = 0u;
        auto warp_max_count = 0u;
        auto clock_count = 0u;
        auto call_count = 0u;
        size_t instruction_index = 0u;
        auto first_buffer_read_index = std::numeric_limits<size_t>::max();
        auto first_buffer_write_index = std::numeric_limits<size_t>::max();
        auto clock_index = std::numeric_limits<size_t>::max();
        auto volatile_write_index = std::numeric_limits<size_t>::max();
        for (auto *block : roundtrip_kernel->basic_blocks()) {
            for (auto *inst : block->instructions()) {
                if (inst->isa<ResourceReadInst>()) {
                    auto op = static_cast<ResourceReadInst *>(inst)->op();
                    if (op == ResourceReadOp::BUFFER_READ) {
                        buffer_read_count++;
                        first_buffer_read_index = std::min(first_buffer_read_index, instruction_index);
                    } else if (op == ResourceReadOp::BUFFER_VOLATILE_READ) {
                        volatile_read_count++;
                    }
                } else if (inst->isa<ResourceWriteInst>()) {
                    if (static_cast<ResourceWriteInst *>(inst)->op() == ResourceWriteOp::BUFFER_VOLATILE_WRITE) {
                        volatile_write_index = std::min(volatile_write_index, instruction_index);
                    }
                    first_buffer_write_index = std::min(first_buffer_write_index, instruction_index);
                } else if (inst->isa<AtomicInst>()) {
                    atomic_count += static_cast<AtomicInst *>(inst)->op() == AtomicOp::FETCH_ADD;
                } else if (inst->isa<ThreadGroupInst>()) {
                    auto op = static_cast<ThreadGroupInst *>(inst)->op();
                    warp_sum_count += op == ThreadGroupOp::WARP_ACTIVE_SUM;
                    warp_max_count += op == ThreadGroupOp::WARP_ACTIVE_MAX;
                } else if (inst->isa<ClockInst>()) {
                    clock_count++;
                    clock_index = std::min(clock_index, instruction_index);
                } else if (inst->isa<CallInst>()) {
                    call_count++;
                }
                instruction_index++;
            }
        }
        expect(buffer_read_count == 1u);
        expect(volatile_read_count == 1u);
        expect(atomic_count == 1u);
        expect(warp_sum_count == 1u);
        expect(warp_max_count == 1u);
        expect(clock_count == 1u);
        expect(call_count == 2u);
        expect(first_buffer_read_index < first_buffer_write_index);
        expect(clock_index < volatile_write_index);
    };

    "xir_to_ast_direct_emits_void_effects_once"_test = [] {
        Module module;
        auto *void_callable = module.create_callable(nullptr);
        auto *callable_body = void_callable->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callable_body);
        b.return_void();

        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        bool condition_value = true;
        auto *condition = module.create_constant(Type::of<bool>(), &condition_value);
        b.set_insertion_point(body);
        b.assert_(condition, "assert");
        b.assume_(condition, "assume");
        b.synchronize_block();
        b.call(nullptr, void_callable, {});
        b.return_void();

        auto ast = xir_to_ast_translate(*kernel, {});
        expect(ast != nullptr);
        auto roundtrip = ast_to_xir_translate(ast->function(), {});
        expect(roundtrip != nullptr);
        expect(xir_verify_module(roundtrip.get()).succeeded());
        auto *roundtrip_kernel = first_kernel_definition(roundtrip.get());
        expect(roundtrip_kernel != nullptr);
        auto assert_count = 0u;
        auto assume_count = 0u;
        auto sync_count = 0u;
        auto call_count = 0u;
        for (auto *block : roundtrip_kernel->basic_blocks()) {
            for (auto *inst : block->instructions()) {
                assert_count += inst->isa<AssertInst>();
                assume_count += inst->isa<AssumeInst>();
                sync_count += inst->isa<ThreadGroupInst>() && static_cast<ThreadGroupInst *>(inst)->op() == ThreadGroupOp::SYNCHRONIZE_BLOCK;
                call_count += inst->isa<CallInst>();
            }
        }
        expect(assert_count == 1u);
        expect(assume_count == 1u);
        expect(sync_count == 1u);
        expect(call_count == 1u);
    };

    "xir_to_ast_direct_continue_executes_loop_update"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *body = kernel->create_body_block();
        auto *uint_type = Type::of<uint32_t>();
        auto *zero = module.create_constant_zero(uint_type);
        auto *one = module.create_constant_one(uint_type);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *counter = b.alloca_local(uint_type);
        auto *mirror = b.alloca_local(uint_type);
        b.store(counter, zero);
        b.store(mirror, zero);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        b.cond_br(condition, loop_body, merge);
        b.set_insertion_point(loop_body);
        auto *if_inst = b.if_(condition);
        auto *true_block = if_inst->create_true_block();
        auto *false_block = if_inst->create_false_block();
        b.set_insertion_point(true_block);
        b.continue_(update);
        b.set_insertion_point(false_block);
        b.break_(merge);
        b.set_insertion_point(update);
        auto *value = b.load(uint_type, counter);
        auto *next = b.call(uint_type, ArithmeticOp::BINARY_ADD, {value, one});
        b.store(counter, next);
        auto *mirror_value = b.load(uint_type, mirror);
        auto *mirror_next = b.call(uint_type, ArithmeticOp::BINARY_ADD, {mirror_value, one});
        b.store(mirror, mirror_next);
        b.br(prepare);
        b.set_insertion_point(merge);
        b.return_void();

        auto ast = xir_to_ast_translate(*kernel, {});
        expect(ast != nullptr);
        auto roundtrip = ast_to_xir_translate(ast->function(), {});
        expect(roundtrip != nullptr);
        expect(xir_verify_module(roundtrip.get()).succeeded());
        auto *roundtrip_kernel = first_kernel_definition(roundtrip.get());
        expect(roundtrip_kernel != nullptr);
        auto continue_count = 0u;
        auto updated_continue_count = 0u;
        auto simple_loop_count = 0u;
        for (auto *block : roundtrip_kernel->basic_blocks()) {
            auto saw_add = false;
            auto saw_store = false;
            for (auto *inst : block->instructions()) {
                simple_loop_count += inst->isa<SimpleLoopInst>();
                saw_add |= inst->isa<ArithmeticInst>() && static_cast<ArithmeticInst *>(inst)->op() == ArithmeticOp::BINARY_ADD;
                saw_store |= inst->isa<StoreInst>();
                if (inst->isa<ContinueInst>()) {
                    continue_count++;
                    auto *target = static_cast<ContinueInst *>(inst)->target_block();
                    expect(target != nullptr);
                    if (target == nullptr) { continue; }
                    auto target_has_add = false;
                    auto target_has_store = false;
                    for (auto *target_inst : target->instructions()) {
                        target_has_add |= target_inst->isa<ArithmeticInst>() && static_cast<ArithmeticInst *>(target_inst)->op() == ArithmeticOp::BINARY_ADD;
                        target_has_store |= target_inst->isa<StoreInst>();
                    }
                    updated_continue_count += (saw_add && saw_store) || (target_has_add && target_has_store);
                }
            }
        }
        expect(simple_loop_count == 1u);
        expect(continue_count == 1u);
        expect(updated_continue_count == 1u);
    };

    "xir_to_ast_direct_preserves_nearest_break_continue_scopes"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *selector = kernel->create_value_argument(Type::of<int32_t>());
        auto *body = kernel->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        b.cond_br(condition, loop_body, loop_merge);
        b.set_insertion_point(loop_body);
        auto *switch_inst = b.switch_(selector);
        auto *case_block = switch_inst->create_case_block(0);
        auto *default_block = switch_inst->create_default_block();
        auto *switch_merge = switch_inst->create_merge_block();
        b.set_insertion_point(case_block);
        b.break_(switch_merge);
        b.set_insertion_point(default_block);
        b.continue_(update);
        b.set_insertion_point(switch_merge);
        b.break_(loop_merge);
        b.set_insertion_point(update);
        b.br(prepare);
        b.set_insertion_point(loop_merge);
        b.return_void();

        auto ast = xir_to_ast_translate(*kernel, {});
        expect(ast != nullptr);
        StatementCounter counter;
        ast->body()->accept(counter);
        expect(counter.loops == 1u);
        expect(counter.switches == 1u);
        expect(counter.breaks == 3u);
        expect(counter.continues == 1u);

        auto roundtrip = ast_to_xir_translate(ast->function(), {});
        expect(roundtrip != nullptr);
        expect(xir_verify_module(
                   roundtrip.get(),
                   {.require_canonical_break_continue_targets = true})
                   .succeeded());
    };

    "xir_to_ast_roundtrip_preserves_u64_switch_case_bits"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *selector = kernel->create_value_argument(Type::of<uint64_t>());
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *switch_inst = builder.switch_(selector);
        constexpr auto low_word_ones = uint64_t{0x00000000ffffffffull};
        constexpr auto all_ones = uint64_t{0xffffffffffffffffull};
        auto *low_word_block = switch_inst->create_case_block(low_word_ones);
        auto *all_ones_block = switch_inst->create_case_block(all_ones);
        auto *default_block = switch_inst->create_default_block();
        auto *merge_block = switch_inst->create_merge_block();
        for (auto *block : {low_word_block, all_ones_block, default_block}) {
            builder.set_insertion_point(block);
            builder.br(merge_block);
        }
        builder.set_insertion_point(merge_block);
        builder.return_void();

        auto ast = xir_to_ast_translate(*kernel, {});
        expect(ast != nullptr);
        if (ast == nullptr) { return; }
        auto roundtrip = ast_to_xir_translate(ast->function(), {});
        expect(roundtrip != nullptr);
        auto *definition = first_kernel_definition(roundtrip.get());
        expect(definition != nullptr);
        if (definition == nullptr) { return; }
        auto found = false;
        definition->traverse_instructions([&](Instruction *instruction) noexcept {
            if (!instruction->isa<SwitchInst>()) { return; }
            auto *value = static_cast<SwitchInst *>(instruction);
            expect(value->value()->type() == Type::of<uint64_t>());
            expect(value->case_count() == 2u);
            expect(value->case_value(0u) == low_word_ones);
            expect(value->case_value(1u) == all_ones);
            found = true;
        });
        expect(found);
    };

    "xir_to_ast_roundtrip_preserves_rint_semantics"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *buffer = kernel->create_resource_argument(
            Type::buffer(Type::of<float>()));
        auto *vector_buffer = kernel->create_resource_argument(
            Type::buffer(Type::of<float4>()));
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *zero = module.create_constant_zero(Type::of<uint32_t>());
        auto *input = builder.call(
            Type::of<float>(), ResourceReadOp::BUFFER_READ,
            {buffer, zero});
        auto *rounded = builder.call(
            Type::of<float>(), ArithmeticOp::RINT, {input});
        builder.call(
            ResourceWriteOp::BUFFER_WRITE,
            {buffer, zero, rounded});
        auto *vector_input = builder.call(
            Type::of<float4>(), ResourceReadOp::BUFFER_READ,
            {vector_buffer, zero});
        auto *vector_rounded = builder.call(
            Type::of<float4>(), ArithmeticOp::RINT, {vector_input});
        builder.call(
            ResourceWriteOp::BUFFER_WRITE,
            {vector_buffer, zero, vector_rounded});
        builder.return_void();

        auto ast = xir_to_ast_translate(*kernel, {});
        expect(ast != nullptr);
        if (ast == nullptr) { return; }
        auto roundtrip = ast_to_xir_translate(ast->function(), {});
        expect(roundtrip != nullptr);
        expect(xir_verify_module(roundtrip.get()).succeeded());
        auto *definition = first_kernel_definition(roundtrip.get());
        expect(definition != nullptr);
        if (definition == nullptr) { return; }
        auto rint_count = 0u;
        auto round_count = 0u;
        definition->traverse_instructions(
            [&](Instruction *instruction) noexcept {
                if (!instruction->isa<ArithmeticInst>()) { return; }
                auto op =
                    static_cast<ArithmeticInst *>(instruction)->op();
                rint_count += op == ArithmeticOp::RINT;
                round_count += op == ArithmeticOp::ROUND;
            });
        expect(rint_count == 2u);
        expect(round_count == 0u);
    };

    "xir_to_ast_roundtrip_preserves_matrix_arithmetic_partition"_test = [] {
        Module module;
        auto *callable = module.create_callable(nullptr);
        auto *matrix = callable->create_value_argument(
            Type::of<float4x4>());
        auto *other_matrix = callable->create_value_argument(
            Type::of<float4x4>());
        auto *vector = callable->create_value_argument(
            Type::of<float4>());
        // Pure, unused AST expressions are intentionally not statements. Make
        // every operation externally observable so the round trip checks the
        // opcode partition rather than dead-expression retention.
        auto *neg_output = callable->create_reference_argument(
            Type::of<float4x4>());
        auto *add_output = callable->create_reference_argument(
            Type::of<float4x4>());
        auto *sub_output = callable->create_reference_argument(
            Type::of<float4x4>());
        auto *component_mul_output = callable->create_reference_argument(
            Type::of<float4x4>());
        auto *div_output = callable->create_reference_argument(
            Type::of<float4x4>());
        auto *matrix_vector_output = callable->create_reference_argument(
            Type::of<float4>());
        auto *vector_matrix_output = callable->create_reference_argument(
            Type::of<float4>());
        auto *matrix_matrix_output = callable->create_reference_argument(
            Type::of<float4x4>());
        auto *body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.store(neg_output, builder.call(
                                      Type::of<float4x4>(),
                                      ArithmeticOp::MATRIX_COMP_NEG, {matrix}));
        builder.store(add_output, builder.call(
                                      Type::of<float4x4>(),
                                      ArithmeticOp::MATRIX_COMP_ADD,
                                      {matrix, other_matrix}));
        builder.store(sub_output, builder.call(
                                      Type::of<float4x4>(),
                                      ArithmeticOp::MATRIX_COMP_SUB,
                                      {matrix, other_matrix}));
        builder.store(component_mul_output, builder.call(
                                                Type::of<float4x4>(),
                                                ArithmeticOp::MATRIX_COMP_MUL,
                                                {matrix, other_matrix}));
        builder.store(div_output, builder.call(
                                      Type::of<float4x4>(),
                                      ArithmeticOp::MATRIX_COMP_DIV,
                                      {matrix, other_matrix}));
        builder.store(matrix_vector_output, builder.call(
                                                Type::of<float4>(),
                                                ArithmeticOp::MATRIX_LINALG_MUL,
                                                {matrix, vector}));
        builder.store(vector_matrix_output, builder.call(
                                                Type::of<float4>(),
                                                ArithmeticOp::MATRIX_LINALG_MUL,
                                                {vector, matrix}));
        builder.store(matrix_matrix_output, builder.call(
                                                Type::of<float4x4>(),
                                                ArithmeticOp::MATRIX_LINALG_MUL,
                                                {matrix, other_matrix}));
        builder.return_void();

        auto ast = xir_to_ast_translate(*callable, {});
        expect(ast != nullptr);
        if (ast == nullptr) { return; }
        auto rebuilt = ast_to_xir_translate(
            ast->function(), {});
        expect(rebuilt != nullptr);
        expect(xir_verify_module(rebuilt.get()).succeeded());

        std::array<size_t, 6u> counts{};
        for (auto *function : rebuilt->function_list()) {
            auto *definition = function->definition();
            if (definition == nullptr) { continue; }
            definition->traverse_instructions(
                [&](Instruction *instruction) noexcept {
                    if (!instruction->isa<ArithmeticInst>()) {
                        return;
                    }
                    switch (static_cast<ArithmeticInst *>(
                                instruction)
                                ->op()) {
                        case ArithmeticOp::MATRIX_COMP_NEG:
                            ++counts[0u];
                            break;
                        case ArithmeticOp::MATRIX_COMP_ADD:
                            ++counts[1u];
                            break;
                        case ArithmeticOp::MATRIX_COMP_SUB:
                            ++counts[2u];
                            break;
                        case ArithmeticOp::MATRIX_COMP_MUL:
                            ++counts[3u];
                            break;
                        case ArithmeticOp::MATRIX_COMP_DIV:
                            ++counts[4u];
                            break;
                        case ArithmeticOp::MATRIX_LINALG_MUL:
                            ++counts[5u];
                            break;
                        default: break;
                    }
                });
        }
        expect(counts[0u] == 1u);
        expect(counts[1u] == 1u);
        expect(counts[2u] == 1u);
        expect(counts[3u] == 1u);
        expect(counts[4u] == 1u);
        expect(counts[5u] == 3u);
    };

    "xir_to_ast_emits_loop_update_for_boundary_selection_arm"_test = [] {
        Module module;
        auto *callable = module.create_callable(nullptr);
        auto *loop_condition = callable->create_value_argument(
            Type::of<bool>());
        auto *take_body = callable->create_value_argument(
            Type::of<bool>());
        auto *entry = callable->create_body_block();
        auto *prepare = callable->create_basic_block();
        auto *body = callable->create_basic_block();
        auto *work = callable->create_basic_block();
        auto *update = callable->create_basic_block();
        auto *merge = callable->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(Type::of<uint32_t>());
        builder.store(
            state,
            module.create_constant_zero(Type::of<uint32_t>()));
        auto *loop = builder.loop();
        loop->set_prepare_block(prepare);
        loop->set_body_block(body);
        loop->set_update_block(update);
        loop->set_merge_block(merge);

        builder.set_insertion_point(prepare);
        builder.cond_br(loop_condition, body, merge);

        // Canonical physical loop-boundary selection:
        //   true  -> normal fallthrough/selection merge
        //   false -> loop update
        // The update arm is not an ordinary recursive CFG region.
        builder.set_insertion_point(body);
        auto *guard = builder.if_(take_body);
        guard->set_true_target(work);
        guard->set_false_target(update);
        guard->set_merge_block(work);

        builder.set_insertion_point(work);
        builder.store(
            state,
            module.create_constant_one(Type::of<uint32_t>()));
        builder.continue_(update);

        builder.set_insertion_point(update);
        auto *value = builder.load(Type::of<uint32_t>(), state);
        auto *next = builder.call(
            // Keep this as a generic LoopInst rather than allowing the
            // translator's canonical induction-variable matcher to turn it
            // into a ForStmt. The regression specifically exercises a
            // selection arm that targets the physical update boundary of an
            // enclosing generic loop.
            Type::of<uint32_t>(), ArithmeticOp::BINARY_SUB,
            {value,
             module.create_constant_one(Type::of<uint32_t>())});
        builder.store(state, next);
        builder.br(prepare);

        builder.set_insertion_point(merge);
        builder.return_void();

        auto verification = xir_verify_function(
            callable,
            {.require_no_phi = true,
             .require_canonical_break_continue_targets = true});
        expect(verification.succeeded());
        auto ast = xir_to_ast_translate(*callable, {});
        expect(ast != nullptr);
        if (ast == nullptr) { return; }
        StatementCounter counter;
        ast->body()->accept(counter);
        expect(counter.loops == 1u)
            << "expected one generic loop, got " << counter.loops;
        // The canonical ContinueInst targeting the loop update is represented
        // by normal loop-tail fallthrough. Only the boundary-selection arm
        // needs an explicit AST ContinueStmt.
        expect(counter.continues == 1u)
            << "the direct update arm must terminate with one synthesized "
               "continue; got "
            << counter.continues;
        auto rebuilt = ast_to_xir_translate(
            ast->function(), {});
        expect(rebuilt != nullptr);
        if (rebuilt == nullptr) { return; }
        expect(xir_verify_module(rebuilt.get()).succeeded());
        auto update_count = size_t{0u};
        for (auto *function : rebuilt->function_list()) {
            auto *definition = function->definition();
            if (definition == nullptr) { continue; }
            definition->traverse_instructions(
                [&](Instruction *instruction) noexcept {
                    if (instruction->isa<ArithmeticInst>() &&
                        static_cast<ArithmeticInst *>(instruction)->op() ==
                            ArithmeticOp::BINARY_SUB) {
                        ++update_count;
                    }
                });
        }
        expect(update_count == 2u)
            << "both the normal loop tail and the boundary-selection arm "
               "must execute the update slice exactly once; got "
            << update_count << " copies";
    };

    "xir_to_ast_for_boundary_continue_uses_header_step_once"_test = [] {
        Module module;
        auto *callable = module.create_callable(nullptr);
        auto *loop_condition = callable->create_value_argument(
            Type::of<bool>());
        auto *take_body = callable->create_value_argument(
            Type::of<bool>());
        auto *entry = callable->create_body_block();
        auto *prepare = callable->create_basic_block();
        auto *body = callable->create_basic_block();
        auto *work = callable->create_basic_block();
        auto *update = callable->create_basic_block();
        auto *merge = callable->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *induction = builder.alloca_local(
            Type::of<uint32_t>());
        auto *work_state = builder.alloca_local(
            Type::of<uint32_t>());
        builder.store(
            induction,
            module.create_constant_zero(Type::of<uint32_t>()));
        builder.store(
            work_state,
            module.create_constant_zero(Type::of<uint32_t>()));
        auto *loop = builder.loop();
        loop->set_prepare_block(prepare);
        loop->set_body_block(body);
        loop->set_update_block(update);
        loop->set_merge_block(merge);

        builder.set_insertion_point(prepare);
        builder.cond_br(loop_condition, body, merge);
        builder.set_insertion_point(body);
        auto *guard = builder.if_(take_body);
        guard->set_true_target(work);
        guard->set_false_target(update);
        guard->set_merge_block(work);
        builder.set_insertion_point(work);
        builder.store(
            work_state,
            module.create_constant_one(Type::of<uint32_t>()));
        builder.continue_(update);
        builder.set_insertion_point(update);
        auto *value = builder.load(
            Type::of<uint32_t>(), induction);
        auto *next = builder.call(
            Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD,
            {value,
             module.create_constant_one(Type::of<uint32_t>())});
        builder.store(induction, next);
        builder.br(prepare);
        builder.set_insertion_point(merge);
        builder.return_void();

        expect(xir_verify_function(
                   callable,
                   {.require_no_phi = true,
                    .require_canonical_break_continue_targets = true})
                   .succeeded());
        auto ast = xir_to_ast_translate(*callable, {});
        expect(ast != nullptr);
        if (ast == nullptr) { return; }
        StatementCounter counter;
        ast->body()->accept(counter);
        expect(counter.fors == 1u);
        expect(counter.continues == 1u)
            << "the direct update arm must become a for-loop continue";

        auto rebuilt = ast_to_xir_translate(
            ast->function(), {});
        expect(rebuilt != nullptr);
        if (rebuilt == nullptr) { return; }
        expect(xir_verify_module(rebuilt.get()).succeeded());
        auto step_count = size_t{0u};
        for (auto *function : rebuilt->function_list()) {
            auto *definition = function->definition();
            if (definition == nullptr) { continue; }
            definition->traverse_instructions(
                [&](Instruction *instruction) noexcept {
                    if (instruction->isa<ArithmeticInst>() &&
                        static_cast<ArithmeticInst *>(instruction)->op() ==
                            ArithmeticOp::BINARY_ADD) {
                        ++step_count;
                    }
                });
        }
        expect(step_count == 1u)
            << "ForStmt continue executes its header step implicitly; the "
               "physical update slice must not be duplicated";
    };

#if __has_include(<unistd.h>) && __has_include(<sys/wait.h>)
    "xir_to_ast_direct_rejects_break_without_structured_scope"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *exit = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *break_inst = b.break_(exit);
        b.set_insertion_point(exit);
        b.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto strict = xir_verify_module(
            &module, {.require_canonical_break_continue_targets = true});
        expect(!strict.succeeded());
        auto has_expected_error = std::any_of(
            strict.errors.cbegin(), strict.errors.cend(),
            [&](auto &&error) noexcept {
                return error.block == body &&
                       error.instruction == break_inst &&
                       error.message.find(
                           "Break target is not the nearest enclosing structured break target.") !=
                           luisa::string::npos;
            });
        expect(has_expected_error);
        expect(terminates_with_abort([&] {
            static_cast<void>(xir_to_ast_translate(*kernel, {}));
        }));
    };
#endif

    "xir_to_ast_direct_structured_if"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *buffer = kernel->create_resource_argument(Type::buffer(Type::of<float>()));
        auto *body = kernel->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *idx = module.create_thread_id();
        auto *zero = module.create_constant_zero(Type::of<uint>());
        auto *x = b.call(Type::of<uint>(), ArithmeticOp::EXTRACT, {idx, zero});
        auto *value = b.call(Type::of<float>(), ResourceReadOp::BUFFER_READ, {buffer, x});
        float threshold = 0.0f;
        auto *c = module.create_constant(Type::of<float>(), &threshold);
        auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_GREATER, {value, c});
        auto *if_inst = b.if_(cond);
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(if_inst->create_true_block());
        b.call(ResourceWriteOp::BUFFER_WRITE, {buffer, x, value});
        b.br(merge);
        b.set_insertion_point(if_inst->create_false_block());
        b.call(ResourceWriteOp::BUFFER_WRITE, {buffer, x, c});
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();

        auto ast = xir_to_ast_translate(*kernel, {});
        expect(ast != nullptr);
        StatementCounter counter;
        ast->body()->accept(counter);
        expect(counter.ifs == 1u);
        expect(counter.exprs == 2u);
        expect(counter.returns == 1u);
    };

    "xir_to_ast_value_checkpoint_work_is_branch_local"_test = [] {
        auto translate = [](size_t dominating_binding_count) noexcept {
            Module module;
            auto *kernel = module.create_kernel();
            auto *condition = kernel->create_value_argument(
                Type::of<bool>());
            luisa::vector<Value *> dominating_values;
            dominating_values.reserve(dominating_binding_count);
            for (auto i = 0u; i < dominating_binding_count; ++i) {
                dominating_values.emplace_back(
                    kernel->create_value_argument(Type::of<uint>()));
            }
            auto *body = kernel->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *if_inst = b.if_(condition);
            auto *merge = if_inst->create_merge_block();
            auto *one = module.create_constant_one(Type::of<uint>());
            b.set_insertion_point(if_inst->create_true_block());
            static_cast<void>(b.call(
                Type::of<uint>(), ArithmeticOp::BINARY_ADD,
                {dominating_values.front(), one}));
            b.br(merge);
            b.set_insertion_point(if_inst->create_false_block());
            static_cast<void>(b.call(
                Type::of<uint>(), ArithmeticOp::BINARY_ADD,
                {dominating_values.back(), one}));
            b.br(merge);
            b.set_insertion_point(merge);
            b.return_void();

            XIR2ASTTranslationStatistics statistics;
            auto ast = xir_to_ast_translate(
                *kernel,
                {.statistics = &statistics,
                 .verify_value_map_checkpoints = true});
            expect(ast != nullptr);
            return statistics;
        };

        auto small = translate(4u);
        auto large = translate(256u);
        expect(large.peak_value_map_size >
               small.peak_value_map_size);
        expect(small.value_map_checkpoint_count == 2u);
        expect(large.value_map_checkpoint_count == 2u);
        expect(small.value_map_rollback_work > 0u);
        expect(large.value_map_rollback_work ==
               small.value_map_rollback_work)
            << "checkpoint work must depend on branch-local insertions, not the retained map prefix";
    };

    "xir_to_ast_direct_terminal_null_merge_selections"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *selector = kernel->create_value_argument(Type::of<int>());
        auto *body = kernel->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *if_inst = b.if_(condition);
        auto *true_block = if_inst->create_true_block();
        auto *false_block = if_inst->create_false_block();
        b.set_insertion_point(true_block);
        b.return_void();
        b.set_insertion_point(false_block);
        auto *switch_inst = b.switch_(selector);
        auto *case_block = switch_inst->create_case_block(1);
        auto *default_block = switch_inst->create_default_block();
        auto *switch_merge = switch_inst->create_merge_block();
        b.set_insertion_point(case_block);
        b.return_void();
        b.set_insertion_point(default_block);
        b.return_void();
        b.set_insertion_point(switch_merge);
        b.unreachable_();

        auto ast = xir_to_ast_translate(*kernel, {});
        expect(ast != nullptr);
        StatementCounter counter;
        ast->body()->accept(counter);
        expect(counter.ifs == 1u);
        expect(counter.switches == 1u);
        expect(counter.returns == 3u);
    };

    "xir_to_ast_normalize_repairs_unterminated_disconnected_block"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        static_cast<void>(kernel->create_basic_block());

        xir_to_ast_normalize_module(&module);
        for (auto *block : kernel->basic_blocks()) {
            expect(block->is_terminated());
        }
        expect(xir_verify_module(
                   &module, {.require_no_phi = true})
                   .succeeded());
        expect(xir_to_ast_translate(*kernel, {}) != nullptr);
    };

    "xir_to_ast_direct_cond_br_with_nested_reconvergence"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *local = b.alloca_local(Type::of<uint32_t>());
        bool cond_value = true;
        auto *cond = module.create_constant(Type::of<bool>(), &cond_value);
        bool nested_cond_value = false;
        auto *nested_cond = module.create_constant(Type::of<bool>(), &nested_cond_value);
        uint32_t one = 1u;
        uint32_t two = 2u;
        uint32_t three = 3u;
        auto *one_c = module.create_constant(Type::of<uint32_t>(), &one);
        auto *two_c = module.create_constant(Type::of<uint32_t>(), &two);
        auto *three_c = module.create_constant(Type::of<uint32_t>(), &three);
        auto *true_block = kernel->create_basic_block();
        auto *false_block = kernel->create_basic_block();
        auto *true_true_block = kernel->create_basic_block();
        auto *true_false_block = kernel->create_basic_block();
        auto *true_join = kernel->create_basic_block();
        auto *merge = kernel->create_basic_block();
        b.cond_br(cond, true_block, false_block);

        b.set_insertion_point(true_block);
        b.cond_br(nested_cond, true_true_block, true_false_block);
        b.set_insertion_point(true_true_block);
        b.store(local, one_c);
        b.br(true_join);
        b.set_insertion_point(true_false_block);
        b.store(local, two_c);
        b.br(true_join);
        b.set_insertion_point(true_join);
        b.br(merge);

        b.set_insertion_point(false_block);
        b.store(local, three_c);
        b.br(merge);

        b.set_insertion_point(merge);
        b.return_void();

        auto ast = xir_to_ast_translate(*kernel, {});
        expect(ast != nullptr);
        StatementCounter counter;
        ast->body()->accept(counter);
        expect(counter.ifs == 2u);
        expect(counter.stores == 3u);
        expect(counter.returns == 1u);
    };
}

void reg_xir2ast_ast_roundtrip() {

    "xir_to_ast_ast_xir_arithmetic_roundtrip"_test = [] {
        Kernel1D kernel = [](BufferFloat buffer) {
            auto idx = dispatch_id().x;
            auto value = buffer->read(idx);
            auto y = value * 2.0f + 1.0f;
            buffer->write(idx, y);
        };
        auto original = ast_to_xir_translate(kernel.function()->function(), {});
        xir_to_ast_normalize_module(original.get());
        auto *def = first_kernel_definition(original.get());
        expect(def != nullptr);
        auto ast = xir_to_ast_translate(*def, {});
        expect(ast != nullptr);
        auto rebuilt = ast_to_xir_translate(ast->function(), {});
        expect(rebuilt != nullptr);
        auto text = xir_to_text_translate(rebuilt.get(), false);
        expect(text.find("arithmetic binary_mul") != string::npos);
        expect(text.find("resource_write buffer_write") != string::npos);
    };

    "xir_to_ast_ast_xir_control_flow_roundtrip"_test = [] {
        Kernel1D kernel = [](BufferFloat buffer) {
            auto idx = dispatch_id().x;
            auto value = buffer->read(idx);
            $if (value > 0.0f) {
                buffer->write(idx, value);
            }
            $else {
                buffer->write(idx, 0.0f);
            };
        };
        auto original = ast_to_xir_translate(kernel.function()->function(), {});
        xir_to_ast_normalize_module(original.get());
        auto *def = first_kernel_definition(original.get());
        auto ast = xir_to_ast_translate(*def, {});
        expect(ast != nullptr);
        auto rebuilt = ast_to_xir_translate(ast->function(), {});
        expect(rebuilt != nullptr);
        auto text = xir_to_text_translate(rebuilt.get(), false);
        expect(text.find("if") != string::npos);
        expect(text.find("resource_write buffer_write") != string::npos);
    };

    "xir_to_ast_preserves_bound_resource_arguments"_test = [] {
        Kernel1D kernel = [](BufferFloat captured, BufferFloat runtime) {
            auto idx = dispatch_id().x;
            runtime->write(idx, captured->read(idx));
        };
        compute::Function function = kernel.function()->function();
        auto binding = compute::Function::Binding{compute::Function::BufferBinding{0x1234u, 16u, 64u}};
        auto config = XIR2ASTConfig{.bound_arguments = luisa::span{&binding, 1u}};
        auto original = ast_to_xir_translate(function, {});
        auto *def = first_kernel_definition(original.get());
        auto ast = xir_to_ast_translate(*def, config);
        expect(ast != nullptr);
        expect(ast->bound_arguments().size() == 1u);
        expect(ast->unbound_arguments().size() == 1u);
        expect(luisa::holds_alternative<compute::Function::BufferBinding>(ast->bound_arguments().front()));
    };

    "xir_to_ast_roundtrips_lowered_autodiff"_test = [] {
        Kernel1D kernel = [](BufferFloat output) {
            auto index = dispatch_id().x;
            auto x = def(2.0f);
            $autodiff {
                requires_grad(x);
                auto y = x * x;
                backward(y);
                output->write(index, grad(x));
            };
        };
        auto ast = backend_detail::lower_autodiff_to_ast(
            kernel.function()->function());
        expect(ast != nullptr);
        expect(!ast->function().requires_autodiff());
        auto rebuilt = ast_to_xir_translate(ast->function(), {});
        auto text = xir_to_text_translate(rebuilt.get(), false);
        expect(text.find("autodiff") == string::npos);
        expect(text.find("resource_write buffer_write") != string::npos);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_xir2ast_direct();
    reg_xir2ast_ast_roundtrip();
    return 0;
}
