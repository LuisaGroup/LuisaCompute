// Test for coroutine CFG distillation and malformed-graph rejection.

#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/core/stl/format.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/clock.h>
#include <luisa/xir/instructions/indexed_branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_reg2mem.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/translators/xir2text.h>
#include <luisa/xir/verifier.h>
#include <luisa/xir/special_register.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] KernelFunction *make_kernel_with_body(Module &m, BasicBlock *&body_out) noexcept {
    auto *k = m.create_kernel();
    body_out = k->create_body_block();
    return k;
}

}// namespace

void reg_coro_cfg_distill() {

    "no_suspend_single_scope"_test = [] {
        // given: a function with no coroutine instructions
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();

        // when
        auto result = coro_cfg_distill_pass_run_on_function(k);

        // then: 1 scope, no suspend info, not terminal
        expect(result.scopes.size() == 1u);
        expect(result.scopes[0].blocks.size() == 1u);
        expect(result.scopes[0].blocks[0] == body);
        expect(!result.scopes[0].suspend_token.has_value());
        expect(!result.scopes[0].suspend_name.has_value());
        expect(!result.scopes[0].is_terminal);
        expect(result.edges.size() == 1u);
        expect(result.edges[0].empty());
        expect(result.boundary_verifier_count == 1u);

        auto verification_transaction =
            begin_xir_pass_verification_transaction(&m);
        auto enclosed = coro_cfg_distill_pass_run_on_function(
            k,
            {.verification_transaction =
                 &verification_transaction});
        expect(enclosed.succeeded());
        expect(enclosed.scopes.size() == result.scopes.size());
        expect(enclosed.boundary_verifier_count == 0u);
        expect(verification_transaction.verify_output().succeeded());
    };

    "single_suspend_two_scopes"_test = [] {
        // given: CFG with one suspend point
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;

        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *suspend_bb = k->create_basic_block();
        auto *resume_bb = k->create_basic_block();

        b.set_insertion_point(body);
        b.cond_br(cond, suspend_bb, resume_bb);

        b.set_insertion_point(suspend_bb);
        b.coro_suspend(42u, "checkpoint", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(42u, nullptr);
        b.return_void();

        // when
        auto result = coro_cfg_distill_pass_run_on_function(k);

        // then: 2 scopes, scope 0 has suspend, scope 1 is continuation
        expect(result.scopes.size() == 2u);

        // scope 0
        expect(result.scopes[0].scope_id == 0);
        expect(result.scopes[0].blocks.size() >= 1u);// body is in scope 0
        expect(result.scopes[0].suspend_token.has_value());
        expect(*result.scopes[0].suspend_token == 42u);
        expect(result.scopes[0].suspend_name.has_value());
        expect(*result.scopes[0].suspend_name == "checkpoint");
        expect(!result.scopes[0].is_terminal);

        // scope 1
        expect(result.scopes[1].scope_id == 1);
        expect(!result.scopes[1].suspend_token.has_value());
        expect(!result.scopes[1].is_terminal);

        // edges
        expect(result.edges.size() == 2u);
    };

    "designated_replayable_value_is_spilled_only_on_export_edge"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume_first = kernel->create_basic_block();
        auto *resume_second = kernel->create_basic_block();
        XIRBuilder b;
        auto *dispatch_id = m.create_special_register(
            DerivedSpecialRegisterTag::DISPATCH_ID);
        uint32_t component_x = 0u;
        uint32_t bias_value = 17u;
        auto *x = m.create_constant(
            Type::of<uint32_t>(), &component_x);
        auto *bias = m.create_constant(
            Type::of<uint32_t>(), &bias_value);
        b.set_insertion_point(entry);
        auto *tid = b.call(
            Type::of<uint>(), ArithmeticOp::EXTRACT,
            {dispatch_id, x});
        auto *hint = b.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {tid, bias});
        std::array<luisa::string, 1u> export_names{
            luisa::string{"coro_hint"}};
        std::array<Value *, 1u> export_values{hint};
        b.coro_suspend(
            151u, "sort", nullptr,
            luisa::span{export_names},
            luisa::span{export_values});
        b.set_insertion_point(resume_first);
        b.coro_resume(151u, nullptr);
        b.coro_suspend(157u, "done", nullptr);
        b.set_insertion_point(resume_second);
        b.coro_resume(157u, nullptr);
        b.return_void();

        auto result =
            coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        auto field = std::find_if(
            result.frame_values.begin(),
            result.frame_values.end(),
            [&](const auto &value) noexcept {
                return value.value == hint;
            });
        expect(field != result.frame_values.end());
        if (field != result.frame_values.end()) {
            expect(std::find(field->aliases.begin(),
                             field->aliases.end(),
                             "coro_hint") !=
                   field->aliases.end());
        }
        const CoroCfgDistillResult::Edge *sort_edge = nullptr;
        const CoroCfgDistillResult::Edge *done_edge = nullptr;
        for (auto &edge : result.transition_edges) {
            if (edge.token == 151u) { sort_edge = &edge; }
            if (edge.token == 157u) { done_edge = &edge; }
        }
        expect(sort_edge != nullptr);
        expect(done_edge != nullptr);
        if (sort_edge != nullptr) {
            expect(std::find(sort_edge->store_values.begin(),
                             sort_edge->store_values.end(),
                             hint) != sort_edge->store_values.end());
            expect(std::find(sort_edge->live_values.begin(),
                             sort_edge->live_values.end(),
                             hint) != sort_edge->live_values.end());
        }
        if (done_edge != nullptr) {
            expect(std::find(done_edge->store_values.begin(),
                             done_edge->store_values.end(),
                             hint) == done_edge->store_values.end());
            expect(std::find(done_edge->live_values.begin(),
                             done_edge->live_values.end(),
                             hint) == done_edge->live_values.end());
        }
    };

    "complete_suspend_extensions_survive_cfg_distillation"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *input = b.clock();
        auto *output = b.alloca_local(Type::of<float>());
        output->set_name("stage_output");

        luisa::vector<CoroSuspendExtensionPtr> extensions;
        extensions.emplace_back(make_coro_suspend_extension_data(
            "com.example.nn-shade", 7u,
            CoroSuspendFallback::reject,
            {{.name = "input",
              .access = CoroSuspendBindingAccess::read,
              .lifetime = CoroSuspendBindingLifetime::queued,
              .index = 0u},
             {.name = "output",
              .access = CoroSuspendBindingAccess::write,
              .lifetime = CoroSuspendBindingLifetime::resumed,
              .index = 1u}},
            {{.name = "enabled", .value = true},
             {.name = "label", .value = luisa::string{"neural-sdf"}},
             {.name = "scale", .value = 0.85}}));
        extensions.emplace_back(make_coro_suspend_annotation_data(
            "luisa.coro.schedule.sort", 1u,
            CoroSuspendFallback::ignore,
            {{.name = "key",
              .access = CoroSuspendBindingAccess::read,
              .lifetime = CoroSuspendBindingLifetime::queued,
              .index = 2u}},
            {{.name = "range", .value = uint64_t{4096u}}}));
        luisa::vector<Value *> binding_values{input, output, input};
        b.coro_suspend(
            149u, "external-stage", nullptr, {}, {},
            std::move(extensions), luisa::span{binding_values});
        b.set_insertion_point(resume);
        b.coro_resume(149u, nullptr);
        static_cast<void>(b.load(Type::of<float>(), output));
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto result = coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        const CoroCfgDistillResult::Scope::SuspendPoint *point = nullptr;
        for (auto &scope : result.scopes) {
            for (auto &candidate : scope.suspend_points) {
                if (candidate.token == 149u) { point = &candidate; }
            }
        }
        const CoroCfgDistillResult::Edge *edge = nullptr;
        for (auto &candidate : result.transition_edges) {
            if (candidate.is_suspend && candidate.token == 149u) {
                edge = &candidate;
            }
        }
        expect(point != nullptr);
        expect(edge != nullptr);
        if (point == nullptr || edge == nullptr) { return; }
        for (auto *owner : {&point->extension_owner,
                            &edge->extension_owner}) {
            expect(owner->extensions.size() == 2u);
            expect(owner->binding_values == binding_values);
            if (owner->extensions.size() != 2u) { continue; }
            auto &&semantic = owner->extensions[0u];
            auto &&annotation = owner->extensions[1u];
            expect(semantic->schema() == "com.example.nn-shade");
            expect(semantic->version() == 7u);
            expect(!semantic->is_annotation());
            expect(semantic->fallback() ==
                   CoroSuspendFallback::reject);
            expect(semantic->bindings().size() == 2u);
            expect(semantic->attributes().size() == 3u);
            expect(annotation->schema() ==
                   "luisa.coro.schedule.sort");
            expect(annotation->is_annotation());
            expect(annotation->fallback() ==
                   CoroSuspendFallback::ignore);
            expect(annotation->bindings().size() == 1u);
            expect(annotation->attributes().size() == 1u);
        }
        auto input_field = std::find_if(
            result.frame_values.begin(), result.frame_values.end(),
            [&](const auto &value) noexcept {
                return value.value == input;
            });
        auto output_field = std::find_if(
            result.frame_values.begin(), result.frame_values.end(),
            [&](const auto &value) noexcept {
                return value.value == output;
            });
        expect(input_field != result.frame_values.end());
        expect(output_field != result.frame_values.end());
        expect(edge->extension_binding_frame_value_indices.size() == 3u);
        if (input_field != result.frame_values.end() &&
            output_field != result.frame_values.end() &&
            edge->extension_binding_frame_value_indices.size() == 3u) {
            auto input_index = static_cast<size_t>(
                input_field - result.frame_values.begin());
            auto output_index = static_cast<size_t>(
                output_field - result.frame_values.begin());
            expect(edge->extension_binding_frame_value_indices[0u] ==
                   luisa::vector<size_t>{input_index});
            expect(edge->extension_binding_frame_value_indices[1u] ==
                   luisa::vector<size_t>{output_index});
            expect(edge->extension_binding_frame_value_indices[2u] ==
                   luisa::vector<size_t>{input_index});
            expect(std::find(edge->store_frame_value_indices.begin(),
                             edge->store_frame_value_indices.end(),
                             input_index) !=
                   edge->store_frame_value_indices.end());
            // A write-only stage output is allocated and live, but the source
            // continuation must not spill (and thus read) its old value.
            expect(std::find(edge->store_frame_value_indices.begin(),
                             edge->store_frame_value_indices.end(),
                             output_index) ==
                   edge->store_frame_value_indices.end());
            expect(std::find(edge->killed_frame_value_indices.begin(),
                             edge->killed_frame_value_indices.end(),
                             output_index) !=
                   edge->killed_frame_value_indices.end());
            expect(std::find(edge->live_frame_value_indices.begin(),
                             edge->live_frame_value_indices.end(),
                             output_index) !=
                   edge->live_frame_value_indices.end());
        }
    };

    "extension_bindings_reuse_existing_dataflow_atom_and_slot"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *state = b.clock();
        state->set_name("shared_state");

        luisa::vector<CoroSuspendExtensionPtr> extensions;
        extensions.emplace_back(make_coro_suspend_extension_data(
            "com.example.stage", 1u,
            CoroSuspendFallback::reject,
            {{.name = "stage-input",
              .access = CoroSuspendBindingAccess::read,
              .lifetime = CoroSuspendBindingLifetime::resumed,
              .index = 0u}},
            {}));
        extensions.emplace_back(make_coro_suspend_annotation_data(
            "luisa.coro.schedule.sort", 1u,
            CoroSuspendFallback::ignore,
            {{.name = "sort-key",
              .access = CoroSuspendBindingAccess::read,
              .lifetime = CoroSuspendBindingLifetime::queued,
              .index = 1u}},
            {}));
        luisa::vector<luisa::string> export_names{"legacy"};
        luisa::vector<Value *> export_values{state};
        luisa::vector<Value *> binding_values{state, state};
        b.coro_suspend(
            153u, "shared", nullptr,
            luisa::span{export_names}, luisa::span{export_values},
            std::move(extensions), luisa::span{binding_values});
        b.set_insertion_point(resume);
        b.coro_resume(153u, nullptr);
        static_cast<void>(b.call(
            Type::of<uint64_t>(), ArithmeticOp::BINARY_ADD,
            {state, m.create_constant_one(Type::of<uint64_t>())}));
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto result = coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(std::count_if(
                   result.frame_values.begin(),
                   result.frame_values.end(),
                   [&](const auto &value) noexcept {
                       return value.value == state;
                   }) == 1u);
        expect(result.frame_values.size() == 1u);
        expect(result.frame_slots.size() == 1u);
        const CoroCfgDistillResult::Edge *edge = nullptr;
        for (auto &candidate : result.transition_edges) {
            if (candidate.is_suspend && candidate.token == 153u) {
                edge = &candidate;
            }
        }
        expect(edge != nullptr);
        if (edge != nullptr) {
            expect(edge->extension_binding_frame_value_indices.size() ==
                   2u);
            if (edge->extension_binding_frame_value_indices.size() == 2u) {
                expect(edge->extension_binding_frame_value_indices[0u] ==
                       luisa::vector<size_t>{0u});
                expect(edge->extension_binding_frame_value_indices[1u] ==
                       luisa::vector<size_t>{0u});
            }
            expect(edge->store_frame_value_indices ==
                   luisa::vector<size_t>{0u});
            expect(edge->live_frame_value_indices ==
                   luisa::vector<size_t>{0u});
        }
    };

    "read_write_extension_binding_uses_one_existing_memory_slot"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *state = b.alloca_local(Type::of<float>());
        state->set_name("throughput");
        b.store(state, m.create_constant_one(Type::of<float>()));
        luisa::vector<CoroSuspendExtensionPtr> extensions;
        extensions.emplace_back(make_coro_suspend_extension_data(
            "com.example.nn-shade", 1u,
            CoroSuspendFallback::reject,
            {{.name = "throughput",
              .access = CoroSuspendBindingAccess::read_write,
              .lifetime = CoroSuspendBindingLifetime::resumed,
              .index = 0u}},
            {}));
        luisa::vector<Value *> binding_values{state};
        b.coro_suspend(
            155u, "read-write", nullptr, {}, {},
            std::move(extensions), luisa::span{binding_values});
        b.set_insertion_point(resume);
        b.coro_resume(155u, nullptr);
        static_cast<void>(b.load(Type::of<float>(), state));
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto result = coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(result.frame_values.size() == 1u);
        expect(result.frame_slots.size() == 1u);
        const CoroCfgDistillResult::Edge *edge = nullptr;
        for (auto &candidate : result.transition_edges) {
            if (candidate.is_suspend && candidate.token == 155u) {
                edge = &candidate;
            }
        }
        expect(edge != nullptr);
        if (edge != nullptr) {
            expect(edge->extension_binding_frame_value_indices ==
                   luisa::vector<luisa::vector<size_t>>{{0u}});
            expect(edge->store_frame_value_indices ==
                   luisa::vector<size_t>{0u});
            expect(edge->killed_frame_value_indices ==
                   luisa::vector<size_t>{0u});
            expect(edge->live_frame_value_indices ==
                   luisa::vector<size_t>{0u});
        }
    };

    "diagnostic_name_does_not_designate_frame_abi"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume = kernel->create_basic_block();
        XIRBuilder b;
        auto *one = m.create_constant_one(Type::of<uint>());
        b.set_insertion_point(entry);
        auto *named = b.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD, {one, one});
        named->set_name("coro_hint");
        b.coro_suspend(163u, "sort", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(163u, nullptr);
        b.return_void();

        auto result =
            coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(std::none_of(
            result.frame_values.begin(), result.frame_values.end(),
            [](const auto &value) noexcept {
                return std::find(value.aliases.begin(), value.aliases.end(),
                                 "coro_hint") != value.aliases.end();
            }));
        expect(std::none_of(
            result.frame_values.begin(), result.frame_values.end(),
            [&](const auto &value) noexcept {
                return value.value == named;
            }));
    };

    "partial_export_on_bypassed_token_is_rejected"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *suspend_block = kernel->create_basic_block();
        auto *bypass_block = kernel->create_basic_block();
        auto *resume_block = kernel->create_basic_block();
        XIRBuilder b;
        auto *condition = m.create_constant_one(Type::of<bool>());
        auto *hint = m.create_constant_one(Type::of<uint>());
        std::array<luisa::string, 1u> export_names{
            luisa::string{"coro_hint"}};
        std::array<Value *, 1u> export_values{hint};

        b.set_insertion_point(entry);
        b.cond_br(condition, suspend_block, bypass_block);
        b.set_insertion_point(suspend_block);
        b.coro_suspend(167u, "sort", nullptr,
                       luisa::span{export_names},
                       luisa::span{export_values});
        b.set_insertion_point(bypass_block);
        b.br(resume_block);
        b.set_insertion_point(resume_block);
        b.coro_resume(167u, nullptr);
        b.return_void();

        auto result =
            coro_cfg_distill_pass_run_on_function(kernel);

        expect(!result.succeeded());
        expect(result.invalid_cfg_error_count == 1u);
    };

    "three_suspends_four_scopes"_test = [] {
        // given: CFG with three suspend points (linear chain)
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());

        // suspend 1
        auto *s1 = k->create_basic_block();
        auto *r1 = k->create_basic_block();

        b.set_insertion_point(body);
        b.cond_br(cond, s1, r1);

        b.set_insertion_point(s1);
        b.coro_suspend(1u, "s1", nullptr);

        b.set_insertion_point(r1);
        b.coro_resume(1u, nullptr);

        // suspend 2
        auto *s2 = k->create_basic_block();
        auto *r2 = k->create_basic_block();
        b.cond_br(cond, s2, r2);

        b.set_insertion_point(s2);
        b.coro_suspend(2u, "s2", nullptr);

        b.set_insertion_point(r2);
        b.coro_resume(2u, nullptr);

        // suspend 3
        auto *s3 = k->create_basic_block();
        auto *r3 = k->create_basic_block();
        b.cond_br(cond, s3, r3);

        b.set_insertion_point(s3);
        b.coro_suspend(3u, "s3", nullptr);

        b.set_insertion_point(r3);
        b.coro_resume(3u, nullptr);
        b.return_void();

        // when
        auto result = coro_cfg_distill_pass_run_on_function(k);

        // then: 4 scopes
        expect(result.scopes.size() == 4u);

        // verify suspend tokens
        expect(result.scopes[0].suspend_token.has_value());
        expect(*result.scopes[0].suspend_token == 1u);
        expect(result.scopes[1].suspend_token.has_value());
        expect(*result.scopes[1].suspend_token == 2u);
        expect(result.scopes[2].suspend_token.has_value());
        expect(*result.scopes[2].suspend_token == 3u);
        expect(!result.scopes[3].suspend_token.has_value());

        // verify no scope is terminal except possibly the last
        expect(!result.scopes[0].is_terminal);
        expect(!result.scopes[1].is_terminal);
        expect(!result.scopes[2].is_terminal);
        expect(!result.scopes[3].is_terminal);

        // verify scope block counts
        for (size_t i = 0; i < 4u; ++i) {
            expect(result.scopes[i].blocks.size() >= 1u);
        }

        // edges
        expect(result.edges.size() == 4u);
    };

    "suspend_token_values_match"_test = [] {
        // given: two suspend points with distinct tokens and names
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());

        auto *s1 = k->create_basic_block();
        auto *r1 = k->create_basic_block();

        b.set_insertion_point(body);
        b.cond_br(cond, s1, r1);

        b.set_insertion_point(s1);
        b.coro_suspend(100u, "alpha", nullptr);

        b.set_insertion_point(r1);
        b.coro_resume(100u, nullptr);

        auto *s2 = k->create_basic_block();
        auto *r2 = k->create_basic_block();
        b.cond_br(cond, s2, r2);

        b.set_insertion_point(s2);
        b.coro_suspend(200u, "beta", nullptr);

        b.set_insertion_point(r2);
        b.coro_resume(200u, nullptr);
        b.return_void();

        // when
        auto result = coro_cfg_distill_pass_run_on_function(k);

        // then: token values match the suspend instructions
        expect(result.scopes.size() == 3u);

        expect(result.scopes[0].suspend_token.has_value());
        expect(*result.scopes[0].suspend_token == 100u);
        expect(result.scopes[0].suspend_name.has_value());
        expect(*result.scopes[0].suspend_name == "alpha");

        expect(result.scopes[1].suspend_token.has_value());
        expect(*result.scopes[1].suspend_token == 200u);
        expect(result.scopes[1].suspend_name.has_value());
        expect(*result.scopes[1].suspend_name == "beta");

        expect(!result.scopes[2].suspend_token.has_value());
    };

    "terminal_scope"_test = [] {
        // given: a coroutine that ends with CoroTerminateInst
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());

        auto *s1 = k->create_basic_block();
        auto *r1 = k->create_basic_block();

        b.set_insertion_point(body);
        b.cond_br(cond, s1, r1);

        b.set_insertion_point(s1);
        b.coro_suspend(1u, "middle", nullptr);

        b.set_insertion_point(r1);
        b.coro_resume(1u, nullptr);

        auto *term_bb = k->create_basic_block();
        b.br(term_bb);

        b.set_insertion_point(term_bb);
        b.coro_terminate();

        // when
        auto result = coro_cfg_distill_pass_run_on_function(k);

        // then: last scope is terminal
        expect(result.scopes.size() >= 1u);
        auto &last = result.scopes.back();
        expect(last.is_terminal);
    };

    "scope_contains_suspend_block"_test = [] {
        // given: a single-suspend CFG — verify the suspend block is in the first scope
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());

        auto *suspend_bb = k->create_basic_block();
        auto *resume_bb = k->create_basic_block();

        b.set_insertion_point(body);
        b.cond_br(cond, suspend_bb, resume_bb);

        b.set_insertion_point(suspend_bb);
        b.coro_suspend(7u, "test", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(7u, nullptr);
        b.return_void();

        // when
        auto result = coro_cfg_distill_pass_run_on_function(k);

        // then: suspend block is in scope 0, resume block is in scope 1
        expect(result.scopes.size() == 2u);

        bool suspend_found = false;
        for (auto *bb : result.scopes[0].blocks) {
            if (bb == suspend_bb) { suspend_found = true; }
        }
        expect(suspend_found);

        bool resume_found = false;
        for (auto *bb : result.scopes[1].blocks) {
            if (bb == resume_bb) { resume_found = true; }
        }
        expect(resume_found);
    };

    "module_pass_iterates_all_functions"_test = [] {
        // given: module with a kernel and a callable (neither with coroutine instructions)
        Module m;
        {
            auto *k = m.create_kernel();
            auto *body = k->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            b.return_void();
        }
        {
            auto *c = m.create_callable(nullptr);
            auto *body = c->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            b.return_void();
        }

        // when
        auto count = coro_cfg_distill_pass_run_on_module(&m);

        // then: processes both definition functions
        expect(count == 2u);
    };

    // ── edge case: adjacent suspends ───────────────────────────────────
    "adjacent_suspends"_test = [] {
        // given: two suspends with minimal code between them (resume
        // of first immediately branches to second suspend)
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());

        auto *s1 = k->create_basic_block();
        auto *r1 = k->create_basic_block();

        b.set_insertion_point(body);
        b.cond_br(cond, s1, r1);

        b.set_insertion_point(s1);
        b.coro_suspend(10u, "first", nullptr);

        // scope 1: resume, then immediately branch to next suspend
        b.set_insertion_point(r1);
        b.coro_resume(10u, nullptr);

        auto *s2 = k->create_basic_block();
        auto *r2 = k->create_basic_block();
        b.cond_br(cond, s2, r2);

        b.set_insertion_point(s2);
        b.coro_suspend(20u, "second", nullptr);

        b.set_insertion_point(r2);
        b.coro_resume(20u, nullptr);
        b.return_void();

        // when
        auto result = coro_cfg_distill_pass_run_on_function(k);

        // then: 3 scopes (body→s1 | r1→s2 | r2)
        expect(result.scopes.size() == 3u);

        expect(result.scopes[0].suspend_token.has_value());
        expect(*result.scopes[0].suspend_token == 10u);
        expect(result.scopes[0].suspend_name.has_value());
        expect(*result.scopes[0].suspend_name == "first");

        expect(result.scopes[1].suspend_token.has_value());
        expect(*result.scopes[1].suspend_token == 20u);
        expect(result.scopes[1].suspend_name.has_value());
        expect(*result.scopes[1].suspend_name == "second");

        expect(!result.scopes[2].suspend_token.has_value());

        // no scope marked terminal
        expect(!result.scopes[0].is_terminal);
        expect(!result.scopes[1].is_terminal);
        expect(!result.scopes[2].is_terminal);

        // r1 and s2 should coexist in scope 1 (adjacent)
        bool r1_in_scope1 = false;
        bool s2_in_scope1 = false;
        for (auto *bb : result.scopes[1].blocks) {
            if (bb == r1) { r1_in_scope1 = true; }
            if (bb == s2) { s2_in_scope1 = true; }
        }
        expect(r1_in_scope1);
        expect(s2_in_scope1);

        expect(result.edges.size() == 3u);
    };

    // ── edge case: suspend inside a conditional branch ────────────────
    "suspend_in_conditional"_test = [] {
        // given: an if/else where only one branch contains a suspend;
        // the other branch skips it entirely and merges afterward
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *always_true = m.create_constant_one(Type::of<bool>());

        auto *branch_a = k->create_basic_block();
        auto *branch_b = k->create_basic_block();
        auto *merge = k->create_basic_block();

        auto *s1 = k->create_basic_block();
        auto *r1 = k->create_basic_block();

        b.set_insertion_point(body);
        b.cond_br(always_true, branch_a, branch_b);

        // branch A: contains a suspend
        b.set_insertion_point(branch_a);
        b.cond_br(always_true, s1, r1);

        b.set_insertion_point(s1);
        b.coro_suspend(1u, "in_branch", nullptr);

        b.set_insertion_point(r1);
        b.coro_resume(1u, nullptr);
        b.br(merge);

        // branch B: no suspend, goes straight to merge
        b.set_insertion_point(branch_b);
        b.br(merge);

        b.set_insertion_point(merge);
        b.return_void();

        // when
        auto result = coro_cfg_distill_pass_run_on_function(k);

        // then: at least 2 scopes, exactly one scope has a suspend
        expect(result.scopes.size() >= 2u);

        size_t suspend_scopes = 0u;
        for (auto &scope : result.scopes) {
            if (scope.suspend_token.has_value()) { suspend_scopes++; }
        }
        expect(suspend_scopes == 1u);

        // suspend block must live in the scope that owns the suspend
        bool s1_in_suspend_scope = false;
        for (auto &scope : result.scopes) {
            if (scope.suspend_token.has_value()) {
                for (auto *bb : scope.blocks) {
                    if (bb == s1) { s1_in_suspend_scope = true; }
                }
            }
        }
        expect(s1_in_suspend_scope);

        // merge block must appear in at least one scope
        bool merge_found = false;
        for (auto &scope : result.scopes) {
            for (auto *bb : scope.blocks) {
                if (bb == merge) { merge_found = true; }
            }
        }
        expect(merge_found);
    };

    // ── edge case: suspend inside a loop ──────────────────────────────
    "suspend_in_loop"_test = [] {
        // given: a loop whose body contains a suspend point;
        // the back-edge goes through the resume block back to the header
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *always_true = m.create_constant_one(Type::of<bool>());
        auto *loop_cond = m.create_constant_one(Type::of<bool>());

        auto *loop_hdr = k->create_basic_block();
        auto *loop_body = k->create_basic_block();
        auto *s1 = k->create_basic_block();
        auto *r1 = k->create_basic_block();
        auto *exit = k->create_basic_block();

        b.set_insertion_point(body);
        b.br(loop_hdr);

        b.set_insertion_point(loop_hdr);
        b.cond_br(loop_cond, loop_body, exit);

        b.set_insertion_point(loop_body);
        b.cond_br(always_true, s1, r1);

        b.set_insertion_point(s1);
        b.coro_suspend(1u, "in_loop", nullptr);

        b.set_insertion_point(r1);
        b.coro_resume(1u, nullptr);
        b.br(loop_hdr);// back-edge through resume

        b.set_insertion_point(exit);
        b.return_void();

        // when
        auto result = coro_cfg_distill_pass_run_on_function(k);

        // then: at least 2 scopes, suspend is found, no crash on cycle
        expect(result.scopes.size() >= 2u);

        bool suspend_found = false;
        for (auto &scope : result.scopes) {
            if (scope.suspend_token.has_value() &&
                *scope.suspend_token == 1u) {
                suspend_found = true;
            }
        }
        expect(suspend_found);

        // edges array matches scope count
        expect(result.edges.size() == result.scopes.size());

        // all blocks from the kernel appear in at least one scope
        size_t total_blocks = 0u;
        for (auto &scope : result.scopes) {
            total_blocks += scope.blocks.size();
        }
        expect(total_blocks >= 6u);// body, loop_hdr, loop_body, s1, r1, exit
    };

    "loop_must_kill_uses_greatest_fixed_point"_test = [] {
        // Definite definition is a must property. The loop header equation is
        //
        //   K_header = K_entry intersect K_backedge.
        //
        // Since state is initialized before the loop and no path can undo a
        // definition, the greatest fixed point contains state. Initializing
        // the backedge to the empty set instead selects the smaller, invalid
        // fixed point and spuriously promotes this loop-local value into the
        // coroutine frame.
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *loop_cond = m.create_constant_one(Type::of<bool>());
        auto *zero = m.create_constant_zero(Type::of<int>());

        auto *loop_header = k->create_basic_block();
        auto *loop_backedge = k->create_basic_block();
        auto *suspend_block = k->create_basic_block();
        auto *resume_block = k->create_basic_block();

        b.set_insertion_point(body);
        auto *state = b.alloca_local(Type::of<int>());
        state->set_name("loop_local_state");
        b.store(state, zero);
        b.br(loop_header);

        b.set_insertion_point(loop_header);
        auto *loaded = b.load(Type::of<int>(), state);
        static_cast<void>(loaded);
        b.cond_br(loop_cond, loop_backedge, suspend_block);

        b.set_insertion_point(loop_backedge);
        b.br(loop_header);

        b.set_insertion_point(suspend_block);
        b.coro_suspend(1u, "after-loop", nullptr);

        b.set_insertion_point(resume_block);
        b.coro_resume(1u, nullptr);
        b.return_void();

        auto result = coro_cfg_distill_pass_run_on_function(k);

        expect(result.succeeded());
        expect(result.scopes.size() == 2u);
        expect(std::find(result.scopes[0u].external_values.begin(),
                         result.scopes[0u].external_values.end(),
                         state) == result.scopes[0u].external_values.end());
        expect(std::find_if(result.frame_values.begin(),
                            result.frame_values.end(),
                            [&](auto &field) noexcept {
                                return field.value == state;
                            }) == result.frame_values.end());
        auto suspend_kills_state = false;
        for (auto &edge : result.transition_edges) {
            if (edge.is_suspend && edge.token == 1u &&
                std::find(edge.killed_values.begin(),
                          edge.killed_values.end(),
                          state) != edge.killed_values.end()) {
                suspend_kills_state = true;
            }
        }
        expect(suspend_kills_state);
    };

    "for_if_suspend_liveness"_test = [] {
        // given: for (...) { if (...) { suspend } } with a local updated
        // before the suspend and used after resume
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *loop_cond = m.create_constant_one(Type::of<bool>());
        auto *if_cond = m.create_constant_one(Type::of<bool>());
        auto *zero = m.create_constant_zero(Type::of<int>());
        auto *one = m.create_constant_one(Type::of<int>());

        auto *loop_hdr = k->create_basic_block();
        auto *loop_body = k->create_basic_block();
        auto *suspend_bb = k->create_basic_block();
        auto *resume_bb = k->create_basic_block();
        auto *after_if = k->create_basic_block();
        auto *exit = k->create_basic_block();

        b.set_insertion_point(body);
        auto *state = b.alloca_local(Type::of<int>());
        state->set_name("state");
        b.store(state, zero);
        b.br(loop_hdr);

        b.set_insertion_point(loop_hdr);
        b.cond_br(loop_cond, loop_body, exit);

        b.set_insertion_point(loop_body);
        auto *old_state = b.load(Type::of<int>(), state);
        auto *new_state = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {old_state, one});
        b.store(state, new_state);
        b.cond_br(if_cond, suspend_bb, after_if);

        b.set_insertion_point(suspend_bb);
        b.coro_suspend(1u, "in_for_if", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(1u, nullptr);
        b.br(after_if);

        b.set_insertion_point(after_if);
        auto *reloaded_state = b.load(Type::of<int>(), state);
        auto *next_state = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {reloaded_state, one});
        b.store(state, next_state);
        b.br(loop_hdr);

        b.set_insertion_point(exit);
        b.return_void();

        // when
        auto result = coro_cfg_distill_pass_run_on_function(k);

        // then: the updated local is stored on suspend edges into the
        // continuation scope, including the loop-carried self edge
        expect(result.scopes.size() == 2u);
        auto has_state_store = [](const CoroCfgDistillResult::Edge &edge) noexcept {
            for (auto &name : edge.store_variables) {
                if (name == "state") { return true; }
            }
            return false;
        };
        bool entry_edge_ok = false;
        bool loop_edge_ok = false;
        for (auto &edge : result.transition_edges) {
            if (edge.token != 1u) { continue; }
            if (edge.from_scope == 0u && edge.to_scope == 1u) {
                entry_edge_ok = has_state_store(edge);
            }
            if (edge.from_scope == 1u && edge.to_scope == 1u) {
                loop_edge_ok = has_state_store(edge);
            }
        }
        expect(entry_edge_ok);
        expect(loop_edge_ok);
    };

    "per_edge_store_excludes_post_suspend_touches"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *zero = m.create_constant_zero(Type::of<int>());
        auto *one = m.create_constant_one(Type::of<int>());
        auto two_value = 2;
        auto *two = m.create_constant(Type::of<int>(), &two_value);

        auto *suspend_bb = k->create_basic_block();
        auto *resume_bb = k->create_basic_block();

        b.set_insertion_point(body);
        auto *state = b.alloca_local(Type::of<int>());
        state->set_name("state");
        auto *late = b.alloca_local(Type::of<int>());
        late->set_name("late");
        b.store(state, one);
        b.store(late, zero);
        b.cond_br(cond, suspend_bb, resume_bb);

        b.set_insertion_point(suspend_bb);
        b.coro_suspend(1u, "s", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(1u, nullptr);
        b.store(late, two);
        auto *a = b.load(Type::of<int>(), state);
        auto *c = b.load(Type::of<int>(), late);
        auto *sum = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {a, c});
        static_cast<void>(sum);
        b.return_void();

        auto result = coro_cfg_distill_pass_run_on_function(k);
        const CoroCfgDistillResult::Edge *suspend_edge = nullptr;
        for (auto &edge : result.transition_edges) {
            if (edge.is_suspend && edge.token == 1u) {
                suspend_edge = &edge;
                break;
            }
        }
        expect(suspend_edge != nullptr);
        bool stores_state = false;
        bool stores_late = false;
        if (suspend_edge != nullptr) {
            for (auto &name : suspend_edge->store_variables) {
                if (name == "state") { stores_state = true; }
                if (name == "late") { stores_late = true; }
            }
        }
        expect(stores_state);
        expect(!stores_late);
    };

    "cross_scope_branch_has_transition_store"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *one = m.create_constant_one(Type::of<int>());

        auto *suspend_bb = k->create_basic_block();
        auto *resume_bb = k->create_basic_block();
        auto *skip_bb = k->create_basic_block();
        auto *merge_bb = k->create_basic_block();

        b.set_insertion_point(body);
        auto *state = b.alloca_local(Type::of<int>());
        state->set_name("state");
        b.store(state, one);
        b.cond_br(cond, suspend_bb, skip_bb);

        b.set_insertion_point(suspend_bb);
        b.coro_suspend(1u, "s", nullptr);

        b.set_insertion_point(skip_bb);
        b.br(resume_bb);

        b.set_insertion_point(resume_bb);
        b.coro_resume(1u, nullptr);
        b.br(merge_bb);

        b.set_insertion_point(merge_bb);
        auto *loaded = b.load(Type::of<int>(), state);
        static_cast<void>(loaded);
        b.return_void();

        auto result = coro_cfg_distill_pass_run_on_function(k);
        bool found_branch_edge = false;
        bool branch_stores_state = false;
        bool found_suspend_edge = false;
        for (auto &edge : result.transition_edges) {
            if (edge.is_suspend) {
                found_suspend_edge = true;
            } else if (edge.from_scope == 0u && edge.to_scope == 1u && edge.exit_block == skip_bb) {
                found_branch_edge = true;
                for (auto &name : edge.store_variables) {
                    if (name == "state") { branch_stores_state = true; }
                }
            }
        }
        expect(found_suspend_edge);
        expect(found_branch_edge);
        expect(branch_stores_state);
    };

    "distilled_scopes_may_share_bypass_merge_blocks"_test = [] {
        // Scope regions are rooted reachability sets rather than a partition.
        // The shared merge is reached directly by the entry scope and through
        // the resume root by the continuation scope. Dense dataflow must use
        // an explicit (scope, block) membership relation; assigning the block
        // one global local index loses one of these two executions.
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *one = m.create_constant_one(Type::of<int>());

        auto *suspend_block = k->create_basic_block();
        auto *bypass_block = k->create_basic_block();
        auto *resume_block = k->create_basic_block();
        auto *shared_merge = k->create_basic_block();

        b.set_insertion_point(body);
        auto *state = b.alloca_local(Type::of<int>());
        state->set_name("shared_merge_state");
        b.store(state, one);
        b.cond_br(cond, suspend_block, bypass_block);

        b.set_insertion_point(suspend_block);
        b.coro_suspend(1u, "shared-merge", nullptr);

        b.set_insertion_point(bypass_block);
        b.br(shared_merge);

        b.set_insertion_point(resume_block);
        b.coro_resume(1u, nullptr);
        // This value is reachable only from the logical resume root. Ordinary
        // raw-CFG traversal from the function body cannot see it, but the
        // shared coroutine value domain must still assign it a coordinate.
        auto *resume_only = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {one, one});
        static_cast<void>(resume_only);
        b.br(shared_merge);

        b.set_insertion_point(shared_merge);
        auto *loaded = b.load(Type::of<int>(), state);
        static_cast<void>(loaded);
        b.return_void();

        auto result = coro_cfg_distill_pass_run_on_function(k);
        expect(result.succeeded());
        expect(result.scopes.size() == 2u);
        auto merge_membership_count = size_t{0u};
        for (auto &scope : result.scopes) {
            if (std::find(scope.blocks.begin(), scope.blocks.end(),
                          shared_merge) != scope.blocks.end()) {
                ++merge_membership_count;
            }
        }
        expect(merge_membership_count == 2u);
        auto suspend_stores_state = false;
        for (auto &edge : result.transition_edges) {
            if (!edge.is_suspend || edge.token != 1u) { continue; }
            suspend_stores_state =
                std::find(edge.store_values.begin(),
                          edge.store_values.end(), state) !=
                edge.store_values.end();
        }
        expect(suspend_stores_state);
    };

    "frame_values_sorted_by_alignment_and_size"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *one_i = m.create_constant_one(Type::of<int>());
        auto *one_f = m.create_constant_one(Type::of<float>());
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *float2_ty = Type::of<float2>();

        auto *suspend_bb = k->create_basic_block();
        auto *resume_bb = k->create_basic_block();

        b.set_insertion_point(body);
        auto *small = b.alloca_local(Type::of<int>());
        small->set_name("small");
        auto *medium = b.alloca_local(Type::of<float>());
        medium->set_name("medium");
        auto *large = b.alloca_local(float2_ty);
        large->set_name("large");
        b.store(small, one_i);
        b.store(medium, one_f);
        auto *large_value = b.call(
            float2_ty, ArithmeticOp::AGGREGATE, {one_f, one_f});
        b.store(large, large_value);
        b.cond_br(cond, suspend_bb, resume_bb);

        b.set_insertion_point(suspend_bb);
        b.coro_suspend(1u, "s", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(1u, nullptr);
        auto *loaded_small = b.load(Type::of<int>(), small);
        auto *loaded_medium = b.load(Type::of<float>(), medium);
        auto *loaded_large = b.load(float2_ty, large);
        auto *loaded_large_x = b.call(Type::of<float>(), ArithmeticOp::EXTRACT, {loaded_large, m.create_constant_zero(Type::of<uint32_t>())});
        auto *medium_i = b.static_cast_(Type::of<int>(), loaded_medium);
        auto *large_i = b.static_cast_(Type::of<int>(), loaded_large_x);
        auto *sum0 = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {loaded_small, medium_i});
        auto *sum1 = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {sum0, large_i});
        static_cast<void>(sum1);
        b.return_void();

        auto result = coro_cfg_distill_pass_run_on_function(k);
        expect(result.frame_values.size() == 3u);
        expect(result.frame_values[0u].name == "large");
        expect(result.frame_values[0u].type == float2_ty);
        expect(result.frame_values[1u].type->alignment() >= result.frame_values[2u].type->alignment());
    };

    "frame_abi_decomposes_padding_into_minimal_packed_fields"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume = kernel->create_basic_block();
        auto *padded = Type::structure(
            {Type::of<float2>(), Type::of<float>()});
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *state = b.alloca_local(padded);
        state->set_name("padded_state");
        b.store(state, m.create_constant_zero(padded));
        b.coro_suspend(211u, "padded", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(211u, nullptr);
        static_cast<void>(b.load(padded, state));
        b.return_void();

        auto result = coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(result.frame_values.size() == 2u);
        if (result.frame_values.size() == 2u) {
            expect(result.frame_values[0u].value == state);
            expect(result.frame_values[0u].access_chain ==
                   luisa::vector<uint32_t>{0u});
            expect(result.frame_values[0u].type == Type::of<float2>());
            expect(result.frame_values[1u].value == state);
            expect(result.frame_values[1u].access_chain ==
                   luisa::vector<uint32_t>{1u});
            expect(result.frame_values[1u].type == Type::of<float>());
        }
        expect(result.scopes.size() == 2u);
        if (result.scopes.size() == 2u) {
            expect(result.scopes[1u].live_in_frame_value_indices ==
                   luisa::vector<size_t>{0u, 1u});
        }
        const CoroCfgDistillResult::Edge *suspend_edge = nullptr;
        for (auto &edge : result.transition_edges) {
            if (edge.is_suspend && edge.token == 211u) {
                suspend_edge = &edge;
                break;
            }
        }
        expect(suspend_edge != nullptr);
        if (suspend_edge != nullptr) {
            expect(suspend_edge->live_frame_value_indices ==
                   luisa::vector<size_t>{0u, 1u});
            expect(suspend_edge->store_frame_value_indices ==
                   luisa::vector<size_t>{0u, 1u});
        }
    };

    "frame_abi_decomposes_complete_ssa_float3_value"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *tick = b.clock();
        auto *x = b.static_cast_(Type::of<float>(), tick);
        auto *one = m.create_constant_one(Type::of<float>());
        auto *y = b.call(
            Type::of<float>(), ArithmeticOp::BINARY_ADD, {x, one});
        auto *z = b.call(
            Type::of<float>(), ArithmeticOp::BINARY_ADD, {y, one});
        auto *state = b.call(
            Type::of<float3>(), ArithmeticOp::AGGREGATE, {x, y, z});
        state->set_name("ssa_float3_state");
        b.coro_suspend(213u, "ssa-float3", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(213u, nullptr);
        static_cast<void>(b.call(
            Type::of<float3>(), ArithmeticOp::BINARY_ADD,
            {state, m.create_constant_zero(Type::of<float3>())}));
        b.return_void();

        auto result = coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(Type::of<float3>()->size() == 16u);
        expect(result.frame_values.size() == 3u);
        if (result.frame_values.size() == 3u) {
            for (auto i = 0u; i < 3u; ++i) {
                expect(result.frame_values[i].value == state);
                expect(result.frame_values[i].access_chain ==
                       luisa::vector<uint32_t>{i});
                expect(result.frame_values[i].type == Type::of<float>());
            }
        }
        expect(result.frame_slots.size() == 3u);
        expect(result.scopes.size() == 2u);
        if (result.scopes.size() == 2u) {
            expect(result.scopes[1u].live_in_frame_value_indices ==
                   luisa::vector<size_t>{0u, 1u, 2u});
        }
    };

    "interfering_boolean_frame_values_pack_into_distinct_uint_bits"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *tick = b.clock();
        luisa::vector<Value *> flags;
        for (auto i = 0u; i < 6u; ++i) {
            auto threshold = uint64_t{i + 1u};
            auto *limit = m.create_constant(
                Type::of<uint64_t>(), &threshold);
            auto *flag = b.call(
                Type::of<bool>(), ArithmeticOp::BINARY_LESS,
                {tick, limit});
            flag->set_name(luisa::format("packed_flag_{}", i));
            flags.emplace_back(flag);
        }
        b.coro_suspend(217u, "packed-bools", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(217u, nullptr);
        auto *combined = flags.front();
        for (auto *flag : luisa::span{flags}.subspan(1u)) {
            combined = b.call(
                Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
                {combined, flag});
        }
        static_cast<void>(combined);
        b.return_void();

        auto result = coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(result.frame_values.size() == 6u);
        expect(result.frame_slots.size() == 1u);
        if (result.frame_slots.size() == 1u) {
            expect(result.frame_slots.front().type == Type::of<uint>());
        }
        if (result.frame_values.size() == 6u) {
            for (auto i = 0u; i < 6u; ++i) {
                expect(result.frame_values[i].value == flags[i]);
                expect(result.frame_values[i].type == Type::of<bool>());
                expect(result.frame_values[i].slot == 0u);
                expect(result.frame_values[i].bit_offset.has_value());
                if (result.frame_values[i].bit_offset) {
                    expect(*result.frame_values[i].bit_offset == i);
                }
            }
        }
    };

    "frame_abi_keeps_no_padding_aggregate_whole"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume = kernel->create_basic_block();
        auto *packed = Type::of<float2>();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *state = b.alloca_local(packed);
        state->set_name("packed_state");
        b.store(state, m.create_constant_zero(packed));
        b.coro_suspend(223u, "packed", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(223u, nullptr);
        static_cast<void>(b.load(packed, state));
        b.return_void();

        auto result = coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(result.frame_values.size() == 1u);
        if (result.frame_values.size() == 1u) {
            expect(result.frame_values.front().value == state);
            expect(result.frame_values.front().access_chain.empty());
            expect(result.frame_values.front().type == packed);
        }
    };

    "frame_slot_order_fills_fixed_prefix_alignment_hole"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume = kernel->create_basic_block();
        auto *packed_type = Type::of<float2>();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *packed = b.alloca_local(packed_type);
        auto *scalar = b.alloca_local(Type::of<float>());
        packed->set_name("packed_state");
        scalar->set_name("scalar_state");
        b.store(packed, m.create_constant_zero(packed_type));
        b.store(scalar, m.create_constant_zero(Type::of<float>()));
        b.coro_suspend(225u, "prefix-hole", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(225u, nullptr);
        static_cast<void>(b.load(packed_type, packed));
        static_cast<void>(b.load(Type::of<float>(), scalar));
        b.return_void();

        auto result = coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(result.frame_values.size() == 2u);
        expect(result.frame_slots.size() == 2u);
        if (result.frame_slots.size() == 2u) {
            // Seven reserved uints end at byte 28. Scheduling the scalar first
            // reaches byte 32 without padding, so float2 can follow at its
            // natural alignment. Alignment-descending order would occupy 48 B.
            expect(result.frame_slots[0u].type == Type::of<float>());
            expect(result.frame_slots[1u].type == packed_type);
            constexpr size_t scheduler_reserved_field_count = 7u;
            luisa::vector<const Type *> members(
                scheduler_reserved_field_count, Type::of<uint>());
            for (auto &slot : result.frame_slots) {
                members.emplace_back(slot.type);
            }
            expect(Type::structure(members)->size() == 40u);
        }
        for (auto &value : result.frame_values) {
            if (value.value == scalar) { expect(value.slot == 0u); }
            if (value.value == packed) { expect(value.slot == 1u); }
        }
    };

    "frame_abi_field_limit_keeps_large_aggregate_whole"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume = kernel->create_basic_block();
        // Eleven padded float3 elements would require 33 scalar fields. The
        // bounded planner must retain the aggregate instead of exploding the
        // generated continuation ABI and spill code.
        auto *large = Type::array(Type::of<float3>(), 11u);
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *state = b.alloca_local(large);
        state->set_name("large_state");
        b.store(state, m.create_constant_zero(large));
        b.coro_suspend(227u, "large", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(227u, nullptr);
        static_cast<void>(b.load(large, state));
        b.return_void();

        auto result = coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(result.frame_values.size() == 1u);
        if (result.frame_values.size() == 1u) {
            expect(result.frame_values.front().value == state);
            expect(result.frame_values.front().access_chain.empty());
            expect(result.frame_values.front().type == large);
        }
    };

    "disjoint_partial_store_preserves_dormant_field_in_frame"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *pair_type = Type::structure({Type::of<float>(), Type::of<float>()});
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *state = b.alloca_local(pair_type);
        state->set_name("state");
        b.store(state, m.create_constant_zero(pair_type));
        b.coro_suspend(1u, "first", nullptr);

        auto *resume_first = k->create_basic_block();
        b.set_insertion_point(resume_first);
        b.coro_resume(1u, nullptr);
        uint32_t first_index = 0u;
        auto *first = m.create_constant(Type::of<uint32_t>(), &first_index);
        auto *first_ptr = b.gep(Type::of<float>(), state, {first});
        b.store(first_ptr, m.create_constant_one(Type::of<float>()));
        b.coro_suspend(2u, "second", nullptr);

        auto *resume_second = k->create_basic_block();
        b.set_insertion_point(resume_second);
        b.coro_resume(2u, nullptr);
        uint32_t second_index = 1u;
        auto *second = m.create_constant(Type::of<uint32_t>(), &second_index);
        auto *second_ptr = b.gep(Type::of<float>(), state, {second});
        auto *value = b.load(Type::of<float>(), second_ptr);
        static_cast<void>(value);
        b.return_void();

        auto result = coro_cfg_distill_pass_run_on_function(k);
        expect(result.succeeded());
        expect(result.frame_values.size() == 1u);
        if (result.frame_values.size() == 1u) {
            expect(result.frame_values.front().value == state);
            expect(result.frame_values.front().type == Type::of<float>());
            expect(result.frame_values.front().access_chain ==
                   luisa::vector<uint32_t>{1u});
        }
        expect(result.scopes.size() == 3u);
        if (result.scopes.size() == 3u) {
            // Scope 1 writes field 0 only. Field 1 remains resident in the
            // frame and must not be reloaded merely to store it unchanged at
            // the next suspension.
            expect(result.scopes[1u]
                       .live_in_frame_value_indices.empty());
            expect(result.scopes[1u].live_in_values.empty());
        }
        const CoroCfgDistillResult::Edge *first_edge = nullptr;
        const CoroCfgDistillResult::Edge *second_edge = nullptr;
        for (auto &edge : result.transition_edges) {
            if (!edge.is_suspend) { continue; }
            if (edge.token == 1u) { first_edge = &edge; }
            if (edge.token == 2u) { second_edge = &edge; }
        }
        expect(first_edge != nullptr);
        expect(second_edge != nullptr);
        if (first_edge != nullptr) {
            expect(first_edge->store_frame_value_indices.size() == 1u);
        }
        if (second_edge != nullptr) {
            expect(second_edge->live_frame_value_indices.size() == 1u);
            expect(second_edge->store_frame_value_indices.empty());
            expect(std::find(second_edge->live_values.begin(),
                             second_edge->live_values.end(), state) !=
                   second_edge->live_values.end());
            expect(std::find(second_edge->store_values.begin(),
                             second_edge->store_values.end(), state) ==
                   second_edge->store_values.end());
        }
    };

    "descendant_store_splits_enclosing_observation_without_reloading_sibling"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume_store = kernel->create_basic_block();
        auto *resume_load = kernel->create_basic_block();
        auto *pair = Type::structure(
            {Type::of<float>(), Type::of<float>()});
        auto *outer = Type::structure({pair, Type::of<float>()});
        uint32_t zero_value = 0u;
        auto *zero = m.create_constant(
            Type::of<uint32_t>(), &zero_value);
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *state = b.alloca_local(outer);
        state->set_name("enclosing_state");
        b.store(state, m.create_constant_zero(outer));
        b.coro_suspend(3u, "before-partial-store", nullptr);

        b.set_insertion_point(resume_store);
        b.coro_resume(3u, nullptr);
        auto *pair_pointer = b.gep(pair, state, {zero});
        auto *first_pointer = b.gep(
            Type::of<float>(), pair_pointer, {zero});
        b.store(first_pointer, m.create_constant_one(Type::of<float>()));
        b.coro_suspend(5u, "after-partial-store", nullptr);

        b.set_insertion_point(resume_load);
        b.coro_resume(5u, nullptr);
        auto *resumed_pair = b.gep(pair, state, {zero});
        static_cast<void>(b.load(pair, resumed_pair));
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto result = coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(result.frame_values.size() == 2u);
        if (result.frame_values.size() == 2u) {
            expect(result.frame_values[0u].value == state);
            expect(result.frame_values[0u].access_chain ==
                   (luisa::vector<uint32_t>{0u, 0u}));
            expect(result.frame_values[0u].type == Type::of<float>());
            expect(result.frame_values[1u].value == state);
            expect(result.frame_values[1u].access_chain ==
                   (luisa::vector<uint32_t>{0u, 1u}));
            expect(result.frame_values[1u].type == Type::of<float>());
        }
        expect(result.scopes.size() == 3u);
        if (result.scopes.size() == 3u) {
            // pair.x is defined in this scope while pair.y remains resident
            // in its independent frame slot. Neither field needs an entry
            // reload before the write.
            expect(result.scopes[1u]
                       .live_in_frame_value_indices.empty());
        }
        const CoroCfgDistillResult::Edge *first_edge = nullptr;
        const CoroCfgDistillResult::Edge *second_edge = nullptr;
        for (auto &edge : result.transition_edges) {
            if (!edge.is_suspend) { continue; }
            if (edge.token == 3u) { first_edge = &edge; }
            if (edge.token == 5u) { second_edge = &edge; }
        }
        expect(first_edge != nullptr);
        expect(second_edge != nullptr);
        if (first_edge != nullptr) {
            expect(first_edge->store_frame_value_indices.size() == 1u);
        }
        if (second_edge != nullptr) {
            expect(second_edge->store_frame_value_indices.size() == 1u);
        }
    };

    "dynamic_descendant_store_preserves_unsplit_aggregate"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *selector =
            kernel->create_value_argument(Type::of<uint32_t>());
        auto *resume_store = kernel->create_basic_block();
        auto *resume_load = kernel->create_basic_block();
        auto *pair = Type::array(Type::of<float>(), 2u);
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *state = b.alloca_local(pair);
        state->set_name("dynamic_partial_state");
        b.store(state, m.create_constant_zero(pair));
        b.coro_suspend(7u, "before-dynamic-store", nullptr);

        b.set_insertion_point(resume_store);
        b.coro_resume(7u, nullptr);
        auto *element = b.gep(Type::of<float>(), state, {selector});
        b.store(element, m.create_constant_one(Type::of<float>()));
        b.coro_suspend(11u, "after-dynamic-store", nullptr);

        b.set_insertion_point(resume_load);
        b.coro_resume(11u, nullptr);
        static_cast<void>(b.load(pair, state));
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto result = coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(result.frame_values.size() == 1u);
        if (result.frame_values.size() == 1u) {
            expect(result.frame_values.front().value == state);
            expect(result.frame_values.front().access_chain.empty());
            expect(result.frame_values.front().type == pair);
        }
        expect(result.scopes.size() == 3u);
        if (result.scopes.size() == 3u) {
            expect(result.scopes[1u]
                       .live_in_frame_value_indices.size() == 1u);
        }
        for (auto token : {7u, 11u}) {
            auto found = false;
            for (auto &edge : result.transition_edges) {
                if (edge.is_suspend && edge.token == token) {
                    expect(edge.store_frame_value_indices.size() == 1u);
                    found = true;
                }
            }
            expect(found);
        }
    };

    "duplicate_alloca_names_get_distinct_frame_field_names"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *lhs = b.alloca_local(Type::of<float>());
        auto *rhs = b.alloca_local(Type::of<float>());
        lhs->set_name("duplicate");
        rhs->set_name("duplicate");
        b.store(lhs, m.create_constant_one(Type::of<float>()));
        b.store(rhs, m.create_constant_one(Type::of<float>()));
        b.coro_suspend(1u, "checkpoint", nullptr);

        auto *resume = k->create_basic_block();
        b.set_insertion_point(resume);
        b.coro_resume(1u, nullptr);
        auto *lhs_value = b.load(Type::of<float>(), lhs);
        auto *rhs_value = b.load(Type::of<float>(), rhs);
        auto *sum = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD,
                           {lhs_value, rhs_value});
        static_cast<void>(sum);
        b.return_void();

        auto result = coro_cfg_distill_pass_run_on_function(k);
        expect(result.frame_values.size() == 2u);
        expect(result.frame_values[0u].name != result.frame_values[1u].name);
    };

    "structured_switch_is_rejected_until_destructured"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *selector = k->create_value_argument(Type::of<uint32_t>());
        auto *entry = k->create_body_block();
        auto *case_block = k->create_basic_block();
        auto *default_block = k->create_basic_block();
        auto *merge = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *switch_inst = b.switch_(selector);
        switch_inst->set_default_block(default_block);
        switch_inst->add_case(7u, case_block);
        switch_inst->set_merge_block(merge);
        b.set_insertion_point(case_block);
        b.br(merge);
        b.set_insertion_point(default_block);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto before = xir_to_text_translate(&m, true);
        auto rejected = coro_cfg_distill_pass_run_on_function(k);
        expect(!rejected.succeeded());
        expect(rejected.structured_cfg_error_count == 1u);
        expect(rejected.invalid_input_error_count == 0u);
        expect(rejected.invalid_cfg_error_count == 0u);
        expect(rejected.scopes.empty());
        expect(xir_to_text_translate(&m, true) == before);
        expect(xir_verify_module(&m).succeeded());

        auto destructured = destructure_cfg_pass_run_on_function(k);
        expect(destructured.succeeded());
        expect(destructured.destructured_switch_count == 1u);
        expect(entry->terminator()->isa<IndexedBranchInst>());
        auto accepted = coro_cfg_distill_pass_run_on_function(k);
        expect(accepted.succeeded());
        expect(accepted.scopes.size() == 1u);
        expect(xir_verify_module(&m).succeeded());
    };

    "null_and_declaration_inputs_fail_closed"_test = [] {
        Module m;
        auto *external = m.create_external_function(nullptr);
        auto null_result =
            coro_cfg_distill_pass_run_on_function(nullptr);
        auto external_result =
            coro_cfg_distill_pass_run_on_function(external);
        expect(!null_result.succeeded());
        expect(null_result.invalid_input_error_count == 1u);
        expect(null_result.scopes.empty());
        expect(!external_result.succeeded());
        expect(external_result.invalid_input_error_count == 1u);
        expect(external_result.scopes.empty());
        expect(coro_cfg_distill_pass_run_on_module(nullptr) == 0u);
    };

    "missing_and_duplicate_coroutine_tokens_fail_closed"_test = [] {
        {
            Module m;
            auto *k = m.create_kernel();
            auto *body = k->create_body_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            b.coro_suspend(7u, "missing_resume", nullptr);

            expect(xir_verify_module(&m).succeeded());
            auto *before = body->terminator();
            auto result = coro_cfg_distill_pass_run_on_function(k);
            expect(!result.succeeded());
            expect(result.invalid_cfg_error_count == 1u);
            expect(result.scopes.empty());
            expect(body->terminator() == before);
            expect(xir_verify_module(&m).succeeded());
        }
        {
            Module m;
            auto *k = m.create_kernel();
            auto *body = k->create_body_block();
            auto *resume0 = k->create_basic_block();
            auto *resume1 = k->create_basic_block();
            XIRBuilder b;
            b.set_insertion_point(body);
            b.coro_suspend(9u, "duplicate_resume", nullptr);
            b.set_insertion_point(resume0);
            b.coro_resume(9u, nullptr);
            b.return_void();
            b.set_insertion_point(resume1);
            b.coro_resume(9u, nullptr);
            b.return_void();

            expect(xir_verify_module(&m).succeeded());
            auto *before_suspend = body->terminator();
            auto *before_resume0 = resume0->terminator();
            auto *before_resume1 = resume1->terminator();
            auto result = coro_cfg_distill_pass_run_on_function(k);
            expect(!result.succeeded());
            expect(result.invalid_cfg_error_count == 1u);
            expect(result.scopes.empty());
            expect(body->terminator() == before_suspend);
            expect(resume0->terminator() == before_resume0);
            expect(resume1->terminator() == before_resume1);
            expect(xir_verify_module(&m).succeeded());
        }
    };

    "phi_cfg_is_rejected_until_reg2mem_makes_edges_explicit"_test = [] {
        Module m;
        auto *callable = m.create_callable(nullptr);
        auto *condition =
            callable->create_value_argument(Type::of<bool>());
        auto *entry = callable->create_body_block();
        auto *left = callable->create_basic_block();
        auto *right = callable->create_basic_block();
        auto *join = callable->create_basic_block();
        auto *resume = callable->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.cond_br(condition, left, right);
        b.set_insertion_point(left);
        b.br(join);
        b.set_insertion_point(right);
        b.br(join);
        b.set_insertion_point(join);
        auto *phi = b.phi(Type::of<int>());
        phi->add_incoming(
            m.create_constant_zero(Type::of<int>()), left);
        phi->add_incoming(
            m.create_constant_one(Type::of<int>()), right);
        phi->set_name("edge_selected_value");
        auto *sum = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD,
            {phi, m.create_constant_one(Type::of<int>())});
        static_cast<void>(sum);
        b.coro_suspend(17u, "phi", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(17u, nullptr);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto rejected =
            coro_cfg_distill_pass_run_on_function(callable);
        expect(!rejected.succeeded());
        expect(rejected.invalid_cfg_error_count == 1u);
        expect(rejected.scopes.empty());
        expect(phi->is_linked());

        auto lowered = coro_reg2mem_pass_run_on_module(&m);
        expect(lowered.lowered_phi_count == 1u);
        expect(!phi->is_linked());
        expect(xir_verify_module(&m).succeeded());
        auto accepted =
            coro_cfg_distill_pass_run_on_function(callable);
        expect(accepted.succeeded());
        expect(accepted.scopes.size() == 2u);
    };

    "two_resume_identities_cannot_share_one_block"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *left_suspend = kernel->create_basic_block();
        auto *right_suspend = kernel->create_basic_block();
        auto *shared_resume = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.cond_br(
            kernel->create_value_argument(Type::of<bool>()),
            left_suspend, right_suspend);
        b.set_insertion_point(left_suspend);
        b.coro_suspend(21u, "left", nullptr);
        b.set_insertion_point(right_suspend);
        b.coro_suspend(22u, "right", nullptr);
        b.set_insertion_point(shared_resume);
        auto *first = b.coro_resume(21u, nullptr);
        auto *second = b.coro_resume(22u, nullptr);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto result =
            coro_cfg_distill_pass_run_on_function(kernel);
        expect(!result.succeeded());
        expect(result.invalid_cfg_error_count == 1u);
        expect(result.scopes.empty());
        expect(first->is_linked());
        expect(second->is_linked());
        expect(xir_verify_module(&m).succeeded());
    };

    "entry_block_cannot_alias_a_resume_root"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *suspend = kernel->create_basic_block();
        auto *exit = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *resume = b.coro_resume(27u, nullptr);
        b.cond_br(
            kernel->create_value_argument(Type::of<bool>()),
            suspend, exit);
        b.set_insertion_point(suspend);
        auto *suspend_inst =
            b.coro_suspend(27u, "entry-alias", nullptr);
        b.set_insertion_point(exit);
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto result =
            coro_cfg_distill_pass_run_on_function(kernel);
        expect(!result.succeeded());
        expect(result.invalid_cfg_error_count == 1u);
        expect(result.scopes.empty());
        expect(resume->is_linked());
        expect(suspend->terminator() == suspend_inst);
        expect(xir_verify_module(&m).succeeded());
    };

    "non_void_coroutine_is_rejected_before_continuation_abi_change"_test = [] {
        Module m;
        auto *callable = m.create_callable(Type::of<int>());
        auto *entry = callable->create_body_block();
        auto *resume = callable->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *suspend =
            b.coro_suspend(31u, "non-void", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(31u, nullptr);
        b.return_(m.create_constant_one(Type::of<int>()));

        expect(xir_verify_module(&m).succeeded());
        auto result =
            coro_cfg_distill_pass_run_on_function(callable);
        expect(!result.succeeded());
        expect(result.invalid_cfg_error_count == 1u);
        expect(result.scopes.empty());
        expect(entry->terminator() == suspend);
        expect(xir_verify_module(&m).succeeded());
    };

    "cheap_argument_rooted_expression_is_replayed_not_framed"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *x = kernel->create_argument(Type::of<float>(), false);
        auto *resume = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *one = m.create_constant_one(Type::of<float>());
        auto *replay = b.call(
            Type::of<float>(), ArithmeticOp::BINARY_ADD,
            {x, one});
        b.coro_suspend(41u, "replay", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(41u, nullptr);
        static_cast<void>(b.call(
            Type::of<float>(), ArithmeticOp::BINARY_MUL,
            {replay, one}));
        b.return_void();

        auto result =
            coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(result.scopes.size() == 2u);
        expect(std::none_of(
            result.frame_values.begin(), result.frame_values.end(),
            [&](const auto &field) noexcept {
                return field.value == replay;
            }));
        expect(result.scopes[1u].live_in_values.empty());
    };

    "replay_cost_is_bounded_to_prevent_continuation_code_growth"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *x = kernel->create_argument(Type::of<float>(), false);
        auto *resume = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *one = m.create_constant_one(Type::of<float>());
        auto *v1 = b.call(
            Type::of<float>(), ArithmeticOp::BINARY_ADD,
            {x, one});
        auto *v2 = b.call(
            Type::of<float>(), ArithmeticOp::BINARY_ADD,
            {v1, one});
        auto *v3 = b.call(
            Type::of<float>(), ArithmeticOp::BINARY_ADD,
            {v2, one});
        b.coro_suspend(43u, "bounded-replay", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(43u, nullptr);
        static_cast<void>(b.call(
            Type::of<float>(), ArithmeticOp::BINARY_MUL,
            {v3, one}));
        b.return_void();

        auto result =
            coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(std::any_of(
            result.frame_values.begin(), result.frame_values.end(),
            [&](const auto &field) noexcept {
                return field.value == v3;
            }));
        expect(std::find(
                   result.scopes[1u].live_in_values.begin(),
                   result.scopes[1u].live_in_values.end(),
                   v3) !=
               result.scopes[1u].live_in_values.end());
    };

    "expression_depending_on_load_is_never_replayed"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *state = b.alloca_local(Type::of<float>());
        auto *one = m.create_constant_one(Type::of<float>());
        b.store(state, one);
        auto *loaded = b.load(Type::of<float>(), state);
        auto *derived = b.call(
            Type::of<float>(), ArithmeticOp::BINARY_ADD,
            {loaded, one});
        b.coro_suspend(47u, "loaded-value", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(47u, nullptr);
        static_cast<void>(b.call(
            Type::of<float>(), ArithmeticOp::BINARY_MUL,
            {derived, one}));
        b.return_void();

        auto result =
            coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(std::any_of(
            result.frame_values.begin(), result.frame_values.end(),
            [&](const auto &field) noexcept {
                return field.value == derived;
            }));
    };

    "disjoint_anonymous_values_share_one_exact_typed_frame_slot"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume_first = kernel->create_basic_block();
        auto *resume_second = kernel->create_basic_block();
        XIRBuilder b;
        auto *one = m.create_constant_one(Type::of<uint64_t>());
        b.set_insertion_point(entry);
        auto *first = b.clock();
        b.coro_suspend(59u, "first", nullptr);
        b.set_insertion_point(resume_first);
        b.coro_resume(59u, nullptr);
        static_cast<void>(b.call(
            Type::of<uint64_t>(), ArithmeticOp::BINARY_ADD,
            {first, one}));
        auto *second = b.clock();
        b.coro_suspend(61u, "second", nullptr);
        b.set_insertion_point(resume_second);
        b.coro_resume(61u, nullptr);
        static_cast<void>(b.call(
            Type::of<uint64_t>(), ArithmeticOp::BINARY_ADD,
            {second, one}));
        b.return_void();

        auto result =
            coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(result.frame_values.size() == 2u);
        expect(result.frame_slots.size() == 1u);
        auto first_field = std::find_if(
            result.frame_values.begin(), result.frame_values.end(),
            [&](const auto &field) noexcept {
                return field.value == first;
            });
        auto second_field = std::find_if(
            result.frame_values.begin(), result.frame_values.end(),
            [&](const auto &field) noexcept {
                return field.value == second;
            });
        expect(first_field != result.frame_values.end());
        expect(second_field != result.frame_values.end());
        if (first_field != result.frame_values.end() &&
            second_field != result.frame_values.end()) {
            expect(first_field->slot == second_field->slot);
            expect(first_field->type == second_field->type);
        }
    };

    "simultaneously_live_values_interfere_in_frame_coloring"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *lhs = b.clock();
        auto *rhs = b.clock();
        b.coro_suspend(67u, "pair", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(67u, nullptr);
        static_cast<void>(b.call(
            Type::of<uint64_t>(), ArithmeticOp::BINARY_ADD,
            {lhs, rhs}));
        b.return_void();

        auto result =
            coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(result.frame_values.size() == 2u);
        expect(result.frame_slots.size() == 2u);
        if (result.frame_values.size() == 2u) {
            expect(result.frame_values[0u].slot !=
                   result.frame_values[1u].slot);
        }
    };

    "dormant_pass_through_value_interferes_with_transition_store"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume_first = kernel->create_basic_block();
        auto *resume_second = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *pass_through = b.clock();
        b.coro_suspend(107u, "first", nullptr);
        b.set_insertion_point(resume_first);
        b.coro_resume(107u, nullptr);
        auto *newly_stored = b.clock();
        b.coro_suspend(109u, "second", nullptr);
        b.set_insertion_point(resume_second);
        b.coro_resume(109u, nullptr);
        static_cast<void>(b.call(
            Type::of<uint64_t>(), ArithmeticOp::BINARY_ADD,
            {pass_through, newly_stored}));
        b.return_void();

        auto result =
            coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(result.frame_values.size() == 2u);
        expect(result.frame_slots.size() == 2u);
        const CoroCfgDistillResult::Edge *second_edge = nullptr;
        for (auto &edge : result.transition_edges) {
            if (edge.is_suspend && edge.token == 109u) {
                second_edge = &edge;
                break;
            }
        }
        expect(second_edge != nullptr);
        if (second_edge != nullptr) {
            expect(std::find(
                       second_edge->live_values.begin(),
                       second_edge->live_values.end(),
                       pass_through) != second_edge->live_values.end());
            expect(std::find(
                       second_edge->live_values.begin(),
                       second_edge->live_values.end(),
                       newly_stored) != second_edge->live_values.end());
            expect(std::find(
                       second_edge->store_values.begin(),
                       second_edge->store_values.end(),
                       pass_through) == second_edge->store_values.end());
            expect(std::find(
                       second_edge->store_values.begin(),
                       second_edge->store_values.end(),
                       newly_stored) != second_edge->store_values.end());
        }
    };

    "ssa_metadata_names_do_not_prevent_safe_slot_sharing"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume_first = kernel->create_basic_block();
        auto *resume_second = kernel->create_basic_block();
        XIRBuilder b;
        auto *one = m.create_constant_one(Type::of<uint64_t>());
        b.set_insertion_point(entry);
        auto *first = b.clock();
        first->set_name("named_first");
        b.coro_suspend(71u, "first", nullptr);
        b.set_insertion_point(resume_first);
        b.coro_resume(71u, nullptr);
        static_cast<void>(b.call(
            Type::of<uint64_t>(), ArithmeticOp::BINARY_ADD,
            {first, one}));
        auto *second = b.clock();
        second->set_name("named_second");
        b.coro_suspend(73u, "second", nullptr);
        b.set_insertion_point(resume_second);
        b.coro_resume(73u, nullptr);
        static_cast<void>(b.call(
            Type::of<uint64_t>(), ArithmeticOp::BINARY_ADD,
            {second, one}));
        b.return_void();

        auto result =
            coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(result.frame_values.size() == 2u);
        expect(result.frame_slots.size() == 1u);
        if (result.frame_values.size() == 2u) {
            expect(result.frame_values[0u].slot ==
                   result.frame_values[1u].slot);
        }
    };

    "disjoint_unnamed_alloca_values_share_frame_storage"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume_first = kernel->create_basic_block();
        auto *resume_second = kernel->create_basic_block();
        XIRBuilder b;
        auto *one = m.create_constant_one(Type::of<float>());
        b.set_insertion_point(entry);
        auto *first = b.alloca_local(Type::of<float>());
        auto *second = b.alloca_local(Type::of<float>());
        b.store(first, one);
        b.coro_suspend(89u, "first", nullptr);
        b.set_insertion_point(resume_first);
        b.coro_resume(89u, nullptr);
        static_cast<void>(b.load(Type::of<float>(), first));
        b.store(second, one);
        b.coro_suspend(97u, "second", nullptr);
        b.set_insertion_point(resume_second);
        b.coro_resume(97u, nullptr);
        static_cast<void>(b.load(Type::of<float>(), second));
        b.return_void();

        auto result =
            coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(result.frame_values.size() == 2u);
        expect(result.frame_slots.size() == 1u);
        if (result.frame_values.size() == 2u) {
            expect(result.frame_values[0u].slot ==
                   result.frame_values[1u].slot);
        }
    };

    "named_allocas_share_storage_but_keep_logical_aliases"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume_first = kernel->create_basic_block();
        auto *resume_second = kernel->create_basic_block();
        XIRBuilder b;
        auto *one = m.create_constant_one(Type::of<float>());
        b.set_insertion_point(entry);
        auto *first = b.alloca_local(Type::of<float>());
        auto *second = b.alloca_local(Type::of<float>());
        first->set_name("named_first");
        second->set_name("named_second");
        b.store(first, one);
        b.coro_suspend(101u, "first", nullptr);
        b.set_insertion_point(resume_first);
        b.coro_resume(101u, nullptr);
        static_cast<void>(b.load(Type::of<float>(), first));
        b.store(second, one);
        b.coro_suspend(103u, "second", nullptr);
        b.set_insertion_point(resume_second);
        b.coro_resume(103u, nullptr);
        static_cast<void>(b.load(Type::of<float>(), second));
        b.return_void();

        auto result =
            coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(result.frame_values.size() == 2u);
        expect(result.frame_slots.size() == 1u);
        if (result.frame_values.size() == 2u) {
            expect(result.frame_values[0u].slot ==
                   result.frame_values[1u].slot);
            expect(result.frame_values[0u].name !=
                   result.frame_values[1u].name);
        }
    };

    "static_disjoint_aggregate_paths_form_independent_frame_atoms"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume = kernel->create_basic_block();
        auto *pair = Type::structure(
            {Type::of<float>(), Type::of<float>()});
        XIRBuilder b;
        uint32_t zero_value = 0u;
        uint32_t one_value = 1u;
        auto *zero = m.create_constant(
            Type::of<uint32_t>(), &zero_value);
        auto *one = m.create_constant(
            Type::of<uint32_t>(), &one_value);
        b.set_insertion_point(entry);
        auto *state = b.alloca_local(pair);
        state->set_name("state");
        auto *entry_first = b.gep(Type::of<float>(), state, {zero});
        auto *entry_second = b.gep(Type::of<float>(), state, {one});
        b.store(entry_first, m.create_constant_zero(Type::of<float>()));
        b.store(entry_second, m.create_constant_one(Type::of<float>()));
        b.coro_suspend(113u, "aggregate-path", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(113u, nullptr);
        auto *resume_second = b.gep(Type::of<float>(), state, {one});
        static_cast<void>(b.load(Type::of<float>(), resume_second));
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto result = coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(result.frame_values.size() == 1u);
        expect(result.frame_slots.size() == 1u);
        if (result.frame_values.size() == 1u) {
            auto &value = result.frame_values.front();
            expect(value.value == state);
            expect(value.type == Type::of<float>());
            expect(value.access_chain == luisa::vector<uint32_t>{1u});
            expect(value.name == "state.1");
        }
        expect(result.scopes.size() == 2u);
        if (result.scopes.size() == 2u) {
            expect(result.scopes[1u]
                       .live_in_frame_value_indices.size() == 1u);
            expect(result.scopes[1u].live_in_values.size() == 1u);
            if (!result.scopes[1u].live_in_values.empty()) {
                expect(result.scopes[1u].live_in_values.front() == state);
            }
        }
    };

    "flat_dynamic_aggregate_index_remains_one_whole_atom"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *selector =
            kernel->create_value_argument(Type::of<uint32_t>());
        auto *resume = kernel->create_basic_block();
        auto *pair = Type::array(Type::of<float>(), 2u);
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *state = b.alloca_local(pair);
        state->set_name("dynamic_state");
        auto *entry_element =
            b.gep(Type::of<float>(), state, {selector});
        b.store(entry_element, m.create_constant_one(Type::of<float>()));
        b.coro_suspend(127u, "dynamic-index", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(127u, nullptr);
        auto *resume_element =
            b.gep(Type::of<float>(), state, {selector});
        static_cast<void>(b.load(Type::of<float>(), resume_element));
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto result = coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(result.frame_values.size() == 1u);
        if (result.frame_values.size() == 1u) {
            expect(result.frame_values.front().value == state);
            expect(result.frame_values.front().type == pair);
            expect(result.frame_values.front().access_chain.empty());
            expect(result.frame_values.front().name == "dynamic_state");
        }
    };

    "nested_dynamic_index_excludes_unrelated_sibling_subaggregate"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *selector =
            kernel->create_value_argument(Type::of<uint32_t>());
        auto *resume = kernel->create_basic_block();
        auto *phase = Type::array(Type::of<float>(), 4u);
        auto *unrelated = Type::array(Type::of<float>(), 8u);
        auto *state_type = Type::structure({phase, unrelated});
        uint32_t zero_value = 0u;
        auto *zero = m.create_constant(
            Type::of<uint32_t>(), &zero_value);
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *state = b.alloca_local(state_type);
        state->set_name("nested_dynamic_state");
        auto *entry_element = b.gep(
            Type::of<float>(), state, {zero, selector});
        b.store(entry_element,
                m.create_constant_one(Type::of<float>()));
        b.coro_suspend(129u, "nested-dynamic-index", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(129u, nullptr);
        auto *resume_element = b.gep(
            Type::of<float>(), state, {zero, selector});
        static_cast<void>(
            b.load(Type::of<float>(), resume_element));
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto result = coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(result.frame_values.size() == 1u);
        if (result.frame_values.size() == 1u) {
            expect(result.frame_values.front().value == state);
            expect(result.frame_values.front().type == phase);
            expect(result.frame_values.front().access_chain ==
                   luisa::vector<uint32_t>{0u});
            expect(result.frame_values.front().name ==
                   "nested_dynamic_state.0");
        }
    };

    "typed_reference_escape_preserves_only_later_observed_subtree"_test = [] {
        Module m;
        auto *pair = Type::structure(
            {Type::of<float>(), Type::of<float>()});
        auto *observer = m.create_callable(nullptr);
        static_cast<void>(observer->create_reference_argument(pair));
        auto *observer_entry = observer->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(observer_entry);
        b.return_void();

        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume = kernel->create_basic_block();
        uint32_t one_value = 1u;
        auto *one = m.create_constant(
            Type::of<uint32_t>(), &one_value);
        b.set_insertion_point(entry);
        auto *state = b.alloca_local(pair);
        state->set_name("escaped_state");
        static_cast<void>(b.call(nullptr, observer, {state}));
        b.coro_suspend(131u, "reference-escape", nullptr);
        b.set_insertion_point(resume);
        b.coro_resume(131u, nullptr);
        auto *second = b.gep(Type::of<float>(), state, {one});
        static_cast<void>(b.load(Type::of<float>(), second));
        b.return_void();

        expect(xir_verify_module(&m).succeeded());
        auto result = coro_cfg_distill_pass_run_on_function(kernel);

        expect(result.succeeded());
        expect(result.frame_values.size() == 1u);
        if (result.frame_values.size() == 1u) {
            expect(result.frame_values.front().value == state);
            expect(result.frame_values.front().type == Type::of<float>());
            expect(result.frame_values.front().access_chain ==
                   luisa::vector<uint32_t>{1u});
        }
    };
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    reg_coro_cfg_distill();
    return 0;
}
