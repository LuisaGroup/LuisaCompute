// Test for XIR scalar-evolution analysis.
// This test covers:
// - owned result lifetime and mutation invalidation
// - malformed loop/backedge and null-module rejection
// - compatibility-cache lifetime invalidation

#include "ut/ut.hpp"

#include <luisa/ast/type.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/scalar_evolution.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct LoopFixture {
    PhiInst *phi;
    ArithmeticInst *increment;
    Constant *one;
};

[[nodiscard]] LoopFixture build_loop(Module &module, KernelFunction *function) noexcept {
    auto *body = function->create_body_block();
    XIRBuilder builder;
    builder.set_insertion_point(body);
    auto *loop = builder.loop();
    auto *prepare = loop->create_prepare_block();
    auto *loop_body = loop->create_body_block();
    auto *update = loop->create_update_block();
    auto *merge = loop->create_merge_block();
    auto *zero = module.create_constant_zero(Type::of<int32_t>());
    auto *one = module.create_constant_one(Type::of<int32_t>());
    int32_t bound_value = 8;
    auto *bound = module.create_constant(Type::of<int32_t>(), &bound_value);
    builder.set_insertion_point(prepare);
    auto *phi = builder.phi(Type::of<int32_t>(), {{zero, body}});
    auto *condition = builder.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {phi, bound});
    builder.cond_br(condition, loop_body, merge);
    builder.set_insertion_point(loop_body);
    builder.br(update);
    builder.set_insertion_point(update);
    auto *increment = builder.call(Type::of<int32_t>(), ArithmeticOp::BINARY_ADD, {phi, one});
    phi->add_incoming(increment, update);
    builder.br(prepare);
    builder.set_insertion_point(merge);
    builder.return_void();
    return {.phi = phi, .increment = increment, .one = one};
}

}// namespace

int main() {

    "scev_owned_handle_is_current_until_ir_mutation"_test = [] {
        Module module;
        auto *function = module.create_kernel();
        auto fixture = build_loop(module, function);
        SCEVAnalysis analysis;
        auto info = analysis.analyze(function);
        expect(info.succeeded());
        expect(info.analyzed_loop_count == 1u);
        expect(analysis.is_current());
        auto *owned = analysis.get(fixture.phi);
        expect(owned != nullptr);
        expect(owned->kind() == SCEV::Kind::ADD_REC);

        auto legacy_info = scev_pass_run_on_function(function);
        expect(legacy_info.succeeded());
        expect(analysis.get(fixture.phi) == owned);

        fixture.increment->set_operand(1u, module.create_constant_zero(Type::of<int32_t>()));
        expect(!analysis.is_current());
        expect(analysis.get(fixture.phi) == nullptr);
        expect(scev_get_for_value(fixture.phi) == nullptr);

        auto refreshed = analysis.analyze(function);
        expect(refreshed.succeeded());
        expect(analysis.is_current());
        auto *simplified = analysis.get(fixture.phi);
        expect(simplified != nullptr);
        expect(simplified->kind() == SCEV::Kind::CONSTANT);
    };

    "scev_rejects_malformed_loop_and_null_inputs"_test = [] {
        Module module;
        auto *function = module.create_kernel();
        auto *body = function->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.loop();

        SCEVAnalysis analysis;
        auto malformed = analysis.analyze(function);
        expect(!malformed.succeeded());
        expect(malformed.analyzed_loop_count == 0u);
        expect(malformed.rejected_loop_count == 1u);

        auto *empty_function = module.create_kernel();
        auto empty = analysis.analyze(empty_function);
        expect(!empty.succeeded());
        expect(empty.invalid_function_count == 1u);
        expect(!analysis.is_current());
        expect(analysis.function() == nullptr);
        auto null_module = scev_pass_run_on_module(nullptr);
        expect(!null_module.succeeded());
        expect(null_module.invalid_function_count == 1u);
    };

    "scev_nested_loops_use_deterministic_owning_loop"_test = [] {
        Module module;
        auto *function = module.create_kernel();
        auto *entry = function->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *outer = builder.loop();
        auto *outer_prepare = outer->create_prepare_block();
        auto *outer_body = outer->create_body_block();
        auto *outer_update = outer->create_update_block();
        auto *outer_merge = outer->create_merge_block();
        auto *zero = module.create_constant_zero(Type::of<int32_t>());
        auto *one = module.create_constant_one(Type::of<int32_t>());
        int32_t bound_value = 4;
        auto *bound = module.create_constant(Type::of<int32_t>(), &bound_value);

        builder.set_insertion_point(outer_prepare);
        auto *outer_phi = builder.phi(Type::of<int32_t>(), {{zero, entry}});
        auto *outer_condition = builder.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {outer_phi, bound});
        builder.cond_br(outer_condition, outer_body, outer_merge);

        builder.set_insertion_point(outer_body);
        auto *inner = builder.loop();
        auto *inner_prepare = inner->create_prepare_block();
        auto *inner_body = inner->create_body_block();
        auto *inner_update = inner->create_update_block();
        auto *inner_merge = inner->create_merge_block();

        builder.set_insertion_point(inner_prepare);
        auto *inner_phi = builder.phi(Type::of<int32_t>(), {{zero, outer_body}});
        auto *inner_condition = builder.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {inner_phi, bound});
        builder.cond_br(inner_condition, inner_body, inner_merge);
        builder.set_insertion_point(inner_body);
        auto *mixed = builder.call(
            Type::of<int32_t>(), ArithmeticOp::BINARY_ADD,
            {inner_phi, outer_phi});
        builder.br(inner_update);
        builder.set_insertion_point(inner_update);
        auto *inner_increment = builder.call(Type::of<int32_t>(), ArithmeticOp::BINARY_ADD, {inner_phi, one});
        inner_phi->add_incoming(inner_increment, inner_update);
        builder.br(inner_prepare);
        builder.set_insertion_point(inner_merge);
        builder.br(outer_update);

        builder.set_insertion_point(outer_update);
        auto *outer_increment = builder.call(Type::of<int32_t>(), ArithmeticOp::BINARY_ADD, {outer_phi, one});
        outer_phi->add_incoming(outer_increment, outer_update);
        builder.br(outer_prepare);
        builder.set_insertion_point(outer_merge);
        builder.return_void();

        SCEVAnalysis analysis;
        // Establish the normal storage order first, then deliberately move the
        // block containing the inner LoopInst before the entry block. Nested
        // SCEV ownership is a CFG property and must not depend on this
        // intrusive-list implementation detail.
        auto initial_info = analysis.analyze(function);
        expect(initial_info.succeeded());
        expect(initial_info.analyzed_loop_count == 2u);
        expect(analysis.get(outer_phi)->kind() == SCEV::Kind::ADD_REC);
        expect(analysis.get(inner_phi)->kind() == SCEV::Kind::ADD_REC);
        auto outer_body_owner = outer_body->remove_self();
        expect(outer_body_owner != nullptr);
        function->basic_blocks().push_front(std::move(outer_body_owner));
        expect(!analysis.is_current());
        for (auto iteration = 0u; iteration < 4u; ++iteration) {
            auto info = analysis.analyze(function);
            expect(info.succeeded());
            expect(info.analyzed_loop_count == 2u);
            auto *outer_scev = analysis.get(outer_phi);
            auto *inner_scev = analysis.get(inner_phi);
            auto *mixed_scev = analysis.get(mixed);
            expect(outer_scev != nullptr);
            expect(inner_scev != nullptr);
            expect(mixed_scev != nullptr);
            expect(outer_scev->kind() == SCEV::Kind::ADD_REC);
            expect(inner_scev->kind() == SCEV::Kind::ADD_REC);
            expect(mixed_scev->kind() == SCEV::Kind::ADD);
            expect(static_cast<const SCEVAddRec *>(outer_scev)->loop() == outer);
            expect(static_cast<const SCEVAddRec *>(inner_scev)->loop() == inner);
        }
        expect(xir_verify_module(&module).succeeded());
    };

    "scev_legacy_cache_is_cleared_with_function_lifetime"_test = [] {
        Instruction *expired = nullptr;
        {
            Module module;
            auto *function = module.create_kernel();
            auto fixture = build_loop(module, function);
            expired = fixture.phi;
            auto info = scev_pass_run_on_function(function);
            expect(info.succeeded());
            expect(scev_get_for_value(expired) != nullptr);
        }
        expect(scev_get_for_value(expired) == nullptr);
    };

    "scev_owned_handle_expires_with_function_lifetime"_test = [] {
        SCEVAnalysis analysis;
        Instruction *expired = nullptr;
        {
            Module module;
            auto *function = module.create_kernel();
            auto fixture = build_loop(module, function);
            expired = fixture.phi;
            auto info = analysis.analyze(function);
            expect(info.succeeded());
            expect(analysis.is_current());
            expect(analysis.get(expired) != nullptr);
        }
        expect(!analysis.is_current());
        expect(analysis.function() == nullptr);
        expect(analysis.get(expired) == nullptr);
    };

    "scev_rejects_loop_without_update_backedge"_test = [] {
        Module module;
        auto *function = module.create_kernel();
        auto *entry = function->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *loop = builder.loop();
        auto *prepare = loop->create_prepare_block();
        auto *body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        auto *zero = module.create_constant_zero(Type::of<int32_t>());
        auto *one = module.create_constant_one(Type::of<int32_t>());
        builder.set_insertion_point(prepare);
        auto *phi = builder.phi(Type::of<int32_t>(), {{zero, entry}});
        builder.cond_br(module.create_constant_one(Type::of<bool>()), body, update);
        builder.set_insertion_point(body);
        builder.br(merge);
        builder.set_insertion_point(update);
        auto *increment = builder.call(Type::of<int32_t>(), ArithmeticOp::BINARY_ADD, {phi, one});
        phi->add_incoming(increment, update);
        builder.br(merge);
        builder.set_insertion_point(merge);
        builder.return_void();

        SCEVAnalysis analysis;
        auto info = analysis.analyze(function);
        expect(!info.succeeded());
        expect(info.rejected_loop_count == 1u);
        expect(analysis.get(phi) == nullptr);
    };

    "scev_keeps_strict_float_recurrences_unknown"_test = [] {
        Module module;
        auto *function = module.create_kernel();
        auto *entry = function->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *loop = builder.loop();
        auto *prepare = loop->create_prepare_block();
        auto *body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        float zero_value = 0.0f;
        float one_value = 1.0f;
        float bound_value = 4.0f;
        auto *zero = module.create_constant(Type::of<float>(), &zero_value);
        auto *one = module.create_constant(Type::of<float>(), &one_value);
        auto *bound = module.create_constant(Type::of<float>(), &bound_value);

        builder.set_insertion_point(prepare);
        auto *phi = builder.phi(Type::of<float>(), {{zero, entry}});
        auto *condition = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS, {phi, bound});
        builder.cond_br(condition, body, merge);
        builder.set_insertion_point(body);
        builder.br(update);
        builder.set_insertion_point(update);
        auto *increment = builder.call(
            Type::of<float>(), ArithmeticOp::BINARY_ADD, {phi, one});
        phi->add_incoming(increment, update);
        builder.br(prepare);
        builder.set_insertion_point(merge);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        SCEVAnalysis analysis;
        auto info = analysis.analyze(function);
        expect(info.succeeded());
        expect(info.analyzed_loop_count == 1u);
        auto *phi_scev = analysis.get(phi);
        auto *increment_scev = analysis.get(increment);
        expect(phi_scev != nullptr);
        expect(increment_scev != nullptr);
        expect(phi_scev->kind() == SCEV::Kind::UNKNOWN);
        expect(increment_scev->kind() == SCEV::Kind::UNKNOWN);
        expect(xir_verify_module(&module).succeeded());
    };

    return 0;
}
