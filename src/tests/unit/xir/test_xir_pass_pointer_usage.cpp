// Test for XIR pointer-usage analysis.
// This test covers:
// - precise and conservative aggregate field masks
// - alias propagation, CFG joins, loops, and backward liveness
// - mutation and function-lifetime invalidation
// - malformed projection rejection

#include "ut/ut.hpp"

#include <luisa/ast/type.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/pointer_usage.h>

#include <array>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] Constant *uint_constant(Module &module, uint32_t value) noexcept {
    return module.create_constant(Type::of<uint32_t>(), &value);
}

}// namespace

int main() {

    "pointer_usage_precise_fields_and_backward_liveness"_test = [] {
        Module module;
        auto *function = module.create_kernel();
        auto *body = function->create_body_block();
        auto *pair_type = Type::structure({Type::of<int32_t>(), Type::array(Type::of<int32_t>(), 2u)});
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *root = builder.alloca_local(pair_type);
        auto *field0 = builder.gep(Type::of<int32_t>(), root, {uint_constant(module, 0u)});
        auto *field1 = builder.gep(Type::array(Type::of<int32_t>(), 2u), root, {uint_constant(module, 1u)});
        static_cast<void>(builder.load(field1->type(), field1));
        builder.store(field0, module.create_constant_one(Type::of<int32_t>()));
        builder.return_void();

        PointerUsageAnalysis analysis;
        auto info = analysis.analyze(function);
        expect(info.succeeded());
        expect(info.tracked_pointer_count == 3u);
        auto *root_in = analysis.in_usage(body, root);
        auto *root_out = analysis.out_usage(body, root);
        auto *field0_out = analysis.out_usage(body, field0);
        auto *field1_in = analysis.in_usage(body, field1);
        expect(root_in != nullptr);
        expect(root_out != nullptr);
        expect(field0_out != nullptr);
        expect(field1_in != nullptr);
        expect(root_in->live.access(0u).none());
        expect(root_in->live.access(1u).all());
        expect(root_out->kill.access(0u).all());
        expect(root_out->kill.access(1u).none());
        expect(root_out->touch.access(0u).all());
        expect(root_out->touch.access(1u).none());
        expect(field0_out->kill.access().all());
        expect(field1_in->live.access().all());
    };

    "pointer_usage_branch_kill_intersection"_test = [] {
        Module module;
        auto *function = module.create_kernel();
        auto *body = function->create_body_block();
        auto *left = function->create_basic_block();
        auto *right = function->create_basic_block();
        auto *merge = function->create_basic_block();
        auto *condition = function->create_value_argument(Type::of<bool>());
        auto *pair_type = Type::structure({Type::of<int32_t>(), Type::of<int32_t>()});
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *root = builder.alloca_local(pair_type);
        auto *field0 = builder.gep(Type::of<int32_t>(), root, {uint_constant(module, 0u)});
        builder.cond_br(condition, left, right);
        builder.set_insertion_point(left);
        builder.store(field0, module.create_constant_one(Type::of<int32_t>()));
        builder.br(merge);
        builder.set_insertion_point(right);
        builder.br(merge);
        builder.set_insertion_point(merge);
        builder.return_void();

        PointerUsageAnalysis analysis;
        auto info = analysis.analyze(function);
        expect(info.succeeded());
        auto *merge_in = analysis.in_usage(merge, root);
        expect(merge_in != nullptr);
        expect(merge_in->kill.access(0u).none());
        expect(merge_in->touch.access(0u).all());
        expect(merge_in->kill.access(1u).none());
        expect(merge_in->touch.access(1u).none());
    };

    "pointer_usage_dynamic_index_is_conservative_for_base"_test = [] {
        Module module;
        auto *function = module.create_kernel();
        auto *index = function->create_value_argument(Type::of<uint32_t>());
        auto *body = function->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *root = builder.alloca_local(Type::array(Type::of<int32_t>(), 4u));
        auto *element = builder.gep(Type::of<int32_t>(), root, {index});
        builder.store(element, module.create_constant_one(Type::of<int32_t>()));
        builder.return_void();

        PointerUsageAnalysis analysis;
        auto info = analysis.analyze(function);
        expect(info.succeeded());
        expect(info.conservative_access_count >= 1u);
        auto *root_out = analysis.out_usage(body, root);
        auto *element_out = analysis.out_usage(body, element);
        expect(root_out != nullptr);
        expect(element_out != nullptr);
        expect(root_out->touch.access().all());
        expect(root_out->kill.access().none());
        expect(element_out->touch.access().all());
        expect(element_out->kill.access().all());
    };

    "pointer_usage_rejects_malformed_projection"_test = [] {
        Module module;
        auto *function = module.create_kernel();
        auto *body = function->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *root = builder.alloca_local(Type::array(Type::of<int32_t>(), 2u));
        auto *element = builder.gep(Type::of<int32_t>(), root, {uint_constant(module, 9u)});
        builder.store(element, module.create_constant_one(Type::of<int32_t>()));
        builder.return_void();

        PointerUsageAnalysis analysis;
        auto info = analysis.analyze(function);
        expect(!info.succeeded());
        expect(info.invalid_access_count >= 1u);
        auto *root_out = analysis.out_usage(body, root);
        expect(root_out != nullptr);
        expect(root_out->touch.access().all());
        expect(root_out->kill.access().none());
        auto null_info = pointer_usage_pass_run_on_module(nullptr);
        expect(!null_info.succeeded());
        expect(null_info.invalid_function_count == 1u);
    };

    "pointer_usage_propagates_to_equivalent_and_descendant_views"_test = [] {
        Module module;
        auto *function = module.create_kernel();
        auto *body = function->create_body_block();
        auto *pair_type = Type::structure({Type::of<int32_t>(), Type::of<int32_t>()});
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *root = builder.alloca_local(pair_type);
        auto *field0a = builder.gep(Type::of<int32_t>(), root, {uint_constant(module, 0u)});
        auto *field0b = builder.gep(Type::of<int32_t>(), root, {uint_constant(module, 0u)});
        auto *field1 = builder.gep(Type::of<int32_t>(), root, {uint_constant(module, 1u)});
        builder.store(field0a, module.create_constant_one(Type::of<int32_t>()));
        builder.return_void();

        PointerUsageAnalysis analysis;
        auto info = analysis.analyze(function);
        expect(info.succeeded());
        auto *root_out = analysis.out_usage(body, root);
        auto *field0b_out = analysis.out_usage(body, field0b);
        auto *field1_out = analysis.out_usage(body, field1);
        expect(root_out != nullptr);
        expect(field0b_out != nullptr);
        expect(field1_out != nullptr);
        expect(root_out->kill.access(0u).all());
        expect(root_out->touch.access(0u).all());
        expect(root_out->kill.access(1u).none());
        expect(field0b_out->kill.access().all());
        expect(field0b_out->touch.access().all());
        expect(field1_out->kill.access().none());
        expect(field1_out->touch.access().none());
    };

    "pointer_usage_root_store_kills_all_descendant_views"_test = [] {
        Module module;
        auto *function = module.create_kernel();
        auto *body = function->create_body_block();
        auto *pair_type = Type::structure({Type::of<int32_t>(), Type::of<int32_t>()});
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *root = builder.alloca_local(pair_type);
        auto *field0 = builder.gep(Type::of<int32_t>(), root, {uint_constant(module, 0u)});
        auto *field1 = builder.gep(Type::of<int32_t>(), root, {uint_constant(module, 1u)});
        std::array<int32_t, 2u> zeros{};
        auto *value = module.create_constant(pair_type, zeros.data());
        builder.store(root, value);
        builder.return_void();

        PointerUsageAnalysis analysis;
        auto info = analysis.analyze(function);
        expect(info.succeeded());
        auto *field0_out = analysis.out_usage(body, field0);
        auto *field1_out = analysis.out_usage(body, field1);
        expect(field0_out != nullptr);
        expect(field1_out != nullptr);
        expect(field0_out->kill.access().all());
        expect(field1_out->kill.access().all());
        expect(field0_out->touch.access().all());
        expect(field1_out->touch.access().all());
    };

    "pointer_usage_cross_block_liveness_and_cyclic_cfg_converge"_test = [] {
        Module module;
        auto *function = module.create_kernel();
        auto *condition = function->create_value_argument(Type::of<bool>());
        auto *entry = function->create_body_block();
        auto *loop = function->create_basic_block();
        auto *exit = function->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *root = builder.alloca_local(Type::array(Type::of<int32_t>(), 2u));
        auto *field = builder.gep(Type::of<int32_t>(), root, {uint_constant(module, 0u)});
        builder.br(loop);
        builder.set_insertion_point(loop);
        builder.load(Type::of<int32_t>(), field);
        builder.cond_br(condition, loop, exit);
        builder.set_insertion_point(exit);
        builder.return_void();

        PointerUsageAnalysis analysis;
        auto info = analysis.analyze(function);
        expect(info.succeeded());
        expect(info.analyzed_block_count == 3u);
        auto *entry_out = analysis.out_usage(entry, root);
        auto *loop_in = analysis.in_usage(loop, root);
        expect(entry_out != nullptr);
        expect(loop_in != nullptr);
        expect(entry_out->live.access(0u).all());
        expect(entry_out->live.access(1u).none());
        expect(loop_in->live.access(0u).all());
        expect(loop_in->live.access(1u).none());
    };

    "pointer_usage_queries_reject_stale_and_destroyed_ir"_test = [] {
        PointerUsageAnalysis analysis;
        BasicBlock *expired_block = nullptr;
        Value *expired_pointer = nullptr;
        {
            Module module;
            auto *function = module.create_kernel();
            auto *body = function->create_body_block();
            auto *pair_type = Type::structure({Type::of<int32_t>(), Type::of<int32_t>()});
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *root = builder.alloca_local(pair_type);
            auto *field0 = builder.gep(Type::of<int32_t>(), root, {uint_constant(module, 0u)});
            auto *field1 = builder.gep(Type::of<int32_t>(), root, {uint_constant(module, 1u)});
            auto *store = builder.store(field0, module.create_constant_one(Type::of<int32_t>()));
            builder.return_void();
            expired_block = body;
            expired_pointer = root;

            auto info = analysis.analyze(function);
            expect(info.succeeded());
            expect(analysis.is_current());
            expect(analysis.out_usage(body, root) != nullptr);
            store->set_operand(0u, field1);
            expect(!analysis.is_current());
            expect(analysis.out_usage(body, root) == nullptr);
            info = analysis.analyze(function);
            expect(info.succeeded());
            expect(analysis.is_current());
            expect(analysis.out_usage(body, field1)->kill.access().all());
        }
        expect(!analysis.is_current());
        expect(analysis.function() == nullptr);
        expect(analysis.block_usage(expired_block) == nullptr);
        expect(analysis.out_usage(expired_block, expired_pointer) == nullptr);
    };

    "pointer_usage_rejects_pointer_passed_to_value_formal_fail_closed"_test = [] {
        Module module;
        auto *callee = module.create_callable(nullptr);
        callee->create_value_argument(Type::of<int32_t>());
        auto *callee_body = callee->create_body_block();
        auto *function = module.create_kernel();
        auto *body = function->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(callee_body);
        builder.return_void();
        builder.set_insertion_point(body);
        auto *root = builder.alloca_local(Type::of<int32_t>());
        // Deliberately verifier-invalid: lvalues may only bind reference
        // formals. The analysis must reject this and model an opaque escape.
        builder.call(nullptr, callee, {root});
        builder.return_void();

        PointerUsageAnalysis analysis;
        auto info = analysis.analyze(function);
        expect(!info.succeeded());
        expect(info.invalid_access_count >= 1u);
        expect(info.conservative_access_count >= 1u);
        auto *usage = analysis.out_usage(body, root);
        expect(usage != nullptr);
        expect(usage->touch.access().all());
        expect(usage->kill.access().none());
        expect(analysis.in_usage(body, root)->live.access().all());
    };

    "pointer_usage_unknown_callee_pointer_escape_is_read_write"_test = [] {
        Module module;
        auto *function = module.create_kernel();
        auto *body = function->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *root = builder.alloca_local(
            Type::array(Type::of<int32_t>(), 2u));
        builder.call(nullptr, static_cast<Function *>(nullptr), {root});
        builder.return_void();

        PointerUsageAnalysis analysis;
        auto info = analysis.analyze(function);
        expect(!info.succeeded());
        expect(info.invalid_access_count >= 1u);
        expect(info.conservative_access_count >= 1u);
        auto *usage = analysis.out_usage(body, root);
        expect(usage != nullptr);
        expect(usage->touch.access().all());
        expect(usage->kill.access().none());
        expect(analysis.in_usage(body, root)->live.access().all());
    };

    return 0;
}
