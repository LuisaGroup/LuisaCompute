#include "ut/ut.hpp"

#include <luisa/ast/type_registry.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/metadata/comment.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/defer_local_aggregate_load.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] bool appears_before(BasicBlock *block,
                                  Instruction *first,
                                  Instruction *second) noexcept {
    auto saw_first = false;
    for (auto *instruction : block->instructions()) {
        if (instruction == first) { saw_first = true; }
        if (instruction == second) { return saw_first; }
    }
    return false;
}

[[nodiscard]] Constant *index(Module &module,
                              uint32_t value) noexcept {
    return module.create_constant(Type::of<uint32_t>(), &value);
}

}// namespace

void reg_defer_local_aggregate_load() {

    "static_nested_projection_becomes_one_precise_load"_test = [] {
        Module module;
        auto *pair = Type::structure(
            {Type::of<float>(), Type::of<float>()});
        auto *nested = Type::structure(
            {Type::of<int32_t>(), pair});
        auto *function = module.create_callable(Type::of<float>());
        auto *body = function->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *state = builder.alloca_local(nested);
        builder.store(state, module.create_constant_zero(nested));
        auto *snapshot = builder.load(nested, state);
        auto *inner = builder.call(
            pair, ArithmeticOp::EXTRACT,
            {snapshot, index(module, 1u)});
        auto *leaf = builder.call(
            Type::of<float>(), ArithmeticOp::EXTRACT,
            {inner, index(module, 0u)});
        auto *ret = builder.return_(leaf);

        expect(xir_verify_module(&module).succeeded());
        auto info =
            defer_local_aggregate_load_pass_run_on_function(function);

        expect(info.aggregate_load_count == 1u);
        expect(info.candidate_extract_count == 2u);
        expect(info.rewritten_extract_count == 2u);
        expect(info.inserted_gep_count == 1u);
        expect(info.inserted_load_count == 1u);
        expect(info.removed_aggregate_load_count == 1u);
        expect(ret->return_value()->isa<LoadInst>());
        auto *projected = static_cast<LoadInst *>(ret->return_value());
        expect(projected->type() == Type::of<float>());
        expect(projected->variable()->isa<GEPInst>());
        auto *gep = static_cast<GEPInst *>(projected->variable());
        expect(gep->base() == state);
        expect(gep->index_count() == 2u);
        expect(xir_verify_module(&module).succeeded());
    };

    "projected_load_stays_at_the_original_snapshot"_test = [] {
        Module module;
        auto *pair = Type::structure(
            {Type::of<float>(), Type::of<float>()});
        auto *function = module.create_callable(Type::of<float>());
        auto *body = function->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *state = builder.alloca_local(pair);
        builder.store(state, module.create_constant_zero(pair));
        auto *snapshot = builder.load(pair, state);
        auto *first_pointer = builder.gep(
            Type::of<float>(), state, {index(module, 0u)});
        auto *overwrite = builder.store(
            first_pointer,
            module.create_constant_one(Type::of<float>()));
        auto *old_first = builder.call(
            Type::of<float>(), ArithmeticOp::EXTRACT,
            {snapshot, index(module, 0u)});
        auto *ret = builder.return_(old_first);

        auto info =
            defer_local_aggregate_load_pass_run_on_function(function);

        expect(info.inserted_load_count == 1u);
        expect(ret->return_value()->isa<LoadInst>());
        auto *projected = static_cast<LoadInst *>(ret->return_value());
        // Moving the replacement to the extract would observe the overwrite.
        // It must remain before every instruction that originally followed the
        // aggregate snapshot.
        expect(appears_before(body, projected, overwrite));
        expect(xir_verify_module(&module).succeeded());
    };

    "dynamic_projection_is_a_conservative_boundary"_test = [] {
        Module module;
        auto *pair = Type::array(Type::of<float>(), 2u);
        auto *function = module.create_callable(Type::of<float>());
        auto *dynamic_index =
            function->create_value_argument(Type::of<uint32_t>());
        auto *body = function->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *state = builder.alloca_local(pair);
        auto *snapshot = builder.load(pair, state);
        auto *extract = builder.call(
            Type::of<float>(), ArithmeticOp::EXTRACT,
            {snapshot, dynamic_index});
        auto *ret = builder.return_(extract);

        auto info =
            defer_local_aggregate_load_pass_run_on_function(function);

        expect(!info.changed());
        expect(ret->return_value() == extract);
        expect(snapshot->is_linked());
        expect(xir_verify_module(&module).succeeded());
    };

    "shared_storage_is_not_split_into_non_atomic_snapshots"_test = [] {
        Module module;
        auto *pair = Type::array(Type::of<float>(), 2u);
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *state = builder.alloca_shared(pair);
        auto *snapshot = builder.load(pair, state);
        auto *extract = builder.call(
            Type::of<float>(), ArithmeticOp::EXTRACT,
            {snapshot, index(module, 0u)});
        builder.print("{}", {extract});
        builder.return_void();

        auto info =
            defer_local_aggregate_load_pass_run_on_function(kernel);

        expect(!info.changed());
        expect(snapshot->is_linked());
        expect(extract->is_linked());
        expect(xir_verify_module(&module).succeeded());
    };

    "identical_unannotated_projections_are_value_numbered"_test = [] {
        Module module;
        auto *pair = Type::array(Type::of<float>(), 2u);
        auto *function = module.create_callable(Type::of<float>());
        auto *body = function->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *state = builder.alloca_local(pair);
        auto *snapshot = builder.load(pair, state);
        auto *first = builder.call(
            Type::of<float>(), ArithmeticOp::EXTRACT,
            {snapshot, index(module, 1u)});
        auto *second = builder.call(
            Type::of<float>(), ArithmeticOp::EXTRACT,
            {snapshot, index(module, 1u)});
        auto *sum = builder.call(
            Type::of<float>(), ArithmeticOp::BINARY_ADD,
            {first, second});
        builder.return_(sum);

        auto info =
            defer_local_aggregate_load_pass_run_on_function(function);

        expect(info.rewritten_extract_count == 2u);
        expect(info.inserted_load_count == 1u);
        expect(info.reused_projection_count == 1u);
        expect(sum->operand(0u) == sum->operand(1u));
        expect(xir_verify_module(&module).succeeded());
    };

    "annotated_projection_is_not_a_value_numbering_leader"_test = [] {
        Module module;
        auto *pair = Type::array(Type::of<float>(), 2u);
        auto *function = module.create_callable(Type::of<float>());
        auto *body = function->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *state = builder.alloca_local(pair);
        auto *snapshot = builder.load(pair, state);
        auto *annotated = builder.call(
            Type::of<float>(), ArithmeticOp::EXTRACT,
            {snapshot, index(module, 1u)});
        annotated->add_comment("unique projection");
        auto *plain = builder.call(
            Type::of<float>(), ArithmeticOp::EXTRACT,
            {snapshot, index(module, 1u)});
        auto *sum = builder.call(
            Type::of<float>(), ArithmeticOp::BINARY_ADD,
            {annotated, plain});
        builder.return_(sum);

        auto info =
            defer_local_aggregate_load_pass_run_on_function(function);

        expect(info.rewritten_extract_count == 2u);
        expect(info.inserted_load_count == 2u);
        expect(info.reused_projection_count == 0u);
        expect(sum->operand(0u) != sum->operand(1u));
        expect(static_cast<Instruction *>(sum->operand(0u))
                   ->find_metadata<CommentMD>() != nullptr);
        expect(xir_verify_module(&module).succeeded());
    };

    "annotated_intermediate_extract_is_not_bypassed"_test = [] {
        Module module;
        auto *pair = Type::structure(
            {Type::of<float>(), Type::of<float>()});
        auto *nested = Type::structure({pair, pair});
        auto *function = module.create_callable(Type::of<float>());
        auto *body = function->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *state = builder.alloca_local(nested);
        auto *snapshot = builder.load(nested, state);
        auto *inner = builder.call(
            pair, ArithmeticOp::EXTRACT,
            {snapshot, index(module, 1u)});
        inner->add_comment("projection boundary");
        auto *leaf = builder.call(
            Type::of<float>(), ArithmeticOp::EXTRACT,
            {inner, index(module, 0u)});
        auto *ret = builder.return_(leaf);

        auto info =
            defer_local_aggregate_load_pass_run_on_function(function);

        expect(info.candidate_extract_count == 1u);
        expect(info.rewritten_extract_count == 1u);
        expect(ret->return_value() == leaf);
        expect(leaf->operand(0u)->isa<LoadInst>());
        auto *replacement =
            static_cast<LoadInst *>(leaf->operand(0u));
        expect(replacement->type() == pair);
        expect(replacement->find_metadata<CommentMD>() != nullptr);
        expect(xir_verify_module(&module).succeeded());
    };

    "deferred_field_load_enables_precise_coroutine_frame_atom"_test = [] {
        Module module;
        auto *pair = Type::structure(
            {Type::of<float>(), Type::of<float>()});
        auto *kernel = module.create_kernel();
        auto *entry = kernel->create_body_block();
        auto *resume = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(pair);
        state->set_name("state");
        // A whole write is representable as a kill of every overlapping read
        // atom and therefore must not force a whole-frame fallback.
        builder.store(state, module.create_constant_zero(pair));
        builder.coro_suspend(17u, "field", nullptr);
        builder.set_insertion_point(resume);
        builder.coro_resume(17u, nullptr);
        auto *snapshot = builder.load(pair, state);
        auto *second = builder.call(
            Type::of<float>(), ArithmeticOp::EXTRACT,
            {snapshot, index(module, 1u)});
        builder.print("{}", {second});
        builder.return_void();

        auto defer_info =
            defer_local_aggregate_load_pass_run_on_function(kernel);
        auto frame = coro_cfg_distill_pass_run_on_function(kernel);

        expect(defer_info.rewritten_extract_count == 1u);
        expect(frame.succeeded());
        expect(frame.frame_values.size() == 1u);
        if (frame.frame_values.size() == 1u) {
            expect(frame.frame_values.front().value == state);
            expect(frame.frame_values.front().type == Type::of<float>());
            expect(frame.frame_values.front().access_chain ==
                   luisa::vector<uint32_t>{1u});
            expect(frame.frame_values.front().name == "state.1");
        }
        expect(xir_verify_module(&module).succeeded());
    };

    "null_entry_points_and_report_schema_are_total"_test = [] {
        expect(!defer_local_aggregate_load_pass_run_on_function(
                    nullptr)
                    .changed());
        PassReport report;
        auto info = defer_local_aggregate_load_pass_run_on_module(
            nullptr, &report);
        expect(!info.changed());
        expect(report.entries().size() == 7u);
        for (auto &entry : report.entries()) {
            expect(entry.value == 0u);
        }
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    reg_defer_local_aggregate_load();
    return 0;
}
