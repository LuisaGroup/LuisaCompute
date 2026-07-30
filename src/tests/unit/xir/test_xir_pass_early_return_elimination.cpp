#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/metadata/comment.h>
#include <luisa/xir/metadata/location.h>
#include <luisa/xir/metadata/name.h>
#include <luisa/xir/module.h>
#include <luisa/xir/undefined.h>
#include <luisa/xir/passes/early_return_elimination.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] size_t count_terminator_kind(FunctionDefinition *def, DerivedInstructionTag tag) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        if (bb->terminator()->derived_instruction_tag() == tag) { ++n; }
    });
    return n;
}

[[nodiscard]] size_t count_alloca(FunctionDefinition *def) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        bb->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<AllocaInst>()) { ++n; }
        });
    });
    return n;
}

}// namespace

void reg_early_return_elimination() {

    "no_early_return_void"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = early_return_elimination_pass_run_on_function(k);
        expect(info.removed_return_count == 0u);
        expect(count_terminator_kind(k->definition(), DerivedInstructionTag::RETURN) == 1u);
    };

    "single_early_return_void_in_if"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *if_inst = b.if_(cond);
        auto *t = if_inst->create_true_block();
        auto *f = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(t);
        auto *early_return = b.return_void();
        early_return->set_name("source_early_return");
        early_return->set_location("early_return.cpp", 41);
        early_return->add_comment("preserve early-return source metadata");
        b.set_insertion_point(f);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto info = early_return_elimination_pass_run_on_function(k);
        expect(info.removed_return_count == 1u);
        auto *def = k->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::RETURN) == 1u);
        expect(count_alloca(def) >= 1u);
        auto *replacement = t->terminator();
        expect(replacement->isa<BranchInst>());
        expect(replacement->name().has_value());
        if (replacement->name().has_value()) {
            expect(replacement->name().value() == "source_early_return");
        }
        auto *location = replacement->find_metadata<LocationMD>();
        expect(location != nullptr);
        if (location != nullptr) {
            expect(location->file() == luisa::filesystem::path{"early_return.cpp"});
            expect(location->line() == 41);
        }
        expect(replacement->find_metadata<CommentMD>() != nullptr);
    };

    "single_early_return_nonvoid_in_if"_test = [] {
        Module m;
        auto *c = m.create_callable(Type::of<int>());
        auto *body = c->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *if_inst = b.if_(cond);
        auto *t = if_inst->create_true_block();
        auto *f = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        auto *v0 = m.create_constant_zero(Type::of<int>());
        auto *v1 = m.create_constant_one(Type::of<int>());
        b.set_insertion_point(t);
        b.return_(v0);
        b.set_insertion_point(f);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_(v1);
        auto info = early_return_elimination_pass_run_on_function(c);
        expect(info.removed_return_count == 1u);
        auto *def = c->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::RETURN) == 1u);
        expect(count_alloca(def) >= 2u);
    };

    "two_early_returns_void_in_nested_if"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        auto *outer_if = b.if_(cond);
        auto *ot = outer_if->create_true_block();
        auto *of_ = outer_if->create_false_block();
        auto *outer_merge = outer_if->create_merge_block();
        b.set_insertion_point(ot);
        auto *inner_if = b.if_(cond);
        auto *it = inner_if->create_true_block();
        auto *if2 = inner_if->create_false_block();
        auto *inner_merge = inner_if->create_merge_block();
        b.set_insertion_point(it);
        b.return_void();
        b.set_insertion_point(if2);
        b.br(inner_merge);
        b.set_insertion_point(inner_merge);
        b.return_void();
        b.set_insertion_point(of_);
        b.br(outer_merge);
        b.set_insertion_point(outer_merge);
        b.return_void();
        auto info = early_return_elimination_pass_run_on_function(k);
        expect(info.removed_return_count == 2u);
        auto *def = k->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::RETURN) == 1u);
    };

    "early_return_in_false_branch"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_zero(Type::of<bool>());
        auto *if_inst = b.if_(cond);
        auto *t = if_inst->create_true_block();
        auto *f = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(t);
        b.br(merge);
        b.set_insertion_point(f);
        b.return_void();
        b.set_insertion_point(merge);
        b.return_void();
        auto info = early_return_elimination_pass_run_on_function(k);
        expect(info.removed_return_count == 1u);
        auto *def = k->definition();
        expect(count_terminator_kind(def, DerivedInstructionTag::RETURN) == 1u);
    };

    "no_early_return_nonvoid"_test = [] {
        Module m;
        auto *c = m.create_callable(Type::of<float>());
        auto *body = c->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *v = m.create_constant_one(Type::of<float>());
        b.return_(v);
        auto info = early_return_elimination_pass_run_on_function(c);
        expect(info.removed_return_count == 0u);
        expect(count_terminator_kind(c->definition(), DerivedInstructionTag::RETURN) == 1u);
    };

    "module_run_multiple_functions"_test = [] {
        Module m;
        auto *k1 = m.create_kernel();
        auto *body1 = k1->create_body_block();
        auto *k2 = m.create_kernel();
        auto *body2 = k2->create_body_block();
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());
        b.set_insertion_point(body1);
        auto *if1 = b.if_(cond);
        auto *t1 = if1->create_true_block();
        auto *f1 = if1->create_false_block();
        auto *merge1 = if1->create_merge_block();
        b.set_insertion_point(t1);
        b.return_void();
        b.set_insertion_point(f1);
        b.br(merge1);
        b.set_insertion_point(merge1);
        b.return_void();
        b.set_insertion_point(body2);
        b.return_void();
        auto info = early_return_elimination_pass_run_on_module(&m);
        expect(info.removed_return_count == 1u);
        expect(count_terminator_kind(k1->definition(), DerivedInstructionTag::RETURN) == 1u);
        expect(count_terminator_kind(k2->definition(), DerivedInstructionTag::RETURN) == 1u);
    };

    "conditionalized_merge_chain_repairs_phi_predecessors"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        auto *cond = k->create_value_argument(Type::of<bool>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *outer = b.if_(cond);
        auto *early = outer->create_true_block();
        auto *fallthrough = outer->create_false_block();
        auto *first_merge = outer->create_merge_block();
        b.set_insertion_point(early);
        b.return_void();
        b.set_insertion_point(fallthrough);
        b.br(first_merge);

        b.set_insertion_point(first_merge);
        auto *inner = b.if_(cond);
        auto *inner_true = inner->create_true_block();
        auto *inner_false = inner->create_false_block();
        auto *final_merge = inner->create_merge_block();
        auto *one = m.create_constant_one(Type::of<int>());
        auto *zero = m.create_constant_zero(Type::of<int>());
        b.set_insertion_point(inner_true);
        auto *true_phi = b.phi(Type::of<int>());
        true_phi->add_incoming(one, first_merge);
        b.br(final_merge);
        b.set_insertion_point(inner_false);
        auto *false_phi = b.phi(Type::of<int>());
        false_phi->add_incoming(zero, first_merge);
        b.br(final_merge);
        b.set_insertion_point(final_merge);
        auto *merge_phi = b.phi(Type::of<int>());
        merge_phi->add_incoming(true_phi, inner_true);
        merge_phi->add_incoming(false_phi, inner_false);
        b.return_void();

        auto info = early_return_elimination_pass_run_on_function(k);
        expect(info.removed_return_count == 1u);
        auto *guard = static_cast<IfInst *>(first_merge->terminator());
        auto *taken_wrapper = guard->true_block();
        auto *skipped_wrapper = guard->false_block();
        expect(taken_wrapper->terminator() == inner);
        expect(true_phi->incoming(0u).block == taken_wrapper);
        expect(false_phi->incoming(0u).block == taken_wrapper);
        bool found_skip_incoming = false;
        for (size_t i = 0u; i < merge_phi->incoming_count(); ++i) {
            auto incoming = merge_phi->incoming(i);
            if (incoming.block == skipped_wrapper) {
                expect(incoming.value->isa<Undefined>());
                found_skip_incoming = true;
            }
        }
        expect(found_skip_incoming);
        expect(merge_phi->incoming_count() == 3u);
        expect(xir_verify_module(&m).succeeded());
    };

    "early_return_edge_adds_undef_to_target_merge_phi"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        auto *cond = k->create_value_argument(Type::of<bool>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *if_inst = b.if_(cond);
        auto *early = if_inst->create_true_block();
        auto *fallthrough = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(early);
        b.return_void();
        b.set_insertion_point(fallthrough);
        b.br(merge);
        b.set_insertion_point(merge);
        auto *phi = b.phi(Type::of<int>());
        phi->add_incoming(
            m.create_constant_one(Type::of<int>()), fallthrough);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto info = early_return_elimination_pass_run_on_function(k);

        expect(info.removed_return_count == 1u);
        auto saw_fallthrough = false;
        auto saw_early_undef = false;
        for (auto i = 0u; i < phi->incoming_count(); ++i) {
            auto incoming = phi->incoming(i);
            saw_fallthrough |= incoming.block == fallthrough;
            saw_early_undef |= incoming.block == early &&
                               incoming.value->isa<Undefined>();
        }
        expect(saw_fallthrough);
        expect(saw_early_undef);
        expect(phi->incoming_count() == 2u);
        expect(xir_verify_module(&m).succeeded());
    };

    "nonvoid_early_return_repairs_merge_phi_and_return_value"_test = [] {
        Module m;
        auto *callable = m.create_callable(Type::of<int>());
        auto *body = callable->create_body_block();
        auto *cond = callable->create_value_argument(Type::of<bool>());
        auto *zero = m.create_constant_zero(Type::of<int>());
        auto *one = m.create_constant_one(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *if_inst = b.if_(cond);
        auto *early = if_inst->create_true_block();
        auto *fallthrough = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(early);
        b.return_(zero);
        b.set_insertion_point(fallthrough);
        b.br(merge);
        b.set_insertion_point(merge);
        auto *phi = b.phi(Type::of<int>());
        phi->add_incoming(one, fallthrough);
        b.return_(phi);
        expect(xir_verify_module(&m).succeeded());

        auto info =
            early_return_elimination_pass_run_on_function(callable);

        expect(info.removed_return_count == 1u);
        auto saw_early_undef = false;
        for (auto i = 0u; i < phi->incoming_count(); ++i) {
            auto incoming = phi->incoming(i);
            saw_early_undef |= incoming.block == early &&
                               incoming.value->isa<Undefined>();
        }
        expect(saw_early_undef);
        expect(count_terminator_kind(
                   callable->definition(),
                   DerivedInstructionTag::RETURN) == 1u);
        expect(xir_verify_module(&m).succeeded());
    };

    "early_return_conditionalizes_structured_switch_and_repairs_phis"_test = [] {
        Module m;
        auto *kernel = m.create_kernel();
        auto *body = kernel->create_body_block();
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *selector =
            kernel->create_value_argument(Type::of<int32_t>());
        auto *zero =
            m.create_constant_zero(Type::of<int32_t>());
        auto *one =
            m.create_constant_one(Type::of<int32_t>());
        XIRBuilder b;

        // The early return is in the outer selection. Its normal continuation
        // begins with a SwitchInst, so early-return elimination must move that
        // structured terminator into the guard's taken wrapper without
        // confusing the switch merge role with an executable edge.
        b.set_insertion_point(body);
        auto *outer = b.if_(condition);
        auto *early = outer->create_true_block();
        auto *fallthrough = outer->create_false_block();
        auto *switch_header = outer->create_merge_block();
        b.set_insertion_point(early);
        b.return_void();
        b.set_insertion_point(fallthrough);
        b.br(switch_header);

        b.set_insertion_point(switch_header);
        auto *switch_inst = b.switch_(selector);
        auto *case_block = switch_inst->create_case_block(7u);
        auto *default_block = switch_inst->create_default_block();
        auto *switch_merge = switch_inst->create_merge_block();
        b.set_insertion_point(case_block);
        auto *case_phi = b.phi(
            Type::of<int32_t>(), {{one, switch_header}});
        b.br(switch_merge);
        b.set_insertion_point(default_block);
        auto *default_phi = b.phi(
            Type::of<int32_t>(), {{zero, switch_header}});
        b.br(switch_merge);
        b.set_insertion_point(switch_merge);
        auto *merge_phi = b.phi(
            Type::of<int32_t>(),
            {{case_phi, case_block}, {default_phi, default_block}});
        static_cast<void>(merge_phi);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto info =
            early_return_elimination_pass_run_on_function(kernel);

        expect(info.removed_return_count == 1u);
        expect(switch_header->terminator()->isa<IfInst>());
        auto *guard =
            static_cast<IfInst *>(switch_header->terminator());
        auto *taken_wrapper = guard->true_block();
        auto *skipped_wrapper = guard->false_block();
        expect(taken_wrapper->terminator() == switch_inst);
        expect(switch_inst->parent_block() == taken_wrapper);
        expect(switch_inst->merge_block() == switch_merge);
        expect(switch_inst->case_block(0u) == case_block);
        expect(switch_inst->default_block() == default_block);
        expect(case_phi->incoming(0u).block == taken_wrapper);
        expect(default_phi->incoming(0u).block == taken_wrapper);
        auto saw_skipped_undef = false;
        for (auto i = 0u; i < merge_phi->incoming_count(); ++i) {
            auto incoming = merge_phi->incoming(i);
            saw_skipped_undef |=
                incoming.block == skipped_wrapper &&
                incoming.value->isa<Undefined>();
        }
        expect(saw_skipped_undef);
        expect(xir_verify_module(&m).succeeded());
    };

    "cyclic_structured_merge_chain_is_rejected_without_mutation"_test = [] {
        for (auto two_block_cycle : {false, true}) {
            Module m;
            auto *kernel = m.create_kernel();
            auto *body = kernel->create_body_block();
            auto *condition = kernel->create_value_argument(Type::of<bool>());
            XIRBuilder b;
            b.set_insertion_point(body);
            auto *outer = b.if_(condition);
            auto *outer_true = outer->create_true_block();
            auto *outer_false = outer->create_false_block();
            b.set_insertion_point(outer_true);
            b.return_void();
            b.set_insertion_point(outer_false);
            b.return_void();

            BasicBlock *second = nullptr;
            IfInst *inner = nullptr;
            if (two_block_cycle) {
                second = kernel->create_basic_block();
                outer->set_merge_block(second);
                b.set_insertion_point(second);
                inner = b.if_(condition);
                auto *inner_true = inner->create_true_block();
                auto *inner_false = inner->create_false_block();
                inner->set_merge_block(body);
                b.set_insertion_point(inner_true);
                b.return_void();
                b.set_insertion_point(inner_false);
                b.return_void();
            } else {
                outer->set_merge_block(body);
            }

            auto info =
                early_return_elimination_pass_run_on_function(kernel);
            expect(info.removed_return_count == 0u);
            expect(body->terminator() == outer);
            expect(outer->merge_block() ==
                   (two_block_cycle ? second : body));
            if (two_block_cycle) {
                expect(second->terminator() == inner);
                expect(inner->merge_block() == body);
            }
        }
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_early_return_elimination();
    return 0;
}
