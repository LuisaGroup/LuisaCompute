#include "ut/ut.hpp"
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/ast/type_registry.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

// ---- Insertion point management ----

// ---- Alloca instructions ----

// ---- Load / Store instructions ----

// ---- GEP instruction ----

// ---- Arithmetic instructions ----

// ---- Cast instructions ----

// ---- Phi nodes ----

// ---- Print / Clock instructions ----

// ---- Assert / Assume instructions ----

// ---- Branch terminators ----

// ---- Return terminator ----

// ---- Unreachable / Break / Continue terminators ----

// ---- If control flow ----

// ---- Switch control flow ----

// ---- Loop control flow ----

// ---- Outline ----

// ---- Thread group ----

// ---- Atomic instructions ----

// ---- User / operand manipulation ----

// ---- Use-def chains ----

// ---- is_insertion_point_terminator ----

// ---- Instruction tags to_string ----

// ---- Multiple instructions in one block ----

// ---- Call instruction (function call) ----

void reg_insertion_point() {

    "xir_builder_default_insertion_point"_test = [] {
        XIRBuilder builder;
        expect(builder.insertion_point() == nullptr);
        expect(builder.is_insertion_point_terminator() == false);
    };

    "xir_builder_set_insertion_point_block"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        expect(builder.insertion_point() != nullptr);
    };

    "xir_builder_set_insertion_point_instruction"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *alloc = builder.alloca_local(Type::of<float>());
        builder.set_insertion_point(alloc);
        expect(builder.insertion_point() == alloc);
    };
}

void reg_alloca() {

    "xir_builder_alloca_local"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *alloc = builder.alloca_local(Type::of<float>());
        expect(alloc != nullptr);
        expect(alloc->is_local() == true);
        expect(alloc->is_shared() == false);
        expect(alloc->is_lvalue() == true);
        expect(alloc->type() == Type::of<float>());
        expect(alloc->derived_instruction_tag() == DerivedInstructionTag::ALLOCA);
        expect(alloc->is_terminator() == false);
        expect(alloc->op() == AllocaOp::LOCAL);
    };

    "xir_builder_alloca_shared"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *alloc = builder.alloca_shared(Type::of<int>());
        expect(alloc != nullptr);
        expect(alloc->is_shared() == true);
        expect(alloc->is_local() == false);
        expect(alloc->is_lvalue() == true);
        expect(alloc->op() == AllocaOp::SHARED);
    };

    "xir_builder_alloca_generic"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *alloc = builder.alloca_(Type::of<float>(), AllocaOp::LOCAL);
        expect(alloc != nullptr);
        expect(alloc->is_local() == true);
    };
}

void reg_load_store() {

    "xir_builder_load"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *alloc = builder.alloca_local(Type::of<float>());
        auto *load = builder.load(Type::of<float>(), alloc);
        expect(load != nullptr);
        expect(load->derived_instruction_tag() == DerivedInstructionTag::LOAD);
        expect(load->type() == Type::of<float>());
        expect(load->variable() == alloc);
        expect(load->is_terminator() == false);
    };

    "xir_builder_store"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *alloc = builder.alloca_local(Type::of<int>());
        int val = 42;
        auto *cst = module.create_constant(Type::of<int>(), &val);
        auto *store = builder.store(alloc, cst);
        expect(store != nullptr);
        expect(store->derived_instruction_tag() == DerivedInstructionTag::STORE);
        expect(store->variable() == alloc);
        expect(store->value() == cst);
    };
}

void reg_gep() {

    "xir_builder_gep"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *alloc = builder.alloca_local(Type::of<float3>());
        int idx_val = 0;
        auto *idx = module.create_constant(Type::of<int>(), &idx_val);
        auto *gep = builder.gep(Type::of<float>(), alloc, {idx});
        expect(gep != nullptr);
        expect(gep->derived_instruction_tag() == DerivedInstructionTag::GEP);
        expect(gep->is_lvalue() == true);
        expect(gep->type() == Type::of<float>());
        expect(gep->base() == alloc);
        expect(gep->index_count() == 1u);
    };
}

void reg_arithmetic() {

    "xir_builder_arithmetic_binary_add"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *arg1 = kernel->create_value_argument(Type::of<int>());
        auto *arg2 = kernel->create_value_argument(Type::of<int>());
        auto *add = builder.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {arg1, arg2});
        expect(add != nullptr);
        expect(add->derived_instruction_tag() == DerivedInstructionTag::ARITHMETIC);
        expect(add->type() == Type::of<int>());
        expect(add->op() == ArithmeticOp::BINARY_ADD);
        expect(add->operand_count() == 2u);
        expect(add->is_terminator() == false);
    };

    "xir_builder_arithmetic_unary_minus"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *arg = kernel->create_value_argument(Type::of<float>());
        auto *neg = builder.call(Type::of<float>(), ArithmeticOp::UNARY_MINUS, {arg});
        expect(neg != nullptr);
        expect(neg->op() == ArithmeticOp::UNARY_MINUS);
        expect(neg->operand_count() == 1u);
    };

    "xir_builder_arithmetic_binary_mul"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *arg1 = kernel->create_value_argument(Type::of<float>());
        auto *arg2 = kernel->create_value_argument(Type::of<float>());
        auto *mul = builder.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {arg1, arg2});
        expect(mul != nullptr);
        expect(mul->op() == ArithmeticOp::BINARY_MUL);
    };

    "xir_builder_arithmetic_binary_less"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *arg1 = kernel->create_value_argument(Type::of<int>());
        auto *arg2 = kernel->create_value_argument(Type::of<int>());
        auto *lt = builder.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {arg1, arg2});
        expect(lt != nullptr);
        expect(lt->op() == ArithmeticOp::BINARY_LESS);
    };
}

void reg_cast() {

    "xir_builder_cast_static"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *arg = kernel->create_value_argument(Type::of<int>());
        auto *cst = builder.cast_(Type::of<float>(), CastOp::STATIC_CAST, arg);
        expect(cst != nullptr);
        expect(cst->derived_instruction_tag() == DerivedInstructionTag::CAST);
        expect(cst->type() == Type::of<float>());
        expect(cst->op() == CastOp::STATIC_CAST);
        expect(cst->value() == arg);
    };

    "xir_builder_cast_bitwise"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *arg = kernel->create_value_argument(Type::of<float>());
        auto *bc = builder.bit_cast_(Type::of<uint>(), arg);
        expect(bc != nullptr);
        expect(bc->op() == CastOp::BITWISE_CAST);
        expect(bc->type() == Type::of<uint>());
    };

    "xir_builder_static_cast_convenience"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *arg = kernel->create_value_argument(Type::of<int>());
        auto *sc = builder.static_cast_(Type::of<float>(), arg);
        expect(sc != nullptr);
        expect(sc->type() == Type::of<float>());
    };

    "xir_builder_static_cast_if_unnecessary"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *arg = kernel->create_value_argument(Type::of<float>());
        auto *result = builder.static_cast_if_necessary(Type::of<float>(), arg);
        expect(result == arg) << "cast to same type should return original value";
    };

    "xir_builder_bit_cast_if_unnecessary"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *arg = kernel->create_value_argument(Type::of<float>());
        auto *result = builder.bit_cast_if_necessary(Type::of<float>(), arg);
        expect(result == arg) << "bit_cast to same type should return original value";
    };
}

void reg_phi() {

    "xir_builder_phi_empty"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *phi = builder.phi(Type::of<int>());
        expect(phi != nullptr);
        expect(phi->derived_instruction_tag() == DerivedInstructionTag::PHI);
        expect(phi->type() == Type::of<int>());
        expect(phi->incoming_count() == 0u);
    };

    "xir_builder_phi_with_incomings"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *bb1 = kernel->create_basic_block();
        auto *bb2 = kernel->create_basic_block();
        int v1 = 10;
        int v2 = 20;
        auto *c1 = module.create_constant(Type::of<int>(), &v1);
        auto *c2 = module.create_constant(Type::of<int>(), &v2);
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *phi = builder.phi(Type::of<int>(), {PhiIncoming{c1, bb1}, PhiIncoming{c2, bb2}});
        expect(phi->incoming_count() == 2u);
        auto inc0 = phi->incoming(0);
        expect(inc0.value == c1);
        expect(inc0.block == bb1);
        auto inc1 = phi->incoming(1);
        expect(inc1.value == c2);
        expect(inc1.block == bb2);
    };

    "xir_builder_phi_add_incoming"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *phi = builder.phi(Type::of<float>());
        expect(phi->incoming_count() == 0u);
        auto *bb = kernel->create_basic_block();
        float v = 3.14f;
        auto *c = module.create_constant(Type::of<float>(), &v);
        phi->add_incoming(c, bb);
        expect(phi->incoming_count() == 1u);
        auto inc = phi->incoming(0);
        expect(inc.value == c);
        expect(inc.block == bb);
    };

    "xir_builder_phi_remove_incoming"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *bb1 = kernel->create_basic_block();
        auto *bb2 = kernel->create_basic_block();
        int v1 = 1, v2 = 2;
        auto *c1 = module.create_constant(Type::of<int>(), &v1);
        auto *c2 = module.create_constant(Type::of<int>(), &v2);
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *phi = builder.phi(Type::of<int>(), {PhiIncoming{c1, bb1}, PhiIncoming{c2, bb2}});
        expect(phi->incoming_count() == 2u);
        phi->remove_incoming(0);
        expect(phi->incoming_count() == 1u);
        auto inc = phi->incoming(0);
        expect(inc.value == c2);
        expect(inc.block == bb2);
    };
}

void reg_print_clock() {

    "xir_builder_print"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *arg = kernel->create_value_argument(Type::of<int>());
        auto *p = builder.print(luisa::string{"value = {}"}, {arg});
        expect(p != nullptr);
        expect(p->derived_instruction_tag() == DerivedInstructionTag::PRINT);
        expect(p->format() == "value = {}");
        expect(p->operand_count() == 1u);
    };

    "xir_builder_print_no_args"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *p = builder.print(luisa::string{"hello world"}, {});
        expect(p != nullptr);
        expect(p->format() == "hello world");
        expect(p->operand_count() == 0u);
    };

    "xir_builder_clock"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *clk = builder.clock();
        expect(clk != nullptr);
        expect(clk->derived_instruction_tag() == DerivedInstructionTag::CLOCK);
        expect(clk->is_terminator() == false);
    };
}

void reg_assert_assume() {

    "xir_builder_assert"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        bool val = true;
        auto *cond = module.create_constant(Type::of<bool>(), &val);
        auto *a = builder.assert_(cond, "test assertion");
        expect(a != nullptr);
        expect(a->derived_instruction_tag() == DerivedInstructionTag::ASSERT);
        expect(a->condition() == cond);
        expect(a->message() == "test assertion");
    };

    "xir_builder_assume"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        bool val = true;
        auto *cond = module.create_constant(Type::of<bool>(), &val);
        auto *a = builder.assume_(cond, "test assumption");
        expect(a != nullptr);
        expect(a->derived_instruction_tag() == DerivedInstructionTag::ASSUME);
        expect(a->condition() == cond);
        expect(a->message() == "test assumption");
    };
}

void reg_branches() {

    "xir_builder_br"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *target = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *br = builder.br(target);
        expect(br != nullptr);
        expect(br->derived_instruction_tag() == DerivedInstructionTag::BRANCH);
        expect(br->is_terminator() == true);
        expect(br->target_block() == target);
        expect(body->is_terminated() == true);
        expect(body->terminator() == br);
    };

    "xir_builder_br_null_target"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *br = builder.br(nullptr);
        expect(br != nullptr);
        expect(br->is_terminator() == true);
        expect(br->target_block() == nullptr);
    };

    "xir_builder_cond_br"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *true_bb = kernel->create_basic_block();
        auto *false_bb = kernel->create_basic_block();
        bool val = true;
        auto *cond = module.create_constant(Type::of<bool>(), &val);
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *cbr = builder.cond_br(cond, true_bb, false_bb);
        expect(cbr != nullptr);
        expect(cbr->derived_instruction_tag() == DerivedInstructionTag::CONDITIONAL_BRANCH);
        expect(cbr->is_terminator() == true);
        expect(cbr->condition() == cond);
        expect(cbr->true_block() == true_bb);
        expect(cbr->false_block() == false_bb);
    };
}

void reg_return() {

    "xir_builder_return_void"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *ret = builder.return_void();
        expect(ret != nullptr);
        expect(ret->derived_instruction_tag() == DerivedInstructionTag::RETURN);
        expect(ret->is_terminator() == true);
        expect(ret->return_value() == nullptr);
        expect(body->is_terminated() == true);
    };

    "xir_builder_return_value"_test = [] {
        Module module;
        auto *callable = module.create_callable(Type::of<int>());
        auto *body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        int val = 42;
        auto *c = module.create_constant(Type::of<int>(), &val);
        auto *ret = builder.return_(c);
        expect(ret != nullptr);
        expect(ret->is_terminator() == true);
        expect(ret->return_value() == c);
    };
}

void reg_other_terminators() {

    "xir_builder_unreachable"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *u = builder.unreachable_("should not reach here");
        expect(u != nullptr);
        expect(u->derived_instruction_tag() == DerivedInstructionTag::UNREACHABLE);
        expect(u->is_terminator() == true);
        expect(u->message() == "should not reach here");
    };

    "xir_builder_break"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *brk = builder.break_();
        expect(brk != nullptr);
        expect(brk->derived_instruction_tag() == DerivedInstructionTag::BREAK);
        expect(brk->is_terminator() == true);
    };

    "xir_builder_continue"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *cont = builder.continue_();
        expect(cont != nullptr);
        expect(cont->derived_instruction_tag() == DerivedInstructionTag::CONTINUE);
        expect(cont->is_terminator() == true);
    };
}

void reg_if() {

    "xir_builder_if"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        bool val = true;
        auto *cond = module.create_constant(Type::of<bool>(), &val);
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *if_inst = builder.if_(cond);
        expect(if_inst != nullptr);
        expect(if_inst->derived_instruction_tag() == DerivedInstructionTag::IF);
        expect(if_inst->is_terminator() == true);
        expect(if_inst->condition() == cond);
        // Builder does NOT auto-create sub-blocks; they start null
        expect(if_inst->true_block() == nullptr);
        expect(if_inst->false_block() == nullptr);
        expect(if_inst->merge_block() == nullptr);
        // Explicitly create sub-blocks
        auto *tb = if_inst->create_true_block();
        auto *fb = if_inst->create_false_block();
        auto *mb = if_inst->create_merge_block();
        expect(tb != nullptr);
        expect(fb != nullptr);
        expect(mb != nullptr);
        expect(if_inst->true_block() == tb);
        expect(if_inst->false_block() == fb);
        expect(if_inst->merge_block() == mb);
        expect(tb != fb);
        expect(tb != mb);
        expect(fb != mb);
        expect(if_inst->control_flow_merge() != nullptr);
    };
}

void reg_switch() {

    "xir_builder_switch"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *arg = kernel->create_value_argument(Type::of<int>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *sw = builder.switch_(arg);
        expect(sw != nullptr);
        expect(sw->derived_instruction_tag() == DerivedInstructionTag::SWITCH);
        expect(sw->is_terminator() == true);
        expect(sw->value() == arg);
        expect(sw->default_block() == nullptr);
        expect(sw->merge_block() == nullptr);
        auto *db = sw->create_default_block();
        auto *mb = sw->create_merge_block();
        expect(db != nullptr);
        expect(mb != nullptr);
        expect(sw->default_block() == db);
        expect(sw->merge_block() == mb);
        expect(sw->case_count() == 0u);
    };

    "xir_builder_switch_add_cases"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *arg = kernel->create_value_argument(Type::of<int>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *sw = builder.switch_(arg);
        auto *case0 = sw->create_case_block(0);
        auto *case1 = sw->create_case_block(1);
        auto *case2 = sw->create_case_block(42);
        expect(sw->case_count() == 3u);
        expect(case0 != nullptr);
        expect(case1 != nullptr);
        expect(case2 != nullptr);
        expect(sw->case_value(0) == 0);
        expect(sw->case_value(1) == 1);
        expect(sw->case_value(2) == 42);
        expect(sw->case_block(0) == case0);
        expect(sw->case_block(1) == case1);
        expect(sw->case_block(2) == case2);
    };

    "xir_builder_switch_remove_case"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *arg = kernel->create_value_argument(Type::of<int>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *sw = builder.switch_(arg);
        (void)sw->create_case_block(10);
        (void)sw->create_case_block(20);
        (void)sw->create_case_block(30);
        expect(sw->case_count() == 3u);
        sw->remove_case(1);
        expect(sw->case_count() == 2u);
        expect(sw->case_value(0) == 10);
        expect(sw->case_value(1) == 30);
    };
}

void reg_loop() {

    "xir_builder_loop"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *loop = builder.loop();
        expect(loop != nullptr);
        expect(loop->derived_instruction_tag() == DerivedInstructionTag::LOOP);
        expect(loop->is_terminator() == true);
        expect(loop->prepare_block() == nullptr);
        expect(loop->body_block() == nullptr);
        expect(loop->update_block() == nullptr);
        expect(loop->merge_block() == nullptr);
        auto *pb = loop->create_prepare_block();
        auto *bb = loop->create_body_block();
        auto *ub = loop->create_update_block();
        auto *mb = loop->create_merge_block();
        expect(pb != nullptr);
        expect(bb != nullptr);
        expect(ub != nullptr);
        expect(mb != nullptr);
        expect(loop->prepare_block() == pb);
        expect(loop->body_block() == bb);
        expect(loop->update_block() == ub);
        expect(loop->merge_block() == mb);
        expect(loop->control_flow_merge() != nullptr);
    };

    "xir_builder_simple_loop"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *sloop = builder.simple_loop();
        expect(sloop != nullptr);
        expect(sloop->derived_instruction_tag() == DerivedInstructionTag::SIMPLE_LOOP);
        expect(sloop->is_terminator() == true);
        expect(sloop->body_block() == nullptr);
        expect(sloop->merge_block() == nullptr);
        auto *bb = sloop->create_body_block();
        auto *mb = sloop->create_merge_block();
        expect(bb != nullptr);
        expect(mb != nullptr);
        expect(sloop->body_block() == bb);
        expect(sloop->merge_block() == mb);
        expect(sloop->control_flow_merge() != nullptr);
    };
}

void reg_outline() {

    "xir_builder_outline"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *outline = builder.outline();
        expect(outline != nullptr);
        expect(outline->derived_instruction_tag() == DerivedInstructionTag::OUTLINE);
        expect(outline->is_terminator() == true);
        expect(outline->target_block() == nullptr);
        expect(outline->merge_block() == nullptr);
        auto *tb = outline->create_target_block();
        auto *mb = outline->create_merge_block();
        expect(tb != nullptr);
        expect(mb != nullptr);
        expect(outline->target_block() == tb);
        expect(outline->merge_block() == mb);
    };
}

void reg_thread_group() {

    "xir_builder_synchronize_block"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *sync = builder.synchronize_block();
        expect(sync != nullptr);
        expect(sync->derived_instruction_tag() == DerivedInstructionTag::THREAD_GROUP);
        expect(sync->op() == ThreadGroupOp::SYNCHRONIZE_BLOCK);
        expect(sync->is_terminator() == false);
    };
}

void reg_atomic() {

    "xir_builder_atomic_fetch_add"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *alloc = builder.alloca_local(Type::of<int>());
        int val = 1;
        auto *one = module.create_constant(Type::of<int>(), &val);
        auto *atomic = builder.atomic_fetch_add(Type::of<int>(), alloc, {}, one);
        expect(atomic != nullptr);
        expect(atomic->derived_instruction_tag() == DerivedInstructionTag::ATOMIC);
        expect(atomic->op() == AtomicOp::FETCH_ADD);
    };

    "xir_builder_atomic_exchange"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *alloc = builder.alloca_local(Type::of<int>());
        int val = 99;
        auto *desired = module.create_constant(Type::of<int>(), &val);
        auto *atomic = builder.atomic_exchange(Type::of<int>(), alloc, {}, desired);
        expect(atomic != nullptr);
        expect(atomic->op() == AtomicOp::EXCHANGE);
    };

    "xir_builder_atomic_compare_exchange"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *alloc = builder.alloca_local(Type::of<int>());
        int v1 = 10, v2 = 20;
        auto *expected = module.create_constant(Type::of<int>(), &v1);
        auto *desired = module.create_constant(Type::of<int>(), &v2);
        auto *atomic = builder.atomic_compare_exchange(Type::of<int>(), alloc, {}, expected, desired);
        expect(atomic != nullptr);
        expect(atomic->op() == AtomicOp::COMPARE_EXCHANGE);
    };
}

void reg_user_operands() {

    "xir_user_operand_count"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *arg1 = kernel->create_value_argument(Type::of<int>());
        auto *arg2 = kernel->create_value_argument(Type::of<int>());
        auto *add = builder.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {arg1, arg2});
        expect(add->operand_count() == 2u);
        expect(add->operand(0) == arg1);
        expect(add->operand(1) == arg2);
    };
}

void reg_use_def() {

    "xir_use_list_on_value"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *alloc = builder.alloca_local(Type::of<float>());
        auto *load1 = builder.load(Type::of<float>(), alloc);
        auto *load2 = builder.load(Type::of<float>(), alloc);
        (void)load1;
        (void)load2;
        auto use_count = 0u;
        for ([[maybe_unused]] auto *u : alloc->use_list()) { use_count++; }
        expect(use_count == 2u) << "alloca used by two loads should have 2 uses";
    };

    "xir_replace_all_uses_with"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *alloc1 = builder.alloca_local(Type::of<float>());
        auto *alloc2 = builder.alloca_local(Type::of<float>());
        auto *load = builder.load(Type::of<float>(), alloc1);
        expect(load->variable() == alloc1);
        alloc1->replace_all_uses_with(alloc2);
        expect(load->variable() == alloc2) << "after replace_all_uses_with, load should use alloc2";
        auto use_count_1 = 0u;
        for ([[maybe_unused]] auto *u : alloc1->use_list()) { use_count_1++; }
        expect(use_count_1 == 0u) << "alloc1 should have no uses after replacement";
        auto use_count_2 = 0u;
        for ([[maybe_unused]] auto *u : alloc2->use_list()) { use_count_2++; }
        expect(use_count_2 == 1u) << "alloc2 should have 1 use after replacement";
    };
}

void reg_terminator_check() {

    "xir_builder_insertion_point_is_terminator_false"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *alloc = builder.alloca_local(Type::of<int>());
        builder.set_insertion_point(alloc);
        expect(builder.is_insertion_point_terminator() == false);
    };

    "xir_builder_insertion_point_is_terminator_true"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *ret = builder.return_void();
        builder.set_insertion_point(ret);
        expect(builder.is_insertion_point_terminator() == true);
    };
}

void reg_instruction_tag_strings() {

    "xir_instruction_tag_to_string"_test = [] {
        expect(to_string(DerivedInstructionTag::IF) == "if");
        expect(to_string(DerivedInstructionTag::SWITCH) == "switch");
        expect(to_string(DerivedInstructionTag::LOOP) == "loop");
        expect(to_string(DerivedInstructionTag::SIMPLE_LOOP) == "simple_loop");
        expect(to_string(DerivedInstructionTag::BRANCH) == "branch");
        expect(to_string(DerivedInstructionTag::CONDITIONAL_BRANCH) == "conditional_branch");
        expect(to_string(DerivedInstructionTag::UNREACHABLE) == "unreachable");
        expect(to_string(DerivedInstructionTag::BREAK) == "break");
        expect(to_string(DerivedInstructionTag::CONTINUE) == "continue");
        expect(to_string(DerivedInstructionTag::RETURN) == "return");
        expect(to_string(DerivedInstructionTag::PHI) == "phi");
        expect(to_string(DerivedInstructionTag::ALLOCA) == "alloca");
        expect(to_string(DerivedInstructionTag::LOAD) == "load");
        expect(to_string(DerivedInstructionTag::STORE) == "store");
        expect(to_string(DerivedInstructionTag::GEP) == "gep");
        expect(to_string(DerivedInstructionTag::ATOMIC) == "atomic");
        expect(to_string(DerivedInstructionTag::ARITHMETIC) == "arithmetic");
        expect(to_string(DerivedInstructionTag::THREAD_GROUP) == "thread_group");
        expect(to_string(DerivedInstructionTag::CALL) == "call");
        expect(to_string(DerivedInstructionTag::CAST) == "cast");
        expect(to_string(DerivedInstructionTag::PRINT) == "print");
        expect(to_string(DerivedInstructionTag::CLOCK) == "clock");
        expect(to_string(DerivedInstructionTag::ASSERT) == "assert");
        expect(to_string(DerivedInstructionTag::ASSUME) == "assume");
        expect(to_string(DerivedInstructionTag::OUTLINE) == "outline");
    };
}

void reg_multi_inst() {

    "xir_builder_multiple_instructions_in_block"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *a1 = builder.alloca_local(Type::of<float>());
        auto *a2 = builder.alloca_local(Type::of<int>());
        int val = 7;
        auto *cst = module.create_constant(Type::of<int>(), &val);
        (void)builder.store(a2, cst);
        (void)builder.load(Type::of<float>(), a1);
        auto count = 0u;
        for ([[maybe_unused]] auto *inst : body->instructions()) { count++; }
        expect(count == 4u) << "block should have 4 instructions (2 alloca + 1 store + 1 load)";
    };
}

void reg_call() {

    "xir_builder_call_function"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *callee = module.create_callable(Type::of<float>());
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *arg = kernel->create_value_argument(Type::of<float>());
        auto *call = builder.call(Type::of<float>(), static_cast<Function *>(callee), {arg});
        expect(call != nullptr);
        expect(call->derived_instruction_tag() == DerivedInstructionTag::CALL);
        expect(call->type() == Type::of<float>());
    };
}

int main(int argc, char *argv[]) {

    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_insertion_point();
    reg_alloca();
    reg_load_store();
    reg_gep();
    reg_arithmetic();
    reg_cast();
    reg_phi();
    reg_print_clock();
    reg_assert_assume();
    reg_branches();
    reg_return();
    reg_other_terminators();
    reg_if();
    reg_switch();
    reg_loop();
    reg_outline();
    reg_thread_group();
    reg_atomic();
    reg_user_operands();
    reg_use_def();
    reg_terminator_check();
    reg_instruction_tag_strings();
    reg_multi_inst();
    reg_call();
    return 0;
}
