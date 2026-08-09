#include "ut/ut.hpp"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/indexed_branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <luisa/xir/translators/xir2text.h>
#include <luisa/xir/verifier.h>

#include <cstdint>
#include <limits>

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

[[nodiscard]] size_t count_blocks(FunctionDefinition *def) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *) noexcept { ++n; });
    return n;
}

[[nodiscard]] size_t count_isa_branch(FunctionDefinition *def) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (bb->is_terminated() && bb->terminator()->isa<BranchInst>()) ++n;
    });
    return n;
}

[[nodiscard]] size_t count_isa_cond_branch(FunctionDefinition *def) noexcept {
    size_t n = 0u;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (bb->is_terminated() && bb->terminator()->isa<ConditionalBranchInst>()) ++n;
    });
    return n;
}

}// namespace

void reg_simplify_cfg() {

    "simplify_empty_function"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = simplify_cfg_pass_run_on_function(k);
        expect(info.folded_constant_cond_br_count == 0u);
        expect(info.threaded_empty_block_count == 0u);
        expect(info.removed_unreachable_block_count == 0u);
    };

    "fold_constant_true_cond_br"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *t = def->create_basic_block();
        auto *f = def->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        b.cond_br(cond, t, f);
        b.set_insertion_point(t);
        b.return_void();
        b.set_insertion_point(f);
        b.return_void();
        auto info = simplify_cfg_pass_run_on_function(k);
        expect(info.folded_constant_cond_br_count == 1u);
        expect(count_isa_cond_branch(def) == 0u);
        expect(info.merged_straight_line_count == 1u);
        expect(count_blocks(def) == 1u);
        expect(info.removed_unreachable_block_count == 1u);
    };

    "fold_constant_false_cond_br"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *t = def->create_basic_block();
        auto *f = def->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_zero(Type::of<bool>());
        b.cond_br(cond, t, f);
        b.set_insertion_point(t);
        b.return_void();
        b.set_insertion_point(f);
        b.return_void();
        auto info = simplify_cfg_pass_run_on_function(k);
        expect(info.folded_constant_cond_br_count == 1u);
        expect(info.removed_unreachable_block_count == 1u);
        expect(count_isa_cond_branch(def) == 0u);
    };

    "malformed_constant_conditional_with_null_taken_edge_terminates"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *fallthrough = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *branch = b.cond_br(
            m.create_constant_one(Type::of<bool>()),
            nullptr, fallthrough);
        b.set_insertion_point(fallthrough);
        b.return_void();
        auto block_count = count_blocks(k->definition());

        auto info = simplify_cfg_pass_run_on_function(k);
        expect(!info.changed());
        expect(count_blocks(k->definition()) == block_count);
        expect(body->terminator() == branch);
        expect(branch->true_block() == nullptr);
        expect(branch->false_block() == fallthrough);
    };

    "preserve_constant_false_structured_loop_prepare"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *loop = b.loop();
        auto *prepare = loop->create_prepare_block();
        auto *body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        b.set_insertion_point(prepare);
        auto *prepare_branch = b.cond_br(
            m.create_constant_zero(Type::of<bool>()), body, merge);
        b.set_insertion_point(body);
        b.br(update);
        b.set_insertion_point(update);
        b.br(prepare);
        b.set_insertion_point(merge);
        b.return_void();
        expect(xir_verify_module(&m).succeeded());

        auto before = xir_to_text_translate(&m, true);
        auto info = simplify_cfg_pass_run_on_function(k);
        auto after = xir_to_text_translate(&m, true);
        expect(info.folded_constant_cond_br_count == 0u);
        expect(prepare->terminator() == prepare_branch);
        expect(before == after);
        expect(xir_verify_module(&m).succeeded());
    };

    "fold_constant_cond_br_removes_dropped_phi_incoming"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *live = def->create_basic_block();
        auto *join = def->create_basic_block();
        auto *from_dropped_edge = m.create_constant_zero(Type::of<int>());
        auto *from_live_edge = m.create_constant_one(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(body);
        b.cond_br(m.create_constant_one(Type::of<bool>()), live, join);
        b.set_insertion_point(live);
        b.br(join);
        b.set_insertion_point(join);
        auto *phi = b.phi(Type::of<int>());
        phi->add_incoming(from_dropped_edge, body);
        phi->add_incoming(from_live_edge, live);
        b.return_void();

        auto info = simplify_cfg_pass_run_on_function(k);
        expect(info.folded_constant_cond_br_count == 1u);
        expect(body->terminator()->isa<BranchInst>());
        expect(static_cast<BranchInst *>(body->terminator())->target_block() == live);
        expect(phi->incoming_count() == 1u);
        expect(phi->incoming(0u).block == live);
        expect(phi->incoming(0u).value == from_live_edge);
        size_t predecessor_count = 0u;
        join->traverse_predecessors(false, [&](BasicBlock *pred) noexcept {
            expect(pred == live);
            ++predecessor_count;
        });
        expect(predecessor_count == 1u);
    };

    "thread_empty_block"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *mid = def->create_basic_block();
        auto *tail = def->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.br(mid);
        b.set_insertion_point(mid);
        b.br(tail);
        b.set_insertion_point(tail);
        b.return_void();
        auto info = simplify_cfg_pass_run_on_function(k);
        expect(info.threaded_empty_block_count >= 1u);
        expect(count_blocks(def) == 1u);
    };

    "remove_unreachable_block"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *dead = def->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        b.set_insertion_point(dead);
        b.return_void();
        auto info = simplify_cfg_pass_run_on_function(k);
        expect(info.removed_unreachable_block_count == 1u);
        expect(count_blocks(def) == 1u);
    };

    "remove_unreachable_cfg_cycle_detaches_before_release"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        auto *def = k->definition();
        auto *dead_a = def->create_basic_block();
        auto *dead_b = def->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        b.return_void();
        b.set_insertion_point(dead_a);
        auto *x = b.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {m.create_constant_one(Type::of<uint>()),
             m.create_constant_one(Type::of<uint>())});
        b.br(dead_b);
        b.set_insertion_point(dead_b);
        static_cast<void>(b.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {x, m.create_constant_one(Type::of<uint>())}));
        b.br(dead_a);
        expect(xir_verify_module(&m).succeeded());

        auto info = simplify_cfg_pass_run_on_function(k);
        expect(info.removed_unreachable_block_count == 2u);
        expect(count_blocks(def) == 1u);
        expect(xir_verify_module(&m).succeeded());
    };

    "switch_default_redirect_via_thread"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *val = m.create_undefined(Type::of<int>());
        auto *sw = b.switch_(val);
        auto *def_bb = sw->create_default_block();
        auto *case_bb = sw->create_case_block(0);
        auto *switch_merge = sw->create_merge_block();
        auto *real_default = def->create_basic_block();
        auto *real_case = def->create_basic_block();
        b.set_insertion_point(def_bb);
        b.br(real_default);
        b.set_insertion_point(case_bb);
        b.br(real_case);
        b.set_insertion_point(real_default);
        b.return_void();
        b.set_insertion_point(real_case);
        b.return_void();
        b.set_insertion_point(switch_merge);
        b.unreachable_();
        auto info = simplify_cfg_pass_run_on_function(k);
        expect(info.threaded_empty_block_count >= 2u);
        expect(sw->default_block() == real_default);
        expect(sw->case_block(0) == real_case);
    };

    "structured_if_thread_preserves_if"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_undefined(Type::of<bool>());
        auto *if_inst = b.if_(cond);
        auto *true_bb = if_inst->create_true_block();
        auto *false_bb = if_inst->create_false_block();
        auto *merge_bb = if_inst->create_merge_block();
        b.set_insertion_point(true_bb);
        b.br(merge_bb);
        b.set_insertion_point(false_bb);
        b.return_void();
        b.set_insertion_point(merge_bb);
        b.return_void();

        auto info = simplify_cfg_pass_run_on_function(k);
        expect(info.threaded_empty_block_count == 1u);
        expect(body->terminator()->isa<IfInst>());
        expect(if_inst->true_block() == merge_bb);
        expect(if_inst->false_block() == false_bb);
        expect(count_blocks(def) == 3u);
    };

    "idempotent_on_simplified"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info1 = simplify_cfg_pass_run_on_function(k);
        auto info2 = simplify_cfg_pass_run_on_function(k);
        expect(info1.folded_constant_cond_br_count == 0u);
        expect(info2.folded_constant_cond_br_count == 0u);
        expect(info2.threaded_empty_block_count == 0u);
        expect(info2.removed_unreachable_block_count == 0u);
    };

    "module_entry_point"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *t = def->create_basic_block();
        auto *f = def->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = m.create_constant_one(Type::of<bool>());
        b.cond_br(cond, t, f);
        b.set_insertion_point(t);
        b.return_void();
        b.set_insertion_point(f);
        b.return_void();
        auto info = simplify_cfg_pass_run_on_module(&m);
        expect(info.folded_constant_cond_br_count == 1u);
    };

    "preserve_constant_structured_switch"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        XIRBuilder b;
        b.set_insertion_point(body);
        int32_t selector_v = 7;
        auto *selector = m.create_constant(Type::of<int>(), &selector_v);
        auto *sw = b.switch_(selector);
        auto *default_bb = sw->create_default_block();
        auto *case_bb = sw->create_case_block(7);
        auto *switch_merge = sw->create_merge_block();
        b.set_insertion_point(default_bb);
        b.return_void();
        b.set_insertion_point(case_bb);
        b.return_void();
        b.set_insertion_point(switch_merge);
        b.unreachable_();
        auto info = simplify_cfg_pass_run_on_function(k);
        expect(info.folded_switch_count == 0u);
        expect(body->terminator()->isa<SwitchInst>());
        expect(body->terminator() == sw);
        expect(count_blocks(def) == 3u);
    };

    "preserve_degenerate_structured_switch"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *selector = m.create_undefined(Type::of<int>());
        auto *sw = b.switch_(selector);
        auto *target = sw->create_default_block();
        sw->add_case(1, target);
        sw->add_case(2, target);
        auto *switch_merge = sw->create_merge_block();
        b.set_insertion_point(target);
        b.return_void();
        b.set_insertion_point(switch_merge);
        b.unreachable_();
        auto info = simplify_cfg_pass_run_on_function(k);
        expect(info.folded_switch_count == 0u);
        expect(body->terminator()->isa<SwitchInst>());
        expect(body->terminator() == sw);
    };

    "fold_constant_indexed_branch"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        uint32_t selector_value = 7u;
        auto *selector = m.create_constant(
            Type::of<uint32_t>(), &selector_value);
        b.set_insertion_point(body);
        auto *indexed_branch = b.indexed_branch(selector);
        auto *default_block = indexed_branch->create_default_block();
        auto *case_block = indexed_branch->create_case_block(7u);
        b.set_insertion_point(default_block);
        b.return_void();
        b.set_insertion_point(case_block);
        b.return_void();

        auto info = simplify_cfg_pass_run_on_function(k);
        expect(info.folded_switch_count == 1u);
        expect(!body->terminator()->isa<IndexedBranchInst>());
        expect(body->terminator()->isa<ReturnInst>());
    };

    "fold_signed_narrow_indexed_branch_case"_test = [] {
        Module m;
        auto *callable = m.create_callable(Type::of<uint32_t>());
        auto *body = callable->create_body_block();
        int8_t selector_value = -1;
        uint32_t zero_value = 0u;
        uint32_t one_value = 1u;
        auto *selector = m.create_constant(
            Type::of<int8_t>(), &selector_value);
        auto *zero = m.create_constant(
            Type::of<uint32_t>(), &zero_value);
        auto *one = m.create_constant(
            Type::of<uint32_t>(), &one_value);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *indexed_branch = b.indexed_branch(selector);
        auto *default_block =
            indexed_branch->create_default_block();
        auto *case_block = indexed_branch->create_case_block(
            std::numeric_limits<uint64_t>::max());
        b.set_insertion_point(default_block);
        b.return_(zero);
        b.set_insertion_point(case_block);
        b.return_(one);

        expect(indexed_branch->case_value(0u) == uint64_t{0xffu});
        auto info = simplify_cfg_pass_run_on_function(callable);
        expect(info.folded_switch_count == 1u);
        expect(body->terminator()->isa<ReturnInst>());
        expect(static_cast<ReturnInst *>(body->terminator())
                   ->return_value() == one);
    };

    "malformed_constant_indexed_branch_with_null_taken_edge_terminates"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *case_block = k->create_basic_block();
        uint32_t selector_value = 7u;
        auto *selector = m.create_constant(
            Type::of<uint32_t>(), &selector_value);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *indexed = b.indexed_branch(selector);
        indexed->set_default_block(nullptr);
        indexed->add_case(8u, case_block);
        b.set_insertion_point(case_block);
        b.return_void();
        auto block_count = count_blocks(k->definition());

        auto info = simplify_cfg_pass_run_on_function(k);
        expect(!info.changed());
        expect(count_blocks(k->definition()) == block_count);
        expect(body->terminator() == indexed);
        expect(indexed->default_block() == nullptr);
        expect(indexed->case_block(0u) == case_block);
    };

    "merge_straight_line_blocks"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *tail = def->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.br(tail);
        b.set_insertion_point(tail);
        auto *undef = m.create_undefined(Type::of<int>());
        b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {undef, undef});
        b.return_void();
        auto info = simplify_cfg_pass_run_on_function(k);
        expect(info.merged_straight_line_count == 1u);
        expect(info.changed());
        expect(count_blocks(def) == 1u);
    };

    "merge_straight_line_chain"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *mid = def->create_basic_block();
        auto *tail = def->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.br(mid);
        b.set_insertion_point(mid);
        auto *undef = m.create_undefined(Type::of<int>());
        b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {undef, undef});
        b.br(tail);
        b.set_insertion_point(tail);
        b.call(Type::of<int>(), ArithmeticOp::BINARY_SUB, {undef, undef});
        b.return_void();
        auto info = simplify_cfg_pass_run_on_function(k);
        expect(info.merged_straight_line_count == 2u);
        expect(info.changed());
        expect(count_blocks(def) == 1u);
    };

    "merge_straight_line_long_chain_is_batched"_test = [] {
        Module module;
        BasicBlock *body;
        auto *kernel = make_kernel_with_body(module, body);
        auto *definition = kernel->definition();
        constexpr auto edge_count = size_t{256u};
        luisa::vector<BasicBlock *> blocks;
        blocks.reserve(edge_count + 1u);
        blocks.emplace_back(body);
        for (auto i = size_t{0u}; i < edge_count; ++i) {
            blocks.emplace_back(
                definition->create_basic_block());
        }
        auto *undefined =
            module.create_undefined(Type::of<int>());
        XIRBuilder builder;
        for (auto i = size_t{0u}; i < edge_count; ++i) {
            builder.set_insertion_point(blocks[i]);
            builder.call(
                Type::of<int>(), ArithmeticOp::BINARY_ADD,
                {undefined, undefined});
            builder.br(blocks[i + 1u]);
        }
        builder.set_insertion_point(blocks.back());
        builder.return_void();

        auto info =
            simplify_cfg_pass_run_on_function(kernel);
        expect(info.merged_straight_line_count == edge_count);
        expect(info.straight_line_scan_count == 2u)
            << "one mutating maximal-chain scan plus one fixed-point "
               "confirmation must replace one full scan per edge";
        expect(info.straight_line_block_visit_count <=
               2u * (edge_count + 1u))
            << "straight-line work must remain linear in the physical "
               "input blocks plus contracted edges";
        expect(count_blocks(definition) == 1u);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_simplify_cfg();
    return 0;
}
