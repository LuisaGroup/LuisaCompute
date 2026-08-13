#include "ut/ut.hpp"
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/convergence_region.h>
#include <luisa/xir/passes/dom_tree.h>

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

} // namespace

void reg_convergence_region() {

    "empty_function_no_regions"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto dom = compute_dom_tree(k);
        auto cri = compute_convergence_regions(k, dom);
        expect(cri.top_level != nullptr);
        expect(cri.top_level->children.empty());
    };

    "single_if_construct_region"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *cond = k->create_value_argument(Type::of<bool>());
        auto *if_inst = b.if_(cond);
        auto *t = if_inst->create_true_block();
        auto *f = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(t);
        b.br(merge);
        b.set_insertion_point(f);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto dom = compute_dom_tree(k);
        auto cri = compute_convergence_regions(k, dom);
        expect(cri.top_level != nullptr);
        expect(cri.top_level->children.size() == 1u);
        auto *if_region = cri.top_level->children[0].get();
        expect(if_region->entry == body);
        expect(if_region->convergence_merge == merge);
        expect(if_region->blocks.contains(body));
        expect(if_region->blocks.contains(t));
        expect(if_region->blocks.contains(f));
        expect(if_region->blocks.contains(merge) == false);
    };

    "nested_if_detects_parent_child"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        XIRBuilder b;
        auto *cond = k->create_value_argument(Type::of<bool>());
        b.set_insertion_point(body);
        auto *outer = b.if_(cond);
        auto *ot = outer->create_true_block();
        auto *of = outer->create_false_block();
        auto *omerge = outer->create_merge_block();
        b.set_insertion_point(ot);
        auto *inner = b.if_(cond);
        auto *it = inner->create_true_block();
        auto *if_ = inner->create_false_block();
        auto *imerge = inner->create_merge_block();
        b.set_insertion_point(it);
        b.br(imerge);
        b.set_insertion_point(if_);
        b.br(imerge);
        b.set_insertion_point(imerge);
        b.br(omerge);
        b.set_insertion_point(of);
        b.br(omerge);
        b.set_insertion_point(omerge);
        b.return_void();
        auto dom = compute_dom_tree(k);
        auto cri = compute_convergence_regions(k, dom);
        expect(cri.top_level != nullptr);
        expect(cri.top_level->children.size() == 1u);
        auto *outer_region = cri.top_level->children[0].get();
        expect(outer_region->entry == body);
        expect(outer_region->children.size() >= 1u);
        bool found_inner = false;
        for (auto &child : outer_region->children) {
            if (child->entry == ot) {
                expect(child->convergence_merge == imerge);
                found_inner = true;
            }
        }
        expect(found_inner);
    };

    "simple_loop_region_includes_body"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *def = k->definition();
        auto *cond = k->create_value_argument(Type::of<bool>());
        auto *header = def->create_basic_block();
        auto *exit_bb = def->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.br(header);
        b.set_insertion_point(header);
        auto *sl = b.simple_loop();
        auto *lb = sl->create_body_block();
        auto *merge = sl->create_merge_block();
        b.set_insertion_point(lb);
        b.cond_br(cond, exit_bb, header);
        b.set_insertion_point(exit_bb);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto dom = compute_dom_tree(k);
        auto cri = compute_convergence_regions(k, dom);
        expect(cri.top_level != nullptr);
        bool found_loop = false;
        for (auto &child : cri.top_level->children) {
            if (child->entry == header) {
                expect(child->blocks.contains(lb));
                found_loop = true;
            }
        }
        expect(found_loop);
    };

    "switch_construct_region"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        int v0 = 0;
        auto *val = m.create_constant(Type::of<int>(), &v0);
        b.set_insertion_point(body);
        auto *sw = b.switch_(val);
        auto *c0 = sw->create_case_block(0);
        auto *c1 = sw->create_case_block(1);
        auto *def_bb = sw->create_default_block();
        auto *merge = sw->create_merge_block();
        b.set_insertion_point(c0);
        b.br(merge);
        b.set_insertion_point(c1);
        b.br(merge);
        b.set_insertion_point(def_bb);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto dom = compute_dom_tree(k);
        auto cri = compute_convergence_regions(k, dom);
        expect(cri.top_level != nullptr);
        expect(cri.top_level->children.size() == 1u);
        auto *sw_region = cri.top_level->children[0].get();
        expect(sw_region->entry == body);
        expect(sw_region->convergence_merge == merge);
        expect(sw_region->blocks.contains(c0));
        expect(sw_region->blocks.contains(c1));
        expect(sw_region->blocks.contains(def_bb));
    };

    "find_region_null_for_unreachable"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto dom = compute_dom_tree(k);
        auto cri = compute_convergence_regions(k, dom);
        auto *orphan = reinterpret_cast<BasicBlock *>(0xdead);
        auto *region = cri.find_region(orphan);
        expect(region == nullptr);
    };

    "find_region_locates_nested_construct"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = k->create_value_argument(Type::of<bool>());
        b.set_insertion_point(body);
        auto *if_inst = b.if_(cond);
        auto *t = if_inst->create_true_block();
        auto *f = if_inst->create_false_block();
        auto *merge = if_inst->create_merge_block();
        b.set_insertion_point(t);
        b.br(merge);
        b.set_insertion_point(f);
        b.br(merge);
        b.set_insertion_point(merge);
        b.return_void();
        auto dom = compute_dom_tree(k);
        auto cri = compute_convergence_regions(k, dom);
        auto *region = cri.find_region(t);
        expect(region != nullptr);
        expect(region->entry == body);
    };

    "convergence_region_rejects_null_and_mismatched_dom_tree"_test = [] {
        Module first;
        BasicBlock *first_body;
        auto *first_kernel = make_kernel_with_body(first, first_body);
        XIRBuilder b;
        b.set_insertion_point(first_body);
        b.return_void();
        auto first_dom = compute_dom_tree(first_kernel);

        auto null_info = compute_convergence_regions(nullptr, first_dom);
        expect(null_info.top_level == nullptr);

        Module second;
        BasicBlock *second_body;
        auto *second_kernel = make_kernel_with_body(second, second_body);
        b.set_insertion_point(second_body);
        b.return_void();
        auto mismatched =
            compute_convergence_regions(second_kernel, first_dom);
        expect(mismatched.top_level == nullptr);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_convergence_region();
    return 0;
}
