
#include "ut/ut.hpp"
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/passes/indvar_simplify.h>
#include <cstdio>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;

int main() {
    "debug"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        
        b.set_insertion_point(body);
        auto *loop = b.loop();
        auto *prep = loop->create_prepare_block();
        auto *lbody = loop->create_body_block();
        auto *upd = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        
        b.set_insertion_point(prep);
        auto *c0 = m.create_constant_zero(Type::of<int>());
        auto *c1 = m.create_constant_one(Type::of<int>());
        auto *iv = b.phi(Type::of<int>(), {{c0, body}});
        auto *prep_true = m.create_constant_one(Type::of<bool>());
        b.cond_br(prep_true, lbody, merge);
        
        b.set_insertion_point(lbody);
        b.br(upd);
        
        b.set_insertion_point(upd);
        auto *inc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv, c1});
        iv->add_incoming(inc, upd);
        auto *true_const = m.create_constant_one(Type::of<bool>());
        b.cond_br(true_const, prep, merge);
        
        b.set_insertion_point(merge);
        b.return_void();
        
        printf("iv parent_block: %p\n", (void*)iv->parent_block());
        printf("prepare_block: %p\n", (void*)prep);
        printf("update_block: %p\n", (void*)upd);
        printf("loop prepare_block(): %p\n", (void*)loop->prepare_block());
        printf("loop update_block(): %p\n", (void*)loop->update_block());
        printf("iv incoming_count: %zu\n", iv->incoming_count());
        for (size_t i = 0; i < iv->incoming_count(); ++i) {
            auto inc_info = iv->incoming(i);
            printf("  incoming[%zu]: block=%p value=%p\n", i, (void*)inc_info.block, (void*)inc_info.value);
        }
        printf("use_list size (approx): iterating...\n");
        int use_count = 0;
        for (auto *use : iv->use_list()) {
            printf("  use: user=%p\n", (void*)use->user());
            use_count++;
        }
        printf("use_count: %d\n", use_count);
        
        auto info = indvar_simplify_pass_run_on_function(k);
        printf("removed_dead_iv_count: %zu\n", info.removed_dead_iv_count);
        printf("simplified_iv_count: %zu\n", info.simplified_iv_count);
    };
    return 0;
}
