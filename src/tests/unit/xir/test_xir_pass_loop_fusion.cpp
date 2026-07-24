// Test for the conservative XIR loop-fusion contract.

#include "ut/ut.hpp"
#include <luisa/core/logging.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/passes/loop_fusion.h>
#include <luisa/xir/translators/xir2text.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;

namespace {

[[nodiscard]] KernelFunction *make_kernel_with_body(Module &m, BasicBlock *&body_out) noexcept {
    auto *k = m.create_kernel();
    body_out = k->create_body_block();
    return k;
}

struct CountedLoopFixture {
    Module &m;
    BasicBlock *body{nullptr};
    KernelFunction *k{nullptr};
    LoopInst *loop{nullptr};
    BasicBlock *prep{nullptr};
    BasicBlock *lbody{nullptr};
    BasicBlock *upd{nullptr};
    BasicBlock *merge{nullptr};
    PhiInst *iv{nullptr};
    XIRBuilder b;

    CountedLoopFixture(Module &module_, BasicBlock *parent, int32_t bound, int32_t step = 1) noexcept
        : m(module_) {
        b.set_insertion_point(parent);
        loop = b.loop();
        prep = loop->create_prepare_block();
        lbody = loop->create_body_block();
        upd = loop->create_update_block();
        merge = loop->create_merge_block();

        b.set_insertion_point(prep);
        auto *c0 = m.create_constant_zero(Type::of<int>());
        iv = b.phi(Type::of<int>(), {{c0, parent}});
        auto *bound_const = [&]() {
            int32_t v = bound;
            return m.create_constant(Type::of<int>(), &v);
        }();
        auto *cmp = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv, bound_const});
        b.cond_br(cmp, lbody, merge);

        b.set_insertion_point(upd);
        auto *step_const = [&]() {
            int32_t v = step;
            return m.create_constant(Type::of<int>(), &v);
        }();
        auto *inc = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {iv, step_const});
        iv->add_incoming(inc, upd);
        b.br(prep);
    }
};

[[nodiscard]] Value *gep_array_element(XIRBuilder &b, Module &m, Value *array_alloca,
                                       Value *index, const Type *elem_type) noexcept {
    return b.gep(elem_type, array_alloca, {index});
}

}// namespace

void reg_loop_fusion() {

    "loop_fusion_rejects_structured_independent_loops"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);

        constexpr int32_t n = 8;
        auto *arr_a = b.alloca_local(Type::array(Type::of<float>(), n));
        auto *arr_b = b.alloca_local(Type::array(Type::of<float>(), n));

        CountedLoopFixture loop1(m, body, n);
        b.set_insertion_point(loop1.lbody);
        auto *gep_a = gep_array_element(b, m, arr_a, loop1.iv, Type::of<float>());
        auto *val = m.create_constant_zero(Type::of<float>());
        b.store(gep_a, val);
        b.br(loop1.upd);

        b.set_insertion_point(loop1.merge);
        CountedLoopFixture loop2(m, loop1.merge, n);
        b.set_insertion_point(loop2.lbody);
        auto *gep_b = gep_array_element(b, m, arr_b, loop2.iv, Type::of<float>());
        b.store(gep_b, val);
        b.br(loop2.upd);

        b.set_insertion_point(loop2.merge);
        b.return_void();

        auto *loop1_term = body->terminator();
        auto *loop2_term = loop1.merge->terminator();
        auto info = loop_fusion_pass_run_on_function(k);
        expect(info.fused_loop_count == 0u);
        expect(info.structured_cfg_error_count == 1u);
        expect(body->terminator() == loop1_term);
        expect(loop1.merge->terminator() == loop2_term);
        expect(loop1.loop->merge_block() == loop1.merge);
        expect(loop2.loop->merge_block() == loop2.merge);
    };

    "loop_fusion_rejects_structured_dependent_loops"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);

        constexpr int32_t n = 8;
        auto *arr_a = b.alloca_local(Type::array(Type::of<float>(), n));
        auto *arr_b = b.alloca_local(Type::array(Type::of<float>(), n));

        CountedLoopFixture loop1(m, body, n);
        b.set_insertion_point(loop1.lbody);
        auto *gep_a = gep_array_element(b, m, arr_a, loop1.iv, Type::of<float>());
        auto *val = m.create_constant_zero(Type::of<float>());
        b.store(gep_a, val);
        b.br(loop1.upd);

        b.set_insertion_point(loop1.merge);
        CountedLoopFixture loop2(m, loop1.merge, n);
        b.set_insertion_point(loop2.lbody);
        auto *gep_a2 = gep_array_element(b, m, arr_a, loop2.iv, Type::of<float>());
        auto *ld = b.load(Type::of<float>(), gep_a2);
        auto *gep_b = gep_array_element(b, m, arr_b, loop2.iv, Type::of<float>());
        b.store(gep_b, ld);
        b.br(loop2.upd);

        b.set_insertion_point(loop2.merge);
        b.return_void();

        auto info = loop_fusion_pass_run_on_function(k);
        expect(info.fused_loop_count == 0u);
        expect(info.structured_cfg_error_count == 1u);
        expect(body->terminator() == loop1.loop);
        expect(loop1.merge->terminator() == loop2.loop);
    };

    "loop_fusion_empty_module"_test = [] {
        Module m;
        auto info = loop_fusion_pass_run_on_module(&m);
        expect(info.fused_loop_count == 0u);
        expect(info.succeeded());
    };

    "loop_fusion_no_loop"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();

        auto info = loop_fusion_pass_run_on_function(k);
        expect(info.fused_loop_count == 0u);
        expect(info.succeeded());
    };
}

namespace {

struct PlainCountedLoop {
    BasicBlock *preheader;
    BasicBlock *header;
    BasicBlock *latch;
    BasicBlock *exit;
    PhiInst *iv;
};

// Build one counted loop whose body writes a constant to `buffer` at `iv`.
// Returns the blocks so tests can chain another loop at `exit`.
// The loop's exit block is created fresh; it becomes the next loop's
// preheader when chaining loops, or the function exit for the last one.
[[nodiscard]] PlainCountedLoop build_plain_counted_loop(
    Module &m, XIRBuilder &b, BasicBlock *preheader,
    uint32_t bound_value, ResourceArgument *buffer,
    bool write_constant_body) noexcept {
    PlainCountedLoop loop;
    loop.preheader = preheader;
    auto *defn = preheader->parent_function();
    loop.header = defn->create_basic_block();
    loop.latch = defn->create_basic_block();
    loop.exit = defn->create_basic_block();
    auto *zero = m.create_constant_zero(Type::of<uint>());
    auto *one = m.create_constant_one(Type::of<uint>());
    auto *bound = m.create_constant(Type::of<uint>(), &bound_value);
    auto *one_f = m.create_constant_one(Type::of<float>());

    b.set_insertion_point(preheader);
    b.br(loop.header);
    b.set_insertion_point(loop.header);
    loop.iv = b.phi(Type::of<uint>());
    auto *cond = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS,
                        {loop.iv, bound});
    b.cond_br(cond, loop.latch, loop.exit);
    b.set_insertion_point(loop.latch);
    if (write_constant_body) {
        b.call(ResourceWriteOp::BUFFER_WRITE, {buffer, loop.iv, one_f});
    }
    auto *next = b.call(Type::of<uint>(), ArithmeticOp::BINARY_ADD,
                        {loop.iv, one});
    b.br(loop.header);
    loop.iv->add_incoming(zero, preheader);
    loop.iv->add_incoming(next, loop.latch);
    return loop;
}

void expect_module_valid(Module &m) noexcept {
    auto verification = xir_verify_module(&m);
    expect(verification.succeeded())
        << (verification.errors.empty() ? "unknown XIR verification error" :
                                          verification.errors.front().message.c_str());
}

[[nodiscard]] size_t count_cond_br(FunctionDefinition *def) noexcept {
    auto count = 0u;
    def->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<ConditionalBranchInst>()) { count++; }
    });
    return count;
}

}// namespace

void reg_loop_fusion_plain_cfg() {

    "plain_adjacent_loops_fuse_into_one_loop"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        auto *buf_a = k->create_resource_argument(Type::buffer(Type::of<float>()));
        auto *buf_b = k->create_resource_argument(Type::buffer(Type::of<float>()));
        XIRBuilder b;
        auto loop1 = build_plain_counted_loop(m, b, entry, 8u, buf_a, true);
        auto loop2 = build_plain_counted_loop(m, b, loop1.exit, 8u, buf_b, true);
        b.set_insertion_point(loop2.exit);
        b.return_void();

        auto info = loop_fusion_pass_run_on_function(k);
        expect(info.fused_loop_count == 1u);
        // one conditional branch remains (the fused loop's check)
        expect(count_cond_br(k->definition()) == 1u);
        // both buffer writes are still present
        auto write_count = 0u;
        k->definition()->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ResourceWriteInst>()) { write_count++; }
        });
        expect(write_count == 2u);
        expect_module_valid(m);
    };

    "plain_loops_with_different_trip_counts_do_not_fuse"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        auto *buf_a = k->create_resource_argument(Type::buffer(Type::of<float>()));
        auto *buf_b = k->create_resource_argument(Type::buffer(Type::of<float>()));
        XIRBuilder b;
        auto loop1 = build_plain_counted_loop(m, b, entry, 8u, buf_a, true);
        auto loop2 = build_plain_counted_loop(m, b, loop1.exit, 16u, buf_b, true);
        b.set_insertion_point(loop2.exit);
        b.return_void();

        auto info = loop_fusion_pass_run_on_function(k);
        expect(info.fused_loop_count == 0u);
        expect(count_cond_br(k->definition()) == 2u);
        expect_module_valid(m);
    };

    "plain_loops_with_write_read_dependence_do_not_fuse"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        auto *buf_a = k->create_resource_argument(Type::buffer(Type::of<float>()));
        XIRBuilder b;
        auto loop1 = build_plain_counted_loop(m, b, entry, 8u, buf_a, true);
        // second loop READS the buffer the first loop writes
        auto *def = k->definition();
        auto *preheader2 = loop1.exit;
        auto *header2 = def->create_basic_block();
        auto *latch2 = def->create_basic_block();
        auto *exit2 = def->create_basic_block();
        auto *zero = m.create_constant_zero(Type::of<uint>());
        auto *one = m.create_constant_one(Type::of<uint>());
        uint32_t bound_value = 8u;
        auto *bound = m.create_constant(Type::of<uint>(), &bound_value);
        b.set_insertion_point(preheader2);
        b.br(header2);
        b.set_insertion_point(header2);
        auto *iv2 = b.phi(Type::of<uint>());
        auto *cond2 = b.call(Type::of<bool>(), ArithmeticOp::BINARY_LESS, {iv2, bound});
        b.cond_br(cond2, latch2, exit2);
        b.set_insertion_point(latch2);
        auto *read = b.call(Type::of<float>(), ResourceReadOp::BUFFER_READ, {buf_a, iv2});
        static_cast<void>(read);
        auto *next2 = b.call(Type::of<uint>(), ArithmeticOp::BINARY_ADD, {iv2, one});
        b.br(header2);
        iv2->add_incoming(zero, preheader2);
        iv2->add_incoming(next2, latch2);
        b.set_insertion_point(exit2);
        b.return_void();

        auto info = loop_fusion_pass_run_on_function(k);
        expect(info.fused_loop_count == 0u);
        expect_module_valid(m);
    };

    "plain_three_adjacent_loops_fuse_into_one"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *k = make_kernel_with_body(m, entry);
        auto *buf_a = k->create_resource_argument(Type::buffer(Type::of<float>()));
        auto *buf_b = k->create_resource_argument(Type::buffer(Type::of<float>()));
        auto *buf_c = k->create_resource_argument(Type::buffer(Type::of<float>()));
        XIRBuilder b;
        auto loop1 = build_plain_counted_loop(m, b, entry, 8u, buf_a, true);
        auto loop2 = build_plain_counted_loop(m, b, loop1.exit, 8u, buf_b, true);
        auto loop3 = build_plain_counted_loop(m, b, loop2.exit, 8u, buf_c, true);
        b.set_insertion_point(loop3.exit);
        b.return_void();

        auto info = loop_fusion_pass_run_on_function(k);
        expect(info.fused_loop_count == 2u);
        expect(count_cond_br(k->definition()) == 1u);
        auto write_count = 0u;
        k->definition()->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ResourceWriteInst>()) { write_count++; }
        });
        expect(write_count == 3u);
        expect_module_valid(m);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_loop_fusion();
    reg_loop_fusion_plain_cfg();
    return 0;
}
