#include "ut/ut.hpp"

#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/lower_ray_query_loop.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] size_t count_functions(Module &m) noexcept {
    size_t n = 0u;
    for ([[maybe_unused]] auto *function : m.function_list()) { ++n; }
    return n;
}

[[nodiscard]] size_t count_blocks(FunctionDefinition *def) noexcept {
    size_t n = 0u;
    for ([[maybe_unused]] auto *block : def->basic_blocks()) { ++n; }
    return n;
}

struct RayQueryFixture {
    KernelFunction *kernel;
    BasicBlock *body;
    RayQueryLoopInst *loop;
    BasicBlock *dispatch;
    BasicBlock *merge;
    RayQueryDispatchInst *dispatch_inst;
    BasicBlock *surface;
    BasicBlock *procedural;
};

[[nodiscard]] RayQueryFixture make_fixture(Module &m) noexcept {
    auto *kernel = m.create_kernel();
    auto *body = kernel->create_body_block();
    XIRBuilder b;
    b.set_insertion_point(body);
    auto *query = b.alloca_local(Type::of<int>());
    auto *loop = b.ray_query_loop();
    auto *dispatch = loop->create_dispatch_block();
    auto *merge = loop->create_merge_block();
    b.set_insertion_point(dispatch);
    auto *dispatch_inst = b.ray_query_dispatch(query);
    dispatch_inst->set_exit_block(merge);
    auto *surface = dispatch_inst->create_on_surface_candidate_block();
    auto *procedural = dispatch_inst->create_on_procedural_candidate_block();
    b.set_insertion_point(procedural);
    b.br(dispatch);
    b.set_insertion_point(merge);
    b.return_void();
    return {kernel, body, loop, dispatch, merge, dispatch_inst, surface, procedural};
}

}// namespace

void register_tests() {
    "single_exit_handlers_are_outlined"_test = [] {
        Module m;
        auto f = make_fixture(m);
        XIRBuilder b;
        b.set_insertion_point(f.surface);
        b.br(f.dispatch);

        auto info = lower_ray_query_loop_pass_run_on_function(f.kernel);
        expect(info.lowered_loop_count == 1u);
        expect(info.error_count == 0u);
        expect(info.succeeded());
        bool found_pipeline = false;
        f.body->traverse_instructions([&](Instruction *inst) noexcept {
            found_pipeline |= inst->isa<RayQueryPipelineInst>();
        });
        expect(found_pipeline);
        expect(count_functions(m) == 3u);
    };

    "multiple_handler_exits_are_rejected_atomically"_test = [] {
        Module m;
        auto f = make_fixture(m);
        auto *cond = f.kernel->create_value_argument(Type::of<bool>());
        auto *left = f.kernel->create_basic_block();
        auto *right = f.kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(f.surface);
        auto *split = b.cond_br(cond, left, right);
        b.set_insertion_point(left);
        auto *left_exit = b.br(f.dispatch);
        b.set_insertion_point(right);
        auto *right_exit = b.br(f.dispatch);
        auto function_count = count_functions(m);
        auto block_count = count_blocks(f.kernel->definition());

        auto info = lower_ray_query_loop_pass_run_on_function(f.kernel);
        expect(info.lowered_loop_count == 0u);
        expect(info.error_count == 1u);
        expect(!info.succeeded());
        expect(count_functions(m) == function_count);
        expect(count_blocks(f.kernel->definition()) == block_count);
        expect(f.body->terminator() == f.loop);
        expect(f.dispatch->terminator() == f.dispatch_inst);
        expect(f.surface->terminator() == split);
        expect(left->terminator() == left_exit);
        expect(right->terminator() == right_exit);
    };

    "shared_handler_tail_with_phi_is_rejected_before_outlining"_test = [] {
        Module m;
        auto f = make_fixture(m);
        auto *shared_tail = f.kernel->create_basic_block();
        auto *zero = m.create_constant_zero(Type::of<int>());
        auto *one = m.create_constant_one(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(f.surface);
        auto *surface_exit = b.br(shared_tail);
        // Replace the fixture's original procedural exit with the shared tail.
        f.procedural->terminator()->remove_self();
        b.set_insertion_point(f.procedural);
        auto *procedural_exit = b.br(shared_tail);
        b.set_insertion_point(shared_tail);
        auto *join_phi = b.phi(Type::of<int>());
        join_phi->add_incoming(zero, f.surface);
        join_phi->add_incoming(one, f.procedural);
        auto *tail_exit = b.br(f.dispatch);
        auto function_count = count_functions(m);
        auto block_count = count_blocks(f.kernel->definition());

        auto info = lower_ray_query_loop_pass_run_on_function(f.kernel);
        expect(info.lowered_loop_count == 0u);
        expect(info.error_count == 1u);
        expect(!info.succeeded());
        expect(count_functions(m) == function_count);
        expect(count_blocks(f.kernel->definition()) == block_count);
        expect(f.body->terminator() == f.loop);
        expect(f.dispatch->terminator() == f.dispatch_inst);
        expect(f.surface->terminator() == surface_exit);
        expect(f.procedural->terminator() == procedural_exit);
        expect(shared_tail->terminator() == tail_exit);
        expect(join_phi->is_linked());
        expect(join_phi->incoming_count() == 2u);
        expect(join_phi->incoming(0u).block == f.surface);
        expect(join_phi->incoming(1u).block == f.procedural);
    };

    "invalid_later_loop_keeps_earlier_valid_loop_unchanged"_test = [] {
        Module m;
        auto *kernel = m.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder b;

        // First loop is fully valid and would be outlined without a
        // function-wide preflight.
        b.set_insertion_point(body);
        auto *query0 = b.alloca_local(Type::of<int>());
        auto *loop0 = b.ray_query_loop();
        auto *dispatch0 = loop0->create_dispatch_block();
        auto *merge0 = loop0->create_merge_block();
        b.set_insertion_point(dispatch0);
        auto *dispatch_inst0 = b.ray_query_dispatch(query0);
        dispatch_inst0->set_exit_block(merge0);
        auto *surface0 = dispatch_inst0->create_on_surface_candidate_block();
        auto *procedural0 = dispatch_inst0->create_on_procedural_candidate_block();
        b.set_insertion_point(surface0);
        auto *surface_exit0 = b.br(dispatch0);
        b.set_insertion_point(procedural0);
        auto *procedural_exit0 = b.br(dispatch0);

        // The later loop has two surface exits and must reject the complete
        // function before the first callback or alloca move is created.
        b.set_insertion_point(merge0);
        auto *query1 = b.alloca_local(Type::of<int>());
        auto *loop1 = b.ray_query_loop();
        auto *dispatch1 = loop1->create_dispatch_block();
        auto *merge1 = loop1->create_merge_block();
        b.set_insertion_point(dispatch1);
        auto *dispatch_inst1 = b.ray_query_dispatch(query1);
        dispatch_inst1->set_exit_block(merge1);
        auto *surface1 = dispatch_inst1->create_on_surface_candidate_block();
        auto *procedural1 = dispatch_inst1->create_on_procedural_candidate_block();
        auto *left = kernel->create_basic_block();
        auto *right = kernel->create_basic_block();
        auto *cond = kernel->create_value_argument(Type::of<bool>());
        b.set_insertion_point(surface1);
        auto *split = b.cond_br(cond, left, right);
        b.set_insertion_point(left);
        auto *left_exit = b.br(dispatch1);
        b.set_insertion_point(right);
        auto *right_exit = b.br(dispatch1);
        b.set_insertion_point(procedural1);
        auto *procedural_exit1 = b.br(dispatch1);
        b.set_insertion_point(merge1);
        b.return_void();
        auto function_count = count_functions(m);
        auto block_count = count_blocks(kernel->definition());

        auto info = lower_ray_query_loop_pass_run_on_function(kernel);
        expect(info.lowered_loop_count == 0u);
        expect(info.error_count == 1u);
        expect(!info.succeeded());
        expect(count_functions(m) == function_count);
        expect(count_blocks(kernel->definition()) == block_count);
        expect(body->terminator() == loop0);
        expect(dispatch0->terminator() == dispatch_inst0);
        expect(surface0->terminator() == surface_exit0);
        expect(procedural0->terminator() == procedural_exit0);
        expect(merge0->terminator() == loop1);
        expect(dispatch1->terminator() == dispatch_inst1);
        expect(surface1->terminator() == split);
        expect(left->terminator() == left_exit);
        expect(right->terminator() == right_exit);
        expect(procedural1->terminator() == procedural_exit1);
        expect(query0->parent_block() == body);
        expect(query1->parent_block() == merge0);
    };

    "null_handler_is_rejected_before_outlining"_test = [] {
        Module m;
        auto *kernel = m.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *query = b.alloca_local(Type::of<int>());
        auto *loop = b.ray_query_loop();
        auto *dispatch = loop->create_dispatch_block();
        auto *merge = loop->create_merge_block();
        b.set_insertion_point(dispatch);
        auto *dispatch_inst = b.ray_query_dispatch(query);
        dispatch_inst->set_exit_block(merge);
        auto *surface = dispatch_inst->create_on_surface_candidate_block();
        b.set_insertion_point(surface);
        b.br(dispatch);
        b.set_insertion_point(merge);
        b.return_void();
        auto function_count = count_functions(m);
        auto block_count = count_blocks(kernel->definition());

        auto info = lower_ray_query_loop_pass_run_on_function(kernel);

        expect(info.lowered_loop_count == 0u);
        expect(info.error_count == 1u);
        expect(!info.succeeded());
        expect(count_functions(m) == function_count);
        expect(count_blocks(kernel->definition()) == block_count);
        expect(body->terminator() == loop);
        expect(dispatch->terminator() == dispatch_inst);
        expect(dispatch_inst->on_procedural_candidate_block() == nullptr);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    register_tests();
    return 0;
}
