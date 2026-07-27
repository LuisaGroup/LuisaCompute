#include "ut/ut.hpp"
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/metadata/location.h>
#include <luisa/xir/module.h>
#include <luisa/xir/op.h>
#include <luisa/xir/passes/early_cse.h>
#include <luisa/xir/verifier.h>

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

void reg_early_cse() {

    "cse_eliminates_duplicate_add"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        int one = 1, two = 2;
        auto *c1 = m.create_constant(Type::of<int>(), &one);
        auto *c2 = m.create_constant(Type::of<int>(), &two);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *add1 = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {c1, c2});
        auto *add2 = b.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {c1, c2});
        b.return_(add2);
        size_t before = 0;
        body->traverse_instructions([&](Instruction *) noexcept { ++before; });
        auto info = early_cse_pass_run_on_function(k);
        expect(info.eliminated_inst_count == 1u);
        size_t after = 0;
        body->traverse_instructions([&](Instruction *) noexcept { ++after; });
        expect(after == before - 1u);
    };

    "cse_preserves_side_effects"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        int one = 1;
        auto *c1 = m.create_constant(Type::of<int>(), &one);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *alloca1 = b.alloca_local(Type::of<int>());
        b.store(alloca1, c1);
        auto *alloca2 = b.alloca_local(Type::of<int>());
        b.store(alloca2, c1);
        b.return_void();
        size_t before = 0;
        body->traverse_instructions([&](Instruction *) noexcept { ++before; });
        auto info = early_cse_pass_run_on_function(k);
        expect(info.eliminated_inst_count == 0u);
        size_t after = 0;
        body->traverse_instructions([&](Instruction *) noexcept { ++after; });
        expect(after == before);
    };

    "cse_handles_empty_function"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = early_cse_pass_run_on_function(k);
        expect(info.eliminated_inst_count == 0u);
    };

    "cse_does_not_merge_coro_resume_side_effects"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *resume0 = b.coro_resume(7u, nullptr);
        auto *resume1 = b.coro_resume(7u, nullptr);
        b.return_void();
        auto info = early_cse_pass_run_on_function(k);
        expect(info.eliminated_inst_count == 0u);
        expect(resume0->is_linked());
        expect(resume1->is_linked());
    };

    "cse_does_not_merge_mutable_accel_queries"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *accel = k->create_resource_argument(Type::of<Accel>());
        auto *instance = m.create_constant_zero(Type::of<uint>());
        auto *new_user_id = m.create_constant_one(Type::of<uint>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sink = b.alloca_local(Type::of<uint2>());
        auto *before = b.call(
            Type::of<uint>(), ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID,
            {accel, instance});
        b.call(ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID,
               {accel, instance, new_user_id});
        auto *after = b.call(
            Type::of<uint>(), ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID,
            {accel, instance});
        auto *pair = b.call(Type::of<uint2>(), ArithmeticOp::AGGREGATE,
                            {before, after});
        b.store(sink, pair);
        b.return_void();

        auto info = early_cse_pass_run_on_function(k);

        expect(info.eliminated_inst_count == 0u);
        expect(pair->operand(0u) == before);
        expect(pair->operand(1u) == after);
    };

    "cse_still_merges_stable_buffer_size_queries"_test = [] {
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *buffer =
            k->create_resource_argument(Type::buffer(Type::of<float>()));
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *sink = b.alloca_local(Type::of<uint2>());
        auto *size0 = b.call(Type::of<uint>(), ResourceQueryOp::BUFFER_SIZE,
                             {buffer});
        auto *size1 = b.call(Type::of<uint>(), ResourceQueryOp::BUFFER_SIZE,
                             {buffer});
        auto *pair = b.call(Type::of<uint2>(), ArithmeticOp::AGGREGATE,
                            {size0, size1});
        b.store(sink, pair);
        b.return_void();

        auto info = early_cse_pass_run_on_function(k);

        expect(info.eliminated_inst_count == 1u);
        expect(pair->operand(0u) == size0);
        expect(pair->operand(1u) == size0);
    };

    "cse_annotated_duplicate_keeps_distinct_metadata_owner"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int2>());
        auto *x = f->create_value_argument(Type::of<int>());
        auto *body = f->create_body_block();
        auto *one = m.create_constant_one(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *first = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {x, one});
        auto *annotated = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {x, one});
        annotated->set_location("early_cse_metadata.cpp", 9);
        auto *pair = b.call(
            Type::of<int2>(), ArithmeticOp::AGGREGATE,
            {first, annotated});
        b.return_(pair);

        expect(xir_verify_module(&m).succeeded());
        auto info = early_cse_pass_run_on_function(f);
        expect(info.eliminated_inst_count == 0u);
        expect(pair->operand(0u) == first);
        expect(pair->operand(1u) == annotated);
        expect(annotated->find_metadata<LocationMD>() != nullptr);
        expect(xir_verify_module(&m).succeeded());
    };

    "cse_annotated_leader_does_not_absorb_plain_duplicate"_test = [] {
        Module m;
        auto *f = m.create_callable(Type::of<int2>());
        auto *x = f->create_value_argument(Type::of<int>());
        auto *body = f->create_body_block();
        auto *one = m.create_constant_one(Type::of<int>());
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *annotated = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {x, one});
        annotated->set_location("early_cse_leader.cpp", 17);
        auto *plain = b.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {x, one});
        auto *pair = b.call(
            Type::of<int2>(), ArithmeticOp::AGGREGATE,
            {annotated, plain});
        b.return_(pair);

        expect(xir_verify_module(&m).succeeded());
        auto info = early_cse_pass_run_on_function(f);
        expect(info.eliminated_inst_count == 0u);
        expect(pair->operand(0u) == annotated);
        expect(pair->operand(1u) == plain);
        expect(annotated->find_metadata<LocationMD>() != nullptr);
        expect(plain->metadata_list().empty());
        expect(xir_verify_module(&m).succeeded());
    };

    "cse_null_inputs_are_noops"_test = [] {
        expect(early_cse_pass_run_on_function(nullptr)
                   .eliminated_inst_count == 0u);
        expect(early_cse_pass_run_on_module(nullptr)
                   .eliminated_inst_count == 0u);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_early_cse();
    return 0;
}
