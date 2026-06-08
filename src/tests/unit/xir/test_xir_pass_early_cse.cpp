#include "ut/ut.hpp"
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/module.h>
#include <luisa/xir/op.h>
#include <luisa/xir/passes/early_cse.h>

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
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_early_cse();
    return 0;
}
