// Regression tests for projected coroutine scope dataflow and its worklist
// schedule. Semantic equivalence is checked against the independent pointer
// oracle through the explicit diagnostic environment switch.

#include "ut/ut.hpp"

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>

#include <luisa/ast/type_registry.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void set_environment_variable(
    const char *name, const char *value) noexcept {
#ifdef _WIN32
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
#endif
}

struct ScopedEnvironmentVariable {
    std::string name;
    std::optional<std::string> previous;

    ScopedEnvironmentVariable(
        const char *env_name, const char *value)
        : name{env_name} {
        if (auto *old_value = std::getenv(env_name)) {
            previous.emplace(old_value);
        }
        set_environment_variable(name.c_str(), value);
    }

    ~ScopedEnvironmentVariable() noexcept {
        set_environment_variable(
            name.c_str(),
            previous ? previous->c_str() : nullptr);
    }

    ScopedEnvironmentVariable(
        const ScopedEnvironmentVariable &) = delete;
    ScopedEnvironmentVariable &operator=(
        const ScopedEnvironmentVariable &) = delete;
};

}// namespace

int main(int argc, char *argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    "stats_are_reset_for_rejected_input"_test = [] {
        CoroCfgDistillStats stats{
            .value_atom_count = 1u,
            .scope_count = 2u,
            .projected_scope_atom_count = 3u,
            .max_projected_scope_atom_count = 4u,
            .block_membership_count = 5u,
            .must_block_evaluation_count = 6u,
            .may_block_evaluation_count = 7u};
        auto result = coro_cfg_distill_pass_run_on_function(
            nullptr, {.stats = &stats});
        expect(!result.succeeded());
        expect(stats.value_atom_count == 0u);
        expect(stats.scope_count == 0u);
        expect(stats.projected_scope_atom_count == 0u);
        expect(stats.max_projected_scope_atom_count == 0u);
        expect(stats.block_membership_count == 0u);
        expect(stats.must_block_evaluation_count == 0u);
        expect(stats.may_block_evaluation_count == 0u);
    };

    "scope_projection_preserves_full_dataflow_fixed_point"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *entry = kernel->create_body_block();
        auto *left = kernel->create_basic_block();
        auto *right = kernel->create_basic_block();
        auto *suspend = kernel->create_basic_block();
        auto *resume = kernel->create_basic_block();
        XIRBuilder builder;
        auto *condition =
            module.create_constant_one(Type::of<bool>());
        auto *one = module.create_constant_one(Type::of<int>());

        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(Type::of<int>());
        state->set_name("projected_state");
        builder.cond_br(condition, left, right);

        builder.set_insertion_point(left);
        auto *left_value = builder.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {one, one});
        builder.store(state, left_value);
        builder.br(suspend);

        builder.set_insertion_point(right);
        auto *right_value = builder.call(
            Type::of<int>(), ArithmeticOp::BINARY_SUB, {one, one});
        builder.store(state, right_value);
        builder.br(suspend);

        builder.set_insertion_point(suspend);
        builder.coro_suspend(17u, "project", nullptr);

        builder.set_insertion_point(resume);
        builder.coro_resume(17u, nullptr);
        auto *chain = static_cast<Value *>(one);
        for (auto i = 0u; i < 256u; ++i) {
            chain = builder.call(
                Type::of<int>(), ArithmeticOp::BINARY_ADD,
                {chain, one});
        }
        static_cast<void>(chain);
        static_cast<void>(builder.load(Type::of<int>(), state));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        ScopedEnvironmentVariable verify_oracle{
            "LUISA_CORO_VERIFY_DENSE_DATAFLOW", "1"};
        CoroCfgDistillStats stats;
        auto result = coro_cfg_distill_pass_run_on_function(
            kernel, {.stats = &stats});

        expect(result.succeeded());
        expect(stats.scope_count == 2u);
        expect(stats.value_atom_count > 256u);
        expect(stats.projected_scope_atom_count <
               stats.value_atom_count * stats.scope_count);
        expect(stats.max_projected_scope_atom_count <
               stats.value_atom_count);
        // Both induced scope CFGs are acyclic. Reverse postorder evaluates
        // every forward equation once; there is no backedge to trigger a
        // revisit in either the must or may fixed point.
        expect(stats.must_block_evaluation_count ==
               stats.block_membership_count);
        expect(stats.may_block_evaluation_count ==
               stats.block_membership_count);
        expect(std::find_if(
                   result.frame_values.begin(),
                   result.frame_values.end(),
                   [&](const auto &field) noexcept {
                       return field.value == state;
                   }) != result.frame_values.end());
    };
    return 0;
}
