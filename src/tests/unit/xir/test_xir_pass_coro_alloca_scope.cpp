// Tests for coroutine-semantic local-allocation lifetime contraction.

#include "ut/ut.hpp"

#include <algorithm>
#include <array>

#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_alloca_scope.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] KernelFunction *make_kernel(
    Module &module, BasicBlock *&entry) noexcept {
    auto *kernel = module.create_kernel();
    entry = kernel->create_body_block();
    return kernel;
}

[[nodiscard]] bool frame_contains(
    const CoroCfgDistillResult &cfg, Value *value) noexcept {
    return std::any_of(
        cfg.frame_values.begin(), cfg.frame_values.end(),
        [value](auto &&field) noexcept { return field.value == value; });
}

[[nodiscard]] size_t instruction_index(
    BasicBlock *block, Instruction *needle) noexcept {
    auto index = size_t{0u};
    for (auto *instruction : block->instructions()) {
        if (instruction == needle) { return index; }
        ++index;
    }
    return index;
}

[[nodiscard]] CallableFunction *make_ray_query_capture_handler(
    Module &module, const Type *query_type,
    bool read_capture, bool write_capture) noexcept {
    auto *handler = module.create_callable(nullptr);
    handler->create_reference_argument(query_type);
    auto *capture = handler->create_reference_argument(Type::of<uint>());
    XIRBuilder builder;
    builder.set_insertion_point(handler->create_body_block());
    if (read_capture) {
        static_cast<void>(builder.load(Type::of<uint>(), capture));
    }
    if (write_capture) {
        builder.store(
            capture, module.create_constant_one(Type::of<uint>()));
    }
    builder.return_void();
    return handler;
}

void check_counted_array_capacity_guard(
    std::uint32_t guard_capacity, bool write_element,
    bool expect_contraction) {
    Module module;
    BasicBlock *entry;
    auto *kernel = make_kernel(module, entry);
    auto *allocated = kernel->create_value_argument(Type::of<bool>());
    auto *repeat = kernel->create_value_argument(Type::of<bool>());
    auto *read_index = kernel->create_value_argument(Type::of<uint>());
    auto *resume = kernel->create_basic_block();
    auto *header = kernel->create_basic_block();
    auto *append = kernel->create_basic_block();
    auto *skip = kernel->create_basic_block();
    auto *latch = kernel->create_basic_block();
    auto *done = kernel->create_basic_block();
    auto *array_type = Type::array(Type::of<uint>(), 2u);
    XIRBuilder builder;

    builder.set_insertion_point(entry);
    auto *array = builder.alloca_local(array_type);
    auto *count = builder.alloca_local(Type::of<uint>());
    builder.coro_suspend(20u, "counted-capacity-guard", nullptr);

    builder.set_insertion_point(resume);
    builder.coro_resume(20u, nullptr);
    builder.store(count, module.create_constant_zero(Type::of<uint>()));
    auto *sentinel = builder.gep(
        Type::of<uint>(), array,
        {module.create_constant_zero(Type::of<uint>())});
    builder.store(sentinel, module.create_constant_zero(Type::of<uint>()));
    builder.br(header);

    builder.set_insertion_point(header);
    auto *guard_count = builder.load(Type::of<uint>(), count);
    auto *has_capacity = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_LESS,
        {guard_count,
         module.create_constant(Type::of<uint>(), &guard_capacity)});
    // Mirrors retained = allocated && count < capacity in the renderer.
    // The true edge proves both conjuncts even when repeat permits more loop
    // iterations than the physical array can hold.
    auto *retained = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_BIT_AND,
        {allocated, has_capacity});
    builder.cond_br(retained, append, skip);

    builder.set_insertion_point(append);
    auto *append_index = builder.load(Type::of<uint>(), count);
    if (write_element) {
        auto *element = builder.gep(
            Type::of<uint>(), array, {append_index});
        builder.store(
            element, module.create_constant_one(Type::of<uint>()));
    }
    auto *old_count = builder.load(Type::of<uint>(), count);
    auto *next_count = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {old_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, next_count);
    builder.br(latch);

    builder.set_insertion_point(skip);
    builder.br(latch);

    builder.set_insertion_point(latch);
    builder.cond_br(repeat, header, done);

    builder.set_insertion_point(done);
    auto *current_count = builder.load(Type::of<uint>(), count);
    auto *in_prefix = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_LESS,
        {read_index, current_count});
    auto *bounded_index = builder.call(
        Type::of<uint>(), ArithmeticOp::SELECT,
        {module.create_constant_zero(Type::of<uint>()),
         read_index, in_prefix});
    auto *selected = builder.gep(
        Type::of<uint>(), array, {bounded_index});
    static_cast<void>(builder.load(Type::of<uint>(), selected));
    builder.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto before = coro_cfg_distill_pass_run_on_function(kernel);
    expect(before.succeeded());
    expect(frame_contains(before, array));

    auto original_block = array->parent_block();
    auto info = coro_alloca_scope_pass_run_on_function(kernel);
    expect(info.initialized_prefix_proof_count ==
           (expect_contraction ? 1u : 0u));
    expect(info.rejected_prior_lifetime_observation_count ==
           (expect_contraction ? 0u : 1u));
    expect(array->parent_block() ==
           (expect_contraction ? resume : original_block));
    expect(xir_verify_module(&module).succeeded());

    auto after = coro_cfg_distill_pass_run_on_function(kernel);
    expect(after.succeeded());
    expect(frame_contains(after, array) == !expect_contraction);
}

void check_counted_array_control_guard(
    bool mutate_index_after_guard, bool expect_contraction) {
    Module module;
    BasicBlock *entry;
    auto *kernel = make_kernel(module, entry);
    auto *resume = kernel->create_basic_block();
    auto *append = kernel->create_basic_block();
    auto *guard_snapshot = kernel->create_basic_block();
    auto *guard = kernel->create_basic_block();
    auto *guard_proxy = kernel->create_basic_block();
    auto *consume = kernel->create_basic_block();
    auto *done = kernel->create_basic_block();
    auto *array_type = Type::array(Type::of<uint>(), 4u);
    XIRBuilder builder;

    builder.set_insertion_point(entry);
    auto *array = builder.alloca_local(array_type);
    array->set_name("control_guard_array");
    auto *count = builder.alloca_local(Type::of<uint>());
    count->set_name("control_guard_count");
    auto *index = builder.alloca_local(Type::of<uint>());
    index->set_name("control_guard_index");
    auto *count_copy = builder.alloca_local(Type::of<uint>());
    auto *index_copy = builder.alloca_local(Type::of<uint>());
    auto *less_copy = builder.alloca_local(Type::of<bool>());
    auto *exit_copy = builder.alloca_local(Type::of<bool>());
    builder.coro_suspend(21u, "counted-control-guard", nullptr);

    builder.set_insertion_point(resume);
    builder.coro_resume(21u, nullptr);
    builder.store(count, module.create_constant_zero(Type::of<uint>()));
    builder.br(append);

    builder.set_insertion_point(append);
    auto *append_index = builder.load(Type::of<uint>(), count);
    auto *element = builder.gep(Type::of<uint>(), array, {append_index});
    builder.store(element, module.create_constant_one(Type::of<uint>()));
    auto *old_count = builder.load(Type::of<uint>(), count);
    auto *next_count = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {old_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, next_count);
    builder.store(index, module.create_constant_zero(Type::of<uint>()));
    builder.br(guard_snapshot);

    builder.set_insertion_point(guard_snapshot);
    auto *loaded_count = builder.load(Type::of<uint>(), count);
    builder.store(count_copy, loaded_count);
    builder.br(guard);

    builder.set_insertion_point(guard);
    auto *loaded_index = builder.load(Type::of<uint>(), index);
    auto *copied_count = builder.load(Type::of<uint>(), count_copy);
    auto *in_prefix = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_LESS,
        {loaded_index, copied_count});
    builder.store(less_copy, in_prefix);
    auto *copied_less = builder.load(Type::of<bool>(), less_copy);
    auto *should_exit = builder.call(
        Type::of<bool>(), ArithmeticOp::UNARY_BIT_NOT,
        {copied_less});
    builder.store(exit_copy, should_exit);
    auto *copied_exit = builder.load(Type::of<bool>(), exit_copy);
    builder.cond_br(copied_exit, done, guard_proxy);

    builder.set_insertion_point(guard_proxy);
    builder.br(consume);

    builder.set_insertion_point(consume);
    if (mutate_index_after_guard) {
        auto *stale_index = builder.load(Type::of<uint>(), index);
        auto *changed_index = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {stale_index, module.create_constant_one(Type::of<uint>())});
        builder.store(index, changed_index);
    }
    auto *read_index = builder.load(Type::of<uint>(), index);
    builder.store(index_copy, read_index);
    auto *copied_index = builder.load(Type::of<uint>(), index_copy);
    auto *selected = builder.gep(Type::of<uint>(), array, {copied_index});
    static_cast<void>(builder.load(Type::of<uint>(), selected));
    auto *old_index = builder.load(Type::of<uint>(), index);
    auto *next_index = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {old_index, module.create_constant_one(Type::of<uint>())});
    builder.store(index, next_index);
    builder.br(guard);

    builder.set_insertion_point(done);
    builder.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto before = coro_cfg_distill_pass_run_on_function(kernel);
    expect(before.succeeded());
    expect(frame_contains(before, array));

    auto original_block = array->parent_block();
    auto info = coro_alloca_scope_pass_run_on_function(kernel);
    expect(info.initialized_prefix_proof_count ==
           (expect_contraction ? 1u : 0u));
    expect(info.rejected_prior_lifetime_observation_count >=
           (expect_contraction ? 0u : 1u));
    expect(array->parent_block() ==
           (expect_contraction ? append : original_block));
    expect(xir_verify_module(&module).succeeded());

    auto after = coro_cfg_distill_pass_run_on_function(kernel);
    expect(after.succeeded());
    expect(frame_contains(after, array) == !expect_contraction);
}

void check_counted_array_outer_lifetime_placement() {
    Module module;
    BasicBlock *entry;
    auto *kernel = make_kernel(module, entry);
    auto *read_index = kernel->create_value_argument(Type::of<uint>());
    auto *resume = kernel->create_basic_block();
    auto *header = kernel->create_basic_block();
    auto *append = kernel->create_basic_block();
    auto *read_guard = kernel->create_basic_block();
    auto *read = kernel->create_basic_block();
    auto *latch = kernel->create_basic_block();
    auto *done = kernel->create_basic_block();
    auto *array_type = Type::array(Type::of<uint>(), 4u);
    XIRBuilder builder;

    builder.set_insertion_point(entry);
    auto *array = builder.alloca_local(array_type);
    auto *count = builder.alloca_local(Type::of<uint>());
    auto *read_cursor = builder.alloca_local(Type::of<uint>());
    array->set_name("outer_lifetime_array");
    builder.coro_suspend(22u, "counted-outer-lifetime", nullptr);

    builder.set_insertion_point(resume);
    builder.coro_resume(22u, nullptr);
    builder.store(count, module.create_constant_zero(Type::of<uint>()));
    builder.br(header);

    builder.set_insertion_point(header);
    auto *guard_count = builder.load(Type::of<uint>(), count);
    std::uint32_t capacity = 4u;
    auto *has_capacity = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_LESS,
        {guard_count,
         module.create_constant(Type::of<uint>(), &capacity)});
    builder.cond_br(has_capacity, append, done);

    builder.set_insertion_point(append);
    auto *append_index = builder.load(Type::of<uint>(), count);
    auto *element = builder.gep(Type::of<uint>(), array, {append_index});
    builder.store(element, module.create_constant_one(Type::of<uint>()));
    auto *old_count = builder.load(Type::of<uint>(), count);
    auto *next_count = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {old_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, next_count);
    builder.br(read_guard);

    builder.set_insertion_point(read_guard);
    builder.store(read_cursor, read_index);
    auto *cursor = builder.load(Type::of<uint>(), read_cursor);
    auto *current_count = builder.load(Type::of<uint>(), count);
    auto *in_prefix = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_LESS,
        {cursor, current_count});
    builder.cond_br(in_prefix, read, latch);

    builder.set_insertion_point(read);
    auto *read_cursor_value = builder.load(Type::of<uint>(), read_cursor);
    auto *selected = builder.gep(
        Type::of<uint>(), array, {read_cursor_value});
    static_cast<void>(builder.load(Type::of<uint>(), selected));
    builder.br(latch);

    builder.set_insertion_point(latch);
    builder.br(header);

    builder.set_insertion_point(done);
    builder.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto before = coro_cfg_distill_pass_run_on_function(kernel);
    expect(before.succeeded());
    expect(frame_contains(before, array));

    auto info = coro_alloca_scope_pass_run_on_function(kernel);
    expect(info.initialized_prefix_proof_count == 1u);
    expect(array->parent_block() == resume);
    expect(xir_verify_module(&module).succeeded());

    auto after = coro_cfg_distill_pass_run_on_function(kernel);
    expect(after.succeeded());
    expect(!frame_contains(after, array));
}

void check_counted_array_saved_allocation_ticket(
    bool mutate_ticket_after_increment,
    bool ticket_comes_from_counter,
    bool expect_contraction) {
    Module module;
    BasicBlock *entry;
    auto *kernel = make_kernel(module, entry);
    auto *resume = kernel->create_basic_block();
    auto *append = kernel->create_basic_block();
    auto *forward = kernel->create_basic_block();
    auto *consume = kernel->create_basic_block();
    auto *done = kernel->create_basic_block();
    auto *array_type = Type::array(Type::of<uint>(), 4u);
    XIRBuilder builder;

    builder.set_insertion_point(entry);
    auto *array = builder.alloca_local(array_type);
    array->set_name("saved_ticket_array");
    auto *count = builder.alloca_local(Type::of<uint>());
    count->set_name("saved_ticket_count");
    auto *ticket = builder.alloca_local(Type::of<uint>());
    ticket->set_name("saved_ticket_index");
    auto *forwarded_ticket = builder.alloca_local(Type::of<uint>());
    auto *unrelated = builder.alloca_local(Type::of<uint>());
    builder.coro_suspend(23u, "counted-saved-ticket", nullptr);

    builder.set_insertion_point(resume);
    builder.coro_resume(23u, nullptr);
    builder.store(count, module.create_constant_zero(Type::of<uint>()));
    builder.store(unrelated, module.create_constant_one(Type::of<uint>()));
    builder.br(append);

    builder.set_insertion_point(append);
    auto *ticket_source = builder.load(
        Type::of<uint>(),
        ticket_comes_from_counter ? count : unrelated);
    builder.store(ticket, ticket_source);
    auto *append_index = builder.load(Type::of<uint>(), count);
    auto *element = builder.gep(Type::of<uint>(), array, {append_index});
    builder.store(element, module.create_constant_one(Type::of<uint>()));
    auto *old_count = builder.load(Type::of<uint>(), count);
    auto *next_count = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {old_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, next_count);
    builder.br(forward);

    builder.set_insertion_point(forward);
    if (mutate_ticket_after_increment) {
        auto *old_ticket = builder.load(Type::of<uint>(), ticket);
        auto *next_ticket = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {old_ticket, module.create_constant_one(Type::of<uint>())});
        builder.store(ticket, next_ticket);
    }
    auto *saved_ticket = builder.load(Type::of<uint>(), ticket);
    builder.store(forwarded_ticket, saved_ticket);
    builder.br(consume);

    builder.set_insertion_point(consume);
    auto *read_index = builder.load(Type::of<uint>(), forwarded_ticket);
    auto *selected = builder.gep(Type::of<uint>(), array, {read_index});
    static_cast<void>(builder.load(Type::of<uint>(), selected));
    builder.br(done);

    builder.set_insertion_point(done);
    builder.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto before = coro_cfg_distill_pass_run_on_function(kernel);
    expect(before.succeeded());
    expect(frame_contains(before, array));

    auto original_block = array->parent_block();
    auto info = coro_alloca_scope_pass_run_on_function(kernel);
    expect(info.initialized_prefix_proof_count ==
           (expect_contraction ? 1u : 0u));
    expect(array->parent_block() ==
           (expect_contraction ? resume : original_block));
    expect(xir_verify_module(&module).succeeded());

    auto after = coro_cfg_distill_pass_run_on_function(kernel);
    expect(after.succeeded());
    expect(frame_contains(after, array) == !expect_contraction);
}

void check_counted_array_boolean_guarded_ticket(
    bool mark_failed_allocation_valid,
    bool expect_contraction) {
    Module module;
    BasicBlock *entry;
    auto *kernel = make_kernel(module, entry);
    auto *can_allocate =
        kernel->create_value_argument(Type::of<bool>());
    auto *resume = kernel->create_basic_block();
    auto *success = kernel->create_basic_block();
    auto *failure = kernel->create_basic_block();
    auto *merge = kernel->create_basic_block();
    auto *guard = kernel->create_basic_block();
    auto *consume = kernel->create_basic_block();
    auto *done = kernel->create_basic_block();
    auto *array_type = Type::array(Type::of<uint>(), 4u);
    XIRBuilder builder;

    builder.set_insertion_point(entry);
    auto *array = builder.alloca_local(array_type);
    array->set_name("guarded_ticket_array");
    auto *count = builder.alloca_local(Type::of<uint>());
    auto *ticket = builder.alloca_local(Type::of<uint>());
    auto *valid = builder.alloca_local(Type::of<bool>());
    auto *forwarded_ticket = builder.alloca_local(Type::of<uint>());
    auto *forwarded_valid = builder.alloca_local(Type::of<bool>());
    builder.coro_suspend(24u, "counted-guarded-ticket", nullptr);

    builder.set_insertion_point(resume);
    builder.coro_resume(24u, nullptr);
    builder.store(count, module.create_constant_zero(Type::of<uint>()));
    auto *old_count = builder.load(Type::of<uint>(), count);
    builder.store(ticket, old_count);
    builder.store(valid, module.create_constant_zero(Type::of<bool>()));
    builder.cond_br(can_allocate, success, failure);

    builder.set_insertion_point(success);
    auto *append_index = builder.load(Type::of<uint>(), count);
    auto *element = builder.gep(Type::of<uint>(), array, {append_index});
    builder.store(element, module.create_constant_one(Type::of<uint>()));
    auto *current_count = builder.load(Type::of<uint>(), count);
    auto *next_count = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {current_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, next_count);
    builder.store(valid, module.create_constant_one(Type::of<bool>()));
    builder.br(merge);

    builder.set_insertion_point(failure);
    if (mark_failed_allocation_valid) {
        builder.store(valid, module.create_constant_one(Type::of<bool>()));
    }
    builder.br(merge);

    builder.set_insertion_point(merge);
    auto *merged_ticket = builder.load(Type::of<uint>(), ticket);
    builder.store(forwarded_ticket, merged_ticket);
    auto *merged_valid = builder.load(Type::of<bool>(), valid);
    builder.store(forwarded_valid, merged_valid);
    builder.br(guard);

    builder.set_insertion_point(guard);
    auto *is_valid = builder.load(Type::of<bool>(), forwarded_valid);
    builder.cond_br(is_valid, consume, done);

    builder.set_insertion_point(consume);
    auto *read_index = builder.load(Type::of<uint>(), forwarded_ticket);
    auto *selected = builder.gep(Type::of<uint>(), array, {read_index});
    static_cast<void>(builder.load(Type::of<uint>(), selected));
    builder.br(done);

    builder.set_insertion_point(done);
    builder.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto before = coro_cfg_distill_pass_run_on_function(kernel);
    expect(before.succeeded());
    expect(frame_contains(before, array));
    auto original_block = array->parent_block();
    auto info = coro_alloca_scope_pass_run_on_function(kernel);
    expect(info.initialized_prefix_proof_count ==
           (expect_contraction ? 1u : 0u));
    expect(array->parent_block() ==
           (expect_contraction ? resume : original_block));
    expect(xir_verify_module(&module).succeeded());
    auto after = coro_cfg_distill_pass_run_on_function(kernel);
    expect(after.succeeded());
    expect(frame_contains(after, array) == !expect_contraction);
}

void check_counted_array_conjunctive_guarded_ticket(
    bool guard_emission_path, bool expect_contraction) {
    Module module;
    BasicBlock *entry;
    auto *kernel = make_kernel(module, entry);
    auto *emission_path =
        kernel->create_value_argument(Type::of<bool>());
    auto *survives_cutoff =
        kernel->create_value_argument(Type::of<bool>());
    auto *can_allocate =
        kernel->create_value_argument(Type::of<bool>());
    auto *resume = kernel->create_basic_block();
    auto *emission = kernel->create_basic_block();
    auto *allocate = kernel->create_basic_block();
    auto *allocation_success = kernel->create_basic_block();
    auto *allocation_failure = kernel->create_basic_block();
    auto *allocation_merge = kernel->create_basic_block();
    auto *mode_merge = kernel->create_basic_block();
    auto *setup = kernel->create_basic_block();
    auto *consume = kernel->create_basic_block();
    auto *done = kernel->create_basic_block();
    auto *array_type = Type::array(Type::of<uint>(), 4u);
    XIRBuilder builder;

    builder.set_insertion_point(entry);
    auto *array = builder.alloca_local(array_type);
    array->set_name("conjunctive_guard_array");
    auto *count = builder.alloca_local(Type::of<uint>());
    count->set_name("conjunctive_guard_count");
    auto *ticket = builder.alloca_local(Type::of<uint>());
    ticket->set_name("conjunctive_guard_ticket");
    auto *valid = builder.alloca_local(Type::of<bool>());
    valid->set_name("conjunctive_guard_valid");
    auto *setup_active = builder.alloca_local(Type::of<bool>());
    setup_active->set_name("conjunctive_guard_setup_active");
    auto *emission_copy = builder.alloca_local(Type::of<bool>());
    emission_copy->set_name("conjunctive_guard_emission");
    auto *not_emission_copy = builder.alloca_local(Type::of<bool>());
    not_emission_copy->set_name("conjunctive_guard_not_emission");
    builder.coro_suspend(25u, "counted-conjunctive-guard", nullptr);

    builder.set_insertion_point(resume);
    builder.coro_resume(25u, nullptr);
    builder.store(count, module.create_constant_zero(Type::of<uint>()));
    builder.store(ticket, module.create_constant_zero(Type::of<uint>()));
    builder.store(valid, module.create_constant_zero(Type::of<bool>()));
    builder.store(setup_active, module.create_constant_zero(Type::of<bool>()));
    builder.store(emission_copy, emission_path);
    auto *is_emission = builder.load(Type::of<bool>(), emission_copy);
    builder.cond_br(is_emission, emission, allocate);

    builder.set_insertion_point(emission);
    builder.store(setup_active, survives_cutoff);
    builder.br(mode_merge);

    builder.set_insertion_point(allocate);
    auto *old_count = builder.load(Type::of<uint>(), count);
    builder.store(ticket, old_count);
    builder.cond_br(
        can_allocate, allocation_success, allocation_failure);

    builder.set_insertion_point(allocation_success);
    auto *append_index = builder.load(Type::of<uint>(), count);
    auto *element = builder.gep(Type::of<uint>(), array, {append_index});
    builder.store(element, module.create_constant_one(Type::of<uint>()));
    auto *current_count = builder.load(Type::of<uint>(), count);
    auto *next_count = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {current_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, next_count);
    builder.store(valid, module.create_constant_one(Type::of<bool>()));
    builder.br(allocation_merge);

    builder.set_insertion_point(allocation_failure);
    builder.br(allocation_merge);

    builder.set_insertion_point(allocation_merge);
    auto *allocated = builder.load(Type::of<bool>(), valid);
    builder.store(setup_active, allocated);
    builder.br(mode_merge);

    builder.set_insertion_point(mode_merge);
    auto *active = builder.load(Type::of<bool>(), setup_active);
    builder.cond_br(active, setup, done);

    builder.set_insertion_point(setup);
    if (guard_emission_path) {
        auto *emission_value =
            builder.load(Type::of<bool>(), emission_copy);
        auto *not_emission = builder.call(
            Type::of<bool>(), ArithmeticOp::UNARY_BIT_NOT,
            {emission_value});
        builder.store(not_emission_copy, not_emission);
        auto *guard = builder.load(Type::of<bool>(), not_emission_copy);
        builder.cond_br(guard, consume, done);
    } else {
        builder.br(consume);
    }

    builder.set_insertion_point(consume);
    auto *selected_index = builder.load(Type::of<uint>(), ticket);
    auto *selected = builder.gep(
        Type::of<uint>(), array, {selected_index});
    static_cast<void>(builder.load(Type::of<uint>(), selected));
    builder.br(done);

    builder.set_insertion_point(done);
    builder.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto before = coro_cfg_distill_pass_run_on_function(kernel);
    expect(before.succeeded());
    expect(frame_contains(before, array));
    auto original_block = array->parent_block();
    auto info = coro_alloca_scope_pass_run_on_function(kernel);
    expect(info.initialized_prefix_proof_count ==
           (expect_contraction ? 1u : 0u));
    expect(array->parent_block() ==
           (expect_contraction ? resume : original_block));
    expect(xir_verify_module(&module).succeeded());
    auto after = coro_cfg_distill_pass_run_on_function(kernel);
    expect(after.succeeded());
    expect(frame_contains(after, array) == !expect_contraction);
}

void check_counted_array_nonwrapping_decrement(
    bool guard_positive_count, bool expect_contraction) {
    Module module;
    BasicBlock *entry;
    auto *kernel = make_kernel(module, entry);
    auto *create_initial_element =
        kernel->create_value_argument(Type::of<bool>());
    auto *resume = kernel->create_basic_block();
    auto *initial_append = kernel->create_basic_block();
    auto *initial_skip = kernel->create_basic_block();
    auto *initial_merge = kernel->create_basic_block();
    auto *decrement = kernel->create_basic_block();
    auto *replacement_append = kernel->create_basic_block();
    auto *consume = kernel->create_basic_block();
    auto *done = kernel->create_basic_block();
    auto *array_type = Type::array(Type::of<uint>(), 4u);
    XIRBuilder builder;

    builder.set_insertion_point(entry);
    auto *array = builder.alloca_local(array_type);
    array->set_name("nonwrapping_decrement_array");
    auto *count = builder.alloca_local(Type::of<uint>());
    count->set_name("nonwrapping_decrement_count");
    auto *ticket = builder.alloca_local(Type::of<uint>());
    ticket->set_name("nonwrapping_decrement_ticket");
    builder.coro_suspend(26u, "counted-nonwrapping-decrement", nullptr);

    builder.set_insertion_point(resume);
    builder.coro_resume(26u, nullptr);
    builder.store(count, module.create_constant_zero(Type::of<uint>()));
    builder.cond_br(
        create_initial_element, initial_append, initial_skip);

    builder.set_insertion_point(initial_append);
    auto *initial_index = builder.load(Type::of<uint>(), count);
    auto *initial_element = builder.gep(
        Type::of<uint>(), array, {initial_index});
    builder.store(
        initial_element, module.create_constant_one(Type::of<uint>()));
    auto *initial_count = builder.load(Type::of<uint>(), count);
    auto *initial_next = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {initial_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, initial_next);
    builder.br(initial_merge);

    builder.set_insertion_point(initial_skip);
    builder.br(initial_merge);

    builder.set_insertion_point(initial_merge);
    if (guard_positive_count) {
        auto *current_count = builder.load(Type::of<uint>(), count);
        auto *positive = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_GREATER,
            {current_count,
             module.create_constant_zero(Type::of<uint>())});
        builder.cond_br(positive, decrement, done);
    } else {
        builder.br(decrement);
    }

    builder.set_insertion_point(decrement);
    auto *count_before_decrement = builder.load(Type::of<uint>(), count);
    auto *decremented = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_SUB,
        {count_before_decrement,
         module.create_constant_one(Type::of<uint>())});
    builder.store(count, decremented);
    builder.br(replacement_append);

    builder.set_insertion_point(replacement_append);
    auto *replacement_index = builder.load(Type::of<uint>(), count);
    builder.store(ticket, replacement_index);
    auto *replacement_element = builder.gep(
        Type::of<uint>(), array, {replacement_index});
    builder.store(
        replacement_element, module.create_constant_one(Type::of<uint>()));
    auto *replacement_count = builder.load(Type::of<uint>(), count);
    auto *replacement_next = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {replacement_count,
         module.create_constant_one(Type::of<uint>())});
    builder.store(count, replacement_next);
    builder.br(consume);

    builder.set_insertion_point(consume);
    auto *read_index = builder.load(Type::of<uint>(), ticket);
    auto *selected = builder.gep(
        Type::of<uint>(), array, {read_index});
    static_cast<void>(builder.load(Type::of<uint>(), selected));
    builder.br(done);

    builder.set_insertion_point(done);
    builder.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto before = coro_cfg_distill_pass_run_on_function(kernel);
    expect(before.succeeded());
    expect(frame_contains(before, array));
    auto original_block = array->parent_block();
    auto info = coro_alloca_scope_pass_run_on_function(kernel);
    expect(info.initialized_prefix_proof_count ==
           (expect_contraction ? 1u : 0u));
    expect(array->parent_block() ==
           (expect_contraction ? resume : original_block));
    expect(xir_verify_module(&module).succeeded());
    auto after = coro_cfg_distill_pass_run_on_function(kernel);
    expect(after.succeeded());
    expect(frame_contains(after, array) == !expect_contraction);
}

void check_counted_array_masked_witness(
    bool assume_initial_mask_zero,
    bool set_mask_without_append,
    bool overwrite_mask_with_unknown,
    bool reset_count_after_initial_contract,
    bool route_flags_through_copy,
    bool expect_contraction) {
    Module module;
    BasicBlock *entry;
    auto *kernel = make_kernel(module, entry);
    auto *initial_flags =
        kernel->create_value_argument(Type::of<uint>());
    auto *append_closure =
        kernel->create_value_argument(Type::of<bool>());
    auto *choose_dynamic =
        kernel->create_value_argument(Type::of<bool>());
    auto *dynamic_index =
        kernel->create_value_argument(Type::of<uint>());
    auto *unknown_flags =
        kernel->create_value_argument(Type::of<uint>());
    auto *resume = kernel->create_basic_block();
    auto *append = kernel->create_basic_block();
    auto *skip_append = kernel->create_basic_block();
    auto *allocation_merge = kernel->create_basic_block();
    auto *pick = kernel->create_basic_block();
    auto *dynamic_guard = kernel->create_basic_block();
    auto *dynamic_pick = kernel->create_basic_block();
    auto *pick_merge = kernel->create_basic_block();
    auto *done = kernel->create_basic_block();
    auto *array_type = Type::array(Type::of<uint>(), 4u);
    constexpr uint32_t scatter_mask = 0x14u;
    XIRBuilder builder;

    builder.set_insertion_point(entry);
    auto *array = builder.alloca_local(array_type);
    array->set_name("masked_witness_array");
    auto *count = builder.alloca_local(Type::of<uint>());
    count->set_name("masked_witness_count");
    auto *flags = builder.alloca_local(Type::of<uint>());
    flags->set_name("masked_witness_flags");
    auto *flag_copy = builder.alloca_local(Type::of<uint>());
    flag_copy->set_name("masked_witness_flag_copy");
    auto *sampled = builder.alloca_local(Type::of<uint>());
    sampled->set_name("masked_witness_sampled");
    auto *candidate = builder.alloca_local(Type::of<uint>());
    candidate->set_name("masked_witness_candidate");
    builder.coro_suspend(28u, "counted-masked-witness", nullptr);

    builder.set_insertion_point(resume);
    builder.coro_resume(28u, nullptr);
    if (!reset_count_after_initial_contract &&
        !route_flags_through_copy) {
        builder.store(count, module.create_constant_zero(Type::of<uint>()));
    }
    builder.store(flags, initial_flags);
    builder.store(candidate, dynamic_index);
    if (assume_initial_mask_zero) {
        auto *initial = builder.load(Type::of<uint>(), flags);
        auto *masked = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_BIT_AND,
            {initial, module.create_constant(
                          Type::of<uint>(), &scatter_mask)});
        auto *is_zero = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
            {masked, module.create_constant_zero(Type::of<uint>())});
        builder.assume_(is_zero, "static flags contain no closure bits");
    }
    // A counter reset changes the truth of C>0 but does not change S. The
    // analysis must therefore retain an independently proved S&M==0 fact
    // across the reset instead of invalidating the implication wholesale.
    if (route_flags_through_copy) {
        auto *initial = builder.load(Type::of<uint>(), flags);
        builder.store(flag_copy, initial);
        // Model a lowering-created snapshot that precedes the counted
        // storage reset. A fresh lifetime placed after this reset would cut
        // the exact def-use chain needed by the later guarded read.
        builder.store(count, module.create_constant_zero(Type::of<uint>()));
    } else if (reset_count_after_initial_contract) {
        builder.store(count, module.create_constant_zero(Type::of<uint>()));
    }
    builder.cond_br(append_closure, append, skip_append);

    builder.set_insertion_point(append);
    auto *append_index = builder.load(Type::of<uint>(), count);
    auto *element = builder.gep(Type::of<uint>(), array, {append_index});
    builder.store(element, module.create_constant_one(Type::of<uint>()));
    auto *old_count = builder.load(Type::of<uint>(), count);
    auto *next_count = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {old_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, next_count);
    auto *old_flags = builder.load(
        Type::of<uint>(),
        route_flags_through_copy ? flag_copy : flags);
    auto *new_flags = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_BIT_OR,
        {old_flags,
         module.create_constant(Type::of<uint>(), &scatter_mask)});
    builder.store(flags, new_flags);
    builder.br(allocation_merge);

    builder.set_insertion_point(skip_append);
    if (set_mask_without_append) {
        auto *old_flags = builder.load(
            Type::of<uint>(),
            route_flags_through_copy ? flag_copy : flags);
        auto *new_flags = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_BIT_OR,
            {old_flags,
             module.create_constant(Type::of<uint>(), &scatter_mask)});
        builder.store(flags, new_flags);
    }
    builder.br(allocation_merge);

    builder.set_insertion_point(allocation_merge);
    if (overwrite_mask_with_unknown) {
        builder.store(flags, unknown_flags);
    }
    auto *current_flags = builder.load(Type::of<uint>(), flags);
    auto *masked = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_BIT_AND,
        {current_flags,
         module.create_constant(Type::of<uint>(), &scatter_mask)});
    auto *has_scatter = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_NOT_EQUAL,
        {masked, module.create_constant_zero(Type::of<uint>())});
    builder.cond_br(has_scatter, pick, done);

    builder.set_insertion_point(pick);
    builder.store(sampled, module.create_constant_zero(Type::of<uint>()));
    builder.br(dynamic_guard);

    builder.set_insertion_point(dynamic_guard);
    auto *current_count = builder.load(Type::of<uint>(), count);
    auto *candidate_index = builder.load(Type::of<uint>(), candidate);
    auto *in_prefix = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_LESS,
        {candidate_index, current_count});
    auto *choose_in_prefix = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_BIT_AND,
        {choose_dynamic, in_prefix});
    builder.cond_br(choose_in_prefix, dynamic_pick, pick_merge);

    builder.set_insertion_point(dynamic_pick);
    auto *selected_candidate = builder.load(Type::of<uint>(), candidate);
    builder.store(sampled, selected_candidate);
    builder.br(pick_merge);

    builder.set_insertion_point(pick_merge);
    auto *selected_index = builder.load(Type::of<uint>(), sampled);
    auto *selected = builder.gep(
        Type::of<uint>(), array, {selected_index});
    static_cast<void>(builder.load(Type::of<uint>(), selected));
    builder.br(done);

    builder.set_insertion_point(done);
    builder.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto before = coro_cfg_distill_pass_run_on_function(kernel);
    expect(before.succeeded());
    expect(frame_contains(before, array));
    auto original_block = array->parent_block();
    auto info = coro_alloca_scope_pass_run_on_function(kernel);
    expect(info.initialized_prefix_proof_count ==
           (expect_contraction ? 1u : 0u));
    expect(array->parent_block() ==
           (expect_contraction ? resume : original_block));
    expect(xir_verify_module(&module).succeeded());
    auto after = coro_cfg_distill_pass_run_on_function(kernel);
    expect(after.succeeded());
    expect(frame_contains(after, array) == !expect_contraction);
}

void check_counted_array_masked_expression_update(
    bool update_may_set_scatter, bool expect_contraction) {
    Module module;
    BasicBlock *entry;
    auto *kernel = make_kernel(module, entry);
    auto *append_closure =
        kernel->create_value_argument(Type::of<bool>());
    auto *select_update =
        kernel->create_value_argument(Type::of<bool>());
    auto *resume = kernel->create_basic_block();
    auto *append = kernel->create_basic_block();
    auto *skip_append = kernel->create_basic_block();
    auto *merge = kernel->create_basic_block();
    auto *pick = kernel->create_basic_block();
    auto *done = kernel->create_basic_block();
    auto *array_type = Type::array(Type::of<uint>(), 4u);
    constexpr uint32_t scatter_mask = 0x14u;
    constexpr uint32_t unrelated_flag = 0x08u;
    XIRBuilder builder;

    builder.set_insertion_point(entry);
    auto *array = builder.alloca_local(array_type);
    array->set_name("masked_expression_array");
    auto *count = builder.alloca_local(Type::of<uint>());
    count->set_name("masked_expression_count");
    auto *flags = builder.alloca_local(Type::of<uint>());
    flags->set_name("masked_expression_flags");
    auto *dynamic_update = builder.alloca_local(Type::of<uint>());
    dynamic_update->set_name("masked_expression_update");
    builder.coro_suspend(30u, "counted-masked-expression", nullptr);

    builder.set_insertion_point(resume);
    builder.coro_resume(30u, nullptr);
    builder.store(count, module.create_constant_zero(Type::of<uint>()));
    builder.store(flags, module.create_constant_zero(Type::of<uint>()));
    builder.cond_br(append_closure, append, skip_append);

    builder.set_insertion_point(append);
    auto *index = builder.load(Type::of<uint>(), count);
    auto *element = builder.gep(Type::of<uint>(), array, {index});
    builder.store(element, module.create_constant_one(Type::of<uint>()));
    auto *old_count = builder.load(Type::of<uint>(), count);
    auto *new_count = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {old_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, new_count);
    auto *old_flags = builder.load(Type::of<uint>(), flags);
    auto *new_flags = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_BIT_OR,
        {old_flags,
         module.create_constant(Type::of<uint>(), &scatter_mask)});
    builder.store(flags, new_flags);
    builder.br(merge);

    builder.set_insertion_point(skip_append);
    builder.br(merge);

    builder.set_insertion_point(merge);
    auto *selected_bit = module.create_constant(
        Type::of<uint>(), update_may_set_scatter ?
                              &scatter_mask :
                              &unrelated_flag);
    auto *selected_update = builder.call(
        Type::of<uint>(), ArithmeticOp::SELECT,
        {module.create_constant_zero(Type::of<uint>()),
         selected_bit, select_update});
    builder.store(dynamic_update, selected_update);
    auto *current_flags = builder.load(Type::of<uint>(), flags);
    auto *current_update = builder.load(Type::of<uint>(), dynamic_update);
    auto *updated_flags = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_BIT_OR,
        {current_flags, current_update});
    builder.store(flags, updated_flags);
    auto *final_flags = builder.load(Type::of<uint>(), flags);
    auto *masked_flags = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_BIT_AND,
        {final_flags,
         module.create_constant(Type::of<uint>(), &scatter_mask)});
    auto *has_scatter = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_NOT_EQUAL,
        {masked_flags, module.create_constant_zero(Type::of<uint>())});
    builder.cond_br(has_scatter, pick, done);

    builder.set_insertion_point(pick);
    auto *selected = builder.gep(
        Type::of<uint>(), array,
        {module.create_constant_zero(Type::of<uint>())});
    static_cast<void>(builder.load(Type::of<uint>(), selected));
    builder.br(done);

    builder.set_insertion_point(done);
    builder.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto before = coro_cfg_distill_pass_run_on_function(kernel);
    expect(before.succeeded());
    expect(frame_contains(before, array));
    auto original_block = array->parent_block();
    auto info = coro_alloca_scope_pass_run_on_function(kernel);
    expect(info.initialized_prefix_proof_count ==
           (expect_contraction ? 1u : 0u));
    expect(array->parent_block() ==
           (expect_contraction ? resume : original_block));
    expect(xir_verify_module(&module).succeeded());
    auto after = coro_cfg_distill_pass_run_on_function(kernel);
    expect(after.succeeded());
    expect(frame_contains(after, array) == !expect_contraction);
}

void check_counted_array_exhausted_capacity_guard(
    bool positive_initial_capacity, bool expect_contraction) {
    Module module;
    BasicBlock *entry;
    auto *kernel = make_kernel(module, entry);
    auto *preallocate =
        kernel->create_value_argument(Type::of<bool>());
    auto *resume = kernel->create_basic_block();
    auto *first_append = kernel->create_basic_block();
    auto *first_skip = kernel->create_basic_block();
    auto *publish = kernel->create_basic_block();
    auto *second_append = kernel->create_basic_block();
    auto *second_skip = kernel->create_basic_block();
    auto *pick_guard = kernel->create_basic_block();
    auto *pick = kernel->create_basic_block();
    auto *done = kernel->create_basic_block();
    auto *array_type = Type::array(Type::of<uint>(), 1u);
    constexpr uint32_t scatter_mask = 0x14u;
    XIRBuilder builder;

    builder.set_insertion_point(entry);
    auto *array = builder.alloca_local(array_type);
    array->set_name("exhausted_capacity_array");
    auto *count = builder.alloca_local(Type::of<uint>());
    count->set_name("exhausted_capacity_count");
    auto *left = builder.alloca_local(Type::of<uint>());
    left->set_name("exhausted_capacity_left");
    auto *flags = builder.alloca_local(Type::of<uint>());
    flags->set_name("exhausted_capacity_flags");
    builder.coro_suspend(31u, "counted-exhausted-capacity", nullptr);

    builder.set_insertion_point(resume);
    builder.coro_resume(31u, nullptr);
    builder.store(count, module.create_constant_zero(Type::of<uint>()));
    const uint32_t initial_capacity = positive_initial_capacity ? 1u : 0u;
    builder.store(
        left,
        module.create_constant(Type::of<uint>(), &initial_capacity));
    builder.store(flags, module.create_constant_zero(Type::of<uint>()));
    auto *initial_left = builder.load(Type::of<uint>(), left);
    auto *initial_left_nonzero = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_NOT_EQUAL,
        {initial_left, module.create_constant_zero(Type::of<uint>())});
    auto *take_first = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_BIT_AND,
        {preallocate, initial_left_nonzero});
    builder.cond_br(take_first, first_append, first_skip);

    builder.set_insertion_point(first_append);
    auto *first_index = builder.load(Type::of<uint>(), count);
    auto *first_element = builder.gep(
        Type::of<uint>(), array, {first_index});
    builder.store(
        first_element, module.create_constant_one(Type::of<uint>()));
    auto *first_count = builder.load(Type::of<uint>(), count);
    auto *next_first_count = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {first_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, next_first_count);
    auto *first_left = builder.load(Type::of<uint>(), left);
    auto *next_first_left = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_SUB,
        {first_left, module.create_constant_one(Type::of<uint>())});
    builder.store(left, next_first_left);
    builder.br(publish);

    builder.set_insertion_point(first_skip);
    builder.br(publish);

    // Cycles' transparent closure publishes its type flag before attempting
    // closure_alloc. If capacity is exhausted, an earlier allocation proves
    // count>0; otherwise the following append succeeds. The renderer must not
    // annotate this lifetime -- the compiler has to prove that disjunction.
    builder.set_insertion_point(publish);
    auto *old_flags = builder.load(Type::of<uint>(), flags);
    auto *new_flags = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_BIT_OR,
        {old_flags,
         module.create_constant(Type::of<uint>(), &scatter_mask)});
    builder.store(flags, new_flags);
    auto *remaining = builder.load(Type::of<uint>(), left);
    auto *has_capacity = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_NOT_EQUAL,
        {remaining, module.create_constant_zero(Type::of<uint>())});
    builder.cond_br(has_capacity, second_append, second_skip);

    builder.set_insertion_point(second_append);
    auto *second_index = builder.load(Type::of<uint>(), count);
    auto *second_element = builder.gep(
        Type::of<uint>(), array, {second_index});
    builder.store(
        second_element, module.create_constant_one(Type::of<uint>()));
    auto *second_count = builder.load(Type::of<uint>(), count);
    auto *next_second_count = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {second_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, next_second_count);
    builder.br(pick_guard);

    builder.set_insertion_point(second_skip);
    builder.br(pick_guard);

    builder.set_insertion_point(pick_guard);
    auto *current_flags = builder.load(Type::of<uint>(), flags);
    auto *masked_flags = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_BIT_AND,
        {current_flags,
         module.create_constant(Type::of<uint>(), &scatter_mask)});
    auto *has_scatter = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_NOT_EQUAL,
        {masked_flags, module.create_constant_zero(Type::of<uint>())});
    builder.cond_br(has_scatter, pick, done);

    builder.set_insertion_point(pick);
    auto *selected = builder.gep(
        Type::of<uint>(), array,
        {module.create_constant_zero(Type::of<uint>())});
    static_cast<void>(builder.load(Type::of<uint>(), selected));
    builder.br(done);

    builder.set_insertion_point(done);
    builder.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto before = coro_cfg_distill_pass_run_on_function(kernel);
    expect(before.succeeded());
    expect(frame_contains(before, array));
    auto original_block = array->parent_block();
    auto info = coro_alloca_scope_pass_run_on_function(kernel);
    expect(info.initialized_prefix_proof_count ==
           (expect_contraction ? 1u : 0u));
    expect(array->parent_block() ==
           (expect_contraction ? resume : original_block));
    expect(xir_verify_module(&module).succeeded());
    auto after = coro_cfg_distill_pass_run_on_function(kernel);
    expect(after.succeeded());
    expect(frame_contains(after, array) == !expect_contraction);
}

void check_counted_array_impossible_masked_edge(
    bool constrain_unbacked_path, bool expect_contraction,
    bool derive_mode_before_suspend = false,
    bool allow_emission_transition = false) {
    Module module;
    BasicBlock *entry;
    auto *kernel = make_kernel(module, entry);
    auto *initial_mode =
        kernel->create_value_argument(Type::of<uint>());
    auto *select_transition =
        kernel->create_value_argument(Type::of<bool>());
    auto *mode_false = derive_mode_before_suspend ?
                           kernel->create_basic_block() :
                           nullptr;
    auto *mode_true = derive_mode_before_suspend ?
                          kernel->create_basic_block() :
                          nullptr;
    auto *mode_merge = derive_mode_before_suspend ?
                           kernel->create_basic_block() :
                           nullptr;
    auto *resume = kernel->create_basic_block();
    auto *mode_update = derive_mode_before_suspend ?
                            kernel->create_basic_block() :
                            nullptr;
    auto *unbacked = kernel->create_basic_block();
    auto *append = kernel->create_basic_block();
    auto *merge = kernel->create_basic_block();
    auto *pick = kernel->create_basic_block();
    auto *done = kernel->create_basic_block();
    auto *array_type = Type::array(Type::of<uint>(), 4u);
    constexpr uint32_t emission_mask = 0x20u;
    constexpr uint32_t scatter_mask = 0x14u;
    XIRBuilder builder;

    builder.set_insertion_point(entry);
    auto *array = builder.alloca_local(array_type);
    array->set_name("impossible_masked_edge_array");
    auto *count = builder.alloca_local(Type::of<uint>());
    count->set_name("impossible_masked_edge_count");
    auto *flags = builder.alloca_local(Type::of<uint>());
    flags->set_name("impossible_masked_edge_flags");
    auto *mode = builder.alloca_local(Type::of<uint>());
    mode->set_name("impossible_masked_edge_mode");
    auto *mode_source = builder.alloca_local(Type::of<uint>());
    mode_source->set_name("impossible_masked_edge_mode_source");
    if (derive_mode_before_suspend) {
        builder.store(
            mode_source,
            module.create_constant_zero(Type::of<uint>()));
        builder.cond_br(select_transition, mode_true, mode_false);

        builder.set_insertion_point(mode_false);
        auto *false_mode = builder.load(Type::of<uint>(), mode_source);
        constexpr uint32_t false_arm_bits = 0x40u;
        auto *false_arm = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_BIT_OR,
            {false_mode,
             module.create_constant(
                 Type::of<uint>(), &false_arm_bits)});
        builder.store(mode_source, false_arm);
        builder.br(mode_merge);

        builder.set_insertion_point(mode_true);
        auto *true_mode = builder.load(Type::of<uint>(), mode_source);
        const auto true_arm_bits = allow_emission_transition ?
                                       emission_mask :
                                       uint32_t{0x80u};
        auto *true_arm = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_BIT_OR,
            {true_mode,
             module.create_constant(
                 Type::of<uint>(), &true_arm_bits)});
        builder.store(mode_source, true_arm);
        builder.br(mode_merge);

        builder.set_insertion_point(mode_merge);
        builder.store(
            mode, builder.load(Type::of<uint>(), mode_source));
        // The copied value is durable even though its source is overwritten
        // before the suspend. This prevents exact load forwarding from
        // collapsing the test back to one alloca and exercises sparse
        // cross-slot known-zero propagation.
        builder.store(mode_source, initial_mode);
    }
    builder.coro_suspend(29u, "counted-impossible-masked-edge", nullptr);

    builder.set_insertion_point(resume);
    builder.coro_resume(29u, nullptr);
    if (!derive_mode_before_suspend) {
        builder.store(mode, initial_mode);
    }
    builder.store(count, module.create_constant_zero(Type::of<uint>()));
    builder.store(flags, module.create_constant_zero(Type::of<uint>()));
    if (derive_mode_before_suspend) {
        constexpr uint32_t unused_suffix_index = 3u;
        auto *unused_suffix = builder.gep(
            Type::of<uint>(), array,
            {module.create_constant(Type::of<uint>(),
                                    &unused_suffix_index)});
        builder.store(
            unused_suffix,
            module.create_constant_zero(Type::of<uint>()));
        builder.br(mode_update);
        builder.set_insertion_point(mode_update);
        auto *current_mode = builder.load(Type::of<uint>(), mode);
        constexpr uint32_t resumed_false_bits = 0x100u;
        auto *false_arm = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_BIT_OR,
            {current_mode,
             module.create_constant(
                 Type::of<uint>(), &resumed_false_bits)});
        const auto resumed_true_bits = allow_emission_transition ?
                                           emission_mask :
                                           uint32_t{0x200u};
        auto *true_arm = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_BIT_OR,
            {current_mode,
             module.create_constant(
                 Type::of<uint>(), &resumed_true_bits)});
        // The prefix relation domain deliberately does not interpret
        // arbitrary SELECT expressions. Edge reachability must instead use
        // the independent whole-CFG known-bits theorem at the later guard.
        auto *selected_mode = builder.call(
            Type::of<uint>(), ArithmeticOp::SELECT,
            {false_arm, true_arm, select_transition});
        builder.store(mode, selected_mode);
    }
    if (constrain_unbacked_path) {
        auto *current_mode = builder.load(Type::of<uint>(), mode);
        auto *masked_mode = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_BIT_AND,
            {current_mode,
             module.create_constant(Type::of<uint>(), &emission_mask)});
        auto *mode_is_zero = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
            {masked_mode,
             module.create_constant_zero(Type::of<uint>())});
        builder.assume_(mode_is_zero, "emission path is unreachable");
    }
    auto *current_mode = builder.load(Type::of<uint>(), mode);
    auto *masked_mode = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_BIT_AND,
        {current_mode,
         module.create_constant(Type::of<uint>(), &emission_mask)});
    auto *take_unbacked = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_NOT_EQUAL,
        {masked_mode, module.create_constant_zero(Type::of<uint>())});
    builder.cond_br(take_unbacked, unbacked, append);

    // This path deliberately creates the same invalid implication as an
    // emission-only closure setup: it publishes a scatter flag without
    // adding an initialized closure. It is safe only when the incoming
    // masked contract proves the edge unreachable.
    builder.set_insertion_point(unbacked);
    auto *unbacked_flags = builder.load(Type::of<uint>(), flags);
    auto *unbacked_scatter = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_BIT_OR,
        {unbacked_flags,
         module.create_constant(Type::of<uint>(), &scatter_mask)});
    builder.store(flags, unbacked_scatter);
    builder.br(merge);

    builder.set_insertion_point(append);
    auto *index = builder.load(Type::of<uint>(), count);
    auto *element = builder.gep(Type::of<uint>(), array, {index});
    builder.store(element, module.create_constant_one(Type::of<uint>()));
    auto *old_count = builder.load(Type::of<uint>(), count);
    auto *new_count = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {old_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, new_count);
    auto *append_flags = builder.load(Type::of<uint>(), flags);
    auto *append_scatter = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_BIT_OR,
        {append_flags,
         module.create_constant(Type::of<uint>(), &scatter_mask)});
    builder.store(flags, append_scatter);
    builder.br(merge);

    builder.set_insertion_point(merge);
    auto *current_flags = builder.load(Type::of<uint>(), flags);
    auto *masked_flags = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_BIT_AND,
        {current_flags,
         module.create_constant(Type::of<uint>(), &scatter_mask)});
    auto *has_scatter = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_NOT_EQUAL,
        {masked_flags, module.create_constant_zero(Type::of<uint>())});
    builder.cond_br(has_scatter, pick, done);

    builder.set_insertion_point(pick);
    auto *selected = builder.gep(
        Type::of<uint>(), array,
        {module.create_constant_zero(Type::of<uint>())});
    static_cast<void>(builder.load(Type::of<uint>(), selected));
    builder.br(done);

    builder.set_insertion_point(done);
    builder.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto before = coro_cfg_distill_pass_run_on_function(kernel);
    expect(before.succeeded());
    expect(frame_contains(before, array));
    auto original_block = array->parent_block();
    auto info = coro_alloca_scope_pass_run_on_function(kernel);
    expect(info.initialized_prefix_proof_count ==
           (expect_contraction ? 1u : 0u));
    expect(array->parent_block() ==
           (expect_contraction ? resume : original_block));
    expect(xir_verify_module(&module).succeeded());
    auto after = coro_cfg_distill_pass_run_on_function(kernel);
    expect(after.succeeded());
    expect(frame_contains(after, array) == !expect_contraction);
}

void check_counted_array_rollback_tail(
    uint32_t ticket_offset, bool include_ticket_identity,
    bool ticket_from_append, bool join_nonrollback,
    bool expect_contraction) {
    Module module;
    BasicBlock *entry;
    auto *kernel = make_kernel(module, entry);
    auto *prepend_element =
        kernel->create_value_argument(Type::of<bool>());
    auto *dynamic_ticket =
        kernel->create_value_argument(Type::of<uint>());
    auto *request_rollback =
        kernel->create_value_argument(Type::of<bool>());
    auto *resume = kernel->create_basic_block();
    auto *prepend = kernel->create_basic_block();
    auto *skip_prepend = kernel->create_basic_block();
    auto *append = kernel->create_basic_block();
    auto *rollback = kernel->create_basic_block();
    auto *consume = kernel->create_basic_block();
    auto *done = kernel->create_basic_block();
    auto *array_type = Type::array(Type::of<uint>(), 4u);
    XIRBuilder builder;

    builder.set_insertion_point(entry);
    auto *array = builder.alloca_local(array_type);
    array->set_name("rollback_tail_array");
    auto *count = builder.alloca_local(Type::of<uint>());
    count->set_name("rollback_tail_count");
    auto *ticket = builder.alloca_local(Type::of<uint>());
    ticket->set_name("rollback_tail_ticket");
    builder.coro_suspend(27u, "counted-rollback-tail", nullptr);

    builder.set_insertion_point(resume);
    builder.coro_resume(27u, nullptr);
    builder.store(count, module.create_constant_zero(Type::of<uint>()));
    builder.cond_br(prepend_element, prepend, skip_prepend);

    const auto append_one = [&](BasicBlock *block,
                                BasicBlock *next) noexcept {
        builder.set_insertion_point(block);
        auto *index = builder.load(Type::of<uint>(), count);
        auto *element = builder.gep(Type::of<uint>(), array, {index});
        builder.store(element, module.create_constant_one(Type::of<uint>()));
        auto *old_count = builder.load(Type::of<uint>(), count);
        auto *new_count = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {old_count, module.create_constant_one(Type::of<uint>())});
        builder.store(count, new_count);
        builder.br(next);
    };
    append_one(prepend, append);

    builder.set_insertion_point(skip_prepend);
    builder.br(append);

    // The second append guarantees C>0 while retaining a dynamic C in {1,2}.
    // The ticket is unrelated until the branch establishes I+1=C.
    builder.set_insertion_point(append);
    auto *append_index = builder.load(Type::of<uint>(), count);
    auto *append_element = builder.gep(
        Type::of<uint>(), array, {append_index});
    builder.store(
        append_element, module.create_constant_one(Type::of<uint>()));
    auto *old_count = builder.load(Type::of<uint>(), count);
    auto *new_count = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {old_count, module.create_constant_one(Type::of<uint>())});
    auto *ticket_value = ticket_from_append ?
                             static_cast<Value *>(append_index) :
                             static_cast<Value *>(dynamic_ticket);
    builder.store(ticket, ticket_value);
    builder.store(count, new_count);
    auto *current_count = builder.load(Type::of<uint>(), count);
    auto *positive = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_GREATER,
        {current_count, module.create_constant_zero(Type::of<uint>())});
    Value *rollback_guard = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_BIT_AND,
        {request_rollback, positive});
    if (include_ticket_identity) {
        auto *current_ticket = builder.load(Type::of<uint>(), ticket);
        auto *offset_ticket = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {current_ticket,
             module.create_constant(Type::of<uint>(), &ticket_offset)});
        auto *is_last = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
            {offset_ticket, current_count});
        rollback_guard = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_BIT_AND,
            {rollback_guard, is_last});
    }
    builder.cond_br(
        rollback_guard, rollback,
        join_nonrollback ? consume : done);

    builder.set_insertion_point(rollback);
    auto *count_before_rollback = builder.load(Type::of<uint>(), count);
    auto *rolled_back_count = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_SUB,
        {count_before_rollback,
         module.create_constant_one(Type::of<uint>())});
    builder.store(count, rolled_back_count);
    builder.br(consume);

    builder.set_insertion_point(consume);
    auto *read_index = builder.load(Type::of<uint>(), ticket);
    auto *selected = builder.gep(Type::of<uint>(), array, {read_index});
    static_cast<void>(builder.load(Type::of<uint>(), selected));
    builder.br(done);

    builder.set_insertion_point(done);
    builder.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto before = coro_cfg_distill_pass_run_on_function(kernel);
    expect(before.succeeded());
    expect(frame_contains(before, array));
    auto original_block = array->parent_block();
    auto info = coro_alloca_scope_pass_run_on_function(kernel);
    expect(info.initialized_prefix_proof_count ==
           (expect_contraction ? 1u : 0u));
    expect(array->parent_block() ==
           (expect_contraction ? resume : original_block));
    expect(xir_verify_module(&module).succeeded());
    auto after = coro_cfg_distill_pass_run_on_function(kernel);
    expect(after.succeeded());
    expect(frame_contains(after, array) == !expect_contraction);
}

}// namespace

void register_coro_alloca_scope_tests() {

    "continuation_loop_begins_fresh_dynamic_scratch_lifetime"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *index = kernel->create_value_argument(Type::of<uint>());
        auto *suspend = kernel->create_basic_block();
        auto *resume = kernel->create_basic_block();
        auto *array_type = Type::array(Type::of<uint>(), 4u);
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(array_type);
        scratch->set_name("phase_scratch");
        builder.br(suspend);

        builder.set_insertion_point(suspend);
        builder.coro_suspend(7u, "phase", nullptr);

        builder.set_insertion_point(resume);
        auto *resume_inst = builder.coro_resume(7u, nullptr);
        auto *element = builder.gep(Type::of<uint>(), scratch, {index});
        builder.store(element, module.create_constant_one(Type::of<uint>()));
        static_cast<void>(builder.load(Type::of<uint>(), element));
        builder.br(suspend);

        expect(xir_verify_module(&module).succeeded());
        auto before = coro_cfg_distill_pass_run_on_function(kernel);
        expect(before.succeeded());
        expect(frame_contains(before, scratch));

        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.invalid_semantic_cfg_count == 0u);
        expect(info.scanned_local_alloca_count == 1u);
        expect(info.contracted_alloca_count == 1u);
        expect(info.cross_block_contraction_count == 1u);
        expect(info.intra_block_contraction_count == 0u);
        expect(info.definite_initialization_proof_count == 1u);
        expect(info.rejected_prior_lifetime_observation_count == 0u);
        expect(scratch->parent_block() == resume);
        expect(instruction_index(resume, resume_inst) <
               instruction_index(resume, scratch));
        expect(instruction_index(resume, scratch) <
               instruction_index(resume, element));
        expect(xir_verify_module(&module).succeeded());

        auto after = coro_cfg_distill_pass_run_on_function(kernel);
        expect(after.succeeded());
        expect(!frame_contains(after, scratch));
    };

    "sibling_uses_contract_to_nearest_common_dominator"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *left = kernel->create_basic_block();
        auto *right = kernel->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        auto *unrelated = builder.clock();
        auto *branch = builder.cond_br(condition, left, right);
        builder.set_insertion_point(left);
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.return_void();
        builder.set_insertion_point(right);
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.contracted_alloca_count == 1u);
        expect(info.intra_block_contraction_count == 1u);
        expect(info.cross_block_contraction_count == 0u);
        expect(scratch->parent_block() == entry);
        expect(instruction_index(entry, unrelated) <
               instruction_index(entry, scratch));
        expect(instruction_index(entry, scratch) <
               instruction_index(entry, branch));
        expect(xir_verify_module(&module).succeeded());
    };

    "loop_carried_state_prevents_lifetime_contraction"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *header = kernel->create_basic_block();
        auto *initialize = kernel->create_basic_block();
        auto *consume = kernel->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *state = builder.alloca_local(Type::of<uint>());
        builder.br(header);

        builder.set_insertion_point(header);
        auto *first_iteration = builder.phi(
            Type::of<bool>(),
            {{module.create_constant_one(Type::of<bool>()), entry},
             {module.create_constant_zero(Type::of<bool>()), consume}});
        builder.cond_br(first_iteration, initialize, consume);

        builder.set_insertion_point(initialize);
        builder.store(
            state, module.create_constant_one(Type::of<uint>()));
        builder.br(consume);

        builder.set_insertion_point(consume);
        static_cast<void>(builder.load(Type::of<uint>(), state));
        builder.br(header);

        expect(xir_verify_module(&module).succeeded());
        auto original_index = instruction_index(entry, state);
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.definite_initialization_proof_count == 0u);
        expect(info.rejected_prior_lifetime_observation_count == 1u);
        expect(info.contracted_alloca_count == 0u);
        expect(state->parent_block() == entry);
        expect(instruction_index(entry, state) == original_index);
        expect(xir_verify_module(&module).succeeded());
    };

    "all_paths_initialize_before_observation"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *branch = kernel->create_basic_block();
        auto *left = kernel->create_basic_block();
        auto *right = kernel->create_basic_block();
        auto *join = kernel->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        builder.br(branch);
        builder.set_insertion_point(branch);
        auto *branch_term = builder.cond_br(condition, left, right);
        builder.set_insertion_point(left);
        builder.store(
            scratch, module.create_constant_one(Type::of<uint>()));
        builder.br(join);
        builder.set_insertion_point(right);
        builder.store(
            scratch, module.create_constant_zero(Type::of<uint>()));
        builder.br(join);
        builder.set_insertion_point(join);
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.definite_initialization_proof_count == 1u);
        expect(info.rejected_prior_lifetime_observation_count == 0u);
        expect(info.contracted_alloca_count == 1u);
        expect(info.cross_block_contraction_count == 1u);
        expect(scratch->parent_block() == branch);
        expect(instruction_index(branch, scratch) <
               instruction_index(branch, branch_term));
        expect(xir_verify_module(&module).succeeded());
    };

    "disjoint_subaggregate_stores_jointly_initialize_parent"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *phase = kernel->create_basic_block();
        auto *array_type = Type::array(Type::of<uint>(), 2u);
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(array_type);
        builder.br(phase);
        builder.set_insertion_point(phase);
        auto *element_0 = builder.gep(
            Type::of<uint>(), scratch,
            {module.create_constant_zero(Type::of<uint>())});
        builder.store(
            element_0, module.create_constant_zero(Type::of<uint>()));
        auto *element_1 = builder.gep(
            Type::of<uint>(), scratch,
            {module.create_constant_one(Type::of<uint>())});
        builder.store(
            element_1, module.create_constant_one(Type::of<uint>()));
        static_cast<void>(builder.load(array_type, scratch));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.definite_initialization_proof_count == 1u);
        expect(info.rejected_prior_lifetime_observation_count == 0u);
        expect(info.contracted_alloca_count == 1u);
        expect(scratch->parent_block() == phase);
        expect(xir_verify_module(&module).succeeded());
    };

    "partial_parent_initialization_prevents_contraction"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *phase = kernel->create_basic_block();
        auto *array_type = Type::array(Type::of<uint>(), 2u);
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(array_type);
        builder.br(phase);
        builder.set_insertion_point(phase);
        auto *element_0 = builder.gep(
            Type::of<uint>(), scratch,
            {module.create_constant_zero(Type::of<uint>())});
        builder.store(
            element_0, module.create_constant_zero(Type::of<uint>()));
        static_cast<void>(builder.load(array_type, scratch));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.definite_initialization_proof_count == 0u);
        expect(info.rejected_prior_lifetime_observation_count == 1u);
        expect(info.contracted_alloca_count == 0u);
        expect(scratch->parent_block() == entry);
        expect(xir_verify_module(&module).succeeded());
    };

    "write_only_reference_call_starts_a_fresh_lifetime"_test = [] {
        Module module;
        auto *writer = module.create_callable(nullptr);
        auto *output = writer->create_reference_argument(Type::of<uint>());
        XIRBuilder builder;
        builder.set_insertion_point(writer->create_body_block());
        builder.store(output, module.create_constant_one(Type::of<uint>()));
        builder.return_void();

        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *phase = kernel->create_basic_block();
        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        builder.br(phase);
        builder.set_insertion_point(phase);
        auto *call = builder.call(nullptr, writer, {scratch});
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.definite_initialization_proof_count == 1u);
        expect(info.rejected_prior_lifetime_observation_count == 0u);
        expect(info.contracted_alloca_count == 1u);
        expect(scratch->parent_block() == phase);
        expect(instruction_index(phase, scratch) <
               instruction_index(phase, call));
        expect(xir_verify_module(&module).succeeded());
    };

    "reference_call_read_before_write_preserves_prior_lifetime"_test = [] {
        Module module;
        auto *reader_writer = module.create_callable(nullptr);
        auto *argument = reader_writer->create_reference_argument(
            Type::of<uint>());
        XIRBuilder builder;
        builder.set_insertion_point(reader_writer->create_body_block());
        static_cast<void>(builder.load(Type::of<uint>(), argument));
        builder.store(argument, module.create_constant_one(Type::of<uint>()));
        builder.return_void();

        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *phase = kernel->create_basic_block();
        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        builder.br(phase);
        builder.set_insertion_point(phase);
        builder.call(nullptr, reader_writer, {scratch});
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.definite_initialization_proof_count == 0u);
        expect(info.rejected_prior_lifetime_observation_count == 1u);
        expect(info.contracted_alloca_count == 0u);
        expect(scratch->parent_block() == entry);
        expect(xir_verify_module(&module).succeeded());
    };

    "conditional_reference_write_is_not_a_must_definition"_test = [] {
        Module module;
        auto *conditional_writer = module.create_callable(nullptr);
        auto *output = conditional_writer->create_reference_argument(
            Type::of<uint>());
        auto *condition = conditional_writer->create_value_argument(
            Type::of<bool>());
        XIRBuilder builder;
        builder.set_insertion_point(conditional_writer->create_body_block());
        auto *selection = builder.if_(condition);
        auto *merge = selection->create_merge_block();
        builder.set_insertion_point(selection->create_true_block());
        builder.store(output, module.create_constant_one(Type::of<uint>()));
        builder.br(merge);
        builder.set_insertion_point(selection->create_false_block());
        builder.br(merge);
        builder.set_insertion_point(merge);
        builder.return_void();

        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *phase = kernel->create_basic_block();
        auto *caller_condition = kernel->create_value_argument(
            Type::of<bool>());
        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        builder.br(phase);
        builder.set_insertion_point(phase);
        builder.call(nullptr, conditional_writer,
                     {scratch, caller_condition});
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.definite_initialization_proof_count == 0u);
        expect(info.rejected_prior_lifetime_observation_count == 1u);
        expect(info.contracted_alloca_count == 0u);
        expect(scratch->parent_block() == entry);
        expect(xir_verify_module(&module).succeeded());
    };

    "aliased_reference_formals_observe_before_any_must_write"_test = [] {
        Module module;
        auto *callee = module.create_callable(nullptr);
        // Keep the writer first to ensure signature order cannot make its Must
        // definition mask the later formal's old-value observation.
        auto *writer = callee->create_reference_argument(Type::of<uint>());
        auto *reader = callee->create_reference_argument(Type::of<uint>());
        XIRBuilder builder;
        builder.set_insertion_point(callee->create_body_block());
        static_cast<void>(builder.load(Type::of<uint>(), reader));
        builder.store(writer, module.create_constant_one(Type::of<uint>()));
        builder.return_void();

        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *phase = kernel->create_basic_block();
        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        builder.br(phase);
        builder.set_insertion_point(phase);
        builder.call(nullptr, callee, {scratch, scratch});
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.definite_initialization_proof_count == 0u);
        expect(info.rejected_prior_lifetime_observation_count == 1u);
        expect(info.contracted_alloca_count == 0u);
        expect(scratch->parent_block() == entry);
        expect(xir_verify_module(&module).succeeded());
    };

    "ray_query_write_only_capture_starts_a_fresh_lifetime"_test = [] {
        Module module;
        auto *query_type = Type::custom("LC_RayQueryAll");
        auto *surface = make_ray_query_capture_handler(
            module, query_type, false, true);
        auto *procedural = make_ray_query_capture_handler(
            module, query_type, false, false);

        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *query = kernel->create_reference_argument(query_type);
        auto *phase = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        builder.br(phase);
        builder.set_insertion_point(phase);
        std::array<Value *, 1u> captures{scratch};
        auto *pipeline = builder.ray_query_pipeline(
            query, surface, procedural,
            luisa::span<Value *const>{captures});
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.definite_initialization_proof_count == 1u);
        expect(info.rejected_prior_lifetime_observation_count == 0u);
        expect(info.contracted_alloca_count == 1u);
        expect(scratch->parent_block() == phase);
        expect(instruction_index(phase, scratch) <
               instruction_index(phase, pipeline));
        expect(xir_verify_module(&module).succeeded());
    };

    "ray_query_callback_read_preserves_prior_lifetime"_test = [] {
        Module module;
        auto *query_type = Type::custom("LC_RayQueryAll");
        auto *surface = make_ray_query_capture_handler(
            module, query_type, true, false);
        auto *procedural = make_ray_query_capture_handler(
            module, query_type, false, false);

        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *query = kernel->create_reference_argument(query_type);
        auto *phase = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        builder.br(phase);
        builder.set_insertion_point(phase);
        std::array<Value *, 1u> captures{scratch};
        builder.ray_query_pipeline(
            query, surface, procedural,
            luisa::span<Value *const>{captures});
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.definite_initialization_proof_count == 0u);
        expect(info.rejected_prior_lifetime_observation_count == 1u);
        expect(info.contracted_alloca_count == 0u);
        expect(scratch->parent_block() == entry);
        expect(xir_verify_module(&module).succeeded());
    };

    "ray_query_callback_write_is_not_a_pipeline_must_definition"_test = [] {
        Module module;
        auto *query_type = Type::custom("LC_RayQueryAll");
        auto *surface = make_ray_query_capture_handler(
            module, query_type, false, true);
        auto *procedural = make_ray_query_capture_handler(
            module, query_type, false, true);

        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *query = kernel->create_reference_argument(query_type);
        auto *phase = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        builder.br(phase);
        builder.set_insertion_point(phase);
        std::array<Value *, 1u> captures{scratch};
        builder.ray_query_pipeline(
            query, surface, procedural,
            luisa::span<Value *const>{captures});
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.definite_initialization_proof_count == 0u);
        expect(info.rejected_prior_lifetime_observation_count == 1u);
        expect(info.contracted_alloca_count == 0u);
        expect(scratch->parent_block() == entry);
        expect(xir_verify_module(&module).succeeded());
    };

    "single_full_definition_moves_with_lifetime_start"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *phase = kernel->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        auto *definition = builder.store(
            scratch, module.create_constant_one(Type::of<uint>()));
        auto *unrelated = builder.clock();
        builder.br(phase);
        builder.set_insertion_point(phase);
        auto *observation = builder.load(Type::of<uint>(), scratch);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.delayed_first_definition_count == 1u);
        expect(info.cross_block_first_definition_delay_count == 1u);
        expect(info.intra_block_first_definition_delay_count == 0u);
        expect(info.contracted_alloca_count == 1u);
        expect(scratch->parent_block() == phase);
        expect(definition->parent_block() == phase);
        expect(unrelated->parent_block() == entry);
        expect(instruction_index(phase, scratch) <
               instruction_index(phase, definition));
        expect(instruction_index(phase, definition) <
               instruction_index(phase, observation));
        expect(xir_verify_module(&module).succeeded());
    };

    "undefined_full_definition_moves_with_fresh_lifetime"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *phase = kernel->create_basic_block();
        auto *array_type = Type::array(Type::of<float4>(), 3u);
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(array_type);
        auto *seed = module.create_undefined(array_type);
        auto *definition = builder.store(scratch, seed);
        auto *unrelated = builder.clock();
        builder.br(phase);
        builder.set_insertion_point(phase);
        auto *observation = builder.load(array_type, scratch);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.delayed_first_definition_count == 1u);
        expect(info.cross_block_first_definition_delay_count == 1u);
        expect(info.contracted_alloca_count == 1u);
        expect(scratch->parent_block() == phase);
        expect(definition->parent_block() == phase);
        expect(definition->value() == seed);
        expect(unrelated->parent_block() == entry);
        expect(instruction_index(phase, scratch) <
               instruction_index(phase, definition));
        expect(instruction_index(phase, definition) <
               instruction_index(phase, observation));
        expect(xir_verify_module(&module).succeeded());
    };

    "single_definition_can_initialize_each_new_loop_lifetime"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *repeat = kernel->create_value_argument(Type::of<bool>());
        auto *header = kernel->create_basic_block();
        auto *done = kernel->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        auto *definition = builder.store(
            scratch, module.create_constant_one(Type::of<uint>()));
        builder.br(header);
        builder.set_insertion_point(header);
        auto *unrelated = builder.clock();
        auto *observation = builder.load(Type::of<uint>(), scratch);
        builder.cond_br(repeat, header, done);
        builder.set_insertion_point(done);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.delayed_first_definition_count == 1u);
        expect(info.cross_block_first_definition_delay_count == 1u);
        expect(scratch->parent_block() == header);
        expect(definition->parent_block() == header);
        expect(instruction_index(header, unrelated) <
               instruction_index(header, scratch));
        expect(instruction_index(header, scratch) <
               instruction_index(header, definition));
        expect(instruction_index(header, definition) <
               instruction_index(header, observation));
        expect(xir_verify_module(&module).succeeded());
    };

    "multiple_definitions_are_not_first_definition_delayed"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *phase = kernel->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        auto *unrelated = builder.clock();
        auto *first_definition = builder.store(
            scratch, module.create_constant_zero(Type::of<uint>()));
        builder.br(phase);
        builder.set_insertion_point(phase);
        auto *second_definition = builder.store(
            scratch, module.create_constant_one(Type::of<uint>()));
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.delayed_first_definition_count == 0u);
        expect(first_definition->parent_block() == entry);
        expect(second_definition->parent_block() == phase);
        expect(scratch->parent_block() == entry);
        expect(instruction_index(entry, unrelated) <
               instruction_index(entry, scratch));
        expect(instruction_index(entry, scratch) <
               instruction_index(entry, first_definition));
        expect(xir_verify_module(&module).succeeded());
    };

    "correlated_runtime_predicate_proves_guarded_initialization"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *start = kernel->create_basic_block();
        auto *initialize = kernel->create_basic_block();
        auto *skip_initialize = kernel->create_basic_block();
        auto *retest = kernel->create_basic_block();
        auto *observe = kernel->create_basic_block();
        auto *done = kernel->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        builder.br(start);
        builder.set_insertion_point(start);
        auto *start_branch = builder.cond_br(
            condition, initialize, skip_initialize);
        builder.set_insertion_point(initialize);
        builder.store(
            scratch, module.create_constant_one(Type::of<uint>()));
        builder.br(retest);
        builder.set_insertion_point(skip_initialize);
        builder.br(retest);
        builder.set_insertion_point(retest);
        builder.cond_br(condition, observe, done);
        builder.set_insertion_point(observe);
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.br(done);
        builder.set_insertion_point(done);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.definite_initialization_proof_count == 1u);
        expect(info.guarded_initialization_proof_count == 1u);
        expect(info.rejected_prior_lifetime_observation_count == 0u);
        expect(info.contracted_alloca_count == 1u);
        expect(scratch->parent_block() == start);
        expect(instruction_index(start, scratch) <
               instruction_index(start, start_branch));
        expect(xir_verify_module(&module).succeeded());
    };

    "structurally_numbered_predicates_prove_the_same_guard"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *value = kernel->create_value_argument(Type::of<uint>());
        auto *limit = kernel->create_value_argument(Type::of<uint>());
        auto *start = kernel->create_basic_block();
        auto *initialize = kernel->create_basic_block();
        auto *skip_initialize = kernel->create_basic_block();
        auto *retest = kernel->create_basic_block();
        auto *observe = kernel->create_basic_block();
        auto *done = kernel->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        builder.br(start);
        builder.set_insertion_point(start);
        auto *first_test = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS,
            {value, limit});
        builder.cond_br(first_test, initialize, skip_initialize);
        builder.set_insertion_point(initialize);
        builder.store(
            scratch, module.create_constant_one(Type::of<uint>()));
        builder.br(retest);
        builder.set_insertion_point(skip_initialize);
        builder.br(retest);
        builder.set_insertion_point(retest);
        auto *second_test = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS,
            {value, limit});
        builder.cond_br(second_test, observe, done);
        builder.set_insertion_point(observe);
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.br(done);
        builder.set_insertion_point(done);
        builder.return_void();

        expect(first_test != second_test);
        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.guarded_initialization_proof_count == 1u);
        expect(info.contracted_alloca_count == 1u);
        expect(scratch->parent_block() == start);
        expect(xir_verify_module(&module).succeeded());
    };

    "unrelated_runtime_predicate_cannot_prove_initialization"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *initialize_condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *observe_condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *start = kernel->create_basic_block();
        auto *initialize = kernel->create_basic_block();
        auto *skip_initialize = kernel->create_basic_block();
        auto *retest = kernel->create_basic_block();
        auto *observe = kernel->create_basic_block();
        auto *done = kernel->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        builder.br(start);
        builder.set_insertion_point(start);
        builder.cond_br(
            initialize_condition, initialize, skip_initialize);
        builder.set_insertion_point(initialize);
        builder.store(
            scratch, module.create_constant_one(Type::of<uint>()));
        builder.br(retest);
        builder.set_insertion_point(skip_initialize);
        builder.br(retest);
        builder.set_insertion_point(retest);
        builder.cond_br(observe_condition, observe, done);
        builder.set_insertion_point(observe);
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.br(done);
        builder.set_insertion_point(done);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.definite_initialization_proof_count == 0u);
        expect(info.guarded_initialization_proof_count == 0u);
        expect(info.rejected_prior_lifetime_observation_count == 1u);
        expect(info.contracted_alloca_count == 0u);
        expect(scratch->parent_block() == entry);
        expect(xir_verify_module(&module).succeeded());
    };

    "opposite_runtime_predicate_cannot_prove_initialization"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *start = kernel->create_basic_block();
        auto *initialize = kernel->create_basic_block();
        auto *skip_initialize = kernel->create_basic_block();
        auto *retest = kernel->create_basic_block();
        auto *observe = kernel->create_basic_block();
        auto *done = kernel->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        builder.br(start);
        builder.set_insertion_point(start);
        builder.cond_br(condition, initialize, skip_initialize);
        builder.set_insertion_point(initialize);
        builder.store(
            scratch, module.create_constant_one(Type::of<uint>()));
        builder.br(retest);
        builder.set_insertion_point(skip_initialize);
        builder.br(retest);
        builder.set_insertion_point(retest);
        builder.cond_br(condition, done, observe);
        builder.set_insertion_point(observe);
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.br(done);
        builder.set_insertion_point(done);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.definite_initialization_proof_count == 0u);
        expect(info.guarded_initialization_proof_count == 0u);
        expect(info.rejected_prior_lifetime_observation_count == 1u);
        expect(info.contracted_alloca_count == 0u);
        expect(scratch->parent_block() == entry);
        expect(xir_verify_module(&module).succeeded());
    };

    "guarded_proof_stops_at_monotone_failing_read"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *phase = kernel->create_basic_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        builder.br(phase);
        builder.set_insertion_point(phase);
        // This observation already disproves a fresh lifetime. Keep a large
        // predicate-rich suffix in the reverse use slice to ensure the
        // guarded solver does not explore it before reporting the monotone
        // failure.
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        auto *cursor = phase;
        for (auto i = 0u; i < 128u; ++i) {
            auto *left = kernel->create_basic_block();
            auto *right = kernel->create_basic_block();
            auto *join = kernel->create_basic_block();
            builder.set_insertion_point(cursor);
            builder.cond_br(condition, left, right);
            builder.set_insertion_point(left);
            builder.br(join);
            builder.set_insertion_point(right);
            builder.br(join);
            cursor = join;
        }
        builder.set_insertion_point(cursor);
        builder.store(
            scratch, module.create_constant_one(Type::of<uint>()));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.definite_initialization_proof_count == 0u);
        expect(info.guarded_initialization_proof_count == 0u);
        expect(info.rejected_prior_lifetime_observation_count == 1u);
        expect(info.guarded_initialization_state_evaluation_count == 1u);
        expect(info.contracted_alloca_count == 0u);
        expect(scratch->parent_block() == entry);
        expect(xir_verify_module(&module).succeeded());
    };

    "counted_array_prefix_contracts_without_initializing_unused_suffix"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *append_condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *read_index =
            kernel->create_value_argument(Type::of<uint>());
        auto *resume = kernel->create_basic_block();
        auto *append = kernel->create_basic_block();
        auto *skip = kernel->create_basic_block();
        auto *consume = kernel->create_basic_block();
        auto *array_type = Type::array(Type::of<uint>(), 4u);
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *array = builder.alloca_local(array_type);
        auto *count = builder.alloca_local(Type::of<uint>());
        builder.coro_suspend(7u, "counted", nullptr);

        builder.set_insertion_point(resume);
        builder.coro_resume(7u, nullptr);
        builder.store(
            count, module.create_constant_zero(Type::of<uint>()));
        auto *sentinel = builder.gep(
            Type::of<uint>(), array,
            {module.create_constant_zero(Type::of<uint>())});
        builder.store(
            sentinel, module.create_constant_zero(Type::of<uint>()));
        builder.cond_br(append_condition, append, skip);

        builder.set_insertion_point(append);
        auto *append_index = builder.load(Type::of<uint>(), count);
        auto *element = builder.gep(
            Type::of<uint>(), array, {append_index});
        builder.store(
            element, module.create_constant_one(Type::of<uint>()));
        auto *old_count = builder.load(Type::of<uint>(), count);
        auto *next_count = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {old_count, module.create_constant_one(Type::of<uint>())});
        builder.store(count, next_count);
        builder.br(consume);

        builder.set_insertion_point(skip);
        builder.br(consume);

        builder.set_insertion_point(consume);
        auto *current_count = builder.load(Type::of<uint>(), count);
        auto *in_prefix = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS,
            {read_index, current_count});
        auto *bounded_index = builder.call(
            Type::of<uint>(), ArithmeticOp::SELECT,
            {module.create_constant_zero(Type::of<uint>()),
             read_index, in_prefix});
        auto *selected = builder.gep(
            Type::of<uint>(), array, {bounded_index});
        static_cast<void>(builder.load(Type::of<uint>(), selected));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto before = coro_cfg_distill_pass_run_on_function(kernel);
        expect(before.succeeded());
        expect(frame_contains(before, array));

        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.initialized_prefix_proof_count == 1u);
        expect(info.rejected_prior_lifetime_observation_count == 0u);
        expect(array->parent_block() == resume);
        expect(xir_verify_module(&module).succeeded());

        auto after = coro_cfg_distill_pass_run_on_function(kernel);
        expect(after.succeeded());
        expect(!frame_contains(after, array));
    };

    "counted_array_capacity_guard_preserves_prefix_in_loop"_test = [] {
        check_counted_array_capacity_guard(2u, true, true);
    };

    "counted_array_control_guard_proves_dynamic_read"_test = [] {
        check_counted_array_control_guard(false, true);
    };

    "counted_array_control_guard_is_killed_by_index_store"_test = [] {
        check_counted_array_control_guard(true, false);
    };

    "counted_array_uses_outer_fresh_lifetime_before_inner_loop"_test = [] {
        check_counted_array_outer_lifetime_placement();
    };

    "counted_array_saved_preincrement_ticket_is_in_new_prefix"_test = [] {
        check_counted_array_saved_allocation_ticket(false, true, true);
    };

    "counted_array_saved_ticket_relation_is_killed_by_store"_test = [] {
        check_counted_array_saved_allocation_ticket(true, true, false);
    };

    "counted_array_unrelated_saved_ticket_is_rejected"_test = [] {
        check_counted_array_saved_allocation_ticket(false, false, false);
    };

    "counted_array_valid_guard_restores_saved_ticket_relation"_test = [] {
        check_counted_array_boolean_guarded_ticket(false, true);
    };

    "counted_array_invalid_guard_correlation_is_rejected"_test = [] {
        check_counted_array_boolean_guarded_ticket(true, false);
    };

    "counted_array_conjunctive_guard_restores_saved_ticket_relation"_test = [] {
        check_counted_array_conjunctive_guarded_ticket(true, true);
    };

    "counted_array_conjunctive_guard_is_required_for_saved_ticket"_test = [] {
        check_counted_array_conjunctive_guarded_ticket(false, false);
    };

    "counted_array_positive_guard_preserves_prefix_across_decrement"_test = [] {
        check_counted_array_nonwrapping_decrement(true, true);
    };

    "counted_array_possibly_wrapping_decrement_is_rejected"_test = [] {
        check_counted_array_nonwrapping_decrement(false, false);
    };

    "counted_array_masked_witness_proves_cycles_picker_default"_test = [] {
        check_counted_array_masked_witness(
            true, false, false, false, false, true);
    };

    "counted_array_masked_zero_survives_counter_reset"_test = [] {
        check_counted_array_masked_witness(
            true, false, false, true, false, true);
    };

    "counted_array_masked_witness_survives_copy_chain"_test = [] {
        check_counted_array_masked_witness(
            true, false, false, false, true, true);
    };

    "counted_array_masked_witness_requires_initial_zero_contract"_test = [] {
        check_counted_array_masked_witness(
            false, false, false, false, false, false);
    };

    "counted_array_masked_witness_rejects_unbacked_flag_set"_test = [] {
        check_counted_array_masked_witness(
            true, true, false, false, false, false);
    };

    "counted_array_masked_witness_is_killed_by_unknown_store"_test = [] {
        check_counted_array_masked_witness(
            true, false, true, false, false, false);
    };

    "counted_array_masked_relation_survives_unrelated_select_or"_test = [] {
        check_counted_array_masked_expression_update(false, true);
    };

    "counted_array_masked_relation_rejects_scatter_select_or"_test = [] {
        check_counted_array_masked_expression_update(true, false);
    };

    "counted_array_exhausted_capacity_implies_nonempty_prefix"_test = [] {
        check_counted_array_exhausted_capacity_guard(true, true);
    };

    "counted_array_zero_capacity_does_not_imply_nonempty_prefix"_test = [] {
        check_counted_array_exhausted_capacity_guard(false, false);
    };

    "counted_array_prunes_impossible_masked_nonzero_edge"_test = [] {
        check_counted_array_impossible_masked_edge(true, true);
    };

    "counted_array_keeps_possible_masked_nonzero_edge"_test = [] {
        check_counted_array_impossible_masked_edge(false, false);
    };

    "counted_array_prunes_masked_edge_after_known_zero_select"_test = [] {
        check_counted_array_impossible_masked_edge(
            false, true, true, false);
    };

    "counted_array_keeps_masked_edge_after_possible_nonzero_select"_test = [] {
        check_counted_array_impossible_masked_edge(
            false, false, true, true);
    };

    "counted_array_rollback_preserves_initialized_last_slot"_test = [] {
        check_counted_array_rollback_tail(
            1u, true, false, false, true);
    };

    "counted_array_rollback_join_preserves_guarded_safe_disjunction"_test = [] {
        check_counted_array_rollback_tail(
            1u, true, true, true, true);
    };

    "counted_array_rollback_rejects_unrelated_tail_ticket"_test = [] {
        check_counted_array_rollback_tail(
            1u, false, false, false, false);
    };

    "counted_array_rollback_rejects_nonlast_ticket_identity"_test = [] {
        check_counted_array_rollback_tail(
            2u, true, false, false, false);
    };

    "counted_array_bounded_prefix_accepts_wider_logical_counter"_test = [] {
        // Executions which dereference the physical array are in-bounds by
        // the XIR memory contract. The bounded-prefix theorem therefore does
        // not require the logical counter guard to equal the array extent.
        check_counted_array_capacity_guard(3u, true, true);
    };

    "counted_array_increment_without_element_definition_is_rejected"_test = [] {
        check_counted_array_capacity_guard(2u, false, false);
    };

    "counted_array_prefix_resolves_block_local_multi_store_copies"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *append_condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *copy_arm =
            kernel->create_value_argument(Type::of<bool>());
        auto *read_index =
            kernel->create_value_argument(Type::of<uint>());
        auto *resume = kernel->create_basic_block();
        auto *append = kernel->create_basic_block();
        auto *skip = kernel->create_basic_block();
        auto *consume = kernel->create_basic_block();
        auto *left = kernel->create_basic_block();
        auto *right = kernel->create_basic_block();
        auto *array_type = Type::array(Type::of<uint>(), 4u);
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *array = builder.alloca_local(array_type);
        auto *count = builder.alloca_local(Type::of<uint>());
        auto *transported_index = builder.alloca_local(Type::of<uint>());
        builder.coro_suspend(17u, "counted-multi-store-copy", nullptr);

        builder.set_insertion_point(resume);
        builder.coro_resume(17u, nullptr);
        builder.store(
            count, module.create_constant_zero(Type::of<uint>()));
        auto *sentinel = builder.gep(
            Type::of<uint>(), array,
            {module.create_constant_zero(Type::of<uint>())});
        builder.store(
            sentinel, module.create_constant_zero(Type::of<uint>()));
        builder.cond_br(append_condition, append, skip);

        builder.set_insertion_point(append);
        auto *append_index = builder.load(Type::of<uint>(), count);
        auto *element = builder.gep(
            Type::of<uint>(), array, {append_index});
        builder.store(
            element, module.create_constant_one(Type::of<uint>()));
        auto *old_count = builder.load(Type::of<uint>(), count);
        auto *next_count = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {old_count, module.create_constant_one(Type::of<uint>())});
        builder.store(count, next_count);
        builder.br(consume);

        builder.set_insertion_point(skip);
        builder.br(consume);

        builder.set_insertion_point(consume);
        builder.cond_br(copy_arm, left, right);

        const auto emit_safe_read = [&](BasicBlock *block) noexcept {
            builder.set_insertion_point(block);
            auto *current_count = builder.load(Type::of<uint>(), count);
            auto *in_prefix = builder.call(
                Type::of<bool>(), ArithmeticOp::BINARY_LESS,
                {read_index, current_count});
            auto *bounded_index = builder.call(
                Type::of<uint>(), ArithmeticOp::SELECT,
                {module.create_constant_zero(Type::of<uint>()),
                 read_index, in_prefix});
            builder.store(transported_index, bounded_index);
            auto *copy = builder.load(Type::of<uint>(), transported_index);
            auto *selected = builder.gep(
                Type::of<uint>(), array, {copy});
            static_cast<void>(
                builder.load(Type::of<uint>(), selected));
            builder.return_void();
        };
        emit_safe_read(left);
        emit_safe_read(right);

        expect(xir_verify_module(&module).succeeded());
        auto before = coro_cfg_distill_pass_run_on_function(kernel);
        expect(before.succeeded());
        expect(frame_contains(before, array));

        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.initialized_prefix_proof_count == 1u);
        expect(info.rejected_prior_lifetime_observation_count == 0u);
        expect(array->parent_block() == resume);
        expect(xir_verify_module(&module).succeeded());

        auto after = coro_cfg_distill_pass_run_on_function(kernel);
        expect(after.succeeded());
        expect(!frame_contains(after, array));
    };

    "counted_array_prefix_does_not_guess_reaching_store_across_edge"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *copy_arm =
            kernel->create_value_argument(Type::of<bool>());
        auto *read_index =
            kernel->create_value_argument(Type::of<uint>());
        auto *resume = kernel->create_basic_block();
        auto *left = kernel->create_basic_block();
        auto *right = kernel->create_basic_block();
        auto *merge = kernel->create_basic_block();
        auto *array_type = Type::array(Type::of<uint>(), 4u);
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *array = builder.alloca_local(array_type);
        auto *count = builder.alloca_local(Type::of<uint>());
        auto *transported_index = builder.alloca_local(Type::of<uint>());
        builder.coro_suspend(18u, "counted-cross-edge-copy", nullptr);

        builder.set_insertion_point(resume);
        builder.coro_resume(18u, nullptr);
        builder.store(
            count, module.create_constant_zero(Type::of<uint>()));
        auto *sentinel = builder.gep(
            Type::of<uint>(), array,
            {module.create_constant_zero(Type::of<uint>())});
        builder.store(
            sentinel, module.create_constant_zero(Type::of<uint>()));
        auto *append_index = builder.load(Type::of<uint>(), count);
        auto *element = builder.gep(
            Type::of<uint>(), array, {append_index});
        builder.store(
            element, module.create_constant_one(Type::of<uint>()));
        auto *old_count = builder.load(Type::of<uint>(), count);
        auto *next_count = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {old_count, module.create_constant_one(Type::of<uint>())});
        builder.store(count, next_count);
        auto *current_count = builder.load(Type::of<uint>(), count);
        auto *in_prefix = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS,
            {read_index, current_count});
        auto *bounded_index = builder.call(
            Type::of<uint>(), ArithmeticOp::SELECT,
            {module.create_constant_zero(Type::of<uint>()),
             read_index, in_prefix});
        builder.cond_br(copy_arm, left, right);

        builder.set_insertion_point(left);
        builder.store(transported_index, bounded_index);
        builder.br(merge);

        builder.set_insertion_point(right);
        builder.store(transported_index, bounded_index);
        builder.br(merge);

        builder.set_insertion_point(merge);
        auto *copy = builder.load(Type::of<uint>(), transported_index);
        auto *selected = builder.gep(
            Type::of<uint>(), array, {copy});
        static_cast<void>(builder.load(Type::of<uint>(), selected));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto original_block = array->parent_block();
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.initialized_prefix_proof_count == 0u);
        expect(info.rejected_prior_lifetime_observation_count == 1u);
        expect(array->parent_block() == original_block);
        expect(xir_verify_module(&module).succeeded());
    };

    "counted_array_prefix_does_not_treat_recurrence_as_copy"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *read_index =
            kernel->create_value_argument(Type::of<uint>());
        auto *resume = kernel->create_basic_block();
        auto *array_type = Type::array(Type::of<uint>(), 4u);
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *array = builder.alloca_local(array_type);
        auto *count = builder.alloca_local(Type::of<uint>());
        auto *state_index = builder.alloca_local(Type::of<uint>());
        builder.coro_suspend(19u, "counted-recurrence", nullptr);

        builder.set_insertion_point(resume);
        builder.coro_resume(19u, nullptr);
        builder.store(
            count, module.create_constant_zero(Type::of<uint>()));
        auto *sentinel = builder.gep(
            Type::of<uint>(), array,
            {module.create_constant_zero(Type::of<uint>())});
        builder.store(
            sentinel, module.create_constant_zero(Type::of<uint>()));
        auto *append_index = builder.load(Type::of<uint>(), count);
        auto *element = builder.gep(
            Type::of<uint>(), array, {append_index});
        builder.store(
            element, module.create_constant_one(Type::of<uint>()));
        auto *old_count = builder.load(Type::of<uint>(), count);
        auto *next_count = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {old_count, module.create_constant_one(Type::of<uint>())});
        builder.store(count, next_count);

        builder.store(
            state_index, module.create_constant_zero(Type::of<uint>()));
        auto *old_state = builder.load(Type::of<uint>(), state_index);
        auto *current_count = builder.load(Type::of<uint>(), count);
        auto *in_prefix = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS,
            {read_index, current_count});
        auto *bounded_index = builder.call(
            Type::of<uint>(), ArithmeticOp::SELECT,
            {old_state, read_index, in_prefix});
        builder.store(state_index, bounded_index);
        auto *state_copy = builder.load(Type::of<uint>(), state_index);
        auto *selected = builder.gep(
            Type::of<uint>(), array, {state_copy});
        static_cast<void>(builder.load(Type::of<uint>(), selected));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto original_block = array->parent_block();
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.initialized_prefix_proof_count == 0u);
        expect(info.rejected_prior_lifetime_observation_count == 1u);
        expect(array->parent_block() == original_block);
        expect(xir_verify_module(&module).succeeded());
    };

    "counted_array_increment_before_element_store_is_rejected"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *read_index =
            kernel->create_value_argument(Type::of<uint>());
        auto *resume = kernel->create_basic_block();
        auto *array_type = Type::array(Type::of<uint>(), 4u);
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *array = builder.alloca_local(array_type);
        auto *count = builder.alloca_local(Type::of<uint>());
        builder.coro_suspend(9u, "bad-order", nullptr);

        builder.set_insertion_point(resume);
        builder.coro_resume(9u, nullptr);
        builder.store(
            count, module.create_constant_zero(Type::of<uint>()));
        auto *sentinel = builder.gep(
            Type::of<uint>(), array,
            {module.create_constant_zero(Type::of<uint>())});
        builder.store(
            sentinel, module.create_constant_zero(Type::of<uint>()));
        auto *old_count = builder.load(Type::of<uint>(), count);
        auto *next_count = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {old_count, module.create_constant_one(Type::of<uint>())});
        builder.store(count, next_count);
        auto *late_index = builder.load(Type::of<uint>(), count);
        auto *late_element = builder.gep(
            Type::of<uint>(), array, {late_index});
        builder.store(
            late_element, module.create_constant_one(Type::of<uint>()));
        auto *current_count = builder.load(Type::of<uint>(), count);
        auto *in_prefix = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS,
            {read_index, current_count});
        auto *bounded_index = builder.call(
            Type::of<uint>(), ArithmeticOp::SELECT,
            {module.create_constant_zero(Type::of<uint>()),
             read_index, in_prefix});
        auto *selected = builder.gep(
            Type::of<uint>(), array, {bounded_index});
        static_cast<void>(builder.load(Type::of<uint>(), selected));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto original_block = array->parent_block();
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.initialized_prefix_proof_count == 0u);
        expect(info.rejected_prior_lifetime_observation_count == 1u);
        expect(array->parent_block() == original_block);
        expect(xir_verify_module(&module).succeeded());
    };

    "counted_array_subaggregate_store_is_not_an_extension"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *read_index =
            kernel->create_value_argument(Type::of<uint>());
        auto *resume = kernel->create_basic_block();
        auto *element_type = Type::structure(
            {Type::of<uint>(), Type::of<uint>()});
        auto *array_type = Type::array(element_type, 4u);
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *array = builder.alloca_local(array_type);
        auto *count = builder.alloca_local(Type::of<uint>());
        builder.coro_suspend(10u, "bad-subaggregate-store", nullptr);

        builder.set_insertion_point(resume);
        builder.coro_resume(10u, nullptr);
        builder.store(
            count, module.create_constant_zero(Type::of<uint>()));
        uint32_t sentinel_index_value = 3u;
        auto *sentinel_index = module.create_constant(
            Type::of<uint>(), &sentinel_index_value);
        auto *sentinel = builder.gep(
            element_type, array, {sentinel_index});
        builder.store(sentinel, module.create_constant_zero(element_type));
        auto *append_index = builder.load(Type::of<uint>(), count);
        auto *element = builder.gep(
            element_type, array, {append_index});
        auto *first_field = builder.gep(
            Type::of<uint>(), element,
            {module.create_constant_zero(Type::of<uint>())});
        builder.store(
            first_field, module.create_constant_one(Type::of<uint>()));
        auto *old_count = builder.load(Type::of<uint>(), count);
        auto *next_count = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {old_count, module.create_constant_one(Type::of<uint>())});
        builder.store(count, next_count);
        auto *current_count = builder.load(Type::of<uint>(), count);
        auto *in_prefix = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS,
            {read_index, current_count});
        auto *bounded_index = builder.call(
            Type::of<uint>(), ArithmeticOp::SELECT,
            {sentinel_index, read_index, in_prefix});
        auto *selected = builder.gep(
            element_type, array, {bounded_index});
        static_cast<void>(builder.load(element_type, selected));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto original_block = array->parent_block();
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.initialized_prefix_proof_count == 0u);
        expect(info.rejected_prior_lifetime_observation_count == 1u);
        expect(array->parent_block() == original_block);
        expect(xir_verify_module(&module).succeeded());
    };

    "counted_array_unrelated_read_bound_is_rejected"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *read_index =
            kernel->create_value_argument(Type::of<uint>());
        auto *unrelated_limit =
            kernel->create_value_argument(Type::of<uint>());
        auto *resume = kernel->create_basic_block();
        auto *array_type = Type::array(Type::of<uint>(), 4u);
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *array = builder.alloca_local(array_type);
        auto *count = builder.alloca_local(Type::of<uint>());
        builder.coro_suspend(11u, "bad-bound", nullptr);

        builder.set_insertion_point(resume);
        builder.coro_resume(11u, nullptr);
        builder.store(
            count, module.create_constant_zero(Type::of<uint>()));
        auto *sentinel = builder.gep(
            Type::of<uint>(), array,
            {module.create_constant_zero(Type::of<uint>())});
        builder.store(
            sentinel, module.create_constant_zero(Type::of<uint>()));
        auto *append_index = builder.load(Type::of<uint>(), count);
        auto *element = builder.gep(
            Type::of<uint>(), array, {append_index});
        builder.store(
            element, module.create_constant_one(Type::of<uint>()));
        auto *old_count = builder.load(Type::of<uint>(), count);
        auto *next_count = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {old_count, module.create_constant_one(Type::of<uint>())});
        builder.store(count, next_count);
        auto *unrelated_guard = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS,
            {read_index, unrelated_limit});
        auto *bounded_index = builder.call(
            Type::of<uint>(), ArithmeticOp::SELECT,
            {module.create_constant_zero(Type::of<uint>()),
             read_index, unrelated_guard});
        auto *selected = builder.gep(
            Type::of<uint>(), array, {bounded_index});
        static_cast<void>(builder.load(Type::of<uint>(), selected));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto original_block = array->parent_block();
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.initialized_prefix_proof_count == 0u);
        expect(info.rejected_prior_lifetime_observation_count == 1u);
        expect(array->parent_block() == original_block);
        expect(xir_verify_module(&module).succeeded());
    };

    "counted_array_arbitrary_counter_overwrite_is_rejected"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *read_index =
            kernel->create_value_argument(Type::of<uint>());
        auto *replacement_count =
            kernel->create_value_argument(Type::of<uint>());
        auto *resume = kernel->create_basic_block();
        auto *array_type = Type::array(Type::of<uint>(), 4u);
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *array = builder.alloca_local(array_type);
        auto *count = builder.alloca_local(Type::of<uint>());
        builder.coro_suspend(13u, "bad-counter-overwrite", nullptr);

        builder.set_insertion_point(resume);
        builder.coro_resume(13u, nullptr);
        builder.store(
            count, module.create_constant_zero(Type::of<uint>()));
        auto *sentinel = builder.gep(
            Type::of<uint>(), array,
            {module.create_constant_zero(Type::of<uint>())});
        builder.store(
            sentinel, module.create_constant_zero(Type::of<uint>()));
        auto *append_index = builder.load(Type::of<uint>(), count);
        auto *element = builder.gep(
            Type::of<uint>(), array, {append_index});
        builder.store(
            element, module.create_constant_one(Type::of<uint>()));
        builder.store(count, replacement_count);
        auto *current_count = builder.load(Type::of<uint>(), count);
        auto *in_prefix = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS,
            {read_index, current_count});
        auto *bounded_index = builder.call(
            Type::of<uint>(), ArithmeticOp::SELECT,
            {module.create_constant_zero(Type::of<uint>()),
             read_index, in_prefix});
        auto *selected = builder.gep(
            Type::of<uint>(), array, {bounded_index});
        static_cast<void>(builder.load(Type::of<uint>(), selected));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto original_block = array->parent_block();
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.initialized_prefix_proof_count == 0u);
        expect(info.rejected_prior_lifetime_observation_count == 1u);
        expect(array->parent_block() == original_block);
        expect(xir_verify_module(&module).succeeded());
    };

    "counted_array_partial_branch_extension_is_rejected"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *write_element =
            kernel->create_value_argument(Type::of<bool>());
        auto *read_index =
            kernel->create_value_argument(Type::of<uint>());
        auto *resume = kernel->create_basic_block();
        auto *write = kernel->create_basic_block();
        auto *skip = kernel->create_basic_block();
        auto *merge = kernel->create_basic_block();
        auto *array_type = Type::array(Type::of<uint>(), 4u);
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto *array = builder.alloca_local(array_type);
        auto *count = builder.alloca_local(Type::of<uint>());
        builder.coro_suspend(15u, "bad-partial-extension", nullptr);

        builder.set_insertion_point(resume);
        builder.coro_resume(15u, nullptr);
        builder.store(
            count, module.create_constant_zero(Type::of<uint>()));
        auto *sentinel = builder.gep(
            Type::of<uint>(), array,
            {module.create_constant_zero(Type::of<uint>())});
        builder.store(
            sentinel, module.create_constant_zero(Type::of<uint>()));
        builder.cond_br(write_element, write, skip);

        builder.set_insertion_point(write);
        auto *append_index = builder.load(Type::of<uint>(), count);
        auto *element = builder.gep(
            Type::of<uint>(), array, {append_index});
        builder.store(
            element, module.create_constant_one(Type::of<uint>()));
        builder.br(merge);

        builder.set_insertion_point(skip);
        builder.br(merge);

        builder.set_insertion_point(merge);
        auto *old_count = builder.load(Type::of<uint>(), count);
        auto *next_count = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {old_count, module.create_constant_one(Type::of<uint>())});
        builder.store(count, next_count);
        auto *current_count = builder.load(Type::of<uint>(), count);
        auto *in_prefix = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_LESS,
            {read_index, current_count});
        auto *bounded_index = builder.call(
            Type::of<uint>(), ArithmeticOp::SELECT,
            {module.create_constant_zero(Type::of<uint>()),
             read_index, in_prefix});
        auto *selected = builder.gep(
            Type::of<uint>(), array, {bounded_index});
        static_cast<void>(builder.load(Type::of<uint>(), selected));
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto original_block = array->parent_block();
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.initialized_prefix_proof_count == 0u);
        expect(info.rejected_prior_lifetime_observation_count == 1u);
        expect(array->parent_block() == original_block);
        expect(xir_verify_module(&module).succeeded());
    };

    "phi_pointer_use_is_an_atomic_noop"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *join = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        auto *clock = builder.clock();
        builder.br(join);
        builder.set_insertion_point(join);
        static_cast<void>(builder.phi(
            Type::of<uint>(), {{scratch, entry}}));
        builder.return_void();

        auto original_index = instruction_index(entry, scratch);
        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.rejected_phi_use_count == 1u);
        expect(info.contracted_alloca_count == 0u);
        expect(scratch->parent_block() == entry);
        expect(instruction_index(entry, scratch) == original_index);
        expect(instruction_index(entry, scratch) <
               instruction_index(entry, clock));
    };

    "snapshot_instruction_order_is_sparse_and_exact_after_moves"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        constexpr auto candidate_count = 128u;
        constexpr auto unrelated_instruction_count = 4096u;
        luisa::vector<AllocaInst *> candidates;
        candidates.reserve(candidate_count);
        for (auto i = 0u; i < candidate_count; ++i) {
            candidates.emplace_back(
                builder.alloca_local(Type::of<uint>()));
        }
        auto *one = module.create_constant_one(Type::of<uint>());
        auto *unrelated = static_cast<Value *>(one);
        for (auto i = 0u; i < unrelated_instruction_count; ++i) {
            unrelated = builder.call(
                Type::of<uint>(), ArithmeticOp::BINARY_ADD,
                {unrelated, one});
        }
        for (auto *candidate : candidates) {
            builder.store(candidate, one);
            static_cast<void>(
                builder.load(Type::of<uint>(), candidate));
        }
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto info = coro_alloca_scope_pass_run_on_function(
            kernel, {.verify_instruction_order = true});

        expect(info.scanned_local_alloca_count == candidate_count);
        expect(info.contracted_alloca_count == candidate_count);
        expect(info.delayed_first_definition_count == candidate_count);
        // One insertion-point query and one strict definition-before-load
        // query per candidate. User inspection depends on the one observation
        // per coordinate, not on the 4,096 unrelated block instructions.
        expect(info.instruction_order_query_count ==
               2u * candidate_count);
        expect(info.placement_user_inspection_count ==
               candidate_count);
        for (auto *candidate : candidates) {
            expect(candidate->parent_block() == entry);
        }
        expect(xir_verify_module(&module).succeeded());
    };

    "invalid_suspend_resume_pair_is_an_atomic_noop"_test = [] {
        Module module;
        BasicBlock *entry;
        auto *kernel = make_kernel(module, entry);
        auto *resume_a = kernel->create_basic_block();
        auto *resume_b = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *scratch = builder.alloca_local(Type::of<uint>());
        builder.coro_suspend(17u, "ambiguous", nullptr);
        builder.set_insertion_point(resume_a);
        builder.coro_resume(17u, nullptr);
        static_cast<void>(builder.load(Type::of<uint>(), scratch));
        builder.return_void();
        builder.set_insertion_point(resume_b);
        builder.coro_resume(17u, nullptr);
        builder.return_void();

        auto info = coro_alloca_scope_pass_run_on_function(kernel);
        expect(info.invalid_semantic_cfg_count == 1u);
        expect(info.contracted_alloca_count == 0u);
        expect(scratch->parent_block() == entry);
    };

    "null_and_declaration_are_noops"_test = [] {
        expect(!coro_alloca_scope_pass_run_on_function(nullptr).changed());
        Module module;
        auto *callable = module.create_callable(nullptr);
        expect(!coro_alloca_scope_pass_run_on_function(callable).changed());
    };
}

int main() {
    register_coro_alloca_scope_tests();
    return 0;
}
